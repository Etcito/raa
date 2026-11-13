/*
MIT License

Copyright (c) 2026 SamG-Coder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
// Adapted from solstice@7e465b599c13da165948ebaa44df824c36c3ae07
// crates/solstice-render/shaders/vg/cull.wgsl.
// Rewritten for RAA's S32 assembly and standard-Z depth: `cull_module_wgsl()` concatenates
// `depth.wgsl`, `pass_view.wgsl` (which declares `PassView`, the scene twins, `CullState` and
// the queue and visible records) and this body; the solstice include preprocessor is not
// ported. The scene data sits at group 2 and the per-phase work buffers at group 3. The page
// requests, the installed-page and missing-children bits, the texture-detail requests and
// `NO_TEXTURE` are removed: RAA never pages cluster data.
//
// Hierarchical LOD selection and culling, one breadth-first level per dispatch. WebGPU has no
// way for a dispatch to grow its own workload, so traversal runs as a fixed series of
// dispatches over a shared queue: each reads the range appended by the previous one and
// appends the next level behind it (see cull_control.wgsl).
//
// Occlusion culling uses two phases. Phase 0 tests against the previous frame's depth, which is
// mostly right, and defers everything it rejects to a retry list. After that result is
// rasterized and a fresh depth pyramid built, phase 1 re-tests the retry list and draws
// whatever had been wrongly rejected.

// Clusters projecting smaller than this are rasterized in the compute shader.
const SOFTWARE_MAX_DIAMETER_PX: f32 = 40.0;
// The same for depth maps, where the compute rasterizer only pays off for the smallest
// clusters: it costs a pass to lay its depth over the map, which a few mid-sized clusters do
// not earn back, while the pixel-sized clusters of a wood's leaves do many times over.
const SOFTWARE_MAX_DIAMETER_TEXELS: f32 = 16.0;

@group(2) @binding(0) var<uniform> view: PassView;
@group(2) @binding(1) var<storage, read> instances: array<VgInstance>;
@group(2) @binding(2) var<storage, read> nodes: array<VgNode>;
@group(2) @binding(3) var<storage, read> clusters: array<VgCluster>;
@group(2) @binding(4) var hzb: texture_2d<f32>;

@group(3) @binding(5) var<storage, read_write> state: CullState;
@group(3) @binding(6) var<storage, read_write> queue: array<QueueEntry>;
@group(3) @binding(7) var<storage, read_write> retry: array<QueueEntry>;
@group(3) @binding(8) var<storage, read_write> visible: array<VisibleCluster>;

fn push_queue(entry: QueueEntry) {
    let slot = atomicAdd(&state.write_head, 1u);
    if slot < arrayLength(&queue) {
        queue[slot] = entry;
    } else {
        atomicAdd(&state.dropped, 1u);
    }
}

fn push_retry(entry: QueueEntry) {
    let slot = atomicAdd(&state.retry_count, 1u);
    if slot < arrayLength(&retry) {
        retry[slot] = entry;
    } else {
        atomicAdd(&state.dropped, 1u);
    }
}

// Software clusters fill the list from the front and hardware from the back, so both can
// append concurrently and each still forms one contiguous range. An entry is claimed from a
// single count of both before either end is written, so the two ranges can never meet,
// however many threads arrive at once. The cluster word holds the absolute cluster index and
// the flags of the list it went to. Returns false, and counts the loss, if the list is full.
fn emit_cluster(instance: u32, cluster: u32, software: bool) -> bool {
    let capacity = arrayLength(&visible);
    if atomicAdd(&state.reserved, 1u) >= capacity {
        atomicAdd(&state.dropped, 1u);
        return false;
    }
    var word = cluster & 0xffffffu;
    if software {
        word |= VISIBLE_SOFTWARE;
    }
    if view.phase == 1u {
        word |= VISIBLE_SECOND;
    }
    if software {
        visible[atomicAdd(&state.sw_count, 1u)] = VisibleCluster(instance, word);
    } else {
        visible[capacity - 1u - atomicAdd(&state.hw_count, 1u)] = VisibleCluster(instance, word);
    }
    return true;
}

// Screen-space size of an error metric living on a sphere. Using the sphere's nearest point
// makes this conservative and, since parent spheres contain their children, monotonic up the
// hierarchy.
fn projected_error(center: vec3<f32>, radius: f32, error: f32) -> f32 {
    if view.is_ortho != 0u {
        var pixels = error * view.lod_scale;
        // Drawn for a camera: error that camera cannot see is not worth drawing here either.
        // A caster's error shows where its shadow lands, anywhere along the line from it away
        // from the light, so the camera's distance to that line bounds how near the error can
        // be seen. A parent's line sweeps a volume holding its children's, so this stays
        // monotonic up the hierarchy, as the view's own measure is; the smaller of the two is
        // too.
        if view.eye_lod_scale > 0.0 {
            let to_eye = view.eye_position - center;
            let along = max(dot(to_eye, -view.eye_light), 0.0);
            let nearest = sqrt(max(dot(to_eye, to_eye) - along * along, 0.0));
            let distance = max(nearest - radius, view.eye_near);
            pixels = min(pixels, error * view.eye_lod_scale / distance);
        }
        return pixels;
    }
    let distance = max(length(center - view.camera_position) - radius, view.near);
    return error * view.lod_scale / distance;
}

fn in_frustum(center: vec3<f32>, radius: f32) -> bool {
    for (var i = 0u; i < 6u; i++) {
        let p = view.planes[i];
        if dot(p.xyz, center) + p.w < -radius {
            return false;
        }
    }
    for (var i = 0u; i < view.receiver_plane_count; i++) {
        let p = view.receiver_planes[i];
        if dot(p.xyz, center) + p.w < -radius {
            return false;
        }
    }
    return true;
}

// Tests a world-space sphere against the depth pyramid. The pyramid stores the farthest depth
// per texel (the largest, under standard-Z), so a sphere whose nearest point is farther than
// every stored surface is hidden. The sphere's nearest point is projected with the occlusion
// view, a finite-far perspective projection, rather than solstice's reversed infinite near/z.
fn passes_occlusion(center: vec3<f32>, radius: f32) -> bool {
    if view.occlusion_enabled == 0u {
        return true;
    }
    let center_clip = view.occlusion_view * vec4<f32>(center, 1.0);
    let distance = center_clip.w;
    if distance <= radius + view.near {
        return true; // Touches the near plane: cannot be projected safely.
    }
    // The sphere's nearest point lies one radius towards the camera along the view axis: the
    // projection's depth depends on that view distance alone, so this point has the smallest
    // depth. The axis is the world direction in the projection's fourth row; without one
    // (an orthographic or plain view matrix) there is nothing safe to test.
    let w_row = vec4<f32>(
        view.occlusion_view[0][3],
        view.occlusion_view[1][3],
        view.occlusion_view[2][3],
        view.occlusion_view[3][3],
    );
    let backward = w_row.xyz;
    if dot(backward, backward) < 1.0e-12 {
        return true;
    }
    let nearest_point = center + normalize(backward) * radius;
    let nearest_clip = view.occlusion_view * vec4<f32>(nearest_point, 1.0);
    let nearest = nearest_clip.z / nearest_clip.w;

    // Tight projected bounds of a sphere (Mara & McGuire 2013), from its distance along the
    // view axis and the projection's scale.
    let center_ndc = center_clip.xy / distance;
    let tan_half_angle = radius / sqrt(distance * distance - radius * radius);
    let half = view.proj_scale * tan_half_angle * 0.5;
    let uv_min = clamp(center_ndc * 0.5 + 0.5 - half, vec2<f32>(0.0), vec2<f32>(1.0));
    let uv_max = clamp(center_ndc * 0.5 + 0.5 + half, vec2<f32>(0.0), vec2<f32>(1.0));

    let extent = (uv_max - uv_min) * view.hzb_size;
    let last_level = f32(textureNumLevels(hzb) - 1u);
    let level = u32(clamp(ceil(log2(max(max(extent.x, extent.y), 1.0))), 0.0, last_level));
    let dims = vec2i(textureDimensions(hzb, level));
    let t0 = clamp(vec2i(uv_min * vec2<f32>(dims)), vec2i(0), dims - 1);
    let t1 = clamp(vec2i(uv_max * vec2<f32>(dims)), vec2i(0), dims - 1);

    var farthest = textureLoad(hzb, t0, level).r;
    farthest = raa_depth_farthest(farthest, textureLoad(hzb, vec2i(t1.x, t0.y), level).r);
    farthest = raa_depth_farthest(farthest, textureLoad(hzb, vec2i(t0.x, t1.y), level).r);
    farthest = raa_depth_farthest(farthest, textureLoad(hzb, t1, level).r);

    // Standard-Z: the sphere passes while its nearest point is not farther than the deepest
    // surface of the tile.
    return !raa_depth_nearer(farthest, nearest);
}

// A queued cluster is in the cut when its own error is fine enough for this view, and its
// bounds survive the frustum and occlusion tests.
fn try_emit(entry: QueueEntry, cluster_index: u32, instance: VgInstance) {
    let cluster = clusters[cluster_index];
    let scale = instance_max_scale(instance);
    let lod_center = transform_point(instance, cluster.lod_center);
    let lod_radius = cluster.lod_radius * scale;
    let lod_error = cluster.lod_error * scale;
    if projected_error(lod_center, lod_radius, lod_error) > view.lod_threshold {
        return;
    }

    let center = transform_point(instance, cluster.center);
    let radius = cluster.radius * scale;
    if !in_frustum(center, radius) {
        return;
    }
    if !passes_occlusion(center, radius) {
        if view.phase == 0u {
            push_retry(QueueEntry(entry.instance, cluster_index | CLUSTER_BIT));
        }
        return;
    }

    // 2 means the list is consumed by a compute pass, not a rasterizer: voxelization.
    var software = view.allow_software == 2u;
    if view.allow_software == 1u {
        if view.is_ortho != 0u {
            // Orthographic: the same size everywhere.
            software = 2.0 * radius * view.lod_scale < SOFTWARE_MAX_DIAMETER_TEXELS;
        } else {
            let distance = length(center - view.camera_position) - radius;
            if distance > view.near {
                software = 2.0 * radius * view.lod_scale / distance < SOFTWARE_MAX_DIAMETER_PX;
            }
        }
    }
    if !emit_cluster(entry.instance, cluster_index, software) {
        return;
    }
    atomicAdd(&state.triangle_count, cluster_triangle_count(cluster));
}

fn instance_movable(instance: VgInstance) -> bool {
    return (instance.ids.z & MOVABLE) != 0u;
}

// Whether this view draws an instance at all.
fn view_draws(instance: VgInstance) -> bool {
    switch view.instance_filter {
        case INSTANCES_STATIC: {
            return !instance_movable(instance);
        }
        case INSTANCES_MOVABLE: {
            return instance_movable(instance);
        }
        case INSTANCES_GI: {
            // P4 adds the flag for movable instances that stay in the voxels; until it
            // exists, a GI view draws only what cannot move.
            return !instance_movable(instance);
        }
        default: {
            return true;
        }
    }
}

// Queues the hierarchy root of every instance the view draws and whose object bounds are
// inside the frustum and receiver planes.
@compute @workgroup_size(64)
fn seed(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < arrayLength(&instances) {
        let instance = instances[id.x];
        if view_draws(instance) {
            let center = transform_point(instance, instance.bounds.xyz);
            let radius = instance.bounds.w * instance_max_scale(instance);
            if in_frustum(center, radius) {
                push_queue(QueueEntry(id.x, instance_root(instance)));
            }
        }
    }
}

// One thread per queue entry written by the previous level: an inner node appends its
// children, a leaf appends its clusters, and a cluster is LOD-tested, tested against the
// frustum and the pyramid, and emitted when it survives.
@compute @workgroup_size(64)
fn process(@builtin(global_invocation_id) id: vec3<u32>) {
    let slot = state.read_begin + id.x;
    if slot >= state.read_end || slot >= arrayLength(&queue) {
        return;
    }
    let entry = queue[slot];
    let instance = instances[entry.instance];

    if (entry.item & CLUSTER_BIT) != 0u {
        try_emit(entry, entry.item & ~CLUSTER_BIT, instance);
        return;
    }

    let node = nodes[entry.item];
    // If even the coarsest parent below is good enough, everything below is finer than
    // required and this subtree cannot contribute to the cut.
    let lod_center = transform_point(instance, node.lod_center);
    let lod_radius = node.lod_radius * instance_max_scale(instance);
    let scaled_error = node.max_parent_error * instance_max_scale(instance);
    let parent_pixels = projected_error(lod_center, lod_radius, scaled_error);
    if parent_pixels <= view.lod_threshold {
        return;
    }

    let center = transform_point(instance, node.center);
    let radius = node.radius * instance_max_scale(instance);
    if !in_frustum(center, radius) {
        return;
    }
    if !passes_occlusion(center, radius) {
        if view.phase == 0u {
            push_retry(entry);
        }
        return;
    }

    if node.is_leaf == 0u {
        for (var i = 0u; i < node.count; i++) {
            push_queue(QueueEntry(entry.instance, instance.ranges.x + node.first + i));
        }
        return;
    }

    // Leaf: queue the set's clusters, each of which is selected on its own error below.
    for (var i = 0u; i < node.count; i++) {
        let cluster_index = instance.ranges.y + node.first + i;
        push_queue(QueueEntry(entry.instance, cluster_index | CLUSTER_BIT));
    }
}
