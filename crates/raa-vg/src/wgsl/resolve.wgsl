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
// crates/solstice-render/shaders/vg/resolve.wgsl:1-283.
// Bindings move to group 2 (contract `vg-resolve-module`). The external include, the
// sea-history bindings, the material texture tiers and their samplers and the emissive
// output are removed: no sampler is left, so the surface is read with loads and written
// with storage textures. The scene twins come from `pass_view.wgsl`.

// The S32 resolve frame record, 320 bytes, the twin of `VgFrameGpu` in vg/resolve.rs:
// view_proj @0, unjittered_view_proj @64, previous_view_proj @128, inv_view_proj @192,
// viewport @256, camera_jitter @272 (current xy, previous zw), camera_pos @288 and
// flags @304 (debug view code, history valid, 0, 0).
struct VgFrame {
    view_proj: mat4x4<f32>,
    unjittered_view_proj: mat4x4<f32>,
    previous_view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    viewport: vec4<f32>,
    camera_jitter: vec4<f32>,
    camera_pos: vec4<f32>,
    flags: vec4<u32>,
}

// The S10 `SurfaceMaterialGpu` record, 256 bytes. Only `base` @0 and `header` @96 are read:
// `header.x` is the model (2 = a registered material function) and `header.y` its function id.
struct VgMaterial {
    base: vec4<f32>,
    emissive: vec4<f32>,
    surface: vec4<f32>,
    sheen: vec4<f32>,
    transmission: vec4<f32>,
    attenuation: vec4<f32>,
    header: vec4<u32>,
    p: array<vec4<f32>, 8>,
    fur: vec4<f32>,
}

@group(2) @binding(0) var<uniform> frame: VgFrame;
@group(2) @binding(1) var<storage, read> vis_depth: array<u32>;
@group(2) @binding(2) var<storage, read> vis_id: array<u32>;
@group(2) @binding(3) var<storage, read> visible: array<VisibleCluster>;
@group(2) @binding(4) var<storage, read> instances: array<VgInstance>;
@group(2) @binding(5) var<storage, read> clusters: array<VgCluster>;
@group(2) @binding(6) var<storage, read> vertices: array<PackedVertex>;
@group(2) @binding(7) var<storage, read> triangles: array<u32>;
@group(2) @binding(8) var<storage, read> materials: array<VgMaterial>;
@group(2) @binding(9) var out_ids: texture_storage_2d<rgba32uint, write>;
@group(2) @binding(10) var out_normal: texture_storage_2d<rgba16float, write>;
@group(2) @binding(11) var out_uv_motion: texture_storage_2d<rgba32float, write>;
@group(2) @binding(12) var out_albedo: texture_storage_2d<rgba8unorm, write>;

// `SurfaceMaterialGpu.header.x`: a registered material function.
const MATERIAL_MODEL_FUNCTION: u32 = 2u;

fn pixel_to_ndc(pixel: vec2<f32>, viewport: vec2<f32>) -> vec2<f32> {
    let uv = pixel / viewport;
    return vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
}

// Perspective-correct barycentrics of a pixel inside a triangle (solstice resolve.wgsl:80-94).
fn barycentrics(clip: array<vec4<f32>, 3>, ndc: vec2<f32>) -> vec3<f32> {
    let inv_w = vec3<f32>(1.0 / clip[0].w, 1.0 / clip[1].w, 1.0 / clip[2].w);
    let p0 = clip[0].xy * inv_w.x;
    let p1 = clip[1].xy * inv_w.y;
    let p2 = clip[2].xy * inv_w.z;
    let e0 = p1 - p0;
    let e1 = p2 - p0;
    let d = ndc - p0;
    let inv_det = 1.0 / (e0.x * e1.y - e0.y * e1.x);
    let screen_b = (d.x * e1.y - d.y * e1.x) * inv_det;
    let screen_c = (e0.x * d.y - e0.y * d.x) * inv_det;
    let screen = vec3<f32>(1.0 - screen_b - screen_c, screen_b, screen_c);
    let weighted = screen * inv_w;
    return weighted / (weighted.x + weighted.y + weighted.z);
}

fn interp2(b: vec3<f32>, a0: vec2<f32>, a1: vec2<f32>, a2: vec2<f32>) -> vec2<f32> {
    return a0 * b.x + a1 * b.y + a2 * b.z;
}

fn interp3(b: vec3<f32>, a0: vec3<f32>, a1: vec3<f32>, a2: vec3<f32>) -> vec3<f32> {
    return a0 * b.x + a1 * b.y + a2 * b.z;
}

// The signed area of the triangle projected to NDC: positive when counter-clockwise (front).
fn projected_area(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> f32 {
    let e0 = b - a;
    let e1 = c - a;
    return e0.x * e1.y - e0.y * e1.x;
}

// Cluster colours: the PCG hash of the cluster index (solstice resolve.wgsl:131-134).
// Handoff to the task "raa-vg: lod, raster and occlusion debug colours": it extends the
// `frame.flags.x` branch of `main` below with codes 2, 3 and 4, reusing this helper plus
// `cluster_index`, `level`, `entry.cluster` and `material.base.rgb`, which are already in
// scope there. No separate entry point; the debug word layout is the state it reads.
fn hash_color(value: u32) -> vec3<f32> {
    let h = pcg(value);
    return vec3<f32>(f32(h & 255u), f32((h >> 8u) & 255u), f32((h >> 16u) & 255u)) / 255.0 * 0.8 + 0.2;
}

// Piecewise sRGB OETF of a value in [0, 1].
fn encode_srgb(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    let lo = x * 12.92;
    let hi = 1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(hi, lo, x <= vec3<f32>(0.0031308));
}

// `VgFrame.flags.x` = 2 (rank table of `DebugView`): the DAG level as a ramp from blue at
// level 0 through green at level 6 to red at level 12 and above, linear between the stops.
fn lod_ramp(level: u32) -> vec3<f32> {
    let t = f32(min(level, 12u));
    let blue = vec3<f32>(0.1, 0.3, 1.0);
    let green = vec3<f32>(0.2, 0.9, 0.3);
    let red = vec3<f32>(1.0, 0.2, 0.1);
    if (t <= 6.0) {
        return mix(blue, green, t / 6.0);
    }
    return mix(green, red, (t - 6.0) / 6.0);
}

// Code 3: hardware rasterisation is blue, software rasterisation is orange.
fn raster_color(cluster_word: u32) -> vec3<f32> {
    return select(
        vec3<f32>(0.15, 0.55, 1.0),
        vec3<f32>(1.0, 0.6, 0.1),
        (cluster_word & VISIBLE_SOFTWARE) != 0u,
    );
}

// Code 4: the first occlusion phase is green, a cluster recovered by the second is magenta.
fn occlusion_color(cluster_word: u32) -> vec3<f32> {
    return select(
        vec3<f32>(0.25, 0.8, 0.35),
        vec3<f32>(0.95, 0.2, 0.75),
        (cluster_word & VISIBLE_SECOND) != 0u,
    );
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<u32>(frame.viewport.xy);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let pixel = id.xy;
    let index = pixel.y * size.x + pixel.x;
    let key = vis_depth[index];
    if (key == 0u) {
        textureStore(out_ids, pixel, vec4<u32>(0u));
        textureStore(out_normal, pixel, vec4<f32>(0.0));
        textureStore(out_uv_motion, pixel, vec4<f32>(0.0));
        textureStore(out_albedo, pixel, vec4<f32>(0.0));
        return;
    }

    let depth = raa_depth_from_key(key);
    let packed_id = vis_id[index];
    let visible_index = packed_id >> 7u;
    let triangle_index = packed_id & 0x7fu;
    let entry = visible[visible_index];
    let cluster_index = visible_cluster_index(entry);
    let instance = instances[entry.instance];
    let cluster = clusters[cluster_index];
    let material = materials[instance.ids.y];

    let packed = triangles[instance.ranges.w + cluster.triangle_offset + triangle_index];
    let v0 = vertices[instance.ranges.z + cluster.vertex_offset + (packed & 0xffu)];
    let v1 = vertices[instance.ranges.z + cluster.vertex_offset + ((packed >> 8u) & 0xffu)];
    let v2 = vertices[instance.ranges.z + cluster.vertex_offset + ((packed >> 16u) & 0xffu)];

    let local0 = decode_position(instance, v0);
    let local1 = decode_position(instance, v1);
    let local2 = decode_position(instance, v2);
    let world0 = transform_point(instance, local0);
    let world1 = transform_point(instance, local1);
    let world2 = transform_point(instance, local2);
    let clip = array<vec4<f32>, 3>(
        frame.view_proj * vec4<f32>(world0, 1.0),
        frame.view_proj * vec4<f32>(world1, 1.0),
        frame.view_proj * vec4<f32>(world2, 1.0),
    );

    let viewport = frame.viewport.xy;
    let center = vec2<f32>(pixel) + vec2<f32>(0.5);
    let ndc = pixel_to_ndc(center, viewport);
    let b = barycentrics(clip, ndc);

    let ndc0 = clip[0].xy / clip[0].w;
    let ndc1 = clip[1].xy / clip[1].w;
    let ndc2 = clip[2].xy / clip[2].w;
    let face_sign = select(-1.0, 1.0, projected_area(ndc0, ndc1, ndc2) > 0.0);

    let world = interp3(b, world0, world1, world2);
    let normal = normalize(interp3(
        b,
        transform_normal(instance, decode_normal(v0)),
        transform_normal(instance, decode_normal(v1)),
        transform_normal(instance, decode_normal(v2)),
    ));
    let uv = interp2(b, decode_uv(instance, v0), decode_uv(instance, v1), decode_uv(instance, v2));

    // S12 UV motion: current unjittered NDC minus previous, positive X right and Y down.
    var motion = vec2<f32>(0.0);
    if ((instance.ids.z & PREVIOUS_VALID) != 0u && frame.flags.y != 0u) {
        let local = interp3(b, local0, local1, local2);
        let prev_world = transform_point_prev(instance, local);
        let current_clip = frame.unjittered_view_proj * vec4<f32>(world, 1.0);
        let previous_clip = frame.previous_view_proj * vec4<f32>(prev_world, 1.0);
        if (current_clip.w > 0.0 && previous_clip.w > 0.0) {
            let current_ndc = current_clip.xy / current_clip.w;
            let previous_ndc = previous_clip.xy / previous_clip.w - frame.camera_jitter.zw;
            motion = vec2<f32>(
                (current_ndc.x - previous_ndc.x) * 0.5,
                (previous_ndc.y - current_ndc.y) * 0.5,
            );
        }
    }

    let level = min(cluster_level(cluster), 63u);
    var debug_word = cluster_index;
    debug_word = debug_word | (level << 24u);
    debug_word = debug_word | (entry.cluster & VISIBLE_SOFTWARE);
    debug_word = debug_word | (entry.cluster & VISIBLE_SECOND);

    var material_fn = 0u;
    if (material.header.x == MATERIAL_MODEL_FUNCTION) {
        material_fn = material.header.y;
    }

    // The base colour sRGB-encoded, or the debug colour of the active view (rank table of
    // `DebugView`): 1 cluster colours, 2 the DAG level ramp, 3 hardware versus software and
    // 4 the first phase versus recovered by the second. Codes 2..4 are sRGB-encoded too.
    let debug_view = frame.flags.x;
    var albedo = encode_srgb(material.base.rgb);
    if (debug_view == 1u) {
        albedo = hash_color(cluster_index);
    } else if (debug_view == 2u) {
        albedo = encode_srgb(lod_ramp(level));
    } else if (debug_view == 3u) {
        albedo = encode_srgb(raster_color(entry.cluster));
    } else if (debug_view == 4u) {
        albedo = encode_srgb(occlusion_color(entry.cluster));
    }

    textureStore(
        out_ids,
        pixel,
        vec4<u32>(instance.ids.x + 1u, material_fn, debug_word, bitcast<u32>(depth)),
    );
    textureStore(out_normal, pixel, vec4<f32>(normal, face_sign));
    textureStore(out_uv_motion, pixel, vec4<f32>(uv, motion));
    textureStore(out_albedo, pixel, vec4<f32>(albedo, 1.0));
}
