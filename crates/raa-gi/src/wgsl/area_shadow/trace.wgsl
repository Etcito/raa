// S44 mid-tier area-light shadows: the trace kernel of `gi.area-shadow`. One invocation per
// viewport pixel traces one stratified shadow ray towards every mid-tier area light of the
// slot table (contract 'area-stochastic-gpu'): first through the depth pyramid `gi.screen`
// published this frame, then through the GI volume. It writes the weighted visibility S and
// the weight U of each slot into the raw history layers, plus the pixel's linear view depth.
//
// This is RAA's own code; it adapts no solstice source. The estimator it feeds is the ratio
// estimator of Heitz, Hill and McGuire, "Combining Analytic Direct Illumination and Stochastic
// Shadows", I3D 2018: the forward pass blends V = clamp(S / U, 0, 1) into the area-light
// visibility. The stratum cell, R2 jitter, stratum uv and sample weight mirror the CPU twins
// of crates/raa-gi/src/parts/area_shadow/cpu.rs (contract 'area-stochastic-cpu'); the PCG hash,
// the depth pyramid and the volume trace come from the screen modules composed ahead of this
// file.
//
// Group-2 binding registry (binding number to resource):
// 0 raa_gi_frame, 1 raa_gi_depth, 2 raa_gi_normal, 4 raa_gi_pyramid, 5 raa_area_raw_out,
// 6 raa_area_depth_out, 8 raa_area_table, 9 raa_area_params. Bindings 3 and 7 stay declared
// by the composed screen files and are unused here.

// The per-frame trace parameters (16 B), contract 'area-shadow-trace-wgsl'.
struct AreaShadowParamsGpu {
    frame_index: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

// The raw history layers (layer g = slots 2g and 2g+1 as S, U, S, U).
@group(2) @binding(5) var raa_area_raw_out: texture_storage_2d_array<rgba16float, write>;
// The positive linear view depth of the pixel (0 on far pixels).
@group(2) @binding(6) var raa_area_depth_out: texture_storage_2d<r32float, write>;
// The mid-tier slot table, contract 'area-stochastic-gpu' (also group 1 binding 19).
@group(2) @binding(8) var<uniform> raa_area_table: RaaAreaStochasticGpu;
// The frame counter of this trace.
@group(2) @binding(9) var<uniform> raa_area_params: AreaShadowParamsGpu;

// Per-axis R2 sequence constants of the per-frame jitter, as in `cpu::R2_ALPHA`.
const RAA_AREA_R2_ALPHA: vec2<f32> = vec2<f32>(0.7548777, 0.5698403);

// `fract(x) = x - floor(x)`, the WGSL and CPU twin definition.
fn raa_area_fract(x: f32) -> f32 {
    return x - floor(x);
}

// The side of the stratum grid: 3 for nine cells, 2 otherwise (four cells).
fn raa_area_cell_side(cells: u32) -> u32 {
    return select(2u, 3u, cells == 9u);
}

// The stratum a pixel's shadow ray uses this frame: the frame counter plus a per-pixel PCG
// offset, modulo the cell count, so `side * side` consecutive frames visit every cell once.
fn raa_area_stratum_cell(cells: u32, frame: u32, pixel: vec2<u32>) -> u32 {
    let side = raa_area_cell_side(cells);
    let offset = raa_gi_probe_pcg(pixel.x * 7919u + pixel.y * 104729u);
    return (frame + offset) % (side * side);
}

// The per-frame jitter inside a stratum, in [0, 1) per component: an R2 sequence over the
// frame (wrapped every 4096 frames) plus a per-pixel rotation from the PCG hash.
fn raa_area_r2_jitter(frame: u32, pixel: vec2<u32>) -> vec2<f32> {
    let k = f32(frame % 4096u);
    let seed = pixel.x * 2957u + pixel.y * 6113u;
    let rotation = vec2<f32>(
        raa_gi_probe_hash_to_unit(seed),
        raa_gi_probe_hash_to_unit(seed + 1u),
    );
    return vec2<f32>(
        raa_area_fract(0.5 + k * RAA_AREA_R2_ALPHA.x + rotation.x),
        raa_area_fract(0.5 + k * RAA_AREA_R2_ALPHA.y + rotation.y),
    );
}

// The unit-square position of a shadow-ray target: the jittered point of one stratum cell of
// a `side x side` grid over the rectangle.
fn raa_area_stratum_uv(cells: u32, cell: u32, jitter: vec2<f32>) -> vec2<f32> {
    let side = raa_area_cell_side(cells);
    return vec2<f32>(
        (f32(cell % side) + jitter.x) / f32(side),
        (f32(cell / side) + jitter.y) / f32(side),
    );
}

// The geometric weight of one shadow sample: the cosine at the receiver and at the light,
// over the squared distance (clamped at 1e-4). Zero at coincidence, where the direction is
// undefined.
fn raa_area_sample_weight(
    receiver: vec3<f32>,
    normal: vec3<f32>,
    point: vec3<f32>,
    travel: vec3<f32>,
) -> f32 {
    let d = point - receiver;
    let r = length(d);
    if r == 0.0 {
        return 0.0;
    }
    let l = d / r;
    let cos_surface = max(dot(normal, l), 0.0);
    let cos_light = max(dot(travel, -l), 0.0);
    return cos_surface * cos_light / max(r * r, 1e-4);
}

// The hard-ray volume path of the LocalVisibility provider: 1 with no volume; otherwise the
// near and far levels at both ends pick the coarser voxel, the origin clears the own surface
// by 1.5 voxels and the walk stops 3 voxels short of the light.
fn raa_area_volume_visibility(
    world: vec3<f32>,
    normal: vec3<f32>,
    dir: vec3<f32>,
    distance: f32,
) -> f32 {
    let count = raa_gi_volume_params.level_count;
    if count == 0u {
        return 1.0;
    }
    let near_level = min(raa_gi_level_at(world, 0u), count - 1u);
    let light_position = world + dir * distance;
    let far_level = min(raa_gi_level_at(light_position, near_level), count - 1u);
    let voxel = max(
        raa_gi_volume_params.levels[near_level].w,
        raa_gi_volume_params.levels[far_level].w,
    );
    let origin = world + normal * (1.5 * voxel);
    let max_t = max(distance - 3.0 * voxel, 0.0);
    return raa_gi_visibility(origin, dir, voxel, max_t, 24u);
}

// The visibility of the shadow ray from `world` to `point` in [0, 1]: the depth pyramid first
// (when the pass published one), then the GI volume. A screen hit that covers the ray
// completely is a full shadow; otherwise the volume modifies the remaining visibility.
fn raa_area_ray_visibility(world: vec3<f32>, normal: vec3<f32>, point: vec3<f32>) -> f32 {
    let to_point = point - world;
    let distance = length(to_point);
    if distance <= 0.0 {
        return 1.0;
    }
    let dir = to_point / distance;
    var occlusion = 0.0;
    if textureDimensions(raa_gi_pyramid).x > 1u {
        let view_depth = -(raa_gi_frame.inv_view * vec4<f32>(world, 1.0)).z;
        let origin = world + normal * (0.01 + 0.002 * view_depth);
        let hit = raa_gi_trace_screen(origin, dir, distance - 0.05, 48u, 0.15);
        if hit.hit {
            occlusion = hit.weight;
        }
    }
    if occlusion >= 0.999 {
        return 0.0;
    }
    let volume = raa_area_volume_visibility(world, normal, dir, distance);
    return min(1.0 - occlusion, volume);
}

// One invocation per viewport pixel: fill one stratum sample per mid-tier slot and write the
// history groups of the slots the limit covers. Layers past `groups_for(limit)` stay untouched.
@compute @workgroup_size(8, 8)
fn raa_gi_area_trace(@builtin(global_invocation_id) id: vec3<u32>) {
    let viewport = vec2<u32>(raa_gi_frame.viewport.xy);
    if any(id.xy >= viewport) {
        return;
    }
    let pixel = id.xy;
    let limit = min(raa_area_table.params.x, 8u);
    let groups = (limit + 1u) / 2u;
    let depth = textureLoad(raa_gi_depth, vec2<i32>(pixel), 0);
    var weights: array<vec2<f32>, 8>;
    for (var i = 0u; i < 8u; i = i + 1u) {
        weights[i] = vec2<f32>(0.0);
    }
    var view_depth = 0.0;
    if !raa_depth_is_far(depth) {
        let centre = vec2<f32>(pixel) + vec2<f32>(0.5);
        let position = raa_gi_world_position(centre, depth);
        let normal = raa_gi_world_normal(vec2<i32>(pixel));
        view_depth = -raa_gi_view_position(centre, depth).z;
        let cells = raa_area_table.params.y;
        let frame_index = raa_area_params.frame_index;
        for (var s = 0u; s < limit; s = s + 1u) {
            let packed = raa_area_table.lights[s / 4u][s % 4u];
            if packed == 0xFFFFFFFFu {
                continue;
            }
            let light = raa_local_lights[packed];
            if light.kind != 3u {
                continue;
            }
            let cell = raa_area_stratum_cell(cells, frame_index + s, pixel);
            let jitter = raa_area_r2_jitter(frame_index + 7u * s, pixel);
            let uv = raa_area_stratum_uv(cells, cell, jitter);
            let basis = raa_area_basis(light.direction_cos_outer.xyz);
            let point = light.position_range.xyz
                + basis[0] * (uv.x - 0.5) * light.width
                + basis[1] * (uv.y - 0.5) * light.height;
            let weight = raa_area_sample_weight(
                position,
                normal,
                point,
                light.direction_cos_outer.xyz,
            );
            if weight > 0.0 {
                let visibility = raa_area_ray_visibility(position, normal, point);
                weights[s] = vec2<f32>(weight * visibility, weight);
            }
        }
    }
    for (var g = 0u; g < groups; g = g + 1u) {
        let first = weights[2u * g];
        let second = weights[2u * g + 1u];
        textureStore(
            raa_area_raw_out,
            vec2<i32>(pixel),
            i32(g),
            vec4<f32>(first.x, first.y, second.x, second.y),
        );
    }
    textureStore(raa_area_depth_out, vec2<i32>(pixel), vec4<f32>(view_depth));
}
