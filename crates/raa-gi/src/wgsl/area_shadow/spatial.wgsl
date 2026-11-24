// S44 mid-tier area-light shadows: the spatial half of the `gi.area-shadow` denoiser, a
// separable 7-tap bilateral filter in the pattern of crates/raa-gtao/src/wgsl/denoise.wgsl:23-56
// (Gaussian times a depth term, normalised weights, horizontal then vertical) extended with a
// normal term and a 2-pixel stride. `raa_gi_area_spatial_h` filters the reprojected history
// along x into scratch, and `raa_gi_area_spatial_v` filters that along y and writes the
// ratio estimator V = clamp(S / U, 0, 1) into the mask.
//
// This is RAA's own code; it adapts no solstice source. It serves the ratio estimator of
// Heitz, Hill and McGuire, "Combining Analytic Direct Illumination and Stochastic Shadows",
// I3D 2018, whose temporal half is temporal.wgsl. `raa_area_bilateral_weight` and
// `raa_area_ratio_visibility` mirror the CPU twins of
// crates/raa-gi/src/parts/area_shadow/cpu.rs (contract 'area-stochastic-cpu') with the same
// constants.
//
// Group-0 binding registry (binding number to resource):
// 1 raa_area_current_depth, 2 raa_area_normal, 3 raa_area_src, 4 raa_area_src_b,
// 5 raa_area_dst, 7 raa_area_dispatch. The horizontal entry reads no `raa_area_src_b`; the
// vertical entry reads both sources.

// The per-frame dispatch record (16 B), contract 'area-shadow-denoise-wgsl': `index` is the
// history group or mask layer its bind group covers (g for the history pair, l for the mask),
// `usable` the history validity, `pairs` the history group count (1 or 2) and `limit` the
// mid-tier slot limit.
struct AreaDispatchGpu {
    index: u32,
    usable: u32,
    pairs: u32,
    limit: u32,
}

@group(0) @binding(1) var raa_area_current_depth: texture_2d<f32>;
@group(0) @binding(2) var raa_area_normal: texture_2d<f32>;
@group(0) @binding(3) var raa_area_src: texture_2d<f32>;
@group(0) @binding(4) var raa_area_src_b: texture_2d<f32>;
@group(0) @binding(5) var raa_area_dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var<uniform> raa_area_dispatch: AreaDispatchGpu;

// Normalised Gaussian of the separable filter, sigma 2 (`cpu::BILATERAL_GAUSS`).
const RAA_AREA_BILATERAL_GAUSS: array<f32, 7> = array<f32, 7>(
    0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125,
);

// Distance in pixels between two taps, `cpu::SPATIAL_STRIDE` (2).
override RAA_AREA_SPATIAL_STRIDE: i32 = 2;

// Exponent of the normal term, `cpu::NORMAL_POWER` (8).
override RAA_AREA_NORMAL_POWER: i32 = 8;

// Below this weight sum the ratio estimator reports full visibility, `cpu::RATIO_EPSILON`.
const RAA_AREA_RATIO_EPSILON: f32 = 1e-6;

// The weight of one bilateral tap: the Gaussian times a depth term and a normal term
// (`max(dot, 0)^RAA_AREA_NORMAL_POWER`). Taps past the seventh have no weight.
fn raa_area_bilateral_weight(
    tap: u32,
    center_depth: f32,
    tap_depth: f32,
    center_normal: vec3<f32>,
    tap_normal: vec3<f32>,
) -> f32 {
    if tap >= 7u {
        return 0.0;
    }
    let depth_term = exp(-abs(tap_depth - center_depth) / (0.02 * max(center_depth, 1e-4)));
    let normal_term = pow(max(dot(center_normal, tap_normal), 0.0), f32(RAA_AREA_NORMAL_POWER));
    return RAA_AREA_BILATERAL_GAUSS[tap] * depth_term * normal_term;
}

// The ratio estimator (Heitz, Hill, McGuire, I3D 2018): visibility `s / u` clamped to [0, 1],
// where `s` is the weighted visible sum and `u` the weight sum. With no weight, full visibility.
fn raa_area_ratio_visibility(s: f32, u: f32) -> f32 {
    if u <= RAA_AREA_RATIO_EPSILON {
        return 1.0;
    }
    return clamp(s / u, 0.0, 1.0);
}

// The filtered value of `src` at `coord` along `axis` (x or y): seven taps
// `RAA_AREA_SPATIAL_STRIDE` pixels apart, each clamped to the image, a sky tap weighs 0, and
// each tap uses the same weight for both channels of each slot. A sky centre copies `src`.
fn raa_area_bilateral(src: texture_2d<f32>, coord: vec2<i32>, axis: vec2<i32>) -> vec4<f32> {
    let center = textureLoad(src, coord, 0);
    let center_depth = textureLoad(raa_area_current_depth, coord, 0).r;
    if center_depth <= 0.0 {
        return center;
    }
    let max_pixel = vec2<i32>(textureDimensions(raa_area_current_depth)) - vec2<i32>(1);
    let center_normal = normalize(textureLoad(raa_area_normal, coord, 0).xyz);
    var sum = vec4<f32>(0.0);
    var weight_sum = 0.0;
    for (var tap = 0u; tap < 7u; tap = tap + 1u) {
        let offset = (i32(tap) - 3) * RAA_AREA_SPATIAL_STRIDE * axis;
        let sp = clamp(coord + offset, vec2<i32>(0), max_pixel);
        let tap_depth = textureLoad(raa_area_current_depth, sp, 0).r;
        if tap_depth <= 0.0 {
            continue;
        }
        let tap_normal = normalize(textureLoad(raa_area_normal, sp, 0).xyz);
        let weight = raa_area_bilateral_weight(
            tap,
            center_depth,
            tap_depth,
            center_normal,
            tap_normal,
        );
        sum = sum + textureLoad(src, sp, 0) * weight;
        weight_sum = weight_sum + weight;
    }
    return sum / max(weight_sum, 1e-6);
}

// The horizontal half: filter the reprojected history along x into the raw layers. Bindings:
// src = the history pair group's write view, src_b = the same view, dst = scratch layer g.
@compute @workgroup_size(8, 8)
fn raa_gi_area_spatial_h(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(raa_area_dst);
    if any(id.xy >= size) {
        return;
    }
    let coord = vec2<i32>(id.xy);
    if raa_area_dispatch.index * 2u >= raa_area_dispatch.limit {
        textureStore(raa_area_dst, coord, textureLoad(raa_area_src, coord, 0));
        return;
    }
    textureStore(raa_area_dst, coord, raa_area_bilateral(raa_area_src, coord, vec2<i32>(1, 0)));
}

// The vertical half and the ratio estimator: filter `src` along y, and `src_b` when the frame
// has two history groups, then write V for the four slots of this mask layer. Bindings:
// src = scratch layer 2l, src_b = scratch layer 2l + 1 (the first again with one group), dst = mask
// layer l. Slots past the limit and sky pixels write 1.
@compute @workgroup_size(8, 8)
fn raa_gi_area_spatial_v(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(raa_area_dst);
    if any(id.xy >= size) {
        return;
    }
    let coord = vec2<i32>(id.xy);
    if textureLoad(raa_area_current_depth, coord, 0).r <= 0.0 {
        textureStore(raa_area_dst, coord, vec4<f32>(1.0));
        return;
    }
    let first = raa_area_bilateral(raa_area_src, coord, vec2<i32>(0, 1));
    var second = first;
    if raa_area_dispatch.pairs == 2u {
        second = raa_area_bilateral(raa_area_src_b, coord, vec2<i32>(0, 1));
    }
    let limit = raa_area_dispatch.limit;
    let slot = raa_area_dispatch.index * 4u;
    let visibility = vec4<f32>(
        select(1.0, raa_area_ratio_visibility(first.x, first.y), slot < limit),
        select(1.0, raa_area_ratio_visibility(first.z, first.w), slot + 1u < limit),
        select(1.0, raa_area_ratio_visibility(second.x, second.y), slot + 2u < limit),
        select(1.0, raa_area_ratio_visibility(second.z, second.w), slot + 3u < limit),
    );
    textureStore(raa_area_dst, coord, visibility);
}
