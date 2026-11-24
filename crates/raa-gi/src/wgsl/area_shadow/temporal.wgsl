// S44 mid-tier area-light shadows: the temporal half of the `gi.area-shadow` denoiser. One
// invocation per viewport pixel reprojects the previous frame's history pair through the S12
// motion target with `raa_temporal_raster_uv` of raa-taa's `REPROJECT_WGSL`, which this file is
// composed after, and blends it into this frame's trace output channel by channel.
//
// This is RAA's own code; it adapts no solstice source. It serves the ratio estimator of
// Heitz, Hill and McGuire, "Combining Analytic Direct Illumination and Stochastic Shadows",
// I3D 2018, whose spatial half is spatial.wgsl. `raa_area_blend_pair` and
// `raa_area_depth_consistent` mirror the CPU twins of
// crates/raa-gi/src/parts/area_shadow/cpu.rs (contract 'area-stochastic-cpu') with the same
// constants. FrameGpu `temporal.z` (camera history valid) gates the blend; `temporal.y` (TAA
// accumulation) is not required, so the shadow history works with TAA off.
//
// Group-0 binding registry (binding number to resource):
// 0 raa_area_frame (RaaTemporalFrame), 1 raa_area_current_depth, 2 raa_area_motion,
// 3 raa_area_raw, 4 raa_area_history, 5 raa_area_previous_depth, 6 raa_area_history_out,
// 7 raa_area_dispatch, 8 raa_area_normal.

// The per-frame dispatch record (16 B), contract 'area-shadow-denoise-wgsl': `index` is the
// history group or mask layer its bind group covers (g for the history pair, l for the mask),
// `usable` packs validity, two slot rejection bits, and the anti-lag flag, `pairs` the history group count (1 or 2) and `limit` the
// mid-tier slot limit.
struct AreaDispatchGpu {
    index: u32,
    usable: u32,
    pairs: u32,
    limit: u32,
}

@group(0) @binding(0) var<uniform> raa_area_frame: RaaTemporalFrame;
@group(0) @binding(1) var raa_area_current_depth: texture_2d<f32>;
@group(0) @binding(2) var raa_area_motion: texture_2d<f32>;
@group(0) @binding(3) var raa_area_raw: texture_2d<f32>;
@group(0) @binding(4) var raa_area_history: texture_2d<f32>;
@group(0) @binding(5) var raa_area_previous_depth: texture_2d<f32>;
@group(0) @binding(6) var raa_area_history_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var<uniform> raa_area_dispatch: AreaDispatchGpu;
@group(0) @binding(8) var raa_area_normal: texture_2d<f32>;

// Current sample weight of the legacy blend, `cpu::TEMPORAL_ALPHA` (0.1).
override RAA_AREA_TEMPORAL_ALPHA: f32 = 0.1;

// Relative depth tolerance of the reprojected history test, `cpu::DEPTH_TOLERANCE` (0.05).
override RAA_AREA_DEPTH_TOLERANCE: f32 = 0.05;

// Use the same current weight for the numerator and denominator of each slot.
fn raa_area_blend_pair(
    history: vec2<f32>, current: vec2<f32>, evidence: vec2<f32>, usable: bool, anti_lag: bool,
) -> vec2<f32> {
    if !usable {
        return current;
    }
    var alpha = RAA_AREA_TEMPORAL_ALPHA;
    if anti_lag && evidence.y > 1e-6 && history.y > 1e-6 {
        let disagreement = abs(evidence.x / evidence.y - history.x / history.y);
        alpha = mix(alpha, 1.0, smoothstep(0.15, 0.6, disagreement));
    }
    return history + (current - history) * alpha;
}

// Pool current ray evidence only on the receiver surface. In particular this gate
// sees visibility changes on stationary receivers whose own motion is zero.
fn raa_area_current_evidence(pixel: vec2<i32>, size: vec2<u32>, depth: f32) -> vec4<f32> {
    let position = raa_area_receiver_position(pixel, depth, raa_area_frame);
    let normal = normalize(textureLoad(raa_area_normal, pixel, 0).xyz);
    var evidence = vec4<f32>(0.0);
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let tap = pixel + vec2<i32>(x, y);
            if all(tap >= vec2<i32>(0)) && all(tap < vec2<i32>(size)) {
                let tap_depth = textureLoad(raa_area_current_depth, tap, 0).r;
                if raa_area_depth_consistent(tap_depth, depth) {
                    let tap_normal = normalize(textureLoad(raa_area_normal, tap, 0).xyz);
                    if dot(normal, tap_normal) >= 0.5 {
                        let delta = raa_area_receiver_position(tap, tap_depth, raa_area_frame) - position;
                        let weight = raa_area_receiver_weight(delta, normal, tap_normal, depth);
                        evidence += weight * textureLoad(raa_area_raw, tap, 0);
                    }
                }
            }
        }
    }
    return evidence;
}

// Whether a reprojected linear depth agrees with the current one: the previous depth must be
// positive and within `RAA_AREA_DEPTH_TOLERANCE` of it, relative to the current depth.
fn raa_area_depth_consistent(previous: f32, current: f32) -> bool {
    return previous > 0.0 && abs(previous - current) <= RAA_AREA_DEPTH_TOLERANCE * current;
}

struct RaaAreaHistory {
    pair: vec4<f32>,
    coverage: f32,
}

// Validate each texel of the bilinear footprint before loading its shadow pair.
fn raa_area_sample_history(sample_pos: vec2<f32>, size: vec2<u32>, depth: f32) -> RaaAreaHistory {
    let base = vec2<i32>(floor(sample_pos));
    let fraction = fract(sample_pos);
    var sum = vec4<f32>(0.0);
    var coverage = 0.0;
    for (var y = 0; y < 2; y++) {
        for (var x = 0; x < 2; x++) {
            let offset = vec2<i32>(x, y);
            let tap = base + offset;
            let axis_weight = select(1.0 - fraction, fraction, offset == vec2<i32>(1));
            let weight = axis_weight.x * axis_weight.y;
            if weight > 0.0 && all(tap >= vec2<i32>(0)) && all(tap < vec2<i32>(size))
                && raa_area_depth_consistent(textureLoad(raa_area_previous_depth, tap, 0).r, depth) {
                sum += weight * textureLoad(raa_area_history, tap, 0);
                coverage += weight;
            }
        }
    }
    if coverage <= 1e-6 {
        return RaaAreaHistory(vec4<f32>(0.0), 0.0);
    }
    return RaaAreaHistory(sum / coverage, min(coverage, 1.0));
}

// One invocation per viewport pixel: reproject the history pair through the S12 motion target
// and write the blended S, U, S, U. A pixel the trace left at depth 0 (sky or no cover) writes
// zeros.
@compute @workgroup_size(8, 8)
fn raa_gi_area_temporal(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(raa_area_history_out);
    if any(id.xy >= size) {
        return;
    }
    let pixel = vec2<i32>(id.xy);
    let depth = textureLoad(raa_area_current_depth, pixel, 0).r;
    if depth <= 0.0 {
        textureStore(raa_area_history_out, pixel, vec4<f32>(0.0));
        return;
    }
    let raw = textureLoad(raa_area_raw, pixel, 0);
    if (raa_area_dispatch.usable & 1u) == 0u || RAA_AREA_TEMPORAL_ALPHA == 1.0 {
        textureStore(raa_area_history_out, pixel, raw);
        return;
    }
    let uv = (vec2<f32>(id.xy) + vec2<f32>(0.5)) * raa_area_frame.viewport.zw;
    let motion = textureLoad(raa_area_motion, pixel, 0);
    let reprojected = raa_temporal_raster_uv(uv, motion, raa_area_frame.camera_jitter);
    if raa_area_frame.temporal.z == 0u || reprojected.z <= 0.5 {
        textureStore(raa_area_history_out, pixel, raw);
        return;
    }
    // Keep stationary, unjittered receivers on integer texel centres even at odd extents.
    let sample_pos = vec2<f32>(pixel) + (reprojected.xy - uv) * vec2<f32>(size);
    let history = raa_area_sample_history(sample_pos, size, depth);
    if history.coverage == 0.0 {
        textureStore(raa_area_history_out, pixel, raw);
        return;
    }
    let anti_lag = (raa_area_dispatch.usable & 8u) != 0u;
    var evidence = raw;
    if anti_lag {
        evidence = raa_area_current_evidence(pixel, size, depth);
    }
    let blended = vec4<f32>(
        raa_area_blend_pair(history.pair.xy, raw.xy, evidence.xy, (raa_area_dispatch.usable & 2u) == 0u, anti_lag),
        raa_area_blend_pair(history.pair.zw, raw.zw, evidence.zw, (raa_area_dispatch.usable & 4u) == 0u, anti_lag),
    );
    // Missing footprint coverage contributes the current pair, including at disocclusions.
    textureStore(raa_area_history_out, pixel, mix(raw, blended, history.coverage));
}
