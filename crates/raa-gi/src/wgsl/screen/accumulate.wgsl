// The temporal accumulation shared by the two irradiance sources of the screen part (the probe
// integrate and the ReSTIR GI denoise). Per pixel it reprojects the previous frame's history and
// its moments through the S12 motion target, keeps each of the four bilinear history taps only
// when its stored linear depth and normal describe this pixel's surface (a disocclusion test),
// tests the kept history against this frame's light (a lighting change test, below) and blends
// with the weight alpha = max(1 / n, 1 / historyMax), n the pixel's history length: the average
// of the last n frames, so a still view keeps converging until historyMax and a moving one keeps
// the history that stays valid. A rejected pixel starts again at n = 1.
//
// The change test never moves the history's colour: z = |current mean - history| / (sigma +
// slack), sigma^2 the larger of the current frame's spatial variance and the history's per-frame
// (temporal) variance, i.e. the noise one frame has. Up to z = clampGamma the history is kept
// whole. From 2 clampGamma a darkening cuts at once. Other changes need two consecutive
// departures in the same direction. A strong unconfirmed brightening holds the previous
// irradiance and moments, since a single probe sampling a bright emitter can resemble a
// lighting change. A confirmed cut keeps at most historyMin - 1 frames and restarts the moments
// from this frame's neighbourhood. The first and second moment signs mark pending brightening
// and darkening, respectively. The pass writes the history (irradiance rgb, occlusion a), the
// moments record, the a-trous input (irradiance rgb, variance of the mean a), the debug record
// and the statistics.
//
// RAA's own code. The moments, the variance of the mean and the variance-guided a-trous that
// follows (atrous.wgsl) are those of Schied et al., "Spatiotemporal Variance-Guided Filtering",
// HPG 2017; the clamp is the luminance form of raa-taa's neighbourhood box (taa.wgsl). Every
// function `raa_gi_acc_*` has a CPU twin in raa_gi::parts::screen::accumulate_cpu with the same
// name minus the prefix.
//
// Composed after probe_common.wgsl (GiFrameGpu, GiScreenGpu, bindings 0 1 2 7). Group-2
// bindings declared here: 3 motion, 4 previous history, 6 history out, 8 previous moments,
// 9 moments out, 10 a-trous input out, 11 debug out, 12 statistics counters.

@group(2) @binding(3) var raa_gi_motion: texture_2d<f32>;
@group(2) @binding(4) var raa_gi_previous: texture_2d<f32>;
@group(2) @binding(6) var raa_gi_output: texture_storage_2d<rgba16float, write>;
@group(2) @binding(8) var raa_gi_meta_previous: texture_2d<u32>;
@group(2) @binding(9) var raa_gi_meta_out: texture_storage_2d<rgba32uint, write>;
@group(2) @binding(10) var raa_gi_filter_out: texture_storage_2d<rgba16float, write>;
@group(2) @binding(11) var raa_gi_debug_out: texture_storage_2d<rgba32uint, write>;
@group(2) @binding(12) var<storage, read_write> raa_gi_counters: array<atomic<u32>, 4>;

// A history tap is kept when its linear depth is within this fraction of the depth expected for
// this pixel's surface (`accumulate_cpu::DEPTH_TOLERANCE`).
override RAA_GI_ACC_DEPTH_TOLERANCE: f32 = 0.04;
// ... and its normal within this cosine of this pixel's normal (`accumulate_cpu::NORMAL_COS`).
override RAA_GI_ACC_NORMAL_COS: f32 = 0.9;
// Below this bilinear weight of kept taps the history is rejected.
override RAA_GI_ACC_MIN_WEIGHT: f32 = 0.05;
// The relative slack of the change test, so a nearly noiseless signal does not trip it on
// rounding (`accumulate_cpu::CHANGE_SLACK`).
override RAA_GI_ACC_CHANGE_SLACK: f32 = 0.05;
// The shortest history the change test applies to (`accumulate_cpu::CHANGE_MIN`).
override RAA_GI_ACC_CHANGE_MIN: f32 = 4.0;
// The f16 ceiling of the stored values.
const RAA_GI_ACC_HALF_MAX: f32 = 65000.0;
// The horizon search of `raa_gi_screen_occlusion`: directions, steps per side and radius (m).
override RAA_GI_OCCLUSION_DIRECTIONS: u32 = 3u;
override RAA_GI_OCCLUSION_STEPS: u32 = 4u;
override RAA_GI_OCCLUSION_RADIUS: f32 = 0.9;

// Debug states of a pixel (`raa_gi_debug_out` z.x).
const RAA_GI_ACC_REJECTED: u32 = 0u;
const RAA_GI_ACC_KEPT: u32 = 1u;
const RAA_GI_ACC_CLAMPED: u32 = 2u;
const RAA_GI_ACC_UNSEEN: u32 = 3u;

fn raa_gi_acc_luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Signed octahedral encoding of a unit vector into [-1, 1]^2, and back.
fn raa_gi_acc_oct_encode(n: vec3<f32>) -> vec2<f32> {
    let p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    if n.z >= 0.0 {
        return p;
    }
    let signs = vec2<f32>(select(-1.0, 1.0, p.x >= 0.0), select(-1.0, 1.0, p.y >= 0.0));
    return (vec2<f32>(1.0) - abs(p.yx)) * signs;
}

fn raa_gi_acc_oct_decode(e: vec2<f32>) -> vec3<f32> {
    var v = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if v.z < 0.0 {
        let signs = vec2<f32>(select(-1.0, 1.0, v.x >= 0.0), select(-1.0, 1.0, v.y >= 0.0));
        let folded = (vec2<f32>(1.0) - abs(v.yx)) * signs;
        v = vec3<f32>(folded.x, folded.y, v.z);
    }
    return normalize(v);
}

// The moments record of one pixel: first and second moments of the luminance, history length,
// linear view depth (0 for no surface) and world normal.
struct RaaGiAccMeta {
    m1: f32,
    m2: f32,
    n: f32,
    depth: f32,
    normal: vec3<f32>,
}

fn raa_gi_acc_pack(m: RaaGiAccMeta) -> vec4<u32> {
    return vec4<u32>(
        bitcast<u32>(m.m1),
        bitcast<u32>(m.m2),
        pack2x16float(vec2<f32>(m.n, min(m.depth, RAA_GI_ACC_HALF_MAX))),
        pack2x16snorm(raa_gi_acc_oct_encode(m.normal)),
    );
}

fn raa_gi_acc_unpack(v: vec4<u32>) -> RaaGiAccMeta {
    let nd = unpack2x16float(v.z);
    return RaaGiAccMeta(
        bitcast<f32>(v.x),
        bitcast<f32>(v.y),
        nd.x,
        nd.y,
        raa_gi_acc_oct_decode(unpack2x16snorm(v.w)),
    );
}

// The blend weight of a pixel whose history now holds `n` frames: 1 / n, never below
// 1 / n_max. With n counting up from 1 the blend is the plain average of the frames seen.
fn raa_gi_acc_alpha(n: f32, n_max: f32) -> f32 {
    return 1.0 / max(min(n, n_max), 1.0);
}

// The next history length: one more frame, capped at n_max.
fn raa_gi_acc_next_length(n: f32, n_max: f32) -> f32 {
    return min(n + 1.0, max(n_max, 1.0));
}

// Whether a history tap describes this pixel's surface: its stored depth is within
// DEPTH_TOLERANCE of the depth this surface had in the previous view (`expected`, a moving camera)
// or of its current depth (a still camera over a moving object), and its normal is within
// NORMAL_COS. A tap with no surface (depth 0) never matches.
fn raa_gi_acc_tap_matches(
    stored_depth: f32,
    expected_depth: f32,
    current_depth: f32,
    stored_normal: vec3<f32>,
    normal: vec3<f32>,
) -> bool {
    if !(stored_depth > 0.0) {
        return false;
    }
    let moved = abs(stored_depth - expected_depth) <= RAA_GI_ACC_DEPTH_TOLERANCE * expected_depth;
    let still = abs(stored_depth - current_depth) <= RAA_GI_ACC_DEPTH_TOLERANCE * current_depth;
    return (moved || still) && dot(stored_normal, normal) >= RAA_GI_ACC_NORMAL_COS;
}

// The change test's band position, in [0, 1] (`accumulate_cpu::change_keep`): 1 up to z = gamma
// (inside), 0 from z = 2 gamma (a certain change), linear between, z = |mean - history| / (sigma +
// slack), sigma^2 = max(spatial_var, temporal_var).
fn raa_gi_acc_change_keep(history: f32, mean: f32, spatial_var: f32, temporal_var: f32, gamma: f32) -> f32 {
    let sigma = sqrt(max(max(spatial_var, temporal_var), 0.0));
    let slack = RAA_GI_ACC_CHANGE_SLACK * max(abs(history), abs(mean)) + 1.0e-4;
    let z = abs(mean - history) / (sigma + slack);
    return clamp(2.0 - z / max(gamma, 1.0e-3), 0.0, 1.0);
}

// The change test's action: x = cut the history, y = mark the pixel. A strong darkening
// cuts immediately. Require a second frame in the same direction for a brightening or
// a change in the transition band.
fn raa_gi_acc_change_action(keep: f32, confirmed: bool, brightening: bool) -> vec2<bool> {
    if keep >= 1.0 {
        return vec2<bool>(false, false);
    }
    let cut = confirmed || (keep <= 0.0 && !brightening);
    return vec2<bool>(cut, !cut);
}

// The variance the a-trous passes steer by: the variance of the accumulated mean, var / n, once
// the history holds four frames; the spatial variance (or the temporal one, if larger) before.
fn raa_gi_acc_filter_variance(temporal_var: f32, spatial_var: f32, n: f32) -> f32 {
    if n >= 4.0 {
        return max(temporal_var, 0.0) / n;
    }
    return max(max(spatial_var, temporal_var), 0.0);
}

// Interleaved gradient noise (solstice `common/math.wgsl:42-45`), seeded with the frame index.
fn raa_gi_probe_ign(pixel: vec2<f32>, frame: u32) -> f32 {
    let p = pixel + 5.588238 * f32(frame % 64u);
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

// Ground-truth-style occlusion in a horizon search along a few screen-space directions (S39,
// moved here from probe_integrate.wgsl).
fn raa_gi_screen_occlusion(pixel: vec2<f32>, world: vec3<f32>, normal: vec3<f32>, noise: f32) -> f32 {
    let distance = length(world - raa_gi_frame.camera_position.xyz);
    let pixel_radius =
        clamp(RAA_GI_OCCLUSION_RADIUS / distance * raa_gi_frame.viewport.y * 0.8, 4.0, 90.0);
    let size = vec2<i32>(raa_gi_frame.viewport.xy);

    var occlusion = 0.0;
    for (var d = 0u; d < RAA_GI_OCCLUSION_DIRECTIONS; d++) {
        let angle = (f32(d) + noise) / f32(RAA_GI_OCCLUSION_DIRECTIONS) * RAA_GI_PROBE_PI;
        let direction = vec2<f32>(cos(angle), sin(angle));
        for (var side = -1.0; side <= 1.0; side += 2.0) {
            var horizon = 0.0;
            for (var s = 1u; s <= RAA_GI_OCCLUSION_STEPS; s++) {
                let step_fraction = f32(s) - 0.5 + noise * 0.5;
                let offset =
                    direction * side * pixel_radius * step_fraction / f32(RAA_GI_OCCLUSION_STEPS);
                let sample_pixel = clamp(vec2<i32>(pixel + offset), vec2<i32>(0), size - vec2<i32>(1));
                let depth = textureLoad(raa_gi_depth, sample_pixel, 0);
                if raa_depth_is_far(depth) {
                    continue;
                }
                let p = raa_gi_world_position(vec2<f32>(sample_pixel) + vec2<f32>(0.5), depth);
                let delta = p - world;
                let length_delta = length(delta);
                let elevation = dot(delta, normal) / max(length_delta, 1.0e-4);
                let falloff = clamp(1.0 - length_delta / RAA_GI_OCCLUSION_RADIUS, 0.0, 1.0);
                horizon = max(horizon, elevation * falloff);
            }
            occlusion += horizon;
        }
    }
    return clamp(1.0 - occlusion / f32(RAA_GI_OCCLUSION_DIRECTIONS * 2u) * 1.6, 0.0, 1.0);
}

// What the accumulation needs of one pixel.
struct RaaGiAccInput {
    pixel: vec2<i32>,
    world: vec3<f32>,
    normal: vec3<f32>,
    // Linear view depth of the pixel.
    depth: f32,
    // This frame's irradiance (rgb) and occlusion (a).
    current: vec4<f32>,
    // False when this frame has no irradiance for the pixel (no probe on its surface): the
    // history's irradiance is held and only the occlusion blends.
    matched: bool,
    // Luminance mean and variance of the current irradiance around the pixel.
    spatial_mean: f32,
    spatial_var: f32,
    // Debug words: the hit code and the reservoir confidence.
    hit_code: u32,
    confidence: f32,
}

// Writes the far-pixel records: an empty history and no surface.
fn raa_gi_acc_store_far(pixel: vec2<i32>) {
    textureStore(raa_gi_output, pixel, vec4<f32>(0.0, 0.0, 0.0, 1.0));
    textureStore(raa_gi_meta_out, pixel, vec4<u32>(0u));
    textureStore(raa_gi_filter_out, pixel, vec4<f32>(0.0));
    textureStore(raa_gi_debug_out, pixel, vec4<u32>(0u));
}

// The previous history and moments of the pixel, from the bilinear taps that match its surface;
// `found` is false when no tap does or the history is not usable.
struct RaaGiAccHistory {
    color: vec4<f32>,
    m1: f32,
    m2: f32,
    n: f32,
    found: bool,
    // The signed share of the kept taps marked by the change test (positive for brighter).
    marked: f32,
}

fn raa_gi_acc_reproject(in: RaaGiAccInput) -> RaaGiAccHistory {
    var out = RaaGiAccHistory(vec4<f32>(0.0), 0.0, 0.0, 0.0, false, 0.0);
    let m = textureLoad(raa_gi_motion, in.pixel, 0);
    let center = vec2<f32>(in.pixel) + vec2<f32>(0.5);
    let previous_uv = center * raa_gi_frame.viewport.zw - m.xy;
    // The reprojection matrix belongs to the lit copy when one is available.
    // Accumulation taps require the immediately preceding real frame.
    if m.z > 0.5 || raa_gi_screen_params.history_valid == 0u || raa_gi_frame.camera_position.w > 1.0
        || any(previous_uv <= vec2<f32>(0.0)) || any(previous_uv >= vec2<f32>(1.0)) {
        return out;
    }
    let previous_clip = raa_gi_frame.previous_view_proj * vec4<f32>(in.world, 1.0);
    let expected = previous_clip.w;
    let size = raa_gi_frame.viewport.xy;
    let last = vec2<i32>(size) - vec2<i32>(1);
    let p = previous_uv * size - vec2<f32>(0.5);
    let base = vec2<i32>(floor(p));
    let f = p - floor(p);
    var total = 0.0;
    for (var t = 0u; t < 4u; t++) {
        let offset = vec2<i32>(i32(t & 1u), i32(t >> 1u));
        let tap = base + offset;
        if any(tap < vec2<i32>(0)) || any(tap > last) {
            continue;
        }
        let w = select(1.0 - f.x, f.x, offset.x == 1) * select(1.0 - f.y, f.y, offset.y == 1);
        if w <= 0.0 {
            continue;
        }
        let stored = raa_gi_acc_unpack(textureLoad(raa_gi_meta_previous, tap, 0));
        if !raa_gi_acc_tap_matches(stored.depth, expected, in.depth, stored.normal, in.normal) {
            continue;
        }
        out.color += textureLoad(raa_gi_previous, tap, 0) * w;
        out.m1 += abs(stored.m1) * w;
        out.m2 += abs(stored.m2) * w;
        out.marked += select(0.0, w, stored.m1 < 0.0) - select(0.0, w, stored.m2 < 0.0);
        out.n += stored.n * w;
        total += w;
    }
    if total < RAA_GI_ACC_MIN_WEIGHT {
        return RaaGiAccHistory(vec4<f32>(0.0), 0.0, 0.0, 0.0, false, 0.0);
    }
    out.color /= total;
    out.m1 /= total;
    out.m2 /= total;
    out.marked /= total;
    out.n = floor(out.n / total + 0.5);
    out.found = out.n >= 1.0;
    return out;
}

// The accumulation of one pixel: reprojection, clamp, blend, and every record written.
fn raa_gi_acc_resolve(in: RaaGiAccInput) {
    let n_max = raa_gi_screen_params.history_max;
    var history = raa_gi_acc_reproject(in);
    var state = RAA_GI_ACC_REJECTED;
    var mark = 0.0;
    var hold_irradiance = false;
    if history.found {
        state = RAA_GI_ACC_KEPT;
        if in.matched && history.n >= RAA_GI_ACC_CHANGE_MIN {
            let lum_h = raa_gi_acc_luminance(history.color.rgb);
            let temporal_var = max(history.m2 - history.m1 * history.m1, 0.0);
            let keep = raa_gi_acc_change_keep(lum_h, in.spatial_mean, in.spatial_var,
                temporal_var, raa_gi_screen_params.clamp_gamma);
            let brightening = in.spatial_mean > lum_h;
            let direction = select(-1.0, 1.0, brightening);
            let action = raa_gi_acc_change_action(keep, history.marked * direction >= 0.5, brightening);
            mark = select(0.0, direction, action.y);
            hold_irradiance = action.y && keep <= 0.0;
            if action.x {
                history.n = min(history.n, max(f32(raa_gi_screen_params.history_min) - 1.0, 0.0));
                // The moments restart from this frame's neighbourhood: kept, they would mix the
                // light before and after the change into a large temporal variance that hides the
                // rest of the change from the next frames' tests (the light would then fade over
                // the remaining history instead of following the change).
                history.m1 = in.spatial_mean;
                history.m2 = in.spatial_mean * in.spatial_mean + max(in.spatial_var, 0.0);
                state = RAA_GI_ACC_CLAMPED;
            }
        }
    } else {
        history = RaaGiAccHistory(vec4<f32>(0.0, 0.0, 0.0, 1.0), 0.0, 0.0, 0.0, false, 0.0);
    }

    let lum = raa_gi_acc_luminance(in.current.rgb);
    var n = history.n;
    var color = history.color;
    var m1 = history.m1;
    var m2 = history.m2;
    if in.matched && !hold_irradiance {
        n = raa_gi_acc_next_length(history.n, n_max);
        let alpha = raa_gi_acc_alpha(n, n_max);
        color = vec4<f32>(mix(history.color.rgb, in.current.rgb, alpha), 0.0);
        m1 = mix(history.m1, lum, alpha);
        m2 = mix(history.m2, lum * lum, alpha);
    } else if !history.found {
        state = RAA_GI_ACC_UNSEEN;
    }
    // The occlusion is known every frame: it blends at the weight of the history it has.
    let occlusion_alpha = raa_gi_acc_alpha(raa_gi_acc_next_length(history.n, n_max), n_max);
    color.a = mix(history.color.a, in.current.a, occlusion_alpha);
    color = min(color, vec4<f32>(RAA_GI_ACC_HALF_MAX));

    let temporal_var = max(m2 - m1 * m1, 0.0);
    let variance = min(raa_gi_acc_filter_variance(temporal_var, in.spatial_var, n), RAA_GI_ACC_HALF_MAX);
    textureStore(raa_gi_output, in.pixel, color);
    let marked_m1 = select(m1, -max(m1, 1.0e-30), mark > 0.0);
    let marked_m2 = select(m2, -max(m2, 1.0e-30), mark < 0.0);
    textureStore(raa_gi_meta_out, in.pixel,
        raa_gi_acc_pack(RaaGiAccMeta(marked_m1, marked_m2, n, in.depth, in.normal)));
    textureStore(raa_gi_filter_out, in.pixel, vec4<f32>(color.rgb, variance));
    let raw = min(in.current, vec4<f32>(RAA_GI_ACC_HALF_MAX));
    textureStore(raa_gi_debug_out, in.pixel, vec4<u32>(
        pack2x16float(raw.rg),
        pack2x16float(vec2<f32>(raw.b, n / max(n_max, 1.0))),
        pack2x16float(vec2<f32>(f32(state), in.confidence / max(raa_gi_screen_params.m_cap, 1.0))),
        in.hit_code,
    ));

    // Statistics over one pixel in 16 (every fourth row and column).
    if (in.pixel.x & 3) == 0 && (in.pixel.y & 3) == 0 {
        atomicAdd(&raa_gi_counters[0], u32(n));
        atomicAdd(&raa_gi_counters[1], select(0u, 1u, state == RAA_GI_ACC_REJECTED));
        atomicAdd(&raa_gi_counters[2], 1u);
        atomicAdd(&raa_gi_counters[3], select(0u, 1u, state == RAA_GI_ACC_CLAMPED));
    }
}
