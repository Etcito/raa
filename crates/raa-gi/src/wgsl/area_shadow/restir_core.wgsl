// The binding-free core of RAA's ReSTIR (area-light DI, contract 'area-restir-wgsl', and the
// indirect-diffuse ReSTIR GI of the screen part): the constants, the binary16 and octahedral
// codec helpers, the PCG random numbers, weighted reservoir sampling, the GRIS contribution
// weight, the confidence clamp, the boiling filter test, the depth rejection and the
// spatial-reuse disc offsets. Every function mirrors its CPU twin of contract 'area-restir-cpu'
// (module raa_gi::parts::area_shadow::restir_cpu). It declares no binding, so any ReSTIR
// module composes it ahead of its own bindings (restir_common.wgsl for DI,
// screen/restir_gi_common.wgsl for GI). RAA's own code from Bitterli et al. 2020 (ReSTIR) and
// Lin et al. 2022 (GRIS).

// Relative depth tolerance of the reservoir depth rejection.
override RAA_RESTIR_DEPTH_TOLERANCE: f32 = 0.1;

const RAA_RESTIR_TAU: f32 = 6.28318530718;

const RAA_RESTIR_LUMINANCE: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

// The largest finite binary16 magnitude, the weight and confidence clamp.
const RAA_RESTIR_HALF_MAX: f32 = 65504.0;

// `sign(x)` with `sign(0) = +1`, the octahedral convention (WGSL's `sign` would give 0).
fn raa_restir_sign_one(x: f32) -> f32 {
    return select(1.0, -1.0, x < 0.0);
}

// The IEEE binary16 pair codec's rounding domain: NaN and negatives become 0, the largest
// finite binary16 magnitude saturates. WGSL's pack2x16float then rounds to nearest even.
fn raa_restir_half_clamped(x: f32) -> f32 {
    if x != x {
        return 0.0;
    }
    return clamp(x, 0.0, RAA_RESTIR_HALF_MAX);
}

// The uv half of the reservoir codec: NaN becomes 0, the point is kept in the unit square.
fn raa_restir_uv_clamped(x: f32) -> f32 {
    if x != x {
        return 0.0;
    }
    return clamp(x, 0.0, 1.0);
}

// PCG hash, the same 32-bit integer hash as raa_gi_probe_pcg of probe_common.wgsl.
fn raa_restir_pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Hash to a float in [0, 1), the twin of `raa_gi_probe_hash_to_unit`.
fn raa_restir_random(seed: u32) -> f32 {
    return f32(raa_restir_pcg(seed) >> 8u) * (1.0 / 16777216.0);
}

// Octahedral encoding of a unit vector: p = n / (|x| + |y| + |z|), and for z < 0 the folded
// (1 - |p.yx|) * sign(p.xy) with sign(0) = +1. A zero vector encodes (0, 0).
fn raa_restir_oct_encode(n: vec3<f32>) -> vec2<f32> {
    let l1 = abs(n.x) + abs(n.y) + abs(n.z);
    if !(l1 > 0.0) {
        return vec2<f32>(0.0);
    }
    let p = n.xy / l1;
    if n.z / l1 < 0.0 {
        return vec2<f32>(
            (1.0 - abs(p.y)) * raa_restir_sign_one(p.x),
            (1.0 - abs(p.x)) * raa_restir_sign_one(p.y),
        );
    }
    return p;
}

// The unit vector of an octahedral coordinate: the inverse of `raa_restir_oct_encode`,
// normalised.
fn raa_restir_oct_decode(e: vec2<f32>) -> vec3<f32> {
    let z = 1.0 - abs(e.x) - abs(e.y);
    var xy = e;
    if z < 0.0 {
        xy = vec2<f32>(
            (1.0 - abs(e.y)) * raa_restir_sign_one(e.x),
            (1.0 - abs(e.x)) * raa_restir_sign_one(e.y),
        );
    }
    return normalize(vec3<f32>(xy, z));
}

// Weighted reservoir sampling: w_sum already includes w; true when w_sum > 0 and
// xi * w_sum < w.
fn raa_restir_wrs_accept(w_sum: f32, w: f32, xi: f32) -> bool {
    return w_sum > 0.0 && xi * w_sum < w;
}

// The GRIS contribution weight w_sum / target; 0 when target <= 0 or the result is not finite.
// (`target` is a WGSL reserved keyword, so the parameter is `target_value`; the CPU twin calls
// it `target`.)
fn raa_restir_contribution_weight(w_sum: f32, target_value: f32) -> f32 {
    if target_value <= 0.0 {
        return 0.0;
    }
    let result = w_sum / target_value;
    // `result - result` is 0 only for a finite result (NaN != 0), so this is the finite test.
    if result - result != 0.0 {
        return 0.0;
    }
    return result;
}

// The temporally clamped confidence: max(min(previous, cap * current), 0).
fn raa_restir_clamp_confidence(previous: f32, current: f32, cap: f32) -> f32 {
    return max(min(previous, cap * current), 0.0);
}

// The boiling filter: true when the mean is not positive or the weight stays within
// factor times the mean.
fn raa_restir_boiling_keep(weight: f32, mean: f32, factor: f32) -> bool {
    return mean <= 0.0 || weight <= factor * mean;
}

// The reservoir depth rejection: both depths are positive and within
// RAA_RESTIR_DEPTH_TOLERANCE of the centre depth.
fn raa_restir_depth_similar(neighbour: f32, centre: f32) -> bool {
    return centre > 0.0 && neighbour > 0.0
        && abs(neighbour - centre) <= RAA_RESTIR_DEPTH_TOLERANCE * centre;
}

// The pixel offset of spatial-reuse neighbour k: a radius-scaled disc point drawn from the
// per-pixel, per-frame, per-neighbour seed, rounded to whole pixels.
fn raa_restir_spatial_offset(pixel: vec2<u32>, frame: u32, k: u32, radius: f32) -> vec2<i32> {
    let seed = pixel.x * 7919u + pixel.y * 104729u + frame * 31337u + k * 977u;
    let r = radius * sqrt(raa_restir_random(seed));
    let a = RAA_RESTIR_TAU * raa_restir_random(seed + 1u);
    return vec2<i32>(round(vec2<f32>(r * cos(a), r * sin(a))));
}

// The luminance of a linear RGB colour, with the weights of contract 'area-restir-cpu'.
fn raa_restir_luminance(c: vec3<f32>) -> f32 {
    return dot(c, RAA_RESTIR_LUMINANCE);
}

// The sRGB decode of one component: c / 12.92 below 0.04045, else ((c + 0.055) / 1.055)^2.4.
fn raa_restir_srgb_decode(c: vec3<f32>) -> vec3<f32> {
    return select(
        c / 12.92,
        pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4)),
        c > vec3<f32>(0.04045),
    );
}
