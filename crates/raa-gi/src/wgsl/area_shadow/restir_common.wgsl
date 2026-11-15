// M7 ReSTIR DI for the area-light shadows (S12 order 43), common share 1 of contract
// 'area-restir-wgsl': the group-2 bindings 12-19, the `RestirParamsGpu` and
// `RaaRestirReservoir` records, and the reservoir codec and reuse helpers every ReSTIR stage
// calls. This is RAA's own code from Bitterli et al. 2020 (ReSTIR), Wyman and Panteleev,
// HPG 2021, and Lin et al. 2022 (GRIS). Bevy Solari's restir.wesl is not among the supplied
// checkouts and no URL for it is given, so nothing is adapted or copied from it and no
// licence header applies. NVIDIA RTXDI is never read or copied.
//
// Every function mirrors its CPU twin of contract 'area-restir-cpu' (module
// raa_gi::parts::area_shadow::restir_cpu); the WGSL names keep the same names with the
// `raa_restir_` prefix. `RaaTemporalFrame` comes from raa_taa::REPROJECT_WGSL, which every
// composed stage module includes; share 2 adds the surface, spherical-rectangle sampling,
// target and LTC luminance functions to this file.
//
// Group-2 binding registry of contract 'area-restir-wgsl' (binding number to resource):
// 12 raa_restir_motion (S12 motion), 13 raa_restir_frame: RaaTemporalFrame (only temporal.z is
// read), 14 raa_restir_previous_depth (the depth pair's read view), 15 raa_restir_params:
// RestirParamsGpu (64 B), 16 raa_restir_current (reservoir storage, read_write), 17
// raa_restir_history (reservoir history, read_write), 18 raa_restir_ltc (filterable atlas),
// 19 raa_restir_material (S12 material). Bindings 0-11 belong to the composed stage files
// (restir_initial.wgsl, restir_temporal.wgsl, restir_spatial.wgsl): 0 is the 432-byte
// GiFrameGpu this file reads through raa_gi_frame. Binding 9 is not in this layout.

// The per-frame ReSTIR parameters (64 B), little-endian. `frame_index` is the ReSTIR frame
// counter; `history_usable` is 1 when raa_restir_history holds a committed frame of this
// allocation; `remap` lane s is the current slot of previous slot s, or 0xFFFFFFFF.
struct RestirParamsGpu {
    frame_index: u32,
    history_usable: u32,
    candidates: u32,
    spatial_samples: u32,
    remap: array<vec4<u32>, 2>,
    spatial_radius: f32,
    m_cap: f32,
    boiling_factor: f32,
    normal_cos: f32,
}

// One reservoir in registers: the stable area-stochastic slot (0..7, 0xFF empty, bits 8-31 of
// the packed word are zero), the point in the light rectangle's unit square, the unbiased
// contribution weight W, the confidence M and the octahedral world normal of the owner pixel.
struct RaaRestirReservoir {
    slot: u32,
    uv: vec2<f32>,
    weight: f32,
    confidence: f32,
    normal_oct: vec2<f32>,
}

// The S12 motion target of the pixel (the lit frame's reprojection input).
@group(2) @binding(12) var raa_restir_motion: texture_2d<f32>;
// The lit FrameGpu (S10) of this frame; only temporal.z (camera history valid) is read.
@group(2) @binding(13) var<uniform> raa_restir_frame: RaaTemporalFrame;
// The previous frame's linear view depth (the depth pair's read view).
@group(2) @binding(14) var raa_restir_previous_depth: texture_2d<f32>;
// The per-frame ReSTIR parameters (64 B).
@group(2) @binding(15) var<uniform> raa_restir_params: RestirParamsGpu;
// This frame's reservoir records, one vec4<u32> per pixel at index y * width + x.
@group(2) @binding(16) var<storage, read_write> raa_restir_current: array<vec4<u32>>;
// The previous committed frame's reservoir records (next frame's history).
@group(2) @binding(17) var<storage, read_write> raa_restir_history: array<vec4<u32>>;
// The part's own BRDF/LTC atlas from raa_pipeline::create_brdf_atlas, read through
// `raa_ltc_uv` with the group-1 `raa_linear_clamp` sampler.
@group(2) @binding(18) var raa_restir_ltc: texture_2d<f32>;
// The S12 material target: sRGB-encoded F0 rgb and metallic a.
@group(2) @binding(19) var raa_restir_material: texture_2d<f32>;

// Relative depth tolerance of the reservoir depth rejection.
const RAA_RESTIR_DEPTH_TOLERANCE: f32 = 0.1;
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

// The empty reservoir: slot 0xFF and every other field zero.
fn raa_restir_empty() -> RaaRestirReservoir {
    return RaaRestirReservoir(0xFFu, vec2<f32>(0.0), 0.0, 0.0, vec2<f32>(0.0));
}

// Packs a reservoir into its four WGSL words: x = slot & 0xFF, y = pack2x16float(uv in the
// unit square), z = pack2x16float((W, M) in [0, 65504], NaN -> 0) and w =
// pack2x16float(normal_oct).
fn raa_restir_pack(r: RaaRestirReservoir) -> vec4<u32> {
    return vec4<u32>(
        r.slot & 0xFFu,
        pack2x16float(vec2<f32>(
            raa_restir_uv_clamped(r.uv.x),
            raa_restir_uv_clamped(r.uv.y),
        )),
        pack2x16float(vec2<f32>(
            raa_restir_half_clamped(r.weight),
            raa_restir_half_clamped(r.confidence),
        )),
        pack2x16float(r.normal_oct),
    );
}

// Unpacks the four WGSL words of `raa_restir_pack`; the slot keeps its low eight bits.
fn raa_restir_unpack(v: vec4<u32>) -> RaaRestirReservoir {
    let weight_confidence = unpack2x16float(v.z);
    return RaaRestirReservoir(
        v.x & 0xFFu,
        unpack2x16float(v.y),
        weight_confidence.x,
        weight_confidence.y,
        unpack2x16float(v.w),
    );
}

// The reservoir storage index of `pixel`: y * width + x, width = u32(viewport.x).
fn raa_restir_index(pixel: vec2<u32>) -> u32 {
    return pixel.y * u32(raa_gi_frame.viewport.x) + pixel.x;
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

// The current slot of the previous slot `slot` from raa_restir_params.remap; 0xFFFFFFFF for a
// slot >= 8 or a lane that did not survive.
fn raa_restir_remap(slot: u32) -> u32 {
    if slot >= 8u {
        return 0xFFFFFFFFu;
    }
    return raa_restir_params.remap[slot / 4u][slot % 4u];
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
