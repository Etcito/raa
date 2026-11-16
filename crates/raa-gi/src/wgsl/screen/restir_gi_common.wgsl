// ReSTIR GI for the indirect diffuse of the screen part (GI mode `restir`), common share: the
// parameters, the reservoir record and codec, the receiver surface, the target function, the
// reconnection-shift Jacobian and the reservoir merge. Ouyang et al., "ReSTIR GI: Path
// Resampling for Real-Time Path Tracing", HPG 2021: a reservoir holds one secondary-path sample
// (the hit point x_s, its normal n_s and its outgoing radiance L toward the visible point),
// its unbiased contribution weight W and its confidence M; reuse resamples those samples
// between pixels and frames. RAA's own code; it builds on the binding-free ReSTIR core
// (area_shadow/restir_core.wgsl: binary16 and octahedral codec helpers, PCG numbers, WRS, GRIS
// weight, confidence clamp, boiling test, depth rejection, disc offsets) shared with the
// area-light ReSTIR DI. Every function has a CPU twin in
// raa_gi::parts::screen::restir_gi_cpu with the same name minus the `raa_rgi_` prefix.
//
// Composed after PRELUDE_WGSL, GI_VOLUME_WGSL, DEPTH_WGSL, probe_common.wgsl (GiFrameGpu and
// group-2 bindings 0-2, 7) and restir_core.wgsl. Group-2 bindings declared here: 8 the S12
// motion, 10 the previous frame's reservoirs (read), 11 this frame's reservoirs (read_write),
// 12 the RestirGiParamsGpu record.

// The per-frame ReSTIR GI parameters (32 B). `history_usable` is 1 when binding 10 holds the
// reservoirs of the previous frame of this allocation and the camera history is valid.
struct RestirGiParamsGpu {
    frame_index: u32,
    history_usable: u32,
    spatial_samples: u32,
    screen_traces: u32,
    spatial_radius: f32,
    m_cap: f32,
    boiling_factor: f32,
    normal_cos: f32,
}

// One reservoir in registers. `position`/`normal`/`radiance`: the sample point x_s, its unit
// normal n_s and the radiance it sends toward the visible point. `weight` is W in the solid-angle
// measure of the owner's visible point, `confidence` M (0: empty). `owner_normal` and
// `owner_depth` (linear view depth) describe the owner pixel's surface for the temporal
// rejection.
struct RaaRgiReservoir {
    position: vec3<f32>,
    normal: vec3<f32>,
    radiance: vec3<f32>,
    weight: f32,
    confidence: f32,
    owner_normal: vec3<f32>,
    owner_depth: f32,
}

// A receiver: world position, world unit normal, linear view depth; `valid` is false on a far
// pixel.
struct RaaRgiSurface {
    position: vec3<f32>,
    normal: vec3<f32>,
    depth: f32,
    valid: bool,
}

@group(2) @binding(8) var raa_rgi_motion: texture_2d<f32>;
@group(2) @binding(10) var<storage, read> raa_rgi_previous: array<vec4<u32>>;
@group(2) @binding(11) var<storage, read_write> raa_rgi_reservoirs: array<vec4<u32>>;
@group(2) @binding(12) var<uniform> raa_rgi_params: RestirGiParamsGpu;

// Largest Jacobian a spatial neighbour may carry before it is rejected (and its inverse the
// smallest): beyond it the shift sits at a grazing angle or folds distances by more than an
// order of magnitude, and the reused sample turns into a firefly.
const RAA_RGI_JACOBIAN_MAX: f32 = 10.0;
// Where a sky sample is placed along its direction, in metres: far enough that every reuse
// within the screen sees it along the same direction (its Jacobian tends to 1).
const RAA_RGI_SKY_DISTANCE: f32 = 1000.0;
const RAA_RGI_F32_MAX: f32 = 0x1.fffffep+127f;

fn raa_rgi_finite(x: f32) -> f32 {
    return select(0.0, x, x >= 0.0 && x <= RAA_RGI_F32_MAX);
}

// The empty reservoir: every field zero (M = 0).
fn raa_rgi_empty() -> RaaRgiReservoir {
    return RaaRgiReservoir(vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0), 0.0, 0.0,
        vec3<f32>(0.0, 0.0, 1.0), 0.0);
}

// Packs a reservoir into two vec4<u32>: a = (x_s as f32 bits, pack2x16float(W, M)),
// b = (pack2x16snorm(oct n_s), pack2x16float(L.rg), pack2x16float(L.b, owner depth),
// pack2x16snorm(oct owner normal)). W, M, L and the depth go through the core's binary16 clamp
// (NaN and negatives to 0, saturating at 65504).
fn raa_rgi_pack_a(r: RaaRgiReservoir) -> vec4<u32> {
    return vec4<u32>(
        bitcast<vec3<u32>>(r.position),
        pack2x16float(vec2<f32>(raa_restir_half_clamped(r.weight), raa_restir_half_clamped(r.confidence))),
    );
}

fn raa_rgi_pack_b(r: RaaRgiReservoir) -> vec4<u32> {
    return vec4<u32>(
        pack2x16snorm(raa_restir_oct_encode(r.normal)),
        pack2x16float(vec2<f32>(raa_restir_half_clamped(r.radiance.r), raa_restir_half_clamped(r.radiance.g))),
        pack2x16float(vec2<f32>(raa_restir_half_clamped(r.radiance.b), raa_restir_half_clamped(r.owner_depth))),
        pack2x16snorm(raa_restir_oct_encode(r.owner_normal)),
    );
}

fn raa_rgi_unpack(a: vec4<u32>, b: vec4<u32>) -> RaaRgiReservoir {
    let wm = unpack2x16float(a.w);
    let rg = unpack2x16float(b.y);
    let bd = unpack2x16float(b.z);
    return RaaRgiReservoir(
        bitcast<vec3<f32>>(a.xyz),
        raa_restir_oct_decode(unpack2x16snorm(b.x)),
        vec3<f32>(rg, bd.x),
        wm.x,
        wm.y,
        raa_restir_oct_decode(unpack2x16snorm(b.w)),
        bd.y,
    );
}

// The first of the two vec4 records of `pixel`: 2 * (y * width + x).
fn raa_rgi_index(pixel: vec2<u32>) -> u32 {
    return 2u * (pixel.y * u32(raa_gi_frame.viewport.x) + pixel.x);
}

fn raa_rgi_load_previous(pixel: vec2<u32>) -> RaaRgiReservoir {
    let i = raa_rgi_index(pixel);
    return raa_rgi_unpack(raa_rgi_previous[i], raa_rgi_previous[i + 1u]);
}

fn raa_rgi_load(pixel: vec2<u32>) -> RaaRgiReservoir {
    let i = raa_rgi_index(pixel);
    return raa_rgi_unpack(raa_rgi_reservoirs[i], raa_rgi_reservoirs[i + 1u]);
}

fn raa_rgi_store(pixel: vec2<u32>, r: RaaRgiReservoir) {
    let i = raa_rgi_index(pixel);
    raa_rgi_reservoirs[i] = raa_rgi_pack_a(r);
    raa_rgi_reservoirs[i + 1u] = raa_rgi_pack_b(r);
}

// The receiver of `pixel` from the S12 depth and normal targets.
fn raa_rgi_surface(pixel: vec2<i32>) -> RaaRgiSurface {
    let depth = textureLoad(raa_gi_depth, pixel, 0);
    if raa_depth_is_far(depth) {
        return RaaRgiSurface(vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0), 0.0, false);
    }
    let centre = vec2<f32>(pixel) + vec2<f32>(0.5);
    let view = raa_gi_view_position(centre, depth);
    return RaaRgiSurface(
        raa_gi_world_position(centre, depth),
        raa_gi_world_normal(pixel),
        -view.z,
        true,
    );
}

// The target function p_hat of a sample at a receiver: the luminance of the radiance it sends
// times the receiver's cosine, max(n_v . w, 0) with w toward x_s; 0 when the receiver lies
// behind the sample's surface (n_s . -w <= 0), whose radiance leaves only its front.
fn raa_rgi_target(x_v: vec3<f32>, n_v: vec3<f32>, x_s: vec3<f32>, n_s: vec3<f32>, radiance: vec3<f32>) -> f32 {
    let d = x_s - x_v;
    let len = length(d);
    if !(len > 0.0) {
        return 0.0;
    }
    let w = d / len;
    let cos_v = dot(n_v, w);
    if cos_v <= 0.0 || dot(n_s, -w) <= 0.0 {
        return 0.0;
    }
    return raa_rgi_finite(raa_restir_luminance(radiance) * cos_v);
}

// The Jacobian of the reconnection shift that moves a sample x_s (normal n_s) from the visible
// point x_q, where its W is a density in x_q's solid angle, to the visible point x_r:
// |d w_r / d w_q| = (|cos phi_r| / |cos phi_q|) (|x_q - x_s|^2 / |x_r - x_s|^2), phi the angle
// at x_s between n_s and the direction to the visible point (Ouyang et al. 2021, eq. 11 written
// for this direction: both solid angles are cos phi dA / d^2 of the same area element dA at
// x_s). A reused weight is W_q times this; 0 for a degenerate configuration.
fn raa_rgi_jacobian(x_r: vec3<f32>, x_q: vec3<f32>, x_s: vec3<f32>, n_s: vec3<f32>) -> f32 {
    let to_r = x_r - x_s;
    let to_q = x_q - x_s;
    let d2_r = dot(to_r, to_r);
    let d2_q = dot(to_q, to_q);
    if !(d2_r > 0.0) || !(d2_q > 0.0) {
        return 0.0;
    }
    let cos_r = abs(dot(n_s, to_r)) * inverseSqrt(d2_r);
    let cos_q = abs(dot(n_s, to_q)) * inverseSqrt(d2_q);
    if !(cos_q > 0.0) {
        return 0.0;
    }
    return raa_rgi_finite((cos_r / cos_q) * (d2_q / d2_r));
}

// A streaming merge state: the kept reservoir, the running weight sum, the summed confidence
// and the kept sample's target at the receiver.
struct RaaRgiMerge {
    kept: RaaRgiReservoir,
    w_sum: f32,
    confidence: f32,
    target_value: f32,
}

fn raa_rgi_merge_start() -> RaaRgiMerge {
    return RaaRgiMerge(raa_rgi_empty(), 0.0, 0.0, 0.0);
}

// Merges reservoir `r`, whose sample has target `target_value` at the receiver (already
// multiplied by the shift Jacobian), with resampling weight target * W * M (Ouyang et al. 2021,
// Algorithm 3 and 4 MERGE). `xi` is a uniform number in [0, 1).
fn raa_rgi_merge(state: ptr<function, RaaRgiMerge>, r: RaaRgiReservoir, target_value: f32, xi: f32) {
    let w = raa_rgi_finite(target_value * r.weight * r.confidence);
    (*state).w_sum = (*state).w_sum + w;
    (*state).confidence = (*state).confidence + r.confidence;
    if raa_restir_wrs_accept((*state).w_sum, w, xi) {
        (*state).kept = r;
        (*state).target_value = target_value;
    }
}

// The cosine-weighted direction of uv in [0, 1)^2 around `n`: cos theta = sqrt(1 - u), pdf
// cos theta / pi.
fn raa_rgi_cosine_direction(uv: vec2<f32>, n: vec3<f32>) -> vec3<f32> {
    let cos_theta = sqrt(max(1.0 - uv.x, 0.0));
    let sin_theta = sqrt(max(uv.x, 0.0));
    let phi = 6.28318530718 * uv.y;
    return raa_gi_probe_basis(n) * vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}
