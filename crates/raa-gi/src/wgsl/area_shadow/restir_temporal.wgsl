// M7 ReSTIR DI for the area-light shadows (S12 order 43), the temporal-reuse stage of contract
// 'area-restir-wgsl': per viewport pixel, the initial reservoir c of raa_restir_current and the
// previous frame's reservoir t of raa_restir_history (reprojected through the S12 motion,
// rejected on depth, normal or slot remap, its confidence clamped) are combined with GRIS (Lin
// et al. 2022), then a workgroup boiling filter (Wyman and Panteleev, HPG 2021, RAA's own
// form) runs and the result goes back into raa_restir_current. RAA's own code; nothing is
// copied from Bevy Solari (not supplied) or NVIDIA RTXDI. Composed after the area-shadow trace
// module, raa_taa::REPROJECT_WGSL and restir_common.wgsl by restir.rs
// `restir_temporal_module_wgsl`. Group-2 use: 0, 1, 2, 8, 12, 13, 14, 15, 16, 17 and 19;
// group-1 use: 12; group 0 is never read.

// The salt that keeps the temporal selection seeds apart from the initial kernel's.
const RAA_RESTIR_TEMPORAL_SALT: u32 = 0x9E3779B9u;

// The boiling-filter reduction: each invocation's q = target * W, then the workgroup mean of
// the positive q that local invocation 0 computes between the two barriers.
var<workgroup> raa_restir_boiling_q: array<f32, 64>;
var<workgroup> raa_restir_boiling_mean: f32;

// One resampled reservoir and the target of its selected sample at the current surface.
struct RaaRestirTemporalResult {
    reservoir: RaaRestirReservoir,
    target_value: f32,
}

// The selection seed of input j (0 = current, 1 = temporal): the initial kernel's
// pixel.x * 7919 + pixel.y * 104729 + frame * 31337 + j * 977, plus the temporal salt, with
// wrapping u32 arithmetic.
fn raa_restir_temporal_seed(pixel: vec2<u32>, frame: u32, j: u32) -> u32 {
    return pixel.x * 7919u + pixel.y * 104729u + frame * 31337u + j * 977u
        + RAA_RESTIR_TEMPORAL_SALT;
}

// A finite non-negative value, or 0.
fn raa_restir_temporal_finite(x: f32) -> f32 {
    return select(0.0, x, x >= 0.0 && x <= RAA_RESTIR_F32_MAX);
}

// The target of reservoir `r` re-evaluated at `surface`; 0 for an empty reservoir or a slot
// that holds no area light this frame.
fn raa_restir_temporal_target(surface: RaaRestirSurface, r: RaaRestirReservoir) -> f32 {
    let packed = raa_restir_slot_light(r.slot);
    if packed == 0xFFFFFFFFu {
        return 0.0;
    }
    return raa_restir_target(surface, raa_local_lights[packed], r.uv);
}

// The previous frame's reservoir of `pixel` with its slot remapped to the current slot, or
// the empty reservoir (slot 0xFF) when the history, the camera history or the reprojection is
// not usable, the depth or normal differs, the reservoir is empty or its slot did not survive.
fn raa_restir_temporal_candidate(
    pixel: vec2<u32>,
    surface: RaaRestirSurface,
) -> RaaRestirReservoir {
    let empty = raa_restir_empty();
    if !surface.valid || raa_restir_params.history_usable == 0u
        || raa_restir_frame.temporal.z == 0u {
        return empty;
    }
    let size = vec2<i32>(raa_gi_frame.viewport.xy);
    let uv = (vec2<f32>(pixel) + vec2<f32>(0.5)) * raa_gi_frame.viewport.zw;
    let rep = raa_temporal_surface_uv(uv, textureLoad(raa_restir_motion, vec2<i32>(pixel), 0));
    if !(rep.z > 0.5) {
        return empty;
    }
    let prev = clamp(vec2<i32>(rep.xy * vec2<f32>(size)), vec2<i32>(0), size - vec2<i32>(1));
    var t = raa_restir_unpack(raa_restir_history[raa_restir_index(vec2<u32>(prev))]);
    let previous_depth = textureLoad(raa_restir_previous_depth, prev, 0).r;
    if !raa_restir_depth_similar(previous_depth, surface.view_depth) {
        return empty;
    }
    if !(dot(raa_restir_oct_decode(t.normal_oct), surface.normal) >= raa_restir_params.normal_cos) {
        return empty;
    }
    if t.slot == 0xFFu {
        return empty;
    }
    // 0xFFFFFFFF for a slot that did not survive; any lane outside the 8 slots is refused too.
    let slot = raa_restir_remap(t.slot);
    if slot >= 8u {
        return empty;
    }
    t.slot = slot;
    return t;
}

// The temporal resampling of one in-viewport pixel: c alone (with its target) without a
// usable t, else the GRIS combination of c and the M-clamped t. An invalid surface gives the
// empty reservoir.
fn raa_restir_temporal_resample(pixel: vec2<u32>) -> RaaRestirTemporalResult {
    let c = raa_restir_unpack(raa_restir_current[raa_restir_index(pixel)]);
    let surface = raa_restir_surface(vec2<i32>(pixel));
    if !surface.valid {
        return RaaRestirTemporalResult(raa_restir_empty(), 0.0);
    }
    let target_c = raa_restir_temporal_target(surface, c);
    var t = raa_restir_temporal_candidate(pixel, surface);
    if t.slot == 0xFFu {
        return RaaRestirTemporalResult(c, target_c);
    }
    t.confidence = raa_restir_clamp_confidence(
        t.confidence,
        max(c.confidence, 1.0),
        raa_restir_params.m_cap,
    );
    let target_t = raa_restir_temporal_target(surface, t);
    let m_sum = c.confidence + t.confidence;
    let m_c = raa_restir_ratio(c.confidence, m_sum);
    let m_t = raa_restir_ratio(t.confidence, m_sum);
    let w_c = raa_restir_temporal_finite(m_c * target_c * c.weight);
    let w_t = raa_restir_temporal_finite(m_t * target_t * t.weight);

    // Streaming weighted reservoir sampling over (c, t).
    let frame = raa_restir_params.frame_index;
    var kept = RaaRestirTemporalResult(raa_restir_empty(), 0.0);
    let xi = vec2<f32>(
        raa_restir_random(raa_restir_temporal_seed(pixel, frame, 0u)),
        raa_restir_random(raa_restir_temporal_seed(pixel, frame, 1u)),
    );
    var w_sum = w_c;
    if raa_restir_wrs_accept(w_sum, w_c, xi.x) {
        kept = RaaRestirTemporalResult(c, target_c);
    }
    w_sum = w_sum + w_t;
    if raa_restir_wrs_accept(w_sum, w_t, xi.y) {
        kept = RaaRestirTemporalResult(t, target_t);
    }
    kept.reservoir.weight = raa_restir_contribution_weight(w_sum, kept.target_value);
    kept.reservoir.confidence = m_sum;
    kept.reservoir.normal_oct = raa_restir_oct_encode(surface.normal);
    return kept;
}

// Every invocation of the workgroup reaches both barriers (no return before them):
// invocations outside the viewport carry q = 0 and skip the final write.
@compute @workgroup_size(8, 8)
fn raa_gi_area_restir_temporal(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let inside = all(id.xy < vec2<u32>(raa_gi_frame.viewport.xy));
    var result = RaaRestirTemporalResult(raa_restir_empty(), 0.0);
    if inside {
        result = raa_restir_temporal_resample(id.xy);
    }
    let q = raa_restir_temporal_finite(result.target_value * result.reservoir.weight);
    raa_restir_boiling_q[lane] = q;
    workgroupBarrier();
    if lane == 0u {
        var sum = 0.0;
        var count = 0u;
        for (var i = 0u; i < 64u; i = i + 1u) {
            let value = raa_restir_boiling_q[i];
            if value > 0.0 {
                sum = sum + value;
                count = count + 1u;
            }
        }
        raa_restir_boiling_mean = sum / f32(max(count, 1u));
    }
    workgroupBarrier();
    if !raa_restir_boiling_keep(q, raa_restir_boiling_mean, raa_restir_params.boiling_factor) {
        result.reservoir.weight = 0.0;
    }
    if inside {
        raa_restir_current[raa_restir_index(id.xy)] = raa_restir_pack(result.reservoir);
    }
}
