// M7 ReSTIR DI for the area-light shadows (S12 order 43), the spatial-reuse and resolve stage
// of contract 'area-restir-wgsl'. Per viewport pixel p, the canonical reservoir of
// raa_restir_current (written by the temporal stage) and up to raa_restir_params.spatial_samples
// neighbour reservoirs within raa_restir_params.spatial_radius pixels (rejected on depth,
// normal, tangent-plane separation or an empty reservoir) are combined by GRIS with the
// generalized balance heuristic (Lin et al. 2022): m_i = M_i T_i(y_i) / sum_j M_j T_j(y_i), every target at participant j's
// own current-frame surface, and w_i = m_i T_p(y_i) W_i. The merged reservoir becomes the next
// frame's temporal history in raa_restir_history. The resolve writes (S, U~, S, U~) into raw
// layer 0 of raa_area_raw_out, the history group 0 (slots 0 and 1) of the M6 denoiser: U~ is
// the analytic unshadowed LTC luminance summed over the slot-table area lights and S = T_p W v,
// v the pixel's second visibility ray towards the merged sample (`raa_area_ray_visibility` of
// trace.wgsl), so the ratio estimator's V = S / U~ is 0 in an umbra.
//
// RAA's own code; nothing is copied from Bevy Solari (not supplied) or NVIDIA RTXDI. Composed
// after the area-shadow trace module, raa_taa::REPROJECT_WGSL and restir_common.wgsl by
// crates/raa-gi/src/parts/area_shadow/restir.rs `restir_spatial_module_wgsl`. The light point,
// slot lookup, target and LTC luminance come from the share-2 Handoff at restir_common.wgsl:391
// `raa_restir_light_point`. Group-2 use: 0, 1, 2, 4, 5, 8, 10, 11, 15, 16, 17, 18 and 19;
// group-1 use: 6, 12, 17 and 18; group 0 is never read.

// The salt that keeps the spatial selection seeds apart from the initial and temporal ones.
const RAA_RESTIR_SPATIAL_SALT: u32 = 0x85EBCA6Bu;

// The participant count bound: the canonical reservoir and at most four neighbours.
const RAA_RESTIR_SPATIAL_PARTICIPANTS: u32 = 5u;

// The selection seed of participant i (0 = canonical): pixel.x * 7919 + pixel.y * 104729 +
// frame * 31337 + i * 977, plus the spatial salt, with wrapping u32 arithmetic.
fn raa_restir_spatial_seed(pixel: vec2<u32>, frame: u32, i: u32) -> u32 {
    return pixel.x * 7919u + pixel.y * 104729u + frame * 31337u + i * 977u
        + RAA_RESTIR_SPATIAL_SALT;
}

// A finite non-negative value, or 0.
fn raa_restir_spatial_finite(x: f32) -> f32 {
    return select(0.0, x, x >= 0.0 && x <= RAA_RESTIR_F32_MAX);
}

@compute @workgroup_size(8, 8)
fn raa_gi_area_restir_spatial(@builtin(global_invocation_id) id: vec3<u32>) {
    let viewport = vec2<u32>(raa_gi_frame.viewport.xy);
    if any(id.xy >= viewport) {
        return;
    }
    let pixel = id.xy;
    let index = raa_restir_index(pixel);
    let surface_p = raa_restir_surface(vec2<i32>(pixel));
    if !surface_p.valid {
        raa_restir_history[index] = raa_restir_pack(raa_restir_empty());
        textureStore(raa_area_raw_out, vec2<i32>(pixel), 0, vec4<f32>(0.0));
        return;
    }

    // The participants: the canonical reservoir with surface_p, then every accepted neighbour
    // with its own surface.
    var surfaces: array<RaaRestirSurface, 5>;
    var reservoirs: array<RaaRestirReservoir, 5>;
    surfaces[0] = surface_p;
    reservoirs[0] = raa_restir_unpack(raa_restir_current[index]);
    var count = 1u;
    let frame = raa_restir_params.frame_index;
    let samples = min(raa_restir_params.spatial_samples, RAA_RESTIR_SPATIAL_PARTICIPANTS - 1u);
    for (var k = 0u; k < samples; k = k + 1u) {
        let offset = raa_restir_spatial_offset(pixel, frame, k, raa_restir_params.spatial_radius);
        if all(offset == vec2<i32>(0)) {
            continue;
        }
        let q = vec2<i32>(pixel) + offset;
        if any(q < vec2<i32>(0)) || any(q >= vec2<i32>(viewport)) {
            continue;
        }
        let surface_q = raa_restir_surface(q);
        if !surface_q.valid
            || !raa_restir_depth_similar(surface_q.view_depth, surface_p.view_depth)
            || !(dot(surface_q.normal, surface_p.normal) >= raa_restir_params.normal_cos) {
            continue;
        }
        if raa_area_receiver_weight(surface_q.position - surface_p.position,
            surface_p.normal, surface_q.normal, surface_p.view_depth) <= 0.0 {
            continue;
        }
        let neighbour = raa_restir_unpack(raa_restir_current[raa_restir_index(vec2<u32>(q))]);
        if neighbour.slot == 0xFFu {
            continue;
        }
        surfaces[count] = surface_q;
        reservoirs[count] = neighbour;
        count = count + 1u;
    }

    // GRIS over the participants with streaming weighted reservoir sampling. A sample whose
    // slot holds no area light this frame has a zero target everywhere and is never kept.
    var merged = raa_restir_empty();
    var kept_target = 0.0;
    var m_sum = 0.0;
    var w_sum = 0.0;
    for (var i = 0u; i < count; i = i + 1u) {
        let r = reservoirs[i];
        m_sum = m_sum + r.confidence;
        let packed = raa_restir_slot_light(r.slot);
        if packed == 0xFFFFFFFFu {
            continue;
        }
        let light = raa_local_lights[packed];
        var own = 0.0;
        var at_p = 0.0;
        var denominator = 0.0;
        for (var j = 0u; j < count; j = j + 1u) {
            let t = raa_restir_target(surfaces[j], light, r.uv);
            denominator = denominator + reservoirs[j].confidence * t;
            if j == i {
                own = t;
            }
            if j == 0u {
                at_p = t;
            }
        }
        let m = raa_restir_ratio(r.confidence * own, denominator);
        let w = raa_restir_spatial_finite(m * at_p * r.weight);
        w_sum = w_sum + w;
        let xi = raa_restir_random(raa_restir_spatial_seed(pixel, frame, i));
        if raa_restir_wrs_accept(w_sum, w, xi) {
            merged = r;
            kept_target = at_p;
        }
    }
    // Nothing kept (w_sum <= 0) leaves the empty slot and a zero target, so W = 0.
    merged.weight = raa_restir_contribution_weight(w_sum, kept_target);
    merged.confidence = m_sum;
    merged.normal_oct = raa_restir_oct_encode(surface_p.normal);
    raa_restir_history[index] = raa_restir_pack(merged);

    // The resolve. U~ covers the lights the candidates are drawn from; an empty merged
    // reservoir traces no ray and writes (0, U~, 0, U~).
    var u_tilde = 0.0;
    let limit = min(raa_area_table.params.x, 8u);
    for (var slot = 0u; slot < limit; slot = slot + 1u) {
        let light = raa_restir_slot_light(slot);
        if light != 0xFFFFFFFFu {
            u_tilde = u_tilde + raa_restir_ltc_luminance(surface_p, raa_local_lights[light]);
        }
    }
    var s = 0.0;
    let light_sel = raa_restir_slot_light(merged.slot);
    if merged.slot != 0xFFu && merged.weight > 0.0 && light_sel != 0xFFFFFFFFu {
        let point = raa_restir_light_point(raa_local_lights[light_sel], merged.uv);
        let v = raa_area_ray_visibility(surface_p.position, surface_p.normal, point);
        s = kept_target * merged.weight * v;
    }
    textureStore(raa_area_raw_out, vec2<i32>(pixel), 0, vec4<f32>(s, u_tilde, s, u_tilde));
}
