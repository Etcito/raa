// M7 ReSTIR DI for the area-light shadows (S12 order 43), the initial-candidates stage of
// contract 'area-restir-wgsl'. One invocation per viewport pixel draws
// raa_restir_params.candidates light samples by resampled importance sampling (Talbot et al.
// 2005, as streamed in Bitterli et al. 2020): the light among the slot-table area lights by its
// unshadowed LTC luminance U_s, the point by Urena et al. EGSR 2013 spherical-rectangle
// sampling, each weighted by the Lambert + GGX target over its source PDF with m_i =
// 1 / candidates. The kept sample then takes one visibility ray (visibility reuse, Bitterli et
// al. 2020 section 6): the depth pyramid, the GI volume, then the raa-sdf-trace volume, through
// `raa_area_ray_visibility` of trace.wgsl. The kernel writes the pixel's reservoir into
// raa_restir_current and its linear view depth into raa_area_depth_out.
//
// This is RAA's own code; Bevy Solari is not supplied and nothing is copied from it or from
// NVIDIA RTXDI. The composed module (crates/raa-gi/src/parts/area_shadow/restir.rs
// `restir_initial_module_wgsl`) puts the area-shadow trace module (prelude, cloud shadow, GI
// volume, depth helpers, probe_common.wgsl, screen_trace.wgsl, trace.wgsl),
// raa_taa::REPROJECT_WGSL and restir_common.wgsl ahead of this file. Group-2 use: 0, 1, 2, 4,
// 6, 8, 10, 11, 15, 16, 18 and 19; group-1 use: 6, 12, 17 and 18; group 0 is never read.

// The per-pixel seed of candidate k, lane j (0..3): pixel.x * 7919 + pixel.y * 104729 +
// frame * 31337 + (4k + j) * 977, with wrapping u32 arithmetic as in the CPU twins.
fn raa_restir_initial_seed(pixel: vec2<u32>, frame: u32, k: u32, j: u32) -> u32 {
    return pixel.x * 7919u + pixel.y * 104729u + frame * 31337u + (4u * k + j) * 977u;
}

@compute @workgroup_size(8, 8)
fn raa_gi_area_restir_initial(@builtin(global_invocation_id) id: vec3<u32>) {
    let viewport = vec2<u32>(raa_gi_frame.viewport.xy);
    if any(id.xy >= viewport) {
        return;
    }
    let pixel = id.xy;
    let index = raa_restir_index(pixel);
    let surface = raa_restir_surface(vec2<i32>(pixel));
    if !surface.valid {
        raa_restir_current[index] = raa_restir_pack(raa_restir_empty());
        textureStore(raa_area_depth_out, vec2<i32>(pixel), vec4<f32>(0.0));
        return;
    }
    textureStore(raa_area_depth_out, vec2<i32>(pixel), vec4<f32>(surface.view_depth));

    // U_s per slot and the packed light index of each slot (0 and 0xFFFFFFFF when the slot
    // holds no area light).
    let limit = min(raa_area_table.params.x, 8u);
    var luminance: array<f32, 8>;
    var lights: array<u32, 8>;
    for (var s = 0u; s < 8u; s = s + 1u) {
        luminance[s] = 0.0;
        lights[s] = 0xFFFFFFFFu;
    }
    var total = 0.0;
    for (var s = 0u; s < limit; s = s + 1u) {
        let packed = raa_restir_slot_light(s);
        lights[s] = packed;
        if packed != 0xFFFFFFFFu {
            let u = raa_restir_ltc_luminance(surface, raa_local_lights[packed]);
            if u > 0.0 && u <= RAA_RESTIR_F32_MAX {
                luminance[s] = u;
                total = total + u;
            }
        }
    }

    var reservoir = raa_restir_empty();
    reservoir.confidence = 1.0;
    reservoir.normal_oct = raa_restir_oct_encode(surface.normal);
    if !(total > 0.0 && total <= RAA_RESTIR_F32_MAX) {
        raa_restir_current[index] = raa_restir_pack(reservoir);
        return;
    }

    // Streaming RIS over the candidates: w = target / (p * candidates), p the product of the
    // slot probability U_s / total and the area PDF of the spherical-rectangle sample.
    let candidates = raa_restir_params.candidates;
    let frame = raa_restir_params.frame_index;
    var w_sum = 0.0;
    var kept_slot = 0xFFFFFFFFu;
    var kept_uv = vec2<f32>(0.0);
    var kept_target = 0.0;
    for (var k = 0u; k < candidates; k = k + 1u) {
        let pick = raa_restir_random(raa_restir_initial_seed(pixel, frame, k, 0u)) * total;
        var slot = 0xFFFFFFFFu;
        var cumulative = 0.0;
        for (var s = 0u; s < limit; s = s + 1u) {
            if luminance[s] > 0.0 {
                slot = s;
                cumulative = cumulative + luminance[s];
                if pick < cumulative {
                    break;
                }
            }
        }
        if slot == 0xFFFFFFFFu {
            continue;
        }
        let light = raa_local_lights[lights[slot]];
        let xi = vec2<f32>(
            raa_restir_random(raa_restir_initial_seed(pixel, frame, k, 1u)),
            raa_restir_random(raa_restir_initial_seed(pixel, frame, k, 2u)),
        );
        let rect = raa_restir_sample_rect(light, surface.position, xi);
        let to_point = raa_restir_light_point(light, rect.uv) - surface.position;
        let r = length(to_point);
        var cos_y = 0.0;
        if r > 0.0 && r <= RAA_RESTIR_F32_MAX {
            cos_y = dot(light.direction_cos_outer.xyz, -(to_point / r));
        }
        let p = (luminance[slot] / total) * raa_restir_area_pdf(rect.solid_angle, cos_y, r);
        let target_value = raa_restir_target(surface, light, rect.uv);
        var w = 0.0;
        if p > 0.0 {
            w = target_value / (p * f32(candidates));
        }
        if !(w >= 0.0 && w <= RAA_RESTIR_F32_MAX) {
            w = 0.0;
        }
        w_sum = w_sum + w;
        let xi_keep = raa_restir_random(raa_restir_initial_seed(pixel, frame, k, 3u));
        if raa_restir_wrs_accept(w_sum, w, xi_keep) {
            kept_slot = slot;
            kept_uv = rect.uv;
            kept_target = target_value;
        }
    }

    // Visibility reuse: one shadow ray towards the kept point; an occluded sample keeps its
    // slot and uv with W = 0.
    if kept_slot != 0xFFFFFFFFu {
        let light = raa_local_lights[lights[kept_slot]];
        let point = raa_restir_light_point(light, kept_uv);
        let visibility = raa_area_ray_visibility(surface.position, surface.normal, point);
        reservoir.slot = kept_slot;
        reservoir.uv = kept_uv;
        reservoir.weight = raa_restir_contribution_weight(w_sum, kept_target) * visibility;
    }
    raa_restir_current[index] = raa_restir_pack(reservoir);
}
