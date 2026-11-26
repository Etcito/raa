// Forward lightingOutput "layers" / "layers-signals" / "layers-interlaced"
// (VARIANT_FLAG_FORWARD_LAYERS): the opaque
// lighting of raa_opaque_lighting split into demodulated layers and per-pixel signals so that,
// for an opaque lit fragment,
//
//     combined = d0 * s + (kd * e + s0) * a + em + r
//
// - d0: the sun direct term with shadow = 1 (same raa_slot_direct call, linear in radiance).
// - s:  the sun visibility combined used (raa_slot_shadow, mixed with the spherical probes' sun
//       visibility beyond directSunRange).
// - a:  the AO combined used, c.ao = min(material ao, raa_slot_ao).
// - e:  w x the probe GI rgb fed to raa_spherical_blend_diffuse (DDGI, else NRC, else
//       spherical), before the screen-AO ratio; w its blend weight.
// - kd: the factor of (probe rgb x w x a) in raa_slot_ambient, measured by linearity:
//       A(0, (k, 1)) - A(0, (0, 1)) with k = 1 / max(material ao, 1e-4) when the probe GI takes
//       the screen AO (DDGI/spherical screenAo), else k = 1. A(ao, sph) is raa_slot_ambient
//       with c.ao = ao and c.spherical_diffuse = sph.
// - s0: A(1, sph0) - A(0, sph0): the AO-dependent ambient (environment specular with specular
//       occlusion, SH diffuse with multi-bounce AO, fallback ambient) at ao = 1.
// - em: emissive + local lights + A(0, sph0) (AO-free ambient, e.g. sheen) + the screen
//       indirect (raa_slot_indirect) when no probe GI is enabled.
// - r:  the residual, only in "layers-signals": combined - the recomposition. Zero up to
//       rounding except (1) where A is not linear in c.ao (multi-bounce AO, specular
//       occlusion), (2) probe weight w < 1 (the layers assume w = 1 whenever a probe GI feature
//       bit is set: the legacy diffuse x (1 - w) and the screen indirect x (1 - w) land in r),
//       (3) a probe GI without screen AO (NRC, screenAo off: combined does not multiply e by a).
// sph0 = (0, 0, 0, 1) when a probe GI feature bit is set (w = 1), else (0, 0, 0, 0).
// Unlit (flat model or UNLIT): em = base + emissive, every other layer 0, s = a = 1, e = 0.
//
// The layers never read the sun shadow, the screen AO or the probe GI ("layers" skips them);
// "layers-signals" evaluates the signals once more (raa_opaque_signals) and the full
// raa_opaque_lighting for r; "layers-interlaced" evaluates them only at the phase pixel of an
// R frame (pass_forward_layers_interlaced.wgsl) and has no r.

struct RaaOpaqueLayers {
    d0: vec3<f32>,
    kd: vec3<f32>,
    s0: vec3<f32>,
    em: vec3<f32>,
}

struct RaaOpaqueSignals {
    shadow: f32,
    ao: f32,
    e: vec3<f32>,
    w: f32,
}

// The probe GI kind combined uses: 1 DDGI, 2 NRC, 3 spherical, 0 none (raa_opaque_lighting's
// precedence). Reads only frame feature bits.
fn raa_layers_probe_kind(capture: bool) -> u32 {
    if capture || raa_feature_enabled(RAA_DIRECT_ONLY_FEATURE) { return 0u; }
    if raa_feature_enabled(RAA_DDGI_GI_FEATURE) { return 1u; }
    if raa_feature_enabled(RAA_NRC_GI_FEATURE) { return 2u; }
    if raa_feature_enabled(RAA_SPHERICAL_GI_FEATURE) { return 3u; }
    return 0u;
}

// Whether combined scales the probe rgb by c.ao / max(material ao, 1e-4) (screenAo).
fn raa_layers_probe_screen_ao(kind: u32) -> bool {
    if kind == 1u { return raa_ddgi_screen_ao(); }
    if kind == 3u { return raa_spherical_screen_ao(); }
    return false;
}

// Spherical sphere-only direct: combined evaluates no local light (raa_opaque_lighting).
fn raa_layers_sphere_direct(kind: u32) -> bool {
    return kind == 3u && raa_spherical_direct_range() >= 0.0;
}

fn raa_opaque_layers(in: MaterialInput, s: MaterialOutput) -> RaaOpaqueLayers {
    let capture = raa_feature_enabled(RAA_PLANAR_CAPTURE_FEATURE);
    let unlit = raa_pass_unlit(in.record);
    let kind = raa_layers_probe_kind(capture);
    // Shadow 1 and screen AO 1: c.ao = min(material ao, 1) = material ao (overridden below).
    var c = raa_pass_shade_ctx(s, in, 1.0, 1.0);
    let d0 = raa_slot_direct(s, c, -raa_frame.sun_dir.xyz,
        raa_directional_radiance(raa_frame.sun_color.rgb));
    let w = select(0.0, 1.0, kind != 0u);
    c.ao = 0.0;
    c.spherical_diffuse = vec4<f32>(vec3<f32>(0.0), w);
    let ambient_free = raa_slot_ambient(s, c);
    c.ao = 1.0;
    let ambient_lit = raa_slot_ambient(s, c);
    var kd = vec3<f32>(0.0);
    if kind != 0u {
        let k = select(1.0, 1.0 / max(s.ao, 1e-4), raa_layers_probe_screen_ao(kind));
        c.ao = 0.0;
        c.spherical_diffuse = vec4<f32>(vec3<f32>(k), 1.0);
        kd = raa_slot_ambient(s, c) - ambient_free;
        c.ao = 1.0;
        c.spherical_diffuse = vec4<f32>(vec3<f32>(0.0), w);
    }
    var indirect = vec3<f32>(0.0);
    if !capture && kind == 0u {
        // No probe GI: combined adds the screen indirect unscaled (w = 0).
        indirect = raa_slot_indirect(s, c);
    }
    var local = vec3<f32>(0.0);
    if !raa_layers_sphere_direct(kind) {
        local = raa_local_lighting(s, c);
    }
    var out: RaaOpaqueLayers;
    let zero = vec3<f32>(0.0);
    out.d0 = select(d0, zero, unlit);
    out.kd = select(kd, zero, unlit);
    out.s0 = select(ambient_lit - ambient_free, zero, unlit);
    out.em = select(s.emissive + local + ambient_free + indirect, s.base_color + s.emissive, unlit);
    return out;
}

// The signals s, a, e and w exactly as raa_opaque_lighting computes them (same calls).
fn raa_opaque_signals(in: MaterialInput, s: MaterialOutput) -> RaaOpaqueSignals {
    let capture = raa_feature_enabled(RAA_PLANAR_CAPTURE_FEATURE);
    let unlit = raa_pass_unlit(in.record);
    var spherical = RaaSphericalResult();
    var sun_far = 0.0;
    if !capture && !unlit && raa_feature_enabled(RAA_SPHERICAL_GI_FEATURE)
        && !raa_feature_enabled(RAA_DDGI_GI_FEATURE) && !raa_feature_enabled(RAA_NRC_GI_FEATURE)
        && !raa_feature_enabled(RAA_DIRECT_ONLY_FEATURE) {
        spherical = raa_spherical_screen(in.position, in.frag_coord.xy, raa_frame.camera_pos.xyz);
        let range = raa_spherical_direct_range();
        if range >= 0.0 {
            sun_far = smoothstep(range, 1.25 * range + 0.01,
                length(in.position - raa_frame.camera_pos.xyz));
        }
    }
    var shadow = 1.0;
    if sun_far < 1.0 {
        shadow = raa_slot_shadow(in.position, in.normal, in.frag_coord);
    }
    shadow = mix(shadow, raa_spherical_sun_visibility, sun_far);
    var ao = 1.0;
    if !capture {
        ao = raa_slot_ao(in.frag_coord);
    }
    var probe = spherical.diffuse;
    if !capture && !unlit && raa_feature_enabled(RAA_DDGI_GI_FEATURE)
        && !raa_feature_enabled(RAA_DIRECT_ONLY_FEATURE) {
        probe = raa_ddgi_evaluate(in.position, s.normal, in.view_dir, in.frag_coord.xy,
            raa_frame.camera_pos.xyz).diffuse;
    } else if !capture && !unlit && raa_feature_enabled(RAA_NRC_GI_FEATURE)
        && !raa_feature_enabled(RAA_DIRECT_ONLY_FEATURE) {
        let nrc_size = vec2<i32>(textureDimensions(raa_gi_screen));
        let nrc = textureLoad(raa_gi_screen,
            clamp(vec2<i32>(in.frag_coord.xy), vec2<i32>(0), nrc_size - vec2<i32>(1)), 0);
        probe = vec4<f32>(nrc.rgb, select(0.0, 1.0, nrc.a > 0.0));
    }
    let w = select(0.0, probe.w, probe.w > 0.0);
    var out: RaaOpaqueSignals;
    out.shadow = select(shadow, 1.0, unlit);
    out.ao = select(min(s.ao, ao), 1.0, unlit);
    out.e = select(probe.rgb * w, vec3<f32>(0.0), unlit);
    out.w = select(w, 0.0, unlit);
    return out;
}
