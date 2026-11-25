// Shared by ordinary forward draws and resolved virtual geometry.
// material_main followed by the instance tint (S10). mat_instance_tint and
// mat_instance_tint_set are set first so material graph code can read them (S23).
fn raa_pass_material(in: MaterialInput, instance: u32) -> MaterialOutput {
    let x = raa_extras[instance];
    raa_set_instance_tint(x);
    return raa_apply_tint(material_main(in), x);
}

// Flat model or UNLIT flag: the fragment is base + emissive.
fn raa_pass_unlit(record: SurfaceMaterialGpu) -> bool {
    return record.header.x == RAA_MODEL_FLAT || (record.header.z & RAA_FLAG_UNLIT) != 0u;
}

// Normal-incidence reflectance: dielectric from the ior (0.04 at ior 1.5), base colour for
// metals.
fn raa_pass_f0(s: MaterialOutput) -> vec3<f32> {
    let ior = clamp(s.ior, 1.0, 3.0);
    let r = (ior - 1.0) / (ior + 1.0);
    return mix(vec3<f32>(r * r), s.base_color, clamp(s.metallic, 0.0, 1.0));
}

fn raa_pass_shade_ctx(s: MaterialOutput, in: MaterialInput, ao: f32, shadow: f32) -> ShadeCtx {
    var c: ShadeCtx;
    c.material_flags = in.record.header.z;
    c.position = in.position;
    c.view_dir = in.view_dir;
    c.frag = in.frag_coord;
    c.n_dot_v = max(dot(s.normal, in.view_dir), 1e-4);
    c.ao = min(s.ao, ao);
    c.shadow = shadow;
    c.f0 = raa_pass_f0(s);
    return c;
}

fn raa_opaque_lighting(in: MaterialInput, s: MaterialOutput) -> vec4<f32> {
    let shadow = raa_slot_shadow(in.position, in.normal, in.frag_coord);
    var ao = 1.0;
    let capture = raa_feature_enabled(RAA_PLANAR_CAPTURE_FEATURE);
    if !capture {
        ao = raa_slot_ao(in.frag_coord);
    }
    var c = raa_pass_shade_ctx(s, in, ao, shadow);
    let unlit = raa_pass_unlit(in.record);
    var spherical_debug = vec4<f32>(0.0);
    if !capture && !unlit && raa_feature_enabled(RAA_DDGI_GI_FEATURE)
        && !raa_feature_enabled(RAA_DIRECT_ONLY_FEATURE) {
        let ddgi = raa_ddgi_evaluate(c.position, s.normal, c.view_dir, c.frag.xy,
            raa_frame.camera_pos.xyz);
        spherical_debug = ddgi.debug;
        c.spherical_diffuse = ddgi.diffuse;
        if raa_ddgi_screen_ao() {
            // Probes 1.5 m apart carry no small-scale occlusion: the DDGI term takes the screen
            // AO like the ambient term it replaces. c.ao = min(material, screen) and the blend
            // multiplies by the material AO, so scale by their ratio.
            c.spherical_diffuse = vec4<f32>(ddgi.diffuse.rgb * (c.ao / max(s.ao, 1e-4)),
                ddgi.diffuse.w);
        }
    } else if !capture && !unlit && raa_feature_enabled(RAA_NRC_GI_FEATURE)
        && !raa_feature_enabled(RAA_DIRECT_ONLY_FEATURE) {
        // Neural radiance cache: indirect irradiance / pi resolved per pixel in the GI screen
        // slot, alpha = view depth (0 where the cache has no surface).
        let nrc_size = vec2<i32>(textureDimensions(raa_gi_screen));
        let nrc = textureLoad(raa_gi_screen,
            clamp(vec2<i32>(in.frag_coord.xy), vec2<i32>(0), nrc_size - vec2<i32>(1)), 0);
        c.spherical_diffuse = vec4<f32>(nrc.rgb, select(0.0, 1.0, nrc.a > 0.0));
    } else if !capture && !unlit && raa_feature_enabled(RAA_SPHERICAL_GI_FEATURE)
        && !raa_feature_enabled(RAA_DIRECT_ONLY_FEATURE) {
        // Screen-resolved by gi.spherical (half resolution); see raa_spherical_screen.
        let spherical = raa_spherical_screen(c.position, c.frag.xy, raa_frame.camera_pos.xyz);
        spherical_debug = spherical.debug;
        c.spherical_diffuse = spherical.diffuse;
    }
    let direct = raa_slot_direct(s, c, -raa_frame.sun_dir.xyz,
            raa_directional_radiance(raa_frame.sun_color.rgb) * shadow);
    let ambient = raa_slot_ambient(s, c);
    var indirect = vec3<f32>(0.0);
    if !capture {
        indirect = raa_slot_indirect(s, c);
        if c.spherical_diffuse.w > 0.0 {
            indirect *= 1.0 - c.spherical_diffuse.w;
        }
    }
    let local = raa_local_lighting(s, c);
    if spherical_debug.a > 0.0 { return spherical_debug; }
    if raa_feature_enabled(RAA_DIRECT_ONLY_FEATURE) {
        return vec4<f32>(select(s.emissive + direct + local, s.base_color + s.emissive, unlit), 1.0);
    }
    if raa_feature_enabled(RAA_INDIRECT_ONLY_FEATURE) {
        return vec4<f32>(select(ambient + indirect, vec3<f32>(0.0), unlit), 1.0);
    }
    let lit = s.emissive + direct + ambient + indirect + local;
    let color = select(lit, s.base_color + s.emissive, unlit);
    return vec4<f32>(color, 1.0);
}
