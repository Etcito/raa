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
    let capture = raa_feature_enabled(RAA_PLANAR_CAPTURE_FEATURE);
    let unlit = raa_pass_unlit(in.record);
    // Screen-resolved by gi.spherical (half resolution); see raa_spherical_screen.
    var spherical = RaaSphericalResult();
    var sphere_direct = false;
    var sun_far = 0.0;
    // DDGI and the radiance cache take precedence over the spherical grid (below).
    if !capture && !unlit && raa_feature_enabled(RAA_SPHERICAL_GI_FEATURE)
        && !raa_feature_enabled(RAA_DDGI_GI_FEATURE) && !raa_feature_enabled(RAA_NRC_GI_FEATURE)
        && !raa_feature_enabled(RAA_DIRECT_ONLY_FEATURE) {
        spherical = raa_spherical_screen(in.position, in.frag_coord.xy, raa_frame.camera_pos.xyz);
        // Sphere-only direct (spherical GI phase 2, `sphereDirect`): local lights come only
        // through the probes (in spherical.diffuse) and, beyond directSunRange, the probes'
        // sun visibility replaces the cascaded shadow, so neither is evaluated here.
        let range = raa_spherical_direct_range();
        sphere_direct = range >= 0.0;
        if sphere_direct {
            sun_far = smoothstep(range, 1.25 * range + 0.01,
                length(in.position - raa_frame.camera_pos.xyz));
        }
    }
    // Under the water body (FrameGpu.reserved) the sun comes along its refracted direction: its
    // visibility is tested where that path enters the water, so the shadow of a wall under water
    // is shorter and the strip next to the wall is lit (raa_water_sun).
    let water = raa_water_unpack(raa_frame.reserved);
    let water_sun = raa_water_sun(water, raa_frame.sun_dir.xyz, in.position);
    var shadow = 1.0;
    if sun_far < 1.0 && water_sun.under {
        // Soft rippled penumbra: the visibility averaged over RAA_WATER_SHADOW_TAPS entry points
        // on a disk around the solved one (raa_water_penumbra), rotated per pixel and per frame
        // (the camera jitter) for TAA, so the caustics fade into the shadow, not on a line.
        let radius = raa_water_penumbra(water, in.position);
        let seed = fract(52.9829189 * fract(dot(in.frag_coord.xy, vec2<f32>(0.06711056, 0.00583715)))
            + 113.7 * dot(raa_frame.camera_jitter.xy, vec2<f32>(1.0, 1.618)));
        shadow = 0.0;
        for (var i = 0u; i < RAA_WATER_SHADOW_TAPS; i++) {
            let o = raa_water_penumbra_tap(i, seed) * radius;
            shadow += raa_slot_shadow(water_sun.entry + vec3<f32>(o.x, 0.0, o.y),
                vec3<f32>(0.0, 1.0, 0.0), in.frag_coord);
        }
        shadow /= f32(RAA_WATER_SHADOW_TAPS);
    } else if sun_far < 1.0 {
        shadow = raa_slot_shadow(water_sun.entry, in.normal, in.frag_coord);
    }
    shadow = mix(shadow, raa_spherical_sun_visibility, sun_far);
    var ao = 1.0;
    if !capture {
        ao = raa_slot_ao(in.frag_coord);
    }
    var c = raa_pass_shade_ctx(s, in, ao, shadow);
    var spherical_debug = spherical.debug;
    c.spherical_diffuse = spherical.diffuse;
    if !capture && raa_feature_enabled(RAA_SPHERICAL_GI_FEATURE) && raa_spherical_screen_ao() {
        // Same as DDGI: the probe term takes the screen AO like the ambient it replaces
        // (c.ao = min(material, screen); the blend multiplies by the material AO).
        c.spherical_diffuse = vec4<f32>(spherical.diffuse.rgb * (c.ao / max(s.ao, 1e-4)),
            spherical.diffuse.w);
    }
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
    }
    var sun_radiance = raa_directional_radiance(raa_frame.sun_color.rgb) * shadow;
    if water_sun.under && shadow > 0.0 {
        // Fresnel transmittance and the caustics of the ripple (raa_water_sun.k), full on an
        // upward receiver (floor), none on a vertical one (wall).
        sun_radiance *= water_sun.transmit * mix(1.0, water_sun.k, saturate(in.normal.y));
    }
    let direct = raa_slot_direct(s, c, water_sun.to_sun, sun_radiance);
    var ambient = raa_slot_ambient(s, c);
    var indirect = vec3<f32>(0.0);
    if !capture {
        indirect = raa_slot_indirect(s, c);
        if c.spherical_diffuse.w > 0.0 {
            indirect *= 1.0 - c.spherical_diffuse.w;
        }
    }
    if water_sun.under {
        // The sky's soft caustics (raa_water_sky_caustic) on the ambient and probe light, full on
        // an upward receiver: no flat tone on the shadowed floor under the water.
        let sky = mix(1.0, water_sun.sky, saturate(in.normal.y));
        ambient *= sky;
        indirect *= sky;
    }
    var local = vec3<f32>(0.0);
    if !sphere_direct {
        local = raa_local_lighting(s, c);
    }
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
