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


// ---- Diffuse transmission (material flag RAA_FLAG_DIFFUSE_TRANSMISSION) ----
// A cheap thin-translucency model for paper, lantern glass, leaves and grass, drawn OPAQUE (not
// the Transparent class): weight t (record.transmission.w) of the diffuse lobe leaves through the
// back. The front diffuse keeps base x (1 - t); the back adds Lambert with albedo
// base x colour x t gathered around the inverted normal -N: the sun (its shadow re-evaluated with
// the receiver pushed along -N), every local light of the tile (n.l, falloff, cone, LocalVisibility
// slot; area lights by their diffuse form factor), and the probe (DDGI) or ambient irradiance read
// with -N. The colour (record.emissive.w) is r + 256 g + 65536 b of 8-bit channels, an exact
// float. Only forward variants whose key carries the flag compile this (RAA_VARIANT_FLAGS): every
// other draw keeps the uber-shader unchanged. As in KHR_materials_diffuse_transmission, metals
// transmit nothing (t x (1 - metallic)) and the parameter is per material, not texturable.
const RAA_DT_PI: f32 = 3.14159265;

fn raa_dt_weight(record: SurfaceMaterialGpu, s: MaterialOutput) -> f32 {
    if (RAA_VARIANT_FLAGS & RAA_FLAG_DIFFUSE_TRANSMISSION) == 0u { return 0.0; }
    if (record.header.z & RAA_FLAG_DIFFUSE_TRANSMISSION) == 0u { return 0.0; }
    return clamp(record.transmission.w, 0.0, 1.0) * (1.0 - clamp(s.metallic, 0.0, 1.0));
}

fn raa_dt_color(record: SurfaceMaterialGpu) -> vec3<f32> {
    let w = u32(clamp(record.emissive.w, 0.0, 16777215.0));
    return vec3<f32>(f32(w & 255u), f32((w >> 8u) & 255u), f32((w >> 16u) & 255u)) / 255.0;
}

// Irradiance / pi from local light `index` arriving around the normal `n` at `p` (the caller
// multiplies by the albedo): point/spot radiance x falloff x cone x n.l / pi x LocalVisibility,
// directional radiance x n.l / pi, area radiance x diffuse form factor x window x GI fade (the
// hero and stochastic shadow tiers of the front term are skipped).
fn raa_dt_local_contribution(index: u32, budget: f32, p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let light = raa_local_lights[index];
    if all(light.color_inv_range_sq.rgb == vec3<f32>(0.0)) { return vec3<f32>(0.0); }
    var direction = -light.direction_cos_outer.xyz;
    var distance = 0.0;
    var falloff = 1.0;
    if light.kind == 0u || light.kind == 1u {
        let to_light = light.position_range.xyz - p;
        distance = length(to_light);
        if distance == 0.0 || distance >= light.position_range.w { return vec3<f32>(0.0); }
        direction = to_light / distance;
        falloff = raa_local_falloff(distance, light.position_range.w);
        // The forward tier and the tile budget of the front term (raa_local_contribution): the
        // rest of the light reaches the back through the probes read with -N.
        if light.width > 0.0 {
            falloff *= 1.0 - smoothstep(0.75 * light.width, light.width, distance);
        }
        falloff *= budget;
        if light.kind == 1u {
            falloff *= raa_local_spot(dot(light.direction_cos_outer.xyz, -direction),
                light.direction_cos_outer.w, light.cos_inner);
        }
    } else if light.kind == 3u {
        if (RAA_VARIANT_FLAGS & RAA_VARIANT_NO_AREA_LIGHTS) != 0u { return vec3<f32>(0.0); }
        if dot(p - light.position_range.xyz, light.direction_cos_outer.xyz) <= 0.0 {
            return vec3<f32>(0.0);
        }
        let d = length(raa_area_representative(light, p) - p);
        if d == 0.0 || d >= light.position_range.w { return vec3<f32>(0.0); }
        let w_direct = clamp(light.color_inv_range_sq.w, 0.0, 1.0);
        let fade = select(1.0, w_direct, raa_feature_enabled(RAA_GI_ACTIVE_FEATURE));
        return raa_area_radiance(light)
            * (raa_area_form_factor(light, p, n) * raa_area_window(d, light.position_range.w) * fade);
    } else if light.kind != 2u {
        return vec3<f32>(0.0);
    }
    let n_dot_l = dot(n, direction);
    if n_dot_l <= 0.0 || falloff <= 0.0 { return vec3<f32>(0.0); }
    var visibility = 1.0;
    if !raa_feature_enabled(RAA_PLANAR_CAPTURE_FEATURE) {
        // The cube shadow around -N: the lantern's own light inside its near radius stays lit.
        visibility = raa_local_shadow(light, p, n);
        if visibility <= 0.0 { return vec3<f32>(0.0); }
        visibility *= clamp(raa_slot_local_visibility(index, p, n, direction, distance), 0.0, 1.0);
    }
    return light.color_inv_range_sq.rgb * (falloff * visibility * n_dot_l / RAA_DT_PI);
}

// The tile's local lights (as raa_local_lighting) around `n`.
fn raa_dt_local(p: vec3<f32>, n: vec3<f32>, frag: vec4<f32>) -> vec3<f32> {
    var result = vec3<f32>(0.0);
    if raa_frame.light_info.x == 0u { return result; }
    if raa_feature_enabled(RAA_PLANAR_CAPTURE_FEATURE) {
        for (var index = 0u; index < min(raa_frame.light_info.x, MAX_LOCAL_LIGHTS); index += 1u) {
            result += raa_dt_local_contribution(index, 1.0, p, n);
        }
        return result;
    }
    let tile = vec2<i32>(floor(frag.xy / f32(TILE_SIZE)));
    let count = min(textureLoad(raa_light_tiles, tile, 0, 0).x, TILE_MAX_LIGHTS);
    for (var word = 1u; word <= count; word += 1u) {
        // Tile word: index in bits 0..15, 255 x (1 - forward weight) in bits 16..23 (light_tiles.wgsl).
        let entry = textureLoad(raa_light_tiles, tile, i32(word), 0).x;
        let index = entry & 0xffffu;
        let budget = 1.0 - f32((entry >> 16u) & 0xffu) / 255.0;
        if index < min(raa_frame.light_info.x, MAX_LOCAL_LIGHTS) {
            result += raa_dt_local_contribution(index, budget, p, n);
        }
    }
    return result;
}

// Environment irradiance / pi around `n` from the S10 SH (same basis as raa-ibl).
fn raa_dt_sh_irradiance(n: vec3<f32>) -> vec3<f32> {
    let sh = raa_env_sh.sh;
    var e = sh[0].rgb * 0.282095;
    e += sh[1].rgb * (0.488603 * n.y);
    e += sh[2].rgb * (0.488603 * n.z);
    e += sh[3].rgb * (0.488603 * n.x);
    e += sh[4].rgb * (1.092548 * n.x * n.y);
    e += sh[5].rgb * (1.092548 * n.y * n.z);
    e += sh[6].rgb * (0.315392 * (3.0 * n.z * n.z - 1.0));
    e += sh[7].rgb * (1.092548 * n.x * n.z);
    e += sh[8].rgb * (0.546274 * (n.x * n.x - n.y * n.y));
    return max(e, vec3<f32>(0.0));
}

// Direct irradiance / pi at the back: the sun (shadow slot evaluated for -N, mixed with the
// spherical sun visibility like the front) plus the local lights.
fn raa_dt_direct(in: MaterialInput, n: vec3<f32>, sun_far: f32) -> vec3<f32> {
    var e = raa_dt_local(in.position, n, in.frag_coord);
    let l = -raa_frame.sun_dir.xyz;
    let n_dot_l = dot(n, l);
    if n_dot_l > 0.0 {
        var shadow = 1.0;
        if sun_far < 1.0 {
            shadow = raa_slot_shadow(in.position, -in.normal, in.frag_coord);
        }
        shadow = mix(shadow, raa_spherical_sun_visibility, sun_far);
        e += raa_directional_radiance(raa_frame.sun_color.rgb) * (shadow * n_dot_l / RAA_DT_PI);
    }
    return e;
}

// Indirect irradiance / pi at the back: DDGI probes read with -N (offset away from the camera)
// over the SH or flat ambient, occluded by the shading AO.
fn raa_dt_indirect(in: MaterialInput, n: vec3<f32>, ao: f32) -> vec3<f32> {
    var e = raa_frame.ambient.rgb;
    if raa_env_sh.info.w >= 0.5 {
        e = raa_dt_sh_irradiance(n) * raa_env_sh.info.y;
    }
    if !raa_feature_enabled(RAA_PLANAR_CAPTURE_FEATURE) && raa_feature_enabled(RAA_DDGI_GI_FEATURE) {
        let probes = raa_ddgi_sample(in.position, n, -in.view_dir);
        e = mix(e, probes.rgb, clamp(probes.w, 0.0, 1.0));
    }
    return e * ao;
}

fn raa_opaque_lighting(in: MaterialInput, shaded: MaterialOutput) -> vec4<f32> {
    // Diffuse transmission keeps 1 - t of the front diffuse (t = 0 outside its variant).
    let dt = raa_dt_weight(in.record, shaded);
    var s = shaded;
    s.base_color = shaded.base_color * (1.0 - dt);
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
    var shadow = 1.0;
    if sun_far < 1.0 {
        shadow = raa_slot_shadow(in.position, in.normal, in.frag_coord);
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
    var local = vec3<f32>(0.0);
    if !sphere_direct {
        local = raa_local_lighting(s, c);
    }
    // The transmitted back term (zero outside the diffuse transmission variant).
    var back_direct = vec3<f32>(0.0);
    var back_indirect = vec3<f32>(0.0);
    if dt > 0.0 && !unlit {
        let albedo = shaded.base_color * raa_dt_color(in.record) * dt;
        let back = -s.normal;
        back_direct = albedo * raa_dt_direct(in, back, sun_far);
        if !raa_feature_enabled(RAA_DIRECT_ONLY_FEATURE) {
            back_indirect = albedo * raa_dt_indirect(in, back, c.ao);
        }
    }
    if spherical_debug.a > 0.0 { return spherical_debug; }
    if raa_feature_enabled(RAA_DIRECT_ONLY_FEATURE) {
        if raa_feature_enabled(RAA_INDIRECT_ONLY_FEATURE) {
            // lightingOutput "shadow": the sun visibility alone (cascades x contact).
            return vec4<f32>(vec3<f32>(shadow), 1.0);
        }
        return vec4<f32>(select(s.emissive + direct + local + back_direct,
            shaded.base_color + s.emissive, unlit), 1.0);
    }
    if raa_feature_enabled(RAA_INDIRECT_ONLY_FEATURE) {
        return vec4<f32>(select(ambient + indirect + back_indirect, vec3<f32>(0.0), unlit), 1.0);
    }
    let lit = s.emissive + direct + ambient + indirect + local + back_direct + back_indirect;
    let color = select(lit, shaded.base_color + s.emissive, unlit);
    return vec4<f32>(color, 1.0);
}
