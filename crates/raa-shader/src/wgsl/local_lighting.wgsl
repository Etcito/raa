// S11 shared local lighting. Local lights cast no shadows by default.
// Point window/incidence adapted from solstice (MIT), Copyright (c) 2026 SamG-Coder.
// solstice@7e465b5, crates/solstice-render/shaders/lighting/deferred.wgsl:72-83.
fn raa_local_falloff(distance: f32, range: f32) -> f32 {
    if distance >= range { return 0.0; }
    let ratio = distance / range;
    let ratio2 = ratio * ratio;
    let window = clamp(1.0 - ratio2 * ratio2, 0.0, 1.0);
    return window * window / max(distance * distance, 0.04);
}

fn raa_local_spot(cosine: f32, outer: f32, inner: f32) -> f32 {
    if inner <= outer { return select(0.0, 1.0, cosine >= outer); }
    return smoothstep(outer, inner, cosine);
}

// Karis representative point over the oriented rectangle, evaluated from the shading
// position. The rectangle emits along +travel only. Callers keep the extents positive;
// a mathematical [0, 0] degenerates to the light position. Kind 3 uses the point for the
// visibility direction and distance and for the range window only.
fn raa_area_representative(light: LightGpu, position: vec3<f32>) -> vec3<f32> {
    let basis = raa_area_basis(light.direction_cos_outer.xyz);
    let offset = position - light.position_range.xyz;
    let half_extent = vec2<f32>(max(light.width, 0.0), max(light.height, 0.0)) * 0.5;
    let along_t = clamp(dot(offset, basis[0]), -half_extent.x, half_extent.x);
    let along_b = clamp(dot(offset, basis[1]), -half_extent.y, half_extent.y);
    return light.position_range.xyz + basis[0] * along_t + basis[1] * along_b;
}

// Feature id 39 (area-light-shading-api, set by setAreaLightShading 'formFactor'): area
// lights evaluate the diffuse form factor only, without the LTC tables.
const RAA_AREA_FORM_FACTOR_FEATURE: u32 = 39u;

// LTC area light (Heitz, Dupuy, Hill, Neubelt, SIGGRAPH 2016): Lambert x (1 - metallic) over
// the whole rectangle plus GGX specular through the inverse LTC matrix, both horizon
// clipped by the clipped-sphere form factor. Table use and Hill's Fresnel follow three.js
// r180 RE_Direct_RectArea_Physical and LTC_Evaluate (MIT, Copyright (c) 2010-2025 three.js
// authors). three.js negates T2 because its corners run the other way; here the corners are
// counter-clockwise from the emitting side and t2 = cross(n, t1) is right-handed.
// Only textureSampleLevel: this runs inside the nonuniform light loop.
fn raa_area_ltc(light: LightGpu, s: MaterialOutput, c: ShadeCtx) -> vec3<f32> {
    let n = s.normal;
    let v = c.view_dir;
    let n_dot_v = clamp(dot(n, v), 0.0, 1.0);
    let roughness = max(s.roughness, 0.0525);
    var t1 = v - n * dot(v, n);
    if dot(t1, t1) < 1e-8 { t1 = raa_area_basis(n)[0]; }
    t1 = normalize(t1);
    let t2 = cross(n, t1);
    let world_to_local = transpose(mat3x3<f32>(t1, t2, n));
    let a = textureSampleLevel(raa_brdf_lut, raa_linear_clamp,
        raa_ltc_uv(0u, roughness, n_dot_v), 0.0);
    let b = textureSampleLevel(raa_brdf_lut, raa_linear_clamp,
        raa_ltc_uv(1u, roughness, n_dot_v), 0.0);
    let m_inv = mat3x3<f32>(vec3<f32>(a.x, 0.0, a.y), vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(a.z, 0.0, a.w));
    let corners = raa_area_corners(light, c.position);
    let fs = raa_area_vector_form_factor(corners, m_inv * world_to_local);
    let specular = raa_area_clipped(length(fs), fs.z);
    let diffuse = raa_area_form_factor(light, c.position, n);
    let f0 = mix(vec3<f32>(0.04), s.base_color, s.metallic);
    let fresnel = f0 * b.x + (vec3<f32>(1.0) - f0) * b.y;
    return raa_area_radiance(light)
        * (s.base_color * (1.0 - s.metallic) * diffuse + fresnel * specular);
}

// Diffuse-only area light (formFactor mode): Lambert x (1 - metallic) times the horizon
// clipped form factor of the rectangle. No table read and no specular.
fn raa_area_diffuse(light: LightGpu, s: MaterialOutput, c: ShadeCtx) -> vec3<f32> {
    return raa_area_radiance(light) * s.base_color * (1.0 - s.metallic)
        * raa_area_form_factor(light, c.position, s.normal);
}

// The caller passes the packed uniform index (not a stable CPU handle) to visibility.
fn raa_local_contribution(index: u32, s: MaterialOutput, c: ShadeCtx) -> vec3<f32> {
    let light = raa_local_lights[index];
    var direction = -light.direction_cos_outer.xyz;
    var distance = 0.0;
    var falloff = 1.0;
    if light.kind == 0u || light.kind == 1u {
        let to_light = light.position_range.xyz - c.position;
        distance = length(to_light);
        // No incident direction exists at coincidence; return finite black before division.
        if distance == 0.0 || distance >= light.position_range.w { return vec3<f32>(0.0); }
        direction = to_light / distance;
        falloff = raa_local_falloff(distance, light.position_range.w);
        if light.kind == 1u {
            falloff *= raa_local_spot(dot(light.direction_cos_outer.xyz, -direction),
                light.direction_cos_outer.w, light.cos_inner);
        }
    } else if light.kind == 3u {
        // S11 area, shaded U x W x V: U integrates the whole rectangle (LTC, or the diffuse
        // form factor under feature 39); the S11 representative point only gives the
        // visibility direction and distance and the range window W. No n.l early-out and
        // no raa_slot_direct: horizon clipping handles surfaces turned away, and area
        // lights use this PbrLite model whatever provider wins the Direct slot.
        let travel = light.direction_cos_outer.xyz;
        // One-sided emission: behind the rectangle contributes nothing.
        if dot(c.position - light.position_range.xyz, travel) <= 0.0 { return vec3<f32>(0.0); }
        let representative = raa_area_representative(light, c.position);
        let to_light = representative - c.position;
        distance = length(to_light);
        if distance == 0.0 || distance >= light.position_range.w { return vec3<f32>(0.0); }
        direction = to_light / distance;
        let window = raa_area_window(distance, light.position_range.w);
        let area_visibility = clamp(raa_slot_local_visibility(index, c.position, s.normal,
            direction, distance), 0.0, 1.0);
        var shading = vec3<f32>(0.0);
        if raa_feature_enabled(RAA_AREA_FORM_FACTOR_FEATURE) {
            shading = raa_area_diffuse(light, s, c);
        } else {
            shading = raa_area_ltc(light, s, c);
        }
        return shading * (window * area_visibility);
    } else if light.kind != 2u {
        // Directional ignores position and range; unknown kinds contribute nothing.
        return vec3<f32>(0.0);
    }
    // Point, spot and directional only.
    if dot(s.normal, direction) <= 0.0 || falloff <= 0.0 { return vec3<f32>(0.0); }
    let visibility = clamp(raa_slot_local_visibility(index, c.position, s.normal,
        direction, distance), 0.0, 1.0);
    let radiance = light.color_inv_range_sq.rgb * (falloff * visibility);
    if !any(radiance > vec3<f32>(0.0)) { return vec3<f32>(0.0); }
    return raa_slot_direct(s, c, direction, radiance);
}

// Handoff: Composer attaches LOCAL_LIGHTING_WGSL after the selected slot providers.
// Forward/transparent/fur callers add raa_local_lighting(s, c) to their lit branch;
// the separate sun call and Flat/UNLIT selection remain with each pass.
// Point, spot and directional lights call Direct once each; area lights return
// raa_area_ltc (or raa_area_diffuse under feature 39) x window x visibility instead.
// c.frag.xy is framebuffer position; light_info.x is the packed active count.
// Binding 13 stores word w at layer w of tile (x, y); layer 0 is the count.
// Providers reached here must use explicit LOD/gradients or textureLoad, no derivatives.
fn raa_local_lighting(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> {
    var result = vec3<f32>(0.0);
    if raa_frame.light_info.x == 0u { return result; }
    let tile = vec2<i32>(floor(c.frag.xy / f32(TILE_SIZE)));
    let count = min(textureLoad(raa_light_tiles, tile, 0, 0).x, TILE_MAX_LIGHTS);
    for (var word = 1u; word <= count; word += 1u) {
        let index = textureLoad(raa_light_tiles, tile, i32(word), 0).x;
        if index < min(raa_frame.light_info.x, MAX_LOCAL_LIGHTS) {
            result += raa_local_contribution(index, s, c);
        }
    }
    return result;
}
