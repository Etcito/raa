// S11 shared local lighting. Point, spot and directional lights call Direct once each and
// read the LocalVisibility slot; area lights shade U x V and layer the visibility tiers in
// V: a hero PCSS shadow map weighted by the map-edge weight, the mid tier's stochastic
// shadow mask blended by w_stoch over the distance band, and the GI cone visibility of the
// LocalVisibility slot; the GI-only far fade of the direct term applies outside V. Point
// window and incidence adapted from solstice (MIT), Copyright (c) 2026 SamG-Coder.
// solstice@7e465b5, crates/solstice-render/shaders/lighting/deferred.wgsl:72-83.
fn raa_local_falloff(distance: f32, range: f32) -> f32 {
    if distance >= range { return 0.0; }
    let ratio = distance / range;
    let ratio2 = ratio * ratio;
    let window = clamp(1.0 - ratio2 * ratio2, 0.0, 1.0);
    return window * window / max(distance * distance, 0.04);
}

// Cached cube shadow of a point or spot light with `shadows` (light.height = slot + 1, see
// raa-sdf-trace mesh/local_shadow.rs, which bakes the faces by tracing the scene BVH from the
// light). `p` is the receiver, `n` the normal it is lit around (the caller's -N for diffuse
// transmission). The receiver is pushed 1.5 texels along `n` and compared with a half-texel
// tolerance, with 2x2 bilinear PCF inside the face. A receiver closer to the light than the bake's
// near radius (the light's own housing, e.g. a lantern's glass) always passes: every stored
// distance is at least that radius. A texel of 0 (not baked yet) and a slot outside the bound
// array read as unoccluded. Only textureLoad: this runs inside the nonuniform light loop.
fn raa_local_shadow_face(v: vec3<f32>) -> vec4<f32> {
    let a = abs(v);
    if a.x >= a.y && a.x >= a.z {
        if v.x > 0.0 { return vec4<f32>(0.0, -v.z / a.x, -v.y / a.x, a.x); }
        return vec4<f32>(1.0, v.z / a.x, -v.y / a.x, a.x);
    }
    if a.y >= a.z {
        if v.y > 0.0 { return vec4<f32>(2.0, v.x / a.y, v.z / a.y, a.y); }
        return vec4<f32>(3.0, v.x / a.y, -v.z / a.y, a.y);
    }
    if v.z > 0.0 { return vec4<f32>(4.0, v.x / a.z, -v.y / a.z, a.z); }
    return vec4<f32>(5.0, -v.x / a.z, -v.y / a.z, a.z);
}

// Own shadow map of a `shadowMode: 'map'` point/spot light (lane 60 = -layer of the sun shadow
// array): layers rendered by shadow.csm after the area heroes with the hero projection
// (raa-lit local_map_view_proj: perspective_rh_zo, near RAA_AREA_HERO_NEAR, far = range, basis
// raa_area_basis of the travel direction). A spot uses one layer whose half-angle tangent is
// the cone's widened by 10 % plus 0.02 (at most RAA_LOCAL_MAP_MAX_TAN); a point light uses six
// layers (+X, -X, +Y, -Y, +Z, -Z) of tangent RAA_LOCAL_MAP_CUBE_TAN, picked by the dominant
// axis. Four comparison taps 1.2 texels apart after a 1.5-texel normal offset; outside the
// frustum reads lit.
const RAA_LOCAL_MAP_CUBE_TAN: f32 = 1.05;
const RAA_LOCAL_MAP_MAX_TAN: f32 = 2.7474774;
const RAA_LOCAL_MAP_NEAR: f32 = 0.05;

fn raa_local_map_shadow(light: LightGpu, p: vec3<f32>, n: vec3<f32>) -> f32 {
    var layer = i32(-light.height + 0.5);
    if u32(layer) >= textureNumLayers(raa_shadow_map) { return 1.0; }
    let to_p = p - light.position_range.xyz;
    var axis = light.direction_cos_outer.xyz;
    var tan_half = RAA_LOCAL_MAP_CUBE_TAN;
    if light.kind == 0u {
        let a = abs(to_p);
        var face = 0;
        if a.x >= a.y && a.x >= a.z {
            face = select(1, 0, to_p.x > 0.0);
            axis = vec3<f32>(sign(to_p.x), 0.0, 0.0);
        } else if a.y >= a.z {
            face = select(3, 2, to_p.y > 0.0);
            axis = vec3<f32>(0.0, sign(to_p.y), 0.0);
        } else {
            face = select(5, 4, to_p.z > 0.0);
            axis = vec3<f32>(0.0, 0.0, sign(to_p.z));
        }
        layer += face;
    } else {
        let c = clamp(light.direction_cos_outer.w, 1e-4, 1.0);
        tan_half = min(sqrt(max(1.0 - c * c, 0.0)) / c * 1.1 + 0.02, RAA_LOCAL_MAP_MAX_TAN);
    }
    let m = raa_area_basis(axis);
    let far = max(light.position_range.w, 2.0 * RAA_LOCAL_MAP_NEAR);
    let res = 1.0 / raa_shadow.params.z;
    let texel = 2.0 * max(dot(to_p, m[2]), RAA_LOCAL_MAP_NEAR) * tan_half / res;
    let d = to_p + n * (1.5 * texel);
    let dz = dot(d, m[2]);
    if dz <= 1.01 * RAA_LOCAL_MAP_NEAR { return 1.0; }
    let ndc = vec2<f32>(-dot(d, m[0]), dot(d, m[1])) / (dz * tan_half);
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) { return 1.0; }
    let depth = far * (dz - RAA_LOCAL_MAP_NEAR) / ((far - RAA_LOCAL_MAP_NEAR) * dz);
    let o = 0.6 / res;
    var lit = 0.0;
    lit += textureSampleCompareLevel(raa_shadow_map, raa_shadow_cmp, uv + vec2<f32>(-o, -o), layer, depth);
    lit += textureSampleCompareLevel(raa_shadow_map, raa_shadow_cmp, uv + vec2<f32>(o, -o), layer, depth);
    lit += textureSampleCompareLevel(raa_shadow_map, raa_shadow_cmp, uv + vec2<f32>(-o, o), layer, depth);
    lit += textureSampleCompareLevel(raa_shadow_map, raa_shadow_cmp, uv + vec2<f32>(o, o), layer, depth);
    return 0.25 * lit;
}

fn raa_local_shadow(light: LightGpu, p: vec3<f32>, n: vec3<f32>) -> f32 {
    if light.kind > 1u { return 1.0; }
    if light.height < -0.5 { return raa_local_map_shadow(light, p, n); }
    if light.height < 0.5 { return 1.0; }
    let slot = u32(light.height + 0.5) - 1u;
    let layers = textureNumLayers(raa_local_shadow_map);
    if 6u * slot + 5u >= layers { return 1.0; }
    let res = f32(textureDimensions(raa_local_shadow_map).x);
    let d0 = length(p - light.position_range.xyz);
    // World size of one texel at the receiver's distance (a cube face spans 2 d).
    let texel = 2.0 * d0 / res;
    let v = p + n * (1.5 * texel) - light.position_range.xyz;
    let d = length(v) - 0.5 * texel;
    let face = raa_local_shadow_face(v);
    let layer = i32(6u * slot + u32(face.x));
    let f = (face.yz * 0.5 + 0.5) * res - 0.5;
    let base = floor(f);
    let t = f - base;
    let hi = i32(res) - 1;
    var lit = vec4<f32>(0.0);
    for (var k = 0; k < 4; k += 1) {
        let texel_xy = clamp(vec2<i32>(base) + vec2<i32>(k & 1, k >> 1), vec2<i32>(0), vec2<i32>(hi));
        let stored = textureLoad(raa_local_shadow_map, texel_xy, layer, 0).x;
        lit[k] = select(0.0, 1.0, stored == 0.0 || d <= stored);
    }
    return mix(mix(lit.x, lit.y, t.x), mix(lit.z, lit.w, t.x), t.y);
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

// VARIANT_FLAG_NO_AREA_LIGHTS (raa-shader lib.rs): the variant's frame has no area light, so
// the area branch is compiled out. RAA_VARIANT_FLAGS is emitted by the composer before the pass.
const RAA_VARIANT_NO_AREA_LIGHTS: u32 = 0x10000000u;

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

// Feature id 29 (set by raa-gi while GI is active): area lights apply the far-fade lane
// w_direct and read the GI cone visibility tier.
const RAA_GI_ACTIVE_FEATURE: u32 = 29u;

// Hero area-light projection: 140 degrees vertical FOV on a square map (the tangent of the
// half angle) and a 0.05 m near plane, matching contract area-shadow-settings.
const RAA_AREA_HERO_TAN_HALF: f32 = 2.7474774;
const RAA_AREA_HERO_NEAR: f32 = 0.05;

// Hero visibility from one layer of raa_shadow_map, as (visibility, map weight). The
// receiver is pushed along its normal by 1.5 texels of the light's own projection; the
// blocker search and the PCF radius widen with the blocker-receiver gap, so the penumbra is
// sharp at contact. The map weight fades the value out over the layer border. Only
// textureLoad and explicit-level samples: this runs inside the nonuniform light loop.
fn raa_area_hero_visibility(light: LightGpu, layer: i32, position: vec3<f32>,
    normal: vec3<f32>, frag: vec4<f32>) -> vec2<f32> {
    let m = raa_area_basis(light.direction_cos_outer.xyz);
    let far = max(light.position_range.w, 2.0 * RAA_AREA_HERO_NEAR);
    let res = 1.0 / raa_shadow.params.z;
    var d = position - light.position_range.xyz;
    let texel_r = 2.0 * dot(d, m[2]) * RAA_AREA_HERO_TAN_HALF / res;
    d = position + normal * (1.5 * texel_r) - light.position_range.xyz;
    let dz = dot(d, m[2]);
    let ndc = vec2<f32>(-dot(d, m[0]), dot(d, m[1])) / (dz * RAA_AREA_HERO_TAN_HALF);
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if dz <= 1.01 * RAA_AREA_HERO_NEAR || any(uv < vec2<f32>(0.0))
        || any(uv > vec2<f32>(1.0)) {
        return vec2<f32>(1.0, 0.0);
    }
    let w_map = smoothstep(0.0, 0.05, min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y)));
    let depth = far * (dz - RAA_AREA_HERO_NEAR) / ((far - RAA_AREA_HERO_NEAR) * dz);
    let size = 0.5 * (light.width + light.height);
    let search_texels = clamp(size * res / (2.0 * RAA_AREA_HERO_TAN_HALF * dz), 1.0, 32.0);
    let blockers = raa_shadow_blockers(layer, uv, depth, search_texels, res);
    if blockers.y < 0.5 { return vec2<f32>(1.0, w_map); }
    let blocker_depth = far * RAA_AREA_HERO_NEAR / (far - blockers.x * (far - RAA_AREA_HERO_NEAR));
    let radius_texels = clamp(0.5 * size * (dz - blocker_depth) / max(blocker_depth, RAA_AREA_HERO_NEAR)
        * res / (2.0 * RAA_AREA_HERO_TAN_HALF * dz), 1.0, 48.0);
    return vec2<f32>(raa_shadow_pcf(layer, uv, depth, radius_texels, res,
        raa_csm_rotation(frag.xy)), w_map);
}

// Mid-tier V (S44): the GI-owned stochastic shadow mask, blended by w_stoch over the band.
// The slot table maps slot s to a packed light index; the mask stores V in layer s / 4,
// channel s % 4, and w_stoch sits beside it in the table. A light without a slot returns
// (1, 0), so V_far keeps the cone and hero tiers unchanged. The bound view is 1x1 with
// (1, 1, 1, 1) while the mask is cleared, and the pixel clamps to its texels like
// slot_indirect.wgsl. Only textureLoad: this runs inside the nonuniform light loop.
fn raa_area_stochastic_visibility(index: u32, frag: vec4<f32>) -> vec2<f32> {
    let size = vec2<i32>(textureDimensions(raa_area_shadow_mask));
    let pixel = clamp(vec2<i32>(frag.xy), vec2<i32>(0), size - vec2<i32>(1));
    for (var s = 0u; s < min(raa_area_stochastic.params.x, 8u); s += 1u) {
        let lane = s % 4u;
        if raa_area_stochastic.lights[s / 4u][lane] == index {
            let visibility = clamp(textureLoad(raa_area_shadow_mask, pixel,
                i32(s / 4u), 0)[lane], 0.0, 1.0);
            return vec2<f32>(visibility,
                clamp(raa_area_stochastic.weights[s / 4u][lane], 0.0, 1.0));
        }
    }
    return vec2<f32>(1.0, 0.0);
}

// The caller passes the packed uniform index (not a stable CPU handle) to visibility.
fn raa_local_contribution(index: u32, budget: f32, s: MaterialOutput, c: ShadeCtx) -> vec3<f32> {
    let light = raa_local_lights[index];
    // Keep authored lights addressable even while off, but skip their shadow and BRDF work.
    if all(light.color_inv_range_sq.rgb == vec3<f32>(0.0)) { return vec3<f32>(0.0); }
    let capture = raa_feature_enabled(RAA_PLANAR_CAPTURE_FEATURE);
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
        // Forward tier (light_cutoff.rs): width > 0 is the forward radius of a point or spot light;
        // beyond it the light is probe-only, faded over [0.75, 1] of the radius.
        if light.width > 0.0 {
            falloff *= 1.0 - smoothstep(0.75 * light.width, light.width, distance);
            if falloff <= 0.0 { return vec3<f32>(0.0); }
        }
        // Forward budget of the tile (light_tiles.wgsl): a weaker light fades to the probes.
        falloff *= budget;
        if falloff <= 0.0 { return vec3<f32>(0.0); }
        if light.kind == 1u {
            falloff *= raa_local_spot(dot(light.direction_cos_outer.xyz, -direction),
                light.direction_cos_outer.w, light.cos_inner);
        }
    } else if light.kind == 3u {
        // Compile-time constant: variants of area-free frames drop the whole branch.
        if (RAA_VARIANT_FLAGS & RAA_VARIANT_NO_AREA_LIGHTS) != 0u { return vec3<f32>(0.0); }
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
        // Direct fade lane: only GI (feature 29) applies the per-light w_direct weight.
        let w_direct = clamp(light.color_inv_range_sq.w, 0.0, 1.0);
        let fade = select(1.0, w_direct, raa_feature_enabled(RAA_GI_ACTIVE_FEATURE));
        if window * fade <= 0.0 { return vec3<f32>(0.0); }
        // Visibility tiers: V_far blends the mid-tier mask V by its w_stoch; w_cone = 1 uses
        // V_far alone, while below 1 the hero map blends in by its own map weight. layer < 4
        // means no hero map was selected this frame.
        let w_cone = clamp(light.cos_inner, 0.0, 1.0);
        let layer = i32(round(light.direction_cos_outer.w));
        var hero = vec2<f32>(1.0, 0.0);
        if !capture && layer >= 4 && w_cone < 1.0 && raa_shadow.params.w > 0.5
            && layer < i32(textureNumLayers(raa_shadow_map)) {
            hero = raa_area_hero_visibility(light, layer, c.position, s.normal, c.frag);
        }
        var stochastic = vec2<f32>(1.0, 0.0);
        if !capture {
            stochastic = raa_area_stochastic_visibility(index, c.frag);
        }
        var v_cone = 1.0;
        if !capture && (w_cone > 0.0 || hero.y < 1.0) && stochastic.y < 1.0 {
            v_cone = clamp(raa_slot_local_visibility(index, c.position, s.normal,
                direction, distance), 0.0, 1.0);
        }
        let v_far = mix(v_cone, stochastic.x, stochastic.y);
        let area_visibility = mix(mix(v_far, hero.x, hero.y), v_far, w_cone);
        var shading = vec3<f32>(0.0);
        if raa_feature_enabled(RAA_AREA_FORM_FACTOR_FEATURE) {
            shading = raa_area_diffuse(light, s, c);
        } else {
            shading = raa_area_ltc(light, s, c);
        }
        return shading * (window * area_visibility * fade);
    } else if light.kind != 2u {
        // Directional ignores position and range; unknown kinds contribute nothing.
        return vec3<f32>(0.0);
    }
    // Point, spot and directional only.
    if dot(s.normal, direction) <= 0.0 || falloff <= 0.0 { return vec3<f32>(0.0); }
    var visibility = 1.0;
    if !capture {
        visibility = raa_local_shadow(light, c.position, s.normal);
        if visibility <= 0.0 { return vec3<f32>(0.0); }
        visibility *= clamp(raa_slot_local_visibility(index, c.position, s.normal,
            direction, distance), 0.0, 1.0);
    }
    let radiance = light.color_inv_range_sq.rgb * (falloff * visibility);
    if !any(radiance > vec3<f32>(0.0)) { return vec3<f32>(0.0); }
    return raa_slot_direct(s, c, direction, radiance);
}

// Handoff: Composer attaches LOCAL_LIGHTING_WGSL after the selected slot providers.
// Forward/transparent/fur callers add raa_local_lighting(s, c) to their lit branch;
// the separate sun call and Flat/UNLIT selection remain with each pass.
// Point, spot and directional lights call Direct once each. Area lights return
// raa_area_ltc (or raa_area_diffuse under feature 39) x window x area_visibility x fade:
// area_visibility blends V_hero (a true hero PCSS map, faded out over the layer border)
// with V_cone (the LocalVisibility slot) by w_cone, and fade is the GI-only w_direct lane.
// c.frag.xy is framebuffer position; light_info.x is the packed active count.
// Binding 13 stores word w at layer w of tile (x, y); layer 0 is the count.
// Providers reached here must use explicit LOD/gradients or textureLoad, no derivatives.
fn raa_local_lighting(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> {
    var result = vec3<f32>(0.0);
    if raa_frame.light_info.x == 0u { return result; }
    if raa_feature_enabled(RAA_PLANAR_CAPTURE_FEATURE) {
        for (var index = 0u; index < min(raa_frame.light_info.x, MAX_LOCAL_LIGHTS); index += 1u) {
            result += raa_local_contribution(index, 1.0, s, c);
        }
        return result;
    }
    let tile = vec2<i32>(floor(c.frag.xy / f32(TILE_SIZE)));
    let count = min(textureLoad(raa_light_tiles, tile, 0, 0).x, TILE_MAX_LIGHTS);
    for (var word = 1u; word <= count; word += 1u) {
        let entry = textureLoad(raa_light_tiles, tile, i32(word), 0).x;
        let index = entry & 0xffffu;
        let budget = 1.0 - f32((entry >> 16u) & 0xffu) / 255.0;
        if index < min(raa_frame.light_info.x, MAX_LOCAL_LIGHTS) {
            result += raa_local_contribution(index, budget, s, c);
        }
    }
    return result;
}
