// raa.liquid.slime: thick, glossy, translucent slime over the Gerstner vertex stage.
//
// The record packs the waves and speeds in `p[0..5]` (src/waves.rs), then
// `p[5] = (color linear rgb, absorption per metre)`, `p[6] = (glossiness, bubbleScale m, 0,
// foamWidth m)` and `p[7] = (foamAmount, detailStrength, detailScale, 0)`. The attenuation
// colour is the slime colour over `1 / absorption` metres; the diffuse base keeps the body
// colour of the non-transmitted part. Small voronoi bubbles darken the body and catch a
// sharp specular dot on their wall. Slow detail normals and a bright crest term make the
// surface wobble and glint; a subtle white foam marks the shore.
//
// ior, thickness and transmission come from the record (desc overrides).

// (core, rim) masks in [0, 1] of the bubbles around the feature point of some voronoi
// cells, at `p` in bubble cells.
fn slime_bubbles(p: vec3<f32>) -> vec2<f32> {
    let d = raa_voronoi3(p);
    let present = step(0.5, raa_hash3(floor(p) + vec3<f32>(19.0, 7.0, 31.0)).x);
    let radius = 0.24;
    let core = 1.0 - smoothstep(radius - 0.06, radius, d.x);
    let rim = 1.0 - smoothstep(0.0, 0.045, abs(d.x - radius));
    return vec2<f32>(core, rim) * present;
}

fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    let color = clamp(in.record.p[5].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let absorption = max(in.record.p[5].w, 1e-3);
    let glossiness = clamp(in.record.p[6].x, 0.0, 1.0);
    let bubble_scale = max(in.record.p[6].y, 1e-3);
    let foam_width = max(in.record.p[6].w, 1e-4);
    let foam_amount = clamp(in.record.p[7].x, 0.0, 1.0);
    let detail_strength = clamp(in.record.p[7].y, 0.0, 1.0);
    let detail_scale = max(in.record.p[7].z, 1e-3);

    o.transmission = clamp(o.transmission, 0.0, 1.0);
    o.metallic = 0.0;
    o.roughness = mix(0.35, 0.02, glossiness);
    o.normal = liquid_detail_normal(
        o.normal,
        in.tangent,
        in.position,
        in.time,
        detail_strength,
        detail_scale,
    );

    let bubbles = slime_bubbles(in.local_position / bubble_scale);
    o.attenuation_color = color * (1.0 - 0.35 * bubbles.x);
    o.attenuation_distance = 1.0 / absorption;
    o.base_color = color * (1.0 - o.transmission) * (1.0 - 0.5 * bubbles.x);

    // A glossy clear coat on the bubble wall makes the specular dots.
    o.roughness = mix(o.roughness, 0.03, bubbles.y);
    o.clearcoat = 0.5 * bubbles.y;
    o.clearcoat_roughness = 0.05;

    let crest = liquid_waves_at(in.record.p, in.local_position.xz, in.time);
    let shore = liquid_shore(in.frag_coord, foam_width);
    let foam = liquid_foam(shore, crest, in.position, in.time, foam_amount);
    return liquid_apply_foam(o, foam);
}
