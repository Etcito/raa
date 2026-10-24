// raa.liquid.slime: thick, glossy, translucent slime over the Gerstner vertex stage.
//
// The record packs the waves and speeds in `p[0..5]` (src/waves.rs), then
// `p[5] = (color linear rgb, absorption per metre)`, `p[6] = (glossiness, bubbleScale m, 0,
// foamWidth m)` and `p[7] = (foamAmount, detailStrength, detailScale, 0)`. The attenuation
// colour is the slime colour over `1 / absorption` metres; the diffuse base keeps the body
// colour of the non-transmitted part. Voronoi bubbles are domed (their height gradient
// tilts the shading normal), darken the body and catch a sharp clear coat highlight on the
// wall. Slow detail normals make the surface wobble; lace foam marks the shore and crests.
//
// ior, thickness and transmission come from the record (desc overrides).

// Bubble height field in [0, 1] at `p` in bubble cells: a dome over the feature point of
// the present cells. The height gradient is taken with finite differences below; `p` is in
// cells, so the small gain below keeps the world-space slope of a dome low.
fn slime_bubble_height(p: vec3<f32>) -> f32 {
    let d = raa_voronoi3(p).x;
    let present = step(0.42, raa_hash3(floor(p) + vec3<f32>(19.0, 7.0, 31.0)).x);
    return (1.0 - smoothstep(0.07, 0.3, d)) * present;
}

// Core and rim masks (0..1) of the nearest bubble. A dome on 60% of the cells with a wide
// bright rim is what makes the bubbles readable at capture distance (cards 5018 and 5030:
// they were too few and too faint).
fn slime_bubble_masks(p: vec3<f32>) -> vec2<f32> {
    let d = raa_voronoi3(p).x;
    let present = step(0.42, raa_hash3(floor(p) + vec3<f32>(19.0, 7.0, 31.0)).x);
    // Larger cells with a wider, brighter rim so the bubbles read at capture distance
    // (card 5031 saw none of the old 5 cm domes).
    let radius = 0.3;
    let core = 1.0 - smoothstep(radius - 0.1, radius, d);
    let rim = 1.0 - smoothstep(0.0, 0.1, abs(d - radius));
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

    let p = in.local_position / bubble_scale;
    let bubbles = slime_bubble_masks(p);
    // A bubble is a lens of slightly thicker slime: the body and the absorbed floor darken a
    // little inside the core (a strong darkening read as dark holes, card 4941 preview),
    // while its rim catches a thin bright skin. The rim goes
    // to a pale green-white, not the body's own green, or it vanishes against the slime
    // (card 5031: no visible bubbles on a green sheet).
    o.attenuation_color = color * (1.0 - 0.2 * bubbles.x);
    o.attenuation_distance = 1.0 / absorption;
    o.base_color = color * (1.0 - o.transmission) * (1.0 - 0.2 * bubbles.x);
    // The bubble skin is a light, opaque ring of slime: it reads as a bubble outline over
    // the floor seen through the body.
    // A saturated light green skin: the old pale green-white ring at 0.85 read as grey
    // circular foam marks on the river (card 4941 judge).
    o.base_color = mix(o.base_color, vec3<f32>(0.5, 1.0, 0.22), 0.6 * bubbles.y);
    o.transmission = o.transmission * (1.0 - 0.5 * bubbles.y);

    // A domed bubble wall: a finite-difference tilt plus a glossy clear coat ring.
    let e = 0.012;
    let hx = slime_bubble_height(p + vec3<f32>(e, 0.0, 0.0))
        - slime_bubble_height(p - vec3<f32>(e, 0.0, 0.0));
    let hz = slime_bubble_height(p + vec3<f32>(0.0, 0.0, e))
        - slime_bubble_height(p - vec3<f32>(0.0, 0.0, e));
    let slope = vec2<f32>(-hx, -hz) * (0.2 / (2.0 * e));
    let m = length(slope);
    let limited = slope * select(1.0, 0.6 / max(m, 1e-6), m > 0.6);
    o.normal = raa_perturb_normal(o.normal, in.tangent, normalize(vec3<f32>(limited, 1.0)));
    o.roughness = mix(o.roughness, 0.02, bubbles.y);
    // A strong but not mirror-wide coat: the glossy green body stays readable under the
    // sky reflection instead of turning into a grey-blue skin.
    o.clearcoat = 0.55;
    o.clearcoat_roughness = mix(0.05, 0.02, bubbles.y);

    let crest = liquid_waves_at(in.record.p, in.local_position.xz, in.time);
    let shore = max(liquid_shore(in.frag_coord, foam_width), liquid_mesh_edge(in.uv, 0.14));
    let foam = liquid_foam(shore, crest, in.position, in.time, foam_amount);
    return liquid_apply_foam(o, foam);
}
