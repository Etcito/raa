// raa.slime: glossy translucent slime with suspended bubbles and a slow colour swirl.
// Bubbles are the centres of `raa_voronoi3(local_position / bubbleScale)` cells, kept for a
// hashed subset of the cells: inside a bubble the transmission drops a little and a bright
// rim marks its edge. A slow `raa_noise3` swirl over time tints the attenuation colour
// between the colour and a lighter, thinner version of it. The colour is the attenuation
// colour over 1 / absorption metres; slime keeps a little body colour in its diffuse base.
// ior, thickness and transmission come from the record (desc overrides).
// p[0].rgb color, p[1] = (absorption per metre, glossiness 0..1, bubbleScale m, 0).

// (core, rim) masks in [0, 1] of the bubbles around the feature point of some cells.
fn slime_bubbles(p: vec3<f32>) -> vec2<f32> {
    let d = raa_voronoi3(p);
    let present = step(0.55, raa_hash3(floor(p) + vec3<f32>(41.0, 17.0, 3.0)).x);
    let radius = 0.22;
    let core = 1.0 - smoothstep(radius - 0.04, radius, d.x);
    let rim = (1.0 - smoothstep(0.0, 0.035, abs(d.x - radius)));
    return vec2<f32>(core, rim) * present;
}

// Swirl in [0, 1] drifting slowly with time.
fn slime_swirl(p: vec3<f32>, t: f32) -> f32 {
    let drift = vec3<f32>(0.07, 0.03, -0.05) * t;
    let warp = raa_noise3(p * 1.3 + drift);
    return clamp(0.5 + 0.5 * raa_noise3(p * 2.1 + vec3<f32>(warp * 1.5) - drift), 0.0, 1.0);
}

fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    let color = clamp(in.record.p[0].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let absorption = max(in.record.p[1].x, 1e-3);
    let glossiness = clamp(in.record.p[1].y, 0.0, 1.0);
    let bubble_scale = max(in.record.p[1].z, 1e-3);

    o.roughness = mix(0.25, 0.02, glossiness);

    let b = slime_bubbles(in.local_position / bubble_scale);
    let t0 = clamp(o.transmission, 0.0, 1.0);
    o.transmission = clamp(t0 * (1.0 - 0.35 * b.x - 0.5 * b.y), 0.0, 1.0);

    let swirl = slime_swirl(in.local_position, in.time);
    let thin = sqrt(color);
    o.attenuation_color = mix(color, thin, 0.4 * swirl);
    o.attenuation_distance = 1.0 / absorption;

    // A little body colour, plus a bright rim where a bubble wall meets the eye.
    let body = color * (1.0 - o.transmission) * 0.5;
    let rim = mix(color, vec3<f32>(1.0), 0.6) * b.y * 0.6;
    o.base_color = (body + rim) * in.color.rgb;
    o.metallic = 0.0;
    o.clearcoat = 0.0;
    return o;
}
