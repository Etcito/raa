// raa.slime: glossy translucent slime with suspended bubbles, a bubbly surface and a slow
// colour swirl. Bubbles are the centres of `raa_voronoi3(local_position / bubbleScale)`
// cells, kept for a hashed subset of the cells: inside a bubble the transmission drops a
// little and a bright rim marks its edge. The surface carries soft bumps (two `raa_noise3`
// octaves at a few bubble sizes) that tilt the shading normal, so the glossy surface
// breaks the sky and sun into small glints. A slow `raa_noise3` swirl over time tints the
// attenuation colour between the colour and a lighter, thinner version of it. The colour
// is the attenuation colour over 1 / absorption metres; slime keeps some body colour in
// its diffuse base, so it stays saturated over any floor. ior, thickness and transmission
// come from the record (desc overrides).
// p[0].rgb color, p[1] = (absorption per metre, glossiness 0..1, bubbleScale m, 0).

// (core, rim) masks in [0, 1] of the bubbles around the feature point of some cells.
fn slime_bubbles(p: vec3<f32>) -> vec2<f32> {
    let d = raa_voronoi3(p);
    let present = step(0.5, raa_hash3(floor(p) + vec3<f32>(41.0, 17.0, 3.0)).x);
    let radius = 0.26;
    let core = 1.0 - smoothstep(radius - 0.05, radius, d.x);
    let rim = 1.0 - smoothstep(0.0, 0.05, abs(d.x - radius));
    return vec2<f32>(core, rim) * present;
}

// Swirl in [0, 1] drifting slowly with time.
fn slime_swirl(p: vec3<f32>, t: f32) -> f32 {
    let drift = vec3<f32>(0.07, 0.03, -0.05) * t;
    let warp = raa_noise3(p * 1.3 + drift);
    return clamp(0.5 + 0.5 * raa_noise3(p * 2.1 + vec3<f32>(warp * 1.5) - drift), 0.0, 1.0);
}

// Height of the bubbly surface (about [-1, 1]) at p in bump cells.
fn slime_height(p: vec3<f32>) -> f32 {
    return 0.65 * raa_noise3(p) + 0.35 * raa_noise3(p * 2.3 + vec3<f32>(7.1, 3.7, 1.3));
}

// Shading normal tilted by the gradient of the bump height (finite differences), with the
// component along n removed so the bumps never flip the surface.
fn slime_bump(n: vec3<f32>, p: vec3<f32>, strength: f32) -> vec3<f32> {
    let e = 0.08;
    let h = slime_height(p);
    let g = vec3<f32>(
        slime_height(p + vec3<f32>(e, 0.0, 0.0)) - h,
        slime_height(p + vec3<f32>(0.0, e, 0.0)) - h,
        slime_height(p + vec3<f32>(0.0, 0.0, e)) - h,
    ) / e;
    let tangential = g - n * dot(g, n);
    return normalize(n - tangential * strength);
}

fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    let color = clamp(in.record.p[0].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    let absorption = max(in.record.p[1].x, 1e-3);
    let glossiness = clamp(in.record.p[1].y, 0.0, 1.0);
    let bubble_scale = max(in.record.p[1].z, 1e-3);

    o.roughness = mix(0.25, 0.02, glossiness);
    // Bumps about three bubbles wide.
    o.normal = slime_bump(o.normal, in.local_position / (3.0 * bubble_scale), 0.2);

    let b = slime_bubbles(in.local_position / bubble_scale);
    let t0 = clamp(o.transmission, 0.0, 1.0);
    o.transmission = clamp(t0 * (1.0 - 0.3 * b.x - 0.45 * b.y), 0.0, 1.0);

    let swirl = slime_swirl(in.local_position, in.time);
    let thin = sqrt(color);
    o.attenuation_color = mix(color, thin, 0.4 * swirl);
    o.attenuation_distance = 1.0 / absorption;

    // Body colour (more where the slime is less transmissive), plus a bright rim where a
    // bubble wall meets the eye.
    let body = color * (0.12 + 0.6 * (1.0 - o.transmission));
    let rim = mix(color, vec3<f32>(1.0), 0.7) * b.y * 0.8;
    o.base_color = (body + rim) * in.color.rgb;
    o.metallic = 0.0;
    o.clearcoat = 0.0;
    return o;
}
