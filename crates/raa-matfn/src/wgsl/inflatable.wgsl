// raa.inflatable: glossy inflated plastic. Gores between `seamCount` meridian seams along
// uv.x bulge like pillows (the shading normal bends towards each seam by `seamDepth`);
// every other gore carries a stripeColor band `stripeWidth` wide (fraction of the gore).
// The clear coat is the record's (desc `clearcoat`, `clearcoatRoughness`, default 0), so a
// desc with clearcoat 0 really has no top layer; the base under it is a satin plastic.
// p[0].rgb color, p[1].rgb stripeColor, p[2] = (stripeWidth, seamDepth, seamCount, 0).

// Position across the gore holding `u`: -1 at its left seam, 0 at its centre, +1 at its
// right seam; .y = 1 on odd gores.
fn inflatable_gore(u: f32, count: f32) -> vec2<f32> {
    let g = u * count;
    let s = fract(g) * 2.0 - 1.0;
    let odd = fract(floor(g) * 0.5) * 2.0;
    return vec2<f32>(s, odd);
}

// 1 inside the centred band of an odd gore, soft edges.
fn inflatable_stripe(gore: vec2<f32>, width: f32) -> f32 {
    let half_w = clamp(width, 0.0, 1.0);
    let inside = 1.0 - smoothstep(half_w - 0.04, half_w + 0.04, abs(gore.x));
    return inside * gore.y * step(0.001, width);
}

// Pillow profile h = depth * (1 - s^4): flat centre, steep at the seams. Returns the
// tangent-space normal of that profile across the gore.
fn inflatable_bend(s: f32, depth: f32) -> vec3<f32> {
    let slope = 8.0 * depth * s * s * s;
    return normalize(vec3<f32>(slope, 0.0, 1.0));
}

fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    let color = in.record.p[0].rgb;
    let stripe_color = in.record.p[1].rgb;
    let stripe_width = in.record.p[2].x;
    let seam_depth = max(in.record.p[2].y, 0.0);
    let seams = round(max(in.record.p[2].z, 0.0));

    let gore = inflatable_gore(in.uv.x, max(seams, 1.0));
    let stripe = inflatable_stripe(gore, stripe_width);
    o.base_color = mix(color, stripe_color, stripe) * in.color.rgb;

    let depth = seam_depth * step(0.5, seams);
    o.normal = raa_perturb_normal(in.normal, in.tangent, inflatable_bend(gore.x, depth));
    // The crease at a seam receives less ambient light.
    let s2 = gore.x * gore.x;
    o.ao = 1.0 - clamp(depth * 3.0, 0.0, 0.5) * s2 * s2 * s2 * s2;

    // Base roughness 0.24: a broad soft highlight the coat's crisp one sits on (the coat
    // weight and roughness stay raa_default_output's, from the record).
    o.metallic = 0.0;
    o.roughness = 0.24;
    return o;
}
