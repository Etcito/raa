// raa.inflatable: glossy inflated plastic. Gores between `seamCount` meridian seams along
// uv.x bulge like pillows (the shading normal bends towards each seam by `seamDepth`);
// every other gore carries a stripeColor band `stripeWidth` wide (fraction of the gore).
// The clear coat is the record's (desc `clearcoat`, `clearcoatRoughness`, default 0), so a
// desc with clearcoat 0 really has no top layer; the base under it is a satin plastic.
// Stripe edges, the pillow bend and both specular lobes are antialiased from the pixel
// footprint (screen derivatives), so seams and glints do not sparkle or stair-step.
// p[0].rgb color, p[1].rgb stripeColor, p[2] = (stripeWidth, seamDepth, seamCount, 0).

// Position across the gore holding `u`: -1 at its left seam, 0 at its centre, +1 at its
// right seam; .y = 1 on odd gores.
fn inflatable_gore(u: f32, count: f32) -> vec2<f32> {
    let g = u * count;
    let s = fract(g) * 2.0 - 1.0;
    let odd = fract(floor(g) * 0.5) * 2.0;
    return vec2<f32>(s, odd);
}

// Gores per pixel along u (the u wrap seam ignored: the smaller of two shifted copies).
fn inflatable_footprint(u: f32, count: f32) -> f32 {
    return min(fwidth(u), fwidth(fract(u + 0.5))) * count;
}

// 1 inside the centred band of an odd gore, edges one pixel soft (`fw` gores per pixel);
// fades to the band's average coverage once a gore pair is under ~2 pixels.
fn inflatable_stripe(gore: vec2<f32>, width: f32, fw: f32) -> f32 {
    let half_w = clamp(width, 0.0, 1.0);
    let edge = max(fw, 0.02);
    let inside = 1.0 - smoothstep(half_w - edge, half_w + edge, abs(gore.x));
    let band = inside * gore.y;
    let average = 0.5 * half_w;
    return mix(band, average, clamp(fw * 2.0 - 0.5, 0.0, 1.0)) * step(0.001, width);
}

// Round pillow h = depth * (1 - s^2) across the gore: the whole panel bulges and the
// seams crease. Returns the tangent-space normal of that profile.
fn inflatable_bend(s: f32, depth: f32) -> vec3<f32> {
    let slope = 4.0 * depth * s;
    return normalize(vec3<f32>(slope, 0.0, 1.0));
}

// Roughness widened by the normal's change over a pixel (Kaplanyan-style variance on
// alpha^2 = roughness^4), capped so a lobe never turns fully diffuse.
fn inflatable_aa_roughness(roughness: f32, variance: f32) -> f32 {
    let r2 = roughness * roughness;
    return clamp(sqrt(sqrt(r2 * r2 + variance)), roughness, 1.0);
}

fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    let color = in.record.p[0].rgb;
    let stripe_color = in.record.p[1].rgb;
    let stripe_width = in.record.p[2].x;
    let seam_depth = max(in.record.p[2].y, 0.0);
    let seams = round(max(in.record.p[2].z, 0.0));
    let count = max(seams, 1.0);

    let gore = inflatable_gore(in.uv.x, count);
    let fw = inflatable_footprint(in.uv.x, count);
    let stripe = inflatable_stripe(gore, stripe_width, fw);
    o.base_color = mix(color, stripe_color, stripe) * in.color.rgb;

    // The bend fades out where a gore is only a few pixels wide (towards the poles).
    let resolved = clamp((0.5 - fw) * 3.0, 0.0, 1.0);
    let depth = seam_depth * step(0.5, seams) * resolved;
    o.normal = raa_perturb_normal(in.normal, in.tangent, inflatable_bend(gore.x, depth));
    // The crease at a seam receives less ambient light.
    let s2 = gore.x * gore.x;
    o.ao = 1.0 - clamp(depth * 3.0, 0.0, 0.5) * s2 * s2 * s2 * s2;

    // Base roughness 0.24: a broad soft highlight the coat's crisp one sits on (the coat
    // weight stays raa_default_output's, from the record); both lobes widen where the
    // shading normal turns fast across a pixel.
    let dn = fwidth(o.normal);
    let variance = min(0.5 * dot(dn, dn), 0.18);
    o.metallic = 0.0;
    o.roughness = inflatable_aa_roughness(0.24, variance);
    o.clearcoat_roughness = inflatable_aa_roughness(o.clearcoat_roughness, variance);
    return o;
}
