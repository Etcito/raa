// raa.candy: hard candy with helical stripes. `stripeCount` stripeColor stripes run around
// uv.x and twist along uv.y by `swirl` turns; the clear coat is the record's (desc
// `clearcoat`, `clearcoatRoughness`, default 0), so clearcoat 0 really removes it.
// Stripe edges are antialiased over one pixel (screen derivatives of the stripe phase).
// p[0].rgb color, p[1].rgb stripeColor, p[2] = (stripeCount, swirl, 0, 0).

// 1 on a stripe, 0 between stripes, edges one pixel soft; half of each period is striped,
// so stripes under ~2 pixels fade to their average 0.5 instead of aliasing.
fn candy_stripe(uv: vec2<f32>, count: f32, swirl: f32) -> f32 {
    let phase = (uv.x + swirl * uv.y) * count;
    // Periods per pixel, ignoring the u wrap seam (the smaller of two shifted copies).
    let shifted = (fract(uv.x + 0.5) + swirl * uv.y) * count;
    let fw = min(fwidth(phase), fwidth(shifted));
    let edge = max(fw, 0.04);
    let tri = abs(fract(phase) - 0.5) * 2.0;
    let stripe = smoothstep(0.5 - edge, 0.5 + edge, tri);
    return mix(stripe, 0.5, clamp(fw * 2.0 - 0.5, 0.0, 1.0)) * step(0.5, count);
}

fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    let color = in.record.p[0].rgb;
    let stripe_color = in.record.p[1].rgb;
    let count = round(max(in.record.p[2].x, 0.0));
    let swirl = in.record.p[2].y;

    let stripe = candy_stripe(in.uv, count, swirl);
    o.base_color = mix(color, stripe_color, stripe) * in.color.rgb;
    // Hard candy gloss under the coat; the coat stays raa_default_output's.
    o.metallic = 0.0;
    o.roughness = 0.25;
    return o;
}
