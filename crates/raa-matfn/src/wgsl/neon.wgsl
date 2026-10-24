// raa.neon: a glowing gas tube, drawn UNLIT (base + emissive). The gas glows `color` x
// `strength` across the tube, following the chord the sight line cuts through it (see
// neon_body), at a level the tone mapper keeps saturated; a narrow hot core along the tube's
// axis adds `coreWhiteness` x neon_core_gain x `strength` of a whitened colour (see
// neon_core), so the centre burns white-hot while the rims keep the saturated colour, and
// the bloom pass spreads the energy as a coloured halo.
// p[0].rgb color, p[1] = (strength, coreWhiteness, 0, 0).

const neon_core_gain: f32 = 4.0;

// Cosine between the tube normal and the view direction, 1 facing the viewer.
fn neon_facing(n: vec3<f32>, v: vec3<f32>) -> f32 {
    return clamp(dot(normalize(n), normalize(v)), 0.0, 1.0);
}

// Relative radiance of the coloured gas: the chord through a cylinder of radius R is 2R (n.v);
// it stays above 0.35 at the silhouette (glass wall and the glow of the far side).
fn neon_body(n_dot_v: f32) -> f32 {
    return mix(0.35, 1.0, n_dot_v);
}

// Weight of the hot core: (n.v)^8, about the central 40 % of the tube's width.
fn neon_core(n_dot_v: f32, whiteness: f32) -> f32 {
    let c2 = n_dot_v * n_dot_v;
    let c4 = c2 * c2;
    return clamp(whiteness, 0.0, 1.0) * c4 * c4;
}

fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    let color = in.record.p[0].rgb * in.color.rgb;
    let strength = max(in.record.p[1].x, 0.0);
    let n_dot_v = neon_facing(in.normal, in.view_dir);
    let body = neon_body(n_dot_v);
    let core = neon_core(n_dot_v, in.record.p[1].y);
    let hot = mix(color, vec3<f32>(1.0), 0.35);

    o.base_color = mix(color * body, vec3<f32>(1.0), core);
    o.emissive = (color * body + hot * (neon_core_gain * core)) * strength;
    o.metallic = 0.0;
    o.clearcoat = 0.0;
    return o;
}
