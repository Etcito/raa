// raa.neon: a glowing gas tube, drawn UNLIT (base + emissive). The gas glows `color` x
// `strength` per unit of sight line, so the radiance follows the chord the sight line cuts
// through the tube (see neon_chord): the centre burns to white in the tone mapper while the
// edges keep the saturated colour, and the bloom pass spreads the energy. The core facing
// the viewer also whitens by `coreWhiteness` x (n.v)^2 so the tube centre reads white-hot.
// p[0].rgb color, p[1] = (strength, coreWhiteness, 0, 0).

// Cosine between the tube normal and the view direction, 1 facing the viewer.
fn neon_facing(n: vec3<f32>, v: vec3<f32>) -> f32 {
    return clamp(dot(normalize(n), normalize(v)), 0.0, 1.0);
}

// Relative radiance across the tube: the chord through a cylinder of radius R is 2R (n.v);
// it stays above 0.3 at the silhouette (glass wall and the glow of the far side).
fn neon_chord(n_dot_v: f32) -> f32 {
    return mix(0.3, 1.0, n_dot_v);
}

// Weight of the white-hot core: 1 where the tube faces the viewer, 0 at its silhouette.
fn neon_core(n_dot_v: f32, whiteness: f32) -> f32 {
    return clamp(whiteness, 0.0, 1.0) * n_dot_v * n_dot_v;
}

fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    let color = in.record.p[0].rgb * in.color.rgb;
    let strength = max(in.record.p[1].x, 0.0);
    let n_dot_v = neon_facing(in.normal, in.view_dir);
    let chord = neon_chord(n_dot_v);
    let core = neon_core(n_dot_v, in.record.p[1].y);

    o.base_color = mix(color, vec3<f32>(1.0), core) * chord;
    o.emissive = color * (strength * chord);
    o.metallic = 0.0;
    o.clearcoat = 0.0;
    return o;
}
