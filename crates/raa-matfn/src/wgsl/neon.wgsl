// raa.neon: a glowing gas tube, drawn UNLIT (base + emissive). The glow is `color` x
// `strength` (the emissive scale the bloom pass spreads); the core facing the viewer
// whitens by `coreWhiteness` x (n.v)^2 so the tube centre reads white-hot.
// p[0].rgb color, p[1] = (strength, coreWhiteness, 0, 0).

// Weight of the white-hot core: 1 where the tube faces the viewer, 0 at its silhouette.
fn neon_core(n: vec3<f32>, v: vec3<f32>, whiteness: f32) -> f32 {
    let n_dot_v = clamp(dot(normalize(n), normalize(v)), 0.0, 1.0);
    return clamp(whiteness, 0.0, 1.0) * n_dot_v * n_dot_v;
}

fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    let color = in.record.p[0].rgb * in.color.rgb;
    let strength = max(in.record.p[1].x, 0.0);
    let core = neon_core(in.normal, in.view_dir, in.record.p[1].y);

    o.base_color = mix(color, vec3<f32>(1.0), core);
    o.emissive = color * strength;
    o.metallic = 0.0;
    o.clearcoat = 0.0;
    return o;
}
