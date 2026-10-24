// raa.glass: smooth or frosted glass. The tint is the attenuation colour (over the record's
// attenuation distance), the roughness comes from the parameter (it also blurs the refraction
// through the opaque-colour mip) and the diffuse base only carries the non-transmitted part.
// ior, thickness and transmission come from the record (desc overrides).
// p[0].rgb tint, p[1] = (roughness, 0, 0, 0).

fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    let tint = max(in.record.p[0].rgb, vec3<f32>(0.0));
    o.roughness = clamp(in.record.p[1].x, 0.03, 1.0);
    o.transmission = clamp(o.transmission, 0.0, 1.0);
    o.attenuation_color = tint;
    o.base_color = tint * in.color.rgb * (1.0 - o.transmission);
    o.metallic = 0.0;
    return o;
}
