// raa.liquid.water: clear animated water over the Gerstner vertex stage.
//
// The record packs the waves and speeds in `p[0..5]` (src/waves.rs), then
// `p[5] = (deepColor linear rgb, clarity)`, `p[6] = (shallowColor linear rgb, foamWidth m)`
// and `p[7] = (foamAmount, detailStrength, detailScale, 0)`. The attenuation colour runs
// from the deep colour to the shallow colour with clarity, and so does the attenuation
// distance (0.15 m murky to 1.5 m clear). The diffuse base only carries the non-transmitted
// part, because the Transmission slot adds the refracted, absorbed picture. Foam comes from
// the shore (the opaque surface within foamWidth of the water) and from the wave crests,
// broken up by noise; it is white, rough and opaque and drops the transmission.
//
// ior, thickness and transmission come from the record (desc overrides).

fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    let deep = max(in.record.p[5].rgb, vec3<f32>(0.0));
    let shallow = max(in.record.p[6].rgb, vec3<f32>(0.0));
    let clarity = clamp(in.record.p[5].w, 0.0, 1.0);
    let foam_width = max(in.record.p[6].w, 1e-4);
    let foam_amount = clamp(in.record.p[7].x, 0.0, 1.0);
    let detail_strength = clamp(in.record.p[7].y, 0.0, 1.0);
    let detail_scale = max(in.record.p[7].z, 1e-3);

    o.transmission = clamp(o.transmission, 0.0, 1.0);
    o.metallic = 0.0;
    o.clearcoat = 0.0;
    o.roughness = 0.02;
    o.normal = liquid_detail_normal(
        o.normal,
        in.tangent,
        in.position,
        in.time,
        detail_strength,
        detail_scale,
    );

    o.attenuation_color = mix(deep, shallow, clarity);
    o.attenuation_distance = mix(0.15, 1.5, clarity);
    o.base_color = shallow * (1.0 - o.transmission);

    let crest = liquid_waves_at(in.record.p, in.local_position.xz, in.time);
    let shore = liquid_shore(in.frag_coord, foam_width);
    let foam = liquid_foam(shore, crest, in.position, in.time, foam_amount);
    return liquid_apply_foam(o, foam);
}
