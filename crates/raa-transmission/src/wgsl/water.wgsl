// raa.water: clear water. A faint ripple shading normal comes from `raa_fbm3` of the world
// position and time, applied through `raa_perturb_normal`. The attenuation colour runs from
// the deep colour to the shallow colour with clarity, and so does the attenuation distance
// (0.3 m murky to 4 m clear). The diffuse base only carries the non-transmitted part.
// ior (1.33 for water is set by the desc), thickness and transmission come from the record.
// p[0].rgb deepColor, p[1].rgb shallowColor, p[2] = (clarity 0..1, 0, 0, 0).

// Tangent-space ripple normal from two decorrelated fbm fields, strong enough that the
// refracted grout lines visibly waver and the surface reflects a broken sky.
fn water_ripple(p: vec3<f32>, t: f32) -> vec3<f32> {
    let q = vec3<f32>(p.x, p.z, p.y) * 3.0;
    let flow = vec3<f32>(0.21, -0.17, 0.0) * t;
    let dx = raa_fbm3(q + flow, 3);
    let dy = raa_fbm3(q.yxz + vec3<f32>(11.3, 5.1, 2.7) - flow, 3);
    return normalize(vec3<f32>(0.14 * dx, 0.14 * dy, 1.0));
}

fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    let deep = max(in.record.p[0].rgb, vec3<f32>(0.0));
    let shallow = max(in.record.p[1].rgb, vec3<f32>(0.0));
    let clarity = clamp(in.record.p[2].x, 0.0, 1.0);

    o.roughness = 0.02;
    o.normal = raa_perturb_normal(o.normal, in.tangent, water_ripple(in.position, in.time));

    o.transmission = clamp(o.transmission, 0.0, 1.0);
    o.attenuation_color = mix(deep, shallow, clarity);
    o.attenuation_distance = mix(0.15, 1.2, clarity);
    o.base_color = shallow * in.color.rgb * (1.0 - o.transmission);
    o.metallic = 0.0;
    o.clearcoat = 0.0;
    return o;
}
