// raa.ice: clear tinted ice with inner cracks, small bubbles and frosted patches.
// Inner cracks are thin bright planes where the F1-F2 distance of
// `raa_voronoi3(local_position / crackScale)` is small (the cell walls); bubbles are the
// centres of finer voronoi cells, kept for a hashed subset of the cells. `raa_fbm3` and
// `cloudiness` vary the frosted roughness. Cracks and bubbles scatter light: they lower the
// transmission locally and raise the roughness. The tint is the attenuation colour over
// 1 / absorption metres; the diffuse base only carries the non-transmitted part.
// ior, thickness and transmission come from the record (desc overrides).
// p[0].rgb tint, p[1] = (absorption per metre, cloudiness 0..1, crackScale m, 0).

// Crack mask in [0, 1]: 1 on a voronoi cell wall, fading over a thin band.
fn ice_cracks(p: vec3<f32>) -> f32 {
    let d = raa_voronoi3(p);
    let wall = d.y - d.x;
    // Only some walls are cracks: break the network up with low-frequency noise.
    let keep = smoothstep(-0.1, 0.25, raa_noise3(p * 0.7 + vec3<f32>(3.1, 7.7, 1.3)));
    return (1.0 - smoothstep(0.0, 0.05, wall)) * keep;
}

// Bubble mask in [0, 1]: a small sphere around the feature point of a third of the cells.
fn ice_bubbles(p: vec3<f32>) -> f32 {
    let d = raa_voronoi3(p);
    let cell = floor(p);
    let present = step(0.66, raa_hash3(cell + vec3<f32>(13.0, 29.0, 7.0)).y);
    return (1.0 - smoothstep(0.08, 0.16, d.x)) * present;
}

fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    let tint = max(in.record.p[0].rgb, vec3<f32>(0.0));
    let absorption = max(in.record.p[1].x, 0.0);
    let cloudiness = clamp(in.record.p[1].y, 0.0, 1.0);
    let crack_scale = max(in.record.p[1].z, 1e-3);

    let q = in.local_position / crack_scale;
    let cracks = ice_cracks(q);
    let bubbles = ice_bubbles(q * 4.0 + vec3<f32>(5.3, 1.9, 8.4));
    let scatter = clamp(cracks + 0.8 * bubbles, 0.0, 1.0);

    // Frost: fbm patches over the surface, stronger with cloudiness.
    let frost = clamp(0.5 + 0.5 * raa_fbm3(q * 1.7, 4), 0.0, 1.0);
    let frosted = cloudiness * smoothstep(0.35, 0.75, frost);
    let base_rough = clamp(o.roughness, 0.03, 1.0);
    o.roughness = clamp(
        mix(base_rough, 0.6, frosted) + 0.5 * scatter,
        0.03,
        1.0,
    );

    let t0 = clamp(o.transmission, 0.0, 1.0);
    let transmission = t0 * (1.0 - 0.65 * scatter) * (1.0 - 0.4 * cloudiness * frost);
    o.transmission = clamp(transmission, 0.0, 1.0);
    o.attenuation_color = tint;
    o.attenuation_distance = 1.0 / max(absorption, 1e-3);

    // Diffuse lobe only for the non-transmitted part; cracks and bubbles read bright.
    let white = mix(tint, vec3<f32>(1.0), 0.7);
    o.base_color = mix(tint, white, scatter) * in.color.rgb * (1.0 - o.transmission);
    o.metallic = 0.0;
    o.clearcoat = 0.0;
    o.normal = normalize(o.normal);
    return o;
}
