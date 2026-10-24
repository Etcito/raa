// raa.ice: clear tinted ice with inner cracks, small bubbles and frosted patches.
// Inner cracks are thin bright planes where the F1-F2 distance of a domain-warped
// `raa_voronoi3(local_position / crackScale)` is small (the cell walls), kept where
// low-frequency noise allows so they form a few irregular fractures instead of a regular
// network; bubbles are the centres of finer voronoi cells, kept for a hashed subset of the
// cells. Both are evaluated at several depths along the refracted view ray, so they sit
// inside the block (parallax) rather than on its faces. Frost raises the roughness and
// whitens: `raa_fbm3` patches scaled by `cloudiness` (also on upward faces) and a rim where
// the geometric normal leaves the axes (rounded edges). A gentle melt ripple bends the
// shading normal. Cracks and bubbles scatter light: they lower the transmission locally,
// raise the roughness, kink the normal (refraction offset along the fracture) and read
// white. The tint is the
// attenuation colour over 1 / absorption metres; the diffuse base only carries the
// non-transmitted part.
// ior, thickness and transmission come from the record (desc overrides).
// p[0].rgb tint, p[1] = (absorption per metre, cloudiness 0..1, crackScale m, 0).

// Smooth 3D noise vector in [-1, 1]^3 (three decorrelated gradient noises).
fn ice_noise_vec(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        raa_noise3(p),
        raa_noise3(p + vec3<f32>(5.2, 1.3, 2.8)),
        raa_noise3(p + vec3<f32>(9.1, 4.7, 6.3)),
    );
}

// Crack mask in [0, 1]: 1 on a (warped) voronoi cell wall, fading over a thin band.
fn ice_cracks(p: vec3<f32>) -> f32 {
    let d = raa_voronoi3(p + 0.45 * ice_noise_vec(p * 0.8));
    let wall = d.y - d.x;
    // Only some walls are cracks: break the network up with low-frequency noise.
    let keep = smoothstep(-0.05, 0.2, raa_noise3(p * 0.45 + vec3<f32>(3.1, 7.7, 1.3)));
    return (1.0 - smoothstep(0.0, 0.08, wall)) * keep;
}

// Bubble mask in [0, 1]: a small sphere around the feature point of a quarter of the cells.
fn ice_bubbles(p: vec3<f32>) -> f32 {
    let d = raa_voronoi3(p);
    let cell = floor(p);
    let present = step(0.75, raa_hash3(cell + vec3<f32>(13.0, 29.0, 7.0)).y);
    return (1.0 - smoothstep(0.1, 0.16, d.x)) * present;
}

fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    let tint = max(in.record.p[0].rgb, vec3<f32>(0.0));
    let absorption = max(in.record.p[1].x, 0.0);
    let cloudiness = clamp(in.record.p[1].y, 0.0, 1.0);
    let crack_scale = max(in.record.p[1].z, 1e-3);

    let n = normalize(o.normal);
    let q = in.local_position / crack_scale;
    // Refracted view ray, used (in cell units) to look into the block.
    let into = refract(-in.view_dir, n, 1.0 / max(o.ior, 1.0));
    var cracks = 0.0;
    var bubbles = 0.0;
    // Three layers 0.15, 0.575 and 1 cell deep along the refracted ray.
    for (var i = 0; i < 3; i = i + 1) {
        let p = q + into * (0.15 + 0.425 * f32(i));
        // Deeper samples fade: the medium in front of them dims what they scatter.
        let fade = 1.0 - 0.25 * f32(i);
        cracks = max(cracks, ice_cracks(p) * fade);
        bubbles = max(bubbles, ice_bubbles(p * 2.0 + vec3<f32>(5.3, 1.9, 8.4)) * fade);
    }
    let scatter = clamp(cracks + 0.9 * bubbles, 0.0, 1.0);

    // Frost: fbm patches over the surface, stronger with cloudiness, and a frosted rim
    // where the geometric normal leaves the axes (rounded edges and corners of a block).
    let frost = clamp(0.5 + 0.5 * raa_fbm3(q * 1.7, 4), 0.0, 1.0);
    let g = abs(normalize(in.normal));
    let rim = smoothstep(0.06, 0.25, 1.0 - max(g.x, max(g.y, g.z))) * (0.35 + 0.65 * frost);
    // Frost settles on upward faces: patches there, scaled by cloudiness.
    let top = smoothstep(0.6, 0.95, n.y) * smoothstep(0.3, 0.7, frost);
    let patches = smoothstep(0.45, 0.8, frost) + top;
    let frosted = clamp(cloudiness * patches + rim * (0.5 + cloudiness), 0.0, 1.0);
    let base_rough = clamp(o.roughness, 0.03, 1.0);
    o.roughness = clamp(mix(base_rough, 0.5, frosted) + 0.25 * scatter, 0.03, 1.0);

    // Melt ripple over the faces, plus a kink across each crack plane.
    let ripple = ice_noise_vec(in.local_position * 6.0);
    let kink = ice_noise_vec(q * 2.3 + vec3<f32>(1.7, 8.2, 3.9));
    o.normal = normalize(n + 0.05 * ripple + 0.3 * cracks * kink);

    let t0 = clamp(o.transmission, 0.0, 1.0);
    let transmission = t0 * (1.0 - 0.55 * scatter) * (1.0 - 0.35 * frosted);
    o.transmission = clamp(transmission, 0.0, 1.0);
    o.attenuation_color = tint;
    o.attenuation_distance = 1.0 / max(absorption, 1e-3);

    // Diffuse lobe only for the non-transmitted part; frost whitens the tint, cracks and
    // bubbles read white.
    let frost_white = mix(tint, vec3<f32>(1.0), 0.6);
    let base = mix(mix(tint, frost_white, frosted), vec3<f32>(1.0), scatter);
    o.base_color = base * in.color.rgb * (1.0 - o.transmission);
    o.metallic = 0.0;
    o.clearcoat = 0.0;
    return o;
}
