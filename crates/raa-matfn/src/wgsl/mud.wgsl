// raa.mud: wet earth. The albedo follows a 5-octave fbm of the world position x `scale`
// between `color` (dry) and `wetColor`; where that fbm mask rises above 1 - `wetness` the
// mud is wet: darker, roughness down to 0.15 and a clear coat of 0.8 (standing water film).
// p[0].rgb color, p[1].rgb wetColor, p[2] = (wetness, scale, 0, 0).

// fbm of the position mapped to [0, 1].
fn mud_mask(p: vec3<f32>, scale: f32) -> f32 {
    return clamp(raa_fbm3(p * scale, 5) * 0.5 + 0.5, 0.0, 1.0);
}

// 1 where the mask is above the wet threshold 1 - wetness, soft edge; 0 for wetness 0.
fn mud_wet(mask: f32, wetness: f32) -> f32 {
    let w = clamp(wetness, 0.0, 1.0);
    let threshold = 1.0 - w;
    return smoothstep(threshold - 0.04, threshold + 0.04, mask) * step(1e-4, w);
}

fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    let color = in.record.p[0].rgb;
    let wet_color = in.record.p[1].rgb;
    let wetness = in.record.p[2].x;
    let scale = max(in.record.p[2].y, 0.0);

    let mask = mud_mask(in.position, scale);
    let wet = mud_wet(mask, wetness);
    let albedo = mix(color, wet_color, mask);
    o.base_color = mix(albedo, wet_color, wet) * in.color.rgb;
    o.metallic = 0.0;
    o.roughness = mix(0.9, 0.15, wet);
    o.clearcoat = 0.8 * wet;
    o.clearcoat_roughness = 0.05;
    o.sheen_color = vec3<f32>(0.0);
    return o;
}
