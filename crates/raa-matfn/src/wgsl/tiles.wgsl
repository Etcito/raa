// raa.tiles: square tiles on a uv grid of `tileSize` (uv units per tile) separated by
// grout lines `groutWidth` wide (uv units). Grout gets `groutColor`, ao 0.5 and roughness
// 0.9; tiles get `color` and roughness 1 - `gloss`.
// p[0].rgb color, p[1].rgb groutColor, p[2] = (tileSize, groutWidth, gloss, 0).

// 1 on a grout line, 0 on a tile, with a narrow soft edge.
fn tiles_grout(uv: vec2<f32>, size: f32, width: f32) -> f32 {
    let s = max(size, 1e-4);
    let f = fract(uv / s);
    // Distance to the nearest tile edge, in uv units.
    let e = min(f, vec2<f32>(1.0) - f) * s;
    let d = min(e.x, e.y);
    let half_w = clamp(width, 0.0, s) * 0.5;
    let soft = max(half_w * 0.15, 1e-5);
    return (1.0 - smoothstep(half_w - soft, half_w + soft, d)) * step(1e-6, width);
}

fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    let color = in.record.p[0].rgb;
    let grout_color = in.record.p[1].rgb;
    let tile_size = in.record.p[2].x;
    let grout_width = in.record.p[2].y;
    let gloss = clamp(in.record.p[2].z, 0.0, 1.0);

    let grout = tiles_grout(in.uv, tile_size, grout_width);
    o.base_color = mix(color, grout_color, grout) * in.color.rgb;
    o.metallic = 0.0;
    o.roughness = mix(max(1.0 - gloss, 0.03), 0.9, grout);
    o.ao = mix(1.0, 0.5, grout);
    o.clearcoat = 0.0;
    return o;
}
