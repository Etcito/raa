// raa.snow: packed snow. White diffuse (`color`, roughness 0.8) with sparkling glints:
// the world position x `scale` is split into the cells of `raa_voronoi3`; every cell
// carries a hashed facet normal (an ice crystal) tilted around the surface normal, and a
// cell whose facet aligns with the half vector of the sun flashes. The glints, lit by the
// sun of FrameGpu and shadowed by `raa_slot_shadow`, are added to `emissive` x `sparkle`.
// p[0].rgb color, p[1] = (sparkle, scale, 0, 0).

// Integer coordinates of the cell owning the nearest feature point of `p` (the same
// search and feature points as `raa_voronoi3`).
fn snow_nearest_cell(p: vec3<f32>) -> vec3<f32> {
    let i = floor(p);
    let f = p - i;
    var best = 8.0;
    var cell = i;
    for (var z = -1; z <= 1; z = z + 1) {
        for (var y = -1; y <= 1; y = y + 1) {
            for (var x = -1; x <= 1; x = x + 1) {
                let o = vec3<f32>(f32(x), f32(y), f32(z));
                let dist = length(o + raa_hash3(i + o) - f);
                if (dist < best) {
                    best = dist;
                    cell = i + o;
                }
            }
        }
    }
    return cell;
}

// Unit facet normal of a crystal: the surface normal tilted by a hashed direction.
fn snow_facet(cell: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let r = raa_hash3(cell + vec3<f32>(41.0, 17.0, 73.0)) * 2.0 - 1.0;
    return normalize(n + 0.6 * r);
}

fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    let color = in.record.p[0].rgb;
    let sparkle = max(in.record.p[1].x, 0.0);
    let scale = max(in.record.p[1].y, 0.0);

    o.base_color = color * in.color.rgb;
    o.metallic = 0.0;
    o.roughness = 0.8;
    o.clearcoat = 0.0;

    let n = normalize(in.normal);
    let v = normalize(in.view_dir);
    let l = -normalize(raa_frame.sun_dir.xyz);
    let h = normalize(l + v);
    let p = in.position * scale;
    // F1: glints are small points around each crystal's feature point.
    let f = raa_voronoi3(p);
    let cell = snow_nearest_cell(p);
    // Only a third of the cells are crystals that can flash.
    let crystal = step(0.66, raa_hash3(cell + vec3<f32>(5.0, 11.0, 23.0)).x);
    let align = pow(clamp(dot(snow_facet(cell, n), h), 0.0, 1.0), 256.0);
    let point = 1.0 - smoothstep(0.1, 0.3, f.x);
    let n_dot_l = clamp(dot(n, l), 0.0, 1.0);
    let shadow = raa_slot_shadow(in.position, n, in.frag_coord);
    let glint = crystal * align * point * n_dot_l * shadow;
    o.emissive = o.emissive + raa_frame.sun_color.rgb * glint * sparkle;
    return o;
}
