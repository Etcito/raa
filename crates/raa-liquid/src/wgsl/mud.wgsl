// raa.liquid.mud: wet, slow, opaque mud over the shared Gerstner vertex stage.
//
// The record packs the waves and speeds in `p[0..5]` (the water layout at 0.25x amplitude
// and 0.08x speed, src/waves.rs), then `p[5] = (color linear rgb, wetness)`,
// `p[6] = (bubbleScale m, bubbleRate pops/s per cell, 0, 0)` and
// `p[7] = (foamAmount, detailStrength, detailScale, 0)`.
//
// Wetness darkens the base colour and adds a sharp clear coat (roughness 0.08) over a
// still rough body (0.7). Slow voronoi bubbles on the XZ plane grow a dome (a normal bump
// plus a glossy highlight) and pop into an expanding ring. Foam is the crest term and
// noise only: an opaque draw runs before the opaque copy of its frame, so the opaque depth
// that `liquid_shore` samples is not written yet; the mud foam marks the moving crests and
// never reads that copy.

// Radial bubble contribution: the XZ gradient of its height field (a slope, dimensionless,
// because both the height and the cell coordinate scale with bubbleScale) and the glossy
// skin coverage in [0, 1].
struct MudBubble {
    gradient: vec2<f32>,
    gloss: f32,
}

// Offset cell of the nearest feature point of the raa_voronoi3 lattice. raa_voronoi3
// returns only the (F1, F2) distances, so the owning cell is recovered with the same
// 27-cell search: the bubble phase follows the cell that draws the dome, so two phases
// never meet inside one bubble at a cell boundary.
fn mud_voronoi_cell(p: vec3<f32>) -> vec3<f32> {
    let i = floor(p);
    let f = p - i;
    var best = 8.0;
    var cell = vec3<f32>(0.0);
    for (var z = -1; z <= 1; z = z + 1) {
        for (var y = -1; y <= 1; y = y + 1) {
            for (var x = -1; x <= 1; x = x + 1) {
                let o = vec3<f32>(f32(x), f32(y), f32(z));
                let r = o + raa_hash3(i + o) - f;
                let dist = length(r);
                if (dist < best) {
                    best = dist;
                    cell = i + o;
                }
            }
        }
    }
    return cell;
}

// One bubble in cell space: `d` is the raa_voronoi3 F1 distance to the feature point of
// `cell` and `life` in [0, 1) is the per-cell phase advanced by the pops per second. The
// first 72% of the cycle is the dome (it grows, holds and fades); the rest is the pop, a
// ring expanding out of the feature point while it thins and dies.
fn mud_bubble(xz: vec2<f32>, cell: vec3<f32>, d: f32, life: f32) -> MudBubble {
    var b: MudBubble;
    b.gradient = vec2<f32>(0.0);
    b.gloss = 0.0;
    if (!(d >= 0.0) || !(life >= 0.0)) {
        return b;
    }
    let feature = cell + raa_hash3(cell);
    let delta = xz - feature.xz;
    // The gradient of a radial height h(d) is h'(d) * delta / d in the XZ plane: `d` is
    // the 3D distance and `delta` the XZ offset from the feature point.
    let denom = max(d, 1e-4);
    let u = clamp(life / 0.72, 0.0, 1.0);
    if (life < 0.72) {
        let radius = mix(0.06, 0.34, sqrt(u));
        let amp = 0.16 * smoothstep(0.0, 0.12, life) * (1.0 - smoothstep(0.60, 0.72, life));
        let x = d / max(radius, 1e-4);
        if (x < 1.0) {
            let profile = 1.0 - x * x;
            b.gradient = delta / denom * (-2.0 * amp * x / max(radius, 1e-4));
            b.gloss = profile;
        }
    } else {
        let pop = clamp((life - 0.72) / 0.28, 0.0, 1.0);
        let radius = mix(0.10, 0.46, pop);
        let width = mix(0.05, 0.09, pop);
        let amp = 0.12 * (1.0 - pop);
        let w = (d - radius) / max(width, 1e-4);
        let ring = exp(-w * w);
        b.gradient = delta / denom * (amp * ring * (-2.0 * w / max(width, 1e-4)));
        b.gloss = ring * (1.0 - pop);
    }
    return b;
}

// Bubbles of the mud at `world_xz` (metres), `scale` metres per cell and `rate` pops per
// second per cell.
fn mud_bubbles(world_xz: vec2<f32>, t: f32, scale: f32, rate: f32) -> MudBubble {
    let xz = world_xz / max(scale, 1e-3);
    let p = vec3<f32>(xz.x, 0.0, xz.y);
    let cell = mud_voronoi_cell(p);
    let d = raa_voronoi3(p).x;
    let phase = raa_hash3(cell + vec3<f32>(31.0, 17.0, 11.0)).x;
    let life = fract(t * max(rate, 0.0) + phase);
    return mud_bubble(xz, cell, d, life);
}

// Tilt the shading normal by the bubble slope, through the tangent frame.
fn mud_bump_normal(n: vec3<f32>, tangent: vec4<f32>, gradient: vec2<f32>) -> vec3<f32> {
    let nn = normalize(n);
    let g = clamp(gradient * 0.5, vec2<f32>(-1.5, -1.5), vec2<f32>(1.5, 1.5));
    return raa_perturb_normal(nn, tangent, vec3<f32>(-g.x, -g.y, 1.0));
}

// Crest foam without the shore term. `liquid_shore` samples the opaque depth copy, but an
// opaque draw runs before the opaque copy of its frame is written, so the copy is still
// the dummy there; mud foam is the moving crest term broken up by noise only.
fn mud_foam(crest: f32, world_pos: vec3<f32>, t: f32, amount: f32) -> f32 {
    let drift = vec3<f32>(0.0, 0.35 * t, 0.21 * t);
    let breakup = clamp(0.5 + 0.5 * raa_noise3(world_pos * 2.7 + drift), 0.0, 1.0);
    return clamp(crest * breakup, 0.0, 1.0) * clamp(amount, 0.0, 1.0);
}

fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    let color = max(in.record.p[5].rgb, vec3<f32>(0.0));
    let wetness = clamp(in.record.p[5].w, 0.0, 1.0);
    let bubble_scale = max(in.record.p[6].x, 1e-3);
    let bubble_rate = max(in.record.p[6].y, 0.0);
    let foam_amount = clamp(in.record.p[7].x, 0.0, 1.0);
    let detail_strength = clamp(in.record.p[7].y, 0.0, 1.0);
    let detail_scale = max(in.record.p[7].z, 1e-3);

    o.metallic = 0.0;
    o.roughness = 0.7;
    o.base_color = color * (1.0 - 0.45 * wetness);
    o.clearcoat = wetness;
    o.clearcoat_roughness = 0.08;

    o.normal = liquid_detail_normal(
        o.normal,
        in.tangent,
        in.position,
        in.time,
        detail_strength,
        detail_scale,
    );

    let bubbles = mud_bubbles(in.local_position.xz, in.time, bubble_scale, bubble_rate);
    o.normal = mud_bump_normal(o.normal, in.tangent, bubbles.gradient);
    o.roughness = mix(o.roughness, 0.09, clamp(bubbles.gloss, 0.0, 1.0));

    let crest = liquid_waves_at(in.record.p, in.local_position.xz, in.time);
    let foam = mud_foam(crest, in.position, in.time, foam_amount);
    let pale = vec3<f32>(0.42, 0.35, 0.26);
    o.base_color = mix(o.base_color, pale, foam);
    o.roughness = mix(o.roughness, 0.9, foam);
    return o;
}
