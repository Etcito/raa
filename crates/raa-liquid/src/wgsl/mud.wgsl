// raa.liquid.mud: wet, slow, opaque mud over the shared Gerstner vertex stage.
//
// The record packs the waves and speeds in `p[0..5]` (src/waves.rs), then
// `p[5] = (color linear rgb, wetness)`,
// `p[6] = (bubbleScale m, bubbleRate pops/s per cell, 0, foamWidth m)` and
// `p[7] = (foamAmount, detailStrength, detailScale, 0)`.
//
// The body is an opaque matte dielectric (roughness 0.5 thin .. 0.62 deep) under a thin
// near-mirror wet coat (0.42 x wetness, coat roughness 0.03..0.05): only the coat reflects
// sharply, so the sky is a Fresnel sheen at grazing angles and the sun tight streaks along
// the swell, and three slow directional ripples in the normal break those streaks up. A
// glossy body under the coat doubled the blue sky reflection over the brown, which AgX
// shows as mauve or lilac (card 4940: pit HDR about (0.22, 0.12, 0.14); card 5070); a rough
// coat read matte (card 5037) and a flat 3 m swell rubbery (card 5051). Depth cues: the deep mud is darker
// and more saturated, the thin mud at the shore and the pads lighter. Slow voronoi bubbles on
// the XZ plane grow a dome (a normal bump plus a glossy coat) and pop into an expanding ring.
//
// Foam: an opaque draw runs before the opaque copy of its frame, so the opaque depth that
// `liquid_shore` samples is not written yet. The shore term instead reads the mud depth
// above the ground from the vertex colour alpha (metres, clamped to [0, 1]; the default
// alpha 1 means deep, no shore), which the mesh author computes from the ground under the
// surface. The crest term adds froth on the steepest crests; no noise mask scatters foam
// inside the pit. The foam is froth: soft round voronoi bubbles (nearest-feature domes of
// two lattices, no cell edges) in light to mid mud brown.

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
// second per cell. About half of the cells carry a bubble: a dome on every cell covered
// the whole surface in a lumpy carpet (card 4940), a third read as too sparse (card 5070).
fn mud_bubbles(world_xz: vec2<f32>, t: f32, scale: f32, rate: f32) -> MudBubble {
    var b: MudBubble;
    b.gradient = vec2<f32>(0.0);
    b.gloss = 0.0;
    let xz = world_xz / max(scale, 1e-3);
    let p = vec3<f32>(xz.x, 0.0, xz.y);
    let cell = mud_voronoi_cell(p);
    let present = step(0.5, raa_hash3(cell + vec3<f32>(5.0, 41.0, 23.0)).x);
    if (present < 0.5) {
        return b;
    }
    let d = raa_voronoi3(p).x;
    let phase = raa_hash3(cell + vec3<f32>(31.0, 17.0, 11.0)).x;
    let life = fract(t * max(rate, 0.0) + phase);
    return mud_bubble(xz, cell, d, life);
}

// Tilt the shading normal by the bubble slope, through the tangent frame.
fn mud_bump_normal(n: vec3<f32>, tangent: vec4<f32>, gradient: vec2<f32>) -> vec3<f32> {
    let nn = normalize(n);
    // A bubble is a low dome, not a lens: a gentle slope keeps the clear coat a coherent
    // sky reflection and never turns the normal on its side. The gain is high enough that
    // the dome catches its own highlight at capture distance.
    let g = clamp(gradient * 0.6, vec2<f32>(-0.9, -0.9), vec2<f32>(0.9, 0.9));
    return raa_perturb_normal(nn, tangent, vec3<f32>(-g.x, -g.y, 1.0));
}

// Secondary ripples of the swell: three slow directional sines (0.2..0.5 m) whose slope
// tilts the shading normal, each faded in and out by a broad drifting noise so the sun
// highlight breaks into moving streaks instead of one smooth lobe. Returns the XZ slope.
fn mud_ripples(world_pos: vec3<f32>, t: f32) -> vec2<f32> {
    let xz = world_pos.xz;
    let d0 = vec2<f32>(0.94, 0.34);
    let d1 = vec2<f32>(-0.42, 0.91);
    let d2 = vec2<f32>(0.62, -0.78);
    let k0 = LIQUID_TAU / 0.47;
    let k1 = LIQUID_TAU / 0.31;
    let k2 = LIQUID_TAU / 0.21;
    let m0 = 0.5 + 0.5 * raa_noise3(vec3<f32>(xz * 0.9, 0.05 * t));
    let m1 = 0.5 + 0.5 * raa_noise3(vec3<f32>(xz * 1.3 + vec2<f32>(7.1, 3.3), 0.07 * t));
    let m2 = 0.5 + 0.5 * raa_noise3(vec3<f32>(xz * 1.7 + vec2<f32>(2.9, 9.4), 0.09 * t));
    let s0 = 0.12 * m0 * cos(k0 * (dot(d0, xz) - 0.12 * t));
    let s1 = 0.09 * m1 * cos(k1 * (dot(d1, xz) - 0.09 * t));
    let s2 = 0.06 * m2 * cos(k2 * (dot(d2, xz) - 0.07 * t));
    return d0 * s0 + d1 * s1 + d2 * s2;
}

// Foam mask in [0, 1]: the shore band where the mud is shallower than `width` metres (the
// vertex colour alpha; the pads' rings ride in it too), with a soft frayed outer edge, plus
// `amount` times the crest term on the swell.
fn mud_foam(depth: f32, crest: f32, world_pos: vec3<f32>, t: f32, width: f32, amount: f32) -> f32 {
    let drift = vec3<f32>(0.0, 0.2 * t, 0.12 * t);
    let n = raa_noise3(world_pos * 3.0 + drift);
    // The band edge wanders by about half its width: a soft, irregular froth line.
    let w = max(width, 1e-4) * mix(0.6, 1.4, 0.5 + 0.5 * n);
    let band = 1.0 - smoothstep(0.15 * w, w, max(depth, 0.0));
    let caps = smoothstep(0.1, 0.3, crest) * clamp(amount, 0.0, 1.0)
        * smoothstep(-0.2, 0.3, raa_noise3(world_pos * 2.2 - drift));
    return clamp(max(band, caps), 0.0, 1.0);
}

// Rounded froth bubbles: the nearest-feature distance of a voronoi lattice turned into a
// soft dome per cell (1 at the feature point, 0 beyond 0.5 cell). Two lattices of
// different sizes overlap, so the froth is a heap of round bubbles with soft edges and no
// cell seams (the F2 - F1 edge pattern it replaces read as crackle).
fn mud_froth_height(xz: vec2<f32>) -> f32 {
    let a = raa_voronoi3(vec3<f32>(xz.x, 0.0, xz.y) / 0.05).x;
    let b = raa_voronoi3(vec3<f32>(xz.x + 3.7, 0.5, xz.y - 1.9) / 0.026).x;
    let ha = sqrt(max(1.0 - (a / 0.5) * (a / 0.5), 0.0));
    let hb = sqrt(max(1.0 - (b / 0.45) * (b / 0.45), 0.0));
    return max(ha, 0.8 * hb);
}

// Froth at `world_pos`: (slope x, slope z, height) of the bubble heap, the slope capped so
// each bubble catches a glint without turning on its side.
fn mud_froth(world_pos: vec3<f32>, t: f32) -> vec3<f32> {
    let xz = world_pos.xz + vec2<f32>(0.0, 0.015 * t);
    let h = mud_froth_height(xz);
    let e = 0.004;
    let hx = mud_froth_height(xz + vec2<f32>(e, 0.0)) - h;
    let hz = mud_froth_height(xz + vec2<f32>(0.0, e)) - h;
    let slope = vec2<f32>(hx, hz) / e * 0.008;
    let m = length(slope);
    let limited = slope * select(1.0, 0.6 / max(m, 1e-6), m > 0.6);
    return vec3<f32>(limited, h);
}

// More saturated colour: `amount` > 1 pushes the channels away from their luma.
fn mud_saturate(c: vec3<f32>, amount: f32) -> vec3<f32> {
    let l = raa_luma(c);
    return max(vec3<f32>(l) + (c - vec3<f32>(l)) * amount, vec3<f32>(0.0));
}

fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    let color = max(in.record.p[5].rgb, vec3<f32>(0.0));
    let wetness = clamp(in.record.p[5].w, 0.0, 1.0);
    let bubble_scale = max(in.record.p[6].x, 1e-3);
    let bubble_rate = max(in.record.p[6].y, 0.0);
    let foam_width = max(in.record.p[6].w, 0.0);
    let foam_amount = clamp(in.record.p[7].x, 0.0, 1.0);
    let detail_strength = clamp(in.record.p[7].y, 0.0, 1.0);
    let detail_scale = max(in.record.p[7].z, 1e-3);
    // Mud depth above the ground in metres (vertex colour alpha, see the header).
    let depth = clamp(in.color.a, 0.0, 1.0);
    let deep = smoothstep(0.03, 0.28, depth);

    o.metallic = 0.0;
    o.opacity = 1.0;
    // Depth cues: deep mud is darker and more saturated, the thin mud at the shore and at
    // the pads lighter and duller; one broad, faint patch layer keeps it from a flat plane.
    let patches = raa_noise3(in.position * 0.7 + vec3<f32>(0.0, 0.0, 0.15 * in.time));
    // No saturation boost: AgX punchy already saturates, and a boosted deep colour read as
    // orange-rust instead of chocolate (card 5056).
    let deep_color = color * 0.78;
    let thin_color = mud_saturate(color * 1.3, 0.9);
    o.base_color = mix(thin_color, deep_color, deep) * (0.94 + 0.08 * patches);
    // An opaque matte body under a thin wet coat: the body itself is rough (its own
    // dielectric reflection is a dull blur), so the only sharp reflection is the coat's,
    // whose Fresnel keeps the sky to a faint sheen at grazing angles and the sun to tight
    // streaks along the swell. A glossy body (roughness 0.22..0.38) under a 0.55 coat
    // mirrored the blue sky over the whole pit at the capture angle and the brown read
    // lilac (card 5070 capture: pit sRGB about (127, 66, 62), blue as high as green).
    o.roughness = mix(0.5, 0.62, deep);
    o.clearcoat = 0.42 * clamp(wetness, 0.0, 1.0);
    o.clearcoat_roughness = mix(0.03, 0.05, deep);

    o.normal = liquid_detail_normal(
        o.normal,
        in.tangent,
        in.position,
        in.time,
        detail_strength,
        detail_scale,
    );
    let ripple = mud_ripples(in.position, in.time);
    o.normal = raa_perturb_normal(o.normal, in.tangent, normalize(vec3<f32>(-ripple.x, -ripple.y, 1.0)));

    let bubbles = mud_bubbles(in.local_position.xz, in.time, bubble_scale, bubble_rate);
    let gloss = clamp(bubbles.gloss, 0.0, 1.0);
    o.normal = mud_bump_normal(o.normal, in.tangent, bubbles.gradient);
    // A bubble skin is smoother and a little lighter than the pit around it, so the domes
    // and pop rings read as glossy bubbles; the colour follows the mud.
    o.roughness = mix(o.roughness, 0.15, gloss);
    o.clearcoat = mix(o.clearcoat, 0.8, gloss);
    o.clearcoat_roughness = mix(o.clearcoat_roughness, 0.015, gloss);
    o.base_color = mix(o.base_color, color * 1.6, 0.55 * gloss);

    // Froth: a creamy mass of small soft round bubbles in tan to cream, concentrated at the
    // shore, in rings around the pads and on crests. Bubbles appear one by one as the mask
    // rises, so the froth edge is bubbly rather than a hard line; inside the mass the
    // bubbles only shade it softly (white pebble-like bubbles on dark mud read as gravel).
    let crest = liquid_waves_at(in.record.p, in.local_position.xz, in.time);
    let mask = mud_foam(depth, crest, in.position, in.time, foam_width, foam_amount);
    let froth = mud_froth(in.position, in.time);
    let foam = smoothstep(0.0, 0.45, froth.z + 0.25 - (1.0 - mask)) * smoothstep(0.0, 0.3, mask);
    // Churned brown froth: the mud's own hue, a little lighter and duller towards the
    // shore line, each bubble lighter at its crown and darker in the gaps between bubbles
    // (cream froth read as pale foam, not churned mud: card 5056).
    let froth_brown = mud_saturate(color * 1.35, 0.85);
    let froth_light = mud_saturate(color * 1.75, 0.75);
    let froth_color = mix(froth_brown, froth_light, smoothstep(0.4, 1.0, mask)) * mix(0.7, 1.08, froth.z);
    o.base_color = mix(o.base_color, froth_color, foam);
    o.normal = raa_perturb_normal(
        o.normal,
        in.tangent,
        normalize(vec3<f32>(-froth.x * foam, -froth.y * foam, 1.0)),
    );
    // Froth is wet but broken up: a softer coat, so it never mirrors the blue sky (a strong
    // coat on the froth read as grey gravel).
    o.roughness = mix(o.roughness, 0.4, foam);
    o.clearcoat = mix(o.clearcoat, 0.45, foam);
    o.clearcoat_roughness = mix(o.clearcoat_roughness, 0.06, foam);
    return o;
}
