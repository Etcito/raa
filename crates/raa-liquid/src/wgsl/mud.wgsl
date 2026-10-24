// raa.liquid.mud: wet, slow, opaque mud over the shared Gerstner vertex stage.
//
// The record packs the waves and speeds in `p[0..5]` (src/waves.rs), then
// `p[5] = (color linear rgb, wetness)`,
// `p[6] = (bubbleScale m, bubbleRate pops/s per cell, 0, foamWidth m)` and
// `p[7] = (foamAmount, detailStrength, detailScale, 0)`.
//
// The body is an opaque semi-glossy dielectric (roughness 0.32 thin .. 0.4 deep) under a
// broad wet coat (0.55 x wetness, coat roughness 0.05..0.09): the coat reflects the sky as
// a Fresnel sheen and the sun as streaks along the swell. A low-frequency fbm height field
// tilts the normal (crests and troughs) and shades the crests lighter, so the slow Gerstner
// swell reads as broad bands under the sun. A sharp coat over a bright body mirrored the
// blue sky as mauve or lilac (cards 4940/5070); the body is now a deep chocolate so the
// sheen stays a highlight over brown. Slow voronoi bubbles on the XZ plane grow a dome (a
// normal bump plus a glossy coat) and pop into an expanding ring; the scene adds a few
// explicit glossy bubble meshes where the field is too sparse to read.
//
// Foam: an opaque draw runs before the opaque copy of its frame, so the opaque depth that
// `liquid_shore` samples is not written yet. The shore term instead reads the mud depth
// above the ground from the vertex colour alpha (metres, clamped to [0, 1]; the default
// alpha 1 means deep, no shore), which the mesh author computes from the ground under the
// surface. The crest term adds froth on the steepest crests. The foam is a soft off-white
// band whose width wanders with large noise and whose outer edge is frayed into the mud by
// fine noise (no per-cell domes: card 5078 read the froth as pebble-like dots).

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
// second per cell. About one cell in six carries a bubble: half of the cells, together with
// the froth, read as a pebble carpet (card 5072 judge: "bumpy lumpy surface").
fn mud_bubbles(world_xz: vec2<f32>, t: f32, scale: f32, rate: f32) -> MudBubble {
    var b: MudBubble;
    b.gradient = vec2<f32>(0.0);
    b.gloss = 0.0;
    let xz = world_xz / max(scale, 1e-3);
    let p = vec3<f32>(xz.x, 0.0, xz.y);
    let cell = mud_voronoi_cell(p);
    let present = step(0.83, raa_hash3(cell + vec3<f32>(5.0, 41.0, 23.0)).x);
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

// Low-frequency swell height: two octaves of gradient noise at `scale` per metre plus a
// slow drift, so the whole pit undulates instead of lying flat (card 5078: "no visible
// swell"). The vertex stage adds the long Gerstner waves on top of this.
fn mud_swell_height(xz: vec2<f32>, t: f32, scale: f32) -> f32 {
    let s = max(scale, 1e-3);
    let p = vec3<f32>(xz.x * s, 0.0, xz.y * s);
    let drift = vec3<f32>(0.15, 0.0, 0.09) * t;
    return raa_fbm3(p + drift, 3) + 0.35 * raa_noise3(p.yzx * 2.1 - drift * 1.7);
}

// XZ slope of the swell height: the low-frequency normal noise that breaks the sun
// highlight into crests and troughs. `strength` scales the tilt.
fn mud_swell_slope(xz: vec2<f32>, t: f32, scale: f32, strength: f32) -> vec2<f32> {
    let e = max(0.05, 0.1 / max(scale, 1e-3));
    let h = mud_swell_height(xz, t, scale);
    let hx = mud_swell_height(xz + vec2<f32>(e, 0.0), t, scale);
    let hz = mud_swell_height(xz + vec2<f32>(0.0, e), t, scale);
    // One long directional band on top of the noise: the wet coat stretches the sky
    // reflection into broad moving bands along the swell.
    let d0 = vec2<f32>(0.94, 0.34);
    let band = 0.25 * cos(LIQUID_TAU / 1.9 * (dot(d0, xz) - 0.1 * t));
    return (vec2<f32>(hx - h, hz - h) / e + d0 * band) * clamp(strength, 0.0, 1.0);
}

// Foam mask in [0, 1]: the shore band where the mud is shallower than `width` metres (the
// vertex colour alpha; the pads' rings ride in it too), its width wandering with large
// noise lumps and its outer edge frayed into the mud by fine noise, plus `amount` times the
// crest term on the swell. A soft band, not the voronoi domes that read as pebbles
// (card 5078).
fn mud_foam(depth: f32, crest: f32, world_pos: vec3<f32>, t: f32, width: f32, amount: f32) -> f32 {
    let xz = world_pos.xz;
    let drift = vec3<f32>(0.0, 0.15 * t, 0.09 * t);
    let lumps = raa_fbm3(vec3<f32>(xz.x * 1.3, 0.0, xz.y * 1.3) + drift, 3);
    let fine = raa_noise3(vec3<f32>(xz.x * 11.0, 1.7, xz.y * 11.0) - drift * 1.4);
    let w = max(width, 1e-4) * (0.55 + 0.9 * (0.5 + 0.5 * lumps));
    let d = max(depth, 0.0);
    let band = 1.0 - smoothstep(0.0, w, d);
    let lace = smoothstep(-0.35, 0.45, 0.7 * fine + 0.5 * lumps);
    // The inner band is solid; only its outer edge thins into the mud.
    let frayed = band * mix(1.0, lace, smoothstep(0.1 * w, w, d));
    let caps = smoothstep(0.12, 0.4, crest) * clamp(amount, 0.0, 1.0)
        * smoothstep(-0.15, 0.45, raa_noise3(vec3<f32>(xz.x * 2.4, 0.4, xz.y * 2.4) - drift));
    return clamp(max(frayed, caps * lace), 0.0, 1.0);
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
    // Relief: the vertex stage supplies the slow Gerstner swell geometry; the fragment adds
    // a low-frequency fbm slope (the sun highlight breaks into crests and troughs) and a
    // finer drifting skin. Card 5078's flat plane came from a gentle swell under an opaque
    // body with almost no normal variation.
    let slope = mud_swell_slope(in.position.xz, in.time, detail_scale, detail_strength * 1.6);
    let low = mud_swell_height(in.position.xz, in.time, detail_scale);
    o.normal = raa_perturb_normal(o.normal, in.tangent, normalize(vec3<f32>(-slope.x, -slope.y, 1.0)));
    o.normal = liquid_detail_normal(
        o.normal,
        in.tangent,
        in.position,
        in.time,
        0.5 * detail_strength,
        detail_scale * 5.0,
    );

    // Depth cues: deep mud is a little darker, the thin mud at the shore and at the pads a
    // little lighter; one broad, faint patch layer keeps it from a flat plane. No
    // saturation boost: AgX punchy already saturates, and a boosted deep colour read as
    // orange-rust instead of chocolate (card 5056). The swell shades crests lighter and
    // troughs darker, together with the low-frequency height field.
    let patches = raa_noise3(in.position * 0.7 + vec3<f32>(0.0, 0.0, 0.15 * in.time));
    let swell = smoothstep(-0.16, 0.16, in.local_position.y);
    let deep_color = color * 0.82;
    let thin_color = color * 1.06;
    o.base_color = mix(thin_color, deep_color, deep) * (0.92 + 0.08 * patches)
        * mix(0.7, 1.24, swell) * mix(0.84, 1.14, 0.5 + 0.5 * low);
    // An opaque body under a wet coat. The body is semi-glossy (roughness 0.32..0.4): its
    // dielectric lobe adds a soft sheen, while the coat gives the sky reflection and sun
    // streaks along the swell. The coat is broad enough (roughness 0.05..0.09) that the
    // blue sky reads as a sheen over the chocolate rather than as a mirror skin (cards
    // 5070/5077 rendered the old sharper coat lilac and maroon).
    o.roughness = mix(0.32, 0.4, deep);
    o.clearcoat = 0.55 * wetness;
    o.clearcoat_roughness = mix(0.05, 0.09, deep);

    let bubbles = mud_bubbles(in.local_position.xz, in.time, bubble_scale, bubble_rate);
    let gloss = clamp(bubbles.gloss, 0.0, 1.0);
    o.normal = mud_bump_normal(o.normal, in.tangent, bubbles.gradient);
    // A bubble skin is smoother and lighter than the pit around it, so the rare large domes
    // and pop rings read as glossy bubbles.
    o.roughness = mix(o.roughness, 0.08, gloss);
    o.clearcoat = mix(o.clearcoat, 0.9, gloss);
    o.clearcoat_roughness = mix(o.clearcoat_roughness, 0.02, gloss);
    o.base_color = mix(o.base_color, mix(color * 0.9, color * 1.55 + vec3<f32>(0.01), 0.5), 0.65 * gloss);

    // Froth: a soft off-white band along the shore line and in a ring around each pad
    // (where the mud is shallower than foamWidth), plus a little on the steepest crests. It
    // tapers into the mud through a noise-frayed edge; card 5078's froth used per-cell
    // voronoi domes and read as pebble-like dots.
    let crest = liquid_waves_at(in.record.p, in.local_position.xz, in.time);
    let foam = mud_foam(depth, crest, in.position, in.time, foam_width, foam_amount);
    // Off-white froth speckled from white to cream, with a fine bubble-scale normal: a soft
    // mat that feathers into the mud, not a flat painted stripe.
    let speckle = 0.5 + 0.5 * raa_noise3(in.position * 17.0);
    // Churned froth: cream foam cells (about 7 cm) separated by soft seams of light brown
    // mud, so the shore reads as frothy brown mud rather than a painted white stripe.
    let fc = in.position.xz * 14.0;
    let cells = raa_voronoi3(vec3<f32>(fc.x, 0.25 * in.time, fc.y));
    let seam = smoothstep(0.0, 0.3, cells.y - cells.x);
    let cream = mix(vec3<f32>(0.8, 0.74, 0.64), vec3<f32>(0.93, 0.89, 0.82), speckle);
    let froth = mix(mix(color * 2.4, cream, 0.35), cream, seam);
    o.base_color = mix(o.base_color, froth, foam);
    let fh0 = raa_noise3(vec3<f32>(in.position.x * 42.0, 1.3, in.position.z * 42.0));
    let fhx = raa_noise3(vec3<f32>((in.position.x + 0.008) * 42.0, 1.3, in.position.z * 42.0));
    let fhz = raa_noise3(vec3<f32>(in.position.x * 42.0, 1.3, (in.position.z + 0.008) * 42.0));
    let fg = clamp(vec2<f32>(fhx - fh0, fhz - fh0) * 2.0 * foam, vec2<f32>(-0.5, -0.5), vec2<f32>(0.5, 0.5));
    o.normal = raa_perturb_normal(o.normal, in.tangent, normalize(vec3<f32>(-fg.x, -fg.y, 1.0)));
    // Foam is rough and opaque with a thin wet skin.
    o.roughness = mix(o.roughness, 0.7, foam);
    o.clearcoat = mix(o.clearcoat, 0.3, foam);
    o.clearcoat_roughness = mix(o.clearcoat_roughness, 0.14, foam);
    return o;
}
