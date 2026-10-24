// raa.liquid.mud: wet, slow, opaque mud over the shared Gerstner vertex stage.
//
// The record packs the waves and speeds in `p[0..5]` (the water layout at 0.4x amplitude
// and 0.08x speed, src/waves.rs), then `p[5] = (color linear rgb, wetness)`,
// `p[6] = (bubbleScale m, bubbleRate pops/s per cell, 0, foamWidth m)` and
// `p[7] = (foamAmount, detailStrength, detailScale, 0)`.
//
// The body is a wet dielectric (roughness 0.2, F0 0.04) under a wet coat (0.6 x wetness,
// coat roughness 0.05): the sky is a visible reflection on the brown, the sun a tight wet
// highlight. Card 5037 judged the rougher body (0.35, coat 0.35 x wetness) matte.
// A mirror coat over the dark body let the blue sky radiance outweigh the brown diffuse
// (measured on the card 4940 capture: pit HDR about (0.22, 0.12, 0.14), blue above green,
// which AgX shows as mauve). Two noise scales mottle the brown. Slow voronoi bubbles on the
// XZ plane grow a dome (a normal bump plus a glossy coat) and pop into an expanding ring.
//
// Foam: an opaque draw runs before the opaque copy of its frame, so the opaque depth that
// `liquid_shore` samples is not written yet. The shore term instead reads the mud depth
// above the ground from the vertex colour alpha (metres, clamped to [0, 1]; the default
// alpha 1 means deep, no shore), which the mesh author computes from the ground under the
// surface. The crest term adds churned streaks on the steepest crests; no noise mask
// scatters foam inside the pit.

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
// second per cell. Only about a third of the cells carry a bubble: a dome on every cell
// covered the whole surface in a lumpy carpet instead of scattered bubbles (card 4940).
fn mud_bubbles(world_xz: vec2<f32>, t: f32, scale: f32, rate: f32) -> MudBubble {
    var b: MudBubble;
    b.gradient = vec2<f32>(0.0);
    b.gloss = 0.0;
    let xz = world_xz / max(scale, 1e-3);
    let p = vec3<f32>(xz.x, 0.0, xz.y);
    let cell = mud_voronoi_cell(p);
    let present = step(0.68, raa_hash3(cell + vec3<f32>(5.0, 41.0, 23.0)).x);
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

// Foam in [0, 1]: the shore band where the mud is shallower than `width` metres (from the
// vertex colour alpha), frayed by a coarse noise along the band only, plus `amount` times
// the steep-crest term.
fn mud_foam(depth: f32, crest: f32, world_pos: vec3<f32>, t: f32, width: f32, amount: f32) -> f32 {
    let shore = 1.0 - smoothstep(0.0, max(width, 1e-4), max(depth, 0.0));
    let drift = vec3<f32>(0.0, 0.2 * t, 0.12 * t);
    let n = raa_noise3(world_pos * 5.0 + drift);
    // The fray thins the outer edge of the band; it never removes the contact line.
    let fray = mix(0.55, 1.0, smoothstep(-0.3, 0.4, n));
    let band = smoothstep(0.0, 1.0, shore * 1.3) * fray;
    let caps = smoothstep(0.18, 0.34, crest) * clamp(amount, 0.0, 1.0);
    return clamp(max(band, caps), 0.0, 1.0);
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

    o.metallic = 0.0;
    o.opacity = 1.0;
    o.roughness = 0.2;
    // Wet mud is darker than dry ground but still reads brown under the clear coat; two
    // noise scales give it the mottled, grainy body of mud instead of a flat coloured
    // plane (card 5018 read the old smooth surface as a pink liquid).
    let grain = raa_fbm3(in.position * 2.4 + vec3<f32>(0.0, 0.0, 0.2 * in.time), 3);
    let patches = raa_noise3(in.position * 0.7 + vec3<f32>(0.0, 0.0, 0.15 * in.time));
    o.base_color = color * (1.0 - 0.15 * wetness) * (0.88 + 0.12 * grain + 0.08 * patches);
    // A wet coat: its sun lobe gives the glossy highlights and the sky shows as a sheen
    // over the brown; MUD_LIGHT's low env specular keeps it from a blue wash (see header).
    o.clearcoat = clamp(0.6 * wetness, 0.0, 1.0);
    o.clearcoat_roughness = 0.05;

    o.normal = liquid_detail_normal(
        o.normal,
        in.tangent,
        in.position,
        in.time,
        detail_strength,
        detail_scale,
    );

    let bubbles = mud_bubbles(in.local_position.xz, in.time, bubble_scale, bubble_rate);
    let gloss = clamp(bubbles.gloss, 0.0, 1.0);
    o.normal = mud_bump_normal(o.normal, in.tangent, bubbles.gradient);
    // The bubble skin is smoother than the body: a tight clear coat highlight on the dome
    // sells the wet bubble, while the body stays a little darker underneath.
    o.roughness = mix(o.roughness, 0.1, gloss);
    o.clearcoat = mix(o.clearcoat, 0.9, 0.8 * gloss);
    o.clearcoat_roughness = mix(o.clearcoat_roughness, 0.02, gloss);
    // A raised bubble is wetter and slightly lighter than the pit around it; it follows the
    // mud colour (a fixed light brown read as pale spots on a dark pit, card 5027) so the
    // surface never drifts to the sky's blue.
    o.base_color = mix(o.base_color, color * 1.5, 0.5 * gloss);

    let crest = liquid_waves_at(in.record.p, in.local_position.xz, in.time);
    let foam = mud_foam(in.color.a, crest, in.position, in.time, foam_width, foam_amount);
    // Churned mud: streaks of lighter and darker saturated brown, still wet and glossy. A
    // pale cream foam read as a tan band around the pit and the pads (card 5037), not as
    // churned mud; the colour stays a multiple of the mud colour so the hue never drifts.
    let churn = raa_fbm3(in.position * vec3<f32>(9.0, 3.0, 9.0) + vec3<f32>(0.0, 0.5 * in.time, 0.0), 3);
    let churned = color * mix(0.6, 1.6, smoothstep(-0.35, 0.35, churn));
    o.base_color = mix(o.base_color, churned, foam);
    // Churned lumps: the churn field also tilts the shading normal inside the band, so the
    // shore reads as rolled, glossy mud rather than a flat stain (card 4941 judge saw no
    // churned mud at the shore).
    let ce = 0.03;
    let cp = in.position * vec3<f32>(9.0, 3.0, 9.0) + vec3<f32>(0.0, 0.5 * in.time, 0.0);
    let cx = raa_fbm3(cp + vec3<f32>(9.0 * ce, 0.0, 0.0), 3) - churn;
    let cz = raa_fbm3(cp + vec3<f32>(0.0, 0.0, 9.0 * ce), 3) - churn;
    let lump = clamp(vec2<f32>(cx, cz) / ce * 0.05 * foam, vec2<f32>(-0.8), vec2<f32>(0.8));
    o.normal = raa_perturb_normal(o.normal, in.tangent, normalize(vec3<f32>(-lump.x, -lump.y, 1.0)));
    o.roughness = mix(o.roughness, 0.3, foam);
    o.clearcoat = o.clearcoat * (1.0 - 0.4 * foam);
    return o;
}
