// Fragment helpers shared by the liquid material functions. Every identifier starts with
// `liquid_` and none uses the engine's `raa_` prefix; the helpers are called by the
// per-liquid `material_main` and use only the S11 prelude and the raa-shader LIB helpers.
//
// They live next to the Gerstner vertex stage of liquid_vertex.wgsl (same module), so the
// crest term recomputed here is exactly the steepness the vertex stage displaced.

// View distance of the fragment at `depth`, reconstructed through the inverse projection.
// `frag.xy` is the pixel coordinate, so the NDC x/y is restored with the frame viewport.
fn liquid_view_distance(frag: vec4<f32>, depth: f32) -> f32 {
    let ndc = vec2<f32>(
        frag.x * raa_frame.viewport.z * 2.0 - 1.0,
        1.0 - frag.y * raa_frame.viewport.w * 2.0,
    );
    let view = raa_frame.inv_proj * vec4<f32>(ndc, depth, 1.0);
    let w = select(view.w, 1.0, abs(view.w) < 1e-6);
    return -view.z / w;
}

// Crest term of the wave sum at `xz`: `1 - J` with the Jacobian `J = 1 - sum Qi k A sin`
// (steepness). It is 0 on flat water and approaches 1 at sharp crests.
fn liquid_waves_at(p: array<vec4<f32>, 8>, xz: vec2<f32>, t: f32) -> f32 {
    let total = liquid_wave_total(p, xz, t);
    let jacobian = 1.0 - total.steepness;
    return clamp(1.0 - jacobian, 0.0, 1.0);
}

// Two scrolling fbm layers moving in different directions, as a height in about [-1, 1].
fn liquid_detail_height(world_pos: vec3<f32>, t: f32, scale: f32) -> f32 {
    let s = max(scale, 1e-3);
    let p = world_pos * s;
    let drift_a = vec3<f32>(0.31, 0.0, 0.19) * t;
    let drift_b = vec3<f32>(-0.23, 0.0, 0.27) * t;
    let a = raa_fbm3(p + drift_a, 3);
    let b = raa_fbm3(p.yzx * 1.7 + vec3<f32>(11.3, 5.1, 2.7) + drift_b, 2);
    return a + 0.6 * b;
}

// Shading normal tilted by the world-space gradient of the two scrolling layers, applied
// through `raa_perturb_normal` with the gradient mapped onto the tangent basis.
fn liquid_detail_normal(
    n: vec3<f32>,
    tangent: vec4<f32>,
    world_pos: vec3<f32>,
    t: f32,
    strength: f32,
    scale: f32,
) -> vec3<f32> {
    let nn = normalize(n);
    let tangent0 = normalize(tangent.xyz - nn * dot(nn, tangent.xyz));
    let bitangent0 = cross(nn, tangent0) * tangent.w;
    let e = 0.1;
    let h0 = liquid_detail_height(world_pos, t, scale);
    let du = (liquid_detail_height(world_pos + tangent0 * e, t, scale) - h0) / e;
    let dv = (liquid_detail_height(world_pos + bitangent0 * e, t, scale) - h0) / e;
    // Cap the tilt so a large fbm gradient can never turn the shading normal on its side
    // (the sky then reads as a mottled blue skin instead of a glossy surface).
    let tilt = vec2<f32>(-du, -dv) * clamp(strength, 0.0, 1.0);
    let m = length(tilt);
    let limited = tilt * select(1.0, 1.2 / max(m, 1e-6), m > 1.2);
    let ts_normal = normalize(vec3<f32>(limited.x, limited.y, 1.0));
    return raa_perturb_normal(nn, tangent, ts_normal);
}

// Shore foam: 1 where the opaque surface seen through the liquid is within `foam_width`
// metres of the liquid surface, 0 where it is deeper. The opaque depth copy is loaded at
// the fragment pixel (clamped to its dimensions); when it is the dummy 1 the returned
// value is 0.
fn liquid_shore(frag: vec4<f32>, foam_width: f32) -> f32 {
    let dims = textureDimensions(raa_opaque_depth);
    let px = min(vec2<u32>(max(frag.xy, vec2<f32>(0.0))), dims - vec2<u32>(1u));
    let opaque_depth = textureLoad(raa_opaque_depth, px, 0).r;
    let surface_dist = liquid_view_distance(frag, frag.z);
    let opaque_dist = liquid_view_distance(frag, opaque_depth);
    return 1.0 - smoothstep(0.0, max(foam_width, 1e-4), opaque_dist - surface_dist);
}

// 1 at the border of the liquid mesh (uv 0/1) falling to 0 `width` (uv fraction) inside.
// A river ends at the channel wall where the opaque depth copy shows only the deep bed,
// so the depth-based shore term is 0 there; this term draws the contact foam line.
fn liquid_mesh_edge(uv: vec2<f32>, width: f32) -> f32 {
    let d = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    return 1.0 - smoothstep(0.0, max(width, 1e-4), d);
}

// Foam amount in [0, 1] from the shore term plus `amount` times the crest term. A two
// octave noise mask broken into lace keeps the foam thin and stringy (a smooth mask turned
// every whitecap into a milky blob, card 4940). The crest band starts low: only the very
// steepest phase reached the old 0.1..0.45 band, so captures showed no whitecaps at all
// (card 5018).
fn liquid_foam(shore: f32, crest: f32, world_pos: vec3<f32>, t: f32, amount: f32) -> f32 {
    let drift = vec3<f32>(0.0, 0.4 * t, 0.23 * t);
    let n = 0.62 * raa_noise3(world_pos * 6.5 + drift) + 0.38 * raa_noise3(world_pos * 15.0 - drift * 1.6);
    // The mask starts above the noise median: a lower start covered half the surface in
    // white blotches at capture distance (card 5027).
    let breakup = smoothstep(0.42, 0.78, clamp(0.5 + 0.5 * n, 0.0, 1.0));
    // Whitecaps only at the sharp crests, so the foam reads as lines rather than a haze.
    let caps = smoothstep(0.05, 0.26, crest) * clamp(amount, 0.0, 1.0);
    return clamp(max(shore, caps) * breakup, 0.0, 1.0);
}

// White, rough and opaque where the foam covers the surface; the transmitted part falls
// by `1 - f`.
fn liquid_apply_foam(o: MaterialOutput, f: f32) -> MaterialOutput {
    var out = o;
    let foam = clamp(f, 0.0, 1.0);
    out.base_color = mix(out.base_color, vec3<f32>(1.0), foam);
    out.roughness = mix(out.roughness, 0.9, foam);
    out.opacity = mix(out.opacity, 1.0, foam);
    out.transmission = out.transmission * (1.0 - foam);
    return out;
}
