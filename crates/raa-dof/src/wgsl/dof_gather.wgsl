// dof: half-res near/far prefilter, 48-tap spiral gather and full-res composite (S12
// `dof`, DOF values). Standalone render module (no PRELUDE), own bindings in group 0; each
// entry uses only the bindings listed on it. Every fragment entry is drawn with the
// fullscreen triangle of vs_main (3 vertices) into an HDR_FORMAT (rgba16float) target.
//
// The CoC (binding 2, written by dof_coc.wgsl's cs_coc) is the signed blur-circle DIAMETER
// in full-res px (negative = near); a blur radius is RADIUS_PER_COC * |CoC|. Every distance
// below is in full-res px. Both half-res layers are premultiplied: rgb = colour * a.
//   fs_prefilter    half res, MRT: 0 near (rgb * near weight, a = near coverage), 1 far
//   fs_gather_near  half res: the prefiltered near layer gathered (a = near coverage)
//   fs_gather_far   half res: the prefiltered far layer gathered, depth ordered
//   fs_composite    full res: sharp hdr, then far by the pixel's CoC, then near by its alpha

struct DofParams {
    inv_proj: mat4x4<f32>, // @0
    lens: vec4<f32>,       // @64 (focus m, aperture term K px, max coc px at viewport_h, near m)
    lens2: vec4<f32>,      // @80 (focal length f m, 0, 0, 0)
    viewport: vec4<f32>,   // @96 (w, h, 1/w, 1/h)
    autofocus: vec4<f32>,  // @112 (u, v, enabled 0|1, blend factor per frame)
}

// fs_composite (viewport, inv_proj).
@group(0) @binding(0) var<uniform> params: DofParams;
// fs_prefilter, fs_composite: hdr.src(), the sharp full-res image (textureLoad only).
@group(0) @binding(1) var hdr_tex: texture_2d<f32>;
// All fragment entries: the full-res signed CoC px (r32float, textureLoad only).
@group(0) @binding(2) var coc_tex: texture_2d<f32>;
// fs_gather_near, fs_gather_far: dilated 16x16 tiles (x = min CoC, y = max |CoC|).
@group(0) @binding(3) var tiles_tex: texture_2d<f32>;
// fs_composite: the S12 depth target (textureLoad only).
@group(0) @binding(4) var depth_tex: texture_depth_2d;
// fs_gather_near: the prefiltered near layer; fs_composite: the gathered near layer.
@group(0) @binding(5) var near_tex: texture_2d<f32>;
// fs_gather_far: the prefiltered far layer; fs_composite: the gathered far layer.
@group(0) @binding(6) var far_tex: texture_2d<f32>;

const TAPS: u32 = 48u;
// pi * (3 - sqrt(5)): round bokeh from a golden-angle (Vogel) spiral, no polygon.
const GOLDEN_ANGLE: f32 = 2.39996323;
// The CoC is a diameter.
const RADIUS_PER_COC: f32 = 0.5;
const TILE: i32 = 16;
const EPS: f32 = 1.0e-4;
// Depth used for sky and invalid pixels in the bilateral weights (m).
const FAR_DEPTH: f32 = 1.0e6;

struct VsOut {
    @builtin(position) position: vec4<f32>,
    // Texture coordinates of the target, (0, 0) top-left.
    @location(0) uv: vec2<f32>,
}

// Fullscreen triangle (3 vertices, no buffers).
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    let p = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: VsOut;
    out.position = vec4<f32>(p * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
    out.uv = vec2<f32>(p.x, 1.0 - p.y);
    return out;
}

// Weight of a pixel in the layer of `side` (-1 near, +1 far): its CoC beyond 0.5 px.
fn dof_layer_weight(coc: f32, side: f32) -> f32 {
    return smoothstep(0.5, 1.5, coc * side);
}

fn dof_load_coc(p: vec2<i32>) -> f32 {
    let last = vec2<i32>(textureDimensions(coc_tex)) - vec2<i32>(1);
    return textureLoad(coc_tex, clamp(p, vec2<i32>(0), last), 0).r;
}

fn dof_load_hdr(p: vec2<i32>) -> vec3<f32> {
    let last = vec2<i32>(textureDimensions(hdr_tex)) - vec2<i32>(1);
    return textureLoad(hdr_tex, clamp(p, vec2<i32>(0), last), 0).rgb;
}

// Half-res texel of the layer of `side` (clamped to the edge).
fn dof_load_layer(t: vec2<i32>, side: f32) -> vec4<f32> {
    let last = vec2<i32>(textureDimensions(near_tex)) - vec2<i32>(1);
    let q = clamp(t, vec2<i32>(0), last);
    if side < 0.0 {
        return textureLoad(near_tex, q, 0);
    }
    return textureLoad(far_tex, q, 0);
}

// Full-res pixel `i` (0..3) of the 2x2 block under half-res texel t.
fn dof_block_pixel(t: vec2<i32>, i: u32) -> vec2<i32> {
    return t * 2 + vec2<i32>(i32(i & 1u), i32(i >> 1u));
}

// Blur radius (px) of half-res texel t in the layer of `side`: the layer-weighted mean
// |CoC| of its 4 full-res pixels (0 when none of them belongs to the layer).
fn dof_texel_radius(t: vec2<i32>, side: f32) -> f32 {
    var sum = 0.0;
    var wsum = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let coc = dof_load_coc(dof_block_pixel(t, i)) * side;
        let w = dof_layer_weight(coc, 1.0);
        sum += w * coc;
        wsum += w;
    }
    return select(0.0, RADIUS_PER_COC * sum / max(wsum, EPS), wsum > EPS);
}

// Scatter-as-gather: a sample of radius r covers a centre at distance d (smoothstep over
// 1 px).
fn dof_cover(r: f32, d: f32) -> f32 {
    return smoothstep(d - 0.5, d + 0.5, r);
}

struct PrefilterOut {
    @location(0) near: vec4<f32>,
    @location(1) far: vec4<f32>,
}

// Half res: each layer averages the 2x2 full-res block weighted by its CoC, so in-focus
// pixels (weight 0) never bleed into a blurred layer.
@fragment
fn fs_prefilter(v: VsOut) -> PrefilterOut {
    let t = vec2<i32>(v.position.xy);
    var near = vec4<f32>(0.0);
    var far = vec4<f32>(0.0);
    for (var i = 0u; i < 4u; i++) {
        let p = dof_block_pixel(t, i);
        let c = dof_load_hdr(p);
        let coc = dof_load_coc(p);
        let wn = dof_layer_weight(coc, -1.0);
        let wf = dof_layer_weight(coc, 1.0);
        near += vec4<f32>(c * wn, wn);
        far += vec4<f32>(c * wf, wf);
    }
    var out: PrefilterOut;
    out.near = near * 0.25;
    out.far = far * 0.25;
    return out;
}

// 48-tap golden-angle spiral over the layer of `side` around half-res texel `frag`; the
// spiral radius is the dilated tile CoC of that layer (near: -min, far: max). Each tap is
// weighted by whether its own disc covers the centre and by 1 / its disc area (energy of a
// scattered sample), so a near-layer sum is directly the coverage of the centre. The far
// layer is depth ordered: a tap behind the centre (larger far CoC) reaches it only within
// the centre's own radius, so neither a sharp foreground nor the far background leaks
// over a nearer far object. Returns a premultiplied (rgb * a, a) half-res texel.
fn dof_gather(frag: vec2<f32>, side: f32) -> vec4<f32> {
    let c = vec2<i32>(frag);
    let tiles_last = vec2<i32>(textureDimensions(tiles_tex)) - vec2<i32>(1);
    let tile = textureLoad(tiles_tex, clamp(c * 2 / TILE, vec2<i32>(0), tiles_last), 0).rg;
    let big_r = RADIUS_PER_COC * max(select(tile.y, -tile.x, side < 0.0), 0.0);
    if big_r < 0.5 {
        // in focus around this tile: the prefiltered layer as is
        return dof_load_layer(c, side);
    }
    let rc = dof_texel_radius(c, side);
    let r_min = big_r * inverseSqrt(f32(TAPS));
    let tap_area = big_r * big_r / f32(TAPS);
    var acc = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var i = 0u; i < TAPS; i++) {
        let fi = f32(i);
        let rho = big_r * sqrt((fi + 0.5) / f32(TAPS));
        let theta = fi * GOLDEN_ANGLE;
        let off = vec2<f32>(cos(theta), sin(theta)) * rho;
        // full-res px offset onto the half-res grid
        let t = vec2<i32>(floor(frag + off * 0.5));
        let s = dof_load_layer(t, side);
        let r = dof_texel_radius(t, side);
        var reach = r;
        if side > 0.0 {
            reach = min(r, rc);
        }
        let rr = max(r, r_min);
        let w = dof_cover(reach, rho) * tap_area / (rr * rr);
        acc += s * w;
        wsum += w;
    }
    if acc.a < EPS {
        return vec4<f32>(0.0);
    }
    let colour = acc.rgb / acc.a;
    var alpha = clamp(acc.a, 0.0, 1.0);
    if side > 0.0 {
        // far: the share of the covering taps that belong to the layer
        alpha = clamp(acc.a / max(wsum, EPS), 0.0, 1.0);
    }
    return vec4<f32>(colour * alpha, alpha);
}

// Half res: the near layer blurred over everything behind it; a = near coverage.
@fragment
fn fs_gather_near(v: VsOut) -> @location(0) vec4<f32> {
    return dof_gather(v.position.xy, -1.0);
}

// Half res: the far layer blurred, depth ordered; a = share of far samples.
@fragment
fn fs_gather_far(v: VsOut) -> @location(0) vec4<f32> {
    return dof_gather(v.position.xy, 1.0);
}

// Linear view depth (m) of full-res pixel p, FAR_DEPTH for sky or an invalid depth.
fn dof_pixel_depth(p: vec2<i32>) -> f32 {
    let last = vec2<i32>(textureDimensions(depth_tex)) - vec2<i32>(1);
    let q = clamp(p, vec2<i32>(0), last);
    let uv = (vec2<f32>(q) + vec2<f32>(0.5)) * params.viewport.zw;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, textureLoad(depth_tex, q, 0), 1.0);
    let vp = params.inv_proj * ndc;
    let z = -vp.z / vp.w;
    return select(FAR_DEPTH, z, z > 0.0 && z < FAR_DEPTH);
}

// Full res into hdr.dst(): bilateral 4-tap upsample of both gathered layers (bilinear
// weights times depth and CoC similarity to this pixel), the far layer blended over the
// sharp hdr by this pixel's far CoC beyond 0.5 px, then the near layer by its alpha.
@fragment
fn fs_composite(v: VsOut) -> @location(0) vec4<f32> {
    let p = vec2<i32>(v.position.xy);
    let sharp = textureLoad(hdr_tex, p, 0);
    let coc_c = dof_load_coc(p);
    let z_c = dof_pixel_depth(p);
    let hp = (vec2<f32>(p) + vec2<f32>(0.5)) * 0.5 - vec2<f32>(0.5);
    let base = vec2<i32>(floor(hp));
    let fr = hp - floor(hp);
    var near = vec4<f32>(0.0);
    var far = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let o = vec2<i32>(i32(i & 1u), i32(i >> 1u));
        let t = base + o;
        let bw = mix(vec2<f32>(1.0) - fr, fr, vec2<f32>(o));
        let q = t * 2;
        let wz = 1.0 / (1.0 + 32.0 * abs(dof_pixel_depth(q) - z_c) / z_c);
        let wc = 1.0 / (1.0 + abs(dof_load_coc(q) - coc_c));
        let w = bw.x * bw.y * (wz * wc + EPS);
        near += dof_load_layer(t, -1.0) * w;
        far += dof_load_layer(t, 1.0) * w;
        wsum += w;
    }
    near /= max(wsum, EPS);
    far /= max(wsum, EPS);
    var colour = sharp.rgb;
    if far.a > EPS {
        colour = mix(colour, far.rgb / far.a, dof_layer_weight(coc_c, 1.0));
    }
    colour = colour * (1.0 - clamp(near.a, 0.0, 1.0)) + near.rgb;
    return vec4<f32>(colour, sharp.a);
}
