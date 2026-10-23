// Physically based bloom (S12 `bloom`, BLOOM values): a 13-tap downsample chain from the
// HDR source (Jimenez 2014, "Next Generation Post Processing in Call of Duty: Advanced
// Warfare"), Karis luma-weighted averaging of the 5 box groups on the first level, a 3x3
// tent upsample scaled by `radius` added (blend One, One) into the next larger level, and
// the composite `mix(hdr, bloom, strength)`. No threshold. Standalone: own bindings in
// group 0, no PRELUDE.

struct BloomParams {
    // 1 / size of the level the pass reads (the HDR source for fs_down_first and
    // fs_composite)
    texel_size: vec2<f32>,
    // upsample tent offset in source texels, [0, 1]
    radius: f32,
    // composite mix factor, [0, 1]
    strength: f32,
}

@group(0) @binding(0) var<uniform> params: BloomParams;
// The level the pass reads: hdr.src() (fs_down_first, fs_composite) or a bloom mip.
@group(0) @binding(1) var src_tex: texture_2d<f32>;
// Linear filtering, clamp to edge.
@group(0) @binding(2) var src_samp: sampler;
// fs_composite only: bloom level 0 after the upsample chain.
@group(0) @binding(3) var bloom_tex: texture_2d<f32>;

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

fn bloom_luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Bilinear sample of the source at `uv` offset by `o` source texels; each tap averages
// the 2x2 texels around it. Negative values are clamped so the Karis weights stay in (0, 1].
fn bloom_tap(uv: vec2<f32>, o: vec2<f32>) -> vec3<f32> {
    let c = textureSampleLevel(src_tex, src_samp, uv + o * params.texel_size, 0.0).rgb;
    return max(c, vec3<f32>(0.0));
}

// The 13 taps of the CoD downsample, grouped into its 5 overlapping 4x4 boxes:
//   a . b . c
//   . j . k .
//   d . e . f
//   . l . m .
//   g . h . i
// box 0 = (j, k, l, m) weight 0.5; boxes 1..4 = the corner quads (a, b, d, e),
// (b, c, e, f), (d, e, g, h), (e, f, h, i) weight 0.125 each.
struct Boxes {
    b0: vec3<f32>,
    b1: vec3<f32>,
    b2: vec3<f32>,
    b3: vec3<f32>,
    b4: vec3<f32>,
}

fn bloom_boxes(uv: vec2<f32>) -> Boxes {
    let a = bloom_tap(uv, vec2<f32>(-2.0, -2.0));
    let b = bloom_tap(uv, vec2<f32>(0.0, -2.0));
    let c = bloom_tap(uv, vec2<f32>(2.0, -2.0));
    let d = bloom_tap(uv, vec2<f32>(-2.0, 0.0));
    let e = bloom_tap(uv, vec2<f32>(0.0, 0.0));
    let f = bloom_tap(uv, vec2<f32>(2.0, 0.0));
    let g = bloom_tap(uv, vec2<f32>(-2.0, 2.0));
    let h = bloom_tap(uv, vec2<f32>(0.0, 2.0));
    let i = bloom_tap(uv, vec2<f32>(2.0, 2.0));
    let j = bloom_tap(uv, vec2<f32>(-1.0, -1.0));
    let k = bloom_tap(uv, vec2<f32>(1.0, -1.0));
    let l = bloom_tap(uv, vec2<f32>(-1.0, 1.0));
    let m = bloom_tap(uv, vec2<f32>(1.0, 1.0));
    var out: Boxes;
    out.b0 = (j + k + l + m) * 0.25;
    out.b1 = (a + b + d + e) * 0.25;
    out.b2 = (b + c + e + f) * 0.25;
    out.b3 = (d + e + g + h) * 0.25;
    out.b4 = (e + f + h + i) * 0.25;
    return out;
}

// Box weight scaled by the Karis factor 1 / (1 + luma) of the box average.
fn bloom_karis(weight: f32, c: vec3<f32>) -> f32 {
    return weight / (1.0 + bloom_luma(c));
}

// First level, from hdr.src(): each box is weighted by its Karis factor and the sum is
// renormalised, so a single very bright texel (firefly) cannot dominate a box, while the
// weighting per box (not per tap) keeps the result smooth (no halo banding).
@fragment
fn fs_down_first(in: VsOut) -> @location(0) vec4<f32> {
    let s = bloom_boxes(in.uv);
    let w0 = bloom_karis(0.5, s.b0);
    let w1 = bloom_karis(0.125, s.b1);
    let w2 = bloom_karis(0.125, s.b2);
    let w3 = bloom_karis(0.125, s.b3);
    let w4 = bloom_karis(0.125, s.b4);
    // Every weight is > 0 for finite input (luma >= 0 after the clamp in bloom_tap).
    let sum = (s.b0 * w0 + s.b1 * w1 + s.b2 * w2 + s.b3 * w3 + s.b4 * w4)
        / (w0 + w1 + w2 + w3 + w4);
    return vec4<f32>(sum, 1.0);
}

// Later levels: the plain 13-tap filter (box weights 0.5 + 4 x 0.125).
@fragment
fn fs_down(in: VsOut) -> @location(0) vec4<f32> {
    let s = bloom_boxes(in.uv);
    let sum = s.b0 * 0.5 + (s.b1 + s.b2 + s.b3 + s.b4) * 0.125;
    return vec4<f32>(sum, 1.0);
}

// 3x3 tent (1 2 1 / 2 4 2 / 1 2 1, / 16) over the smaller level, taps `radius` source
// texels apart. Drawn with blend (One, One) into the next larger level; alpha 0 leaves the
// destination alpha unchanged.
@fragment
fn fs_up(in: VsOut) -> @location(0) vec4<f32> {
    let o = params.texel_size * params.radius;
    let uv = in.uv;
    let a = textureSampleLevel(src_tex, src_samp, uv + vec2<f32>(-o.x, -o.y), 0.0).rgb;
    let b = textureSampleLevel(src_tex, src_samp, uv + vec2<f32>(0.0, -o.y), 0.0).rgb;
    let c = textureSampleLevel(src_tex, src_samp, uv + vec2<f32>(o.x, -o.y), 0.0).rgb;
    let d = textureSampleLevel(src_tex, src_samp, uv + vec2<f32>(-o.x, 0.0), 0.0).rgb;
    let e = textureSampleLevel(src_tex, src_samp, uv, 0.0).rgb;
    let f = textureSampleLevel(src_tex, src_samp, uv + vec2<f32>(o.x, 0.0), 0.0).rgb;
    let g = textureSampleLevel(src_tex, src_samp, uv + vec2<f32>(-o.x, o.y), 0.0).rgb;
    let h = textureSampleLevel(src_tex, src_samp, uv + vec2<f32>(0.0, o.y), 0.0).rgb;
    let i = textureSampleLevel(src_tex, src_samp, uv + vec2<f32>(o.x, o.y), 0.0).rgb;
    let sum = e * 4.0 + (b + d + f + h) * 2.0 + (a + c + g + i);
    return vec4<f32>(sum * (1.0 / 16.0), 0.0);
}

// hdr.src() (exact texel) mixed with bloom level 0 (bilinear upscale), written to hdr.dst().
@fragment
fn fs_composite(in: VsOut) -> @location(0) vec4<f32> {
    let hdr = textureLoad(src_tex, vec2<i32>(in.position.xy), 0);
    let bloom = textureSampleLevel(bloom_tex, src_samp, in.uv, 0.0).rgb;
    return vec4<f32>(mix(hdr.rgb, bloom, params.strength), hdr.a);
}
