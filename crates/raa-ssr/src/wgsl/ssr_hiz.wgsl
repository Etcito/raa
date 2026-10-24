// SSR inputs (S12 `ssr`): the Hi-Z min pyramid of `depth` (compute, 8x8 workgroups) and the
// filtered mip chain of the HDR colour the reflections sample by roughness (fragment, one
// render target per mip). Standalone: own bindings in group 0, no PRELUDE. Each entry uses
// only its own bindings, so each pipeline gets its own layout.

struct HizParams {
    // (src w, src h, dst w, dst h) of the level being written; cs_hiz_copy uses src = dst
    size: vec4<u32>,
}

@group(0) @binding(0) var<uniform> hiz: HizParams;
// cs_hiz_copy: the prepass depth target (Depth32Float, single-sample).
@group(0) @binding(1) var depth: texture_depth_2d;
// cs_hiz_reduce: level n of the pyramid.
@group(0) @binding(2) var hiz_src: texture_2d<f32>;
// Both compute entries: the level written (mip 0 for the copy, n + 1 for the reduce).
@group(0) @binding(3) var hiz_dst: texture_storage_2d<r32float, write>;
// fs_downsample: the previous mip of the HDR_FORMAT colour chain (a single-level view).
@group(0) @binding(4) var color_src: texture_2d<f32>;
// Linear filtering, clamp to edge.
@group(0) @binding(5) var color_samp: sampler;

@compute @workgroup_size(8, 8, 1)
fn cs_hiz_copy(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= hiz.size.z || id.y >= hiz.size.w) {
        return;
    }
    let d = textureLoad(depth, vec2<i32>(id.xy), 0);
    textureStore(hiz_dst, vec2<i32>(id.xy), vec4<f32>(d, 0.0, 0.0, 1.0));
}

// Inclusive source range along one axis of destination index i: the rule of src/hiz.rs
// (`footprint`): [2i, 2i + 1], except that the last destination index (i + 1 == dst_len)
// extends to src_len - 1, so an odd source keeps its extra column or row (a 1-texel source
// maps to [0, 0]).
fn hiz_footprint(i: u32, dst_len: u32, src_len: u32) -> vec2<u32> {
    let start = 2u * i;
    var end = start + 1u;
    if (i + 1u == dst_len) {
        end = src_len - 1u;
    }
    return vec2<u32>(start, end);
}

// Level n + 1 = min over each texel's footprint in level n, the exact rule of src/hiz.rs
// (`reduce_min`): dst size max(1, w / 2) x max(1, h / 2), 2x2 min, and the last column or
// row of an odd source covers 3 source columns or rows (at most 3x3 texels).
@compute @workgroup_size(8, 8, 1)
fn cs_hiz_reduce(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= hiz.size.z || id.y >= hiz.size.w) {
        return;
    }
    let fx = hiz_footprint(id.x, hiz.size.z, hiz.size.x);
    let fy = hiz_footprint(id.y, hiz.size.w, hiz.size.y);
    var m = 3.4028235e38;
    for (var y = fy.x; y <= fy.y; y = y + 1u) {
        for (var x = fx.x; x <= fx.y; x = x + 1u) {
            m = min(m, textureLoad(hiz_src, vec2<i32>(i32(x), i32(y)), 0).r);
        }
    }
    textureStore(hiz_dst, vec2<i32>(id.xy), vec4<f32>(m, 0.0, 0.0, 1.0));
}

struct VsOut {
    @builtin(position) position: vec4<f32>,
    // Texture coordinates of the target, (0, 0) top-left.
    @location(0) uv: vec2<f32>,
}

// Fullscreen triangle (3 vertices, no buffers).
@vertex
fn vs_fullscreen(@builtin(vertex_index) index: u32) -> VsOut {
    let p = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: VsOut;
    out.position = vec4<f32>(p * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
    out.uv = vec2<f32>(p.x, 1.0 - p.y);
    return out;
}

// Bilinear sample of the previous mip at `uv` offset by `o` of its texels.
fn ssr_tap(uv: vec2<f32>, texel: vec2<f32>, o: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(color_src, color_samp, uv + o * texel, 0.0).rgb;
}

// 13-tap downsample (Jimenez 2014, CoD: Advanced Warfare), its 5 overlapping 4x4 boxes:
//   a . b . c
//   . j . k .
//   d . e . f
//   . l . m .
//   g . h . i
// centre box (j, k, l, m) weight 0.5, the 4 corner quads 0.125 each, i.e. per tap
// e 0.125, j k l m 0.125, b d f h 0.0625, a c g i 0.03125. No Karis weighting: the
// chain stays an unbiased blur of the radiance the reflections sample.
@fragment
fn fs_downsample(in: VsOut) -> @location(0) vec4<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(color_src, 0));
    let a = ssr_tap(in.uv, texel, vec2<f32>(-2.0, -2.0));
    let b = ssr_tap(in.uv, texel, vec2<f32>(0.0, -2.0));
    let c = ssr_tap(in.uv, texel, vec2<f32>(2.0, -2.0));
    let d = ssr_tap(in.uv, texel, vec2<f32>(-2.0, 0.0));
    let e = ssr_tap(in.uv, texel, vec2<f32>(0.0, 0.0));
    let f = ssr_tap(in.uv, texel, vec2<f32>(2.0, 0.0));
    let g = ssr_tap(in.uv, texel, vec2<f32>(-2.0, 2.0));
    let h = ssr_tap(in.uv, texel, vec2<f32>(0.0, 2.0));
    let i = ssr_tap(in.uv, texel, vec2<f32>(2.0, 2.0));
    let j = ssr_tap(in.uv, texel, vec2<f32>(-1.0, -1.0));
    let k = ssr_tap(in.uv, texel, vec2<f32>(1.0, -1.0));
    let l = ssr_tap(in.uv, texel, vec2<f32>(-1.0, 1.0));
    let m = ssr_tap(in.uv, texel, vec2<f32>(1.0, 1.0));
    let sum = e * 0.125 + (j + k + l + m) * 0.125 + (b + d + f + h) * 0.0625
        + (a + c + g + i) * 0.03125;
    return vec4<f32>(sum, 1.0);
}
