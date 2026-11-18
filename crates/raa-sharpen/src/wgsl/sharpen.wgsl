// Adapted from solstice@7e465b5 crates/solstice-render/shaders/post/tonemap.wgsl, MIT licence, Copyright (c) 2026 SamG-Coder
// Contrast-adaptive sharpening on the HDR chain (S47): taps are range-compressed, sharpened
// within their neighbourhood range, then expanded back.

struct SharpenParams {
    sharpness: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: SharpenParams;
@group(0) @binding(1) var source: texture_2d<f32>;

struct VsOut {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    let p = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: VsOut;
    out.position = vec4<f32>(p * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
    return out;
}

fn compress(c: vec3<f32>) -> vec3<f32> {
    let p = max(c, vec3<f32>(0.0));
    return p / (1.0 + max(p.r, max(p.g, p.b)));
}

fn expand(d: vec3<f32>) -> vec3<f32> {
    return d / (1.0 - min(max(d.r, max(d.g, d.b)), 0.999));
}

fn tap(pixel: vec2<i32>, offset: vec2<i32>) -> vec4<f32> {
    let last = vec2<i32>(textureDimensions(source)) - vec2<i32>(1);
    return textureLoad(source, clamp(pixel + offset, vec2<i32>(0), last), 0);
}

@fragment
fn fs_sharpen(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    let centre = tap(pixel, vec2<i32>(0, 0));
    let c = compress(centre.rgb);
    let n = compress(tap(pixel, vec2<i32>(0, -1)).rgb);
    let s = compress(tap(pixel, vec2<i32>(0, 1)).rgb);
    let w = compress(tap(pixel, vec2<i32>(-1, 0)).rgb);
    let e = compress(tap(pixel, vec2<i32>(1, 0)).rgb);
    let low = min(c, min(min(n, s), min(w, e)));
    let high = max(c, max(max(n, s), max(w, e)));
    let ratio = min(low, vec3<f32>(1.0) - high) / max(high, vec3<f32>(1e-4));
    let amount = sqrt(clamp(ratio, vec3<f32>(0.0), vec3<f32>(1.0)));
    let weight = -amount * params.sharpness * 0.2;
    let sharpened = (c + (n + s + w + e) * weight) / (vec3<f32>(1.0) + 4.0 * weight);
    let out = clamp(sharpened, low, high);
    return vec4<f32>(expand(out), centre.a);
}
