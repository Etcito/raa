// Adapted from solstice@7e465b5 crates/solstice-render/shaders/post/exposure.wgsl, MIT licence, Copyright (c) 2026 SamG-Coder
// Automatic exposure apply (S46): multiplies the HDR source by the adapted exposure.

struct ExposureParams {
    dt_s: f32,
    first: u32,
    compensation: f32,
    low_percent: f32,
    high_percent: f32,
    speed_up: f32,
    speed_down: f32,
    min_exposure: f32,
    max_exposure: f32,
    frame: u32,
    _pad0: u32,
    _pad1: u32,
}

struct ExposureState {
    average_luminance: f32,
    exposure: f32,
    valid: u32,
    frame: u32,
}

@group(0) @binding(0) var<uniform> params: ExposureParams;
@group(0) @binding(1) var source: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> state: ExposureState;

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

@fragment
fn fs_apply(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    // Phony assignment: keeps binding 0 statically used so the layout keeps it.
    _ = params.frame;
    let src = textureLoad(source, vec2<i32>(position.xy), 0);
    return vec4<f32>(src.rgb * state.exposure, src.a);
}
