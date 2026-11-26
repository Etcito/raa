// Adapted from solstice@7e465b5 crates/solstice-render/shaders/post/exposure.wgsl, MIT licence, Copyright (c) 2026 SamG-Coder
// Automatic exposure (S46): a luminance histogram of every 4th pixel, then the average over a
// percentile band, adapted over time and mapped to middle grey.

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

const BINS: u32 = 128u;
const MIN_LOG_LUMINANCE: f32 = -20.0;
const LOG_LUMINANCE_RANGE: f32 = 30.0;

@group(0) @binding(0) var<uniform> params: ExposureParams;
@group(0) @binding(1) var scene: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>, 128>;
@group(0) @binding(3) var<storage, read_write> state: ExposureState;

@compute @workgroup_size(16, 16)
fn build_histogram(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(scene);
    let pixel = id.xy * 4u;
    if any(pixel >= size) {
        return;
    }
    let c = textureLoad(scene, vec2i(pixel), 0).rgb;
    let luminance = dot(c, vec3f(0.2126, 0.7152, 0.0722));
    if luminance < 1.0e-7 {
        return;
    }
    let position = (log2(luminance) - MIN_LOG_LUMINANCE) / LOG_LUMINANCE_RANGE;
    let bin = u32(clamp(position, 0.0, 1.0) * f32(BINS - 1u));
    atomicAdd(&histogram[bin], 1u);
}

@compute @workgroup_size(1)
fn resolve() {
    var total = 0u;
    for (var i = 0u; i < BINS; i++) {
        total += atomicLoad(&histogram[i]);
    }
    let low_cut = f32(total) * params.low_percent;
    let high_cut = f32(total) * params.high_percent;
    var running = 0.0;
    var sum = 0.0;
    var weight = 0.0;
    for (var i = 0u; i < BINS; i++) {
        let count = f32(atomicExchange(&histogram[i], 0u));
        let next = running + count;
        let used = max(0.0, min(next, high_cut) - max(running, low_cut));
        let log_luminance = MIN_LOG_LUMINANCE + (f32(i) + 0.5) / f32(BINS) * LOG_LUMINANCE_RANGE;
        sum += log_luminance * used;
        weight += used;
        running = next;
    }
    let measured = exp2(sum / max(weight, 1.0));

    var average = measured;
    if total == 0u {
        average = select(1.0, state.average_luminance, state.valid == 1u);
    } else if params.first == 1u || state.valid == 0u {
        average = measured;
    } else {
        let rate = select(params.speed_down, params.speed_up, measured > state.average_luminance);
        average = mix(measured, state.average_luminance, exp(-params.dt_s * rate));
    }
    state.average_luminance = average;
    state.exposure = clamp(
        params.compensation * 0.18 / max(average, 1.0e-6),
        params.min_exposure,
        params.max_exposure,
    );
    state.valid = 1u;
    state.frame = params.frame;
}
