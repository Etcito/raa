// Bounded differentiable foliage update (M7, contract `m7-compute`). One invocation owns
// one optimizer micro-triangle: it reads the 80-byte gradient record the previous stage
// wrote, applies one bounded learning-rate step to the three vertices, normalizes the
// independent normal separately (falling back to the previous normal, then the triangle's
// geometric normal, then +Y when the candidate is degenerate) and anneals the opacity
// toward the params target. A non-finite input or result never reaches the triangle record;
// the terminal freeze in foliage_update.rs refuses a non-finite readback and writes the
// exact opaque opacity.
struct Params {
    count: u32, views: u32, resolution: u32, iteration: u32,
    silhouette: f32, normal: f32, edge: f32, temperature: f32,
    learning_rate: f32, opacity_target: f32, capacity: u32, generation: u32,
    reserved: vec4<u32>,
}
struct Triangle { p0: vec4<f32>, p1: vec4<f32>, p2: vec4<f32>, normal: vec4<f32>, reserved: vec4<f32> }
struct Gradient { dp0: vec4<f32>, dp1: vec4<f32>, dp2: vec4<f32>, normal_opacity: vec4<f32>, reserved: vec4<f32> }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> triangles: array<Triangle>;
@group(0) @binding(2) var<storage, read> gradients: array<Gradient>;

const MAX_POSITION_STEP: f32 = 0.05;
const MAX_NORMAL_STEP: f32 = 1.0;
const MAX_OPACITY_STEP: f32 = 0.25;
const OPACITY_ANNEAL_STEP: f32 = 0.1;
const NORMAL_EPSILON: f32 = 1e-8;

// True for a finite f32 (naga's WGSL front-end has no isNan/isInf builtin; `v - v` is NaN
// for both NaN and infinity, and every comparison against NaN is false).
fn update_finite(v: f32) -> bool {
    return (v - v) == 0.0;
}

fn update_finite3(v: vec3<f32>) -> bool {
    return update_finite(v.x) && update_finite(v.y) && update_finite(v.z);
}

fn update_length(v: vec3<f32>) -> f32 {
    return sqrt(dot(v, v));
}

// One bounded, finite descent step `-learning_rate * gradient`: a non-finite rate or
// gradient contributes no step and a step longer than `limit` is scaled back to it.
fn update_step(gradient: vec3<f32>, learning_rate: f32, limit: f32) -> vec3<f32> {
    if (!update_finite(learning_rate) || !update_finite3(gradient) || !update_finite(limit) || limit <= 0.0) {
        return vec3<f32>(0.0);
    }
    let step = -learning_rate * gradient;
    let length = update_length(step);
    if (!update_finite(length)) { return vec3<f32>(0.0); }
    if (length > limit) { return step * (limit / length); }
    return step;
}

fn update_position(position: vec3<f32>, gradient: vec3<f32>, learning_rate: f32) -> vec3<f32> {
    let next = position + update_step(gradient, learning_rate, MAX_POSITION_STEP);
    if (update_finite3(next)) { return next; }
    if (update_finite3(position)) { return position; }
    return vec3<f32>(0.0);
}

// The independent normal: the bounded candidate normalized on its own. A zero or
// non-finite candidate keeps the normalized previous normal; when that is degenerate too
// the triangle's geometric normal is used, and +Y is the defined final fallback.
fn update_normal(previous: vec3<f32>, gradient: vec3<f32>, learning_rate: f32, p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>) -> vec3<f32> {
    let candidate = previous + update_step(gradient, learning_rate, MAX_NORMAL_STEP);
    let candidate_length = update_length(candidate);
    if (update_finite(candidate_length) && candidate_length > NORMAL_EPSILON) {
        return candidate / candidate_length;
    }
    let previous_length = update_length(previous);
    if (update_finite(previous_length) && previous_length > NORMAL_EPSILON) {
        return previous / previous_length;
    }
    let geometric = cross(p1 - p0, p2 - p0);
    let geometric_length = update_length(geometric);
    if (update_finite(geometric_length) && geometric_length > NORMAL_EPSILON) {
        return geometric / geometric_length;
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}

// A bounded anneal toward the clamped target: a non-finite current restarts from the
// opaque 1, a non-finite target anneals toward it and a remaining distance within one
// OPACITY_ANNEAL_STEP snaps exactly onto the target, so repeated updates terminate on it.
fn anneal_opacity(current: f32, opacity_target: f32) -> f32 {
    let goal = select(1.0, clamp(opacity_target, 0.0, 1.0), update_finite(opacity_target));
    let start = select(1.0, clamp(current, 0.0, 1.0), update_finite(current));
    let delta = goal - start;
    if (abs(delta) <= OPACITY_ANNEAL_STEP) { return goal; }
    return start + sign(delta) * OPACITY_ANNEAL_STEP;
}

fn update_opacity(current: f32, derivative: f32, learning_rate: f32, opacity_target: f32) -> f32 {
    let step = update_step(vec3<f32>(derivative, 0.0, 0.0), learning_rate, MAX_OPACITY_STEP).x;
    return anneal_opacity(clamp(current + step, 0.0, 1.0), opacity_target);
}

@compute @workgroup_size(64)
fn update_triangles(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= params.count) { return; }
    let triangle = triangles[index];
    let gradient = gradients[index];
    let rate = params.learning_rate;
    let p0 = update_position(triangle.p0.xyz, gradient.dp0.xyz, rate);
    let p1 = update_position(triangle.p1.xyz, gradient.dp1.xyz, rate);
    let p2 = update_position(triangle.p2.xyz, gradient.dp2.xyz, rate);
    let normal = update_normal(triangle.normal.xyz, gradient.normal_opacity.xyz, rate, p0, p1, p2);
    let opacity = update_opacity(triangle.normal.w, gradient.normal_opacity.w, rate, params.opacity_target);
    var record: Triangle;
    record.p0 = vec4<f32>(p0, 0.0);
    record.p1 = vec4<f32>(p1, 0.0);
    record.p2 = vec4<f32>(p2, 0.0);
    record.normal = vec4<f32>(normal, opacity);
    record.reserved = vec4<f32>(0.0);
    triangles[index] = record;
}
