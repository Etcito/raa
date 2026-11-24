struct Frame {
    size: vec2<u32>,
    origin: vec2<u32>,
    extent: vec2<u32>,
}
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> raw: array<f32>;
@group(0) @binding(4) var<uniform> frame: Frame;

const scales = array<f32, 12>(1, 1, 1, 1, 1, 1, 1, 0.125, 8, 8, 1, 1);

fn index(p: vec2<u32>) -> u32 {
    return p.y * frame.size.x + p.x;
}
fn tap(p: vec2<u32>, k: u32) -> u32 {
    let offset = vec2<i32>(i32(k % 3), i32(k / 3)) - vec2<i32>(1);
    let q = clamp(vec2<i32>(p) + offset, vec2<i32>(frame.origin),
                  vec2<i32>(frame.origin + frame.extent) - vec2<i32>(1));
    return index(vec2<u32>(q));
}
fn dense(pixel: u32, channel: u32, weight: u32, bias: u32) -> f32 {
    var sum = weights[bias + channel];
    for (var i = 0u; i < 16u; i++) {
        sum += weights[weight + channel * 16u + i] * input[pixel * 16u + i];
    }
    return sum;
}
fn depthwise(p: vec2<u32>, weight: u32, bias: u32) {
    var neighbors: array<u32, 9>;
    for (var k = 0u; k < 9u; k++) {
        neighbors[k] = tap(p, k) * 16u;
    }
    let pixel = index(p) * 16u;
    for (var c = 0u; c < 16u; c++) {
        var sum = weights[bias + c];
        for (var k = 0u; k < 9u; k++) {
            sum += weights[weight + c * 9u + k] * input[neighbors[k] + c];
        }
        output[pixel + c] = max(0.0, sum);
    }
}
fn pointwise(p: vec2<u32>, weight: u32, bias: u32) {
    let pixel = index(p);
    for (var c = 0u; c < 16u; c++) {
        output[pixel * 16u + c] = max(0.0, dense(pixel, c, weight, bias));
    }
}
fn sigmoid(x: f32) -> f32 {
    let e = exp(-abs(x));
    return select(e, 1.0, x >= 0.0) / (1.0 + e);
}
@compute @workgroup_size(8, 8)
fn stem(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id.xy >= frame.extent)) { return; }
    let pixel = index(id.xy + frame.origin);
    let base = pixel * 12u;
    let valid = raw[base + 10u] == 1.0;
    var lanes: array<f32, 12>;
    for (var i = 0u; i < 12u; i++) {
        lanes[i] = select(0.0, raw[base + i] * scales[i], valid || (i != 2u && i != 3u));
    }
    for (var c = 0u; c < 16u; c++) {
        var sum = weights[192u + c];
        for (var i = 0u; i < 12u; i++) {
            sum += weights[c * 12u + i] * lanes[i];
        }
        output[pixel * 16u + c] = max(0.0, sum);
    }
}
@compute @workgroup_size(8, 8)
fn depthwise1(@builtin(global_invocation_id) id: vec3<u32>) {
    if (all(id.xy < frame.extent)) { depthwise(id.xy + frame.origin, 208u, 352u); }
}
@compute @workgroup_size(8, 8)
fn pointwise1(@builtin(global_invocation_id) id: vec3<u32>) {
    if (all(id.xy < frame.extent)) { pointwise(id.xy + frame.origin, 368u, 624u); }
}
@compute @workgroup_size(8, 8)
fn depthwise2(@builtin(global_invocation_id) id: vec3<u32>) {
    if (all(id.xy < frame.extent)) { depthwise(id.xy + frame.origin, 640u, 784u); }
}
@compute @workgroup_size(8, 8)
fn pointwise2(@builtin(global_invocation_id) id: vec3<u32>) {
    if (all(id.xy < frame.extent)) { pointwise(id.xy + frame.origin, 800u, 1056u); }
}
@compute @workgroup_size(8, 8)
fn head(@builtin(global_invocation_id) id: vec3<u32>) {
    if (any(id.xy >= frame.extent)) { return; }
    let p = id.xy + frame.origin;
    let pixel = index(p);
    let base = pixel * 12u;
    var logits: array<f32, 9>;
    logits[0] = dense(pixel, 0u, 1072u, 1248u);
    var largest = logits[0];
    for (var k = 1u; k < 9u; k++) {
        logits[k] = dense(pixel, k, 1072u, 1248u);
        largest = max(largest, logits[k]);
    }
    var spatial_mass: array<f32, 9>;
    var total = 0.0;
    for (var k = 0u; k < 9u; k++) {
        logits[k] -= largest;
        spatial_mass[k] = exp(logits[k]);
        total += spatial_mass[k];
    }
    let log_total = log(total);
    var masses: array<f32, 10>;
    var ratios: array<f32, 10>;
    var occupied: array<bool, 10>;
    var any_weight = false;
    var largest_mass = -3.402823e38;
    for (var k = 0u; k < 9u; k++) {
        output[base + k] = spatial_mass[k] / total;
        let neighbor = tap(p, k) * 12u;
        let weight = raw[neighbor + 1u];
        if (weight > 0.0) {
            masses[k] = logits[k] - log_total + log(weight);
            ratios[k] = raw[neighbor] / weight;
            occupied[k] = true;
            any_weight = true;
            largest_mass = max(largest_mass, masses[k]);
        }
    }
    var history = 0.0;
    if (raw[base + 10u] == 1.0) {
        let gate_logit = dense(pixel, 9u, 1072u, 1248u);
        history = sigmoid(gate_logit);
        let weight = raw[base + 3u];
        if (weight > 0.0) {
            // Cancel (1 - history) from both sums, leaving h/(1-h) = exp(gate_logit).
            masses[9] = gate_logit + log(weight);
            ratios[9] = raw[base + 2u] / weight;
            occupied[9] = true;
            any_weight = true;
            largest_mass = max(largest_mass, masses[9]);
        }
    }
    var visibility = 1.0;
    if (any_weight) {
        // At least one occupied tap has mass one after shifting. Empty support alone
        // selects the fallback, even if unoccupied taps had the largest spatial logits.
        var weighted_visibility = 0.0;
        var weight_sum = 0.0;
        for (var k = 0u; k < 10u; k++) {
            if (occupied[k]) {
                let weight = exp(masses[k] - largest_mass);
                weighted_visibility += weight * ratios[k];
                weight_sum += weight;
            }
        }
        visibility = weighted_visibility / weight_sum;
    }
    output[base + 9u] = history;
    output[base + 10u] = sigmoid(dense(pixel, 10u, 1072u, 1248u));
    output[base + 11u] = visibility;
}
