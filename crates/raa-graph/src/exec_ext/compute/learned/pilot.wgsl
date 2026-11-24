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
    var total = 0.0;
    for (var k = 0u; k < 9u; k++) {
        logits[k] = exp(logits[k] - largest);
        total += logits[k];
    }
    var pair = vec2<f32>(0.0);
    for (var k = 0u; k < 9u; k++) {
        let spatial = logits[k] / total;
        let neighbor = tap(p, k) * 12u;
        pair += spatial * vec2<f32>(raw[neighbor], raw[neighbor + 1u]);
        output[base + k] = spatial;
    }
    var history = 0.0;
    if (raw[base + 10u] == 1.0) {
        history = sigmoid(dense(pixel, 9u, 1072u, 1248u));
        pair = mix(pair, vec2<f32>(raw[base + 2u], raw[base + 3u]), history);
    }
    var visibility = 1.0;
    if (pair.y > 0.0) {
        visibility = pair.x / pair.y;
    }
    output[base + 9u] = history;
    output[base + 10u] = sigmoid(dense(pixel, 10u, 1072u, 1248u));
    output[base + 11u] = visibility;
}
