// Neural radiance cache training: per-sample forward and backward, then the weight gradient
// summed per chunk. Constants NRC_HIDDEN, NRC_FREQS, NRC_LAYERS, NRC_W4, NRC_BATCH, NRC_CHUNK and
// NRC_CHUNKS are prepended; `nrc_mlp.wgsl` follows the bindings.

struct NrcTrain {
    // learning rate, beta1, beta2, epsilon
    adam: vec4<f32>,
    // 1 / (1 - beta1^t), 1 / (1 - beta2^t), EMA weight of the previous average, loss epsilon
    correction: vec4<f32>,
}
@group(0) @binding(0) var<uniform> nrc_train: NrcTrain;
@group(0) @binding(1) var<storage, read> nrc_weights: array<vec4<f32>>;
// Three rows per sample: position.xyz + valid, normal.xyz, target.rgb.
@group(0) @binding(2) var<storage, read> nrc_samples: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> nrc_acts: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> nrc_deltas: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> nrc_partial: array<vec4<f32>>;

//!include nrc_mlp

// One invocation per training sample: forward keeping every layer input, relative L2 loss
// (normalized by the squared predicted luminance plus an epsilon),
// backward.
@compute @workgroup_size(64)
fn nrc_forward_backward(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= NRC_BATCH { return; }
    let out = id.x * NRC_LAYERS * 16u;
    let record = nrc_samples[id.x * 3u];
    if record.w < 0.5 {
        for (var j = 0u; j < NRC_LAYERS * 16u; j++) { nrc_deltas[out + j] = vec4<f32>(0.0); }
        return;
    }
    var x: array<array<vec4<f32>, 16>, NRC_LAYERS>;
    var a: array<vec4<f32>, 16>;
    var b: array<vec4<f32>, 16>;
    nrc_encode(record.xyz, nrc_samples[id.x * 3u + 1u].xyz, &a);
    x[0] = a;
    for (var layer = 0u; layer < NRC_HIDDEN; layer++) {
        nrc_hidden_layer(layer, &a, &b);
        a = b;
        x[layer + 1u] = a;
    }
    let prediction = nrc_output_layer(&a);
    let wanted = nrc_samples[id.x * 3u + 2u].rgb;
    let luminance = dot(max(prediction, vec3<f32>(0.0)), vec3<f32>(0.2126, 0.7152, 0.0722));
    let gradient = 2.0 * (prediction - wanted) / (luminance * luminance + nrc_train.correction.w);
    var delta: array<vec4<f32>, 16>;
    delta[0] = vec4<f32>(gradient, 0.0);
    for (var j = 0u; j < 16u; j++) {
        nrc_acts[out + NRC_HIDDEN * 16u + j] = x[NRC_HIDDEN][j];
        nrc_deltas[out + NRC_HIDDEN * 16u + j] = delta[j];
    }
    var rows = 3u;
    for (var step = 0u; step < NRC_HIDDEN; step++) {
        // Delta of this layer's outputs to the delta of its inputs.
        let layer = NRC_HIDDEN - step;
        let base = layer * 1024u;
        var back: array<vec4<f32>, 16>;
        for (var i = 0u; i < rows; i++) {
            let d = delta[i >> 2u][i & 3u];
            for (var j = 0u; j < 16u; j++) { back[j] += nrc_weights[base + i * 16u + j] * d; }
        }
        for (var j = 0u; j < 16u; j++) {
            delta[j] = select(vec4<f32>(0.0), back[j], x[layer][j] > vec4<f32>(0.0));
            nrc_acts[out + (layer - 1u) * 16u + j] = x[layer - 1u][j];
            nrc_deltas[out + (layer - 1u) * 16u + j] = delta[j];
        }
        rows = 64u;
    }
}

// One invocation per (chunk, layer, four output rows): sum of delta_i * input_j over the chunk.
@compute @workgroup_size(64)
fn nrc_gradient(@builtin(global_invocation_id) id: vec3<u32>) {
    let per_chunk = NRC_LAYERS * 16u;
    let chunk = id.x / per_chunk;
    if chunk >= NRC_CHUNKS { return; }
    let rest = id.x % per_chunk;
    let layer = rest / 16u;
    let row4 = rest % 16u;
    if layer == NRC_HIDDEN && row4 > 0u { return; }
    var sum: array<array<vec4<f32>, 16>, 4>;
    for (var n = 0u; n < NRC_CHUNK; n++) {
        let at = ((chunk * NRC_CHUNK + n) * NRC_LAYERS + layer) * 16u;
        let d = nrc_deltas[at + row4];
        if all(d == vec4<f32>(0.0)) { continue; }
        for (var j = 0u; j < 16u; j++) {
            let a = nrc_acts[at + j];
            sum[0][j] += a * d.x;
            sum[1][j] += a * d.y;
            sum[2][j] += a * d.z;
            sum[3][j] += a * d.w;
        }
    }
    let rows = select(4u, 3u, layer == NRC_HIDDEN);
    for (var k = 0u; k < rows; k++) {
        let to = chunk * NRC_W4 + layer * 1024u + (row4 * 4u + k) * 16u;
        for (var j = 0u; j < 16u; j++) { nrc_partial[to + j] = sum[k][j]; }
    }
}
