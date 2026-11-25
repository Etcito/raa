// Neural radiance cache optimizer: reduces the per-chunk gradients, applies Adam and keeps the
// exponential moving average of the weights that rendering reads. Constants are prepended.

struct NrcTrain {
    adam: vec4<f32>,
    correction: vec4<f32>,
}
@group(0) @binding(0) var<uniform> nrc_train: NrcTrain;
@group(0) @binding(1) var<storage, read_write> nrc_trained: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> nrc_average: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> nrc_moment1: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> nrc_moment2: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> nrc_partial: array<vec4<f32>>;

@compute @workgroup_size(64)
fn nrc_adam(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= NRC_W4 { return; }
    var g = vec4<f32>(0.0);
    for (var chunk = 0u; chunk < NRC_CHUNKS; chunk++) { g += nrc_partial[chunk * NRC_W4 + id.x]; }
    g /= f32(NRC_BATCH);
    // A non-finite gradient (degenerate sample) must not poison the cache.
    if any(abs(g) > vec4<f32>(1e20)) || any(g != g) { return; }
    let m = mix(g, nrc_moment1[id.x], nrc_train.adam.y);
    let v = mix(g * g, nrc_moment2[id.x], nrc_train.adam.z);
    nrc_moment1[id.x] = m;
    nrc_moment2[id.x] = v;
    let w = nrc_trained[id.x] - nrc_train.adam.x * (m * nrc_train.correction.x)
        / (sqrt(v * nrc_train.correction.y) + vec4<f32>(nrc_train.adam.w));
    nrc_trained[id.x] = w;
    nrc_average[id.x] = mix(w, nrc_average[id.x], nrc_train.correction.z);
}
