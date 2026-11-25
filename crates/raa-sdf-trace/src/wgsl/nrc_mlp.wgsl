// Neural radiance cache network (Müller et al. 2021), fully fused: one invocation evaluates the
// whole MLP. 64 inputs, NRC_HIDDEN ReLU layers of 64, 3 linear outputs, no bias, FP32.
// The includer declares `nrc_weights: array<vec4<f32>>` (row-major, 16 vec4 per output row) and
// the constants NRC_HIDDEN and NRC_FREQS.

fn nrc_tri(x: f32) -> f32 { return abs(fract(x) * 2.0 - 1.0) * 2.0 - 1.0; }

// One-blob encoding (k = 4) of a value in [-1, 1].
fn nrc_blob4(v: f32) -> vec4<f32> {
    let d = (vec4<f32>(v * 0.5 + 0.5) - vec4<f32>(0.125, 0.375, 0.625, 0.875)) * 4.0;
    return exp(-0.5 * d * d);
}

// Position in [0, 1]^3 (triangle-wave frequencies 2^0..2^(NRC_FREQS-1) per axis), unit normal
// (one-blob per component), then the raw normal, the raw position and constants.
fn nrc_encode(p: vec3<f32>, n: vec3<f32>, x: ptr<function, array<vec4<f32>, 16>>) {
    for (var axis = 0u; axis < 3u; axis++) {
        let v = p[axis];
        for (var k = 0u; k < 3u; k++) {
            var lane = vec4<f32>(0.0);
            for (var c = 0u; c < 4u; c++) {
                let f = k * 4u + c;
                if f < NRC_FREQS { lane[c] = nrc_tri(v * f32(1u << f)); }
            }
            (*x)[axis * 3u + k] = lane;
        }
    }
    (*x)[9] = nrc_blob4(n.x);
    (*x)[10] = nrc_blob4(n.y);
    (*x)[11] = nrc_blob4(n.z);
    (*x)[12] = vec4<f32>(n, 1.0);
    (*x)[13] = vec4<f32>(p * 2.0 - 1.0, 1.0);
    (*x)[14] = vec4<f32>(0.0);
    (*x)[15] = vec4<f32>(0.0);
}

fn nrc_hidden_layer(layer: u32, a: ptr<function, array<vec4<f32>, 16>>,
                    o: ptr<function, array<vec4<f32>, 16>>) {
    let base = layer * 1024u;
    for (var i4 = 0u; i4 < 16u; i4++) {
        var r: vec4<f32>;
        for (var k = 0u; k < 4u; k++) {
            let row = base + (i4 * 4u + k) * 16u;
            var sum = 0.0;
            for (var j = 0u; j < 16u; j++) { sum += dot(nrc_weights[row + j], (*a)[j]); }
            r[k] = sum;
        }
        (*o)[i4] = max(r, vec4<f32>(0.0));
    }
}

fn nrc_output_layer(a: ptr<function, array<vec4<f32>, 16>>) -> vec3<f32> {
    let base = NRC_HIDDEN * 1024u;
    var r: vec3<f32>;
    for (var k = 0u; k < 3u; k++) {
        var sum = 0.0;
        for (var j = 0u; j < 16u; j++) { sum += dot(nrc_weights[base + k * 16u + j], (*a)[j]); }
        r[k] = sum;
    }
    return r;
}

// Raw network output for a normalized position and a unit normal.
fn nrc_query(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    var a: array<vec4<f32>, 16>;
    var b: array<vec4<f32>, 16>;
    nrc_encode(p, n, &a);
    for (var layer = 0u; layer < NRC_HIDDEN; layer++) {
        if (layer & 1u) == 0u { nrc_hidden_layer(layer, &a, &b); } else { nrc_hidden_layer(layer, &b, &a); }
    }
    if (NRC_HIDDEN & 1u) == 1u { return nrc_output_layer(&b); }
    return nrc_output_layer(&a);
}
