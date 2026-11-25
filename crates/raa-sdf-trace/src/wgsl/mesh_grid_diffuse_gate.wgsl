const GRID_DIFFUSE_LEARNED: bool = true;

fn grid_diffuse_gate(features: array<f32, 8>) -> f32 {
    var logit = bitcast<f32>(GRID_GATE_WEIGHTS[80]);
    for (var j = 0u; j < 8u; j++) {
        var activation = bitcast<f32>(GRID_GATE_WEIGHTS[64u + j]);
        for (var k = 0u; k < 8u; k++) {
            activation += features[k] * bitcast<f32>(GRID_GATE_WEIGHTS[8u * j + k]);
        }
        logit += max(0.0, activation) * bitcast<f32>(GRID_GATE_WEIGHTS[72u + j]);
    }
    return exp(clamp(logit, -4.0, 4.0));
}
