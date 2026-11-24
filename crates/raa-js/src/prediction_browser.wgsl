// Browser target samples are exact scene-linear HDR supplied at cutoff.
// Alpha one certifies a sample; missing/invalid samples reject the candidate.
@group(0) @binding(0) var<storage, read> work: RaaPredictionRefreshWork;
@group(0) @binding(1) var samples: texture_2d<f32>;
@group(0) @binding(2) var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(64)
fn refresh(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= raa_prediction_refresh_count(work.count, arrayLength(&work.pixels)) {
        return;
    }
    let xy = raa_prediction_refresh_pixel(work.pixels[id.x], textureDimensions(samples).x);
    textureStore(output, xy, textureLoad(samples, xy, 0));
}
