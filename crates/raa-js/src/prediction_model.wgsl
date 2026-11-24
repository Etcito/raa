// raa.generated-frame-model/1, FP32 NHWC. Warp and repair validity are
// established by the analytic recorder; this decoder cannot create validity.
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read_write> stem: array<array<f32, 12>>;
@group(0) @binding(2) var warp_image: texture_2d<f32>;
@group(0) @binding(3) var fresh_image: texture_2d<f32>;
@group(0) @binding(4) var depth_image: texture_2d<f32>;
@group(0) @binding(5) var motion_image: texture_2d<f32>;
@group(0) @binding(6) var lighting_image: texture_2d<u32>;
@group(0) @binding(7) var<uniform> source_age: f32;
@group(0) @binding(8) var baseline: texture_2d<f32>;
@group(0) @binding(9) var output_image: texture_storage_2d<rgba16float, write>;

fn log_color(rgb: vec3<f32>) -> vec3<f32> { return log(1.0 + max(rgb, vec3<f32>(0.0))); }
fn position(p: vec2<u32>, size: vec2<u32>) -> u32 { return p.y * size.x + p.x; }

@compute @workgroup_size(8, 8)
fn generated_stem(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(warp_image);
    if any(id.xy >= size) { return; }
    let p = vec2<i32>(id.xy);
    let warp = textureLoad(warp_image, p, 0);
    let fresh = textureLoad(fresh_image, min(p, vec2<i32>(textureDimensions(fresh_image)) - 1), 0);
    let motion = textureLoad(motion_image, p, 0);
    let depth = textureLoad(depth_image, p, 0).x;
    let changed = textureLoad(lighting_image, min(p, vec2<i32>(textureDimensions(lighting_image)) - 1), 0).x;
    let vw = warp.a > 0.5;
    let vf = fresh.a > 0.5;
    let w = log_color(select(vec3<f32>(0.0), warp.rgb, vw));
    let f = log_color(select(vec3<f32>(0.0), fresh.rgb, vf));
    let channels = array<f32, 14>(w.r, w.g, w.b, f.r, f.g, f.b, f32(vw), f32(vf),
        clamp(depth, 0.0, 1.0), motion.x, motion.y, f32(changed != 0u), source_age, motion.z);
    stem[position(id.xy, size)] = generated_encode(channels);
}

fn generated_encode(channels: array<f32, 14>) -> array<f32, 12> {
    var encoded: array<f32, 12>;
    for (var row = 0u; row < 12u; row++) {
        var value = weights[168u + row];
        for (var c = 0u; c < 14u; c++) {
            value = fma(weights[row * 14u + c], channels[c], value);
        }
        encoded[row] = max(0.0, value);
    }
    return encoded;
}

@compute @workgroup_size(8, 8)
fn generated_residual(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(warp_image);
    if any(id.xy >= size) { return; }
    let p = vec2<i32>(id.xy);
    let existing = textureLoad(baseline, p, 0);
    if textureLoad(warp_image, p, 0).a <= 0.5 || textureLoad(fresh_image, min(p, vec2<i32>(textureDimensions(fresh_image)) - 1), 0).a > 0.5 {
        textureStore(output_image, p, existing);
        return;
    }
    textureStore(output_image, p, vec4<f32>(generated_rgb(size, id.xy, existing.rgb), existing.a));
}

fn generated_rgb(size: vec2<u32>, xy: vec2<u32>, existing: vec3<f32>) -> vec3<f32> {
    let p = vec2<i32>(xy);
    var spatial: array<f32, 12>;
    for (var c = 0u; c < 12u; c++) { spatial[c] = weights[288u + c]; }
    for (var tap = 0u; tap < 9u; tap++) {
        let q = clamp(p + vec2<i32>(i32(tap % 3u), i32(tap / 3u)) - 1, vec2<i32>(0), vec2<i32>(size) - 1);
        let loaded = stem[position(vec2<u32>(q), size)];
        for (var c = 0u; c < 12u; c++) {
            spatial[c] = fma(weights[180u + c * 9u + tap], loaded[c], spatial[c]);
        }
    }
    let center = stem[position(xy, size)];
    var features: array<f32, 12>;
    for (var row = 0u; row < 12u; row++) {
        var value = weights[444u + row];
        for (var c = 0u; c < 12u; c++) {
            value = fma(weights[300u + row * 12u + c], spatial[c], value);
        }
        features[row] = max(0.0, center[row] + value);
    }
    let base = log_color(existing);
    var rgb: vec3<f32>;
    for (var row = 0u; row < 3u; row++) {
        var value = weights[492u + row];
        for (var c = 0u; c < 12u; c++) {
            value = fma(weights[456u + row * 12u + c], features[c], value);
        }
        rgb[row] = clamp(exp(base[row] + 0.25 * tanh(value)) - 1.0, 0.0, 65504.0);
    }
    return select(existing, rgb, all(abs(rgb) <= vec3<f32>(65504.0)));
}
