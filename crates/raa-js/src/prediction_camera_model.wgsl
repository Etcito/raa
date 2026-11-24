// The camera-only profile proves a fully covered static view. Fresh repair,
// lighting change, and motion rejection inputs are therefore exactly zero.
@compute @workgroup_size(8, 8)
fn reproject_model_stem(@builtin(global_invocation_id) id: vec3<u32>) {
    let extent = textureDimensions(output_hdr);
    if any(id.xy >= extent) { return; }
    let sample = optical_sample(id.xy, extent);
    textureStore(output_hdr, vec2<i32>(id.xy), sample.hdr);
    textureStore(output_depth, vec2<i32>(id.xy), vec4<f32>(sample.depth, 0.0, 0.0, 0.0));
    // Match the half-float warp quantization seen by the unfused FP32 stem.
    let rgb = vec3<f32>(unpack2x16float(pack2x16float(sample.hdr.rg)),
                        unpack2x16float(pack2x16float(sample.hdr.ba)).x);
    let color = log_color(rgb);
    let depth = clamp(sample.depth, 0.0, 1.0);
    var encoded: array<f32, 12>;
    for (var row = 0u; row < 12u; row++) {
        let offset = row * 14u;
        var value = weights[168u + row];
        value = fma(weights[offset], color.r, value);
        value = fma(weights[offset + 1u], color.g, value);
        value = fma(weights[offset + 2u], color.b, value);
        value = fma(weights[offset + 6u], sample.hdr.a, value);
        value = fma(weights[offset + 8u], depth, value);
        value = fma(weights[offset + 9u], sample.motion.x, value);
        value = fma(weights[offset + 10u], sample.motion.y, value);
        value = fma(weights[offset + 12u], source_age, value);
        encoded[row] = max(0.0, value);
    }
    stem[position(id.xy, extent)] = encoded;
}

@compute @workgroup_size(8, 8)
fn reproject_model_residual(@builtin(global_invocation_id) id: vec3<u32>) {
    let extent = textureDimensions(baseline);
    if any(id.xy >= extent) { return; }
    let existing = textureLoad(baseline, vec2<i32>(id.xy), 0);
    if existing.a <= 0.5 {
        textureStore(output_image, vec2<i32>(id.xy), existing);
        return;
    }
    let rgb = generated_rgb(extent, id.xy, existing.rgb);
    textureStore(output_image, vec2<i32>(id.xy), vec4<f32>(rgb, existing.a));
}
