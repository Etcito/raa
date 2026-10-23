// Source sky cube mip k from mip k-1 by 2x2 box average (raa-ibl filter, one dispatch per
// source mip 1..5). Standalone compute module (no PRELUDE), own bindings in group 0.
// Workgroup 8x8x1 over the destination texels; global invocation z = face (array layer,
// order +X,-X,+Y,-Y,+Z,-Z). Dispatch (ceil(size / 8), ceil(size / 8), 6).

// Mip k-1 of the source cube viewed as a 6-layer 2d array (one mip, read with textureLoad).
@group(0) @binding(0) var src_mip: texture_2d_array<f32>;
// Mip k of the source cube viewed as a 6-layer 2d array (one mip).
@group(0) @binding(1) var dst_mip: texture_storage_2d_array<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn downsample(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(dst_mip);
    if (id.x >= size.x || id.y >= size.y || id.z >= textureNumLayers(dst_mip)) {
        return;
    }
    let layer = id.z;
    // Clamped so an odd source size never reads past its last texel.
    let last = vec2<i32>(textureDimensions(src_mip)) - vec2<i32>(1);
    let p = vec2<i32>(id.xy) * 2;
    let c = textureLoad(src_mip, min(p, last), layer, 0)
        + textureLoad(src_mip, min(p + vec2<i32>(1, 0), last), layer, 0)
        + textureLoad(src_mip, min(p + vec2<i32>(0, 1), last), layer, 0)
        + textureLoad(src_mip, min(p + vec2<i32>(1, 1), last), layer, 0);
    textureStore(dst_mip, vec2<i32>(id.xy), layer, c * 0.25);
}
