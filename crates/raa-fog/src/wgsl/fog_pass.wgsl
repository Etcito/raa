// Fog pass (S12 `fog`, order 68): composed as PRELUDE_WGSL + fog_slot.wgsl + this file.
// Group 0 = the LitHandle frame bind group, group 1 = lighting, group 2 private (below).
// Reads hdr.src() and the scene depth, applies `raa_height_apply` per pixel and writes HDR.

// Scene depth (Depth32Float, clip z [0, 1], 1 = sky).
@group(2) @binding(0) var fog_depth: texture_depth_2d;
// hdr.src().
@group(2) @binding(1) var fog_hdr_src: texture_2d<f32>;
// Linear clamp sampler for hdr.src().
@group(2) @binding(2) var fog_samp: sampler;

struct FogVsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Fullscreen triangle (3 vertices); uv (0, 0) top-left.
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> FogVsOut {
    let p = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: FogVsOut;
    out.position = vec4<f32>(p * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
    out.uv = vec2<f32>(p.x, 1.0 - p.y);
    return out;
}

// World position of target coordinate `uv` at clip depth `z` (inv_proj then inv_view).
fn fog_unproject(uv: vec2<f32>, z: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, z, 1.0);
    let v = raa_frame.inv_proj * ndc;
    let view_pos = v.xyz / v.w;
    return (raa_frame.inv_view * vec4<f32>(view_pos, 1.0)).xyz;
}

@fragment
fn fs_main(in: FogVsOut) -> @location(0) vec4<f32> {
    let src = textureSampleLevel(fog_hdr_src, fog_samp, in.uv, 0.0);
    let dims = textureDimensions(fog_depth);
    let texel = min(vec2<u32>(in.position.xy), dims - vec2<u32>(1u));
    let depth = textureLoad(fog_depth, texel, 0);
    let camera = raa_frame.camera_pos.xyz;
    let sky = depth >= 1.0;
    var world: vec3<f32>;
    if (sky) {
        // Sky: only the view direction matters; take it through the near plane.
        let near = fog_unproject(in.uv, 0.0);
        world = camera + normalize(near - camera);
    } else {
        world = fog_unproject(in.uv, depth);
    }
    return vec4<f32>(raa_height_apply(src.rgb, world, camera, sky), src.a);
}
