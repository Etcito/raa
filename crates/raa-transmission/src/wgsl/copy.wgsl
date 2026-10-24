// The `opaque.copy` pass (S12, order 70): the mip chain of the opaque HDR colour copy by
// 2x2 filtered downsample (`fs_downsample`) and the opaque depth copy into R32Float
// (`fs_depth`). Mip 0 is a copy_texture_to_texture of hdr.src(), not a draw. Standalone: own
// bindings in group 0, no PRELUDE.

// fs_downsample: the previous mip (a single-mip view), linear filtering, clamp to edge.
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_samp: sampler;
// fs_depth: the single-sample prepass depth target.
@group(0) @binding(2) var depth_tex: texture_depth_2d;

struct VsOut {
    @builtin(position) position: vec4<f32>,
}

// Fullscreen triangle (3 vertices, no buffers).
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    let p = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: VsOut;
    out.position = vec4<f32>(p * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
    return out;
}

// Target texel p of mip k averages the 2x2 block 2p..2p+1 of mip k - 1 with one bilinear
// sample at the block's centre, (2p + 1) / size of mip k - 1.
@fragment
fn fs_downsample(in: VsOut) -> @location(0) vec4<f32> {
    let src_size = vec2<f32>(textureDimensions(src_tex, 0u));
    let p = floor(in.position.xy);
    let uv = (2.0 * p + vec2<f32>(1.0)) / src_size;
    return textureSampleLevel(src_tex, src_samp, uv, 0.0);
}

// Device depth of the pixel, unchanged (the copy has the depth target's size).
@fragment
fn fs_depth(in: VsOut) -> @location(0) f32 {
    return textureLoad(depth_tex, vec2<i32>(floor(in.position.xy)), 0);
}
