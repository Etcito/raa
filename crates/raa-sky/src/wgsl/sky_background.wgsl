// RAA sky background (S12 `sky.background`, forward contributor 900).
//
// Composed after PRELUDE + LIB + SKY_WGSL; groups 0 (frame) and 1 (lighting) are bound as in
// the forward pass. A fullscreen triangle at clip z = 1, drawn with LessEqual and no depth
// write, so it covers only the pixels no geometry wrote. The fragment reconstructs the world
// view ray through inv_proj and inv_view and returns the sky radiance (scene-referred HDR,
// unscaled: the tonemap pass applies exposure).

struct RaaSkyBackgroundVaryings {
    @builtin(position) clip: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32) -> RaaSkyBackgroundVaryings {
    let xy = vec2<f32>(f32((vertex << 1u) & 2u), f32(vertex & 2u)) * 2.0 - vec2<f32>(1.0);
    var out: RaaSkyBackgroundVaryings;
    out.clip = vec4<f32>(xy, 1.0, 1.0);
    return out;
}

// View-space point of an NDC position (w guarded).
fn raa_sky_bg_unproject(ndc: vec3<f32>) -> vec3<f32> {
    let p = raa_frame.inv_proj * vec4<f32>(ndc, 1.0);
    let w = select(p.w, 1e-6, abs(p.w) < 1e-6);
    return p.xyz / w;
}

// World direction of the view ray through the pixel. The ray runs from the near-plane point
// to the mid-depth point, which is right for perspective (also with an infinite far plane,
// where depth 1 does not unproject) and for orthographic projections.
fn raa_sky_bg_world_dir(frag: vec4<f32>) -> vec3<f32> {
    let ndc = vec2<f32>(
        frag.x * raa_frame.viewport.z * 2.0 - 1.0,
        1.0 - frag.y * raa_frame.viewport.w * 2.0,
    );
    let near = raa_sky_bg_unproject(vec3<f32>(ndc, 0.0));
    let mid = raa_sky_bg_unproject(vec3<f32>(ndc, 0.5));
    let world = (raa_frame.inv_view * vec4<f32>(mid - near, 0.0)).xyz;
    let len = length(world);
    if (!(len > 1e-12)) {
        return vec3<f32>(0.0, 0.0, -1.0);
    }
    return world / len;
}

@fragment
fn fs_main(in: RaaSkyBackgroundVaryings) -> @location(0) vec4<f32> {
    let dir = raa_sky_bg_world_dir(in.clip);
    let c = raa_sky_radiance(
        dir,
        raa_env,
        raa_frame.sun_dir.xyz,
        raa_frame.sun_color.rgb,
        raa_frame.camera_pos.w,
    );
    return vec4<f32>(c, 1.0);
}
