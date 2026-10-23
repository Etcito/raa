// RAA sky env pass (S12 `sky.env`): renders one face of the SKY_ENV cube per draw.
//
// Composed after PRELUDE + LIB + SKY_WGSL; groups 0 (frame) and 1 (lighting) are bound as in
// the forward pass. The pass draws vertices 0..3 with the instance range face..face+1, so
// `instance_index` is the cube face in wgpu order +X, -X, +Y, -Y, +Z, -Z. Target: HDR_FORMAT.

struct RaaSkyEnvVaryings {
    @builtin(position) clip: vec4<f32>,
    // Face coordinates in [-1, 1]: u grows with the texel column, v with the texel row.
    @location(0) face_uv: vec2<f32>,
    @location(1) @interpolate(flat, either) face: u32,
}

// Unit direction of face coordinates (u, v); same mapping as raa-ibl's SH projection
// (crates/raa-ibl/src/sh.rs) and the CPU port in crates/raa-sky/src/shaders.rs.
fn raa_sky_env_face_dir(face: u32, uv: vec2<f32>) -> vec3<f32> {
    let u = uv.x;
    let v = uv.y;
    var d: vec3<f32>;
    switch face {
        case 0u: { d = vec3<f32>(1.0, -v, -u); }
        case 1u: { d = vec3<f32>(-1.0, -v, u); }
        case 2u: { d = vec3<f32>(u, 1.0, v); }
        case 3u: { d = vec3<f32>(u, -1.0, -v); }
        case 4u: { d = vec3<f32>(u, -v, 1.0); }
        default: { d = vec3<f32>(-u, -v, -1.0); }
    }
    // |d| >= 1 by construction (one component is +-1).
    return normalize(d);
}

// Fullscreen triangle covering the face.
@vertex
fn vs_main(
    @builtin(vertex_index) vertex: u32,
    @builtin(instance_index) instance: u32,
) -> RaaSkyEnvVaryings {
    let xy = vec2<f32>(f32((vertex << 1u) & 2u), f32(vertex & 2u)) * 2.0 - vec2<f32>(1.0);
    var out: RaaSkyEnvVaryings;
    out.clip = vec4<f32>(xy, 0.0, 1.0);
    // Clip y points up, texel rows go down.
    out.face_uv = vec2<f32>(xy.x, -xy.y);
    out.face = instance;
    return out;
}

@fragment
fn fs_main(in: RaaSkyEnvVaryings) -> @location(0) vec4<f32> {
    let dir = raa_sky_env_face_dir(in.face, in.face_uv);
    let c = raa_sky_radiance(
        dir,
        raa_env,
        raa_frame.sun_dir.xyz,
        raa_frame.sun_color.rgb,
        raa_frame.camera_pos.w,
    );
    return vec4<f32>(c, 1.0);
}
