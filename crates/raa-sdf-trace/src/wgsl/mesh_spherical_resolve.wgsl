// Screen resolve of the spherical lookup at 1/RAA_SPHERICAL_RESOLVE_SCALE resolution: one
// texel per block evaluates the full probe lookup at the G-buffer surface (prepass depth and
// view-space normal) and stores irradiance / pi, the blend weight and the camera distance in
// the resolve rows (header row 3: z first row, w width | height << 16). Forward only blends
// the four nearest texels (raa_spherical_screen), so the lookup's registers and loads leave
// the forward shader and run on a quarter of the pixels.

fn spherical_resolve_world(uv: vec2<f32>, z: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, z, 1.0);
    let p = frame.inv_view * frame.inv_proj * ndc;
    return p.xyz / p.w;
}

@compute @workgroup_size(8, 8, 1)
fn spherical_resolve(@builtin(global_invocation_id) id: vec3<u32>) {
    let screen = raa_spherical_data[3];
    let size = vec2<u32>(screen.w & 0xffffu, screen.w >> 16u);
    if screen.z == 0u || id.x >= size.x || id.y >= size.y { return; }
    let row = screen.z + id.y * size.x + id.x;
    let full = textureDimensions(depth);
    let pixel = min(id.xy * RAA_SPHERICAL_RESOLVE_SCALE, full - 1u);
    let z = textureLoad(depth, vec2<i32>(pixel), 0);
    let normal_view = textureLoad(normals, vec2<i32>(pixel), 0).xyz;
    if z >= 1.0 || dot(normal_view, normal_view) <= 0.0 {
        raa_spherical_data[row] = vec4<u32>(0u);
        return;
    }
    let p = spherical_resolve_world((vec2<f32>(pixel) + 0.5) / vec2<f32>(full), z);
    let n = normalize((frame.inv_view * vec4<f32>(normal_view, 0.0)).xyz);
    let distance = length(frame.camera_pos.xyz - p);
    let result = raa_spherical_evaluate(p, n);
    // xyz: rgb (f32); w: camera distance (f32 with the low 12 mantissa bits dropped),
    // blend weight in 10 bits, flags (bit 0 resolved, bit 1 diagnostic colour instead).
    // No f16 packing: the lookup validates without the float16-in-float32 capability.
    var value = result.diffuse;
    var flags = 1u;
    if result.debug.w > 0.0 {
        value = result.debug;
        flags = 3u;
    }
    let rgb = clamp(value.rgb, vec3<f32>(0.0), vec3<f32>(65504.0));
    let blend = u32(round(clamp(value.w, 0.0, 1.0) * 1023.0));
    raa_spherical_data[row] = vec4<u32>(bitcast<vec3<u32>>(rgb),
        (bitcast<u32>(distance) & 0xfffff000u) | (blend << 2u) | flags);
}
