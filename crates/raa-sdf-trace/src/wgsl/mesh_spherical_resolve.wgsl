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
    let trace = raa_spherical_data[6].w;
    raa_spherical_tracing = trace != 0u && id.x + 1u == (trace & 0xffffu) && id.y + 1u == (trace >> 16u);
    let distance = length(frame.camera_pos.xyz - p);
    let result = raa_spherical_evaluate(p, n);
    // xyz: rgb (f32); w: camera distance (f32 top 16 bits), blend weight (8 bits),
    // probe sun visibility (6 bits), flags (bit 0 resolved, bit 1 diagnostic colour).
    // No f16 packing: the lookup validates without the float16-in-float32 capability.
    var value = result.diffuse;
    var flags = 1u;
    var sun = 1.0;
    if result.debug.w > 0.0 {
        value = result.debug;
        flags = 3u;
    } else if raa_spherical_direct_range() >= 0.0 {
        let direct = spherical_resolve_direct(p, n);
        value = vec4<f32>(value.rgb + direct.rgb, value.w);
        sun = direct.w;
    }
    let rgb = clamp(value.rgb, vec3<f32>(0.0), vec3<f32>(65504.0));
    let blend = u32(round(clamp(value.w, 0.0, 1.0) * 255.0));
    let visibility = u32(round(clamp(sun, 0.0, 1.0) * 63.0));
    raa_spherical_data[row] = vec4<u32>(bitcast<vec3<u32>>(rgb),
        (bitcast<u32>(distance) & 0xffff0000u) | (blend << 8u) | (visibility << 2u) | flags);
}

// Sphere-only direct: a local light's diffuse irradiance / pi at the receiver (exact
// falloff, range window and cone; panels as points at their centre).
fn spherical_resolve_light(index: u32, p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let light = lights[index];
    var l = -light.direction_outer.xyz;
    var weight = 1.0;
    if light.kind != 2u {
        let delta = light.position_range.xyz - p;
        let d = length(delta);
        if d >= light.position_range.w { return vec3<f32>(0.0); }
        l = delta / max(d, 1e-4);
        let window = saturate(1.0 - pow(d / light.position_range.w, 4.0));
        weight = window * window / max(d * d, 0.04);
        if light.kind == 1u {
            weight *= smoothstep(light.direction_outer.w, light.inner, dot(light.direction_outer.xyz, -l));
        }
    }
    return light.radiance.rgb * (weight * max(dot(n, l), 0.0) * 0.31830988);
}

// rgb: the local lights of the two strongest probes, each shadowed by the probes'
// visibility blended with the same soft top-2 weights as the irradiance (a light a probe
// did not list counts as invisible there, so lights fade rather than pop); w: their sun
// visibility.
fn spherical_resolve_direct(p: vec3<f32>, n: vec3<f32>) -> vec4<f32> {
    let cell = raa_spherical_coordinate(p);
    if cell.desc.x == 0u { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    let top = raa_spherical_top(p, n, cell);
    let a = top.w0 - top.w2;
    let b = max(top.w1 - top.w2, 0.0);
    if a <= 1e-7 { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    let mask = cell.desc.z;
    let r0 = raa_spherical_data[raa_spherical_slot(top.e0, mask) + RAA_SPHERICAL_LIGHT_ROW];
    var r1 = vec4<u32>(0u, 0xffffffffu, 0u, 0u);
    if b > 1e-7 { r1 = raa_spherical_data[raa_spherical_slot(top.e1, mask) + RAA_SPHERICAL_LIGHT_ROW]; }
    let norm = 1.0 / (a + b);
    let ids = array<u32, 4>(r0.y & 0xffffu, r0.y >> 16u, r1.y & 0xffffu, r1.y >> 16u);
    let vis = array<f32, 4>(bitcast<f32>(r0.z), bitcast<f32>(r0.w), bitcast<f32>(r1.z), bitcast<f32>(r1.w));
    var rgb = vec3<f32>(0.0);
    for (var k = 0u; k < 4u; k++) {
        let id = ids[k];
        if id == 0xffffu { continue; }
        // Each light once: the second probe's entries skip lights the first one listed.
        if k >= 2u && (id == ids[0] || id == ids[1]) { continue; }
        var v = 0.0;
        if ids[0] == id { v += a * vis[0]; }
        if ids[1] == id { v += a * vis[1]; }
        if ids[2] == id { v += b * vis[2]; }
        if ids[3] == id { v += b * vis[3]; }
        rgb += spherical_resolve_light(id, p, n) * (v * norm);
    }
    let sun = (a * bitcast<f32>(r0.x) + b * bitcast<f32>(r1.x)) * norm;
    return vec4<f32>(rgb, sun);
}
