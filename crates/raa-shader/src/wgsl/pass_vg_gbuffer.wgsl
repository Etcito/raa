// RAA virtualized-geometry G-buffer pass (S32): one fullscreen draw per material function
// over the vg.resolve surface. It evaluates the material on the rebuilt MaterialInput and
// writes the S12 normal, material and motion targets plus the resolved standard-Z depth; a
// pixel of another material function or without a virtualized surface is discarded after
// every call.

struct RaaVgGbufferTargets {
    // S12 `normal` (Rgba16Float): view-space shading normal xyz, roughness.
    @location(0) normal: vec4<f32>,
    // S12 `material` (Rgba8Unorm): F0 rgb sRGB-encoded, metallic.
    @location(1) material: vec4<f32>,
    // S12 `motion` (Rg16Float): signed UV displacement, positive X right and Y down.
    @location(2) motion: vec2<f32>,
    // The resolve output's standard-Z depth.
    @builtin(frag_depth) frag_depth: f32,
}

// Fullscreen triangle without a vertex buffer: (-1, -1), (3, -1), (-1, 3) at z 0.
@vertex
fn vs_main(@builtin(vertex_index) raa_vertex_index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((raa_vertex_index << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(raa_vertex_index & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

// material_main followed by the instance tint (S10). mat_instance_tint and
// mat_instance_tint_set are set first so material graph code can read them (S23).
fn raa_pass_material(in: MaterialInput, instance: u32) -> MaterialOutput {
    let x = raa_extras[instance];
    raa_set_instance_tint(x);
    return raa_apply_tint(material_main(in), x);
}

// Normal-incidence reflectance: dielectric from the ior (0.04 at ior 1.5), base colour for
// metals (wgsl/pass_prepass.wgsl).
fn raa_pass_f0(s: MaterialOutput) -> vec3<f32> {
    let ior = clamp(s.ior, 1.0, 3.0);
    let r = (ior - 1.0) / (ior + 1.0);
    return mix(vec3<f32>(r * r), s.base_color, clamp(s.metallic, 0.0, 1.0));
}

// Piecewise sRGB OETF of a value in [0, 1] (wgsl/pass_prepass.wgsl).
fn raa_pass_encode_srgb(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    let lo = x * 12.92;
    let hi = 1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(hi, lo, x <= vec3<f32>(0.0031308));
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> RaaVgGbufferTargets {
    let raa_vg_surface = raa_vg_material_input(frag);
    let in = raa_vg_surface.input;
    let s = raa_pass_material(in, raa_vg_surface.slot);
    let n = normalize((raa_frame.view * vec4<f32>(s.normal, 0.0)).xyz);
    var out: RaaVgGbufferTargets;
    out.normal = vec4<f32>(n, clamp(s.roughness, 0.0, 1.0));
    out.material = vec4<f32>(raa_pass_encode_srgb(raa_pass_f0(s)), clamp(s.metallic, 0.0, 1.0));
    out.motion = raa_vg_surface.motion;
    out.frag_depth = raa_vg_surface.depth;
    if (!raa_vg_surface.covered) {
        discard;
    }
    return out;
}
