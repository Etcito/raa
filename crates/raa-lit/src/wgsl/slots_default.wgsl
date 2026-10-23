// Default slot providers of raa-lit (provider `raa.default`, priority 0, S11 defaults).
// One section per slot, each starting with a `// @slot <name>` line; a section is composed
// alone (another provider may win the other slots), so it carries its own private helpers,
// all prefixed `raa_default_`.

// @slot deform
// Identity: the local position and normal pass through unchanged.
fn raa_slot_deform(p: vec3<f32>, n: vec3<f32>, x: InstanceExtraGpu, t: f32) -> VertexMaterialOutput {
    return VertexMaterialOutput(p, n);
}

// @slot shadow
// The sun is fully visible.
fn raa_slot_shadow(world_pos: vec3<f32>, n: vec3<f32>, frag: vec4<f32>) -> f32 {
    return 1.0;
}

// @slot ao
// No screen-space occlusion.
fn raa_slot_ao(frag: vec4<f32>) -> f32 {
    return 1.0;
}

// @slot ambient
// S1 PbrLite ambient term: ambient x base x (1 - 0.5 metallic).
fn raa_slot_ambient(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> {
    return raa_frame.ambient.rgb * s.base_color * (1.0 - 0.5 * s.metallic);
}

// @slot direct
// S1 PbrLite direct term: Lambert x (1 - metallic) + GGX specular with Schlick Fresnel
// (F0 = mix(0.04, base, metallic)) and Smith-Schlick G (k = (roughness + 1)^2 / 8).
fn raa_default_ggx_d(n_dot_h: f32, roughness: f32) -> f32 {
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let d_den = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
    return alpha2 / max(3.14159265 * d_den * d_den, 1e-6);
}

fn raa_default_f0(s: MaterialOutput) -> vec3<f32> {
    return mix(vec3<f32>(0.04), s.base_color, s.metallic);
}

fn raa_default_fresnel_schlick(f0: vec3<f32>, v_dot_h: f32) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - v_dot_h, 5.0);
}

fn raa_default_smith_schlick_g(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    let g_v = n_dot_v / max(n_dot_v * (1.0 - k) + k, 1e-6);
    let g_l = n_dot_l / max(n_dot_l * (1.0 - k) + k, 1e-6);
    return g_v * g_l;
}

fn raa_slot_direct(s: MaterialOutput, c: ShadeCtx, l: vec3<f32>, radiance: vec3<f32>) -> vec3<f32> {
    let n = s.normal;
    let v = c.view_dir;
    let h = normalize(l + v);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(c.n_dot_v, 0.0);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);
    let d = raa_default_ggx_d(n_dot_h, s.roughness);
    let fresnel = raa_default_fresnel_schlick(raa_default_f0(s), v_dot_h);
    let g = raa_default_smith_schlick_g(n_dot_v, n_dot_l, s.roughness);
    let specular = d * g * fresnel / max(4.0 * n_dot_v * n_dot_l, 1e-6);
    let diffuse = s.base_color / 3.14159265 * (1.0 - s.metallic);
    return (diffuse + specular) * radiance * n_dot_l;
}

// @slot transmission
// Nothing transmitted.
fn raa_slot_transmission(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> {
    return vec3<f32>(0.0);
}

// @slot fog
// Identity: no fog.
fn raa_slot_fog(color: vec3<f32>, world_pos: vec3<f32>, c: ShadeCtx) -> vec3<f32> {
    return color;
}
