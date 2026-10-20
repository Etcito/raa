// raa-material shading (S1). No bindings and no entry points: raa-render owns those.

const RAA_PI: f32 = 3.14159265358979;
const RAA_EPS: f32 = 1e-4;

// One element of the read-only storage buffer `array<MaterialGpu>` (48 B).
struct MaterialGpu {
    // Linear rgb, a = 1.
    base_color: vec4<f32>,
    // Linear rgb, w = 0.
    emissive: vec4<f32>,
    // (roughness clamped [0.04, 1], metallic [0, 1], model: 0 Flat / 1 PbrLite, 0).
    params: vec4<f32>,
}

// Per-frame uniform (144 B).
struct FrameUniform {
    view_proj: mat4x4<f32>,   // @0
    camera_pos: vec4<f32>,    // @64
    light_dir: vec4<f32>,     // @80, normalised travel direction of the light, w = 0
    light_color: vec4<f32>,   // @96, rgb x intensity
    ambient: vec4<f32>,       // @112, rgb x intensity
    flags: vec4<u32>,         // @128, x = 1: encode linear to sRGB in the shader
}

// Shades a surface point. `n` is the unit surface normal, `v` the unit direction from
// the point to the camera. Returns linear rgb.
fn raa_shade(m: MaterialGpu, n: vec3<f32>, v: vec3<f32>, f: FrameUniform) -> vec3<f32> {
    let base = m.base_color.rgb;
    if (m.params.z < 0.5) {
        return base + m.emissive.rgb;
    }
    let roughness = m.params.x;
    let metallic = m.params.y;

    let l = -f.light_dir.xyz;
    let h = normalize(l + v);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

    // GGX normal distribution, alpha = roughness^2.
    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let d_den = n_dot_h * n_dot_h * (alpha2 - 1.0) + 1.0;
    let d = alpha2 / max(RAA_PI * d_den * d_den, RAA_EPS);

    // Schlick Fresnel.
    let f0 = mix(vec3<f32>(0.04), base, metallic);
    let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - v_dot_h, 5.0);

    // Smith-Schlick geometry, k = (roughness + 1)^2 / 8.
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    let g_v = n_dot_v / max(n_dot_v * (1.0 - k) + k, RAA_EPS);
    let g_l = n_dot_l / max(n_dot_l * (1.0 - k) + k, RAA_EPS);
    let g = g_v * g_l;

    let specular = d * g * fresnel / max(4.0 * n_dot_v * n_dot_l, RAA_EPS);
    let diffuse = base / RAA_PI * (1.0 - metallic);

    let ambient = f.ambient.rgb * base * (1.0 - 0.5 * metallic);
    return ambient + (diffuse + specular) * f.light_color.rgb * n_dot_l + m.emissive.rgb;
}

// Piecewise sRGB OETF (linear to sRGB-encoded), per channel. Applied by the forward
// shader when `flags.x == 1`.
fn raa_encode_srgb(c: vec3<f32>) -> vec3<f32> {
    let x = max(c, vec3<f32>(0.0));
    let lo = x * 12.92;
    let hi = 1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(hi, lo, x <= vec3<f32>(0.0031308));
}
