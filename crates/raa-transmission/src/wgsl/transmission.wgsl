// Transmission slot provider `screen` of raa-transmission (S11): light refracted through
// the surface is read from the copy of the opaque HDR colour and depth (group 1 bindings 9
// and 10), absorbed by Beer-Lambert over the path in the medium and weighted by the Fresnel
// transmittance. Every formula matches the CPU reference `src/absorption.rs`; private
// helpers are prefixed `raa_screen_`.

// Attenuation distance meaning "no absorption" (f32::INFINITY is written as 1e30).
const RAA_SCREEN_NO_ABSORPTION: f32 = 1e30;
// Smallest colour component accepted before the logarithm.
const RAA_SCREEN_MIN_ATTENUATION: f32 = 1e-4;
// Lower bound of the cosine used for the path length.
const RAA_SCREEN_MIN_COS: f32 = 0.2;

// Absorption coefficient per channel: -ln(colour) / distance; a component <= 0 is clamped
// to 1e-4 first, a distance of 1e30 (or <= 0) absorbs nothing.
fn raa_screen_sigma(color: vec3<f32>, distance: f32) -> vec3<f32> {
    if (distance >= RAA_SCREEN_NO_ABSORPTION || distance <= 0.0) {
        return vec3<f32>(0.0);
    }
    let c = select(color, vec3<f32>(RAA_SCREEN_MIN_ATTENUATION), color <= vec3<f32>(0.0));
    return -log(c) / distance;
}

// Path through the medium: thickness / max(dot(-dir, n), 0.2).
fn raa_screen_path_length(thickness: f32, dir: vec3<f32>, n: vec3<f32>) -> f32 {
    return thickness / max(dot(-dir, n), RAA_SCREEN_MIN_COS);
}

// WGSL refract(i, n, eta), except that total internal reflection returns reflect(i, n)
// instead of the zero vector.
fn raa_screen_refract(i: vec3<f32>, n: vec3<f32>, eta: f32) -> vec3<f32> {
    let d = dot(n, i);
    let k = 1.0 - eta * eta * (1.0 - d * d);
    if (k < 0.0) {
        return i - 2.0 * d * n;
    }
    return eta * i - (eta * d + sqrt(k)) * n;
}

// Opaque-colour mip: log2(max(1, roughness^2 x 0.5 x viewport_h x thickness / view_depth)).
fn raa_screen_lod(roughness: f32, viewport_h: f32, thickness: f32, view_depth: f32) -> f32 {
    let x = roughness * roughness * 0.5 * viewport_h * thickness / max(view_depth, 1e-6);
    return log2(max(x, 1.0));
}

// Schlick Fresnel per channel: f0 + (1 - f0) (1 - cos)^5.
fn raa_screen_fresnel(f0: vec3<f32>, cos_theta: f32) -> vec3<f32> {
    let m = pow(1.0 - clamp(cos_theta, 0.0, 1.0), 5.0);
    return f0 + (vec3<f32>(1.0) - f0) * m;
}

// Screen uv of the opaque copy where the refracted ray leaves the medium. The exit point
// is position + dir x thickness: the material thickness is in world metres and the
// instance scale is approximated by 1. When the exit point is behind the camera, or the
// opaque depth there is in front of the surface itself (the ray would pick up an object
// that occludes the surface), the unrefracted fragment position is used.
fn raa_screen_uv(c: ShadeCtx, dir: vec3<f32>, thickness: f32) -> vec2<f32> {
    let straight = c.frag.xy * raa_frame.viewport.zw;
    let exit = c.position + dir * thickness;
    let clip = raa_frame.view_proj * vec4<f32>(exit, 1.0);
    if (clip.w <= 1e-6) {
        return straight;
    }
    let ndc = clip.xy / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    // Pixel of the depth copy; clamped so the 1x1 dummy (depth 1) is always read.
    let size = textureDimensions(raa_opaque_depth);
    let px = vec2<u32>(clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * vec2<f32>(size));
    let depth = textureLoad(raa_opaque_depth, min(px, size - vec2<u32>(1u)), 0).r;
    return select(uv, straight, depth < c.frag.z);
}

// transmitted x T x (1 - F) x transmission, T = exp(-sigma x path) along the refracted
// view ray, F Schlick with the surface f0 at n.v.
fn raa_slot_transmission(s: MaterialOutput, c: ShadeCtx) -> vec3<f32> {
    let n = s.normal;
    let dir = raa_screen_refract(-c.view_dir, n, 1.0 / s.ior);
    let uv = raa_screen_uv(c, dir, s.thickness);
    let view_depth = -(raa_frame.view * vec4<f32>(c.position, 1.0)).z;
    let lod = raa_screen_lod(s.roughness, raa_frame.viewport.y, s.thickness, view_depth);
    let max_lod = f32(textureNumLevels(raa_opaque_color) - 1u);
    let transmitted = textureSampleLevel(raa_opaque_color, raa_linear_clamp, uv, min(lod, max_lod)).rgb;
    let sigma = raa_screen_sigma(s.attenuation_color, s.attenuation_distance);
    let t = exp(-sigma * raa_screen_path_length(s.thickness, dir, n));
    let f = raa_screen_fresnel(c.f0, c.n_dot_v);
    return transmitted * t * (vec3<f32>(1.0) - f) * s.transmission;
}
