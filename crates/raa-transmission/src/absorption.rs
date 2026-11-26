//! CPU reference of the transmission formulas (same as the WGSL slot).

/// Attenuation distance meaning "no absorption" (f32::INFINITY is written as 1e30).
pub const NO_ABSORPTION_DISTANCE: f32 = 1e30;

/// Smallest colour component accepted before the logarithm.
pub const MIN_ATTENUATION: f32 = 1e-4;

/// Lower bound of the cosine used for the path length.
pub const MIN_COS: f32 = 0.2;

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Absorption coefficient per channel: -ln(colour) / distance.
pub fn sigma(attenuation_color: [f32; 3], attenuation_distance: f32) -> [f32; 3] {
    if attenuation_distance >= NO_ABSORPTION_DISTANCE || attenuation_distance <= 0.0 {
        return [0.0; 3];
    }
    attenuation_color.map(|c| {
        let c = if c <= 0.0 { MIN_ATTENUATION } else { c };
        -c.ln() / attenuation_distance
    })
}

/// Path through the medium: thickness / max(dot(-dir, n), 0.2).
pub fn path_length(thickness: f32, dir: [f32; 3], n: [f32; 3]) -> f32 {
    let c = -dot(dir, n);
    thickness / c.max(MIN_COS)
}

/// Beer-Lambert transmittance exp(-sigma x path).
pub fn transmittance(sigma: [f32; 3], path: f32) -> [f32; 3] {
    sigma.map(|s| (-s * path).exp())
}

/// Transmittance from material parameters along a refracted direction.
pub fn transmittance_for(
    attenuation_color: [f32; 3],
    attenuation_distance: f32,
    thickness: f32,
    dir: [f32; 3],
    n: [f32; 3],
) -> [f32; 3] {
    transmittance(
        sigma(attenuation_color, attenuation_distance),
        path_length(thickness, dir, n),
    )
}

/// WGSL `refract(i, n, eta)`; total internal reflection returns `reflect(i, n)`.
pub fn refract(i: [f32; 3], n: [f32; 3], eta: f32) -> [f32; 3] {
    let d = dot(n, i);
    let k = 1.0 - eta * eta * (1.0 - d * d);
    if k < 0.0 {
        return [0, 1, 2].map(|j| i[j] - 2.0 * d * n[j]);
    }
    let s = eta * d + k.sqrt();
    [0, 1, 2].map(|j| eta * i[j] - s * n[j])
}

/// Refracted view ray: refract(-v, n, 1 / ior).
pub fn refract_view(v: [f32; 3], n: [f32; 3], ior: f32) -> [f32; 3] {
    refract(v.map(|x| -x), n, 1.0 / ior)
}

/// Opaque-colour mip: log2(max(1, roughness^2 x 0.5 x viewport_h x thickness / view_depth)).
pub fn transmission_lod(roughness: f32, viewport_h: f32, thickness: f32, view_depth: f32) -> f32 {
    let x = roughness * roughness * 0.5 * viewport_h * thickness / view_depth.max(1e-6);
    x.max(1.0).log2()
}

/// Schlick Fresnel: f0 + (1 - f0) (1 - cos)^5.
pub fn schlick_fresnel(f0: f32, cos_theta: f32) -> f32 {
    let m = (1.0 - cos_theta.clamp(0.0, 1.0)).powi(5);
    f0 + (1.0 - f0) * m
}

/// Submerged opaque surface (WGSL `raa_screen_submerged`): `behind` is the distance from the
/// fragment to the opaque surface seen behind it, `n_dot_v` the view cosine, `dir` the refracted
/// view ray, `sun_dir` the direction the sunlight travels (frame `sun_dir`). When the medium
/// depth along the normal, h = behind x n_dot_v, is below `thickness`, returns
/// `(h / cos_t, h / cos_l, true)`: the view path down to the surface and the path of the light
/// refracted into the medium (cos_t, cos_l clamped to [`MIN_COS`]). Otherwise
/// `(path_length(thickness, dir, n), 0, false)`, the unchanged slab path.
pub fn submerged_paths(
    thickness: f32,
    behind: f32,
    n_dot_v: f32,
    dir: [f32; 3],
    n: [f32; 3],
    sun_dir: [f32; 3],
    ior: f32,
) -> (f32, f32, bool) {
    let h = behind * n_dot_v;
    if behind >= NO_ABSORPTION_DISTANCE || h >= thickness {
        return (path_length(thickness, dir, n), 0.0, false);
    }
    let cos_t = (-dot(dir, n)).max(MIN_COS);
    let l = refract(sun_dir, n, 1.0 / ior.max(1.0));
    let cos_l = (-dot(l, n)).max(MIN_COS);
    (h / cos_t, h / cos_l, true)
}
