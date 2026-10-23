// Procedural sky radiance (raa-sky SKY_WGSL).
//
// Contract function `raa_sky_radiance` plus private helpers prefixed `raa_sky_`. It reads only
// its parameters (never the group bindings), so the `sky.env` cube pass and the
// `sky.background` contributor share it. Needs PRELUDE (EnvGpu) and LIB (raa_fbm3).
// Output: scene-referred linear HDR radiance along the unit direction `dir`.

const raa_sky_pi: f32 = 3.14159265;

// Henyey-Greenstein phase function (per steradian) for the cosine between view and light.
fn raa_sky_hg(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = max(1.0 + g2 - 2.0 * g * clamp(cos_theta, -1.0, 1.0), 1e-4);
    return (1.0 - g2) / (4.0 * raa_sky_pi * denom * sqrt(denom));
}

// 1. Zenith / horizon / ground gradient, times intensity.
fn raa_sky_gradient(dir: vec3<f32>, env: EnvGpu) -> vec3<f32> {
    let softness = max(env.sky_params.y, 1e-3);
    let t = pow(clamp(1.0 - max(dir.y, 0.0), 0.0, 1.0), softness);
    let above = mix(env.zenith.rgb, env.horizon.rgb, t);
    // Below the horizon: quick fade from the horizon colour to the ground.
    let g = smoothstep(0.0, 0.06, max(-dir.y, 0.0));
    let below = mix(env.horizon.rgb, env.ground.rgb, g);
    let c = select(below, above, dir.y >= 0.0);
    return c * max(env.sky_params.x, 0.0);
}

// 2 + 3. Henyey-Greenstein (g 0.76) glow and the limb-darkened sun disc.
fn raa_sky_sun(dir: vec3<f32>, env: EnvGpu, l: vec3<f32>, sun_color: vec3<f32>) -> vec3<f32> {
    let c = dot(dir, l);
    let glow = sun_color * raa_sky_hg(c, 0.76) * 0.12;
    // Angle to the sun centre; atan2 keeps precision at the tiny disc radius.
    let s = length(cross(dir, l));
    let theta = select(atan2(s, c), 0.5 * raa_sky_pi, s < 1e-9 && abs(c) < 1e-9);
    let radius = env.sun_disc.x;
    var disc = vec3<f32>(0.0);
    if (radius > 1e-6 && theta < radius) {
        let r = clamp(theta / radius, 0.0, 1.0);
        let mu = sqrt(max(1.0 - r * r, 0.0));
        let limb = 1.0 - 0.6 * (1.0 - mu);
        // Smooth edge over the outer 15 % of the radius.
        let edge = 1.0 - smoothstep(0.85, 1.0, r);
        // The ground hides the disc.
        let ground = smoothstep(-0.004, 0.004, dir.y);
        disc = sun_color * max(env.sun_disc.y, 0.0) * limb * edge * ground;
    }
    return glow + disc;
}

// 4. Cloud density at a point of the layer (density-field units = metres / scale): 5-octave
// fbm warped by a low-octave fbm offset, remapped over the coverage threshold, times density.
fn raa_sky_cloud_density(q: vec2<f32>, env: EnvGpu) -> f32 {
    let p = vec3<f32>(q.x, 0.0, q.y);
    let warp = vec3<f32>(
        raa_fbm3(p + vec3<f32>(0.0, 3.1, 0.0), 2),
        0.0,
        raa_fbm3(p + vec3<f32>(5.2, 1.3, 7.7), 2),
    );
    let n = raa_fbm3(p * 1.7 + warp * 0.9, 5) * 0.5 + 0.5;
    let coverage = clamp(env.clouds.x, 0.0, 1.0);
    let lo = 1.0 - coverage;
    return smoothstep(lo, lo + 0.35, n) * max(env.clouds.y, 0.0);
}

// 4. Cloud layer: radiance (rgb) and alpha (a) along dir, for dir.y > 0.01.
fn raa_sky_clouds(dir: vec3<f32>, env: EnvGpu, l: vec3<f32>, sun_color: vec3<f32>, time: f32) -> vec4<f32> {
    if (dir.y <= 0.01 || env.clouds.x <= 0.0 || env.clouds.y <= 0.0) {
        return vec4<f32>(0.0);
    }
    // Ray from the origin hits the plane y = altitude.
    let altitude = max(env.clouds.z, 1.0);
    let scale = max(env.clouds.w, 1.0);
    let hit = dir.xz * (altitude / dir.y);
    // Wind (m/s) moves the field; the field is sampled at (xz + wind x time) / scale.
    let q = (hit + env.clouds2.xy * time) / scale;
    let rho = raa_sky_cloud_density(q, env);
    if (rho <= 0.0) {
        return vec4<f32>(0.0);
    }
    // 4-step light march toward the sun (-sun_dir) through the layer.
    let march = l.xz / max(abs(l.y), 0.15) * 0.035;
    var depth = rho * 0.5;
    for (var i = 0; i < 4; i = i + 1) {
        depth = depth + raa_sky_cloud_density(q + march * (f32(i) + 0.5), env);
    }
    depth = depth * 0.6;
    // Beer-powder transmittance.
    let beer = exp(-depth);
    let powder = beer * (1.0 - exp(-2.0 * depth)) * 2.0;
    // Silver lining: HG forward lobe (g 0.6), strongest through thin cloud.
    let silver = max(env.clouds2.z, 0.0) * raa_sky_hg(dot(dir, l), 0.6) * beer * 1.5;
    let sun_term = sun_color * (powder * 0.6 + silver);
    // Ambient sky light on the cloud: mostly zenith, some horizon.
    let ambient = mix(env.horizon.rgb, env.zenith.rgb, 0.6) * max(env.sky_params.x, 0.0) * 0.9;
    let color = max(env.cloud_color.rgb, vec3<f32>(0.0)) * (ambient + sun_term);
    let alpha = (1.0 - exp(-rho * 3.0)) * smoothstep(0.01, 0.15, dir.y);
    return vec4<f32>(color, clamp(alpha, 0.0, 1.0));
}

// Sky radiance along the unit direction `dir`; `sun_dir` is the sun's travel direction (S10),
// `sun_color` its rgb x intensity, `time` in seconds (cloud wind).
fn raa_sky_radiance(dir: vec3<f32>, env: EnvGpu, sun_dir: vec3<f32>, sun_color: vec3<f32>, time: f32) -> vec3<f32> {
    let l = -sun_dir;
    let sun = max(sun_color, vec3<f32>(0.0));
    let base = raa_sky_gradient(dir, env) + raa_sky_sun(dir, env, l, sun);
    let cloud = raa_sky_clouds(dir, env, l, sun, time);
    return max(mix(base, cloud.rgb, cloud.a), vec3<f32>(0.0));
}
