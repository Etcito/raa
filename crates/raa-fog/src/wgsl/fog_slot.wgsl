// Fog slot provider `height` (S11 `Fog`, priority 10): exponential height fog and aerial
// perspective from EnvGpu.fog / fog_color / aerial. Uses only PRELUDE names; every helper
// starts with `raa_height_`. The optical depth mirrors crates/raa-fog/src/integral.rs.

// Distance in metres toward which sky pixels integrate the height fog.
const raa_height_sky_distance_m: f32 = 20000.0;

// Analytic optical depth of the height fog along y(t) = origin_y + dir_y * t over
// [max(desc start, 0), end_m] (integral.rs `optical_depth` with start_m 0).
fn raa_height_optical_depth(origin_y: f32, dir_y: f32, end_m: f32) -> f32 {
    let density = raa_env.fog.x;
    let falloff = raa_env.fog.y;
    let t0 = max(raa_env.fog.w, 0.0);
    let t1 = end_m;
    if (t1 <= t0 || density <= 0.0) {
        return 0.0;
    }
    let k = falloff * dir_y;
    let span = t1 - t0;
    let scale = density * exp(-falloff * (origin_y - raa_env.fog.z));
    var integral: f32;
    if (abs(k * span) < 1e-4) {
        // Near-horizontal series limit: span * (1 - k span / 2 + (k span)^2 / 6).
        let kd = k * span;
        integral = scale * exp(-k * t0) * span * (1.0 - 0.5 * kd + kd * kd / 6.0);
    } else {
        integral = scale * (exp(-k * t0) - exp(-k * t1)) / k;
    }
    return max(integral, 0.0);
}

// Env cube radiance at its max mip (the blurred sky) in direction `dir`.
fn raa_height_env(dir: vec3<f32>) -> vec3<f32> {
    return textureSampleLevel(raa_env_specular, raa_linear_clamp, dir, raa_env_sh.info.z).rgb;
}

// Henyey-Greenstein phase function for the cosine `cos_theta`, anisotropy g.
fn raa_height_hg(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = max(1.0 + g2 - 2.0 * g * cos_theta, 1e-4);
    return (1.0 - g2) / (4.0 * 3.14159265 * denom * sqrt(denom));
}

// Fog radiance seen along `dir`: env at max mip x fog colour + sun inscatter (HG g 0.6).
fn raa_height_radiance(dir: vec3<f32>) -> vec3<f32> {
    let ambient = raa_height_env(dir) * raa_env.fog_color.rgb;
    let cos_sun = dot(dir, -raa_frame.sun_dir.xyz);
    let sun = raa_height_hg(cos_sun, 0.6) * raa_frame.sun_color.rgb * raa_env.fog_color.w;
    return ambient + sun;
}

// Horizontal direction of `dir` (x axis when `dir` is vertical).
fn raa_height_horizontal(dir: vec3<f32>) -> vec3<f32> {
    let h = vec3<f32>(dir.x, 0.0, dir.z);
    let len = length(h);
    if (len < 1e-4) {
        return vec3<f32>(1.0, 0.0, 0.0);
    }
    return h / len;
}

// Height fog then aerial perspective applied to `color` seen at `world_pos` from `camera`.
// Sky pixels (`sky`) get only the height part, integrated to a far distance and weighted by
// closeness to the horizon. Density 0 and aerial strength 0 return `color` unchanged.
fn raa_height_apply(color: vec3<f32>, world_pos: vec3<f32>, camera: vec3<f32>, sky: bool) -> vec3<f32> {
    let to_point = world_pos - camera;
    let dist = length(to_point);
    if (dist <= 0.0) {
        return color;
    }
    let dir = to_point / dist;
    var out = color;

    // (a) Height fog.
    if (raa_env.fog.x > 0.0) {
        var end_m = dist;
        if (sky) {
            end_m = raa_height_sky_distance_m;
        }
        let t = exp(-raa_height_optical_depth(camera.y, dir.y, end_m));
        let fogged = out * t + raa_height_radiance(dir) * (1.0 - t);
        if (sky) {
            let horizon = 1.0 - smoothstep(0.0, 0.25, abs(dir.y));
            out = mix(out, fogged, horizon);
        } else {
            out = fogged;
        }
    }
    if (sky) {
        return out;
    }

    // (b) Aerial perspective toward the env horizon radiance.
    let strength = raa_env.aerial.x;
    if (strength > 0.0) {
        let amount = (1.0 - exp(-dist / max(raa_env.aerial.y, 1e-3))) * strength;
        out = mix(out, raa_height_env(raa_height_horizontal(dir)), clamp(amount, 0.0, 1.0));
    }
    return out;
}

fn raa_slot_fog(color: vec3<f32>, world_pos: vec3<f32>, c: ShadeCtx) -> vec3<f32> {
    return raa_height_apply(color, world_pos, raa_frame.camera_pos.xyz, false);
}
