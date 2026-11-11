// Cloud uniform record and density model of the M2 volumetric cloud backend.
//
// Adapted from Solstice revision 7e465b599c13da165948ebaa44df824c36c3ae07,
// crates/solstice-render/shaders/clouds/params.wgsl:3-24 (the uniform record) and
// crates/solstice-render/shaders/clouds/common.wgsl:19-158 (the density model),
// after Schneider 2015, "The Real-time Volumetric Cloudscapes of Horizon: Zero Dawn".
// MIT License, Copyright (c) 2026 SamG-Coder.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// `clouds/shaders.rs` places the group-2 bindings 0-9 before this text; the file
// declares no binding. Distances and heights are metres above the ground sphere.
// Constants are prefixed `RAA_CLOUD_` and functions `raa_cloud_`.

// Twin of `CloudGpu` in `clouds/resources.rs`. The first 64 bytes are the Solstice
// record; the camera basis, screen lane and previous view-projection follow the RAA
// cloud ABI (`P2-CLOUD-GPU`).
struct CloudGpu {
    wind_offset: vec2<f32>,
    evolution: f32,
    coverage: f32,
    cloud_type: f32,
    base: f32,
    top: f32,
    density: f32,
    cirrus: f32,
    rain: f32,
    shadow_extent: f32,
    enabled: f32,
    shadow_origin: vec2<f32>,
    wind_direction: vec2<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    forward: vec4<f32>,
    screen: vec4<f32>,
    previous_view_proj: mat4x4<f32>,
}

const RAA_CLOUD_PLANET_RADIUS: f32 = 6360000.0;
// Metres covered by one repeat of the weather map, the shape noise and the detail noise.
const RAA_CLOUD_WEATHER_TILE: f32 = 36000.0;
const RAA_CLOUD_SHAPE_TILE: f32 = 7000.0;
const RAA_CLOUD_DETAIL_TILE: f32 = 700.0;
// Extinction per metre of cloud at density 1. Real cumulus is around 0.05: a little over 50 m
// of it hides the sun.
const RAA_CLOUD_EXTINCTION: f32 = 0.05;
// Cloud droplets scatter nearly all the light they meet.
const RAA_CLOUD_ALBEDO: f32 = 0.98;
const RAA_CLOUD_CIRRUS_HEIGHT: f32 = 9000.0;
// Beyond this, clouds are lost in the haze.
const RAA_CLOUD_MAX_DISTANCE: f32 = 90000.0;
// The longest stretch of the layer one ray marches through.
const RAA_CLOUD_MAX_PATH: f32 = 40000.0;

fn raa_cloud_remap(x: f32, low: f32, high: f32, new_low: f32, new_high: f32) -> f32 {
    return new_low + (x - low) / (high - low) * (new_high - new_low);
}

// Distance along a ray from height `h` above the ground, rising at `mu` (the sine of its
// elevation), to where it leaves the sphere at height `shell` that contains it. Written to avoid
// the cancellation that planet-sized numbers cause in the textbook formula.
fn raa_cloud_shell_exit(h: f32, mu: f32, shell: f32) -> f32 {
    let r = RAA_CLOUD_PLANET_RADIUS + h;
    let b = r * mu;
    let c = (h - shell) * (2.0 * RAA_CLOUD_PLANET_RADIUS + h + shell);
    let s = sqrt(max(b * b - c, 0.0));
    return select(-b + s, -c / (b + s), b > 0.0);
}

// Height above the ground at distance `t` along a ray from height `h` rising at `mu`.
fn raa_cloud_height_along(h: f32, mu: f32, t: f32) -> f32 {
    let r = RAA_CLOUD_PLANET_RADIUS + h;
    let grow = t * t + 2.0 * r * t * mu;
    return h + grow / (sqrt(r * r + grow) + r);
}

// The stretch of a ray from height `h` rising at `mu` that lies in the cloud layer, as
// (enter, exit), or exit < enter when it misses: when the ground is in the way, or the layer is
// too far to matter.
fn raa_cloud_layer_span(h: f32, mu: f32) -> vec2<f32> {
    var enter = 0.0;
    var exit = 0.0;
    if h < raa_cloud.base {
        // The ground hides the layer from a ray heading down to meet it.
        let r = RAA_CLOUD_PLANET_RADIUS + h;
        if mu < 0.0
            && r * r * (mu * mu - 1.0) + RAA_CLOUD_PLANET_RADIUS * RAA_CLOUD_PLANET_RADIUS >= 0.0
        {
            return vec2<f32>(1.0, 0.0);
        }
        enter = raa_cloud_shell_exit(h, mu, raa_cloud.base);
        exit = raa_cloud_shell_exit(h, mu, raa_cloud.top);
    } else if h < raa_cloud.top {
        exit = raa_cloud_shell_exit(h, mu, raa_cloud.top);
    } else {
        return vec2<f32>(1.0, 0.0);
    }
    if enter > RAA_CLOUD_MAX_DISTANCE {
        return vec2<f32>(1.0, 0.0);
    }
    return vec2<f32>(enter, min(exit, enter + RAA_CLOUD_MAX_PATH));
}

// How far up the layer `height` is, 0 at its base and 1 at its top.
fn raa_cloud_layer_fraction(height: f32) -> f32 {
    return clamp(
        (height - raa_cloud.base) / max(raa_cloud.top - raa_cloud.base, 1.0),
        0.0,
        1.0,
    );
}

// Where `value` falls among all the weather map's coverage values, from 0 to 1.
fn raa_cloud_coverage_rank(value: f32) -> f32 {
    let x = clamp(value, 0.0, 1.0) * 255.0;
    let i = u32(floor(x));
    return mix(
        raa_cloud_coverage_cdf[i],
        raa_cloud_coverage_cdf[min(i + 1u, 255u)],
        fract(x),
    );
}

// The weather where the drifting map lies over `xz`.
fn raa_cloud_weather_at(xz: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(
        raa_cloud_weather_map,
        raa_cloud_noise_sampler,
        (xz - raa_cloud.wind_offset) / RAA_CLOUD_WEATHER_TILE,
        0.0,
    );
}

// How much of the layer's height cloud of `kind` fills at fraction `h` of the way up: a thin
// sheet low in the layer for stratus, a lumpy band for stratocumulus, and for cumulus a flat
// base rising to rounded tops.
fn raa_cloud_height_profile(h: f32, kind: f32) -> f32 {
    let stratus = smoothstep(0.0, 0.1, h) * (1.0 - smoothstep(0.25, 0.4, h));
    let stratocumulus = smoothstep(0.0, 0.12, h) * (1.0 - smoothstep(0.45, 0.75, h));
    let cumulus = smoothstep(0.0, 0.07, h) * (1.0 - smoothstep(0.6, 1.0, h));
    let low = mix(stratus, stratocumulus, clamp(kind * 2.0, 0.0, 1.0));
    return mix(low, cumulus, clamp(kind * 2.0 - 1.0, 0.0, 1.0));
}

// Cloud density at a point `xz` across the ground and `height` above it. The cheap form skips
// the detail noise, for shadows and for finding cloud before marching it finely.
fn raa_cloud_density(xz: vec2<f32>, height: f32, cheap: bool) -> f32 {
    let h = raa_cloud_layer_fraction(height);
    if h <= 0.0 || h >= 1.0 || raa_cloud.coverage <= 0.001 {
        return 0.0;
    }
    let weather = raa_cloud_weather_at(xz);
    let kind = clamp(raa_cloud.cloud_type + (weather.g - 0.5) * 0.35, 0.0, 1.0);
    // Deep storm clouds spread into anvils at their tops.
    let anvil = raa_cloud.cloud_type
        * smoothstep(3500.0, 6500.0, raa_cloud.top - raa_cloud.base)
        * smoothstep(0.65, 0.95, h);
    let coverage = mix(raa_cloud.coverage, sqrt(raa_cloud.coverage), anvil);
    // How much cloud can form here, 0..1: the weather map's rank among all its values, raised
    // or lowered by the coverage asked for. At coverage 0.5 it is the rank itself; less, and
    // only the highest ranks keep any; more, and even the lowest do. The shape noise is then
    // cut against it, so a little gives small scattered puffs and a lot gives a solid deck.
    let local = clamp(raa_cloud_coverage_rank(weather.r) + coverage * 2.0 - 1.0, 0.0, 1.0);
    if local <= 0.0 {
        return 0.0;
    }
    var profile = raa_cloud_height_profile(h, kind);
    profile = max(profile, anvil * (1.0 - smoothstep(0.95, 1.0, h)));

    // Shapes drift with the weather and boil slowly upwards as they evolve.
    let p = vec3<f32>(
        xz.x - raa_cloud.wind_offset.x,
        height - raa_cloud.evolution,
        xz.y - raa_cloud.wind_offset.y,
    );
    let shape = textureSampleLevel(
        raa_cloud_shape,
        raa_cloud_noise_sampler,
        p / RAA_CLOUD_SHAPE_TILE,
        0.0,
    );
    let cells = shape.g * 0.625 + shape.b * 0.25 + shape.a * 0.125;
    let base_shape = raa_cloud_remap(shape.r, cells - 1.0, 1.0, 0.0, 1.0);
    var density = clamp(
        raa_cloud_remap(base_shape * profile, 1.0 - local, 1.0, 0.0, 1.0),
        0.0,
        1.0,
    ) * local;
    // Even a solid deck is thicker in some places than others, which shows from below as
    // lighter and darker patches. Without this, full coverage would be a featureless sheet.
    density *= 0.35 + 0.65 * smoothstep(0.15, 0.85, base_shape);
    if cheap || density <= 0.0 {
        return density;
    }
    // Wisps at the base and billows at the top, carved from the edges inwards at two scales:
    // puffs of about 100 m and, within them, curls of about 25 m.
    let skew = vec3<f32>(raa_cloud.wind_direction.x, 0.0, raa_cloud.wind_direction.y) * h * 300.0;
    let q = p + skew;
    let detail = textureSampleLevel(
        raa_cloud_detail,
        raa_cloud_noise_sampler,
        q / RAA_CLOUD_DETAIL_TILE,
        0.0,
    );
    let curls = textureSampleLevel(
        raa_cloud_detail,
        raa_cloud_noise_sampler,
        q / (RAA_CLOUD_DETAIL_TILE * 0.27),
        0.0,
    );
    let fine = (detail.r * 0.625 + detail.g * 0.25 + detail.b * 0.125) * 0.7
        + (curls.r * 0.625 + curls.g * 0.25 + curls.b * 0.125) * 0.3;
    let erode = mix(fine, 1.0 - fine, clamp(h * 4.0, 0.0, 1.0));
    // Stratus is smooth; heaped cloud is deeply carved.
    let strength = mix(0.25, 0.55, kind);
    density = clamp(raa_cloud_remap(density, erode * strength, 1.0, 0.0, 1.0), 0.0, 1.0);
    return density;
}

// ---------------------------------------------------------------------------
// Lighting and ray march.
//
// Adapted from Solstice revision 7e465b599c13da165948ebaa44df824c36c3ae07,
// crates/solstice-render/shaders/clouds/common.wgsl:160-354 (lighting, march, cirrus),
// with `above_horizon` from crates/solstice-render/shaders/common/environment.wgsl:101-106
// and `luminance` from crates/solstice-render/shaders/common/math.wgsl:79-81, after
// Hillaire 2016, "Physically Based Sky, Atmosphere and Cloud Rendering in Frostbite".
// MIT License, Copyright (c) 2026 SamG-Coder; the complete notice is at the top of this
// file. Environment reads map to the group-2 bindings of contract `P2-CLOUD-GPU`: the
// transmittance table through `raa_atmosphere_transmittance_uv`, the view's sky table
// through `raa_atmosphere_sky_view_uv`, and the raw dominant radiance in
// `raa_cloud_sky.daylight.rgb * raa_cloud_sky.dominant.w` toward
// `raa_cloud_sky.dominant.xyz`. `with_aerial_perspective` is deliberately not ported:
// raa-fog remains the sole owner of scene aerial perspective. All divisions keep guarded
// denominators so finite inputs cannot produce NaN.

fn raa_cloud_luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn raa_cloud_henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let d = max(1.0 + g * g - 2.0 * g * cos_theta, raa_atmosphere_min_scale);
    return (1.0 - g * g) / (4.0 * raa_atmosphere_pi * d * sqrt(d));
}

// A strong forward lobe for the silver lining towards the sun, and a weaker back lobe.
fn raa_cloud_phase(cos_theta: f32, scale: f32) -> f32 {
    return mix(
        raa_cloud_henyey_greenstein(cos_theta, -0.25 * scale),
        raa_cloud_henyey_greenstein(cos_theta, 0.8 * scale),
        0.7,
    );
}

// Light from the sun scattered towards the eye after crossing `depth` of optical depth inside
// the cloud. Each octave stands for light scattered more times: weaker, spread wider, and
// reaching deeper (Wrenninge et al. 2013).
fn raa_cloud_sun_scattering(depth: f32, cos_theta: f32) -> f32 {
    var total = 0.0;
    var strength = 1.0;
    var reach = 1.0;
    var spread = 1.0;
    for (var i = 0; i < 4; i++) {
        total += strength * raa_cloud_phase(cos_theta, spread) * exp(-depth * reach);
        strength *= 0.5;
        reach *= 0.45;
        spread *= 0.5;
    }
    return total;
}

// How much of the dominant light is above the horizon seen from `radius` (km from the planet
// centre): the horizon dips with height, so cloud keeps the sun after the ground has lost it.
// The transmittance table does not know the ground is there, so this blocks the light instead.
fn raa_cloud_above_horizon(radius: f32, mu: f32) -> f32 {
    let ground = raa_atmosphere_ground_km(raa_cloud_sky);
    let ratio = ground / max(radius, ground);
    let dip = -sqrt(max(1.0 - ratio * ratio, 0.0));
    // Across the sun's own width, so that it sets gradually rather than switching off.
    return smoothstep(dip - 0.006, dip + 0.006, mu);
}

// Radiance of the cloudless sky towards `direction`, from this view's atmosphere sky-view
// table at the landed mapping and the raw daylight intensity of the frame.
fn raa_cloud_sky_radiance(direction: vec3<f32>) -> vec3<f32> {
    let uv = raa_atmosphere_sky_view_uv(direction);
    let scattered = textureSampleLevel(raa_cloud_sky_view, raa_cloud_lut_sampler, uv, 0.0).rgb;
    return max(scattered, vec3<f32>(0.0)) * max(raa_cloud_sky.daylight.w, 0.0);
}

// Dominant light arriving at `height`, after the atmosphere above it. The raw dominant
// radiance is attenuated exactly once here, by this path's own transmittance table.
fn raa_cloud_sun_at_height(height: f32) -> vec3<f32> {
    let r = raa_atmosphere_ground_km(raa_cloud_sky) + height * 0.001;
    let mu = raa_cloud_sky.dominant.y;
    let uv = raa_atmosphere_transmittance_uv(r, mu, raa_cloud_sky);
    let transmittance = textureSampleLevel(raa_cloud_transmittance_lut, raa_cloud_lut_sampler, uv, 0.0).rgb;
    return transmittance * raa_cloud_sky.daylight.rgb * raa_cloud_sky.dominant.w
        * raa_cloud_above_horizon(r, mu);
}

// Optical depth of cloud between a point and the sun, from a few samples along the way that
// grow further apart.
fn raa_cloud_depth_towards_sun(xz: vec2<f32>, height: f32, samples: i32) -> f32 {
    let reach = min(raa_cloud.top - raa_cloud.base, 2500.0);
    let to_sun = raa_cloud_sky.dominant.xyz;
    var depth = 0.0;
    var travelled = 0.0;
    let total = max(samples * (samples + 1) / 2, 1);
    for (var i = 0; i < samples; i++) {
        let step = reach * (f32(i) + 1.0) / f32(total);
        let at = travelled + step * 0.5;
        travelled += step;
        let point_height = height + to_sun.y * at;
        if point_height > raa_cloud.top {
            break;
        }
        let density = raa_cloud_density(xz + to_sun.xz * at, point_height, i >= 2);
        depth += density * step;
    }
    return depth * RAA_CLOUD_EXTINCTION * raa_cloud.density;
}

// The result of one cloud ray: linear HDR radiance scattered towards the eye, the scalar
// transmittance of what lies behind the cloud, and roughly where along the ray the cloud is.
struct RaaCloudSample {
    radiance: vec3<f32>,
    transmittance: f32,
    distance: f32,
}

// Marches the clouds along a ray from the viewer. `steps` sets quality; `jitter` in 0..1
// offsets the samples so that successive frames, blended, cover the gaps between them.
fn raa_cloud_march_clouds(
    direction: vec3<f32>,
    steps: i32,
    light_samples: i32,
    jitter: f32,
) -> RaaCloudSample {
    var result = RaaCloudSample(vec3<f32>(0.0), 1.0, RAA_CLOUD_MAX_DISTANCE);
    if raa_cloud.enabled < 0.5 {
        return result;
    }
    let h0 = raa_cloud_sky.viewer.y;
    let mu = direction.y;
    let span = raa_cloud_layer_span(h0, mu);
    if span.y <= span.x {
        return result;
    }
    let length = span.y - span.x;
    let to_sun = raa_cloud_sky.dominant.xyz;
    let cos_theta = dot(direction, to_sun);
    let darken = 1.0 - 0.6 * raa_cloud.rain;
    // Skylight inside the cloud: from the horizon low down, from overhead higher up. The more
    // of the sky is cloud, the more of that light comes from other clouds, which are grey
    // rather than blue.
    let grey = clamp(raa_cloud.coverage * 0.8, 0.0, 0.8);
    let horizon_sky = raa_cloud_sky_radiance(normalize(vec3<f32>(direction.x, 0.05, direction.z)));
    let zenith_sky = raa_cloud_sky_radiance(vec3<f32>(0.0, 1.0, 0.0));
    let sky_low = mix(horizon_sky, vec3<f32>(raa_cloud_luminance(horizon_sky)), grey) * 0.6;
    let sky_high = mix(zenith_sky, vec3<f32>(raa_cloud_luminance(zenith_sky)), grey);

    var weighted_distance = 0.0;
    var weight = 0.0;
    let step_count = max(steps, 1);
    // Samples crowd towards the near end of the layer, where cloud is seen largest and in most
    // detail, and spread out into the distance.
    for (var i = 0; i < steps; i++) {
        let s0 = max((f32(i) + jitter) / f32(step_count), 0.0);
        let t = span.x + length * pow(s0, 1.6);
        let step = length * (pow(min(s0 + 1.0 / f32(step_count), 1.0), 1.6) - pow(s0, 1.6))
            + 1.0;
        let height = raa_cloud_height_along(h0, mu, t);
        let xz = raa_cloud_sky.viewer.xz + direction.xz * t;
        let coarse = raa_cloud_density(xz, height, true);
        if coarse > 0.0 {
            let density = raa_cloud_density(xz, height, false);
            if density > 0.0 {
                let sigma = density * RAA_CLOUD_EXTINCTION * raa_cloud.density;
                let h = raa_cloud_layer_fraction(height);
                let sun_depth = raa_cloud_depth_towards_sun(xz, height, light_samples);
                // Dark creases where light has had little cloud to scatter in yet: the "powder"
                // look of a sunlit cumulus, strongest looking away from the sun.
                let powder = 1.0 - exp(-sigma * 120.0);
                let powder_weight = 0.5 - 0.5 * cos_theta;
                let sun_light = raa_cloud_sun_at_height(height);
                // Light scattered a few times keeps its direction (the octaves). Light that
                // has crossed hundreds of metres of cloud has scattered so often that it
                // diffuses, and a cloud passes far more of it than its optical depth suggests:
                // the two-stream result T = 1 / (1 + 0.75 (1 - g) tau), for g about 0.85.
                let diffused = 1.0
                    / max(1.0 + 0.11 * sun_depth, raa_atmosphere_min_scale)
                    / (4.0 * raa_atmosphere_pi);
                let sun = sun_light * (raa_cloud_sun_scattering(sun_depth, cos_theta) + diffused)
                    * mix(1.0, powder, powder_weight * 0.6);
                // Skylight inside a cloud also diffuses down through the cloud above it.
                let above_height = mix(height, raa_cloud.top, 0.5);
                let above = raa_cloud_density(xz, above_height, true) * (raa_cloud.top - height)
                    * RAA_CLOUD_EXTINCTION * raa_cloud.density * 0.5;
                let sky = mix(sky_low, sky_high, h)
                    / max(1.0 + 0.11 * above, raa_atmosphere_min_scale);
                // Sunlit ground lights the bases from below, a large part of why real cloud
                // bases are grey rather than black: its radiance is E * albedo / pi, for an
                // albedo of about 0.25, fading with height into the cloud.
                let ground = raa_cloud_sun_at_height(0.0) * max(to_sun.y, 0.0) * 0.25
                    / raa_atmosphere_pi * (1.0 - h) * (1.0 - 0.7 * raa_cloud.coverage);
                let source = (sun + sky + ground) * RAA_CLOUD_ALBEDO * darken;
                let through = exp(-sigma * step);
                // Integrated exactly over the step for a constant medium (Hillaire 2016).
                result.radiance += result.transmittance * source * (1.0 - through);
                weighted_distance += t * result.transmittance * (1.0 - through);
                weight += result.transmittance * (1.0 - through);
                result.transmittance *= through;
                if result.transmittance < 0.01 {
                    // What little would still show through is not worth marching for; call it
                    // none, or the sun's disc, thousands of times brighter, would show.
                    result.transmittance = 0.0;
                    break;
                }
            }
        }
    }
    if weight > 0.0 {
        result.distance = weighted_distance / weight;
    }
    return result;
}

// High cirrus, a thin sheet far above the main layer, behind it.
fn raa_cloud_cirrus_layer(direction: vec3<f32>, behind: RaaCloudSample) -> RaaCloudSample {
    var result = behind;
    if raa_cloud.cirrus <= 0.001 || direction.y <= 0.0 || raa_cloud.enabled < 0.5 {
        return result;
    }
    let h0 = raa_cloud_sky.viewer.y;
    let t = raa_cloud_shell_exit(h0, direction.y, RAA_CLOUD_CIRRUS_HEIGHT);
    let xz = raa_cloud_sky.viewer.xz + direction.xz * t;
    // Cirrus moves faster, in the stronger winds aloft.
    let uv = (xz - raa_cloud.wind_offset * 2.5) / (RAA_CLOUD_WEATHER_TILE * 0.7);
    let streaks = textureSampleLevel(raa_cloud_weather_map, raa_cloud_noise_sampler, uv, 0.0).b;
    let fibres = textureSampleLevel(
        raa_cloud_detail,
        raa_cloud_noise_sampler,
        vec3<f32>(uv * 9.0, 0.37),
        0.0,
    ).r;
    let cover = smoothstep(
        1.0 - raa_cloud.cirrus,
        1.0 - raa_cloud.cirrus + 0.35,
        streaks * (0.75 + 0.5 * fibres),
    );
    let alpha = cover * 0.55 * smoothstep(0.0, 0.06, direction.y);
    if alpha <= 0.0 {
        return result;
    }
    let cos_theta = dot(direction, raa_cloud_sky.dominant.xyz);
    let light = raa_cloud_sun_at_height(RAA_CLOUD_CIRRUS_HEIGHT)
            * raa_cloud_henyey_greenstein(cos_theta, 0.6) * 1.6
        + raa_cloud_sky_radiance(vec3<f32>(0.0, 1.0, 0.0)) * 0.8;
    // Behind the main layer: seen only through it.
    result.radiance += behind.transmittance * light * alpha;
    result.transmittance *= 1.0 - alpha;
    if behind.transmittance > 0.99 {
        result.distance = t;
    }
    return result;
}

// Clouds and cirrus along a ray from the viewer, with no aerial perspective: raa-fog owns
// scene aerial perspective. A disabled model returns the empty result unchanged.
fn raa_cloud_along(
    direction: vec3<f32>,
    steps: i32,
    light_samples: i32,
    jitter: f32,
) -> RaaCloudSample {
    let main = raa_cloud_march_clouds(direction, steps, light_samples, jitter);
    return raa_cloud_cirrus_layer(direction, main);
}
