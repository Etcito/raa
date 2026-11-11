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
