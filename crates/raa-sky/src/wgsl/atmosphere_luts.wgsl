// Generates the three atmosphere lookup tables of the physical sky.
//
// Adapted from Solstice revision 7e465b599c13da165948ebaa44df824c36c3ae07,
// crates/solstice-render/shaders/sky/luts.wgsl:19-185, after Hillaire 2020,
// "A Scalable and Production Ready Sky and Atmosphere Rendering Technique".
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
// Transmittance and multiple scattering depend only on the atmosphere, so they are built
// once. The sky view depends on the sun and the viewer height, so it is rebuilt per view.
// Radiance is per unit of sun illuminance; distances are kilometres.
//
// The module is composed after `atmosphere.wgsl` only, never the lit prelude: every pipeline
// has empty groups 0 and 1 and binds its resources explicitly in group 2:
//   0 PhysicalSkyGpu uniform, 1 transmittance LUT, 2 multiple-scattering LUT,
//   3 filtering sampler, 4 Rgba16Float storage output, 5 directional result buffer.
// The dispatched LUT entry selects which table the storage output receives; the directional
// entry writes its own 16-byte STORAGE buffer, copied to `WeatherLightingGpu` offset 32 by the
// recorder. The two layouts are separate: the LUT pipelines never bind 5 and the directional
// pipeline never binds 2 or 4.

@group(2) @binding(0) var<uniform> raa_atmosphere_lut_sky: PhysicalSkyGpu;
@group(2) @binding(1) var raa_atmosphere_lut_transmittance: texture_2d<f32>;
@group(2) @binding(2) var raa_atmosphere_lut_multi_scatter: texture_2d<f32>;
@group(2) @binding(3) var raa_atmosphere_lut_sampler: sampler;
@group(2) @binding(4) var raa_atmosphere_lut_output: texture_storage_2d<rgba16float, write>;

// The GPU-written directional transmission lane of the S10 `WeatherLightingGpu`: rgb is the
// transmittance toward the dominant light and w is one. The recorder copies the 16 bytes into
// the CPU-uploaded uniform at offset 32, after the dispatches below.
struct RaaDirectionalTransmission {
    value: vec4<f32>,
}

@group(2) @binding(5) var<storage, read_write> raa_atmosphere_directional_result:
    RaaDirectionalTransmission;

// The authored ground albedo, clamped to the physical [0, 1] domain. The `.w` component of
// the lane is the Mie multiplier, applied by `raa_atmosphere_sample_medium` instead.
fn raa_atmosphere_lut_ground_albedo(sky: PhysicalSkyGpu) -> vec3<f32> {
    return clamp(sky.ground_albedo.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
}

// Integral of the extinction over one step: (1 - exp(-extinction * dt)) / extinction.
// The denominator keeps the smallest positive scale and dt is clamped to a finite,
// nonnegative range, so zero or degenerate authored coefficients cannot divide by zero.
fn raa_atmosphere_lut_integrate(extinction: vec3<f32>, dt: f32) -> vec3<f32> {
    let e = max(extinction, vec3<f32>(0.0));
    let step = clamp(dt, 0.0, 1.0e9);
    return (vec3<f32>(1.0) - exp(-e * step)) / max(e, vec3<f32>(raa_atmosphere_min_scale));
}

// Optical depth from `origin` through the top of the atmosphere along `direction`, in 40
// uniform steps. A ray that hits the planet returns the saturated sentinel of the source.
fn raa_atmosphere_lut_optical_depth_to_top(
    origin: vec3<f32>,
    direction: vec3<f32>,
    sky: PhysicalSkyGpu,
) -> vec3<f32> {
    let ground = raa_atmosphere_ground_km(sky);
    let top = raa_atmosphere_top_km(sky);
    if (raa_atmosphere_ray_sphere(origin, direction, ground) > 0.0) {
        return vec3<f32>(1.0e9);
    }
    let span = raa_atmosphere_ray_sphere(origin, direction, top);
    if (!(span > 0.0)) {
        return vec3<f32>(0.0);
    }
    const STEPS: i32 = 40;
    var depth = vec3<f32>(0.0);
    let dt = span / f32(STEPS);
    for (var i = 0; i < STEPS; i++) {
        let p = origin + direction * (f32(i) + 0.5) * dt;
        depth += raa_atmosphere_sample_medium(length(p) - ground, sky).extinction * dt;
    }
    return depth;
}

// Transmittance toward `sun` at `p`, from the transmittance LUT (explicit LOD 0, valid in
// compute). The radius is guarded so a degenerate point cannot divide by zero.
fn raa_atmosphere_lut_sample_transmittance(
    p: vec3<f32>,
    sun: vec3<f32>,
    sky: PhysicalSkyGpu,
) -> vec3<f32> {
    let r = max(length(p), raa_atmosphere_min_radius_km);
    let mu = dot(p / r, sun);
    let uv = raa_atmosphere_transmittance_uv(r, mu, sky);
    return textureSampleLevel(
        raa_atmosphere_lut_transmittance,
        raa_atmosphere_lut_sampler,
        uv,
        0.0,
    ).rgb;
}

// Multiple scattering at `p` for the sun direction: cos-sun on U, normalized height on V.
// The height span keeps the smallest positive scale so authored radii cannot invert it.
fn raa_atmosphere_lut_sample_multi_scatter(
    p: vec3<f32>,
    sun: vec3<f32>,
    sky: PhysicalSkyGpu,
) -> vec3<f32> {
    let ground = raa_atmosphere_ground_km(sky);
    let top = raa_atmosphere_top_km(sky);
    let r = max(length(p), raa_atmosphere_min_radius_km);
    let cos_sun = clamp(dot(p / r, sun), -1.0, 1.0);
    let span = max(top - ground, raa_atmosphere_min_scale);
    let uv = vec2<f32>(
        cos_sun * 0.5 + 0.5,
        clamp((r - ground) / span, 0.0, 1.0),
    );
    return textureSampleLevel(
        raa_atmosphere_lut_multi_scatter,
        raa_atmosphere_lut_sampler,
        uv,
        0.0,
    ).rgb;
}

// Static transmittance LUT: one texel per Bruneton (radius, view zenith) parameter pair,
// integrated through the top of the atmosphere.
@compute @workgroup_size(8, 8)
fn raa_atmosphere_build_transmittance(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(raa_atmosphere_lut_output);
    if (any(id.xy >= size)) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + vec2<f32>(0.5)) / vec2<f32>(size);
    let params = raa_atmosphere_transmittance_params(uv, raa_atmosphere_lut_sky);
    let origin = vec3<f32>(0.0, params.x, 0.0);
    let direction = vec3<f32>(
        sqrt(max(1.0 - params.y * params.y, 0.0)),
        params.y,
        0.0,
    );
    let depth = raa_atmosphere_lut_optical_depth_to_top(
        origin,
        direction,
        raa_atmosphere_lut_sky,
    );
    textureStore(raa_atmosphere_lut_output, id.xy, vec4<f32>(exp(-depth), 1.0));
}

// Static multiple-scattering LUT: second-order scattering toward a point and how much of a
// uniform input it re-emits. Their ratio is the sum of all higher orders as a geometric
// series (Hillaire, equation 10). 8x8 uniform sphere directions, 20 steps each.
@compute @workgroup_size(8, 8)
fn raa_atmosphere_build_multi_scatter(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(raa_atmosphere_lut_output);
    if (any(id.xy >= size)) {
        return;
    }
    let sky = raa_atmosphere_lut_sky;
    let ground = raa_atmosphere_ground_km(sky);
    let top = raa_atmosphere_top_km(sky);
    let uv = (vec2<f32>(id.xy) + vec2<f32>(0.5)) / vec2<f32>(size);
    let cos_sun = clamp(uv.x * 2.0 - 1.0, -1.0, 1.0);
    let span = max(top - ground, raa_atmosphere_min_radius_km);
    let inset = min(0.01, span * 0.25);
    let radius = mix(
        ground + inset,
        max(top - inset, ground + inset),
        clamp(uv.y, 0.0, 1.0),
    );
    let origin = vec3<f32>(0.0, radius, 0.0);
    let sun = vec3<f32>(sqrt(max(1.0 - cos_sun * cos_sun, 0.0)), cos_sun, 0.0);
    let albedo = raa_atmosphere_lut_ground_albedo(sky);

    const SQRT_DIRECTIONS: u32 = 8u;
    const STEPS: i32 = 20;
    var second_order = vec3<f32>(0.0);
    var transfer = vec3<f32>(0.0);
    for (var a = 0u; a < SQRT_DIRECTIONS; a++) {
        for (var b = 0u; b < SQRT_DIRECTIONS; b++) {
            // Uniform directions over the sphere.
            let u = (f32(a) + 0.5) / f32(SQRT_DIRECTIONS);
            let v = (f32(b) + 0.5) / f32(SQRT_DIRECTIONS);
            let cos_theta = 1.0 - 2.0 * u;
            let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
            let phi = raa_atmosphere_tau * v;
            let direction = vec3<f32>(sin_theta * cos(phi), cos_theta, sin_theta * sin(phi));

            var t_max = raa_atmosphere_ray_sphere(origin, direction, top);
            let ground_t = raa_atmosphere_ray_sphere(origin, direction, ground);
            let hits_ground = ground_t > 0.0;
            if (hits_ground) {
                t_max = ground_t;
            }
            if (!(t_max > 0.0)) {
                continue;
            }
            let dt = t_max / f32(STEPS);
            var through = vec3<f32>(1.0);
            var radiance = vec3<f32>(0.0);
            var scatter_transfer = vec3<f32>(0.0);
            for (var i = 0; i < STEPS; i++) {
                let p = origin + direction * (f32(i) + 0.5) * dt;
                let medium = raa_atmosphere_sample_medium(length(p) - ground, sky);
                let scattering = medium.rayleigh + vec3<f32>(medium.mie);
                let step_transmittance = exp(-max(medium.extinction, vec3<f32>(0.0)) * dt);
                let integrate = raa_atmosphere_lut_integrate(medium.extinction, dt);

                let in_sun = raa_atmosphere_lut_sample_transmittance(p, sun, sky);
                // Isotropic phase: this LUT assumes scattering has lost its direction.
                radiance += through * scattering * in_sun / (4.0 * raa_atmosphere_pi) * integrate;
                scatter_transfer += through * scattering * integrate;
                through *= step_transmittance;
            }
            if (hits_ground) {
                let ground_point = origin + direction * t_max;
                let n = raa_atmosphere_unit_direction(ground_point);
                let ground_light = raa_atmosphere_lut_sample_transmittance(ground_point, sun, sky);
                radiance += through * ground_light * max(dot(n, sun), 0.0) * albedo
                    / raa_atmosphere_pi;
            }
            second_order += radiance;
            transfer += scatter_transfer;
        }
    }
    let count = f32(SQRT_DIRECTIONS * SQRT_DIRECTIONS);
    second_order /= count;
    transfer /= count;
    let higher = max(vec3<f32>(1.0) - transfer, vec3<f32>(raa_atmosphere_min_scale));
    textureStore(raa_atmosphere_lut_output, id.xy, vec4<f32>(second_order / higher, 1.0));
}

// Dynamic sky-view LUT: single and multiple scattering along the sky-view direction with 32
// quadratically spaced steps that cluster near the viewer, where most in-scatter happens.
@compute @workgroup_size(8, 8)
fn raa_atmosphere_build_sky_view(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(raa_atmosphere_lut_output);
    if (any(id.xy >= size)) {
        return;
    }
    let sky = raa_atmosphere_lut_sky;
    let ground = raa_atmosphere_ground_km(sky);
    let top = raa_atmosphere_top_km(sky);
    let uv = (vec2<f32>(id.xy) + vec2<f32>(0.5)) / vec2<f32>(size);
    let direction = raa_atmosphere_sky_view_direction(uv);
    let origin = vec3<f32>(0.0, max(sky.viewer.w, raa_atmosphere_min_radius_km), 0.0);
    let sun = raa_atmosphere_unit_direction(sky.sun.xyz);

    var t_max = raa_atmosphere_ray_sphere(origin, direction, top);
    let ground_t = raa_atmosphere_ray_sphere(origin, direction, ground);
    if (ground_t > 0.0) {
        t_max = ground_t;
    }
    if (!(t_max > 0.0)) {
        textureStore(raa_atmosphere_lut_output, id.xy, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    const STEPS: i32 = 32;
    let cos_theta = clamp(dot(direction, sun), -1.0, 1.0);
    let phase_r = raa_atmosphere_rayleigh_phase(cos_theta);
    let phase_m = raa_atmosphere_mie_phase(cos_theta, sky.mie.w);
    var through = vec3<f32>(1.0);
    var radiance = vec3<f32>(0.0);
    var previous_t = 0.0;
    for (var i = 0; i < STEPS; i++) {
        let f = (f32(i) + 0.5) / f32(STEPS);
        let t = f * f * t_max;
        let dt = t - previous_t;
        previous_t = t;
        let p = origin + direction * t;
        let medium = raa_atmosphere_sample_medium(length(p) - ground, sky);
        let step_transmittance = exp(-max(medium.extinction, vec3<f32>(0.0)) * dt);
        let integrate = raa_atmosphere_lut_integrate(medium.extinction, dt);

        let in_sun = raa_atmosphere_lut_sample_transmittance(p, sun, sky);
        let single = medium.rayleigh * phase_r + vec3<f32>(medium.mie * phase_m);
        let scattering = medium.rayleigh + vec3<f32>(medium.mie);
        let multi = raa_atmosphere_lut_sample_multi_scatter(p, sun, sky) * scattering;
        radiance += through * (single * in_sun + multi) * integrate;
        through *= step_transmittance;
    }
    // Below the horizon the ray ends on the planet, which should look like lit ground and
    // not black. This is what shows past the edge of a scene that has no terrain of its own.
    if (ground_t > 0.0) {
        let p = origin + direction * t_max;
        let n = raa_atmosphere_unit_direction(p);
        let lit = max(dot(n, sun), 0.0);
        let ground_light = raa_atmosphere_lut_sample_transmittance(p, sun, sky);
        radiance += through * ground_light * lit * raa_atmosphere_lut_ground_albedo(sky)
            / raa_atmosphere_pi;
    }
    textureStore(raa_atmosphere_lut_output, id.xy, vec4<f32>(radiance, 1.0));
}

// The horizon fade of the directional transmission, adapted from Solstice
// `shaders/common/environment.wgsl:98-113`: the transmittance table does not know the ground
// is there, so a light below the horizon dip is blocked. The 0.006 half-width fades the cut
// across the light's angular size; the authored ground radius replaces the source constant.
const raa_atmosphere_horizon_fade: f32 = 0.006;

fn raa_atmosphere_above_horizon(radius_km: f32, mu: f32, ground_km: f32) -> f32 {
    let ratio = clamp(ground_km / max(radius_km, raa_atmosphere_min_radius_km), 0.0, 1.0);
    let dip = -sqrt(max(1.0 - ratio * ratio, 0.0));
    return smoothstep(dip - raa_atmosphere_horizon_fade, dip + raa_atmosphere_horizon_fade, mu);
}

// Directional transmission: the atmosphere-filtered light toward the dominant light at this
// view's height, with the part below the horizon blocked. One invocation writes the separate
// 16-byte result that the recorder copies into `WeatherLightingGpu.atmosphericTransmission`
// after the table dispatches; the value is (transmittance rgb, 1), never a CPU-written lane.
@compute @workgroup_size(1, 1)
fn raa_atmosphere_build_directional_transmission() {
    let sky = raa_atmosphere_lut_sky;
    let radius = max(sky.viewer.w, raa_atmosphere_min_radius_km);
    let direction = raa_atmosphere_unit_direction(sky.dominant.xyz);
    let mu = clamp(direction.y, -1.0, 1.0);
    let uv = raa_atmosphere_transmittance_uv(radius, mu, sky);
    let filtered = textureSampleLevel(
        raa_atmosphere_lut_transmittance,
        raa_atmosphere_lut_sampler,
        uv,
        0.0,
    ).rgb;
    let horizon = raa_atmosphere_above_horizon(radius, mu, raa_atmosphere_ground_km(sky));
    raa_atmosphere_directional_result.value = vec4<f32>(filtered * horizon, 1.0);
}
