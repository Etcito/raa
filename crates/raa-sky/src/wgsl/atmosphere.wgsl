// Physical atmosphere model and parameterisations (raa-sky atmosphere helpers).
//
// Adapted from Solstice revision 7e465b599c13da165948ebaa44df824c36c3ae07,
// crates/solstice-render/shaders/common/atmosphere.wgsl:4-102, after Hillaire 2020,
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
// Distances are kilometres. Every helper is prefixed `raa_atmosphere_`. The file
// declares the S10 `PhysicalSkyGpu` record but no bindings, so consumers place the
// group-2 binding themselves; the Rust serializer is `atmosphere.rs::to_bytes`.
// All divisions guard their denominators so finite parameters cannot produce NaN.

const raa_atmosphere_pi: f32 = 3.14159265;
const raa_atmosphere_tau: f32 = 6.2831853;

const raa_atmosphere_ground_radius_km: f32 = 6360.0;
const raa_atmosphere_top_radius_km: f32 = 6460.0;

const raa_atmosphere_rayleigh_scattering: vec3<f32> = vec3<f32>(5.802e-3, 13.558e-3, 33.1e-3);
const raa_atmosphere_rayleigh_scale_height_km: f32 = 8.0;
// Several times the clear-air value: desert dust, which pales and warms the sky.
const raa_atmosphere_mie_scattering: f32 = 14.0e-3;
const raa_atmosphere_mie_absorption: f32 = 4.40e-3;
const raa_atmosphere_mie_scale_height_km: f32 = 1.6;
const raa_atmosphere_mie_g: f32 = 0.8;
const raa_atmosphere_ozone_absorption: vec3<f32> = vec3<f32>(0.650e-3, 1.881e-3, 0.085e-3);
const raa_atmosphere_ozone_centre_km: f32 = 25.0;
const raa_atmosphere_ozone_half_width_km: f32 = 15.0;

// Smallest positive divisor/scale used by the guarded division helpers.
const raa_atmosphere_min_scale: f32 = 1.0e-4;
const raa_atmosphere_min_radius_km: f32 = 1.0e-3;

// S10 group-2 binding 0 (private physical-sky ABI, 256 bytes). `viewer.w` is the
// viewer radius in kilometres, already converted by `atmospheric_radius_km`;
// directions point toward the body or light named by the lane.
struct PhysicalSkyGpu {
    viewer: vec4<f32>,
    sun: vec4<f32>,
    moon: vec4<f32>,
    pole: vec4<f32>,
    equinox: vec4<f32>,
    dominant: vec4<f32>,
    daylight: vec4<f32>,
    discs: vec4<f32>,
    ground_albedo: vec4<f32>,
    rayleigh: vec4<f32>,
    mie: vec4<f32>,
    ozone: vec4<f32>,
    planet: vec4<f32>,
    flags: vec4<u32>,
    reserved0: vec4<f32>,
    reserved1: vec4<f32>,
}

struct raa_atmosphere_medium {
    rayleigh: vec3<f32>,
    mie: f32,
    extinction: vec3<f32>,
}

// Ground and top radii of the planet lanes, kept ordered and positive.
fn raa_atmosphere_ground_km(sky: PhysicalSkyGpu) -> f32 {
    return max(sky.planet.x, raa_atmosphere_min_radius_km);
}

fn raa_atmosphere_top_km(sky: PhysicalSkyGpu) -> f32 {
    return max(sky.planet.y, raa_atmosphere_ground_km(sky) + raa_atmosphere_min_radius_km);
}

// Exponential Rayleigh and Mie density and the triangular ozone profile around 25 km.
// `mie.x` is the base scattering coefficient; the authored `ground_albedo.w`
// multiplier is applied here so the upload carries the raw authoring value.
fn raa_atmosphere_sample_medium(height_km: f32, sky: PhysicalSkyGpu) -> raa_atmosphere_medium {
    let h = max(height_km, 0.0);
    let rayleigh_scale = max(sky.rayleigh.w, raa_atmosphere_min_scale);
    let mie_scale = max(sky.mie.z, raa_atmosphere_min_scale);
    let rayleigh = max(sky.rayleigh.rgb, vec3<f32>(0.0)) * exp(-h / rayleigh_scale);
    let mie_density = exp(-h / mie_scale);
    let mie_scattering = max(sky.mie.x, 0.0) * max(sky.ground_albedo.w, 0.0);
    let mie = mie_scattering * mie_density;
    let ozone = max(sky.ozone.rgb, vec3<f32>(0.0))
        * max(0.0, 1.0 - abs(h - raa_atmosphere_ozone_centre_km) / raa_atmosphere_ozone_half_width_km);
    let extinction = rayleigh + vec3<f32>(mie + max(sky.mie.y, 0.0) * mie_density) + ozone;
    return raa_atmosphere_medium(rayleigh, mie, extinction);
}

fn raa_atmosphere_rayleigh_phase(cos_theta: f32) -> f32 {
    let c = clamp(cos_theta, -1.0, 1.0);
    return 3.0 * (1.0 + c * c) / (16.0 * raa_atmosphere_pi);
}

// Cornette-Shanks Mie phase, with `g` clamped away from the singular forward peak.
fn raa_atmosphere_mie_phase(cos_theta: f32, g: f32) -> f32 {
    let gg = clamp(g, -0.99, 0.99);
    let c = clamp(cos_theta, -1.0, 1.0);
    let k = 3.0 / (8.0 * raa_atmosphere_pi) * (1.0 - gg * gg) / (2.0 + gg * gg);
    let d = max(1.0 + gg * gg - 2.0 * gg * c, raa_atmosphere_min_scale);
    return k * (1.0 + c * c) / (d * sqrt(d));
}

// Distance along a ray to a sphere around the planet centre, or -1 if it misses.
fn raa_atmosphere_ray_sphere(origin: vec3<f32>, direction: vec3<f32>, radius: f32) -> f32 {
    let b = dot(origin, direction);
    let c = dot(origin, origin) - radius * radius;
    let discriminant = b * b - c;
    if (discriminant < 0.0) {
        return -1.0;
    }
    let root = sqrt(discriminant);
    if (-b - root > 0.0) {
        return -b - root;
    }
    if (-b + root > 0.0) {
        return -b + root;
    }
    return -1.0;
}

// Unit direction guard: a zero or non-finite length maps to the up vector instead of
// dividing by zero. Raw finite inputs keep their direction.
fn raa_atmosphere_unit_direction(direction: vec3<f32>) -> vec3<f32> {
    let len = length(direction);
    if (!(len > raa_atmosphere_min_scale)) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return direction / len;
}

// Bruneton's mapping for the transmittance LUT: more resolution near the horizon.
// The top-atmosphere span collapses at the ground radius, so its divisor is guarded.
fn raa_atmosphere_transmittance_uv(r: f32, mu: f32, sky: PhysicalSkyGpu) -> vec2<f32> {
    let ground = raa_atmosphere_ground_km(sky);
    let top = raa_atmosphere_top_km(sky);
    let h = sqrt(max(top * top - ground * ground, 0.0));
    let rr = clamp(r, ground, top);
    let mm = clamp(mu, -1.0, 1.0);
    let rho = sqrt(max(rr * rr - ground * ground, 0.0));
    let discriminant = rr * rr * (mm * mm - 1.0) + top * top;
    let d = max(0.0, -rr * mm + sqrt(max(discriminant, 0.0)));
    let d_min = top - rr;
    let d_max = rho + h;
    let span = max(d_max - d_min, raa_atmosphere_min_scale);
    return vec2<f32>((d - d_min) / span, rho / max(h, raa_atmosphere_min_scale));
}

// Inverse of `raa_atmosphere_transmittance_uv`; returns (r, mu).
fn raa_atmosphere_transmittance_params(uv: vec2<f32>, sky: PhysicalSkyGpu) -> vec2<f32> {
    let ground = raa_atmosphere_ground_km(sky);
    let top = raa_atmosphere_top_km(sky);
    let h = sqrt(max(top * top - ground * ground, 0.0));
    let rho = h * clamp(uv.y, 0.0, 1.0);
    let r = sqrt(rho * rho + ground * ground);
    let d_min = top - r;
    let d_max = rho + h;
    let d = d_min + clamp(uv.x, 0.0, 1.0) * max(d_max - d_min, 0.0);
    var mu = 1.0;
    if (d > raa_atmosphere_min_scale) {
        let divisor = max(2.0 * r * d, raa_atmosphere_min_scale);
        mu = clamp((h * h - rho * rho - d * d) / divisor, -1.0, 1.0);
    }
    return vec2<f32>(r, mu);
}

// Sky-view LUT mapping: world azimuth across U and square-rooted elevation across V,
// so the horizon where the sky changes fastest gets the most texels.
fn raa_atmosphere_sky_view_uv(direction: vec3<f32>) -> vec2<f32> {
    let d = raa_atmosphere_unit_direction(direction);
    let azimuth = atan2(d.z, d.x);
    let elevation = asin(clamp(d.y, -1.0, 1.0));
    let v = 0.5 + 0.5 * sign(elevation) * sqrt(abs(elevation) / (raa_atmosphere_pi * 0.5));
    return vec2<f32>(azimuth / raa_atmosphere_tau + 0.5, 1.0 - v);
}

// Inverse of `raa_atmosphere_sky_view_uv`.
fn raa_atmosphere_sky_view_direction(uv: vec2<f32>) -> vec3<f32> {
    let azimuth = (uv.x - 0.5) * raa_atmosphere_tau;
    let s = (1.0 - clamp(uv.y, 0.0, 1.0)) * 2.0 - 1.0;
    let elevation = sign(s) * s * s * raa_atmosphere_pi * 0.5;
    return vec3<f32>(cos(elevation) * cos(azimuth), sin(elevation), cos(elevation) * sin(azimuth));
}
