/*
MIT License

Copyright (c) 2026 SamG-Coder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
// Adapted from solstice@7e465b599c13da165948ebaa44df824c36c3ae07
// shaders/sky/milky_way.wgsl:9-96 and common/math.wgsl:30-39,83-85.
const raa_milky_way_pi: f32 = 3.14159265359;
const raa_milky_way_tau: f32 = 6.28318530718;

fn raa_milky_way_pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn raa_milky_way_hash_to_unit(v: u32) -> f32 {
    return f32(raa_milky_way_pcg(v) >> 8u) * (1.0 / 16777216.0);
}

fn raa_milky_way_sq(x: f32) -> f32 {
    return x * x;
}

// The Milky Way, fixed on the celestial sphere, and the noise it and the moon are made from.
// The Milky Way costs too much to work out for every pixel of sky, so `sky/milky_way_bake.wgsl`
// works it out once into a map of the celestial sphere, which the sky samples.
//
// Self-contained prefixed helpers; no shared bindings or frame/time/weather reads.

// The galactic frame, in equatorial coordinates (J2000): towards the galactic centre in
// Sagittarius, towards galactic longitude 90 degrees in Cygnus, and the north galactic pole.
const raa_milky_way_galactic_centre: vec3f = vec3f(-0.0548756, -0.8734371, -0.4838350);
const raa_milky_way_galactic_90: vec3f = vec3f(0.4941094, -0.4448296, 0.7469822);
const raa_milky_way_galactic_pole: vec3f = vec3f(-0.8676661, -0.1980764, 0.4559838);

// A hash of an integer lattice point, to [0, 1).
fn raa_milky_way_lattice_hash(cell: vec3i, salt: u32) -> f32 {
    let c = bitcast<vec3u>(cell);
    return raa_milky_way_hash_to_unit(raa_milky_way_pcg(c.x ^ raa_milky_way_pcg(c.y ^ raa_milky_way_pcg(c.z ^ salt))));
}

// Smooth value noise in [0, 1].
fn raa_milky_way_value_noise(p: vec3f, salt: u32) -> f32 {
    let cell = vec3i(floor(p));
    let f = fract(p);
    let s = f * f * (3.0 - 2.0 * f);
    let x00 = mix(raa_milky_way_lattice_hash(cell, salt), raa_milky_way_lattice_hash(cell + vec3i(1, 0, 0), salt), s.x);
    let x10 = mix(raa_milky_way_lattice_hash(cell + vec3i(0, 1, 0), salt), raa_milky_way_lattice_hash(cell + vec3i(1, 1, 0), salt), s.x);
    let x01 = mix(raa_milky_way_lattice_hash(cell + vec3i(0, 0, 1), salt), raa_milky_way_lattice_hash(cell + vec3i(1, 0, 1), salt), s.x);
    let x11 = mix(raa_milky_way_lattice_hash(cell + vec3i(0, 1, 1), salt), raa_milky_way_lattice_hash(cell + vec3i(1, 1, 1), salt), s.x);
    return mix(mix(x00, x10, s.y), mix(x01, x11, s.y), s.z);
}

// Fractal noise in [0, 1]: `octaves` layers of value noise, each twice as fine and half as
// strong.
fn raa_milky_way_fbm(p: vec3f, octaves: u32, salt: u32) -> f32 {
    var sum = 0.0;
    var amplitude = 0.5;
    var total = 0.0;
    var q = p;
    for (var i = 0u; i < octaves; i++) {
        sum += amplitude * raa_milky_way_value_noise(q, salt + i * 101u);
        total += amplitude;
        amplitude *= 0.5;
        q = q * 2.03 + vec3f(1.7, 9.2, 4.1);
    }
    return sum / total;
}

// The Milky Way's glow towards `c` (celestial), and in A how dense its faint stars are there.
fn raa_milky_way_glow(c: vec3f) -> vec4f {
    let g = vec3f(dot(c, raa_milky_way_galactic_centre), dot(c, raa_milky_way_galactic_90), dot(c, raa_milky_way_galactic_pole));
    let latitude = asin(clamp(g.z, -1.0, 1.0));
    let longitude = atan2(g.y, g.x);
    let toward_centre = 0.5 + 0.5 * cos(longitude);
    // The disc, thickening and brightening towards the bulge round the centre.
    let width = 0.07 + 0.07 * pow(toward_centre, 4.0);
    let disc = exp(-raa_milky_way_sq(latitude / width)) * (0.2 + 0.8 * pow(toward_centre, 2.5));
    let bulge = exp(-(raa_milky_way_sq(longitude / 0.3) + raa_milky_way_sq(latitude / 0.2)));
    // Star clouds, mottled at several scales, and grainy with stars too faint to see one by one.
    let clouds = raa_milky_way_fbm(g * 7.0, 7u, 71u);
    let grain = 0.75 + 0.5 * raa_milky_way_value_noise(g * 260.0, 73u);
    var glow = (disc * (0.3 + 1.4 * clouds * clouds) + bulge * 0.8) * grain;
    // Dust: dark lanes along the plane, deepest in the Great Rift from Cygnus to Sagittarius.
    // The lane wanders and swells along its length, thickens and thins into clumps, frays at
    // its edges, and throws off fainter side lanes.
    let rift = smoothstep(-0.5, -0.1, longitude) * smoothstep(1.6, 1.1, longitude);
    let lane_offset = 0.012 + 0.03 * (raa_milky_way_fbm(g * 5.0, 5u, 83u) - 0.5);
    let lane_width = 0.01 + 0.024 * raa_milky_way_fbm(g * 8.0, 4u, 87u);
    let ragged = (raa_milky_way_fbm(g * 30.0, 5u, 89u) - 0.5) * 0.014;
    let opacity = smoothstep(0.25, 0.75, raa_milky_way_fbm(g * 11.0, 5u, 91u));
    let lane = exp(-raa_milky_way_sq((latitude - lane_offset + ragged) / lane_width)) * rift * mix(0.35, 0.95, opacity);
    let side_offset = -0.035 + 0.04 * (raa_milky_way_fbm(g * 6.0, 4u, 93u) - 0.5);
    let side = exp(-raa_milky_way_sq((latitude - side_offset + ragged) / 0.008)) * rift * smoothstep(0.5, 0.8, raa_milky_way_fbm(g * 14.0, 4u, 95u));
    let filaments = smoothstep(0.52, 0.78, raa_milky_way_fbm(g * 45.0, 5u, 97u)) * exp(-raa_milky_way_sq(latitude / 0.08));
    let dust = clamp(lane + side * 0.6 + filaments * 0.4, 0.0, 0.9);
    glow *= 1.0 - dust;
    // Old yellow stars towards the core, young blue ones out along the arms; dust reddens.
    let color = mix(vec3f(0.78, 0.86, 1.0), vec3f(1.0, 0.86, 0.68), pow(toward_centre, 3.0))
        * mix(vec3f(1.0), vec3f(1.0, 0.78, 0.6), dust);
    return vec4f(color * glow, disc * (1.0 - dust));
}

// Where `c`, a direction on the celestial sphere, falls on the baked map: right ascension across,
// from 0 to 24 hours, and declination down, from the north pole.
fn raa_milky_way_celestial_uv(c: vec3f) -> vec2f {
    return vec2f(atan2(c.y, c.x) / raa_milky_way_tau + 0.5, acos(clamp(c.z, -1.0, 1.0)) / raa_milky_way_pi);
}

// The direction on the celestial sphere at `uv` on the baked map.
fn raa_milky_way_celestial_direction(uv: vec2f) -> vec3f {
    let ra = (uv.x - 0.5) * raa_milky_way_tau;
    let polar = uv.y * raa_milky_way_pi;
    return vec3f(sin(polar) * cos(ra), sin(polar) * sin(ra), cos(polar));
}

// The map stores the square root of the glow over this range, for precision in the faint parts
// of eight-bit channels.
const raa_milky_way_range: f32 = 4.0;
