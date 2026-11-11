// Cloud noise bake: shape, detail and weather volumes (raa-sky cloud noise compute).
//
// Adapted from Solstice revision 7e465b599c13da165948ebaa44df824c36c3ae07,
// crates/solstice-render/shaders/clouds/noise.wgsl:1-172, after Schneider 2015,
// "The Real-time Volumetric Cloudscapes of Horizon: Zero Dawn".
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
// Every helper and global is prefixed `raa_cloud_noise_`. Groups 0 and 1 are empty;
// the volume write is group 2 binding 0 and the weather map group 2 binding 1.

@group(2) @binding(0) var raa_cloud_noise_volume_out: texture_storage_3d<rgba8unorm, write>;
@group(2) @binding(1) var raa_cloud_noise_map_out: texture_storage_2d<rgba8unorm, write>;

fn raa_cloud_noise_hash(p: vec3i, seed: u32) -> u32 {
    var h = (u32(p.x) * 0x8da6b343u) ^ (u32(p.y) * 0xd8163841u)
        ^ (u32(p.z) * 0xcb1ab31fu) ^ (seed * 0x9e3779b1u);
    h ^= h >> 15u;
    h *= 0x2c1b3c6du;
    h ^= h >> 12u;
    h *= 0x297a2d39u;
    return h ^ (h >> 15u);
}

fn raa_cloud_noise_unit(h: u32) -> f32 {
    return f32(h >> 8u) / 16777216.0;
}

fn raa_cloud_noise_gradient(h: u32) -> vec3f {
    switch h % 12u {
        case 0u: { return vec3f(1.0, 1.0, 0.0); }
        case 1u: { return vec3f(-1.0, 1.0, 0.0); }
        case 2u: { return vec3f(1.0, -1.0, 0.0); }
        case 3u: { return vec3f(-1.0, -1.0, 0.0); }
        case 4u: { return vec3f(1.0, 0.0, 1.0); }
        case 5u: { return vec3f(-1.0, 0.0, 1.0); }
        case 6u: { return vec3f(1.0, 0.0, -1.0); }
        case 7u: { return vec3f(-1.0, 0.0, -1.0); }
        case 8u: { return vec3f(0.0, 1.0, 1.0); }
        case 9u: { return vec3f(0.0, -1.0, 1.0); }
        case 10u: { return vec3f(0.0, 1.0, -1.0); }
        default: { return vec3f(0.0, -1.0, -1.0); }
    }
}

fn raa_cloud_noise_wrap(c: vec3i, period: vec3i) -> vec3i {
    return ((c % period) + period) % period;
}

// Perlin noise in roughly -1..1, tiling every `period` cells along each axis.
fn raa_cloud_noise_perlin(p: vec3f, period: vec3i, seed: u32) -> f32 {
    let cell = vec3i(floor(p));
    let f = p - floor(p);
    let fade = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    var corners: array<f32, 8>;
    for (var i = 0; i < 8; i++) {
        let offset = vec3i(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        let g = raa_cloud_noise_gradient(raa_cloud_noise_hash(
            raa_cloud_noise_wrap(cell + offset, period),
            seed,
        ));
        corners[i] = dot(g, f - vec3f(offset));
    }
    let x0 = mix(corners[0], corners[1], fade.x);
    let x1 = mix(corners[2], corners[3], fade.x);
    let x2 = mix(corners[4], corners[5], fade.x);
    let x3 = mix(corners[6], corners[7], fade.x);
    return mix(mix(x0, x1, fade.y), mix(x2, x3, fade.y), fade.z);
}

fn raa_cloud_noise_perlin_fbm(p: vec3f, period: vec3i, octaves: i32, seed: u32) -> f32 {
    var sum = 0.0;
    var amplitude = 0.5;
    var total = 0.0;
    var scale = 1;
    for (var i = 0; i < octaves; i++) {
        sum += amplitude * raa_cloud_noise_perlin(
            p * f32(scale),
            period * scale,
            seed + u32(i),
        );
        total += amplitude;
        amplitude *= 0.5;
        scale *= 2;
    }
    return sum / total;
}

// Worley noise in 0..1: 1 at the middle of a cell, falling to 0 a cell away.
fn raa_cloud_noise_worley(p: vec3f, period: i32, seed: u32) -> f32 {
    let cell = vec3i(floor(p));
    let f = p - floor(p);
    var nearest = 9.0;
    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let offset = vec3i(x, y, z);
                let h = raa_cloud_noise_hash(
                    raa_cloud_noise_wrap(cell + offset, vec3i(period)),
                    seed,
                );
                let point = vec3f(offset) + vec3f(
                    raa_cloud_noise_unit(h),
                    raa_cloud_noise_unit(h * 747796405u + 1u),
                    raa_cloud_noise_unit(h * 2891336453u + 7u),
                );
                nearest = min(nearest, length(point - f));
            }
        }
    }
    return clamp(1.0 - nearest, 0.0, 1.0);
}

// Three octaves of Worley noise, starting at `period` cells across the tile.
fn raa_cloud_noise_worley_fbm(uvw: vec3f, period: i32, seed: u32) -> f32 {
    return raa_cloud_noise_worley(uvw * f32(period), period, seed) * 0.625
        + raa_cloud_noise_worley(uvw * f32(period * 2), period * 2, seed + 1u) * 0.25
        + raa_cloud_noise_worley(uvw * f32(period * 4), period * 4, seed + 2u) * 0.125;
}

fn raa_cloud_noise_remap(x: f32, low: f32, high: f32, new_low: f32, new_high: f32) -> f32 {
    return new_low + (x - low) / (high - low) * (new_high - new_low);
}

// Stretches a measured narrow band over 0..1 so thresholds mean what they say.
fn raa_cloud_noise_spread(x: f32, low: f32, high: f32) -> f32 {
    return clamp((x - low) / (high - low), 0.0, 1.0);
}

@compute @workgroup_size(4, 4, 4)
fn raa_cloud_noise_shape(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(raa_cloud_noise_volume_out);
    if any(id >= size) {
        return;
    }
    let uvw = (vec3f(id) + 0.5) / vec3f(size);
    // Billowy Perlin noise, pinched into puffs by Worley noise.
    let perlin_value = raa_cloud_noise_perlin_fbm(uvw * 4.0, vec3i(4), 6, 11u) * 0.5 + 0.5;
    let cells = raa_cloud_noise_worley_fbm(uvw, 4, 21u);
    let perlin_worley = clamp(
        raa_cloud_noise_remap(perlin_value, 0.0, 1.0, cells, 1.0),
        0.0,
        1.0,
    );
    let value = vec4f(
        raa_cloud_noise_spread(perlin_worley, 0.58, 0.88),
        raa_cloud_noise_spread(raa_cloud_noise_worley_fbm(uvw, 4, 31u), 0.2, 0.74),
        raa_cloud_noise_spread(raa_cloud_noise_worley_fbm(uvw, 8, 41u), 0.2, 0.74),
        raa_cloud_noise_spread(raa_cloud_noise_worley_fbm(uvw, 16, 51u), 0.22, 0.74),
    );
    textureStore(raa_cloud_noise_volume_out, id, value);
}

@compute @workgroup_size(4, 4, 4)
fn raa_cloud_noise_detail(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(raa_cloud_noise_volume_out);
    if any(id >= size) {
        return;
    }
    let uvw = (vec3f(id) + 0.5) / vec3f(size);
    textureStore(raa_cloud_noise_volume_out, id, vec4f(
        raa_cloud_noise_spread(raa_cloud_noise_worley_fbm(uvw, 2, 61u), 0.24, 0.74),
        raa_cloud_noise_spread(raa_cloud_noise_worley_fbm(uvw, 4, 71u), 0.24, 0.74),
        raa_cloud_noise_spread(raa_cloud_noise_worley_fbm(uvw, 8, 81u), 0.2, 0.72),
        1.0,
    ));
}

@compute @workgroup_size(8, 8)
fn raa_cloud_noise_weather(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(raa_cloud_noise_map_out);
    if any(id.xy >= size) {
        return;
    }
    let uv = vec3f((vec2f(id.xy) + 0.5) / vec2f(size), 0.37);
    // Where cloud can form, in clumps; only the order matters, not the spread.
    let field = raa_cloud_noise_perlin_fbm(uv * vec3f(8.0, 8.0, 1.0), vec3i(8, 8, 1), 5, 91u) * 0.5 + 0.5;
    let clumps = raa_cloud_noise_worley(uv * vec3f(12.0, 12.0, 1.0), 12, 101u);
    let coverage = raa_cloud_noise_spread(field * 0.75 + clumps * 0.25, 0.32, 0.68);
    let kind = raa_cloud_noise_spread(
        raa_cloud_noise_perlin_fbm(uv * vec3f(3.0, 3.0, 1.0), vec3i(3, 3, 1), 3, 111u) * 0.5 + 0.5,
        0.3,
        0.7,
    );
    // Cirrus: streaks drawn out along X, warped so they curve and fray.
    let warp = raa_cloud_noise_perlin_fbm(uv * vec3f(4.0, 4.0, 1.0), vec3i(4, 4, 1), 3, 121u);
    let streaks = raa_cloud_noise_perlin_fbm(
        vec3f(uv.x * 3.0, (uv.y + warp * 0.06) * 24.0, 0.37),
        vec3i(3, 24, 1),
        5,
        131u,
    );
    let cirrus = clamp(streaks * 1.3 + 0.5, 0.0, 1.0)
        * (raa_cloud_noise_perlin_fbm(uv * vec3f(5.0, 5.0, 1.0), vec3i(5, 5, 1), 3, 141u) * 0.5 + 0.6);
    textureStore(raa_cloud_noise_map_out, id.xy, vec4f(coverage, kind, clamp(cirrus, 0.0, 1.0), 1.0));
}
