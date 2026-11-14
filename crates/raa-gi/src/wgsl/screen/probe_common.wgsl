// S39 screen probes: the records, group-2 bindings and maths shared by the probe passes. It
// places one probe per 16-pixel tile, jitters it inside the tile each frame, stratifies and
// rotates each probe's ray set per frame, and reconstructs a pixel's view and world position
// and world normal from the S12 depth and normal targets.
//
// Adapted from solstice@7e465b5,
// solstice:crates/solstice-render/shaders/gi/probe_common.wgsl:1-50 and
// solstice:crates/solstice-render/shaders/common/math.wgsl:30-34 (pcg), :37-39
// (hash_to_unit) and :61-70 (orthonormal_basis); the view and world positions follow
// solstice:crates/solstice-render/shaders/common/frame.wgsl:69-73. MIT License
//
// Copyright (c) 2026 SamG-Coder
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

// Each probe covers this many pixels on a side.
const RAA_GI_PROBE_SPACING: u32 = 16u;
// Each probe traces this many rays on a side; passes override it from the quality setting.
override PROBE_RAYS: u32 = 8u;

const RAA_GI_PROBE_PI: f32 = 3.14159265359;
const RAA_GI_PROBE_TAU: f32 = 6.28318530718;

// The frame record of contract 'gi-records' (432 B), shared by every GI compute pass.
struct GiFrameGpu {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    previous_view_proj: mat4x4<f32>,
    camera_position: vec4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    viewport: vec4<f32>,
    counts: vec4<u32>,
    settings: vec4<f32>,
    flags: vec4<u32>,
}

// The screen-pass state (16 B). history_valid is 1 when the other parity output holds the
// previous integrate result; screen_traces is 1 when screen traces are on and binding 3 holds
// the previous frame's lit HDR; parity is the index (0 or 1) of the output written this frame.
struct GiScreenGpu {
    history_valid: u32,
    screen_traces: u32,
    parity: u32,
    pad: u32,
}

// One grid probe (176 B): SH9 with rgb in xyz, the world position (w = 1 on geometry, 0 on a
// far or sky pixel) and the world unit normal.
struct Probe {
    sh: array<vec4<f32>, 9>,
    position: vec4<f32>,
    normal: vec4<f32>,
}

@group(2) @binding(0) var<uniform> raa_gi_frame: GiFrameGpu;
@group(2) @binding(1) var raa_gi_depth: texture_depth_2d;
@group(2) @binding(2) var raa_gi_normal: texture_2d<f32>;
@group(2) @binding(7) var<uniform> raa_gi_screen_params: GiScreenGpu;

// PCG hash (contract 'gi-screen-cpu'): a well-distributed 32-bit integer hash.
fn raa_gi_probe_pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Hash to a float in [0, 1).
fn raa_gi_probe_hash_to_unit(v: u32) -> f32 {
    return f32(raa_gi_probe_pcg(v) >> 8u) * (1.0 / 16777216.0);
}

// An orthonormal basis around `n` (Duff et al. 2017), returned as three columns.
fn raa_gi_probe_basis(n: vec3<f32>) -> mat3x3<f32> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    return mat3x3<f32>(
        vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x),
        vec3<f32>(b, s + n.y * n.y * a, -n.y),
        n,
    );
}

// The probe grid of a viewport: one probe per 16-pixel tile.
fn raa_gi_probe_grid(viewport: vec2<f32>) -> vec2<u32> {
    return (vec2<u32>(viewport) + vec2<u32>(RAA_GI_PROBE_SPACING - 1u)) / RAA_GI_PROBE_SPACING;
}

// The pixel a probe sits on this frame. Moving it within its tile each frame means that, over
// time, probes sample every part of the surface.
fn raa_gi_probe_pixel(tile: vec2<u32>, viewport: vec2<f32>, frame: u32) -> vec2<u32> {
    let seed = tile.x * 7919u + tile.y * 104729u + (frame % 16u) * 31337u;
    let jitter = vec2<u32>(
        raa_gi_probe_pcg(seed) % RAA_GI_PROBE_SPACING,
        raa_gi_probe_pcg(seed + 1u) % RAA_GI_PROBE_SPACING,
    );
    return min(tile * RAA_GI_PROBE_SPACING + jitter, vec2<u32>(viewport) - vec2<u32>(1u));
}

// The stratified uv of one ray, rotated per frame so the fixed ray count acts like a much
// larger one once the temporal filter averages it.
fn raa_gi_probe_ray_uv(tile: vec2<u32>, ray: vec2<u32>, frame: u32) -> vec2<f32> {
    let seed = tile.x * 2957u + tile.y * 6113u + frame * 409u;
    let jitter = vec2<f32>(
        raa_gi_probe_hash_to_unit(seed + ray.x * 17u + ray.y * 257u),
        raa_gi_probe_hash_to_unit(seed + ray.x * 131u + ray.y * 7u + 1u),
    );
    let rotation = raa_gi_probe_hash_to_unit(seed + 3u);
    var uv = (vec2<f32>(ray) + jitter) / f32(PROBE_RAYS);
    uv.y = fract(uv.y + rotation);
    return uv;
}

// Maps [0, 1]^2 onto the unit hemisphere around +Z with even area, in `normal`'s frame.
fn raa_gi_hemisphere_direction(uv: vec2<f32>, normal: vec3<f32>) -> vec3<f32> {
    let cos_theta = 1.0 - uv.x;
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let phi = RAA_GI_PROBE_TAU * uv.y;
    let local = vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    return raa_gi_probe_basis(normal) * local;
}

// A probe's ray direction in world space.
fn raa_gi_probe_ray_direction(
    tile: vec2<u32>,
    ray: vec2<u32>,
    normal: vec3<f32>,
    frame: u32,
) -> vec3<f32> {
    return raa_gi_hemisphere_direction(raa_gi_probe_ray_uv(tile, ray, frame), normal);
}

// The view position of a pixel-centre coordinate at standard-Z clip depth `depth`: inv_proj
// of its NDC point, divided by w.
fn raa_gi_view_position(pixel: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(
        2.0 * pixel.x * raa_gi_frame.viewport.z - 1.0,
        1.0 - 2.0 * pixel.y * raa_gi_frame.viewport.w,
        depth,
        1.0,
    );
    let view = raa_gi_frame.inv_proj * ndc;
    return view.xyz / view.w;
}

// The world position of a pixel-centre coordinate at standard-Z clip depth `depth`.
fn raa_gi_world_position(pixel: vec2<f32>, depth: f32) -> vec3<f32> {
    let world = raa_gi_frame.inv_view * vec4<f32>(raa_gi_view_position(pixel, depth), 1.0);
    return world.xyz;
}

// The S12 view-space normal at `pixel` (xyz, roughness w), rotated into world space.
fn raa_gi_world_normal(pixel: vec2<i32>) -> vec3<f32> {
    let view_normal = normalize(textureLoad(raa_gi_normal, pixel, 0).xyz);
    return normalize((raa_gi_frame.inv_view * vec4<f32>(view_normal, 0.0)).xyz);
}
