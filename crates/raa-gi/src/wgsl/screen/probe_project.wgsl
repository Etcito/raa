// S39 'project and filter': raa_gi_probe_project compresses each probe's P x P traced rays
// into SH9 (the lower hemisphere mirrored about the probe normal, carrying the mean
// radiance), and raa_gi_probe_filter blends each probe with its neighbours on the same
// surface. Composed after probe_common.wgsl (project module of contract 'gi-screen-wgsl').
//
// Group-2 bindings used: raa_gi_probe_project {0 raa_gi_frame, 1 raa_gi_depth,
// 2 raa_gi_normal, 3 raa_gi_rays, 5 raa_gi_probes_out}; raa_gi_probe_filter {0 raa_gi_frame,
// 4 raa_gi_probes_in, 5 raa_gi_probes_out}.
//
// Adapted from solstice@7e465b5,
// crates/solstice-render/shaders/gi/probe_project.wgsl:13-125. MIT License
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

// The ray atlas written by the trace: (rgb radiance, a hit distance) per ray texel.
@group(2) @binding(3) var raa_gi_rays: texture_2d<f32>;
@group(2) @binding(4) var<storage, read> raa_gi_probes_in: array<Probe>;
@group(2) @binding(5) var<storage, read_write> raa_gi_probes_out: array<Probe>;

fn raa_gi_sh_basis(d: vec3<f32>) -> array<f32, 9> {
    return array<f32, 9>(
        0.282095,
        0.488603 * d.y,
        0.488603 * d.z,
        0.488603 * d.x,
        1.092548 * d.x * d.y,
        1.092548 * d.y * d.z,
        0.315392 * (3.0 * d.z * d.z - 1.0),
        1.092548 * d.x * d.z,
        0.546274 * (d.x * d.x - d.y * d.y),
    );
}

@compute @workgroup_size(8, 8)
fn raa_gi_probe_project(@builtin(global_invocation_id) id: vec3<u32>) {
    let grid = raa_gi_probe_grid(raa_gi_frame.viewport.xy);
    if any(id.xy >= grid) {
        return;
    }
    let tile = id.xy;
    let slot = tile.y * grid.x + tile.x;
    let frame = raa_gi_frame.counts.y;
    var probe: Probe;

    let pixel = raa_gi_probe_pixel(tile, raa_gi_frame.viewport.xy, frame);
    let depth = textureLoad(raa_gi_depth, vec2<i32>(pixel), 0);
    if raa_depth_is_far(depth) {
        probe.position = vec4<f32>(0.0);
        raa_gi_probes_out[slot] = probe;
        return;
    }
    let normal = raa_gi_world_normal(vec2<i32>(pixel));
    probe.position = vec4<f32>(raa_gi_world_position(vec2<f32>(pixel) + vec2<f32>(0.5), depth), 1.0);
    probe.normal = vec4<f32>(normal, 0.0);

    // The rays cover the upper hemisphere only; mirror each sample into the lower
    // hemisphere carrying the mean radiance so the harmonics do not ring below the horizon.
    var mean = vec3<f32>(0.0);
    for (var y = 0u; y < PROBE_RAYS; y++) {
        for (var x = 0u; x < PROBE_RAYS; x++) {
            let texel = tile * PROBE_RAYS + vec2<u32>(x, y);
            mean += textureLoad(raa_gi_rays, vec2<i32>(texel), 0).rgb;
        }
    }
    let ray_count = f32(PROBE_RAYS * PROBE_RAYS);
    mean /= ray_count;

    // Each ray stands for an equal part of the 2 pi steradian hemisphere.
    let weight = RAA_GI_PROBE_TAU / ray_count;
    for (var y = 0u; y < PROBE_RAYS; y++) {
        for (var x = 0u; x < PROBE_RAYS; x++) {
            let texel = tile * PROBE_RAYS + vec2<u32>(x, y);
            let radiance = textureLoad(raa_gi_rays, vec2<i32>(texel), 0).rgb;
            let direction = raa_gi_probe_ray_direction(tile, vec2<u32>(x, y), normal, frame);
            let mirrored = direction - 2.0 * dot(direction, normal) * normal;
            let basis_up = raa_gi_sh_basis(direction);
            let basis_down = raa_gi_sh_basis(mirrored);
            for (var i = 0; i < 9; i++) {
                let contribution = radiance * basis_up[i] + mean * basis_down[i];
                probe.sh[i] = vec4<f32>(probe.sh[i].xyz + contribution * weight, 0.0);
            }
        }
    }
    raa_gi_probes_out[slot] = probe;
}

// Blends each probe with its 3x3 neighbours lying on the same surface.
@compute @workgroup_size(8, 8)
fn raa_gi_probe_filter(@builtin(global_invocation_id) id: vec3<u32>) {
    let grid = raa_gi_probe_grid(raa_gi_frame.viewport.xy);
    if any(id.xy >= grid) {
        return;
    }
    let slot = id.y * grid.x + id.x;
    var probe = raa_gi_probes_in[slot];
    if probe.position.w <= 0.0 {
        raa_gi_probes_out[slot] = probe;
        return;
    }

    var total = 1.0;
    let camera = raa_gi_frame.camera_position.xyz;
    let plane_scale = 4.0 / max(length(probe.position.xyz - camera), 1.0);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if dx == 0 && dy == 0 {
                continue;
            }
            let c = vec2<i32>(id.xy) + vec2<i32>(dx, dy);
            if any(c < vec2<i32>(0)) || any(c >= vec2<i32>(grid)) {
                continue;
            }
            let other = raa_gi_probes_in[u32(c.y) * grid.x + u32(c.x)];
            if other.position.w <= 0.0 {
                continue;
            }
            let normal_weight = pow(max(dot(other.normal.xyz, probe.normal.xyz), 0.0), 8.0);
            let plane = abs(dot(other.position.xyz - probe.position.xyz, probe.normal.xyz));
            let weight = normal_weight * exp(-plane * plane_scale * 8.0);
            if weight > 1.0e-3 {
                for (var i = 0; i < 9; i++) {
                    probe.sh[i] += other.sh[i] * weight;
                }
                total += weight;
            }
        }
    }
    for (var i = 0; i < 9; i++) {
        probe.sh[i] /= total;
    }
    raa_gi_probes_out[slot] = probe;
}
