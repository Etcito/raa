// S39 screen integration: per-pixel indirect diffuse and ambient occlusion. It weights the
// four nearest grid probes bilinearly, down-weighting probes on other surfaces, evaluates
// their cosine-convolved SH irradiance towards the pixel's world normal, adds a horizon
// search in screen space for detail below the probe grid, and blends the result into the
// previous parity output with the S39 temporal alpha.
//
// Adapted from solstice@7e465b5,
// crates/solstice-render/shaders/gi/probe_integrate.wgsl:15-135. MIT License
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
//
// Group-2 binding registry of contract 'gi-screen-wgsl' (binding number to resource):
// integrate {0 raa_gi_frame, 1 raa_gi_depth, 2 raa_gi_normal, 3 raa_gi_motion,
// 4 raa_gi_previous, 5 raa_gi_probes, 6 raa_gi_output, 7 raa_gi_screen_params}.

// The S12 motion (previous uv = uv - motion).
@group(2) @binding(3) var raa_gi_motion: texture_2d<f32>;
// The other parity output: the previous integrate result.
@group(2) @binding(4) var raa_gi_previous: texture_2d<f32>;
// The filtered probes of `probe_project.wgsl`.
@group(2) @binding(5) var<storage, read> raa_gi_probes: array<Probe>;
@group(2) @binding(6) var raa_gi_output: texture_storage_2d<rgba16float, write>;

// Irradiance from harmonics towards `n`: radiance convolved with the cosine lobe. A constant
// radiance c gives pi c.
fn raa_gi_sh_irradiance(probe: Probe, n: vec3<f32>) -> vec3<f32> {
    let a0 = RAA_GI_PROBE_PI;
    let a1 = RAA_GI_PROBE_TAU / 3.0;
    let a2 = RAA_GI_PROBE_PI * 0.25;
    var e = probe.sh[0].xyz * 0.282095 * a0;
    e += (probe.sh[1].xyz * n.y + probe.sh[2].xyz * n.z + probe.sh[3].xyz * n.x) * 0.488603 * a1;
    e += (probe.sh[4].xyz * n.x * n.y * 1.092548
        + probe.sh[5].xyz * n.y * n.z * 1.092548
        + probe.sh[6].xyz * (3.0 * n.z * n.z - 1.0) * 0.315392
        + probe.sh[7].xyz * n.x * n.z * 1.092548
        + probe.sh[8].xyz * (n.x * n.x - n.y * n.y) * 0.546274) * a2;
    return max(e, vec3<f32>(0.0));
}

// Interleaved gradient noise (solstice `common/math.wgsl:42-45`), seeded with the frame
// index `raa_gi_frame.counts.y`, so it rotates each probe's occlusion search over time.
fn raa_gi_probe_ign(pixel: vec2<f32>, frame: u32) -> f32 {
    let p = pixel + 5.588238 * f32(frame % 64u);
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

// Ground-truth-style occlusion in a horizon search along a few screen-space directions.
fn raa_gi_screen_occlusion(pixel: vec2<f32>, world: vec3<f32>, normal: vec3<f32>, noise: f32) -> f32 {
    const RAA_GI_OCCLUSION_DIRECTIONS: u32 = 3u;
    const RAA_GI_OCCLUSION_STEPS: u32 = 4u;
    const RAA_GI_OCCLUSION_RADIUS: f32 = 0.9;
    let distance = length(world - raa_gi_frame.camera_position.xyz);
    let pixel_radius =
        clamp(RAA_GI_OCCLUSION_RADIUS / distance * raa_gi_frame.viewport.y * 0.8, 4.0, 90.0);
    let size = vec2<i32>(raa_gi_frame.viewport.xy);

    var occlusion = 0.0;
    for (var d = 0u; d < RAA_GI_OCCLUSION_DIRECTIONS; d++) {
        let angle = (f32(d) + noise) / f32(RAA_GI_OCCLUSION_DIRECTIONS) * RAA_GI_PROBE_PI;
        let direction = vec2<f32>(cos(angle), sin(angle));
        for (var side = -1.0; side <= 1.0; side += 2.0) {
            var horizon = 0.0;
            for (var s = 1u; s <= RAA_GI_OCCLUSION_STEPS; s++) {
                let step_fraction = f32(s) - 0.5 + noise * 0.5;
                let offset =
                    direction * side * pixel_radius * step_fraction / f32(RAA_GI_OCCLUSION_STEPS);
                let sample_pixel = clamp(vec2<i32>(pixel + offset), vec2<i32>(0), size - vec2<i32>(1));
                let depth = textureLoad(raa_gi_depth, sample_pixel, 0);
                if raa_depth_is_far(depth) {
                    continue;
                }
                let p = raa_gi_world_position(vec2<f32>(sample_pixel) + vec2<f32>(0.5), depth);
                let delta = p - world;
                let length_delta = length(delta);
                let elevation = dot(delta, normal) / max(length_delta, 1.0e-4);
                let falloff = clamp(1.0 - length_delta / RAA_GI_OCCLUSION_RADIUS, 0.0, 1.0);
                horizon = max(horizon, elevation * falloff);
            }
            occlusion += horizon;
        }
    }
    return clamp(1.0 - occlusion / f32(RAA_GI_OCCLUSION_DIRECTIONS * 2u) * 1.6, 0.0, 1.0);
}

// How well a probe stands for the surface at `world` with `normal`: 0 for an empty tile or a
// probe facing away by 90 degrees or more, else the cosine to the fourth times a falloff with
// the probe's distance from the pixel's tangent plane.
const RAA_GI_PROBE_MIN_WEIGHT: f32 = 1.0e-3;

fn raa_gi_probe_match(probe: Probe, world: vec3<f32>, normal: vec3<f32>, plane_scale: f32) -> f32 {
    if probe.position.w <= 0.0 {
        return 0.0;
    }
    let facing = max(dot(probe.normal.xyz, normal), 0.0);
    let facing2 = facing * facing;
    let plane = abs(dot(probe.position.xyz - world, normal));
    return facing2 * facing2 * exp(-plane * plane_scale);
}

// Handoff: card 'raa-gi screen: passes.rs module builders, group-2 layouts and pipelines'
// builds the integrate compute pipeline on `raa_gi_probe_integrate`; today it maps the pixel
// grid, weights the four nearest probes, evaluates `raa_gi_sh_irradiance`, adds
// `raa_gi_screen_occlusion` and blends into the previous parity output under the group-2
// registry above.
@compute @workgroup_size(8, 8)
fn raa_gi_probe_integrate(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<u32>(raa_gi_frame.viewport.xy);
    if any(id.xy >= size) {
        return;
    }
    let pixel = vec2<i32>(id.xy);
    let depth = textureLoad(raa_gi_depth, pixel, 0);
    if raa_depth_is_far(depth) {
        textureStore(raa_gi_output, pixel, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let center = vec2<f32>(id.xy) + vec2<f32>(0.5);
    let normal = raa_gi_world_normal(pixel);
    let world = raa_gi_world_position(center, depth);
    let grid = raa_gi_probe_grid(raa_gi_frame.viewport.xy);
    let grid_i = vec2<i32>(grid);

    // Bilinear over the four nearest tiles, keeping only probes on this pixel's surface. When
    // none of them lies on it (a silhouette or a face narrower than a tile), the 4 x 4 tiles
    // around are searched without the bilinear weight; with no matching probe there the pixel
    // gets no screen irradiance rather than the mean of probes on other surfaces. A floor
    // weight on every probe used to average exactly those, so the faces just outside a lit
    // room (a wall's outer side, the front of its backing) took the room's bounce light.
    let position = center / f32(RAA_GI_PROBE_SPACING) - vec2<f32>(0.5);
    let base = vec2<i32>(floor(position));
    let fraction = position - vec2<f32>(base);
    // Linear view distance: RAA's standard-Z projection replaces solstice's near / depth.
    let plane_scale = 6.0 / max(-raa_gi_view_position(center, depth).z, 1.0);

    var irradiance = vec3<f32>(0.0);
    var total = 0.0;
    for (var dy = 0; dy <= 1; dy++) {
        for (var dx = 0; dx <= 1; dx++) {
            let tile = clamp(base + vec2<i32>(dx, dy), vec2<i32>(0), grid_i - vec2<i32>(1));
            let probe = raa_gi_probes[u32(tile.y) * grid.x + u32(tile.x)];
            let bilinear = select(1.0 - fraction.x, fraction.x, dx == 1)
                * select(1.0 - fraction.y, fraction.y, dy == 1);
            let weight = max(bilinear, 0.001) * raa_gi_probe_match(probe, world, normal, plane_scale);
            if weight > 0.0 {
                irradiance += raa_gi_sh_irradiance(probe, normal) * weight;
                total += weight;
            }
        }
    }
    if total < RAA_GI_PROBE_MIN_WEIGHT {
        irradiance = vec3<f32>(0.0);
        total = 0.0;
        for (var dy = -1; dy <= 2; dy++) {
            for (var dx = -1; dx <= 2; dx++) {
                let tile = base + vec2<i32>(dx, dy);
                if any(tile < vec2<i32>(0)) || any(tile >= grid_i) {
                    continue;
                }
                let probe = raa_gi_probes[u32(tile.y) * grid.x + u32(tile.x)];
                let weight = raa_gi_probe_match(probe, world, normal, plane_scale);
                if weight > 0.0 {
                    irradiance += raa_gi_sh_irradiance(probe, normal) * weight;
                    total += weight;
                }
            }
        }
    }
    if total >= RAA_GI_PROBE_MIN_WEIGHT {
        irradiance /= total;
    } else {
        irradiance = vec3<f32>(0.0);
    }

    let noise = raa_gi_probe_ign(vec2<f32>(id.xy), raa_gi_frame.counts.y);
    let occlusion = raa_gi_screen_occlusion(center, world, normal, noise);
    var current = vec4<f32>(irradiance, occlusion);

    // Temporal blend with RAA's motion convention (previous uv = uv - motion). Without usable
    // history alpha is 1; otherwise the S39 alpha grows with the motion in pixels.
    let motion = textureLoad(raa_gi_motion, pixel, 0).xy;
    let previous_uv = center * raa_gi_frame.viewport.zw - motion;
    if raa_gi_screen_params.history_valid != 0u
        && all(previous_uv > vec2<f32>(0.0))
        && all(previous_uv < vec2<f32>(1.0))
    {
        let moved = length(motion * raa_gi_frame.viewport.xy);
        let alpha = min(0.08 + 0.03 * moved, 0.5);
        let previous_pixel = vec2<i32>(previous_uv * raa_gi_frame.viewport.xy);
        let history = textureLoad(raa_gi_previous, previous_pixel, 0);
        current = mix(history, current, alpha);
    }
    textureStore(raa_gi_output, pixel, current);
}
