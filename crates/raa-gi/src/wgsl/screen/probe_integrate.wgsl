// S39 screen integration: per-pixel indirect diffuse and ambient occlusion. It weights the
// four nearest grid probes bilinearly, down-weighting probes on other surfaces, evaluates
// their cosine-convolved SH irradiance towards the pixel's world normal, adds a horizon
// search in screen space for detail below the probe grid, and hands the result to the shared
// temporal accumulation of accumulate.wgsl (disocclusion test, clamp, 1 / n blend).
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
// 4 raa_gi_previous, 5 raa_gi_probes, 6 raa_gi_output, 7 raa_gi_screen_params,
// 8 raa_gi_meta_previous, 9 raa_gi_meta_out, 10 raa_gi_filter_out, 11 raa_gi_debug_out,
// 12 raa_gi_counters}. Composed after accumulate.wgsl, which declares 3, 4, 6 and 8 to 12 and
// holds the temporal accumulation, the occlusion search and its noise.

// The filtered probes of `probe_project.wgsl`.
@group(2) @binding(5) var<storage, read> raa_gi_probes: array<Probe>;

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

// Running weighted sums of the probe irradiances a pixel blends: their total, the irradiance and
// the luminance moments, for the clamp's spatial statistics.
struct RaaGiProbeSum {
    irradiance: vec3<f32>,
    total: f32,
    l1: f32,
    l2: f32,
}

fn raa_gi_probe_add(sum: ptr<function, RaaGiProbeSum>, probe: Probe, normal: vec3<f32>, weight: f32) {
    let e = raa_gi_sh_irradiance(probe, normal);
    let l = raa_gi_acc_luminance(e);
    (*sum).irradiance += e * weight;
    (*sum).total += weight;
    (*sum).l1 += l * weight;
    (*sum).l2 += l * l * weight;
}

// Handoff: card 'raa-gi screen: passes.rs module builders, group-2 layouts and pipelines'
// builds the integrate compute pipeline on `raa_gi_probe_integrate`; it maps the pixel grid,
// weights the four nearest probes, evaluates `raa_gi_sh_irradiance`, adds
// `raa_gi_screen_occlusion` and hands the result to the shared accumulation
// (`raa_gi_acc_resolve`) under the group-2 registry above.
@compute @workgroup_size(8, 8)
fn raa_gi_probe_integrate(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<u32>(raa_gi_frame.viewport.xy);
    if any(id.xy >= size) {
        return;
    }
    let pixel = vec2<i32>(id.xy);
    let depth = textureLoad(raa_gi_depth, pixel, 0);
    if raa_depth_is_far(depth) {
        raa_gi_acc_store_far(pixel);
        return;
    }
    let center = vec2<f32>(id.xy) + vec2<f32>(0.5);
    let normal = raa_gi_world_normal(pixel);
    let world = raa_gi_world_position(center, depth);
    let view_depth = -raa_gi_view_position(center, depth).z;
    let grid = raa_gi_probe_grid(raa_gi_frame.viewport.xy);
    let grid_i = vec2<i32>(grid);

    // Bilinear over the four nearest tiles, keeping only probes on this pixel's surface. When
    // none of them lies on it (a silhouette or a face narrower than a tile), the 4 x 4 tiles
    // around are searched without the bilinear weight; with no matching probe there the pixel
    // gets no irradiance this frame and holds its history.
    let position = center / f32(RAA_GI_PROBE_SPACING) - vec2<f32>(0.5);
    let base = vec2<i32>(floor(position));
    let fraction = position - vec2<f32>(base);
    // Linear view distance: RAA's standard-Z projection replaces solstice's near / depth.
    let plane_scale = 6.0 / max(view_depth, 1.0);

    var sum = RaaGiProbeSum(vec3<f32>(0.0), 0.0, 0.0, 0.0);
    for (var dy = 0; dy <= 1; dy++) {
        for (var dx = 0; dx <= 1; dx++) {
            let tile = clamp(base + vec2<i32>(dx, dy), vec2<i32>(0), grid_i - vec2<i32>(1));
            let probe = raa_gi_probes[u32(tile.y) * grid.x + u32(tile.x)];
            let bilinear = select(1.0 - fraction.x, fraction.x, dx == 1)
                * select(1.0 - fraction.y, fraction.y, dy == 1);
            let weight = max(bilinear, 0.001) * raa_gi_probe_match(probe, world, normal, plane_scale);
            if weight > 0.0 {
                raa_gi_probe_add(&sum, probe, normal, weight);
            }
        }
    }
    if sum.total < RAA_GI_PROBE_MIN_WEIGHT {
        sum = RaaGiProbeSum(vec3<f32>(0.0), 0.0, 0.0, 0.0);
        for (var dy = -1; dy <= 2; dy++) {
            for (var dx = -1; dx <= 2; dx++) {
                let tile = base + vec2<i32>(dx, dy);
                if any(tile < vec2<i32>(0)) || any(tile >= grid_i) {
                    continue;
                }
                let probe = raa_gi_probes[u32(tile.y) * grid.x + u32(tile.x)];
                let weight = raa_gi_probe_match(probe, world, normal, plane_scale);
                if weight > 0.0 {
                    raa_gi_probe_add(&sum, probe, normal, weight);
                }
            }
        }
    }
    let matched = sum.total >= RAA_GI_PROBE_MIN_WEIGHT;
    var irradiance = vec3<f32>(0.0);
    var mean = 0.0;
    var spread = 0.0;
    if matched {
        irradiance = sum.irradiance / sum.total;
        mean = sum.l1 / sum.total;
        spread = max(sum.l2 / sum.total - mean * mean, 0.0);
    }

    let noise = raa_gi_probe_ign(vec2<f32>(id.xy), raa_gi_frame.counts.y);
    let occlusion = raa_gi_screen_occlusion(center, world, normal, noise);

    // Debug: the hit code of the pixel's own tile probe (probe_project.wgsl stores it in
    // normal.w) and bit 16 on the pixels next to where that probe sits this frame.
    let own_tile = clamp(vec2<i32>(id.xy) / i32(RAA_GI_PROBE_SPACING), vec2<i32>(0), grid_i - vec2<i32>(1));
    let own_probe = raa_gi_probes[u32(own_tile.y) * grid.x + u32(own_tile.x)];
    var hit_code = u32(max(own_probe.normal.w, 0.0));
    let probe_pixel = raa_gi_probe_pixel(vec2<u32>(own_tile), raa_gi_frame.viewport.xy, raa_gi_frame.counts.y);
    let away = abs(vec2<i32>(probe_pixel) - pixel);
    if max(away.x, away.y) <= 1 {
        hit_code |= 0x10000u;
    }

    raa_gi_acc_resolve(RaaGiAccInput(
        pixel,
        world,
        normal,
        view_depth,
        vec4<f32>(irradiance, occlusion),
        matched,
        mean,
        spread,
        hit_code,
        0.0,
    ));
}
