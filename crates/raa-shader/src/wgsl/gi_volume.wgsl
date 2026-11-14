// S37 GI volume tracing: level lookup, atlas sampling, sphere tracing, visibility and the
// outer-edge fade over the clipmap atlas. Consumes `RaaGiVolumeParams`, binding 6
// `raa_linear_clamp`, binding 17 `raa_gi_volume` and binding 18 `raa_gi_volume_params`
// from the including module (PRELUDE_WGSL declares them all); it declares no binding of
// its own and samples only with `textureSampleLevel`, so it is safe in non-uniform control
// flow and in compute. With `level_count == 0` (the dummy) every lookup misses and the
// visibility is 1.
//
// Adapted from solstice@7e465b5,
// crates/solstice-render/shaders/common/environment.wgsl:140-219. MIT License
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

// The finest level in [first, level_count) whose local coordinates lie inside
// [1.5, resolution - 1.5], else level_count. Never returns a level below `first`.
fn raa_gi_level_at(p: vec3<f32>, first: u32) -> u32 {
    let count = raa_gi_volume_params.level_count;
    for (var level = first; level < count; level++) {
        let grid = raa_gi_volume_params.levels[level];
        let local = (p - grid.xyz) / grid.w;
        let margin = 1.5;
        let upper = raa_gi_volume_params.resolution - margin;
        if all(local > vec3<f32>(margin)) && all(local < vec3<f32>(upper)) {
            return level;
        }
    }
    return count;
}

// Radiance in rgb, distance to the nearest surface in a (metres).
fn raa_gi_sample(p: vec3<f32>, level: u32) -> vec4<f32> {
    let inv_resolution = 1.0 / raa_gi_volume_params.resolution;
    let grid = raa_gi_volume_params.levels[level];
    let local = (p - grid.xyz) / grid.w * inv_resolution;
    // Each level occupies one z slab of the atlas. Stay half a texel inside, so that
    // filtering cannot read a neighbouring level.
    let half_texel = 0.5 * inv_resolution;
    let z = clamp(local.z, half_texel, 1.0 - half_texel);
    let atlas_z = (z + f32(level)) / f32(raa_gi_volume_params.level_count);
    return textureSampleLevel(raa_gi_volume, raa_linear_clamp, vec3<f32>(local.xy, atlas_z), 0.0);
}

struct RaaGiHit {
    hit: bool,
    t: f32,
    radiance: vec3<f32>,
}

// Sphere traces the volume from t = start. `hit_voxels` is how close counts as a hit, in
// voxels of the level at the sample.
fn raa_gi_trace(origin: vec3<f32>, dir: vec3<f32>, start: f32, max_t: f32, max_steps: u32, hit_voxels: f32) -> RaaGiHit {
    let count = raa_gi_volume_params.level_count;
    var t = start;
    var level = 0u;
    var measured_level = count;
    var previous_distance = 0.0;
    for (var step = 0u; step < max_steps && t < max_t; step++) {
        let p = origin + dir * t;
        level = raa_gi_level_at(p, level);
        if level >= count {
            break;
        }
        let voxel = raa_gi_volume_params.levels[level].w;
        let s = raa_gi_sample(p, level);
        // A hit needs the ray to be closing in on a surface, measured within a single
        // level: distances from different levels are not comparable.
        let approaching = level == measured_level && s.a <= previous_distance;
        measured_level = level;
        previous_distance = s.a;
        if s.a < voxel * hit_voxels && approaching {
            return RaaGiHit(true, t, s.rgb);
        }
        t += max(s.a - voxel * 0.5, voxel * 0.6);
    }
    return RaaGiHit(false, t, vec3<f32>(0.0));
}

// 0 when the volume blocks the path along `dir`, else 1.
fn raa_gi_visibility(origin: vec3<f32>, dir: vec3<f32>, start: f32, max_t: f32, max_steps: u32) -> f32 {
    return select(1.0, 0.0, raa_gi_trace(origin, dir, start, max_t, max_steps, 0.85).hit);
}

// How far inside the outermost level's XZ footprint `p` is: 0 at its edge, 1 well inside.
fn raa_gi_edge_fade(p: vec3<f32>) -> f32 {
    let count = raa_gi_volume_params.level_count;
    if count == 0u {
        return 1.0;
    }
    let inv_resolution = 1.0 / raa_gi_volume_params.resolution;
    let grid = raa_gi_volume_params.levels[count - 1u];
    let local = (p - grid.xyz) / grid.w * inv_resolution;
    let edge = min(min(local.x, 1.0 - local.x), min(local.z, 1.0 - local.z));
    return smoothstep(0.02, 0.2, edge);
}

// The finest voxel size in metres (level 0).
fn raa_gi_finest_voxel() -> f32 {
    return raa_gi_volume_params.levels[0].w;
}
