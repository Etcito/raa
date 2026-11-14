// GI volume debug views (share 4): one pixel of hdr.dst from the depth surface, the camera ray
// through the inverse view-projection and the voxel field of the clipmap: the first occupied
// 0.2 m voxel of level 0, the traced radiance, the distance contours (sampled along the surface
// normal) or the level bands.
//
// Adapted from solstice@7e465b5,
// crates/solstice-render/shaders/lighting/deferred.wgsl:95-215 (pixel ray, voxel debug view). MIT License
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

@group(2) @binding(0) var<uniform> raa_gi_frame: GiFrameGpu;
@group(2) @binding(1) var raa_gi_depth: texture_depth_2d;
@group(2) @binding(2) var raa_gi_hdr_src: texture_2d<f32>;
@group(2) @binding(3) var<storage, read> raa_gi_voxels_read: array<u32>;
@group(2) @binding(4) var raa_gi_hdr_dst: texture_storage_2d<rgba16float, write>;

const RAA_GI_DEBUG_PI: f32 = 3.14159265359;
// Two 0.2 m finest voxels make one contour period.
const RAA_GI_DEBUG_PERIOD: f32 = 0.4;
// The distance view samples the field this far along the surface normal (seven finest voxels),
// so the sample holds the distance to the other surfaces nearby (the walls, floor, box and
// sphere) up to 1.4 m: three and a half contour periods. A sample a fixed short way in front of
// the surface only reads that surface's own distance, the same everywhere, and shows no bands.
// Twin: `raa_gi::cpu::distance_probe` and `DISTANCE_LIFT_METRES`.
const RAA_GI_DEBUG_LIFT: f32 = 1.4;
const RAA_GI_DEBUG_FAR: f32 = 1.0e30;

struct RaaGiDebugVoxel {
    hit: bool,
    normal: vec3<f32>,
    albedo: vec3<f32>,
}

// Surface position of the depth pixel, back through the inverse view-projection.
fn raa_gi_debug_world(ndc: vec2<f32>, depth: f32) -> vec3<f32> {
    let clip = raa_gi_frame.inv_view_proj * vec4<f32>(ndc, depth, 1.0);
    return clip.xyz / clip.w;
}

// World position (xyz) and depth (w) of a pixel, clamped to the viewport.
fn raa_gi_debug_surface(pixel: vec2<i32>) -> vec4<f32> {
    let size = vec2<i32>(raa_gi_frame.viewport.xy);
    let p = clamp(pixel, vec2<i32>(0), size - vec2<i32>(1));
    let depth = textureLoad(raa_gi_depth, p, 0);
    let uv = (vec2<f32>(p) + vec2<f32>(0.5)) / raa_gi_frame.viewport.xy;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return vec4<f32>(raa_gi_debug_world(ndc, depth), depth);
}

// Surface normal of the depth pixel from its neighbours: along each screen axis, the side whose
// depth is closer to the centre (so silhouettes do not bend it), facing the camera.
fn raa_gi_debug_normal(pixel: vec2<i32>, centre: vec4<f32>, camera: vec3<f32>) -> vec3<f32> {
    let size = vec2<i32>(raa_gi_frame.viewport.xy);
    let right = raa_gi_debug_surface(pixel + vec2<i32>(1, 0));
    let left = raa_gi_debug_surface(pixel - vec2<i32>(1, 0));
    let down = raa_gi_debug_surface(pixel + vec2<i32>(0, 1));
    let up = raa_gi_debug_surface(pixel - vec2<i32>(0, 1));
    let right_gap = select(RAA_GI_DEBUG_FAR, abs(right.w - centre.w), pixel.x + 1 < size.x);
    let left_gap = select(RAA_GI_DEBUG_FAR, abs(centre.w - left.w), pixel.x > 0);
    let down_gap = select(RAA_GI_DEBUG_FAR, abs(down.w - centre.w), pixel.y + 1 < size.y);
    let up_gap = select(RAA_GI_DEBUG_FAR, abs(centre.w - up.w), pixel.y > 0);
    let dx = select(centre.xyz - left.xyz, right.xyz - centre.xyz, right_gap < left_gap);
    let dy = select(centre.xyz - up.xyz, down.xyz - centre.xyz, down_gap < up_gap);
    let toward_camera = camera - centre.xyz;
    var normal = cross(dx, dy);
    if dot(normal, normal) <= 1.0e-20 {
        normal = toward_camera;
    }
    if dot(normal, normal) <= 1.0e-20 {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    normal = normalize(normal);
    return select(normal, -normal, dot(normal, toward_camera) < 0.0);
}

fn raa_gi_debug_level_color(level: u32) -> vec3<f32> {
    switch level {
        case 0u: {
            return vec3<f32>(1.0, 0.15, 0.1);
        }
        case 1u: {
            return vec3<f32>(1.0, 0.55, 0.1);
        }
        case 2u: {
            return vec3<f32>(1.0, 0.95, 0.2);
        }
        case 3u: {
            return vec3<f32>(0.2, 0.85, 0.25);
        }
        default: {
            return vec3<f32>(0.2, 0.4, 1.0);
        }
    }
}

// The first occupied voxel of level 0 along the camera ray: a 3D DDA over at most 3R cells.
// `normal` is the axis of the entered face, so the flat shade shows the 0.2 m blocks.
fn raa_gi_debug_voxel(origin: vec3<f32>, dir: vec3<f32>) -> RaaGiDebugVoxel {
    let miss = RaaGiDebugVoxel(false, vec3<f32>(0.0), vec3<f32>(0.0));
    let n = i32(raa_gi_volume_params.resolution);
    if n == 0 {
        return miss;
    }
    let grid = raa_gi_volume_params.levels[0];
    let p = (origin - grid.xyz) / grid.w;
    // Clip the ray against [0, R]^3; remember the face it enters through.
    var t0 = 0.0;
    var t1 = RAA_GI_DEBUG_FAR;
    var entry_axis = -1;
    var entry_sign = 1.0;
    for (var a = 0; a < 3; a++) {
        if abs(dir[a]) < 1.0e-8 {
            if p[a] < 0.0 || p[a] > f32(n) {
                return miss;
            }
            continue;
        }
        var ta = (0.0 - p[a]) / dir[a];
        var tb = (f32(n) - p[a]) / dir[a];
        if ta > tb {
            let swap = ta;
            ta = tb;
            tb = swap;
        }
        if ta > t0 {
            t0 = ta;
            entry_axis = a;
            entry_sign = dir[a];
        }
        t1 = min(t1, tb);
    }
    if t1 <= max(t0, 0.0) {
        return miss;
    }
    let start = p + dir * max(t0, 0.0);
    var cell = clamp(vec3<i32>(floor(start)), vec3<i32>(0), vec3<i32>(n - 1));
    let step = vec3<i32>(sign(dir));
    var t_max = vec3<f32>(RAA_GI_DEBUG_FAR);
    var t_delta = vec3<f32>(RAA_GI_DEBUG_FAR);
    for (var a = 0; a < 3; a++) {
        if abs(dir[a]) >= 1.0e-8 {
            let boundary = f32(cell[a]) + select(0.0, 1.0, dir[a] > 0.0);
            t_max[a] = max((boundary - p[a]) / dir[a], 0.0);
            t_delta[a] = abs(1.0 / dir[a]);
        }
    }
    var normal = vec3<f32>(0.0);
    if entry_axis >= 0 {
        normal[entry_axis] = -sign(entry_sign);
    } else {
        // The camera starts inside level 0: shade the face that looks back along the ray.
        let magnitude = abs(dir);
        var axis = 0;
        if magnitude.y > magnitude[axis] {
            axis = 1;
        }
        if magnitude.z > magnitude[axis] {
            axis = 2;
        }
        normal[axis] = -sign(dir[axis]);
    }
    for (var i = 0; i < 3 * n; i++) {
        let index = (cell.z * n + cell.y) * n + cell.x;
        let word = raa_gi_voxels_read[index * 3];
        if (word & 0xff000000u) != 0u {
            return RaaGiDebugVoxel(true, normal, unpack4x8unorm(word).rgb);
        }
        var axis = 0;
        if t_max.y < t_max.x {
            axis = 1;
        }
        if t_max.z < t_max[axis] {
            axis = 2;
        }
        if t_max[axis] >= RAA_GI_DEBUG_FAR {
            break;
        }
        cell[axis] += step[axis];
        if cell[axis] < 0 || cell[axis] >= n {
            break;
        }
        normal = vec3<f32>(0.0);
        normal[axis] = -f32(step[axis]);
        t_max[axis] += t_delta[axis];
    }
    return miss;
}

@compute @workgroup_size(8, 8)
fn raa_gi_debug_view(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = raa_gi_frame.viewport.xy;
    if f32(id.x) >= size.x || f32(id.y) >= size.y {
        return;
    }
    let pixel = vec2<i32>(id.xy);
    let uv = (vec2<f32>(id.xy) + vec2<f32>(0.5)) / size;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let far_clip = raa_gi_frame.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let origin = raa_gi_frame.camera_position.xyz;
    let direction = normalize(far_clip.xyz / far_clip.w - origin);
    let depth = textureLoad(raa_gi_depth, pixel, 0);
    let src = textureLoad(raa_gi_hdr_src, pixel, 0).rgb;

    var color = src;
    switch raa_gi_frame.counts.z {
        case 1u: {
            let voxel = raa_gi_debug_voxel(origin, direction);
            if voxel.hit {
                let sun = raa_gi_frame.sun_direction.xyz;
                color = voxel.albedo * (0.35 + 0.65 * max(dot(voxel.normal, sun), 0.0));
            }
        }
        case 2u: {
            let hit = raa_gi_trace(origin, direction, 0.0, raa_gi_frame.settings.w, 128u, 0.85);
            color = select(src * 0.1, hit.radiance, hit.hit);
        }
        case 3u: {
            if depth < 1.0 {
                // The field RAA_GI_DEBUG_LIFT along the surface normal: the distance from there
                // to the nearest other surface, so the bands run parallel to every surface around.
                let centre = raa_gi_debug_surface(pixel);
                let normal = raa_gi_debug_normal(pixel, centre, origin);
                let probe = centre.xyz + normal * RAA_GI_DEBUG_LIFT;
                let level = raa_gi_level_at(probe, 0u);
                var d = 64.0 * raa_gi_finest_voxel();
                if level < raa_gi_volume_params.level_count {
                    d = raa_gi_sample(probe, level).a;
                }
                color = vec3<f32>(0.5 + 0.5 * cos(2.0 * RAA_GI_DEBUG_PI * d / RAA_GI_DEBUG_PERIOD));
            }
        }
        case 4u: {
            if depth < 1.0 {
                let surface = raa_gi_debug_world(ndc, depth);
                let level = raa_gi_level_at(surface, 0u);
                if level < raa_gi_volume_params.level_count {
                    let luma = dot(src, vec3<f32>(0.2126, 0.7152, 0.0722));
                    color = mix(vec3<f32>(luma), raa_gi_debug_level_color(level), 0.6);
                }
            }
        }
        default: {}
    }
    textureStore(raa_gi_hdr_dst, pixel, vec4<f32>(color, 1.0));
}
