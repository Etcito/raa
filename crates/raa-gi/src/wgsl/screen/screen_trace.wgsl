// S39 screen trace: a probe ray marches the closest-depth pyramid (depth_pyramid.wgsl) in
// screen space and hits where it meets the nearest surface of a tile, as in AMD FidelityFX
// SSSR. RAA is standard-Z, so the depth plane crossing is valid when the ray moves towards
// depth (ray.z > 0) and a surface counts as in front through `raa_depth_nearer`, never a
// reversed-Z comparison. The hit's thickness test uses view distances from
// `raa_gi_view_position` and the hit's radiance comes from the gi.history copy of the
// previous frame's lit HDR, reprojected with `previous_view_proj`.
//
// Adapted from solstice@7e465b5,
// crates/solstice-render/shaders/gi/screen_trace.wgsl:15-141 and
// crates/solstice-render/shaders/gi/screen_history.wgsl:9-13. MIT License
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

// Fades to zero towards the screen edges, where the screen's data runs out.
const RAA_GI_SCREEN_EDGE_FADE: f32 = 0.08;

// The previous frame's lit HDR (Rgba16Float), sampled at lod 0.
@group(2) @binding(3) var raa_gi_history: texture_2d<f32>;
// The closest-depth pyramid (R32Float, all mips), read with textureLoad only.
@group(2) @binding(4) var raa_gi_pyramid: texture_2d<f32>;

struct RaaGiScreenHit {
    hit: bool,
    // Distance along the world-space ray.
    t: f32,
    uv: vec2<f32>,
    weight: f32,
}

fn raa_gi_screen_miss() -> RaaGiScreenHit {
    return RaaGiScreenHit(false, 0.0, vec2<f32>(0.0), 0.0);
}

// To uv and standard-Z clip depth, without jitter.
fn raa_gi_screen_project(world: vec3<f32>) -> vec3<f32> {
    let clip = raa_gi_frame.view_proj * vec4<f32>(world, 1.0);
    let ndc = clip.xyz / clip.w;
    return vec3<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, ndc.z);
}

// The positive view distance of a world position, from the w lane of its clip point.
fn raa_gi_screen_view_distance(world: vec3<f32>) -> f32 {
    return (raa_gi_frame.view_proj * vec4<f32>(world, 1.0)).w;
}

// Moves the ray to the edge of its texel at the given mip resolution, or to the tile's depth
// plane if that comes first. Returns whether the ray passed in front of the whole tile.
fn raa_gi_screen_advance(
    start: vec3<f32>,
    inv_ray: vec3<f32>,
    ray: vec3<f32>,
    mip_position: vec2<f32>,
    inv_resolution: vec2<f32>,
    floor_offset: vec2<f32>,
    uv_offset: vec2<f32>,
    surface: f32,
    position: ptr<function, vec3<f32>>,
    t: ptr<function, f32>,
) -> bool {
    let planes = vec3<f32>(
        (floor(mip_position) + floor_offset) * inv_resolution + uv_offset,
        surface,
    );
    var crossing = (planes - start) * inv_ray;
    // Standard Z: only a ray moving towards depth can meet the depth plane from in front.
    crossing.z = select(3.0e38, crossing.z, ray.z > 0.0);
    let t_min = min(min(crossing.x, crossing.y), crossing.z);
    let in_front = raa_depth_nearer((*position).z, surface);
    if in_front {
        *t = t_min;
        *position = start + ray * t_min;
    }
    return in_front && t_min != crossing.z;
}

// Traces from `origin` along unit `direction` for up to `max_distance` metres. `thickness`
// is how far, in metres, a surface is assumed to extend behind what the camera sees.
fn raa_gi_trace_screen(
    origin: vec3<f32>,
    direction: vec3<f32>,
    max_distance: f32,
    max_steps: u32,
    thickness: f32,
) -> RaaGiScreenHit {
    // Stop short of the near plane, where projection breaks down.
    var distance = max_distance;
    let w0 = raa_gi_screen_view_distance(origin);
    let dw = raa_gi_screen_view_distance(origin + direction) - w0;
    if dw < 0.0 {
        distance = min(distance, (w0 - raa_gi_frame.settings.z * 2.0) / -dw);
    }
    if distance <= 0.0 {
        return raa_gi_screen_miss();
    }
    let start = raa_gi_screen_project(origin);
    let end = raa_gi_screen_project(origin + direction * distance);
    let ray = end - start;
    if dot(ray.xy, ray.xy) < 1.0e-12 {
        return raa_gi_screen_miss();
    }
    let inv_ray = select(vec3<f32>(3.0e38), 1.0 / ray, abs(ray) > vec3<f32>(1.0e-12));

    let top_mip = i32(textureNumLevels(raa_gi_pyramid)) - 1;
    let pixel_offset = 0.005 * raa_gi_frame.viewport.zw;
    let uv_offset = select(pixel_offset, -pixel_offset, ray.xy < vec2<f32>(0.0));
    let floor_offset = select(vec2<f32>(1.0), vec2<f32>(0.0), ray.xy < vec2<f32>(0.0));

    // Step off the starting pixel first, so the ray does not hit the surface it leaves.
    var position = start;
    var t = 0.0;
    {
        let planes = (floor(start.xy * raa_gi_frame.viewport.xy) + floor_offset)
            * raa_gi_frame.viewport.zw
            + uv_offset;
        let crossing = (planes - start.xy) * inv_ray.xy;
        t = min(crossing.x, crossing.y);
        position = start + ray * t;
    }

    var mip = 0;
    var steps = 0u;
    while mip >= 0 && steps < max_steps {
        if t > 1.0 || any(position.xy < vec2<f32>(0.0)) || any(position.xy > vec2<f32>(1.0)) {
            return raa_gi_screen_miss();
        }
        let scale = exp2(-f32(mip));
        let mip_position = position.xy * raa_gi_frame.viewport.xy * scale;
        // The odd-edge pyramid's last texel also takes the extra source column or row.
        let dims = vec2<i32>(textureDimensions(raa_gi_pyramid, mip));
        let texel = min(vec2<i32>(floor(mip_position)), dims - vec2<i32>(1));
        let surface = textureLoad(raa_gi_pyramid, texel, mip).r;
        let skipped = raa_gi_screen_advance(
            start,
            inv_ray,
            ray,
            mip_position,
            raa_gi_frame.viewport.zw / scale,
            floor_offset,
            uv_offset,
            surface,
            &position,
            &t,
        );
        mip = select(mip - 1, min(mip + 1, top_mip), skipped);
        steps++;
    }
    if mip >= 0 || t > 1.0 {
        return raa_gi_screen_miss();
    }

    // The ray is level with or behind the surface here. It hits only if that surface is
    // plausibly this thick, and only its front: a ray leaving a surface seen at a grazing
    // angle would otherwise "hit" the same surface, since a pixel there spans metres of depth.
    let pixel = min(
        vec2<i32>(position.xy * raa_gi_frame.viewport.xy),
        vec2<i32>(raa_gi_frame.viewport.xy) - vec2<i32>(1),
    );
    let scene_depth = textureLoad(raa_gi_pyramid, pixel, 0).r;
    if raa_depth_is_far(scene_depth) || position.z <= 0.0 {
        return raa_gi_screen_miss();
    }
    let scene_distance = -raa_gi_view_position(vec2<f32>(pixel) + vec2<f32>(0.5), scene_depth).z;
    let ray_distance = -raa_gi_view_position(position.xy * raa_gi_frame.viewport.xy, position.z).z;
    if ray_distance - scene_distance > thickness + scene_distance * 0.02 {
        return raa_gi_screen_miss();
    }
    if dot(raa_gi_world_normal(pixel), direction) >= 0.0 {
        return raa_gi_screen_miss();
    }
    let edge = min(min(position.x, 1.0 - position.x), min(position.y, 1.0 - position.y));
    return RaaGiScreenHit(
        true,
        t * distance,
        position.xy,
        smoothstep(0.0, RAA_GI_SCREEN_EDGE_FADE, edge),
    );
}

// Last frame's lit colour at a world hit, found by reprojecting the hit with the previous
// frame's view-projection. a is 0 when the hit was off screen or behind the previous camera.
fn raa_gi_screen_hit_radiance(world_hit: vec3<f32>) -> vec4<f32> {
    let clip = raa_gi_frame.previous_view_proj * vec4<f32>(world_hit, 1.0);
    if clip.w <= 0.0 {
        return vec4<f32>(0.0);
    }
    let ndc = clip.xyz / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(textureSampleLevel(raa_gi_history, raa_linear_clamp, uv, 0.0).rgb, 1.0);
}
