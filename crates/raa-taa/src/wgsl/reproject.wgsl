// Temporal reprojection shared by every temporal pass of the TAA milestone (contract
// `raa-temporal-wgsl`): the S10 `FrameGpu` twin `RaaTemporalFrame` and the surface, sky and
// history helpers. This source declares no bindings and no entry points.
//
// Adapted from solstice@7e465b5,
// solstice:crates/solstice-render/shaders/post/taa.wgsl:131-141 (sky reprojection).
// MIT License
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

// S10 `FrameGpu` twin: the 768-byte frame record every temporal pass reads. `view_proj`,
// `proj` and `inv_proj` carry the current camera jitter; `previous_view_proj` is the previous
// jittered view-projection and `camera_jitter` holds the current NDC xy then the previous
// NDC xy. RAA is standard-Z (far plane at depth 1).
struct RaaTemporalFrame {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    camera_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    ambient: vec4<f32>,
    viewport: vec4<f32>,
    clip: vec4<f32>,
    flags: vec4<u32>,
    exposure: vec4<f32>,
    reserved: array<vec4<f32>, 4>,
    previous_view_proj: mat4x4<f32>,
    unjittered_view_proj: mat4x4<f32>,
    camera_jitter: vec4<f32>,
    light_info: vec4<u32>,
    temporal: vec4<u32>,
    reserved2: array<vec4<f32>, 5>,
}

// Strictly inside the [0, 1] uv square, so a bilinear history read never wraps.
fn raa_temporal_on_screen(uv: vec2<f32>) -> bool {
    return all(uv > vec2<f32>(0.0)) && all(uv < vec2<f32>(1.0));
}

// S12 surface reprojection: motion is the current minus the previous UV (previous uv =
// uv - motion.xy), unlike solstice's taa.wgsl which adds it. motion.z is the history reject
// flag of contract 'motion-target-v2': 1.0 means the pixel has no usable object history.
fn raa_temporal_surface_uv(uv: vec2<f32>, motion: vec4<f32>) -> vec3<f32> {
    let previous_uv = uv - motion.xy;
    let usable = motion.z < 0.5 && raa_temporal_on_screen(previous_uv);
    return vec3<f32>(previous_uv, select(0.0, 1.0, usable));
}

// Sky reprojection through the camera alone: the sky has no stored motion. `inv_proj` is the
// inverse of the jittered projection, so the jitter is added to the NDC before the unproject
// and the previous NDC subtracts camera_jitter.zw, as raa_pass_motion does. The sky is a
// direction (w = 0), as raa-sky's screen.wgsl reprojects it. Standard-Z puts the far plane at
// ndc z = 1. A clip w <= 0 has no previous uv and returns the current uv with z = 0.
fn raa_temporal_sky_uv(
    uv: vec2<f32>,
    inv_view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    previous_view_proj: mat4x4<f32>,
    camera_jitter: vec4<f32>,
) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let view_h = inv_proj * vec4<f32>(ndc + camera_jitter.xy, 1.0, 1.0);
    // The direction needs no homogeneous divide, including when the far plane
    // is at infinity and the unprojected point already has w = 0.
    let dir = (inv_view * vec4<f32>(view_h.xyz, 0.0)).xyz;
    let clip = previous_view_proj * vec4<f32>(dir, 0.0);
    if (clip.w <= 0.0) {
        return vec3<f32>(uv, 0.0);
    }
    let previous_ndc = clip.xy / clip.w - camera_jitter.zw;
    let previous_uv = vec2<f32>(previous_ndc.x * 0.5 + 0.5, 0.5 - previous_ndc.y * 0.5);
    return vec3<f32>(previous_uv, select(0.0, 1.0, raa_temporal_on_screen(previous_uv)));
}

// FrameGpu `temporal` = (frame parity, accumulation enabled 0/1, history valid 0/1, 0).
fn raa_temporal_history_usable(temporal: vec4<u32>) -> bool {
    return temporal.y != 0u && temporal.z != 0u;
}
