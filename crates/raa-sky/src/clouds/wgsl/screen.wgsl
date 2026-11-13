// Screen clouds of the M2 volumetric cloud backend, in a buffer of ceil(size * cloud_scale)
// texels (default half resolution). Each texel marches one ray, offset along its length by a
// fixed per-pixel noise shifted by the golden ratio each frame, and blends it into the view's
// previous screen clouds reprojected to where they now appear (cloud-only history, no TAA).
// The blend is a running mean over the frames the camera and the sky stood still
// (`raa_cloud.forward.w` = n): 0.12 while they change, then 1/(n + 6), never below 1/16 while the
// sky is animated, so the offsets average out and the result converges. Texels without usable
// history (first frame, a resize, newly entering the field of view, previously covered by
// geometry) are marched with twice the steps instead of one noisy sample. Texels whose
// full-resolution pixel is covered by geometry are not marched.
//
// Adapted from Solstice revision 7e465b599c13da165948ebaa44df824c36c3ae07,
// crates/solstice-render/shaders/clouds/screen.wgsl:20-86 and
// crates/solstice-render/shaders/common/math.wgsl:42-45 (interleaved gradient noise).
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
// `clouds/shaders.rs` composes this text after `cloud_volume_wgsl()` and the bindings
// 10 `raa_cloud_output` (Rgba16Float, ceil(size * cloud_scale)), 11 `raa_cloud_history` (the
// view's previous result) and 12 `raa_cloud_depth` (texture_depth_2d or
// texture_depth_multisampled_2d; both are read with textureLoad at index 0). RGB is linear
// HDR cloud radiance, A the scalar transmittance; A = -1 marks a texel not marched.
// Adaptations: the ray comes from `raa_cloud.right/up/forward` at the texel centre; standard Z
// (clear 1, far clip 1) replaces the reversed-Z sky test; history is reprojected with
// `raa_cloud.previous_view_proj` and used only when `raa_cloud.forward.w` is not 0.
// `raa_cloud.screen` = (cloud_scale, flags, jitter index, cloud_samples); flags bit 0 is the
// refresh pass of a frozen view (texels that kept a march are copied, the others marched), bit 1
// an animated sky, bit 2 dither off (constant march offset 0.5, one march per texel, no history
// mix: a deterministic image that cannot crawl), bits 3.. the noise block size 1..16 (the hash
// sees floor(texel / size), so a larger size is a coarser grain). With a camera and sky identical
// to the history's, previous_uv lands on this very texel centre and the history is read exactly
// (textureLoad), never bilinearly: repeated blends of a bilinear read at a tiny offset diffuse
// the converged image into a blur. A texel's full-resolution pixel is floor((texel + 0.5) / cloud_scale).

const RAA_CLOUD_SCREEN_LIGHT_SAMPLES: i32 = 5;
// The share of each new frame while the camera or the sky changes (n = 1).
const RAA_CLOUD_SCREEN_MOVING_BLEND: f32 = 0.12;
// The running mean continues the moving blend: 1 / (n + 6) is 0.125 at n = 2.
const RAA_CLOUD_SCREEN_PRIOR: f32 = 6.0;
// The smallest share of a new frame while the sky is animated.
const RAA_CLOUD_SCREEN_ANIMATED_FLOOR: f32 = 0.0625;
// Texels without usable history march this many times the steps.
const RAA_CLOUD_SCREEN_BURST: i32 = 2;
// Transmittance stored by texels that were not marched (CLOUD_SCREEN_UNKNOWN_ALPHA).
const RAA_CLOUD_SCREEN_UNKNOWN: f32 = -1.0;

// Integer hash (PCG) of the pixel and a sequence index, uniform in [0, 1) with no low-frequency
// structure, so no ramp travels across the screen between frames.
fn raa_cloud_hash_noise(pixel: vec2<u32>, sequence: u32) -> f32 {
    var v = pixel.x * 1664525u + pixel.y * 1013904223u + sequence * 747796405u + 2891336453u;
    v = v * 747796405u + 2891336453u;
    let word = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return f32(((word >> 22u) ^ word) >> 8u) * (1.0 / 16777216.0);
}

// The share of the new march in the blend with the history, for n frames accumulated.
fn raa_cloud_screen_blend(n: f32, animated: bool) -> f32 {
    if n < 1.5 {
        return RAA_CLOUD_SCREEN_MOVING_BLEND;
    }
    let mean = 1.0 / (n + RAA_CLOUD_SCREEN_PRIOR);
    return select(mean, max(mean, RAA_CLOUD_SCREEN_ANIMATED_FLOOR), animated);
}

@compute @workgroup_size(8, 8)
fn raa_cloud_screen_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(raa_cloud_output);
    if any(id.xy >= size) {
        return;
    }
    let scale = max(raa_cloud.screen.x, 0.01);
    let flags = u32(raa_cloud.screen.y + 0.5);
    let refresh = (flags & 1u) != 0u;
    let animated = (flags & 2u) != 0u;
    let dither_off = (flags & 4u) != 0u;
    let noise_size = clamp(flags >> 3u, 1u, 16u);
    // Standard Z: the sky keeps the clear depth 1; any nearer depth is geometry.
    let full = vec2<i32>(textureDimensions(raa_cloud_depth));
    let pixel = min(vec2<i32>(floor((vec2<f32>(id.xy) + 0.5) / scale)), full - vec2<i32>(1));
    if textureLoad(raa_cloud_depth, pixel, 0) < 1.0 {
        // Unknown, rather than clear: a later frame must not reuse it as "no cloud".
        textureStore(raa_cloud_output, id.xy, vec4<f32>(0.0, 0.0, 0.0, RAA_CLOUD_SCREEN_UNKNOWN));
        return;
    }
    if refresh {
        // A frozen view stood still: the history is this very texel.
        let kept = textureLoad(raa_cloud_history, vec2<i32>(id.xy), 0);
        if kept.a >= 0.0 {
            textureStore(raa_cloud_output, id.xy, kept);
            return;
        }
    }

    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let direction = normalize(
        raa_cloud.forward.xyz + raa_cloud.right.xyz * ndc.x + raa_cloud.up.xyz * ndc.y,
    );
    // Still frames: a fixed per-pixel noise shifted by the golden ratio each frame (a
    // low-discrepancy sequence, so the accumulated mean converges without flicker). Moving frames:
    // a fresh hash per frame.
    let sequence = u32(raa_cloud.screen.z);
    let block = id.xy / vec2<u32>(noise_size);
    var jitter = raa_cloud_hash_noise(block, sequence);
    if raa_cloud.forward.w >= 1.5 {
        jitter = fract(raa_cloud_hash_noise(block, 0u) + 0.61803398875 * f32(sequence));
    }
    let steps = clamp(i32(raa_cloud.screen.w + 0.5), 8, 256);
    if dither_off {
        // No noise at all: the same march every frame, so nothing crawls and nothing blurs.
        let fixed = raa_cloud_along(direction, steps, RAA_CLOUD_SCREEN_LIGHT_SAMPLES, 0.5);
        textureStore(raa_cloud_output, id.xy, vec4<f32>(fixed.radiance, fixed.transmittance));
        return;
    }

    // Where the clouds were on screen at the last committed march, when it can be used.
    var result = vec4<f32>(0.0);
    var blended = false;
    if raa_cloud.forward.w != 0.0 && !refresh {
        let here = raa_cloud_along(
            direction,
            steps,
            RAA_CLOUD_SCREEN_LIGHT_SAMPLES,
            jitter,
        );
        result = vec4<f32>(here.radiance, here.transmittance);
        // Clouds are far away, so where none were hit the direction alone will do.
        var previous_clip: vec4<f32>;
        if here.transmittance > 0.995 {
            previous_clip = raa_cloud.previous_view_proj * vec4<f32>(direction, 0.0);
        } else {
            let point = raa_cloud_sky.viewer.xyz + direction * here.distance;
            previous_clip = raa_cloud.previous_view_proj * vec4<f32>(point, 1.0);
        }
        if previous_clip.w > 0.0 {
            let previous_ndc = previous_clip.xy / previous_clip.w;
            let previous_uv = vec2<f32>(previous_ndc.x * 0.5 + 0.5, 0.5 - previous_ndc.y * 0.5);
            if all(previous_uv >= vec2<f32>(0.0)) && all(previous_uv <= vec2<f32>(1.0)) {
                // History is used only where all four texels it filters between were marched.
                let marched = textureGather(3, raa_cloud_history, raa_cloud_lut_sampler, previous_uv);
                let at_texel = previous_uv * vec2<f32>(size) - 0.5 - vec2<f32>(id.xy);
                let same_texel = all(abs(at_texel) < vec2<f32>(0.02));
                let exact = textureLoad(raa_cloud_history, vec2<i32>(id.xy), 0);
                if (same_texel && exact.a >= 0.0) || all(marched >= vec4<f32>(0.0)) {
                    var past = textureSampleLevel(
                        raa_cloud_history,
                        raa_cloud_lut_sampler,
                        previous_uv,
                        0.0,
                    );
                    if same_texel {
                        past = exact;
                    }
                    result = mix(
                        past,
                        result,
                        raa_cloud_screen_blend(raa_cloud.forward.w, animated),
                    );
                    blended = true;
                }
            }
        }
    }
    if !blended {
        // No usable history: one noisy sample would be all this texel shows for several frames.
        let burst = raa_cloud_along(
            direction,
            min(steps * RAA_CLOUD_SCREEN_BURST, 512),
            RAA_CLOUD_SCREEN_LIGHT_SAMPLES,
            fract(jitter + 0.5),
        );
        result = vec4<f32>(burst.radiance, burst.transmittance);
    }
    textureStore(raa_cloud_output, id.xy, result);
}
