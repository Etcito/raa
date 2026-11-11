// Screen clouds of the M2 volumetric cloud backend, at half resolution. Each texel marches
// one ray, jittered along its length every frame, and blends it into the view's previous
// screen clouds reprojected to where they now appear (cloud-only history, no TAA). Texels
// whose full-resolution pixel is covered by geometry are not marched.
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
// 10 `raa_cloud_output` (half-resolution Rgba16Float), 11 `raa_cloud_history` (the view's
// previous result) and 12 `raa_cloud_depth` (texture_depth_2d or
// texture_depth_multisampled_2d; both are read with textureLoad at index 0). RGB is linear
// HDR cloud radiance, A the scalar transmittance; A = -1 marks a texel not marched.
// Adaptations: the ray comes from `raa_cloud.right/up/forward` at the half-resolution texel
// centre; the jitter index is `raa_cloud.screen.z`; standard Z (clear 1, far clip 1)
// replaces the reversed-Z sky test; history is reprojected with
// `raa_cloud.previous_view_proj` and used only when `raa_cloud.forward.w` is 1.

const RAA_CLOUD_SCREEN_STEPS: i32 = 56;
const RAA_CLOUD_SCREEN_LIGHT_SAMPLES: i32 = 5;
// The share of each new frame in the blend with the reprojected history.
const RAA_CLOUD_SCREEN_BLEND: f32 = 0.12;
// Transmittance stored by texels that were not marched (CLOUD_SCREEN_UNKNOWN_ALPHA).
const RAA_CLOUD_SCREEN_UNKNOWN: f32 = -1.0;

fn raa_cloud_gradient_noise(pixel: vec2<f32>, frame: u32) -> f32 {
    let p = pixel + 5.588238 * f32(frame % 64u);
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

@compute @workgroup_size(8, 8)
fn raa_cloud_screen_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(raa_cloud_output);
    if any(id.xy >= size) {
        return;
    }
    // Standard Z: the sky keeps the clear depth 1; any nearer depth is geometry.
    let full = vec2<i32>(textureDimensions(raa_cloud_depth));
    let pixel = min(vec2<i32>(id.xy) * 2, full - vec2<i32>(1));
    if textureLoad(raa_cloud_depth, pixel, 0) < 1.0 {
        // Unknown, rather than clear: a later frame must not reuse it as "no cloud".
        textureStore(raa_cloud_output, id.xy, vec4<f32>(0.0, 0.0, 0.0, RAA_CLOUD_SCREEN_UNKNOWN));
        return;
    }

    let uv = (vec2<f32>(id.xy) + 0.5) / max(raa_cloud.screen.xy, vec2<f32>(1.0));
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let direction = normalize(
        raa_cloud.forward.xyz + raa_cloud.right.xyz * ndc.x + raa_cloud.up.xyz * ndc.y,
    );
    let jitter = raa_cloud_gradient_noise(vec2<f32>(id.xy), u32(raa_cloud.screen.z));
    let here = raa_cloud_along(
        direction,
        RAA_CLOUD_SCREEN_STEPS,
        RAA_CLOUD_SCREEN_LIGHT_SAMPLES,
        jitter,
    );
    var result = vec4<f32>(here.radiance, here.transmittance);

    // Where the clouds were on screen at the last committed march. Clouds are far away,
    // so where none were hit the direction alone will do.
    if raa_cloud.forward.w != 0.0 {
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
                if all(marched >= vec4<f32>(0.0)) {
                    let past = textureSampleLevel(
                        raa_cloud_history,
                        raa_cloud_lut_sampler,
                        previous_uv,
                        0.0,
                    );
                    result = mix(past, result, RAA_CLOUD_SCREEN_BLEND);
                }
            }
        }
    }
    textureStore(raa_cloud_output, id.xy, result);
}
