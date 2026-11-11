// Ground cloud shadow of the M2 volumetric cloud backend: how much of the dominant
// light the clouds let through to the ground over a square around the camera.
//
// Adapted from Solstice revision 7e465b599c13da165948ebaa44df824c36c3ae07,
// crates/solstice-render/shaders/clouds/shadow.wgsl:14-47.
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
// `clouds/shaders.rs` composes this text after `cloud_volume_wgsl()` and the output
// binding 10 `raa_cloud_output`. Texel (i, j) covers the ground at
// `raa_cloud.shadow_origin + (ij + 0.5) / size * raa_cloud.shadow_extent` (metres);
// R is the scalar transmission, G and B repeat it and A is 1.

const RAA_CLOUD_SHADOW_SAMPLES: i32 = 16;

@compute @workgroup_size(8, 8)
fn raa_cloud_shadow_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(raa_cloud_output);
    if any(id.xy >= size) {
        return;
    }
    let ground = raa_cloud.shadow_origin
        + (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size) * raa_cloud.shadow_extent;
    var light = 1.0;
    if raa_cloud.enabled > 0.5 && raa_cloud.coverage > 0.001 {
        // From the ground towards the dominant light, across the layer. A low light
        // crosses more of it, up to a limit beyond which the map is too small to matter.
        let to_light = raa_cloud_sky.dominant.xyz;
        let rise = max(to_light.y, 0.08);
        let enter = raa_cloud.base / rise;
        let exit = raa_cloud.top / rise;
        let span = min(exit - enter, 12000.0);
        let step = span / f32(RAA_CLOUD_SHADOW_SAMPLES);
        var depth = 0.0;
        for (var i = 0; i < RAA_CLOUD_SHADOW_SAMPLES; i++) {
            let t = enter + (f32(i) + 0.5) * step;
            let height = to_light.y * t;
            depth += raa_cloud_density(
                ground + to_light.xz * t,
                max(height, raa_cloud.base + 1.0),
                true,
            ) * step;
        }
        depth *= RAA_CLOUD_EXTINCTION * raa_cloud.density;
        // Light scattered on through the cloud still reaches the ground.
        light = max(exp(-depth) * 0.7 + exp(-depth * 0.1) * 0.3, 0.04);
        // Rain falling from the cloud dims the light further.
        light *= 1.0 - 0.3 * raa_cloud.rain;
    }
    textureStore(raa_cloud_output, id.xy, vec4<f32>(light, light, light, 1.0));
}
