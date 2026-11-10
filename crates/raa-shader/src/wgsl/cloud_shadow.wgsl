// S25 ground cloud shadow: how much of the dominant light reaches `world_pos` through the
// clouds. Consumes the group 1 weather declarations already in the prelude (binding 15's
// two-layer transmittance map and binding 16's WeatherLightingGpu) and binding 6's existing
// linear clamp sampler; it declares no binding of its own. Returns 1 when clouds are off, so
// CSM and contact shadows keep their own visibility.
//
// Receiver projection and edge-to-mean fade adapted from solstice@7e465b5,
// crates/solstice-render/shaders/common/environment.wgsl:117-132. MIT License
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

fn raa_cloud_shadow(world_pos: vec3<f32>) -> f32 {
    if raa_weather_lighting.map.w < 0.5 {
        return 1.0;
    }
    // The map holds light at the ground. A point above it is lit along the same line to
    // the dominant light, from where that line meets the ground.
    let to_light = raa_weather_lighting.light.xyz;
    let ground = world_pos.xz - to_light.xz * (world_pos.y / max(to_light.y, 0.08));
    let extent = max(raa_weather_lighting.map.z, 1.0e-6);
    let uv = (ground - raa_weather_lighting.map.xy) / extent;
    let mapped = textureSampleLevel(
        raa_cloud_transmittance,
        raa_linear_clamp,
        clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)),
        0,
        0.0,
    ).r;
    // Beyond the map, its mean stands in, blended in across its outer band, so the far field
    // is the same in every direction instead of stretching edge texels to the horizon.
    let edge = max(abs(uv.x - 0.5), abs(uv.y - 0.5)) * 2.0;
    let mean = textureLoad(raa_cloud_transmittance, vec2<i32>(0, 0), 1, 0).r;
    return mix(mapped, mean, smoothstep(0.75, 1.0, edge));
}
