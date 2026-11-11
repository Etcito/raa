// Mean of the ground cloud shadow map: the average transmission over the square
// around the camera, which lighting fades to beyond the map's edge. One workgroup
// reads every fourth texel each way.
//
// Adapted from Solstice revision 7e465b599c13da165948ebaa44df824c36c3ae07,
// crates/solstice-render/shaders/clouds/shadow_mean.wgsl:7-35.
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
// Self-contained: groups 0 and 1 are empty. Group 2 binding 0 is layer 0 of the
// shadow texture (the map), binding 1 a write view of layer 1; the mean goes to
// texel (0, 0) as (mean, mean, mean, 1).

@group(2) @binding(0) var raa_cloud_shadow_map: texture_2d<f32>;
@group(2) @binding(1) var raa_cloud_shadow_mean: texture_storage_2d<rgba16float, write>;

const RAA_CLOUD_MEAN_THREADS: u32 = 256u;
const RAA_CLOUD_MEAN_STRIDE: u32 = 4u;

var<workgroup> raa_cloud_mean_sums: array<f32, RAA_CLOUD_MEAN_THREADS>;

@compute @workgroup_size(RAA_CLOUD_MEAN_THREADS)
fn raa_cloud_shadow_mean_main(@builtin(local_invocation_index) thread: u32) {
    let size = textureDimensions(raa_cloud_shadow_map) / RAA_CLOUD_MEAN_STRIDE;
    let count = size.x * size.y;
    var sum = 0.0;
    for (var texel = thread; texel < count; texel += RAA_CLOUD_MEAN_THREADS) {
        let at = vec2<u32>(texel % size.x, texel / size.x) * RAA_CLOUD_MEAN_STRIDE
            + RAA_CLOUD_MEAN_STRIDE / 2u;
        sum += textureLoad(raa_cloud_shadow_map, at, 0).r;
    }
    raa_cloud_mean_sums[thread] = sum;
    workgroupBarrier();
    for (var stride = RAA_CLOUD_MEAN_THREADS / 2u; stride > 0u; stride /= 2u) {
        if thread < stride {
            raa_cloud_mean_sums[thread] += raa_cloud_mean_sums[thread + stride];
        }
        workgroupBarrier();
    }
    if thread == 0u {
        let mean = raa_cloud_mean_sums[0] / f32(max(count, 1u));
        textureStore(raa_cloud_shadow_mean, vec2<u32>(0u), vec4<f32>(mean, mean, mean, 1.0));
    }
}
