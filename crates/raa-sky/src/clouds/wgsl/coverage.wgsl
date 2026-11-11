// Cloud coverage CDF: histogram and cumulative share of the weather map coverage.
//
// Adapted from Solstice revision 7e465b599c13da165948ebaa44df824c36c3ae07,
// crates/solstice-render/shaders/clouds/coverage.wgsl:7-31.
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
// Entries are prefixed `raa_cloud_coverage_`. Groups 0 and 1 are empty.

@group(2) @binding(0) var raa_cloud_coverage_map: texture_2d<f32>;
@group(2) @binding(1) var<storage, read_write> raa_cloud_coverage_bins: array<atomic<u32>, 256>;
@group(2) @binding(2) var<storage, read_write> raa_cloud_coverage_cumulative: array<f32, 256>;

@compute @workgroup_size(8, 8)
fn raa_cloud_coverage_histogram(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(raa_cloud_coverage_map);
    if any(id.xy >= size) {
        return;
    }
    let value = textureLoad(raa_cloud_coverage_map, id.xy, 0).r;
    atomicAdd(&raa_cloud_coverage_bins[min(u32(value * 255.0 + 0.5), 255u)], 1u);
}

// The share of the map at or below each value.
@compute @workgroup_size(1)
fn raa_cloud_coverage_accumulate() {
    let size = textureDimensions(raa_cloud_coverage_map);
    let total = f32(size.x * size.y);
    var running = 0u;
    for (var i = 0u; i < 256u; i++) {
        running += atomicLoad(&raa_cloud_coverage_bins[i]);
        raa_cloud_coverage_cumulative[i] = f32(running) / total;
    }
}
