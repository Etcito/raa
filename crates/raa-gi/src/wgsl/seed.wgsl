// Jump flood seeding: marks each occupied voxel of a level as its own seed (GiFloodGpu is declared
// by the module prefix).
//
// Adapted from solstice@7e465b5, crates/solstice-render/shaders/gi/jump_flood.wgsl:37-47. MIT License
//
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

const RAA_GI_NO_SEED: u32 = 0xffffffffu;

@group(2) @binding(0) var<uniform> raa_gi_flood_params: GiFloodGpu;
@group(2) @binding(1) var<storage, read> raa_gi_voxels_read: array<u32>;
@group(2) @binding(2) var<storage, read_write> raa_gi_flood_out: array<u32>;

@compute @workgroup_size(4, 4, 4)
fn raa_gi_seed(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = raa_gi_flood_params.resolution;
    if any(id >= vec3<u32>(n)) {
        return;
    }
    let index = id.z * n * n + id.y * n + id.x;
    let word = raa_gi_voxels_read[(index + raa_gi_flood_params.level_offset) * 3u];
    let occupied = (word & 0xff000000u) != 0u;
    raa_gi_flood_out[index] = select(RAA_GI_NO_SEED, index, occupied);
}
