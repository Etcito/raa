// Jump flood pass along one axis: each voxel keeps the nearest seed of itself and its two
// neighbours at +-step (GiFloodGpu is declared by the module prefix).
//
// Adapted from solstice@7e465b5, crates/solstice-render/shaders/gi/jump_flood.wgsl:49-80. MIT License
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
@group(2) @binding(1) var<storage, read> raa_gi_flood_in: array<u32>;
@group(2) @binding(2) var<storage, read_write> raa_gi_flood_out: array<u32>;

fn raa_gi_flood_index(c: vec3<u32>) -> u32 {
    let n = raa_gi_flood_params.resolution;
    return c.z * n * n + c.y * n + c.x;
}

fn raa_gi_flood_coord(index: u32) -> vec3<f32> {
    let n = raa_gi_flood_params.resolution;
    return vec3<f32>(f32(index % n), f32((index / n) % n), f32(index / (n * n)));
}

@compute @workgroup_size(4, 4, 4)
fn raa_gi_flood(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = i32(raa_gi_flood_params.resolution);
    let c = vec3<i32>(id);
    if any(c >= vec3<i32>(n)) {
        return;
    }
    let here = vec3<f32>(c);
    var best = raa_gi_flood_in[raa_gi_flood_index(id)];
    var best_distance = 1.0e20;
    if best != RAA_GI_NO_SEED {
        let delta = raa_gi_flood_coord(best) - here;
        best_distance = dot(delta, delta);
    }
    for (var side = -1; side <= 1; side += 2) {
        let neighbor = c + raa_gi_flood_params.step * side;
        if any(neighbor < vec3<i32>(0)) || any(neighbor >= vec3<i32>(n)) {
            continue;
        }
        let candidate = raa_gi_flood_in[raa_gi_flood_index(vec3<u32>(neighbor))];
        if candidate != RAA_GI_NO_SEED {
            let delta = raa_gi_flood_coord(candidate) - here;
            let distance = dot(delta, delta);
            if distance < best_distance {
                best = candidate;
                best_distance = distance;
            }
        }
    }
    raa_gi_flood_out[raa_gi_flood_index(id)] = best;
}
