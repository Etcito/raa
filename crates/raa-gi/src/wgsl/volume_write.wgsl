// Writes one clipmap level of the sampled atlas from the compute-side buffers: distance to the
// nearest surface in alpha, and that surface's outgoing radiance in colour.
//
// Near-surface empty voxels copy the radiance of their nearest occupied voxel. Otherwise filtered
// lookups at a ray's hit point would average in the black of empty space, and surfaces would
// darken as rays approach them.
//
// Adapted from solstice@7e465b5, crates/solstice-render/shaders/gi/volume_write.wgsl:10-51. MIT License
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

// The fields follow the S38 record order: wrap_origin @0, level @12, resolution @16,
// voxel_size @20 (the solstice Params put level first).
struct GiWriteGpu {
    wrap_origin: vec3<i32>,
    level: u32,
    resolution: u32,
    voxel_size: f32,
    _pad0: u32,
    _pad1: u32,
}

const RAA_GI_NO_SEED: u32 = 0xffffffffu;
// Far enough for filtered lookups to read valid colour, in voxels.
const COLOR_BAND: f32 = 3.0;
const FAR_DISTANCE_VOXELS: f32 = 64.0;

@group(2) @binding(0) var<uniform> raa_gi_write: GiWriteGpu;
@group(2) @binding(1) var<storage, read> raa_gi_seeds: array<u32>;
@group(2) @binding(2) var<storage, read> raa_gi_radiance_read: array<vec2<u32>>;
@group(2) @binding(3) var raa_gi_atlas_out: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn raa_gi_volume_write(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = raa_gi_write.resolution;
    if any(id >= vec3<u32>(n)) {
        return;
    }
    let level_offset = raa_gi_write.level * n * n * n;
    let seed = raa_gi_seeds[level_offset + id.z * n * n + id.y * n + id.x];

    var distance_voxels = FAR_DISTANCE_VOXELS;
    var color = vec3<f32>(0.0);
    if seed != RAA_GI_NO_SEED {
        let seed_coord = vec3<u32>(seed % n, (seed / n) % n, seed / (n * n));
        distance_voxels = length(vec3<f32>(seed_coord) - vec3<f32>(id));
        if distance_voxels <= COLOR_BAND {
            let world = vec3<i32>(seed_coord) + raa_gi_write.wrap_origin;
            let wrapped = vec3<u32>((world % i32(n) + vec3<i32>(i32(n))) % i32(n));
            color = raa_gi_decode_radiance(
                raa_gi_radiance_read[level_offset + wrapped.z * n * n + wrapped.y * n + wrapped.x],
            );
        }
    }
    let atlas_coord = vec3<u32>(id.x, id.y, id.z + raa_gi_write.level * n);
    textureStore(raa_gi_atlas_out, atlas_coord, vec4<f32>(color, distance_voxels * raa_gi_write.voxel_size));
}
