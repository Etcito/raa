// Writes one clipmap level of the sampled atlas from the compute-side buffers: distance to the
// nearest surface in alpha, and that surface's outgoing radiance in colour.
//
// Near-surface empty voxels copy the radiance of their nearest occupied voxel. Otherwise filtered
// lookups at a ray's hit point would average in the black of empty space, and surfaces would
// darken as rays approach them. The copy is one-sided (RAA): a voxel's radiance is the light its
// surface sends out along its normal, so only the empty voxels in front of that surface (not
// more than half a voxel behind its plane) take it; those behind stay black. A ray arriving at a
// thin wall from the unlit side then reads black instead of the light of the other side, and a
// lit box wall no longer lights the inside of the box.
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
// The voxel words (albedo, normal, emissive per voxel): the seed's normal.
@group(2) @binding(4) var<storage, read> raa_gi_write_voxels: array<u32>;

// Empty voxels further than this behind the seed's surface plane (in voxels) take no radiance.
const RAA_GI_BEHIND: f32 = 0.5;

// Signed octahedral coordinates in [-1, 1]^2 to a unit vector.
fn raa_gi_write_oct_decode(e: vec2<f32>) -> vec3<f32> {
    var v = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if v.z < 0.0 {
        let signs = vec2<f32>(select(-1.0, 1.0, v.x >= 0.0), select(-1.0, 1.0, v.y >= 0.0));
        let folded = (vec2<f32>(1.0) - abs(v.yx)) * signs;
        v = vec3<f32>(folded.x, folded.y, v.z);
    }
    return normalize(v);
}

// Whether an empty voxel at `offset` voxels from its seed lies in front of the seed's surface:
// at most RAA_GI_BEHIND voxels behind the plane through the seed with the seed's normal
// (`cpu::in_front_of_seed`). The seed itself (offset 0) always is.
fn raa_gi_in_front_of_seed(offset: vec3<f32>, normal: vec3<f32>) -> bool {
    return dot(offset, normal) >= -RAA_GI_BEHIND;
}

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
        let seed_normal = raa_gi_write_oct_decode(
            unpack2x16snorm(raa_gi_write_voxels[(level_offset + seed) * 3u + 1u]));
        let in_front = raa_gi_in_front_of_seed(vec3<f32>(id) - vec3<f32>(seed_coord), seed_normal);
        if distance_voxels <= COLOR_BAND && in_front {
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
