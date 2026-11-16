// Driver B of the GI volume: each workgroup voxelizes the clusters the GI cull selected for one
// clipmap level, one lane per triangle; the cluster records come from pass_view.wgsl and the
// coverage scan from voxelize_kernel.wgsl.
//
// Adapted from solstice@7e465b5, crates/solstice-render/shaders/gi/voxelize.wgsl:160-220. MIT License
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

// The S10 `SurfaceMaterialGpu` record, 256 bytes: `base` @0, `emissive` @16, `surface` @32,
// `sheen` @48, `transmission` @64, `attenuation` @80, `header` @96, `p` @112 and `fur` @240.
// The twin of `VgMaterial` in raa-vg resolve.wgsl.
struct RaaGiVgMaterial {
    base: vec4<f32>,
    emissive: vec4<f32>,
    surface: vec4<f32>,
    sheen: vec4<f32>,
    transmission: vec4<f32>,
    attenuation: vec4<f32>,
    header: vec4<u32>,
    p: array<vec4<f32>, 8>,
    fur: vec4<f32>,
}

@group(2) @binding(7) var<storage, read> raa_gi_vg_materials: array<RaaGiVgMaterial>;
// Three words per voxel: albedo with occupancy flag, octahedral normal, emissive RGBM.
@group(2) @binding(8) var<storage, read_write> raa_gi_voxels: array<atomic<u32>>;
@group(2) @binding(9) var<uniform> raa_gi_voxelize: GiVoxelizeGpu;

// One workgroup per selected cluster, striding over the rest; one lane per triangle. The
// software clusters fill `visible[state.sw_base .. state.sw_count)` from the front.
@compute @workgroup_size(128)
fn raa_gi_voxelize_virtual(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let software = state.sw_count - state.sw_base;
    for (var item = wg.x; item < software; item += groups.x) {
        raa_gi_voxelize_cluster_triangle(state.sw_base + item, lane);
    }
}

fn raa_gi_voxelize_cluster_triangle(entry_index: u32, t: u32) {
    let entry = visible[entry_index];
    let cluster = clusters[visible_cluster_index(entry)];
    if t >= cluster_triangle_count(cluster) {
        return;
    }
    let instance = instances[entry.instance];
    let material = raa_gi_vg_materials[instance.ids.y];

    let packed = triangles[instance.ranges.w + cluster.triangle_offset + t];
    let v0 = vertices[instance.ranges.z + cluster.vertex_offset + (packed & 0xffu)];
    let v1 = vertices[instance.ranges.z + cluster.vertex_offset + ((packed >> 8u) & 0xffu)];
    let v2 = vertices[instance.ranges.z + cluster.vertex_offset + ((packed >> 16u) & 0xffu)];
    let world0 = transform_point(instance, decode_position(instance, v0));
    let world1 = transform_point(instance, decode_position(instance, v1));
    let world2 = transform_point(instance, decode_position(instance, v2));

    let face = cross(world1 - world0, world2 - world0);
    let face_length = length(face);
    if face_length <= 0.0 {
        return;
    }
    // No instance tint: the 8-storage-buffer limit leaves no slot for the extras buffer.
    let data = RaaGiVoxelData(
        pack4x8unorm(vec4<f32>(material.base.rgb, 1.0)),
        pack2x16snorm(raa_gi_oct_encode(face / face_length)),
        raa_gi_encode_emissive(material.emissive.rgb),
    );
    let origin = raa_gi_voxelize.origin;
    let voxel = raa_gi_voxelize.voxel_size;
    raa_gi_voxelize_triangle(
        (world0 - origin) / voxel,
        (world1 - origin) / voxel,
        (world2 - origin) / voxel,
        data,
    );
}
