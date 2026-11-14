// Forward voxelization (driver A): the lit instances no other driver claims write their
// world-space triangles into one clipmap level. Each thread handles one triangle of the work
// item in a grid-stride loop; the coverage scan lives in voxelize_kernel.wgsl.
//
// This is RAA code, not adapted from solstice.

struct GiForwardWorkGpu {
    slot: u32,
    first_index: u32,
    triangle_count: u32,
    level: u32,
    _pad0: vec4<u32>,
    _pad1: vec4<u32>,
}

@group(2) @binding(0) var<uniform> raa_gi_voxelize: GiVoxelizeGpu;
@group(2) @binding(1) var<uniform> raa_gi_work: GiForwardWorkGpu;
@group(2) @binding(2) var<storage, read> raa_gi_instances: array<RaaInstanceGpu>;
@group(2) @binding(3) var<storage, read> raa_gi_extras: array<InstanceExtraGpu>;
@group(2) @binding(4) var<storage, read> raa_gi_materials: array<SurfaceMaterialGpu>;
// Stream 0 (S1 Vertex, 24 B): position at floats 0..3, normal at 3..6, per vertex.
@group(2) @binding(5) var<storage, read> raa_gi_stream0: array<f32>;
@group(2) @binding(6) var<storage, read> raa_gi_indices: array<u32>;
@group(2) @binding(7) var<storage, read_write> raa_gi_voxels: array<atomic<u32>>;

@compute @workgroup_size(64)
fn raa_gi_voxelize_forward(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let slot = raa_gi_work.slot;
    let instance = raa_gi_instances[slot];
    let model = mat4x4<f32>(instance.model0, instance.model1, instance.model2, instance.model3);
    let material = raa_gi_materials[instance.material_index];
    let albedo = material.base.rgb * raa_gi_extras[slot].tint.rgb;
    let emissive = material.emissive.rgb;
    let origin = raa_gi_voxelize.origin;
    let voxel = raa_gi_voxelize.voxel_size;
    for (var t = gid.x; t < raa_gi_work.triangle_count; t += groups.x * 64u) {
        let base = raa_gi_work.first_index + 3u * t;
        let i0 = raa_gi_indices[base];
        let i1 = raa_gi_indices[base + 1u];
        let i2 = raa_gi_indices[base + 2u];
        let p0 = vec3<f32>(
            raa_gi_stream0[i0 * 6u],
            raa_gi_stream0[i0 * 6u + 1u],
            raa_gi_stream0[i0 * 6u + 2u],
        );
        let p1 = vec3<f32>(
            raa_gi_stream0[i1 * 6u],
            raa_gi_stream0[i1 * 6u + 1u],
            raa_gi_stream0[i1 * 6u + 2u],
        );
        let p2 = vec3<f32>(
            raa_gi_stream0[i2 * 6u],
            raa_gi_stream0[i2 * 6u + 1u],
            raa_gi_stream0[i2 * 6u + 2u],
        );
        let w0 = (model * vec4<f32>(p0, 1.0)).xyz;
        let w1 = (model * vec4<f32>(p1, 1.0)).xyz;
        let w2 = (model * vec4<f32>(p2, 1.0)).xyz;
        let face = cross(w1 - w0, w2 - w0);
        if dot(face, face) < 1.0e-16 {
            continue;
        }
        let data = RaaGiVoxelData(
            pack4x8unorm(vec4<f32>(albedo, 1.0)),
            pack2x16snorm(raa_gi_oct_encode(normalize(face))),
            raa_gi_encode_emissive(emissive),
        );
        raa_gi_voxelize_triangle(
            (w0 - origin) / voxel,
            (w1 - origin) / voxel,
            (w2 - origin) / voxel,
            data,
        );
    }
}
