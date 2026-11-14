//! WGSL of the GI volume (share 1): the radiance codec, the view-independent voxelize kernel
//! and the seed and jump-flood modules (contract `gi-shaders`, records of `gi-records`).

use std::num::NonZeroU64;

use wgpu::{BindGroupLayoutEntry, BindingType, BufferBindingType, ShaderStages};

/// Radiance codec (`raa_gi_encode_radiance`, `raa_gi_decode_radiance`, range 512).
pub const RADIANCE_CODEC_WGSL: &str = include_str!("wgsl/radiance_codec.wgsl");

/// Voxelize kernel: `GiVoxelizeGpu`, `RaaGiVoxelData` and the functions
/// `raa_gi_voxelize_triangle`, `raa_gi_encode_emissive`, `raa_gi_oct_encode`,
/// `raa_gi_write_voxel` and `raa_gi_clip_span`. They read the globals `raa_gi_voxels` and
/// `raa_gi_voxelize`, which each driver module declares.
///
/// Handoff: card 'raa-gi: shaders share 2, forward voxelize and volume write' takes over
/// through `VOXELIZE_KERNEL_WGSL` and `RADIANCE_CODEC_WGSL`; today this share ships both
/// texts and the kernel as listed above, and declares no binding for the kernel's globals.
pub const VOXELIZE_KERNEL_WGSL: &str = include_str!("wgsl/voxelize_kernel.wgsl");

const SEED_WGSL: &str = include_str!("wgsl/seed.wgsl");
const JUMP_FLOOD_WGSL: &str = include_str!("wgsl/jump_flood.wgsl");

const FLOOD_RECORD_WGSL: &str = "struct GiFloodGpu {
    step: vec3<i32>,
    resolution: u32,
    level_offset: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

";

/// Seed module: `FLOOD_RECORD_WGSL` + `seed.wgsl`, entry `raa_gi_seed` (4x4x4).
///
/// Handoff: card 'raa-gi: variants test' calls `seed_module_wgsl`, `flood_module_wgsl`,
/// `seed_group2_entries` and `flood_group2_entries`; today each returns the module text or the
/// three group-2 entries of contract 'gi-shaders'.
pub fn seed_module_wgsl() -> String {
    format!("{FLOOD_RECORD_WGSL}{SEED_WGSL}")
}

/// Flood module: `FLOOD_RECORD_WGSL` + `jump_flood.wgsl`, entry `raa_gi_flood` (4x4x4).
pub fn flood_module_wgsl() -> String {
    format!("{FLOOD_RECORD_WGSL}{JUMP_FLOOD_WGSL}")
}

fn entry(binding: u32, ty: BindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

fn flood_entries(read_only: bool) -> [BindGroupLayoutEntry; 3] {
    let storage = |read_only| BindingType::Buffer {
        ty: BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    [
        entry(
            0,
            BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: NonZeroU64::new(32),
            },
        ),
        entry(1, storage(read_only)),
        entry(2, storage(false)),
    ]
}

/// Seed group 2: GiFloodGpu (dynamic), voxels read, flood A write.
pub fn seed_group2_entries() -> [BindGroupLayoutEntry; 3] {
    flood_entries(true)
}

/// Flood group 2: GiFloodGpu (dynamic), flood in read, flood out write.
pub fn flood_group2_entries() -> [BindGroupLayoutEntry; 3] {
    flood_entries(true)
}
