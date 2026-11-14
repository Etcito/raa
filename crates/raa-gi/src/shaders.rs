//! WGSL of the GI volume (shares 1, 2 and 3): the radiance codec, the view-independent voxelize
//! kernel, the seed, jump-flood, forward-voxelize and atlas-write modules and the voxel-light
//! module (contract `gi-shaders`, records of `gi-records`).
//!
//! Split note: `no handoff: raa-gi: shaders share 3, voxel lighting takes nothing from this card`
//! and `no handoff: raa-gi: shaders share 4, debug views takes nothing from this card`; each adds
//! its own builders and group-2 entries beside these.

use std::num::NonZeroU64;

use raa_shader::{CLOUD_SHADOW_WGSL, GI_VOLUME_WGSL, PRELUDE_WGSL};
use wgpu::{
    BindGroupLayoutEntry, BindingType, BufferBindingType, ShaderStages, StorageTextureAccess,
    TextureFormat, TextureViewDimension,
};

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
const VOXELIZE_FORWARD_WGSL: &str = include_str!("wgsl/voxelize_forward.wgsl");
const VOLUME_WRITE_WGSL: &str = include_str!("wgsl/volume_write.wgsl");
const VOXEL_LIGHT_WGSL: &str = include_str!("wgsl/voxel_light.wgsl");

const FLOOD_RECORD_WGSL: &str = "struct GiFloodGpu {
    step: vec3<i32>,
    resolution: u32,
    level_offset: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

";

/// `GI_FRAME_RECORD_WGSL` is the 432-byte `GiFrameGpu` struct text of contract 'gi-records',
/// mirrored by [`crate::layouts::GiFrameGpu`]; `voxel_light_module_wgsl` composes it before
/// `voxel_light.wgsl`.
///
/// Handoff: card 'raa-gi: shaders share 4, debug views' reuses `GI_FRAME_RECORD_WGSL` (this
/// file) as the frame record of its debug module; today it holds the `struct GiFrameGpu` text
/// at the S38 member offsets.
const GI_FRAME_RECORD_WGSL: &str = "struct GiFrameGpu {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    previous_view_proj: mat4x4<f32>,
    camera_position: vec4<f32>,
    sun_direction: vec4<f32>,
    sun_color: vec4<f32>,
    viewport: vec4<f32>,
    counts: vec4<u32>,
    settings: vec4<f32>,
    flags: vec4<u32>,
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

/// Forward-voxelize module (driver A): `PRELUDE_WGSL` (for `RaaInstanceGpu`,
/// `InstanceExtraGpu` and `SurfaceMaterialGpu`) + `VOXELIZE_KERNEL_WGSL` + `voxelize_forward.wgsl`,
/// entry `raa_gi_voxelize_forward` (workgroup 64, one thread per triangle, grid-stride).
pub fn voxelize_forward_module_wgsl() -> String {
    format!("{PRELUDE_WGSL}{VOXELIZE_KERNEL_WGSL}{VOXELIZE_FORWARD_WGSL}")
}

/// Atlas-write module: `RADIANCE_CODEC_WGSL` + `volume_write.wgsl`, entry `raa_gi_volume_write`
/// (4x4x4).
pub fn volume_write_module_wgsl() -> String {
    format!("{RADIANCE_CODEC_WGSL}{VOLUME_WRITE_WGSL}")
}

/// Voxel-light module (share 3): `PRELUDE_WGSL` + `CLOUD_SHADOW_WGSL` + `GI_VOLUME_WGSL` +
/// `RADIANCE_CODEC_WGSL` + `GI_FRAME_RECORD_WGSL` + `voxel_light.wgsl`, entry
/// `raa_gi_voxel_light` (4x4x4, one relit level per frame).
///
/// Handoff: card 'raa-gi: variants test' calls `voxel_light_module_wgsl` and
/// `light_group2_entries`; today they return the module text and the four group-2 entries of
/// contract 'gi-shaders'.
pub fn voxel_light_module_wgsl() -> String {
    format!(
        "{PRELUDE_WGSL}{CLOUD_SHADOW_WGSL}{GI_VOLUME_WGSL}{RADIANCE_CODEC_WGSL}\
         {GI_FRAME_RECORD_WGSL}{VOXEL_LIGHT_WGSL}"
    )
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

fn storage_entry(binding: u32, read_only: bool) -> BindGroupLayoutEntry {
    entry(
        binding,
        BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
    )
}

fn uniform_entry(binding: u32, size: u64, has_dynamic_offset: bool) -> BindGroupLayoutEntry {
    entry(
        binding,
        BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset,
            min_binding_size: NonZeroU64::new(size),
        },
    )
}

/// Forward group 2: GiVoxelizeGpu, GiForwardWorkGpu (dynamic, 256-B slots), instances, extras,
/// materials, stream 0, indices and the voxels buffer (six read storage buffers).
pub fn forward_group2_entries() -> [BindGroupLayoutEntry; 8] {
    [
        uniform_entry(0, 32, false),
        uniform_entry(1, 48, true),
        storage_entry(2, true),
        storage_entry(3, true),
        storage_entry(4, true),
        storage_entry(5, true),
        storage_entry(6, true),
        storage_entry(7, false),
    ]
}

/// Voxel-light group 2: GiLightGpu, voxels read, radiance read_write and the GiFrameGpu
/// uniform.
///
/// Handoff: card 'raa-gi: volume.rs share 1, allocation and pipelines' calls
/// `light_group2_entries` and `voxel_light_module_wgsl` to build the relight pipeline; today
/// they return the four group-2 entries and the module text.
pub fn light_group2_entries() -> [BindGroupLayoutEntry; 4] {
    [
        uniform_entry(0, 32, false),
        storage_entry(1, true),
        storage_entry(2, false),
        uniform_entry(3, 432, false),
    ]
}

/// Atlas-write group 2: GiWriteGpu, seeds, radiance and the Rgba16Float atlas storage texture.
pub fn write_group2_entries() -> [BindGroupLayoutEntry; 4] {
    [
        uniform_entry(0, 32, false),
        storage_entry(1, true),
        storage_entry(2, true),
        entry(
            3,
            BindingType::StorageTexture {
                access: StorageTextureAccess::WriteOnly,
                format: TextureFormat::Rgba16Float,
                view_dimension: TextureViewDimension::D3,
            },
        ),
    ]
}
