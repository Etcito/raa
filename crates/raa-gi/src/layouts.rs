//! CPU twins of the S38 `gi-records` GPU layouts: the frame record, the voxelize, flood, light
//! and write records, the forward work slot and the volume parameters. Every record is written
//! little-endian with `to_le_bytes` at the byte offsets of its WGSL struct, padding is zero, and
//! matrices go through `raa_math::Mat4::write_le_bytes`. There is no unsafe code.

use raa_math::Mat4;

/// Byte size of [`GiFrameGpu`] (S38).
pub const GI_FRAME_SIZE: usize = 432;

/// Byte stride of one [`GiForwardWorkGpu`] slot (S38): the dynamic-offset alignment.
pub const GI_WORK_STRIDE: usize = 256;

/// Writes `words` as little-endian 32-bit words starting at byte `at` of `out`.
fn put<const N: usize>(out: &mut [u8], at: usize, words: [[u8; 4]; N]) {
    for (i, word) in words.iter().enumerate() {
        let start = at + 4 * i;
        out[start..start + 4].copy_from_slice(word);
    }
}

/// The per-frame GI record (`GiFrameGpu`, 432 B, S38), the twin of the WGSL frame struct.
///
/// Handoff: `GiFrameGpu` (this file) is serialised today by `GiFrameGpu::to_bytes`: 432 bytes
/// from the fields as they are, with no GPU call. It is taken over by `raa-gi: layouts twin
/// test`, which calls `GiFrameGpu::to_bytes` and reads `GiFrameGpu::SIZE`; the same holds for
/// `GiVoxelizeGpu`, `GiFloodGpu`, `GiLightGpu`, `GiWriteGpu`, `GiForwardWorkGpu` and
/// `GiVolumeParamsGpu` in this file.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct GiFrameGpu {
    /// Current view-projection (offset 0).
    pub view_proj: Mat4,
    /// Inverse of `view_proj` (offset 64).
    pub inv_view_proj: Mat4,
    /// Inverse projection (offset 128).
    pub inv_proj: Mat4,
    /// Inverse view (offset 192).
    pub inv_view: Mat4,
    /// Previous frame's view-projection (offset 256).
    pub previous_view_proj: Mat4,
    /// Camera position, w 0 (offset 320).
    pub camera_position: [f32; 4],
    /// Direction toward the sun, w 0 (offset 336).
    pub sun_direction: [f32; 4],
    /// Linear rgb times intensity, w 0 (offset 352).
    pub sun_color: [f32; 4],
    /// Width, height, 1/width, 1/height (offset 368).
    pub viewport: [f32; 4],
    /// Active local lights, frame index, debug view code, level count (offset 384).
    pub counts: [u32; 4],
    /// Intensity, environment diffuse, near, far (offset 400).
    pub settings: [f32; 4],
    /// History valid, screen traces, local-light shadows, reflections (offset 416).
    pub flags: [u32; 4],
}

impl GiFrameGpu {
    /// Byte size of the record (S38).
    pub const SIZE: usize = GI_FRAME_SIZE;

    /// Serialises the record at the S38 offsets, little-endian.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut out = [0u8; Self::SIZE];
        self.view_proj.write_le_bytes(&mut out[0..64]);
        self.inv_view_proj.write_le_bytes(&mut out[64..128]);
        self.inv_proj.write_le_bytes(&mut out[128..192]);
        self.inv_view.write_le_bytes(&mut out[192..256]);
        self.previous_view_proj.write_le_bytes(&mut out[256..320]);
        put(&mut out, 320, self.camera_position.map(f32::to_le_bytes));
        put(&mut out, 336, self.sun_direction.map(f32::to_le_bytes));
        put(&mut out, 352, self.sun_color.map(f32::to_le_bytes));
        put(&mut out, 368, self.viewport.map(f32::to_le_bytes));
        put(&mut out, 384, self.counts.map(u32::to_le_bytes));
        put(&mut out, 400, self.settings.map(f32::to_le_bytes));
        put(&mut out, 416, self.flags.map(u32::to_le_bytes));
        out
    }
}

/// The voxelize record of one level (`GiVoxelizeGpu`, 32 B, S38).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct GiVoxelizeGpu {
    /// Level origin in metres (offset 0).
    pub origin: [f32; 3],
    /// Voxel size in metres (offset 12).
    pub voxel_size: f32,
    /// Voxels per axis (offset 16).
    pub resolution: u32,
    /// First voxel of the level (offset 20).
    pub level_offset: u32,
}

impl GiVoxelizeGpu {
    /// Byte size of the record (S38).
    pub const SIZE: usize = 32;

    /// Serialises the record; bytes 24..32 are zero padding.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut out = [0u8; Self::SIZE];
        put(&mut out, 0, self.origin.map(f32::to_le_bytes));
        put(&mut out, 12, [self.voxel_size.to_le_bytes()]);
        put(&mut out, 16, [self.resolution.to_le_bytes()]);
        put(&mut out, 20, [self.level_offset.to_le_bytes()]);
        out
    }
}

/// The jump-flood step record (`GiFloodGpu`, 32 B, S38).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct GiFloodGpu {
    /// Flood step per axis, in voxels (offset 0).
    pub step: [i32; 3],
    /// Voxels per axis (offset 12).
    pub resolution: u32,
    /// First voxel of the level (offset 16).
    pub level_offset: u32,
}

impl GiFloodGpu {
    /// Byte size of the record (S38).
    pub const SIZE: usize = 32;

    /// Serialises the record; bytes 20..32 are zero padding.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut out = [0u8; Self::SIZE];
        put(&mut out, 0, self.step.map(i32::to_le_bytes));
        put(&mut out, 12, [self.resolution.to_le_bytes()]);
        put(&mut out, 16, [self.level_offset.to_le_bytes()]);
        out
    }
}

/// The voxel-lighting record (`GiLightGpu`, 32 B, S38).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct GiLightGpu {
    /// Wrap origin of the level in voxels (offset 0).
    pub wrap_origin: [i32; 3],
    /// Level being relit (offset 12).
    pub level: u32,
    /// Checkerboard update counter (offset 16).
    pub updates: u32,
    /// Voxels per axis (offset 20).
    pub resolution: u32,
}

impl GiLightGpu {
    /// Byte size of the record (S38).
    pub const SIZE: usize = 32;

    /// Serialises the record; bytes 24..32 are zero padding.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut out = [0u8; Self::SIZE];
        put(&mut out, 0, self.wrap_origin.map(i32::to_le_bytes));
        put(&mut out, 12, [self.level.to_le_bytes()]);
        put(&mut out, 16, [self.updates.to_le_bytes()]);
        put(&mut out, 20, [self.resolution.to_le_bytes()]);
        out
    }
}

/// The atlas write record (`GiWriteGpu`, 32 B, S38).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct GiWriteGpu {
    /// Wrap origin of the level in voxels (offset 0).
    pub wrap_origin: [i32; 3],
    /// Level written (offset 12).
    pub level: u32,
    /// Voxels per axis (offset 16).
    pub resolution: u32,
    /// Voxel size in metres (offset 20).
    pub voxel_size: f32,
}

impl GiWriteGpu {
    /// Byte size of the record (S38).
    pub const SIZE: usize = 32;

    /// Serialises the record; bytes 24..32 are zero padding.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut out = [0u8; Self::SIZE];
        put(&mut out, 0, self.wrap_origin.map(i32::to_le_bytes));
        put(&mut out, 12, [self.level.to_le_bytes()]);
        put(&mut out, 16, [self.resolution.to_le_bytes()]);
        put(&mut out, 20, [self.voxel_size.to_le_bytes()]);
        out
    }
}

/// One forward voxelize work item (`GiForwardWorkGpu`, 48 B in [`GI_WORK_STRIDE`] slots, S38).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct GiForwardWorkGpu {
    /// Instance slot (offset 0).
    pub slot: u32,
    /// First index of the triangle range (offset 4).
    pub first_index: u32,
    /// Triangles in the range (offset 8).
    pub triangle_count: u32,
    /// Level voxelized (offset 12).
    pub level: u32,
}

impl GiForwardWorkGpu {
    /// Byte size of the record (S38); the slot tail up to [`GI_WORK_STRIDE`] is not written.
    pub const SIZE: usize = 48;

    /// Serialises the record; bytes 16..48 are zero padding.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut out = [0u8; Self::SIZE];
        put(&mut out, 0, [self.slot.to_le_bytes()]);
        put(&mut out, 4, [self.first_index.to_le_bytes()]);
        put(&mut out, 8, [self.triangle_count.to_le_bytes()]);
        put(&mut out, 12, [self.level.to_le_bytes()]);
        out
    }
}

/// The volume parameters passed to `LightingResources::set_gi` (`GiVolumeParamsGpu`, 96 B, S38).
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct GiVolumeParamsGpu {
    /// Per level: origin in metres (xyz) and voxel size (w); unused levels are zero (offset 0).
    pub levels: [[f32; 4]; 5],
    /// Voxels per axis as a float (offset 80).
    pub resolution: f32,
    /// Indirect intensity (offset 84).
    pub intensity: f32,
    /// Level count, 0 when there is no volume (offset 88).
    pub level_count: u32,
    /// Bit 0 valid, bit 1 reflections, bit 2 local-light shadows (offset 92).
    pub flags: u32,
}

impl GiVolumeParamsGpu {
    /// Byte size of the record (S38).
    pub const SIZE: usize = 96;

    /// Serialises the record at the S38 offsets, little-endian.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut out = [0u8; Self::SIZE];
        for (i, level) in self.levels.iter().enumerate() {
            put(&mut out, 16 * i, level.map(f32::to_le_bytes));
        }
        put(&mut out, 80, [self.resolution.to_le_bytes()]);
        put(&mut out, 84, [self.intensity.to_le_bytes()]);
        put(&mut out, 88, [self.level_count.to_le_bytes()]);
        put(&mut out, 92, [self.flags.to_le_bytes()]);
        out
    }
}
