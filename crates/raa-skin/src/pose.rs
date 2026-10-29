//! Per-instance S18 pose buffers: 16 column-major f32 per joint, validated before upload.

use raa_core::{GpuContext, wgpu};

use crate::{MAX_JOINTS, SkinError};

/// Bytes of one joint matrix in a pose buffer.
pub const POSE_BYTES: u64 = 64;

/// Joint count of a pose slice: `skin.pose_length` for an empty, truncated or non-finite
/// pose, `skin.cap_joints` past [`MAX_JOINTS`].
pub fn validate_pose(matrices: &[f32]) -> Result<usize, SkinError> {
    let length_ok = !matrices.is_empty()
        && matrices.len().is_multiple_of(16)
        && matrices.iter().all(|x| x.is_finite());
    if !length_ok {
        return Err(SkinError::new(
            "skin.pose_length",
            "expected 16 finite floats per joint",
        ));
    }
    let joints = matrices.len() / 16;
    if joints > MAX_JOINTS {
        return Err(SkinError::new(
            "skin.cap_joints",
            format!("pose has {joints} joints; the S18 cap is {MAX_JOINTS}"),
        ));
    }
    Ok(joints)
}

/// The pose storage of one instance; it grows when a longer pose arrives.
#[derive(Debug)]
pub struct PoseBuffer(wgpu::Buffer);

impl PoseBuffer {
    pub fn new(gpu: &GpuContext, instance: u32, matrices: &[f32]) -> Result<Self, SkinError> {
        let buffer = allocate(gpu, instance, validate_pose(matrices)?);
        gpu.queue().write_buffer(&buffer, 0, &bytes(matrices));
        Ok(PoseBuffer(buffer))
    }

    /// Uploads `matrices`, reallocating only when they need more joints.
    pub fn write(
        &mut self,
        gpu: &GpuContext,
        instance: u32,
        matrices: &[f32],
    ) -> Result<(), SkinError> {
        let joints = validate_pose(matrices)?;
        if self.0.size() < joints as u64 * POSE_BYTES {
            self.0 = allocate(gpu, instance, joints);
        }
        gpu.queue().write_buffer(&self.0, 0, &bytes(matrices));
        Ok(())
    }

    /// Storage buffer bound at group 0, binding 1 of `skin.wgsl`.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.0
    }
}

fn bytes(matrices: &[f32]) -> Vec<u8> {
    matrices.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn allocate(gpu: &GpuContext, instance: u32, joints: usize) -> wgpu::Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("raa.skin.pose.{instance}")),
        size: joints as u64 * POSE_BYTES,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
