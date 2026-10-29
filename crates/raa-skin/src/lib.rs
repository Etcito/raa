//! S18 skinned-mesh data: validated joints (u16x4) and weights (f32x4), their
//! little-endian GPU upload streams and the rest geometry they skin.

#![forbid(unsafe_code)]

mod cpu;
mod error;
mod gpu;
mod mesh;
mod pass;
mod pose;

pub use cpu::{CpuSkin, skin_cpu};
pub use error::SkinError;
pub use gpu::{SKIN_WGSL, SkinVertexInputs, SkinVertexKernel};
pub use mesh::{MAX_INFLUENCES, MAX_JOINTS, SkinMesh, WEIGHT_SUM_EPS};
pub use pass::{SKIN_FEATURE_BIT, SKIN_ORDER, SKIN_PASS, SkinPass};
pub use pose::{POSE_BYTES, PoseBuffer, validate_pose};

/// Absolute component-wise GPU/CPU tolerance; both values must also be finite.
pub const SKIN_PARITY_EPS: f32 = 1e-5;
