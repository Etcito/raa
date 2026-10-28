//! S18 skinned-mesh data: validated joints (u16x4) and weights (f32x4), their
//! little-endian GPU upload streams and the rest geometry they skin.

#![forbid(unsafe_code)]

mod error;
mod mesh;

pub use error::SkinError;
pub use mesh::{MAX_INFLUENCES, MAX_JOINTS, SkinMesh, WEIGHT_SUM_EPS};
