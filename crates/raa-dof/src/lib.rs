//! Depth of field (S12 `dof`): the descriptor, its validation and the thin-lens circle of
//! confusion, and the `dof` feature pass with its handle.

mod desc;
mod feature;
mod resources;
mod shaders;

pub use desc::{DofDesc, POST_INVALID, REFERENCE_VIEWPORT_H, SENSOR_WIDTH_MM, coc_px};
pub use feature::{DofFeature, DofHandle};
