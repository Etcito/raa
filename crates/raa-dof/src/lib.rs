//! Depth of field (S12 `dof`): the descriptor, its validation and the thin-lens circle of
//! confusion. The `dof` feature pass and its handle arrive with the feature task.

mod desc;
mod resources;
mod shaders;

pub use desc::{DofDesc, POST_INVALID, REFERENCE_VIEWPORT_H, SENSOR_WIDTH_MM, coc_px};
