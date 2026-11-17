//! Contrast-adaptive sharpening on the HDR chain (pass `sharpen`, S47). This share holds the S47
//! constants, the [`SharpenDesc`] settings with their validation (`post.invalid`) and the CPU
//! kernel twins. Later shares of this crate add the shaders and the feature.

mod desc;
mod kernel;
mod shaders;

pub use desc::SharpenDesc;
pub use kernel::{compress, expand, sharpen_texel};
pub use shaders::{FS_SHARPEN, SHARPEN_WGSL, VS_MAIN};

/// Name of the feature pass.
pub const SHARPEN_PASS_NAME: &str = "sharpen";
/// Order of the pass in the frame graph: after `cel.resolve` (190), before `tonemap` (200).
pub const SHARPEN_ORDER: u32 = 195;
/// Strength of the neighbour weight at sharpness 1 (the solstice constant).
pub const SHARPEN_WEIGHT: f32 = 0.2;
/// Upper bound of the compressed maximum channel, so that `expand` never divides by zero.
pub const COMPRESS_LIMIT: f32 = 0.999;
/// Bytes of the `SharpenParams` uniform (`sharpness: f32` @0, 0 @4-16).
pub const PARAMS_SIZE: usize = 16;
