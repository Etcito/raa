//! Screen-space reflections (S12 `ssr`): the descriptor and its validation, the CPU twin of
//! the Hi-Z depth pyramid rule that the GPU reduction follows, and the `ssr` pass.

mod desc;
mod feature;
pub mod hiz;
mod resources;
mod shaders;

pub use desc::SsrDesc;
pub use feature::{SsrFeature, SsrHandle};
