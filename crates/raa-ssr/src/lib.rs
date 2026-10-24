//! Screen-space reflections (S12 `ssr`): the descriptor and its validation, and the CPU
//! twin of the Hi-Z depth pyramid rule that the GPU reduction follows.

mod desc;
pub mod hiz;
mod resources;
mod shaders;

pub use desc::SsrDesc;
