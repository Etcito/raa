//! WGSL composition and validation of the lit renderer (S11). A composed module is the
//! prelude, the helper library, one material function, the winning provider of each slot
//! and the entry points of one pass kind. [`MaterialFnRegistry`] holds the material
//! functions; [`Composer::validate`] runs the naga WGSL front-end and validator so a broken
//! shader fails a CPU test without a GPU.

mod composer;
mod error;
mod ids;
mod registry;

pub use composer::Composer;
pub use error::ShaderError;
pub use ids::{MaterialFnId, PassKind, SlotId, VariantKey};
pub use registry::{MaterialFnEntry, MaterialFnRegistry};
