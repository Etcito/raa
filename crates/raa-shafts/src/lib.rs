//! Volumetric sun shafts (S12 `sunshafts`, order 69): descriptor, validation, the pass
//! shaders and the `sunshafts` feature pass with its handle.

mod desc;
mod feature;
mod shaders;

pub use desc::ShaftsDesc;
pub use feature::{ShaftsFeature, ShaftsHandle};
