//! HDR colour grading before tonemap (S12 `grading`): the [`GradingDesc`] controls, their
//! presets (`neutral`, `candy`, `ice`, `jungle`, `neon`), validation (`post.invalid`), the
//! CAT02 [`white_balance_matrix`] and the [`GradingFeature`] pass with its [`GradingHandle`].

mod desc;
mod feature;
mod shaders;
mod wb;

pub use desc::GradingDesc;
pub use feature::{GradingFeature, GradingHandle};
pub use wb::white_balance_matrix;
