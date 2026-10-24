//! HDR colour grading before tonemap (S12 `grading`): the [`GradingDesc`] controls, their
//! presets (`neutral`, `candy`, `ice`, `jungle`, `neon`), validation (`post.invalid`) and the
//! CAT02 [`white_balance_matrix`].

mod desc;
mod wb;

pub use desc::GradingDesc;
pub use wb::white_balance_matrix;
