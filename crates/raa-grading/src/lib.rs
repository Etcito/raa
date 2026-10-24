//! HDR colour grading before tonemap (S12 `grading`): the [`GradingDesc`] controls, their
//! presets (`neutral`, `candy`, `ice`, `jungle`, `neon`) and validation (`post.invalid`).

mod desc;

pub use desc::GradingDesc;
