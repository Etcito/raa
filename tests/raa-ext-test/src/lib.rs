//! Extends RAA without editing any RAA crate; nothing in the workspace depends on this crate.
//! Includes an external geometry provider and a custom, alpha-blended render pass.

mod star;
mod tint_pass;

pub use star::StarProvider;
pub use tint_pass::{TINT, TintPass};
