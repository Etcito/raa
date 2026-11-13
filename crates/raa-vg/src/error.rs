//! The `vg.*` refusal codes and the helpers that build their [`RaaError`]s (S31, S32).
//!
//! Every refusal is a typed value: a refused setter or path leaves the state it was given
//! unchanged and reports one of these codes with a message naming the field or the reason.

use raa_core::RaaError;

/// A settings field is outside its domain, or a patch is not a valid candidate.
pub const INVALID: &str = "vg.invalid";
/// The instance is skinned, so it stays on the forward path.
pub const SKINNED_UNSUPPORTED: &str = "vg.skinned_unsupported";
/// The instance is deformed (squash or jiggle) or its material defines a vertex function.
pub const DEFORMED_UNSUPPORTED: &str = "vg.deformed_unsupported";
/// The instance material is transparent or fur, so it stays on the forward path.
pub const TRANSPARENT_MATERIAL: &str = "vg.transparent_material";

/// A `vg.invalid` refusal whose message is `<field>: <reason>`; `field` is the S31 camelCase
/// name of the setting.
pub fn invalid(field: &str, reason: impl std::fmt::Display) -> RaaError {
    RaaError::FrameGraph {
        code: INVALID,
        message: format!("{field}: {reason}"),
    }
}

/// A refusal with a stable `vg.*` code and its message.
pub fn refusal(code: &'static str, message: String) -> RaaError {
    RaaError::FrameGraph { code, message }
}
