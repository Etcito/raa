//! The `gi.*` refusal codes and the helper that builds their [`RaaError`]s (S36-S39).
//!
//! Every refusal is a typed value: a refused setter leaves the state it was given unchanged and
//! reports the code with a `<key>: <reason>` message.

use raa_core::RaaError;

/// A global-illumination field is outside its domain, or a patch is not a valid candidate.
pub const GI_INVALID: &str = "gi.invalid";

/// A `gi.invalid` refusal whose message is `<key>: <reason>`; `key` is the S36 camelCase name of
/// the setting or instance field.
pub fn invalid(key: &str, reason: &str) -> RaaError {
    RaaError::FrameGraph {
        code: GI_INVALID,
        message: format!("{key}: {reason}"),
    }
}
