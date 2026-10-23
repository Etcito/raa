//! The crate's refusal type.

use std::fmt;

/// Refusal of a shader or material function operation: a stable code of the form
/// `<area>.<snake_case>` plus a human message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaderError {
    pub code: &'static str,
    pub message: String,
}

impl ShaderError {
    /// WGSL front-end error (message with line:col).
    pub const PARSE: &'static str = "shader.parse";
    /// naga validation error (message with line:col).
    pub const VALIDATE: &'static str = "shader.validate";
    /// Two providers of one slot with the same priority.
    pub const SLOT_CONFLICT: &'static str = "shader.slot_conflict";
    /// A slot has no provider.
    pub const SLOT_MISSING: &'static str = "shader.slot_missing";
    /// A variant names a material function the registry does not hold.
    pub const UNKNOWN_FN: &'static str = "shader.unknown_fn";
    /// A user material function breaks a source rule (message names the reason).
    pub const FN_INVALID: &'static str = "material.fn_invalid";
    /// A material function name is already registered.
    pub const FN_DUPLICATE: &'static str = "material.fn_duplicate";

    pub(crate) fn new(code: &'static str, message: String) -> ShaderError {
        ShaderError { code, message }
    }

    /// Stable machine-readable code of the form `<area>.<snake_case>`.
    pub fn code(&self) -> &'static str {
        self.code
    }
}

impl fmt::Display for ShaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for ShaderError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_have_area_and_snake_case_name() {
        let codes = [
            ShaderError::PARSE,
            ShaderError::VALIDATE,
            ShaderError::SLOT_CONFLICT,
            ShaderError::SLOT_MISSING,
            ShaderError::UNKNOWN_FN,
            ShaderError::FN_INVALID,
            ShaderError::FN_DUPLICATE,
        ];
        for code in codes {
            let (area, name) = code.split_once('.').expect("code has an area");
            assert!(area == "shader" || area == "material", "{code}");
            assert!(!name.is_empty(), "{code}");
            assert!(
                name.chars().all(|c| c.is_ascii_lowercase() || c == '_'),
                "{code}"
            );
        }
    }

    #[test]
    fn display_shows_code_and_message() {
        let e = ShaderError::new(ShaderError::PARSE, "3:7: expected ';'".to_string());
        assert_eq!(e.code(), "shader.parse");
        assert_eq!(e.to_string(), "shader.parse: 3:7: expected ';'");
    }
}
