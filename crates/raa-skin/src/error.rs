//! Typed S18 skin refusals.

use std::fmt;

/// Every S18 skin refusal; `code` is stable (`skin.<snake_case>`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkinError {
    pub code: &'static str,
    pub message: String,
}

impl SkinError {
    pub(crate) fn new(code: &'static str, message: impl Into<String>) -> SkinError {
        SkinError {
            code,
            message: message.into(),
        }
    }

    pub fn code(&self) -> &'static str {
        self.code
    }
}

impl fmt::Display for SkinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for SkinError {}
