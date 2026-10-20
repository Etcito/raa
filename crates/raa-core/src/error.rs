//! The one error type of raa-core.

use std::fmt;

/// Every refusal raa-core can produce. Each variant has a stable code (see [`RaaError::code`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RaaError {
    /// No GPU adapter could be obtained (for example `navigator.gpu` is missing).
    NoAdapter(String),
    /// The adapter refused to create a device.
    RequestDevice(String),
    /// Surface creation, configuration or acquisition failed.
    Surface(String),
    /// A frame graph refusal; `code` is one of the `framegraph.*` codes.
    FrameGraph { code: &'static str, message: String },
    /// A resource (handle, target, buffer) is missing or invalid.
    Resource(String),
    /// A GPU operation failed (validation, mapping, readback).
    Gpu(String),
    /// The requested feature is not supported on this platform or adapter.
    Unsupported(String),
}

impl RaaError {
    /// Stable machine-readable code of the form `<area>.<snake_case>`.
    pub fn code(&self) -> &'static str {
        match self {
            RaaError::NoAdapter(_) => "core.no_adapter",
            RaaError::RequestDevice(_) => "core.request_device",
            RaaError::Surface(_) => "core.surface",
            RaaError::FrameGraph { code, .. } => code,
            RaaError::Resource(_) => "core.resource",
            RaaError::Gpu(_) => "core.gpu",
            RaaError::Unsupported(_) => "core.unsupported",
        }
    }

    /// Human-readable message, without the code.
    pub fn message(&self) -> &str {
        match self {
            RaaError::NoAdapter(m)
            | RaaError::RequestDevice(m)
            | RaaError::Surface(m)
            | RaaError::Resource(m)
            | RaaError::Gpu(m)
            | RaaError::Unsupported(m) => m,
            RaaError::FrameGraph { message, .. } => message,
        }
    }
}

impl fmt::Display for RaaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code(), self.message())
    }
}

impl std::error::Error for RaaError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_variant_has_its_stable_code() {
        let cases = [
            (RaaError::NoAdapter("a".into()), "core.no_adapter"),
            (RaaError::RequestDevice("a".into()), "core.request_device"),
            (RaaError::Surface("a".into()), "core.surface"),
            (RaaError::Resource("a".into()), "core.resource"),
            (RaaError::Gpu("a".into()), "core.gpu"),
            (RaaError::Unsupported("a".into()), "core.unsupported"),
        ];
        for (err, code) in cases {
            assert_eq!(err.code(), code);
        }
    }

    #[test]
    fn frame_graph_error_reports_its_own_code() {
        let err = RaaError::FrameGraph {
            code: "framegraph.duplicate_pass",
            message: "pass `clear` added twice".into(),
        };
        assert_eq!(err.code(), "framegraph.duplicate_pass");
        assert_eq!(err.message(), "pass `clear` added twice");
    }

    #[test]
    fn display_shows_code_then_message() {
        let err = RaaError::NoAdapter("navigator.gpu is missing".into());
        assert_eq!(err.to_string(), "core.no_adapter: navigator.gpu is missing");
    }

    #[test]
    fn codes_follow_area_dot_snake_case() {
        let errs = [
            RaaError::NoAdapter(String::new()),
            RaaError::RequestDevice(String::new()),
            RaaError::Surface(String::new()),
            RaaError::Resource(String::new()),
            RaaError::Gpu(String::new()),
            RaaError::Unsupported(String::new()),
        ];
        for err in errs {
            let (area, name) = err.code().split_once('.').expect("code has a dot");
            assert!(!area.is_empty() && !name.is_empty());
            assert!(name.chars().all(|c| c.is_ascii_lowercase() || c == '_'));
        }
    }

    #[test]
    fn is_a_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(RaaError::Gpu("map failed".into()));
        assert_eq!(err.to_string(), "core.gpu: map failed");
    }
}
