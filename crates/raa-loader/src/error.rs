//! The one error type of raa-loader.

use std::fmt;

use raa_geom::GeomError;
use raa_material::MaterialError;

/// A refusal with a stable S5 code, a JSON-pointer-like path and a message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadError {
    /// Stable code, for example `json.syntax`.
    pub code: &'static str,
    /// Where the problem is (JSON pointer, or `@<offset> <line>:<col>` for JSON syntax).
    pub path: String,
    /// Human-readable message.
    pub message: String,
}

impl LoadError {
    pub(crate) fn new(
        code: &'static str,
        path: impl Into<String>,
        message: impl Into<String>,
    ) -> LoadError {
        LoadError {
            code,
            path: path.into(),
            message: message.into(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn from_geom(e: &GeomError, path: impl Into<String>) -> LoadError {
        let code = match e {
            GeomError::UnknownKind(_) => "kind.unknown",
            GeomError::MissingParam { .. } => "kind.params",
            _ => "value.range",
        };
        LoadError::new(code, path, format!("{} ({})", e, e.code()))
    }

    #[allow(dead_code)]
    pub(crate) fn from_material(e: &MaterialError, path: impl Into<String>) -> LoadError {
        LoadError::new("value.range", path, format!("{} ({})", e.message, e.code))
    }
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {}: {}", self.code, self.path, self.message)
    }
}

impl std::error::Error for LoadError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_format() {
        let e = LoadError::new("json.syntax", "@0 1:1", "boom");
        assert_eq!(e.to_string(), "json.syntax at @0 1:1: boom");
    }

    #[test]
    fn geom_mapping() {
        let e = GeomError::UnknownKind("x".into());
        assert_eq!(LoadError::from_geom(&e, "/a").code, "kind.unknown");
        let e = GeomError::MissingParam {
            kind: "k".into(),
            name: "n".into(),
        };
        assert_eq!(LoadError::from_geom(&e, "/a").code, "kind.params");
        let e = GeomError::InvalidParam {
            kind: "k".into(),
            name: "n".into(),
            reason: "r".into(),
        };
        let l = LoadError::from_geom(&e, "/a");
        assert_eq!(l.code, "value.range");
        assert!(l.message.contains("geom.invalid_param"));
    }

    #[test]
    fn material_mapping() {
        let e = MaterialError {
            code: "material.invalid",
            message: "bad".into(),
        };
        assert_eq!(LoadError::from_material(&e, "/m").code, "value.range");
    }
}
