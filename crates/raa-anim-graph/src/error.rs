//! The `graph.*` refusal type.

use raa_anim::AnimError;
use std::fmt;

/// Stable S19 refusal codes.
pub mod codes {
    pub const FORMAT: &str = "graph.format";
    pub const VERSION: &str = "graph.version";
    pub const SCHEMA: &str = "graph.schema";
    pub const PARAM: &str = "graph.param";
    pub const CLIP_UNKNOWN: &str = "graph.clip_unknown";
    pub const STATE_UNKNOWN: &str = "graph.state_unknown";
    pub const BLEND_SPACE: &str = "graph.blend_space";
    pub const MASK: &str = "graph.mask";
    pub const CAP: &str = "graph.cap";
}

/// A refusal with a stable code and a JSON pointer path. Rig, clip and
/// reader failures from raa-anim keep their own `rig.*`, `clip.*` and `json.*` codes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphError {
    pub code: &'static str,
    pub path: String,
    pub message: String,
}

impl GraphError {
    pub fn new(code: &'static str, path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code,
            path: path.into(),
            message: message.into(),
        }
    }

    /// A count of `got` items above the inclusive `limit` is refused, never truncated.
    pub fn check_cap(got: usize, limit: usize, path: &str, what: &str) -> Result<(), Self> {
        if got > limit {
            return Err(Self::new(
                codes::CAP,
                path,
                format!("{what}: {got} exceeds the limit of {limit}"),
            ));
        }
        Ok(())
    }
}

impl From<AnimError> for GraphError {
    fn from(e: AnimError) -> Self {
        Self {
            code: e.code,
            path: e.path,
            message: e.message,
        }
    }
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {}: {}", self.code, self.path, self.message)
    }
}

impl std::error::Error for GraphError {}
