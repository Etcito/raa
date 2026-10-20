//! Geometry provider trait and shared mesh data types.

use std::fmt;

pub mod analysis;
pub mod builtin;
mod mesh;
mod params;
mod provider;

pub use mesh::MeshData;
pub use params::{ParamValue, Params, Quality};
pub use provider::{GeometryProvider, GeometryRegistry};

/// Every refusal raa-geom can produce. Each variant has a stable code (see [`GeomError::code`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeomError {
    /// No provider is registered under the requested kind.
    UnknownKind(String),
    /// A provider's parameter is absent.
    MissingParam { kind: String, name: String },
    /// A provider's parameter is present but violates its rule.
    InvalidParam {
        kind: String,
        name: String,
        reason: String,
    },
    /// A provider with the same kind is already registered.
    DuplicateKind(String),
    /// Mesh data violates S2; `reason` names the first broken rule.
    InvalidMesh(String),
}

impl GeomError {
    /// Stable machine-readable code of the form `<area>.<snake_case>`.
    pub fn code(&self) -> &'static str {
        match self {
            GeomError::UnknownKind(_) => "geom.unknown_kind",
            GeomError::MissingParam { .. } => "geom.missing_param",
            GeomError::InvalidParam { .. } => "geom.invalid_param",
            GeomError::DuplicateKind(_) => "geom.duplicate_kind",
            GeomError::InvalidMesh(_) => "geom.invalid_mesh",
        }
    }
}

impl fmt::Display for GeomError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GeomError::UnknownKind(kind) => {
                write!(
                    f,
                    "geom.unknown_kind: no geometry provider for kind `{kind}`"
                )
            }
            GeomError::MissingParam { kind, name } => write!(
                f,
                "geom.missing_param: geometry kind `{kind}` has no parameter `{name}`"
            ),
            GeomError::InvalidParam { kind, name, reason } => write!(
                f,
                "geom.invalid_param: geometry kind `{kind}` parameter `{name}` is invalid: {reason}"
            ),
            GeomError::DuplicateKind(kind) => write!(
                f,
                "geom.duplicate_kind: geometry kind `{kind}` is already registered"
            ),
            GeomError::InvalidMesh(reason) => write!(f, "geom.invalid_mesh: {reason}"),
        }
    }
}

impl std::error::Error for GeomError {}
