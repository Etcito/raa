//! The primitive asset format v0 (`raa.prims`).

mod build;
mod model;
mod parse;
pub mod samples;
mod write;

pub use build::{PrimsLoader, build, estimate_triangles};
pub use model::{ColliderDoc, NodeOp, PrimMaterial, PrimNode, PrimsAsset, TransformDoc};
pub use write::to_canonical_json;

use crate::error::LoadError;

/// Parses prims v0 text with the 8 S3 kinds only; the first error in document order wins.
pub fn parse(text: &str) -> Result<PrimsAsset, LoadError> {
    parse::parse_with(text, true)
}

/// Format string of every prims document.
pub const FORMAT: &str = "raa.prims";
/// The only supported format version.
pub const VERSION: u32 = 0;
/// Triangle budget when the document gives none (TRIANGLE_BUDGET_DEFAULT).
pub const DEFAULT_TRIANGLE_BUDGET: u32 = 1500;
/// Tessellation of a published document without `budget.segments` (DEFAULT_SEGMENTS).
pub const DEFAULT_SEGMENTS: u32 = 16;
/// Maximum node count (ASSET_CAPS).
pub const MAX_NODES: usize = 64;
/// Maximum nesting level, roots are level 0 (ASSET_CAPS).
pub const MAX_LEVEL: usize = 8;
