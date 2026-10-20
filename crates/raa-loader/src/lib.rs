//! Asset loading trait and the primitive-asset (prims v0) loader.

mod boolean;
mod collider;
mod error;
pub mod json;
pub mod prims;
mod registry;

pub use boolean::{BoolOp, BooleanHost, BooleanResolver, BooleanTool, NoBooleans};
pub use collider::{Collider, ColliderShape};
pub use error::LoadError;
pub use registry::{AssetLoader, LoaderRegistry};

use raa_geom::{GeometryRegistry, MeshData, Quality};
use raa_material::Material;
use raa_math::Mat4;

/// Reserved name of the default material (linear grey 0.5, roughness 0.6, metallic 0).
#[allow(dead_code)] // used by the prims build step
pub(crate) const DEFAULT_MATERIAL: &str = "_default";

/// What a loader needs besides the bytes.
pub struct LoadContext<'a> {
    /// Providers used to generate primitive meshes.
    pub geometry: &'a GeometryRegistry,
    /// Evaluates subtract/intersect children.
    pub booleans: &'a dyn BooleanResolver,
    /// Tessellation quality of the generated meshes.
    pub quality: Quality,
    /// When true only the 8 S3 kinds are accepted (`kind.unknown`); when false any kind
    /// registered in `geometry` is accepted and its params are validated by the provider.
    pub strict_kinds: bool,
}

/// A loaded asset: meshes, materials, anchors and colliders.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedAsset {
    pub format: String,
    pub id: String,
    pub name: String,
    /// Depth-first pre-order.
    pub meshes: Vec<LoadedMesh>,
    pub materials: Vec<(String, Material)>,
    pub anchors: Vec<(String, Mat4)>,
    pub colliders: Vec<Collider>,
    pub triangles: u32,
    pub budget: u32,
}

/// One closed mesh of a loaded asset.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedMesh {
    pub node_id: String,
    pub kind: String,
    /// Node-local primitive space, booleans applied.
    pub mesh: MeshData,
    /// Asset space.
    pub transform: Mat4,
    /// Resolved material name (inherited when the node has none).
    pub material: String,
}
