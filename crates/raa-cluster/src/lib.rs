//! CPU cluster-DAG builder of RAA's virtualized geometry (S29): a saved mesh becomes clusters of
//! at most 128 triangles, arranged in a hierarchy of simplified levels and built in bounded work
//! units. The build is deterministic, needs no GPU and depends only on raa-math and raa-geom.
//! Adapted parts name their solstice source in their module docs; see README.md.

pub mod bake;
pub mod bounds;
pub mod cut;
pub mod dag;
#[doc(hidden)]
pub mod dvec;
pub mod error;
pub mod group;
pub mod hierarchy;
pub mod input;
pub mod packed;
pub mod partition;
pub mod simplify;
pub mod store;
pub mod thin;
pub mod weld;

pub use bake::{BakeProgress, BakeStage, BakeStatus, ClusterBake, WorkBudget, bake};
pub use bounds::Sphere;
pub use cut::{ClusterMesh, CutMesh, SectionHeader, cut_triangles, select_cut};
pub use error::ClusterError;
pub use input::{BuildOptions, ClusterInput, InputBounds, OwnedClusterInput};
pub use packed::{
    CLUSTER_ALGORITHM, CLUSTER_SECTION_VERSION, MAX_CLUSTER_TRIANGLES, MAX_CLUSTER_VERTICES,
    PACKED_CLUSTER_SIZE, PACKED_NODE_SIZE, PACKED_VERTEX_SIZE, PackedCluster, PackedNode,
    PackedVertex, ROOT_PARENT_ERROR, pack_counts, pack_triangle, unpack_triangle,
};
pub use store::{ClusterSource, ClusterState, ClusterStore, MeshClusterStatus};
