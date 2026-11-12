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
