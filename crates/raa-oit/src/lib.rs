//! Sorted transparency (linked-list OIT): node layout, pool sizing, the CPU reference of
//! the resolve (the WGSL resolve follows `resolve_cpu`) and the `transparent.oit` feature
//! pass ([`OitFeature`], [`OitHandle`]).

mod buffers;
mod feature;
pub mod node;
pub mod resolve_cpu;
mod shaders;

pub use feature::{OitFeature, OitHandle, OitSettings, OitStats};
pub use node::{NODE_SIZE, OitNode, pool_nodes};
pub use resolve_cpu::{DEFAULT_MAX_FRAGMENTS, Fragment, resolve_pixel};
pub use shaders::{
    CLEAR_ENTRY, CLEAR_PARAMS_SIZE, CLEAR_WGSL, CLEAR_WORKGROUP_SIZE, RESOLVE_FS_ENTRY,
    RESOLVE_PARAMS_SIZE, RESOLVE_VS_ENTRY, RESOLVE_WGSL,
};
