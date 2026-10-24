//! Sorted transparency (linked-list OIT): node layout, pool sizing and the
//! CPU reference of the resolve. The WGSL resolve follows `resolve_cpu`.

pub mod node;
pub mod resolve_cpu;

pub use node::{NODE_SIZE, OitNode, pool_nodes};
pub use resolve_cpu::{DEFAULT_MAX_FRAGMENTS, Fragment, resolve_pixel};
