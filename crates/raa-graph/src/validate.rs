//! The validation context and the device limits validation reads (S52, S53).

use crate::catalog::{Catalog, ParamValue};

/// The per-stage binding limits a graph is validated against (RG_LIMITS).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphLimits {
    pub sampled_textures: u32,
    pub samplers: u32,
    pub storage_textures: u32,
    pub storage_buffers: u32,
    pub uniform_buffers: u32,
}

impl GraphLimits {
    /// The device-free limits RG_LIMITS names, for code without a device.
    pub const CORE: GraphLimits = GraphLimits {
        sampled_textures: 16,
        samplers: 16,
        storage_textures: 4,
        storage_buffers: 8,
        uniform_buffers: 12,
    };
}

/// Everything validation reads besides the document itself (S53).
pub struct ValidateCtx<'a> {
    pub catalog: &'a Catalog,
    pub limits: GraphLimits,
    pub installed: &'a dyn Fn(&str) -> bool,
    pub viewports: u32,
    pub engine_options: &'a [(&'static str, ParamValue)],
}
