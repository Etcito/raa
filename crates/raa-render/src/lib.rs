//! Forward renderer as a frame-graph pass.

mod forward;
mod prep;
mod renderer;

pub use forward::{FORWARD_WGSL, ForwardPass};
pub use prep::{FRAME_UNIFORM_SIZE, FramePrep, INSTANCE_STRIDE};
pub use renderer::Renderer;

use raa_core::MeshId;

/// Capacities the renderer preallocates once.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RendererOptions {
    /// Most instances drawn in one frame (default 16384).
    pub max_instances: u32,
    /// Most materials in the storage buffer (default 256).
    pub max_materials: u32,
}

impl Default for RendererOptions {
    fn default() -> Self {
        RendererOptions {
            max_instances: 16384,
            max_materials: 256,
        }
    }
}

/// What one frame submitted.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct FrameCounts {
    /// Instances written to the instance buffer.
    pub instances: u32,
    /// Instanced draw calls.
    pub draw_calls: u32,
    /// Triangles drawn (0 when only [`FramePrep::prepare_cpu`] ran).
    pub triangles: u64,
}

/// One instanced draw: `instance_count` instances of `mesh` starting at `first_instance`
/// in the instance buffer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DrawRange {
    /// Mesh to draw.
    pub mesh: MeshId,
    /// First instance in the instance buffer.
    pub first_instance: u32,
    /// Number of instances.
    pub instance_count: u32,
}
