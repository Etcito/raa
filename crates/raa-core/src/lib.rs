//! Device, queue and surface, resource handles, a minimal frame graph and the error type.

mod clear;
mod error;
mod gpu;
pub mod graph;
mod handle;
mod offscreen;
#[cfg(target_arch = "wasm32")]
mod surface;
#[cfg(not(target_arch = "wasm32"))]
pub mod testing;

pub use clear::ClearPass;
pub use error::RaaError;
pub use gpu::{AdapterSummary, DeviceRequest, GpuContext, GpuOptions};
pub use graph::{
    BACKBUFFER, FrameGraph, PassBuilder, PassId, PrepareContext, RecordContext, RenderPass,
    ResourceId, TargetDesc, TargetSize,
};
pub use handle::{Handle, Pool};
#[cfg(not(target_arch = "wasm32"))]
pub use offscreen::OffscreenTarget;
pub use offscreen::non_empty;
#[cfg(target_arch = "wasm32")]
pub use surface::{CanvasSurface, SurfaceFrame};

/// re-exported so extension crates implement RenderPass without their own wgpu dependency (version always matches)
pub use wgpu;

/// Frame clear colour, linear RGB (0.05, 0.06, 0.08, 1.0).
pub const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.05,
    g: 0.06,
    b: 0.08,
    a: 1.0,
};

/// Engine-wide mesh id shared by the scene, render and material crates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MeshId(pub u32);

/// Engine-wide material id shared by the scene, render and material crates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaterialId(pub u32);

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn ids_are_copy_and_compare_by_value() {
        let a = MeshId(3);
        let b = a;
        assert_eq!(a, b);
        assert_ne!(MeshId(3), MeshId(4));
        assert_eq!(MaterialId(7), MaterialId(7));
    }

    #[test]
    fn ids_order_by_inner_value() {
        let mut ids = vec![MaterialId(5), MaterialId(1), MaterialId(3)];
        ids.sort();
        assert_eq!(ids, vec![MaterialId(1), MaterialId(3), MaterialId(5)]);
        assert!(MeshId(0) < MeshId(u32::MAX));
    }

    #[test]
    fn ids_hash_by_value() {
        let set: HashSet<MeshId> = [MeshId(1), MeshId(1), MeshId(2)].into_iter().collect();
        assert_eq!(set.len(), 2);
        assert!(set.contains(&MeshId(2)));
    }
}
