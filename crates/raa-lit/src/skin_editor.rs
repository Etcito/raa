//! Editor overlays follow the selected instance's posed geometry (S18). The wireframe
//! binds the instance's posed edge segments and the selection outline its posed stream 0.
//! An absent pose or disabled skinning selects the rest geometry in the same frame.

use raa_core::wgpu;

use crate::skin_views::{SKINNED_EDGE_STRIDE, SkinDrawSource};

/// Stride of a rest edge record: the two endpoint positions only.
pub const REST_EDGE_STRIDE: u64 = 24;

/// The geometry one editor overlay draw of an instance binds.
#[derive(Clone, Debug)]
pub struct OverlayGeometry {
    /// The instance's posed stream 0 when skinning is live, else the rest stream 0.
    pub stream0: wgpu::Buffer,
    /// The instance's posed edge segments when live, else the rest edge list.
    pub edges: Option<wgpu::Buffer>,
    /// Record stride of `edges`: [`REST_EDGE_STRIDE`] or [`SKINNED_EDGE_STRIDE`].
    pub edge_stride: u64,
    /// Whether `stream0` and `edges` are the posed geometry of the instance.
    pub posed: bool,
}

impl OverlayGeometry {
    /// Routes `tag` through the skin draw source; `rest_stream0` and `rest_edges` are the
    /// mesh's rest view. Posed geometry is selected only while skinning is enabled.
    pub fn select(
        source: &SkinDrawSource,
        tag: u32,
        rest_stream0: &wgpu::Buffer,
        rest_edges: Option<&wgpu::Buffer>,
    ) -> Self {
        let posed = source.posed(tag);
        let stream0 = source.stream0(tag).unwrap_or_else(|| rest_stream0.clone());
        let (edges, edge_stride) = match source.edges(tag) {
            Some(edges) => (Some(edges), SKINNED_EDGE_STRIDE),
            None => (rest_edges.cloned(), REST_EDGE_STRIDE),
        };
        Self {
            stream0,
            edges,
            edge_stride,
            posed,
        }
    }

    /// Number of edge segments the wireframe draws (zero without an edge list).
    pub fn edge_count(&self) -> u32 {
        self.edges
            .as_ref()
            .map_or(0, |edges| (edges.size() / self.edge_stride) as u32)
    }
}
