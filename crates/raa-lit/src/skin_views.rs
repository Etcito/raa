//! Per-instance skinned mesh views (S18): each view owns a private posed stream 0, an
//! auxiliary transformed-tangent stream and a posed edge-segment buffer, sharing the rest
//! mesh's stream 1 and u32 indices. An absent view or disabled skinning selects the rest.

use std::collections::BTreeMap;

use raa_core::wgpu;
use raa_core::{GpuContext, RaaError};
pub const SKINNED_STREAM0_STRIDE: u64 = 24;
pub const SKINNED_TANGENT_STRIDE: u64 = 16;
pub const SKINNED_EDGE_STRIDE: u64 = 48;
const REST_STREAM0_STRIDE: u64 = 24;
const REST_STREAM1_STRIDE: u64 = 40;
const REST_EDGE_STRIDE: u64 = 24;

/// The rest resources a [`SkinnedView`] shares. Stream 0 only sizes the private streams.
#[derive(Clone, Debug)]
pub struct RestMeshView {
    pub stream1: wgpu::Buffer,
    pub index: wgpu::Buffer,
    pub edges: Option<wgpu::Buffer>,
    pub vertices: u32,
    pub edge_count: u32,
}

impl RestMeshView {
    /// Refuses `render.invalid_mesh` unless stream 0 holds whole 24-byte vertices, stream 1
    /// matches them at 40 B, the index buffer is a non-empty multiple of 4 B and the edge
    /// list a multiple of 24 B (two positions per segment).
    pub fn new(
        stream0: wgpu::Buffer,
        stream1: wgpu::Buffer,
        index: wgpu::Buffer,
        edges: Option<wgpu::Buffer>,
    ) -> Result<Self, RaaError> {
        let vertices = count(stream0.size(), REST_STREAM0_STRIDE, "stream 0")?;
        let expected = vertices * REST_STREAM1_STRIDE;
        if stream1.size() != expected {
            return Err(invalid(format!(
                "rest stream 1 is {} bytes; {vertices} vertices need {expected}",
                stream1.size()
            )));
        }
        if index.size() == 0 || !index.size().is_multiple_of(4) {
            return Err(invalid(format!(
                "rest index buffer is {} bytes, not a positive multiple of 4",
                index.size()
            )));
        }
        let edge_count = match &edges {
            Some(edges) => count(edges.size(), REST_EDGE_STRIDE, "edge list")?,
            None => 0,
        };
        Ok(RestMeshView {
            stream1,
            index,
            edges,
            vertices: u32::try_from(vertices)
                .map_err(|_| invalid(format!("{vertices} vertices exceed u32")))?,
            edge_count: u32::try_from(edge_count)
                .map_err(|_| invalid(format!("{edge_count} edges exceed u32")))?,
        })
    }
}

#[derive(Debug)]
pub struct SkinnedView {
    pub rest: RestMeshView,
    pub stream0: wgpu::Buffer,
    pub tangents: wgpu::Buffer,
    pub edges: Option<wgpu::Buffer>,
}

#[derive(Debug, Default)]
pub struct SkinViews {
    views: BTreeMap<u32, SkinnedView>,
    disabled: bool,
}

impl SkinViews {
    /// Creates (or replaces) the view of `instance`, sized from `rest`. Each view gets its
    /// own `VERTEX | STORAGE | COPY_DST | COPY_SRC` buffers; the rest mesh is never written.
    /// Refusals: `render.invalid_mesh` from [`RestMeshView::new`].
    pub fn add_skinned_view(
        &mut self,
        gpu: &GpuContext,
        instance: u32,
        rest: RestMeshView,
    ) -> Result<(), RaaError> {
        let vertices = u64::from(rest.vertices);
        let stream0 = private(gpu, "stream0", instance, vertices * SKINNED_STREAM0_STRIDE);
        let tangents = private(gpu, "tangents", instance, vertices * SKINNED_TANGENT_STRIDE);
        let edge_bytes = u64::from(rest.edge_count) * SKINNED_EDGE_STRIDE;
        let edges = (rest.edge_count > 0).then(|| private(gpu, "edges", instance, edge_bytes));
        self.views.insert(
            instance,
            SkinnedView {
                rest,
                stream0,
                tangents,
                edges,
            },
        );
        Ok(())
    }

    /// Drops the view of `instance` and its buffers; other views are untouched.
    pub fn remove_skinned_view(&mut self, instance: u32) -> Option<SkinnedView> {
        self.views.remove(&instance)
    }

    /// A disabled `skin` pass draws the rest pose (S18); views and buffers are kept.
    pub fn set_skinning_enabled(&mut self, enabled: bool) {
        self.disabled = !enabled;
    }

    /// The view that draws `instance`: `None` when absent or while disabled.
    pub fn skinned_view(&self, instance: u32) -> Option<&SkinnedView> {
        (!self.disabled).then(|| self.views.get(&instance)).flatten()
    }

    pub fn skinned_stream0(&self, instance: u32) -> Option<&wgpu::Buffer> {
        self.skinned_view(instance).map(|v| &v.stream0)
    }

    pub fn skinned_tangents(&self, instance: u32) -> Option<&wgpu::Buffer> {
        self.skinned_view(instance).map(|v| &v.tangents)
    }

    pub fn skinned_edge_segments(&self, instance: u32) -> Option<&wgpu::Buffer> {
        self.skinned_view(instance).and_then(|v| v.edges.as_ref())
    }

    /// Stream 0 a draw of `instance` binds: private when drawn skinned, else `rest`.
    pub fn stream0_for<'a>(&'a self, instance: u32, rest: &'a wgpu::Buffer) -> &'a wgpu::Buffer {
        self.skinned_stream0(instance).unwrap_or(rest)
    }

    /// Edge segments a draw of `instance` binds: posed when skinned, else the rest list.
    pub fn edges_for<'a>(
        &'a self,
        instance: u32,
        rest: Option<&'a wgpu::Buffer>,
    ) -> Option<&'a wgpu::Buffer> {
        self.skinned_edge_segments(instance).or(rest)
    }
}

fn count(size: u64, stride: u64, name: &str) -> Result<u64, RaaError> {
    if size == 0 || !size.is_multiple_of(stride) {
        return Err(invalid(format!(
            "rest {name} is {size} bytes, not a positive multiple of {stride}"
        )));
    }
    Ok(size / stride)
}

fn invalid(message: String) -> RaaError {
    RaaError::FrameGraph {
        code: "render.invalid_mesh",
        message,
    }
}

fn private(gpu: &GpuContext, kind: &str, instance: u32, size: u64) -> wgpu::Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("raa.lit.skin.{kind}.{instance}")),
        size,
        usage: wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}
