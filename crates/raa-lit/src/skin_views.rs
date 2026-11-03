//! Skinned draw state (S18), shared by the `skin` pass and the lit draws.
//!
//! - The joint **palette**: one storage buffer of joint matrices for every posed instance.
//!   Each instance tag owns a block of whole [`PALETTE_BLOCK`] matrices (256 B, the
//!   WebGPU storage offset alignment), freed blocks are reused, the buffer is re-created
//!   when it grows and the CPU copy is written with one `write_buffer` per frame
//!   ([`SkinViews::flush_palette`]). The lit vertex stage skins posed instances from it:
//!   the S1 instance record carries the block base (bytes 120..124) and the per-vertex
//!   joints/weights come from the mesh's rest skin records, so posed instances of one mesh
//!   and material draw as ONE instanced draw with no per-instance buffers.
//! - Overlay **views** (editor only): a private posed stream 0, tangents and edge segments
//!   for an instance whose outline or wireframe is on; the `skin` pass fills them by
//!   compute from the palette. Nothing else dispatches compute for skinned vertices.
//!
//! An absent pose or disabled skinning selects the rest geometry.

use std::collections::{BTreeMap, BTreeSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::{cell::RefCell, rc::Rc};

use raa_core::wgpu;
use raa_core::{GpuContext, RaaError};

pub const SKINNED_STREAM0_STRIDE: u64 = 24;
pub const SKINNED_TANGENT_STRIDE: u64 = 16;
pub const SKINNED_EDGE_STRIDE: u64 = 48;
/// Stride of a rest skin record (position, normal, tangent, joints u16x4, weights f32x4).
pub const SKIN_RECORD_STRIDE: u64 = 64;
/// Matrices per palette allocation unit: 4 x 64 B = 256 B (storage offset alignment).
pub const PALETTE_BLOCK: u32 = 4;
/// Bytes of one palette matrix.
pub const PALETTE_MATRIX_BYTES: u64 = 64;
/// Byte offset in the S1 instance record of the palette base (in matrices) of a posed draw.
pub const INSTANCE_PALETTE_OFFSET: usize = 120;
const REST_STREAM0_STRIDE: u64 = 24;
const REST_STREAM1_STRIDE: u64 = 40;
const REST_EDGE_STRIDE: u64 = 24;

/// What a posed instance draws with: its palette block base (matrices) and the identity
/// of its mesh's rest skin record buffer. Instances with equal `skin` batch together.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SkinDraw {
    pub base: u32,
    pub skin: u64,
}

#[derive(Debug)]
struct Posed {
    skin: wgpu::Buffer,
    skin_key: u64,
    base: u32,
    capacity: u32,
    joints: Vec<f32>,
}

/// The palette: CPU copy, GPU buffer, free blocks.
#[derive(Debug, Default)]
struct Palette {
    cpu: Vec<f32>,
    bytes: Vec<u8>,
    buffer: Option<wgpu::Buffer>,
    generation: u64,
    free: Vec<(u32, u32)>,
    dirty: bool,
}

impl Palette {
    fn end(&self) -> u32 {
        (self.cpu.len() / 16) as u32
    }

    fn alloc(&mut self, capacity: u32) -> u32 {
        if let Some(i) = self.free.iter().position(|&(_, len)| len >= capacity) {
            let (base, len) = self.free[i];
            if len == capacity {
                self.free.remove(i);
            } else {
                self.free[i] = (base + capacity, len - capacity);
            }
            return base;
        }
        let base = self.end();
        self.cpu.resize((base + capacity) as usize * 16, 0.0);
        base
    }

    fn release(&mut self, base: u32, capacity: u32) {
        self.free.push((base, capacity));
        self.free.sort_unstable();
        let mut merged: Vec<(u32, u32)> = Vec::with_capacity(self.free.len());
        for &(b, l) in &self.free {
            match merged.last_mut() {
                Some(last) if last.0 + last.1 == b => last.1 += l,
                _ => merged.push((b, l)),
            }
        }
        if let Some(&(b, l)) = merged.last()
            && b + l == self.end()
        {
            merged.pop();
            self.cpu.truncate(b as usize * 16);
        }
        self.free = merged;
        self.dirty = true;
    }

    /// Grows the GPU buffer to hold every block (power of two, at least 256 B).
    fn ensure_buffer(&mut self, gpu: &GpuContext) {
        let needed = (u64::from(self.end()) * PALETTE_MATRIX_BYTES).max(256);
        if self.buffer.as_ref().is_some_and(|b| b.size() >= needed) {
            return;
        }
        self.buffer = Some(gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa.lit.skin.palette"),
            size: needed.next_power_of_two(),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        self.generation += 1;
        self.dirty = true;
    }
}

/// Which instances want overlay geometry (mirrors the lit editor state).
#[derive(Debug, Default)]
struct OverlayRequests {
    global: bool,
    wire: BTreeMap<u32, bool>,
    outlines: BTreeSet<u32>,
}

impl OverlayRequests {
    fn wanted(&self, tag: u32) -> bool {
        self.outlines.contains(&tag) || self.wire.get(&tag).copied().unwrap_or(self.global)
    }
}

/// Live skinned state, shared by the skin pass ([`SkinViews`]) and the lit renderer.
#[derive(Debug, Default)]
pub(crate) struct SkinShared {
    disabled: bool,
    posed: BTreeMap<u32, Posed>,
    palette: Palette,
    /// Overlay buffers by tag: posed stream 0 and edge segments.
    overlays: BTreeMap<u32, (wgpu::Buffer, Option<wgpu::Buffer>)>,
    requests: OverlayRequests,
}

impl SkinShared {
    /// The draw key of `tag` while it draws posed (a pose, a palette, skinning enabled).
    pub(crate) fn draw(&self, tag: u32) -> Option<SkinDraw> {
        if self.disabled || self.palette.buffer.is_none() {
            return None;
        }
        self.posed.get(&tag).map(|p| SkinDraw {
            base: p.base,
            skin: p.skin_key,
        })
    }

    /// The last pose of `tag` (16 floats per joint).
    pub(crate) fn joints(&self, tag: u32) -> Option<&[f32]> {
        self.posed.get(&tag).map(|p| p.joints.as_slice())
    }
}

#[derive(Clone, Debug, Default)]
pub struct SkinDrawSource(pub(crate) Rc<RefCell<SkinShared>>);

impl SkinDrawSource {
    /// Whether `tag` draws posed geometry: it has a pose and skinning is enabled.
    pub fn posed(&self, tag: u32) -> bool {
        self.0.borrow().draw(tag).is_some()
    }

    /// The draw key of `tag` while it draws posed.
    pub fn draw(&self, tag: u32) -> Option<SkinDraw> {
        self.0.borrow().draw(tag)
    }

    /// The rest skin record buffer (64 B per vertex) a posed draw of `tag` binds in slot 3.
    pub fn skin_buffer(&self, tag: u32) -> Option<wgpu::Buffer> {
        let s = self.0.borrow();
        s.draw(tag)
            .and_then(|_| s.posed.get(&tag).map(|p| p.skin.clone()))
    }

    /// The palette buffer and its generation (bumped whenever the buffer is re-created).
    pub fn palette(&self) -> Option<(wgpu::Buffer, u64)> {
        let s = self.0.borrow();
        s.palette.buffer.clone().map(|b| (b, s.palette.generation))
    }

    /// The last pose of `tag` (for posed culling bounds) while it draws posed.
    pub fn joints(&self, tag: u32) -> Option<Vec<f32>> {
        let s = self.0.borrow();
        s.draw(tag)
            .and_then(|_| s.posed.get(&tag).map(|p| p.joints.clone()))
    }

    /// Whether `tag` has a live overlay view (editor geometry) while skinning is enabled.
    pub fn overlay_posed(&self, tag: u32) -> bool {
        let s = self.0.borrow();
        !s.disabled && s.overlays.contains_key(&tag)
    }

    /// The posed overlay stream 0 of `tag`; `None` draws the rest mesh.
    pub fn stream0(&self, tag: u32) -> Option<wgpu::Buffer> {
        let s = self.0.borrow();
        (!s.disabled)
            .then(|| s.overlays.get(&tag).map(|v| v.0.clone()))
            .flatten()
    }

    /// The posed overlay edge segments of `tag`; `None` draws the rest edge list.
    pub fn edges(&self, tag: u32) -> Option<wgpu::Buffer> {
        let s = self.0.borrow();
        (!s.disabled)
            .then(|| s.overlays.get(&tag).and_then(|v| v.1.clone()))
            .flatten()
    }

    /// Mirrors the editor: the global wireframe is on, per-tag wireframe overrides (on/off)
    /// and outlined tags. The skin pass builds overlay geometry only for wanted tags.
    pub fn set_overlay_requests(
        &self,
        global: bool,
        wire: impl IntoIterator<Item = (u32, bool)>,
        outlines: impl IntoIterator<Item = u32>,
    ) {
        let mut s = self.0.borrow_mut();
        s.requests.global = global;
        s.requests.wire = wire.into_iter().collect();
        s.requests.outlines = outlines.into_iter().collect();
    }

    /// Whether the editor wants overlay geometry for `tag`.
    pub fn overlay_wanted(&self, tag: u32) -> bool {
        self.0.borrow().requests.wanted(tag)
    }
}

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

/// The editor overlay geometry of one posed instance (filled by the skin pass compute).
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
    source: SkinDrawSource,
}

impl SkinViews {
    /// Connect once; poses, overlay views and enable changes remain live.
    pub fn draw_source(&self) -> SkinDrawSource {
        self.source.clone()
    }

    // ---- palette (draws) ----

    /// Writes the pose of `tag` (16 column-major floats per joint, validated by the
    /// caller) into its palette block, allocating or growing the block and the palette
    /// buffer as needed. `skin` is the mesh's rest skin record buffer (64 B per vertex,
    /// `VERTEX` usage) its posed draws bind. Returns the block base in matrices. The GPU
    /// copy is updated by the next [`flush_palette`](Self::flush_palette).
    pub fn set_pose(
        &mut self,
        gpu: &GpuContext,
        tag: u32,
        skin: &wgpu::Buffer,
        matrices: &[f32],
    ) -> u32 {
        let joints = (matrices.len() / 16) as u32;
        let needed = joints.div_ceil(PALETTE_BLOCK).max(1) * PALETTE_BLOCK;
        let mut guard = self.source.0.borrow_mut();
        let s = &mut *guard;
        let (base, capacity) = match s.posed.get(&tag).map(|p| (p.base, p.capacity)) {
            Some((base, cap)) if cap >= needed => (base, cap),
            Some((base, cap)) => {
                s.palette.release(base, cap);
                (s.palette.alloc(needed), needed)
            }
            None => (s.palette.alloc(needed), needed),
        };
        let start = base as usize * 16;
        s.palette.cpu[start..start + matrices.len()].copy_from_slice(matrices);
        s.palette.dirty = true;
        s.palette.ensure_buffer(gpu);
        let posed = Posed {
            skin: skin.clone(),
            skin_key: buffer_key(skin),
            base,
            capacity,
            joints: matrices.to_vec(),
        };
        s.posed.insert(tag, posed);
        base
    }

    /// Frees the palette block of `tag`; it draws rest again.
    pub fn clear_pose(&mut self, tag: u32) {
        let mut s = self.source.0.borrow_mut();
        if let Some(p) = s.posed.remove(&tag) {
            s.palette.release(p.base, p.capacity);
        }
    }

    /// Whether `tag` has a pose (ignores the enable state).
    pub fn has_pose(&self, tag: u32) -> bool {
        self.source.0.borrow().posed.contains_key(&tag)
    }

    /// Tags with a pose, ascending.
    pub fn posed_tags(&self) -> Vec<u32> {
        self.source.0.borrow().posed.keys().copied().collect()
    }

    /// `(base, capacity)` of the palette block of `tag`, in matrices.
    pub fn palette_block(&self, tag: u32) -> Option<(u32, u32)> {
        self.source
            .0
            .borrow()
            .posed
            .get(&tag)
            .map(|p| (p.base, p.capacity))
    }

    /// The palette buffer (absent until the first pose).
    pub fn palette_buffer(&self) -> Option<wgpu::Buffer> {
        self.source.0.borrow().palette.buffer.clone()
    }

    /// Matrices of the palette (blocks in use, holes included).
    pub fn palette_matrices(&self) -> u32 {
        self.source.0.borrow().palette.end()
    }

    /// The CPU copy of the palette (16 floats per matrix).
    pub fn palette_cpu(&self) -> Vec<f32> {
        self.source.0.borrow().palette.cpu.clone()
    }

    /// Uploads the palette with ONE `write_buffer` when a pose changed since the last
    /// flush. Returns the number of writes issued (0 or 1).
    pub fn flush_palette(&self, gpu: &GpuContext) -> u32 {
        let mut s = self.source.0.borrow_mut();
        let p = &mut s.palette;
        if !p.dirty || p.cpu.is_empty() {
            return 0;
        }
        let Some(buffer) = p.buffer.clone() else {
            return 0;
        };
        p.bytes.clear();
        p.bytes.reserve(p.cpu.len() * 4);
        for x in &p.cpu {
            p.bytes.extend_from_slice(&x.to_le_bytes());
        }
        gpu.queue().write_buffer(&buffer, 0, &p.bytes);
        p.dirty = false;
        1
    }

    // ---- overlay views (editor) ----

    /// Whether the editor wants overlay geometry for `tag` (outline or wireframe on).
    pub fn overlay_wanted(&self, tag: u32) -> bool {
        self.source.overlay_wanted(tag)
    }

    /// Creates (or replaces) the overlay view of `instance`, sized from `rest`. Each view
    /// gets its own `VERTEX | STORAGE | COPY_DST | COPY_SRC` buffers; the rest mesh is never
    /// written. Refusals: `render.invalid_mesh` from [`RestMeshView::new`].
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
        self.source
            .0
            .borrow_mut()
            .overlays
            .insert(instance, (stream0.clone(), edges.clone()));
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

    /// Replaces the culling pose of `instance` (the palette keeps the drawn pose).
    pub fn set_joint_matrices(&mut self, instance: u32, matrices: &[f32]) {
        if let Some(p) = self.source.0.borrow_mut().posed.get_mut(&instance) {
            p.joints.clear();
            p.joints.extend_from_slice(matrices);
        }
    }

    /// Drops the overlay view of `instance` and its buffers; other views are untouched.
    pub fn remove_skinned_view(&mut self, instance: u32) -> Option<SkinnedView> {
        self.source.0.borrow_mut().overlays.remove(&instance);
        self.views.remove(&instance)
    }

    /// A disabled `skin` pass draws the rest pose (S18); poses and views are kept.
    pub fn set_skinning_enabled(&mut self, enabled: bool) {
        self.disabled = !enabled;
        self.source.0.borrow_mut().disabled = !enabled;
    }

    /// Whether skinning is enabled.
    pub fn skinning_enabled(&self) -> bool {
        !self.disabled
    }

    /// Number of live overlay views (ignores the enable state).
    pub fn overlay_views(&self) -> usize {
        self.views.len()
    }

    /// The overlay view of `instance`: `None` when absent or while disabled.
    pub fn skinned_view(&self, instance: u32) -> Option<&SkinnedView> {
        (!self.disabled)
            .then(|| self.views.get(&instance))
            .flatten()
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

    /// Stream 0 an overlay of `instance` binds: posed when it has a view, else `rest`.
    pub fn stream0_for<'a>(&'a self, instance: u32, rest: &'a wgpu::Buffer) -> &'a wgpu::Buffer {
        self.skinned_stream0(instance).unwrap_or(rest)
    }

    /// Edge segments an overlay of `instance` binds: posed when it has a view, else rest.
    pub fn edges_for<'a>(
        &'a self,
        instance: u32,
        rest: Option<&'a wgpu::Buffer>,
    ) -> Option<&'a wgpu::Buffer> {
        self.skinned_edge_segments(instance).or(rest)
    }
}

impl Drop for SkinViews {
    fn drop(&mut self) {
        let mut s = self.source.0.borrow_mut();
        s.overlays.clear();
        s.posed.clear();
        s.palette = Palette::default();
    }
}

/// Identity of a buffer (equal handles hash equal).
fn buffer_key(buffer: &wgpu::Buffer) -> u64 {
    let mut h = DefaultHasher::new();
    buffer.hash(&mut h);
    h.finish()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn palette_blocks_are_aligned_reused_and_trimmed() {
        let mut p = Palette::default();
        let a = p.alloc(4);
        let b = p.alloc(8);
        let c = p.alloc(4);
        assert_eq!((a, b, c, p.end()), (0, 4, 12, 16));
        p.release(b, 8);
        assert_eq!(p.alloc(4), 4, "a freed block is reused first-fit");
        assert_eq!(p.alloc(4), 8);
        p.release(c, 4);
        assert_eq!(p.end(), 12, "a free block at the end shrinks the palette");
        p.release(4, 4);
        p.release(8, 4);
        assert_eq!(p.end(), 4);
        p.release(a, 4);
        assert_eq!((p.end(), p.free.len()), (0, 0));
    }

    #[test]
    fn overlay_requests_follow_editor_rules() {
        let mut r = OverlayRequests::default();
        assert!(!r.wanted(1));
        r.outlines.insert(1);
        assert!(r.wanted(1) && !r.wanted(2));
        r.global = true;
        r.wire.insert(3, false);
        assert!(r.wanted(2) && !r.wanted(3));
    }
}
