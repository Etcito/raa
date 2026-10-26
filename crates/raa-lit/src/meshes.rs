//! The GPU mesh store of the lit renderer: S10 streams 0/1 and the u32 index buffer per
//! [`LitMeshId`], plus one internal [`MeshId`] per sub-mesh (what draws and the scene use).
//!
//! Ids of both kinds are dense from 0 and reused lowest first after a removal.

use std::collections::HashMap;

use raa_core::{GpuContext, MaterialId, MeshId, RaaError};
use raa_geom::MeshData;
use raa_math::Aabb;
use raa_mesh::{AttributedMesh, SubMesh};

/// Vertex buffer slot of S10 stream 0 (position, normal), as in the v0 renderer.
pub(crate) const STREAM0_SLOT: u32 = 0;
/// Vertex buffer slot of the S1 instance stream, as in the v0 renderer.
#[allow(dead_code, reason = "used by the lit pipelines and draws")]
pub(crate) const INSTANCE_SLOT: u32 = 1;
/// Vertex buffer slot of S10 stream 1 (uv, tangent, colour).
pub(crate) const STREAM1_SLOT: u32 = 2;

/// Handle of a mesh uploaded to the lit renderer (all its sub-meshes).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LitMeshId(pub u32);

/// What an uploaded lit mesh holds. `submeshes` is the effective list (one sub-mesh over
/// all indices when the mesh declared none).
#[derive(Clone, Debug, PartialEq)]
pub struct LitMeshInfo {
    pub vertices: u32,
    pub triangles: u32,
    pub submeshes: Vec<SubMesh>,
    pub bounds: Aabb,
}

/// Dense u32 ids from 0; a freed id is handed out again before any new one, lowest first.
#[derive(Clone, Debug, Default)]
pub(crate) struct IdAllocator {
    next: u32,
    /// Freed ids, sorted descending so `pop` yields the lowest.
    free: Vec<u32>,
}

impl IdAllocator {
    /// The lowest free id, else the next fresh one. `None` when u32 ids are exhausted.
    pub(crate) fn alloc(&mut self) -> Option<u32> {
        if let Some(id) = self.free.pop() {
            return Some(id);
        }
        let id = self.next;
        self.next = id.checked_add(1)?;
        Some(id)
    }

    /// Returns `id` to the pool. Ids never allocated or already free are ignored.
    pub(crate) fn release(&mut self, id: u32) {
        if id >= self.next {
            return;
        }
        // Descending order: find the first position whose id is smaller.
        match self.free.binary_search_by(|probe| id.cmp(probe)) {
            Ok(_) => {}
            Err(pos) => self.free.insert(pos, id),
        }
    }
}

/// Where a sub-mesh id draws from.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct SubMeshRecord {
    pub(crate) lit: LitMeshId,
    pub(crate) first_index: u32,
    pub(crate) index_count: u32,
    pub(crate) material: Option<MaterialId>,
}

/// The CPU side of the store: both id allocators, the sub-mesh records and the
/// LitMeshId -> sub-mesh ids map. Needs no GPU, so it is unit-tested alone.
#[derive(Clone, Debug, Default)]
pub(crate) struct SubMeshTable {
    lit_ids: IdAllocator,
    mesh_ids: IdAllocator,
    /// Indexed by `MeshId.0`; `None` for a free id.
    records: Vec<Option<SubMeshRecord>>,
    /// Sub-mesh ids of each live lit mesh, in sub-mesh order.
    by_lit: HashMap<LitMeshId, Vec<MeshId>>,
}

#[allow(dead_code, reason = "used by LitRenderer and LitHandle")]
impl SubMeshTable {
    /// Allocates a lit mesh id and one sub-mesh id per entry of `subs`.
    pub(crate) fn insert(&mut self, subs: &[SubMesh]) -> Result<LitMeshId, RaaError> {
        let lit = LitMeshId(self.lit_ids.alloc().ok_or_else(too_many_meshes)?);
        let mut ids = Vec::with_capacity(subs.len());
        if let Err(e) = self.grow(lit, &mut ids, subs) {
            for id in ids {
                self.free_mesh_id(id);
            }
            self.lit_ids.release(lit.0);
            return Err(e);
        }
        self.by_lit.insert(lit, ids);
        Ok(lit)
    }

    /// Replaces the sub-meshes of `lit`: the first `min(old, new)` ids are kept (records
    /// rewritten), surplus ids are freed and missing ones allocated lowest first.
    pub(crate) fn replace(&mut self, lit: LitMeshId, subs: &[SubMesh]) -> Result<(), RaaError> {
        let mut ids = self.by_lit.remove(&lit).ok_or_else(|| unknown_lit(lit))?;
        while ids.len() > subs.len() {
            if let Some(id) = ids.pop() {
                self.free_mesh_id(id);
            }
        }
        for (id, sub) in ids.iter().zip(subs) {
            if let Some(slot) = self.records.get_mut(id.0 as usize) {
                *slot = Some(record(lit, sub));
            }
        }
        let kept = ids.len();
        let result = self.grow(lit, &mut ids, subs.get(kept..).unwrap_or(&[]));
        self.by_lit.insert(lit, ids);
        result
    }

    /// Frees `lit` and all its sub-mesh ids. Unknown ids are ignored.
    pub(crate) fn remove(&mut self, lit: LitMeshId) {
        if let Some(ids) = self.by_lit.remove(&lit) {
            for id in ids {
                self.free_mesh_id(id);
            }
            self.lit_ids.release(lit.0);
        }
    }

    pub(crate) fn contains(&self, lit: LitMeshId) -> bool {
        self.by_lit.contains_key(&lit)
    }

    pub(crate) fn submesh_ids(&self, lit: LitMeshId) -> Option<&[MeshId]> {
        self.by_lit.get(&lit).map(Vec::as_slice)
    }

    pub(crate) fn record(&self, mesh: MeshId) -> Option<&SubMeshRecord> {
        self.records.get(mesh.0 as usize).and_then(Option::as_ref)
    }

    /// Allocates one id per `subs` entry, appending to `ids` and writing the records.
    fn grow(
        &mut self,
        lit: LitMeshId,
        ids: &mut Vec<MeshId>,
        subs: &[SubMesh],
    ) -> Result<(), RaaError> {
        for sub in subs {
            let id = self.mesh_ids.alloc().ok_or_else(too_many_meshes)?;
            let index = id as usize;
            if index >= self.records.len() {
                self.records.resize(index + 1, None);
            }
            if let Some(slot) = self.records.get_mut(index) {
                *slot = Some(record(lit, sub));
            }
            ids.push(MeshId(id));
        }
        Ok(())
    }

    fn free_mesh_id(&mut self, id: MeshId) {
        if let Some(slot) = self.records.get_mut(id.0 as usize) {
            *slot = None;
        }
        self.mesh_ids.release(id.0);
    }
}

fn record(lit: LitMeshId, sub: &SubMesh) -> SubMeshRecord {
    SubMeshRecord {
        lit,
        first_index: sub.first_index,
        index_count: sub.index_count,
        material: sub.material,
    }
}

/// GPU buffers of one lit mesh (shared by all its sub-mesh ids).
#[derive(Debug)]
pub(crate) struct GpuLitMesh {
    /// S10 stream 0 (S1 24 B: position, normal).
    pub(crate) stream0: wgpu::Buffer,
    /// S10 stream 1 (40 B: uv, tangent, colour).
    pub(crate) stream1: wgpu::Buffer,
    /// u32 triangle-list indices.
    pub(crate) index: wgpu::Buffer,
    info: LitMeshInfo,
    pub(crate) edge_bytes: Vec<u8>,
    pub(crate) edges: Option<wgpu::Buffer>,
}

/// The lit renderer's meshes, behind `LitRenderer::{upload, update, remove, submesh_ids,
/// info}` and `LitHandle::{bind_mesh, submesh_range}`.
#[derive(Debug, Default)]
pub(crate) struct MeshStore {
    table: SubMeshTable,
    /// Indexed by `LitMeshId.0`; `None` for a free id.
    meshes: Vec<Option<GpuLitMesh>>,
}

#[allow(dead_code, reason = "used by LitRenderer and LitHandle")]
impl MeshStore {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Validates `m` (`render.invalid_mesh` carrying the `mesh.*` code), generates missing
    /// tangents, writes streams 0/1 and the index buffer and assigns one [`MeshId`] per
    /// sub-mesh.
    pub(crate) fn upload(
        &mut self,
        gpu: &GpuContext,
        m: &AttributedMesh,
    ) -> Result<LitMeshId, RaaError> {
        let (gpu_mesh, subs) = build(gpu, m)?;
        let lit = self.table.insert(&subs)?;
        let index = lit.0 as usize;
        if index >= self.meshes.len() {
            self.meshes.resize_with(index + 1, || None);
        }
        if let Some(slot) = self.meshes.get_mut(index) {
            *slot = Some(gpu_mesh);
        }
        Ok(lit)
    }

    /// Uploads a v0 [`MeshData`] as one sub-mesh with the S10 attribute defaults.
    pub(crate) fn upload_v0(
        &mut self,
        gpu: &GpuContext,
        m: &MeshData,
    ) -> Result<LitMeshId, RaaError> {
        self.upload(gpu, &AttributedMesh::from_mesh(m.clone()))
    }

    /// Replaces the buffers and sub-meshes of `id`, keeping its sub-mesh ids where the
    /// count allows. Unknown ids give `render.unknown_mesh`; a refused mesh changes nothing.
    pub(crate) fn update(
        &mut self,
        gpu: &GpuContext,
        id: LitMeshId,
        m: &AttributedMesh,
    ) -> Result<(), RaaError> {
        if !self.table.contains(id) {
            return Err(unknown_lit(id));
        }
        let (gpu_mesh, subs) = build(gpu, m)?;
        self.table.replace(id, &subs)?;
        if let Some(slot) = self.meshes.get_mut(id.0 as usize) {
            *slot = Some(gpu_mesh);
        }
        Ok(())
    }

    /// Drops the buffers of `id` and frees it with all its sub-mesh ids. Unknown ids are
    /// ignored.
    pub(crate) fn remove(&mut self, id: LitMeshId) {
        self.table.remove(id);
        if let Some(slot) = self.meshes.get_mut(id.0 as usize) {
            *slot = None;
        }
    }

    pub(crate) fn submesh_ids(&self, id: LitMeshId) -> Option<&[MeshId]> {
        self.table.submesh_ids(id)
    }

    pub(crate) fn info(&self, id: LitMeshId) -> Option<LitMeshInfo> {
        self.lit_mesh(id).map(|m| m.info.clone())
    }

    /// `(first_index, index_count)` of sub-mesh `mesh`, `None` when unknown.
    pub(crate) fn submesh_range(&self, mesh: MeshId) -> Option<(u32, u32)> {
        self.table
            .record(mesh)
            .map(|r| (r.first_index, r.index_count))
    }

    /// The sub-mesh record of `mesh` (lit mesh, range, material).
    pub(crate) fn submesh(&self, mesh: MeshId) -> Option<&SubMeshRecord> {
        self.table.record(mesh)
    }

    /// The buffers holding sub-mesh `mesh` and its `(first_index, index_count)`;
    /// unknown ids give `render.unknown_mesh`.
    pub(crate) fn resolve(&self, mesh: MeshId) -> Result<(&GpuLitMesh, u32, u32), RaaError> {
        self.table
            .record(mesh)
            .and_then(|r| {
                self.lit_mesh(r.lit)
                    .map(|g| (g, r.first_index, r.index_count))
            })
            .ok_or_else(|| RaaError::FrameGraph {
                code: "render.unknown_mesh",
                message: format!("mesh {} is not uploaded", mesh.0),
            })
    }

    pub(crate) fn set_edges(&mut self, id: LitMeshId, bytes: Vec<u8>) {
        if let Some(Some(m)) = self.meshes.get_mut(id.0 as usize) {
            m.edge_bytes = bytes;
            m.edges = None;
        }
    }

    pub(crate) fn prepare_edges(&mut self, gpu: &GpuContext) {
        for mesh in self.meshes.iter_mut().flatten() {
            if mesh.edges.is_none() && !mesh.edge_bytes.is_empty() {
                mesh.edges = Some(buffer_with(
                    gpu,
                    "raa.editor.edges",
                    wgpu::BufferUsages::VERTEX,
                    &mesh.edge_bytes,
                ));
            }
        }
    }

    /// Sets streams 0 and 1 ([`STREAM0_SLOT`], [`STREAM1_SLOT`]) and the index buffer of sub-mesh `mesh`; returns its `(first_index, index_count)`.
    pub(crate) fn bind(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        mesh: MeshId,
    ) -> Result<(u32, u32), RaaError> {
        let (g, first, count) = self.resolve(mesh)?;
        pass.set_vertex_buffer(STREAM0_SLOT, g.stream0.slice(..));
        pass.set_vertex_buffer(STREAM1_SLOT, g.stream1.slice(..));
        pass.set_index_buffer(g.index.slice(..), wgpu::IndexFormat::Uint32);
        Ok((first, count))
    }

    pub(crate) fn lit_mesh(&self, id: LitMeshId) -> Option<&GpuLitMesh> {
        self.meshes.get(id.0 as usize).and_then(Option::as_ref)
    }
}

/// Validates `m`, fills its tangents and creates its three buffers. Returns them with the
/// effective sub-mesh list.
fn build(gpu: &GpuContext, m: &AttributedMesh) -> Result<(GpuLitMesh, Vec<SubMesh>), RaaError> {
    let (vertices, index_total) = check(m)?;
    let generated;
    let mesh = if m.uvs.is_some() && m.tangents.is_none() {
        let mut copy = m.clone();
        copy.ensure_tangents();
        generated = copy;
        &generated
    } else {
        m
    };
    let subs = if mesh.submeshes.is_empty() {
        vec![SubMesh {
            first_index: 0,
            index_count: index_total,
            material: None,
        }]
    } else {
        mesh.submeshes.clone()
    };

    let mut bytes = Vec::new();
    mesh.write_stream0_bytes(&mut bytes);
    let stream0 = buffer_with(gpu, "raa.lit.stream0", wgpu::BufferUsages::VERTEX, &bytes);
    bytes.clear();
    mesh.write_stream1_bytes(&mut bytes);
    let stream1 = buffer_with(gpu, "raa.lit.stream1", wgpu::BufferUsages::VERTEX, &bytes);
    bytes.clear();
    mesh.write_index_bytes(&mut bytes);
    let index = buffer_with(gpu, "raa.lit.indices", wgpu::BufferUsages::INDEX, &bytes);

    let info = LitMeshInfo {
        vertices,
        triangles: index_total / 3,
        submeshes: subs.clone(),
        bounds: mesh.bounds(),
    };
    let gpu_mesh = GpuLitMesh {
        stream0,
        stream1,
        index,
        info,
        edge_bytes: triangle_edge_bytes(m),
        edges: None,
    };
    Ok((gpu_mesh, subs))
}

/// Validates `m` for upload; returns its vertex and index counts. Refusals are
/// `render.invalid_mesh` with the message `mesh.<code>: <reason>`.
fn check(m: &AttributedMesh) -> Result<(u32, u32), RaaError> {
    m.validate().map_err(|e| invalid_mesh(e.to_string()))?;
    let vertices = u32::try_from(m.vertex_count())
        .map_err(|_| invalid_mesh(format!("{} vertices exceed u32", m.vertex_count())))?;
    let index_total = u32::try_from(m.base.indices.len())
        .map_err(|_| invalid_mesh(format!("{} indices exceed u32", m.base.indices.len())))?;
    Ok((vertices, index_total))
}

/// Creates a buffer of `bytes.len()` with `usage | COPY_DST` and writes `bytes` into it.
fn buffer_with(
    gpu: &GpuContext,
    label: &'static str,
    usage: wgpu::BufferUsages,
    bytes: &[u8],
) -> wgpu::Buffer {
    let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !bytes.is_empty() {
        gpu.queue().write_buffer(&buffer, 0, bytes);
    }
    buffer
}

fn invalid_mesh(message: String) -> RaaError {
    RaaError::FrameGraph {
        code: "render.invalid_mesh",
        message,
    }
}

fn unknown_lit(id: LitMeshId) -> RaaError {
    RaaError::FrameGraph {
        code: "render.unknown_mesh",
        message: format!("lit mesh {} is not uploaded", id.0),
    }
}

fn too_many_meshes() -> RaaError {
    RaaError::FrameGraph {
        code: "render.too_many_meshes",
        message: "mesh ids are exhausted (u32)".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subs(n: u32) -> Vec<SubMesh> {
        (0..n)
            .map(|i| SubMesh {
                first_index: i * 3,
                index_count: 3,
                material: Some(MaterialId(i)),
            })
            .collect()
    }

    fn ids(t: &SubMeshTable, lit: LitMeshId) -> Vec<u32> {
        t.submesh_ids(lit)
            .expect("lit mesh is live")
            .iter()
            .map(|m| m.0)
            .collect()
    }

    #[test]
    fn allocator_is_dense_and_reuses_lowest_first() {
        let mut a = IdAllocator::default();
        let got: Vec<u32> = (0..5).filter_map(|_| a.alloc()).collect();
        assert_eq!(got, [0, 1, 2, 3, 4]);
        a.release(3);
        a.release(1);
        a.release(3);
        a.release(99);
        assert_eq!(a.alloc(), Some(1));
        assert_eq!(a.alloc(), Some(3));
        assert_eq!(a.alloc(), Some(5));
    }

    #[test]
    fn submesh_ids_are_dense_across_meshes() {
        let mut t = SubMeshTable::default();
        let a = t.insert(&subs(2)).expect("insert");
        let b = t.insert(&subs(3)).expect("insert");
        assert_eq!((a, b), (LitMeshId(0), LitMeshId(1)));
        assert_eq!(ids(&t, a), [0, 1]);
        assert_eq!(ids(&t, b), [2, 3, 4]);
        let r = t.record(MeshId(3)).expect("record");
        assert_eq!(r.lit, b);
        assert_eq!((r.first_index, r.index_count), (3, 3));
        assert_eq!(r.material, Some(MaterialId(1)));
    }

    #[test]
    fn removed_ids_are_reused_lowest_first() {
        let mut t = SubMeshTable::default();
        let a = t.insert(&subs(2)).expect("insert");
        let b = t.insert(&subs(1)).expect("insert");
        let c = t.insert(&subs(2)).expect("insert");
        assert_eq!(ids(&t, c), [3, 4]);
        t.remove(a);
        t.remove(b);
        assert!(t.submesh_ids(a).is_none());
        assert!(t.record(MeshId(0)).is_none());
        let d = t.insert(&subs(4)).expect("insert");
        assert_eq!(d, LitMeshId(0));
        assert_eq!(ids(&t, d), [0, 1, 2, 5]);
        let e = t.insert(&subs(1)).expect("insert");
        assert_eq!(e, LitMeshId(1));
        assert_eq!(ids(&t, e), [6]);
    }

    #[test]
    fn update_keeps_ids_when_submesh_count_is_unchanged() {
        let mut t = SubMeshTable::default();
        let a = t.insert(&subs(3)).expect("insert");
        let mut new = subs(3);
        new[1].material = None;
        new[1].first_index = 6;
        t.replace(a, &new).expect("replace");
        assert_eq!(ids(&t, a), [0, 1, 2]);
        let r = t.record(MeshId(1)).expect("record");
        assert_eq!((r.first_index, r.material), (6, None));
    }

    #[test]
    fn update_shrinks_and_grows_keeping_leading_ids() {
        let mut t = SubMeshTable::default();
        let a = t.insert(&subs(3)).expect("insert");
        let b = t.insert(&subs(1)).expect("insert");
        t.replace(a, &subs(1)).expect("shrink");
        assert_eq!(ids(&t, a), [0]);
        assert!(t.record(MeshId(2)).is_none());
        t.replace(a, &subs(4)).expect("grow");
        assert_eq!(ids(&t, a), [0, 1, 2, 4]);
        assert_eq!(ids(&t, b), [3]);
    }

    #[test]
    fn replace_of_unknown_mesh_is_refused() {
        let mut t = SubMeshTable::default();
        let err = t.replace(LitMeshId(7), &subs(1)).expect_err("unknown");
        assert_eq!(err.code(), "render.unknown_mesh");
        t.remove(LitMeshId(7));
        assert_eq!(t.insert(&subs(1)).expect("insert"), LitMeshId(0));
    }

    #[test]
    fn invalid_mesh_is_refused_with_the_mesh_code() {
        let err = check(&AttributedMesh::default()).expect_err("empty mesh");
        assert_eq!(err.code(), "render.invalid_mesh");
        assert!(err.message().starts_with("mesh.empty"), "{err}");
    }
}

/// Each edge is one instanced segment, expanded to a pixel-width quad in the shader.
pub(crate) fn edge_bytes(m: &AttributedMesh, edges: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(edges.len() * 12);
    for &i in edges {
        if let Some(p) = m.base.positions.get(i as usize) {
            for x in *p {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
    bytes
}

fn triangle_edge_bytes(m: &AttributedMesh) -> Vec<u8> {
    let mut edges = std::collections::BTreeSet::new();
    for t in m.base.indices.chunks_exact(3) {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            edges.insert((a.min(b), a.max(b)));
        }
    }
    edge_bytes(
        m,
        &edges
            .into_iter()
            .flat_map(|(a, b)| [a, b])
            .collect::<Vec<_>>(),
    )
}

#[cfg(test)]
mod edge_tests {
    use super::*;
    #[test]
    fn polygon_quad_omits_triangle_diagonal() {
        let mut m = AttributedMesh::default();
        m.base.positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        m.base.indices = vec![0, 1, 2, 0, 2, 3];
        assert_eq!(triangle_edge_bytes(&m).len(), 5 * 24);
        assert_eq!(edge_bytes(&m, &[0, 1, 1, 2, 2, 3, 3, 0]).len(), 4 * 24);
        assert!(edge_bytes(&m, &[]).is_empty());
    }
}
