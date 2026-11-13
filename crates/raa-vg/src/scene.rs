//! The S32 scene pool of RAA's virtualized geometry: the 192-byte `VgInstanceGpu` record and its
//! little-endian writer, the section layout of the uploaded pool, the mesh-key registry and the
//! per-instance requests. The pool buffers hold the S29 section records unchanged; they are
//! rebuilt only when the cluster store or the mesh keys change, and they grow in pages.
//! Adapted from solstice (MIT), Copyright (c) 2026 SamG-Coder: `crates/solstice-render/src/vg`
//! scene setup, revision 7e465b5.

use std::collections::BTreeMap;

use raa_cluster::ClusterStore;
use raa_core::{GpuContext, MeshId, wgpu};
use raa_math::Mat4;

/// Bytes of one `VgInstanceGpu` record.
pub const VG_INSTANCE_SIZE: usize = 192;
/// `ids.z` bit: the previous model rows are valid.
pub const INSTANCE_PREVIOUS_VALID: u32 = 1;
/// `ids.z` bit: the instance is double-sided.
pub const INSTANCE_DOUBLE_SIDED: u32 = 2;
/// `ids.z` bit: the instance casts shadows.
pub const INSTANCE_CASTS_SHADOW: u32 = 4;
/// `ids.z` bit: the instance may move.
pub const INSTANCE_MOVABLE: u32 = 8;

const NODE_SIZE: u64 = 48;
const CLUSTER_SIZE: u64 = 48;
const VERTEX_SIZE: u64 = 16;
const TRIANGLE_SIZE: u64 = 4;
const INSTANCE_PAGE: u32 = 64;
const NODE_PAGE: u32 = 64;
const CLUSTER_PAGE: u32 = 64;
const VERTEX_PAGE: u32 = 256;
const TRIANGLE_PAGE: u32 = 256;

/// One virtualized scene instance in the revised S32 layout.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct VgInstanceRecord {
    pub model: Mat4,
    pub previous_rows: [[f32; 4]; 3],
    pub quant: [f32; 4],
    pub uv_quant: [f32; 4],
    pub ranges: [u32; 4],
    pub ids: [u32; 4],
    pub bounds: [f32; 4],
}

/// Writes `record` little-endian at the revised S32 offsets: model @0, previous rows @64/80/96,
/// quant @112, uv_quant @128, ranges @144, ids @160 and bounds @176.
pub fn write_instance(out: &mut [u8; VG_INSTANCE_SIZE], record: &VgInstanceRecord) {
    let mut at = 0;
    for column in &record.model.cols {
        for value in column {
            put_f32(out, at, *value);
            at += 4;
        }
    }
    for row in &record.previous_rows {
        for value in row {
            put_f32(out, at, *value);
            at += 4;
        }
    }
    for value in record.quant {
        put_f32(out, at, value);
        at += 4;
    }
    for value in record.uv_quant {
        put_f32(out, at, value);
        at += 4;
    }
    for value in record.ranges {
        put_u32(out, at, value);
        at += 4;
    }
    for value in record.ids {
        put_u32(out, at, value);
        at += 4;
    }
    for value in record.bounds {
        put_f32(out, at, value);
        at += 4;
    }
}

/// The hierarchy word of `ids.w`: depth in bits 0-7, the section-relative root node in 8-31.
pub fn hierarchy_word(depth: u32, root_relative: u32) -> u32 {
    (depth & 0xff) | ((root_relative & 0x00ff_ffff) << 8)
}

/// The absolute base of one section in the four pool buffers.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SectionRange {
    pub node_base: u32,
    pub cluster_base: u32,
    pub vertex_base: u32,
    pub triangle_base: u32,
}

#[derive(Clone, Copy, Debug)]
struct SectionInfo {
    range: SectionRange,
    hierarchy_depth: u32,
    root_node: u32,
}

/// The lit sub-mesh ids of every cluster-store key: sub-mesh `i` of key `key` is `submeshes[i]`.
#[derive(Clone, Debug, Default)]
pub struct MeshKeys {
    lists: BTreeMap<u32, Vec<MeshId>>,
    by_mesh: BTreeMap<MeshId, (u32, usize)>,
    generation: u64,
}

impl MeshKeys {
    /// Replaces the key's list; the generation advances only when the list changes.
    pub fn set(&mut self, key: u32, submeshes: &[MeshId]) {
        if self.lists.get(&key).is_some_and(|old| old == submeshes) {
            return;
        }
        self.lists.insert(key, submeshes.to_vec());
        self.reindex();
        self.generation += 1;
    }

    /// Removes the key's list, if any.
    pub fn remove(&mut self, key: u32) {
        if self.lists.remove(&key).is_some() {
            self.reindex();
            self.generation += 1;
        }
    }

    /// The store key and sub-mesh ordinal of a lit mesh id.
    pub fn get(&self, mesh: MeshId) -> Option<(u32, usize)> {
        self.by_mesh.get(&mesh).copied()
    }

    /// The registered keys, ascending.
    pub fn keys(&self) -> Vec<u32> {
        self.lists.keys().copied().collect()
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    fn reindex(&mut self) {
        self.by_mesh.clear();
        for (key, meshes) in &self.lists {
            for (ordinal, mesh) in meshes.iter().enumerate() {
                self.by_mesh.insert(*mesh, (*key, ordinal));
            }
        }
    }
}

/// Explicit virtualized-path requests per JS instance tag; `None` removes the tag.
#[derive(Clone, Debug, Default)]
pub struct InstanceRequests {
    requests: BTreeMap<u32, bool>,
}

impl InstanceRequests {
    pub fn set(&mut self, tag: u32, request: Option<bool>) {
        match request {
            Some(value) => {
                self.requests.insert(tag, value);
            }
            None => {
                self.requests.remove(&tag);
            }
        }
    }

    pub fn get(&self, tag: u32) -> Option<bool> {
        self.requests.get(&tag).copied()
    }
}

#[derive(Default)]
struct Pool {
    buffer: Option<wgpu::Buffer>,
    capacity: u32,
}

impl Pool {
    fn upload(
        &mut self,
        gpu: &GpuContext,
        label: &str,
        record_size: u64,
        page: u32,
        bytes: &[u8],
        generation: &mut u64,
    ) {
        let records = u32::try_from(bytes.len() as u64 / record_size).unwrap_or(u32::MAX);
        if self.buffer.is_none() || records > self.capacity {
            let capacity = records.div_ceil(page).max(1) * page;
            self.buffer = Some(gpu.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: u64::from(capacity) * record_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.capacity = capacity;
            *generation += 1;
        }
        if !bytes.is_empty()
            && let Some(buffer) = &self.buffer
        {
            gpu.queue().write_buffer(buffer, 0, bytes);
        }
    }

    fn release(&mut self, generation: &mut u64) {
        if self.buffer.take().is_some() {
            self.capacity = 0;
            *generation += 1;
        }
    }

    fn access(&self) -> Option<&wgpu::Buffer> {
        self.buffer.as_ref()
    }

    fn bytes(&self) -> u64 {
        self.buffer.as_ref().map_or(0, wgpu::Buffer::size)
    }
}

/// The five S32 pool buffers and the layout of the last upload.
#[derive(Default)]
pub struct VgScene {
    instances: Pool,
    nodes: Pool,
    clusters: Pool,
    vertices: Pool,
    triangles: Pool,
    sections: BTreeMap<(u32, usize), SectionInfo>,
    scratch: [Vec<u8>; 4],
    instance_scratch: Vec<u8>,
    last_instance_bytes: Vec<u8>,
    instance_count: u32,
    max_hierarchy_depth: u32,
    generation: u64,
    store_generation: Option<u64>,
    keys_generation: Option<u64>,
}

impl VgScene {
    /// An empty scene: no buffer exists until the first upload.
    pub fn new() -> VgScene {
        VgScene::default()
    }

    /// Re-packs every Ready section of `keys.keys()` in ascending `(key, ordinal)` order when the
    /// store or the keys changed since the last upload; returns true when it rebuilt.
    pub fn upload_pool(&mut self, gpu: &GpuContext, store: &ClusterStore, keys: &MeshKeys) -> bool {
        let store_generation = store.generation();
        let keys_generation = keys.generation();
        if self.store_generation == Some(store_generation)
            && self.keys_generation == Some(keys_generation)
        {
            return false;
        }
        self.store_generation = Some(store_generation);
        self.keys_generation = Some(keys_generation);
        for bytes in &mut self.scratch {
            bytes.clear();
        }
        self.sections.clear();
        let mut bases = SectionRange {
            node_base: 0,
            cluster_base: 0,
            vertex_base: 0,
            triangle_base: 0,
        };
        for key in keys.keys() {
            let Some(sections) = store.sections(key) else {
                continue;
            };
            for (ordinal, mesh) in sections.iter().enumerate() {
                let (cluster_bytes, node_bytes) = mesh.to_section_bytes();
                self.sections.insert(
                    (key, ordinal),
                    SectionInfo {
                        range: bases,
                        hierarchy_depth: mesh.hierarchy_depth,
                        root_node: mesh.root_node,
                    },
                );
                self.scratch[0].extend_from_slice(&node_bytes);
                self.scratch[1].extend_from_slice(&cluster_bytes);
                for vertex in &mesh.vertices {
                    vertex.write_le_bytes(&mut self.scratch[2]);
                }
                for triangle in &mesh.triangles {
                    self.scratch[3].extend_from_slice(&triangle.to_le_bytes());
                }
                bases.node_base = bases.node_base.saturating_add(mesh.nodes.len() as u32);
                bases.cluster_base = bases
                    .cluster_base
                    .saturating_add(mesh.clusters.len() as u32);
                bases.vertex_base = bases.vertex_base.saturating_add(mesh.vertices.len() as u32);
                bases.triangle_base = bases
                    .triangle_base
                    .saturating_add(mesh.triangles.len() as u32);
            }
        }
        self.nodes.upload(
            gpu,
            "raa-vg.nodes",
            NODE_SIZE,
            NODE_PAGE,
            &self.scratch[0],
            &mut self.generation,
        );
        self.clusters.upload(
            gpu,
            "raa-vg.clusters",
            CLUSTER_SIZE,
            CLUSTER_PAGE,
            &self.scratch[1],
            &mut self.generation,
        );
        self.vertices.upload(
            gpu,
            "raa-vg.vertices",
            VERTEX_SIZE,
            VERTEX_PAGE,
            &self.scratch[2],
            &mut self.generation,
        );
        self.triangles.upload(
            gpu,
            "raa-vg.triangles",
            TRIANGLE_SIZE,
            TRIANGLE_PAGE,
            &self.scratch[3],
            &mut self.generation,
        );
        self.instances.upload(
            gpu,
            "raa-vg.instances",
            VG_INSTANCE_SIZE as u64,
            INSTANCE_PAGE,
            &[],
            &mut self.generation,
        );
        true
    }

    /// The absolute bases of a section from the last upload.
    pub fn section_range(&self, key: u32, ordinal: usize) -> Option<SectionRange> {
        self.sections.get(&(key, ordinal)).map(|info| info.range)
    }

    /// The section range, its hierarchy depth and its section-relative root node.
    pub fn section_header(&self, key: u32, ordinal: usize) -> Option<(SectionRange, u32, u32)> {
        self.sections
            .get(&(key, ordinal))
            .map(|info| (info.range, info.hierarchy_depth, info.root_node))
    }

    /// Writes the records in order, growing the buffer in pages of 64; equal bytes write nothing.
    pub fn upload_instances(&mut self, gpu: &GpuContext, records: &[VgInstanceRecord]) {
        self.instance_scratch.clear();
        self.instance_scratch
            .reserve(records.len().saturating_mul(VG_INSTANCE_SIZE));
        let mut bytes = [0u8; VG_INSTANCE_SIZE];
        for record in records {
            write_instance(&mut bytes, record);
            self.instance_scratch.extend_from_slice(&bytes);
        }
        if self.instances.buffer.is_some() && self.instance_scratch == self.last_instance_bytes {
            return;
        }
        self.instances.upload(
            gpu,
            "raa-vg.instances",
            VG_INSTANCE_SIZE as u64,
            INSTANCE_PAGE,
            &self.instance_scratch,
            &mut self.generation,
        );
        self.last_instance_bytes.clear();
        self.last_instance_bytes
            .extend_from_slice(&self.instance_scratch);
        self.instance_count = u32::try_from(records.len()).unwrap_or(u32::MAX);
        self.max_hierarchy_depth = records
            .iter()
            .map(|record| record.ids[3] & 0xff)
            .max()
            .unwrap_or(0);
    }

    pub fn instance_count(&self) -> u32 {
        self.instance_count
    }

    /// The largest hierarchy depth among the uploaded records; 0 when none.
    pub fn max_hierarchy_depth(&self) -> u32 {
        self.max_hierarchy_depth
    }

    pub fn instances(&self) -> Option<&wgpu::Buffer> {
        self.instances.access()
    }

    pub fn nodes(&self) -> Option<&wgpu::Buffer> {
        self.nodes.access()
    }

    pub fn clusters(&self) -> Option<&wgpu::Buffer> {
        self.clusters.access()
    }

    pub fn vertices(&self) -> Option<&wgpu::Buffer> {
        self.vertices.access()
    }

    pub fn triangles(&self) -> Option<&wgpu::Buffer> {
        self.triangles.access()
    }

    /// Advances by one whenever a buffer is created, recreated or dropped.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The summed size of the existing buffers.
    pub fn allocated_bytes(&self) -> u64 {
        self.instances.bytes()
            + self.nodes.bytes()
            + self.clusters.bytes()
            + self.vertices.bytes()
            + self.triangles.bytes()
    }

    /// Drops every buffer and the cached layout; the next upload rebuilds.
    pub fn release(&mut self) {
        self.instances.release(&mut self.generation);
        self.nodes.release(&mut self.generation);
        self.clusters.release(&mut self.generation);
        self.vertices.release(&mut self.generation);
        self.triangles.release(&mut self.generation);
        self.sections.clear();
        self.last_instance_bytes.clear();
        self.instance_count = 0;
        self.max_hierarchy_depth = 0;
        self.store_generation = None;
        self.keys_generation = None;
    }
}

fn put_f32(out: &mut [u8; VG_INSTANCE_SIZE], at: usize, value: f32) {
    out[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(out: &mut [u8; VG_INSTANCE_SIZE], at: usize, value: u32) {
    out[at..at + 4].copy_from_slice(&value.to_le_bytes());
}
