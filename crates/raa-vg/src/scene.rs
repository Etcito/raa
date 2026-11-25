//! The S32 scene pool of RAA's virtualized geometry: the 192-byte `VgInstanceGpu` record and its
//! little-endian writer, the section layout of the uploaded pool, the mesh-key registry and the
//! per-instance requests. The pool buffers hold the S29 section records unchanged; they are
//! appended for newly ready sections, and repacked when a resident section changes.
//! Adapted from solstice (MIT), Copyright (c) 2026 SamG-Coder: `crates/solstice-render/src/vg`
//! scene setup, revision 7e465b5.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use raa_cluster::{ClusterMesh, ClusterState, ClusterStore};
use raa_core::{GpuContext, MaterialId, MeshId, wgpu};
use raa_lit::{FLAG_DOUBLE_SIDED, FLAG_NO_SHADOW, InstanceShape, LitHandle, MaterialFnId};
use raa_math::Mat4;
use raa_scene::{InstanceId, RenderList, Scene};

use crate::eligibility::{InstancePath, refusal, refusal_message};
use crate::settings::VgSettings;

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
/// `ids.z` bit: a raa-gi voxel member; the GI cull filter (InstanceFilter::Gi) draws exactly these.
pub const INSTANCE_GI: u32 = 16;

const NODE_SIZE: u64 = 48;
const CLUSTER_SIZE: u64 = 48;
const VERTEX_SIZE: u64 = 16;
const TRIANGLE_SIZE: u64 = 4;
const INSTANCE_PAGE: u32 = 64;
const NODE_PAGE: u32 = 64;
const CLUSTER_PAGE: u32 = 64;
const VERTEX_PAGE: u32 = 256;
const TRIANGLE_PAGE: u32 = 256;
/// Width in texels of the pool vertex-attribute texture; the WGSL `VG_COLOR_WIDTH` of resolve.wgsl.
pub const VG_COLOR_WIDTH: u32 = 4096;
/// Largest row count of the pool vertex-attribute texture (the WebGPU default 2D dimension).
pub const VG_COLOR_MAX_ROWS: u32 = 8192;
/// Texels of the pool vertex-attribute texture per vertex: the colour, then the shading uv.
pub const VG_ATTRIBUTE_TEXELS: u32 = 2;
/// The colour texel of a vertex without a stream: white, rgba 1.0 as f32 bits.
pub const VG_WHITE: [u32; 4] = [0x3F80_0000; 4];
/// The uv texel of a vertex without a shading uv: the flag word (z) is 0, so the resolve keeps
/// the cluster's own uv.
pub const VG_NO_UV: [u32; 4] = [0; 4];

/// Checks linear rgba floats (4 per vertex) and returns them per vertex, unclamped: a vertex
/// colour may carry data a material reads at full precision (a rest position, a normal
/// component), as the forward path's f32 colour stream does. A length that is not a multiple
/// of 4 or a non-finite component is refused with its reason.
pub fn pack_vertex_colors(colors: &[f32]) -> Result<Vec<[f32; 4]>, String> {
    if !colors.len().is_multiple_of(4) {
        return Err(format!(
            "{} floats are not rgba per vertex (a multiple of 4)",
            colors.len()
        ));
    }
    if colors.iter().any(|c| !c.is_finite()) {
        return Err("colours must be finite".to_string());
    }
    Ok(colors
        .chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect())
}

/// As [`pack_vertex_colors`] for uvs (2 floats per vertex).
pub fn pack_vertex_uvs(uvs: &[f32]) -> Result<Vec<[f32; 2]>, String> {
    if !uvs.len().is_multiple_of(2) {
        return Err(format!(
            "{} floats are not uv per vertex (a multiple of 2)",
            uvs.len()
        ));
    }
    if uvs.iter().any(|c| !c.is_finite()) {
        return Err("uvs must be finite".to_string());
    }
    Ok(uvs.chunks_exact(2).map(|c| [c[0], c[1]]).collect())
}

/// The shading attributes of one section (`VgScene::set_cluster_shading`): the colour and uv
/// a virtualized pixel hands its material in place of the vertex colour and the cluster uv, one
/// per `ClusterMesh` vertex. The forward path draws the same mesh with its own colour and uv
/// streams; a material that keeps data in them (raa-sdfm's field shading: the rest point and
/// the rest normal) reads the same values on both paths.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VgShading {
    pub colors: Option<Vec<[f32; 4]>>,
    pub uvs: Option<Vec<[f32; 2]>>,
}

/// The pool texels of the registered streams, [`VG_ATTRIBUTE_TEXELS`] per vertex: the colour
/// (f32 bits; the shading colour, else the vertex colour, else [`VG_WHITE`]) then the uv
/// (u bits, v bits, 1, 0; [`VG_NO_UV`] without a shading uv), each stream written at the
/// vertex base of its section. A stream whose length differs from its section's vertex count,
/// or of a section the pool does not hold, is ignored.
pub fn pool_color_texels(
    vertex_count: u32,
    sections: &BTreeMap<(u32, usize), (u32, u32)>,
    colors: &BTreeMap<(u32, usize), Vec<[f32; 4]>>,
    shading: &BTreeMap<(u32, usize), VgShading>,
) -> Vec<[u32; 4]> {
    let mut out = Vec::with_capacity(vertex_count as usize * 2);
    for _ in 0..vertex_count {
        out.push(VG_WHITE);
        out.push(VG_NO_UV);
    }
    let bits = |c: &[f32; 4]| {
        [
            c[0].to_bits(),
            c[1].to_bits(),
            c[2].to_bits(),
            c[3].to_bits(),
        ]
    };
    let write_colors = |id: &(u32, usize), stream: &[[f32; 4]], out: &mut Vec<[u32; 4]>| {
        let Some(&(base, count)) = sections.get(id) else {
            return;
        };
        if stream.len() != count as usize {
            return;
        }
        for (k, c) in stream.iter().enumerate() {
            if let Some(texel) = out.get_mut(2 * (base as usize + k)) {
                *texel = bits(c);
            }
        }
    };
    for (id, stream) in colors {
        write_colors(id, stream, &mut out);
    }
    for (id, entry) in shading {
        if let Some(stream) = &entry.colors {
            write_colors(id, stream, &mut out);
        }
        let (Some(uvs), Some(&(base, count))) = (&entry.uvs, sections.get(id)) else {
            continue;
        };
        if uvs.len() != count as usize {
            continue;
        }
        for (k, uv) in uvs.iter().enumerate() {
            if let Some(texel) = out.get_mut(2 * (base as usize + k) + 1) {
                *texel = [uv[0].to_bits(), uv[1].to_bits(), 1, 0];
            }
        }
    }
    out
}

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
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
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
    counts: [u32; 4],
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

/// The CPU questions path planning asks of the lit renderer; implemented for [`LitHandle`] and
/// by tests with a table double.
pub trait SyncSource {
    fn instance_eligibility(&self, tag: u32, mesh: MeshId, material: MaterialId) -> InstanceShape;
    fn material_fn_of(&self, material: MaterialId) -> MaterialFnId;
    fn material_flags(&self, material: MaterialId) -> u32;
    fn previous_instance_row(&self, slot: u32) -> Option<[[f32; 4]; 3]>;

    fn gi_member(&self, tag: u32) -> bool {
        let _ = tag;
        false
    }
}

impl SyncSource for LitHandle {
    fn instance_eligibility(&self, tag: u32, mesh: MeshId, material: MaterialId) -> InstanceShape {
        LitHandle::instance_eligibility(self, tag, mesh, material)
    }

    fn material_fn_of(&self, material: MaterialId) -> MaterialFnId {
        LitHandle::material_fn_of(self, material)
    }

    fn material_flags(&self, material: MaterialId) -> u32 {
        self.material_record(material)
            .map_or(0, |record| record.flags)
    }

    fn previous_instance_row(&self, slot: u32) -> Option<[[f32; 4]; 3]> {
        LitHandle::previous_instance_row(self, slot)
    }
}

/// A [`SyncSource`] view that delegates every question to a wrapped source but answers
/// [`SyncSource::gi_member`] from a set of tags.
pub struct WithGiMembers<'a> {
    source: &'a dyn SyncSource,
    gi_members: &'a BTreeSet<u32>,
}

impl<'a> WithGiMembers<'a> {
    /// `WithGiMembers::new` wraps a source so planning marks `INSTANCE_GI` for exactly the tags
    /// in `gi_members`; every other `SyncSource` question delegates to the wrapped source. The
    /// downstream card 'raa-vg: VgHandle::with_gi_frame and VgHandle::set_gi_members' calls it
    /// from `VgHandle::plan` in crates/raa-vg/src/feature.rs.
    pub fn new(source: &'a dyn SyncSource, gi_members: &'a BTreeSet<u32>) -> WithGiMembers<'a> {
        WithGiMembers { source, gi_members }
    }
}

impl SyncSource for WithGiMembers<'_> {
    fn instance_eligibility(&self, tag: u32, mesh: MeshId, material: MaterialId) -> InstanceShape {
        self.source.instance_eligibility(tag, mesh, material)
    }

    fn material_fn_of(&self, material: MaterialId) -> MaterialFnId {
        self.source.material_fn_of(material)
    }

    fn material_flags(&self, material: MaterialId) -> u32 {
        self.source.material_flags(material)
    }

    fn previous_instance_row(&self, slot: u32) -> Option<[[f32; 4]; 3]> {
        self.source.previous_instance_row(slot)
    }

    fn gi_member(&self, tag: u32) -> bool {
        self.gi_members.contains(&tag)
    }
}

/// One listed scene instance: its identity, its JS tag and what to draw.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SyncItem {
    pub instance: InstanceId,
    pub slot: u32,
    pub tag: u32,
    pub mesh: MeshId,
    pub material: MaterialId,
    pub world: Mat4,
}

/// Reused render-list scratch: [`ItemScratch::fill`] allocates nothing once its capacity fits.
#[derive(Default)]
pub struct ItemScratch {
    list: RenderList,
    tags: Vec<u32>,
    slots: Vec<u32>,
    items: Vec<SyncItem>,
}

impl ItemScratch {
    /// Fills the reused list and tags of `scene` and returns one item per visible instance.
    pub fn fill(&mut self, scene: &Scene) -> &[SyncItem] {
        scene.build_render_list_slots(&mut self.list, &mut self.tags, &mut self.slots);
        self.items.clear();
        self.items.reserve(self.list.items.len());
        for (index, item) in self.list.items.iter().enumerate() {
            let Some(slot) = self.slots.get(index).copied() else {
                continue;
            };
            let Some(instance) = scene.instance_at_slot(slot) else {
                continue;
            };
            self.items.push(SyncItem {
                instance,
                slot,
                tag: self.tags.get(index).copied().unwrap_or(0),
                mesh: item.mesh,
                material: item.material,
                world: item.world,
            });
        }
        &self.items
    }
}

/// The path-planning report of one [`plan_sync`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VgSyncReport {
    pub virtual_instances: u32,
    pub forward_instances: u32,
    pub forward_requested: u32,
    pub refused: BTreeMap<&'static str, u32>,
    pub baking_meshes: u32,
    pub ready_meshes: u32,
    pub material_fns: Vec<MaterialFnId>,
    pub bake_requests: Vec<u32>,
    pub virtual_ids: Vec<InstanceId>,
    pub paths: BTreeMap<u32, InstancePath>,
    pub ineligible: BTreeMap<u32, &'static str>,
}

/// One instance the plan uploads: its section key and ordinal, and the record before the pool
/// fills in the ranges and the hierarchy word's root.
#[derive(Clone, Debug, PartialEq)]
pub struct PlannedInstance {
    pub instance: InstanceId,
    pub key: u32,
    pub ordinal: usize,
    pub record: VgInstanceRecord,
}

/// The CPU plan: the report and the records [`VgScene::upload_plan`] writes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VgPlan {
    pub report: VgSyncReport,
    pub instances: Vec<PlannedInstance>,
}

/// Plans one frame's virtualized paths on the CPU: one [`InstancePath`] per listed JS instance
/// group (tag), the virtual records to upload and the report counts.
pub fn plan_sync(
    items: &[SyncItem],
    source: &dyn SyncSource,
    store: &ClusterStore,
    requests: &InstanceRequests,
    keys: &MeshKeys,
    settings: &VgSettings,
) -> VgPlan {
    let mut plan = VgPlan::default();
    let mut virtual_slots: Vec<(u32, InstanceId)> = Vec::new();

    let mut baking = BTreeSet::new();
    let mut ready = BTreeSet::new();
    for item in items {
        if let Some((key, _)) = keys.get(item.mesh) {
            match store.state(key).state {
                ClusterState::Baking => {
                    baking.insert(key);
                }
                ClusterState::Ready => {
                    ready.insert(key);
                }
                _ => {}
            }
        }
    }
    plan.report.baking_meshes = baking.len() as u32;
    plan.report.ready_meshes = ready.len() as u32;

    let mut groups: BTreeMap<u32, Vec<&SyncItem>> = BTreeMap::new();
    for item in items {
        groups.entry(item.tag).or_default().push(item);
    }

    for (tag, group) in groups {
        let requested = requests.get(tag).unwrap_or(settings.instance_default);
        let mut best: Option<(&'static str, String, u8)> = None;
        for item in &group {
            let shape = source.instance_eligibility(item.tag, item.mesh, item.material);
            if let Some(code) = refusal(&shape) {
                let rank = refusal_rank(code);
                if best.as_ref().is_none_or(|(_, _, best)| rank < *best) {
                    best = Some((code, refusal_message(&shape).unwrap_or_default(), rank));
                }
            }
        }
        if let Some((code, _, _)) = &best {
            plan.report.ineligible.insert(tag, *code);
        }
        if !requested {
            plan.report.forward_instances += 1;
            plan.report.paths.insert(
                tag,
                InstancePath::Forward {
                    pending: false,
                    code: None,
                    message: None,
                },
            );
            continue;
        }
        if let Some((code, message, _)) = best {
            plan.report.forward_instances += 1;
            plan.report.forward_requested += 1;
            *plan.report.refused.entry(code).or_insert(0) += 1;
            plan.report.paths.insert(
                tag,
                InstancePath::Forward {
                    pending: false,
                    code: Some(code),
                    message: Some(message),
                },
            );
            continue;
        }

        let mut refused = None;
        let mut absent = false;
        let mut missing_section = false;
        let mut pending = false;
        let mut bake_keys: Vec<u32> = Vec::new();
        let mut planned: Vec<PlannedInstance> = Vec::new();
        for item in &group {
            let Some((key, ordinal)) = keys.get(item.mesh) else {
                absent = true;
                continue;
            };
            let status = store.state(key);
            match status.state {
                ClusterState::Ready => {
                    if let Some(section) = store.sections(key).and_then(|list| list.get(ordinal)) {
                        planned.push(planned_instance(item, key, ordinal, section, source));
                    } else {
                        missing_section = true;
                    }
                }
                ClusterState::None => {
                    pending = true;
                    if settings.runtime_bake {
                        bake_keys.push(key);
                    }
                }
                ClusterState::Baking => pending = true,
                ClusterState::Refused => {
                    refused = Some((
                        store_refusal_code(status.code),
                        status.message.unwrap_or_default(),
                    ));
                }
            }
        }

        if let Some((code, message)) = refused {
            plan.report.forward_instances += 1;
            plan.report.forward_requested += 1;
            plan.report.paths.insert(
                tag,
                InstancePath::Forward {
                    pending: false,
                    code: Some(code),
                    message: Some(message),
                },
            );
        } else if absent || missing_section {
            plan.report.forward_instances += 1;
            plan.report.forward_requested += 1;
            plan.report.paths.insert(
                tag,
                InstancePath::Forward {
                    pending: false,
                    code: None,
                    message: None,
                },
            );
        } else if pending {
            plan.report.forward_instances += 1;
            plan.report.forward_requested += 1;
            for key in bake_keys {
                if !plan.report.bake_requests.contains(&key) {
                    plan.report.bake_requests.push(key);
                }
            }
            plan.report.paths.insert(
                tag,
                InstancePath::Forward {
                    pending: true,
                    code: None,
                    message: None,
                },
            );
        } else {
            plan.report.virtual_instances += 1;
            plan.report.paths.insert(tag, InstancePath::Virtual);
            for item in &group {
                let function = source.material_fn_of(item.material);
                if !plan.report.material_fns.contains(&function) {
                    plan.report.material_fns.push(function);
                }
            }
            for instance in planned {
                virtual_slots.push((instance.record.ids[0], instance.instance));
                plan.instances.push(instance);
            }
        }
    }

    virtual_slots.sort_by_key(|(slot, _)| *slot);
    plan.report.virtual_ids = virtual_slots.into_iter().map(|(_, id)| id).collect();
    plan.report.material_fns.sort();
    plan.report.bake_requests.sort_unstable();
    plan.report.bake_requests.dedup();
    plan
}

fn refusal_rank(code: &'static str) -> u8 {
    match code {
        crate::error::SKINNED_UNSUPPORTED => 0,
        crate::error::DEFORMED_UNSUPPORTED => 1,
        _ => 2,
    }
}

fn store_refusal_code(code: Option<&'static str>) -> &'static str {
    match code {
        Some(code) if code.starts_with("vg.") => code,
        Some("cluster.empty") => "vg.mesh_empty",
        Some("cluster.too_large") => "vg.mesh_too_large",
        _ => "vg.invalid",
    }
}

fn model_rows(model: &Mat4) -> [[f32; 4]; 3] {
    [
        [
            model.cols[0][0],
            model.cols[1][0],
            model.cols[2][0],
            model.cols[3][0],
        ],
        [
            model.cols[0][1],
            model.cols[1][1],
            model.cols[2][1],
            model.cols[3][1],
        ],
        [
            model.cols[0][2],
            model.cols[1][2],
            model.cols[2][2],
            model.cols[3][2],
        ],
    ]
}

fn planned_instance(
    item: &SyncItem,
    key: u32,
    ordinal: usize,
    section: &ClusterMesh,
    source: &dyn SyncSource,
) -> PlannedInstance {
    let rows = model_rows(&item.world);
    let mut flags = 0;
    let previous_rows = match source.previous_instance_row(item.slot) {
        Some(previous) => {
            flags |= INSTANCE_PREVIOUS_VALID;
            if previous != rows {
                flags |= INSTANCE_MOVABLE;
            }
            previous
        }
        None => rows,
    };
    let material_flags = source.material_flags(item.material);
    if material_flags & FLAG_DOUBLE_SIDED != 0 {
        flags |= INSTANCE_DOUBLE_SIDED;
    }
    if material_flags & FLAG_NO_SHADOW == 0 {
        flags |= INSTANCE_CASTS_SHADOW;
    }
    if source.gi_member(item.tag) {
        flags |= INSTANCE_GI;
    }
    PlannedInstance {
        instance: item.instance,
        key,
        ordinal,
        record: VgInstanceRecord {
            model: item.world,
            previous_rows,
            quant: [
                section.quant_origin[0],
                section.quant_origin[1],
                section.quant_origin[2],
                section.quant_step,
            ],
            uv_quant: [
                section.uv_origin[0],
                section.uv_origin[1],
                section.uv_step[0],
                section.uv_step[1],
            ],
            ranges: [0; 4],
            ids: [
                item.slot,
                item.material.0,
                flags,
                hierarchy_word(section.hierarchy_depth, 0),
            ],
            bounds: [
                section.bounds_center[0],
                section.bounds_center[1],
                section.bounds_center[2],
                section.bounds_radius,
            ],
        },
    }
}

#[derive(Clone, Copy)]
struct PoolSpec {
    label: &'static str,
    record_size: u64,
    page: u32,
}

const GEOMETRY_POOLS: [PoolSpec; 4] = [
    PoolSpec {
        label: "raa-vg.nodes",
        record_size: NODE_SIZE,
        page: NODE_PAGE,
    },
    PoolSpec {
        label: "raa-vg.clusters",
        record_size: CLUSTER_SIZE,
        page: CLUSTER_PAGE,
    },
    PoolSpec {
        label: "raa-vg.vertices",
        record_size: VERTEX_SIZE,
        page: VERTEX_PAGE,
    },
    PoolSpec {
        label: "raa-vg.triangles",
        record_size: TRIANGLE_SIZE,
        page: TRIANGLE_PAGE,
    },
];
const INSTANCE_POOL: PoolSpec = PoolSpec {
    label: "raa-vg.instances",
    record_size: VG_INSTANCE_SIZE as u64,
    page: INSTANCE_PAGE,
};

/// Computes geometric, page-aligned growth within the device's binding limit. The required
/// records still take precedence so an unsupported active range receives the normal device error.
fn grown_capacity(current: u32, required: u32, page: u32, limit: u32) -> u32 {
    let page = u64::from(page);
    let current = u64::from(current);
    let wanted = (current + current.div_ceil(2)).max(u64::from(required));
    let rounded = wanted.div_ceil(page).max(1) * page;
    rounded.min(u64::from(limit)).max(u64::from(required)) as u32
}

#[derive(Default)]
struct Pool {
    buffer: Option<wgpu::Buffer>,
    capacity: u32,
}

impl Pool {
    /// Writes `ranges` of `bytes` (the whole stream) into the buffer. A missing or too small
    /// buffer is reallocated with geometric growth, then the whole stream is written once:
    /// the resident prefix and the new data together. Returns the bytes written.
    fn upload(
        &mut self,
        gpu: &GpuContext,
        spec: PoolSpec,
        bytes: &[u8],
        ranges: impl Iterator<Item = Range<usize>>,
        generation: &mut u64,
    ) -> u64 {
        let records = u32::try_from(bytes.len() as u64 / spec.record_size).unwrap_or(u32::MAX);
        let replacing = self.buffer.is_none() || records > self.capacity;
        if replacing {
            let limits = gpu.device().limits();
            let limit = limits
                .max_buffer_size
                .min(limits.max_storage_buffer_binding_size)
                / spec.record_size;
            let limit = u32::try_from(limit).unwrap_or(u32::MAX);
            let capacity = grown_capacity(self.capacity, records, spec.page, limit);
            self.buffer = Some(gpu.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some(spec.label),
                size: u64::from(capacity) * spec.record_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.capacity = capacity;
            *generation += 1;
        }
        let buffer = self.buffer.as_ref().expect("pool buffer allocated");
        if replacing {
            if !bytes.is_empty() {
                gpu.queue().write_buffer(buffer, 0, bytes);
            }
            return bytes.len() as u64;
        }
        let mut written = 0;
        for range in ranges.filter(|range| !range.is_empty()) {
            written += range.len() as u64;
            gpu.queue()
                .write_buffer(buffer, range.start as u64, &bytes[range]);
        }
        written
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

/// Upload counters of a [`VgScene`] since it was created: what the CPU handed the queue.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct VgPoolTraffic {
    /// Bytes written into the four geometry streams (nodes, clusters, vertices, triangles).
    pub geometry_bytes: u64,
    /// Bytes written into the instance buffer.
    pub instance_bytes: u64,
    /// Bytes written into the pool attribute texture (colours and shading uvs).
    pub attribute_bytes: u64,
    /// Pool updates: a sync that changed the resident sections, or a restore after release.
    pub updates: u64,
    /// Geometry stream buffers created or grown (each rewrites its whole stream once).
    pub reallocations: u64,
    /// Attribute textures created (each writes every vertex texel).
    pub attribute_rebuilds: u64,
    /// Same-key replacements written in place (same section shapes).
    pub in_place: u64,
    /// Same-key replacements with other shapes, appended at the end of the streams.
    pub relocations: u64,
    /// Repacks of the whole pool: a new store, or dead ranges outweighing live ones.
    pub compactions: u64,
    /// Current bytes of the four CPU streams, dead ranges included.
    pub resident_bytes: u64,
    /// Current bytes of dead ranges (removed or relocated sections) awaiting compaction.
    pub dead_bytes: u64,
}

/// One Ready key held by the pool: the store revision of its bytes and its section count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ResidentMesh {
    revision: u64,
    section_count: usize,
}

/// Bytes `start[i]..end[i]` of geometry stream `i` that an upload writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PoolSpan {
    start: [usize; 4],
    end: [usize; 4],
}

/// What one [`PackedPool::sync`] changed.
#[derive(Debug, Default, PartialEq, Eq)]
struct PoolUpdate {
    spans: Vec<PoolSpan>,
    /// A section was added, moved or dropped: the vertex bases of the attribute texture changed.
    layout_changed: bool,
}

/// The CPU copy of the four geometry streams and their section layout. A newly Ready key is
/// appended after the resident ones, so earlier sections keep their bases and only the new
/// bytes are uploaded. A replaced key (a new store revision, e.g. a runtime bake followed by its
/// saved section) is overwritten in place when its section shapes are unchanged, else it is
/// appended and its old range becomes dead. A removed key's range becomes dead. The whole
/// pool is repacked in key order only for a new store or when dead bytes exceed live bytes.
#[derive(Default)]
struct PackedPool {
    residents: BTreeMap<u32, ResidentMesh>,
    sections: BTreeMap<(u32, usize), SectionInfo>,
    /// (vertex base, vertex count) of every resident section.
    vertices: BTreeMap<(u32, usize), (u32, u32)>,
    bytes: [Vec<u8>; 4],
    scratch: [Vec<u8>; 4],
    end: SectionRange,
    /// Dead records per stream.
    dead: [u64; 4],
    /// (store identity, store generation, keys generation) of the last sync.
    generations: Option<(u64, u64, u64)>,
    in_place: u64,
    relocations: u64,
    compactions: u64,
}

impl PackedPool {
    fn clear_layout(&mut self) {
        self.residents.clear();
        self.sections.clear();
        self.vertices.clear();
        self.end = SectionRange::default();
        self.dead = [0; 4];
        for bytes in &mut self.bytes {
            bytes.clear();
        }
    }

    fn dead_bytes(&self) -> u64 {
        (0..4)
            .map(|i| self.dead[i] * GEOMETRY_POOLS[i].record_size)
            .sum()
    }

    fn total_bytes(&self) -> u64 {
        self.bytes.iter().map(|bytes| bytes.len() as u64).sum()
    }

    /// Brings the streams to the Ready sections of `keys` in `store`; `None` when neither the
    /// store nor the keys changed since the last sync.
    fn sync(&mut self, store: &ClusterStore, keys: &MeshKeys) -> Option<PoolUpdate> {
        let generations = (store.identity(), store.generation(), keys.generation());
        if self.generations == Some(generations) {
            return None;
        }
        let fresh = self
            .generations
            .is_none_or(|previous| previous.0 != store.identity());
        self.generations = Some(generations);
        let mut full = fresh;
        let mut layout_changed = fresh;
        if fresh {
            if !self.residents.is_empty() {
                self.compactions += 1;
            }
            self.clear_layout();
        }

        let mut spans = Vec::new();
        let residents: Vec<(u32, ResidentMesh)> =
            self.residents.iter().map(|(&k, &r)| (k, r)).collect();
        for (key, resident) in residents {
            let current = if keys.lists.contains_key(&key) {
                store.ready_sections(key)
            } else {
                None
            };
            match current {
                Some((revision, _)) if revision == resident.revision => {}
                Some((revision, sections))
                    if self.same_shape(key, resident.section_count, sections) =>
                {
                    self.overwrite(key, sections, &mut spans);
                    self.residents.insert(
                        key,
                        ResidentMesh {
                            revision,
                            section_count: sections.len(),
                        },
                    );
                    self.in_place += 1;
                }
                Some(_) => {
                    // Appended again below with the newly Ready keys.
                    self.evict(key);
                    self.relocations += 1;
                    layout_changed = true;
                }
                None => {
                    self.evict(key);
                    layout_changed = true;
                }
            }
        }
        if !full && self.dead_bytes() * 2 > self.total_bytes() {
            self.clear_layout();
            self.compactions += 1;
            full = true;
        }

        let append_start = self.bytes.each_ref().map(Vec::len);
        for &key in keys.lists.keys() {
            if self.residents.contains_key(&key) {
                continue;
            }
            let Some((revision, sections)) = store.ready_sections(key) else {
                continue;
            };
            self.residents.insert(
                key,
                ResidentMesh {
                    revision,
                    section_count: sections.len(),
                },
            );
            for (ordinal, mesh) in sections.iter().enumerate() {
                self.append((key, ordinal), mesh);
            }
            layout_changed = true;
        }
        let end = self.bytes.each_ref().map(Vec::len);
        if full {
            spans.clear();
            spans.push(PoolSpan { start: [0; 4], end });
        } else if append_start != end {
            spans.push(PoolSpan {
                start: append_start,
                end,
            });
        }
        (layout_changed || !spans.is_empty()).then_some(PoolUpdate {
            spans,
            layout_changed,
        })
    }

    fn same_shape(&self, key: u32, section_count: usize, sections: &[ClusterMesh]) -> bool {
        section_count == sections.len()
            && sections.iter().enumerate().all(|(ordinal, mesh)| {
                self.sections
                    .get(&(key, ordinal))
                    .is_some_and(|info| info.counts == section_counts(mesh))
            })
    }

    fn append(&mut self, key: (u32, usize), mesh: &ClusterMesh) {
        let counts = section_counts(mesh);
        self.sections.insert(
            key,
            SectionInfo {
                range: self.end,
                hierarchy_depth: mesh.hierarchy_depth,
                root_node: mesh.root_node,
                counts,
            },
        );
        self.vertices.insert(key, (self.end.vertex_base, counts[2]));
        write_section(&mut self.bytes, mesh);
        self.end.node_base = self.end.node_base.saturating_add(counts[0]);
        self.end.cluster_base = self.end.cluster_base.saturating_add(counts[1]);
        self.end.vertex_base = self.end.vertex_base.saturating_add(counts[2]);
        self.end.triangle_base = self.end.triangle_base.saturating_add(counts[3]);
    }

    /// Drops a resident key; its records stay in the streams as dead bytes.
    fn evict(&mut self, key: u32) {
        let Some(resident) = self.residents.remove(&key) else {
            return;
        };
        for ordinal in 0..resident.section_count {
            if let Some(info) = self.sections.remove(&(key, ordinal)) {
                for i in 0..4 {
                    self.dead[i] += u64::from(info.counts[i]);
                }
            }
            self.vertices.remove(&(key, ordinal));
        }
    }

    /// Writes same-shaped sections over a resident key's records, one span per section.
    fn overwrite(&mut self, key: u32, sections: &[ClusterMesh], spans: &mut Vec<PoolSpan>) {
        for (ordinal, mesh) in sections.iter().enumerate() {
            let info = self
                .sections
                .get_mut(&(key, ordinal))
                .expect("resident section");
            info.hierarchy_depth = mesh.hierarchy_depth;
            info.root_node = mesh.root_node;
            let range = info.range;
            let bases = [
                range.node_base,
                range.cluster_base,
                range.vertex_base,
                range.triangle_base,
            ];
            for bytes in &mut self.scratch {
                bytes.clear();
            }
            write_section(&mut self.scratch, mesh);
            let start: [usize; 4] = std::array::from_fn(|i| {
                (u64::from(bases[i]) * GEOMETRY_POOLS[i].record_size) as usize
            });
            let end: [usize; 4] = std::array::from_fn(|i| start[i] + self.scratch[i].len());
            for i in 0..4 {
                self.bytes[i][start[i]..end[i]].copy_from_slice(&self.scratch[i]);
            }
            spans.push(PoolSpan { start, end });
        }
    }
}

fn section_counts(mesh: &ClusterMesh) -> [u32; 4] {
    [
        mesh.nodes.len(),
        mesh.clusters.len(),
        mesh.vertices.len(),
        mesh.triangles.len(),
    ]
    .map(|count| u32::try_from(count).unwrap_or(u32::MAX))
}

/// Appends one section's records to the four streams, as `ClusterMesh::to_section_bytes` and
/// the vertex/triangle writers lay them out.
fn write_section(bytes: &mut [Vec<u8>; 4], mesh: &ClusterMesh) {
    for node in &mesh.nodes {
        node.write_le_bytes(&mut bytes[0]);
    }
    for cluster in &mesh.clusters {
        cluster.write_le_bytes(&mut bytes[1]);
    }
    for vertex in &mesh.vertices {
        vertex.write_le_bytes(&mut bytes[2]);
    }
    for triangle in &mesh.triangles {
        bytes[3].extend_from_slice(&triangle.to_le_bytes());
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
    packed: PackedPool,
    instance_scratch: Vec<u8>,
    last_instance_bytes: Vec<u8>,
    scratch_items: ItemScratch,
    plan_scratch: Vec<VgInstanceRecord>,
    instance_count: u32,
    max_hierarchy_depth: u32,
    generation: u64,
    /// The per-vertex colour streams by (key, ordinal), linear rgba in the order of
    /// `ClusterMesh::vertices`: the colours a virtualized pixel multiplies its base colour by.
    colors: BTreeMap<(u32, usize), Vec<[f32; 4]>>,
    /// The shading attributes by (key, ordinal), which take precedence over `colors`.
    shading: BTreeMap<(u32, usize), VgShading>,
    color_generation: u64,
    layout_serial: u64,
    /// The `Rgba32Uint` pool attribute texture and the (layout serial, colour generation) it
    /// holds.
    color_texture: Option<(wgpu::Texture, wgpu::TextureView)>,
    color_key: Option<(u64, u64)>,
    /// The geometry buffers hold the CPU streams (false before the first upload and after
    /// `release`, whose CPU streams are kept for the next upload).
    pool_uploaded: bool,
    traffic: VgPoolTraffic,
}

impl VgScene {
    /// An empty scene: no buffer exists until the first upload.
    pub fn new() -> VgScene {
        VgScene::default()
    }

    /// Brings the pool to the Ready sections of `keys`: newly Ready sections are appended and
    /// only their bytes are written; resident ranges stay in place (see [`PackedPool`]). After
    /// [`release`](Self::release) the retained CPU streams are written back whole. Returns true
    /// when anything was uploaded.
    pub fn upload_pool(&mut self, gpu: &GpuContext, store: &ClusterStore, keys: &MeshKeys) -> bool {
        let update = self.packed.sync(store, keys);
        let restore = !self.pool_uploaded;
        let Some(update) = update.or_else(|| restore.then(PoolUpdate::default)) else {
            return false;
        };
        if update.layout_changed || restore {
            self.layout_serial += 1;
        }
        self.pool_uploaded = true;
        self.traffic.updates += 1;
        let pools = [
            &mut self.nodes,
            &mut self.clusters,
            &mut self.vertices,
            &mut self.triangles,
        ];
        for (i, pool) in pools.into_iter().enumerate() {
            let generation = self.generation;
            self.traffic.geometry_bytes += pool.upload(
                gpu,
                GEOMETRY_POOLS[i],
                &self.packed.bytes[i],
                update.spans.iter().map(|span| span.start[i]..span.end[i]),
                &mut self.generation,
            );
            if self.generation != generation {
                self.traffic.reallocations += 1;
            }
        }
        if self.instances.buffer.is_none() {
            self.instances.upload(
                gpu,
                INSTANCE_POOL,
                &[],
                std::iter::empty(),
                &mut self.generation,
            );
        }
        true
    }

    /// The upload counters since this scene was created, with the current stream sizes.
    pub fn traffic(&self) -> VgPoolTraffic {
        VgPoolTraffic {
            in_place: self.packed.in_place,
            relocations: self.packed.relocations,
            compactions: self.packed.compactions,
            resident_bytes: self.packed.total_bytes(),
            dead_bytes: self.packed.dead_bytes(),
            ..self.traffic
        }
    }

    /// Registers (`Some`) or removes (`None`) the per-vertex colours of section `ordinal` of
    /// cluster-store key `key`: linear rgba per vertex of `ClusterMesh::vertices`. A refusal
    /// (length not rgba, non-finite) changes nothing. The pool texture follows on the next
    /// upload.
    pub fn set_cluster_colors(
        &mut self,
        key: u32,
        ordinal: usize,
        colors: Option<&[f32]>,
    ) -> Result<(), String> {
        let packed = colors.map(pack_vertex_colors).transpose()?;
        let changed = match packed {
            Some(packed) => self.colors.insert((key, ordinal), packed.clone()) != Some(packed),
            None => self.colors.remove(&(key, ordinal)).is_some(),
        };
        if changed {
            self.color_generation += 1;
        }
        Ok(())
    }

    /// Registers or removes (both `None`) the shading attributes of section `ordinal` of key
    /// `key` ([`VgShading`]): rgba colours (4 floats per vertex) and uvs (2 per vertex) the
    /// material of a virtualized pixel reads in place of the vertex colour and the cluster uv.
    /// A refusal (length, non-finite) changes nothing.
    pub fn set_cluster_shading(
        &mut self,
        key: u32,
        ordinal: usize,
        colors: Option<&[f32]>,
        uvs: Option<&[f32]>,
    ) -> Result<(), String> {
        let colors = colors.map(pack_vertex_colors).transpose()?;
        let uvs = uvs.map(pack_vertex_uvs).transpose()?;
        let changed = if colors.is_none() && uvs.is_none() {
            self.shading.remove(&(key, ordinal)).is_some()
        } else {
            let entry = VgShading { colors, uvs };
            self.shading.insert((key, ordinal), entry.clone()) != Some(entry)
        };
        if changed {
            self.color_generation += 1;
        }
        Ok(())
    }

    /// Rebuilds the pool attribute texture when the pool layout or the streams changed: none when
    /// no stream is registered or the pool is past `VG_COLOR_WIDTH x VG_COLOR_MAX_ROWS` texels
    /// (every vertex reads white then). A created or dropped texture advances the generation.
    pub fn upload_colors(&mut self, gpu: &GpuContext) {
        let key = (self.layout_serial, self.color_generation);
        if self.color_key == Some(key) {
            return;
        }
        self.color_key = Some(key);
        let rows = (self.packed.end.vertex_base * VG_ATTRIBUTE_TEXELS)
            .div_ceil(VG_COLOR_WIDTH)
            .max(1);
        if (self.colors.is_empty() && self.shading.is_empty())
            || self.packed.end.vertex_base == 0
            || rows > VG_COLOR_MAX_ROWS
        {
            if self.color_texture.take().is_some() {
                self.generation += 1;
            }
            return;
        }
        let mut texels = pool_color_texels(
            self.packed.end.vertex_base,
            &self.packed.vertices,
            &self.colors,
            &self.shading,
        );
        texels.resize((rows * VG_COLOR_WIDTH) as usize, VG_NO_UV);
        let size = wgpu::Extent3d {
            width: VG_COLOR_WIDTH,
            height: rows,
            depth_or_array_layers: 1,
        };
        let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("raa-vg.vertex_colors"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let bytes: Vec<u8> = texels
            .iter()
            .flat_map(|t| t.iter().flat_map(|w| w.to_le_bytes()))
            .collect();
        self.traffic.attribute_bytes += bytes.len() as u64;
        self.traffic.attribute_rebuilds += 1;
        gpu.queue().write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(VG_COLOR_WIDTH * 16),
                rows_per_image: Some(rows),
            },
            size,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.color_texture = Some((texture, view));
        self.generation += 1;
    }

    /// The pool attribute texture (`Rgba32Uint`, `VG_COLOR_WIDTH` wide, [`VG_ATTRIBUTE_TEXELS`]
    /// per vertex), or `None` when every vertex is white with its cluster uv.
    pub fn colors(&self) -> Option<&wgpu::TextureView> {
        self.color_texture.as_ref().map(|(_, view)| view)
    }

    /// The absolute bases of a section from the last upload; `None` after `release`.
    pub fn section_range(&self, key: u32, ordinal: usize) -> Option<SectionRange> {
        self.section_header(key, ordinal).map(|(range, _, _)| range)
    }

    /// The section range, its hierarchy depth and its section-relative root node.
    pub fn section_header(&self, key: u32, ordinal: usize) -> Option<(SectionRange, u32, u32)> {
        if !self.pool_uploaded {
            return None;
        }
        self.packed
            .sections
            .get(&(key, ordinal))
            .map(|info| (info.range, info.hierarchy_depth, info.root_node))
    }

    /// Rebuilds the pool when needed, then writes the plan's records in order with their
    /// section ranges and the hierarchy word's root filled from [`VgScene::section_header`].
    pub fn upload_plan(
        &mut self,
        gpu: &GpuContext,
        store: &ClusterStore,
        keys: &MeshKeys,
        plan: &VgPlan,
    ) {
        self.upload_pool(gpu, store, keys);
        self.upload_colors(gpu);
        self.plan_scratch.clear();
        self.plan_scratch.reserve(plan.instances.len());
        for planned in &plan.instances {
            let mut record = planned.record;
            if let Some((range, depth, root)) = self.section_header(planned.key, planned.ordinal) {
                record.ranges = [
                    range.node_base,
                    range.cluster_base,
                    range.vertex_base,
                    range.triangle_base,
                ];
                record.ids[3] = hierarchy_word(depth, root);
            }
            self.plan_scratch.push(record);
        }
        let records = std::mem::take(&mut self.plan_scratch);
        self.upload_instances(gpu, &records);
        self.plan_scratch = records;
    }

    /// Plans and uploads one frame: [`ItemScratch::fill`] into [`plan_sync`], then
    /// [`VgScene::upload_plan`]; returns the report.
    #[allow(clippy::too_many_arguments)]
    pub fn sync(
        &mut self,
        gpu: &GpuContext,
        store: &ClusterStore,
        lit: &LitHandle,
        requests: &InstanceRequests,
        keys: &MeshKeys,
        scene: &Scene,
        settings: &VgSettings,
    ) -> VgSyncReport {
        let plan = {
            let items = self.scratch_items.fill(scene);
            plan_sync(items, lit, store, requests, keys, settings)
        };
        self.upload_plan(gpu, store, keys, &plan);
        plan.report
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
        self.traffic.instance_bytes += self.instances.upload(
            gpu,
            INSTANCE_POOL,
            &self.instance_scratch,
            std::iter::once(0..self.instance_scratch.len()),
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
            + self.color_texture.as_ref().map_or(0, |_| {
                u64::from(
                    (self.packed.end.vertex_base * VG_ATTRIBUTE_TEXELS)
                        .div_ceil(VG_COLOR_WIDTH)
                        .max(1),
                ) * u64::from(VG_COLOR_WIDTH)
                    * 16
            })
    }

    /// Drops every buffer; the next upload writes the retained CPU streams back whole, after
    /// applying whatever changed in the store or the keys meanwhile.
    pub fn release(&mut self) {
        self.instances.release(&mut self.generation);
        self.nodes.release(&mut self.generation);
        self.clusters.release(&mut self.generation);
        self.vertices.release(&mut self.generation);
        self.triangles.release(&mut self.generation);
        if self.color_texture.take().is_some() {
            self.generation += 1;
        }
        self.color_key = None;
        self.pool_uploaded = false;
        self.last_instance_bytes.clear();
        self.instance_count = 0;
        self.max_hierarchy_depth = 0;
    }
}

fn put_f32(out: &mut [u8; VG_INSTANCE_SIZE], at: usize, value: f32) {
    out[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u32(out: &mut [u8; VG_INSTANCE_SIZE], at: usize, value: u32) {
    out[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

/// CPU-only checks of the incremental pool: no adapter is created.
#[cfg(test)]
mod pool_tests {
    use super::*;
    use raa_cluster::{BuildOptions, ClusterSource, OwnedClusterInput, PackedNode, PackedVertex};

    const SIZES: [u64; 4] = [NODE_SIZE, CLUSTER_SIZE, VERTEX_SIZE, TRIANGLE_SIZE];

    /// A section with exactly `counts` records, each record tagged with `tag` and its index.
    fn section(tag: u32, counts: [u32; 4]) -> ClusterMesh {
        ClusterMesh {
            nodes: (0..counts[0])
                .map(|i| PackedNode {
                    first: tag,
                    count: i,
                    ..PackedNode::default()
                })
                .collect(),
            clusters: (0..counts[1])
                .map(|i| raa_cluster::PackedCluster {
                    vertex_offset: tag,
                    triangle_offset: i,
                    ..Default::default()
                })
                .collect(),
            vertices: (0..counts[2])
                .map(|i| PackedVertex {
                    data: [
                        tag as u16,
                        (tag >> 16) as u16,
                        i as u16,
                        (i >> 16) as u16,
                        0,
                        0,
                        0,
                        0,
                    ],
                })
                .collect(),
            triangles: (0..counts[3]).map(|i| i ^ tag.rotate_left(13)).collect(),
            root_node: counts[0].saturating_sub(1),
            hierarchy_depth: 3,
            ..ClusterMesh::default()
        }
    }

    fn keys_for(list: &[u32]) -> MeshKeys {
        let mut keys = MeshKeys::default();
        for &key in list {
            keys.set(key, &[MeshId(key)]);
        }
        keys
    }

    fn bytes_of(mesh: &ClusterMesh) -> [Vec<u8>; 4] {
        let mut out: [Vec<u8>; 4] = Default::default();
        write_section(&mut out, mesh);
        out
    }

    /// What a full repack in key order would upload: the old `upload_pool` contents.
    fn full_pack(store: &ClusterStore, keys: &MeshKeys) -> [Vec<u8>; 4] {
        let mut out: [Vec<u8>; 4] = Default::default();
        for key in keys.keys() {
            for mesh in store.sections(key).unwrap_or_default() {
                write_section(&mut out, mesh);
            }
        }
        out
    }

    /// Every Ready section of `keys` is resident at its range with its exact records, the
    /// ranges do not overlap, and the vertex table matches the sections.
    fn assert_consistent(pool: &PackedPool, store: &ClusterStore, keys: &MeshKeys) {
        let mut claimed: [Vec<(usize, usize)>; 4] = Default::default();
        let mut expected_sections = 0;
        for key in keys.keys() {
            let Some((revision, sections)) = store.ready_sections(key) else {
                assert!(!pool.residents.contains_key(&key), "key {key} is not ready");
                continue;
            };
            assert_eq!(
                pool.residents.get(&key),
                Some(&ResidentMesh {
                    revision,
                    section_count: sections.len()
                })
            );
            for (ordinal, mesh) in sections.iter().enumerate() {
                expected_sections += 1;
                let info = pool.sections[&(key, ordinal)];
                assert_eq!(info.counts, section_counts(mesh));
                assert_eq!(info.root_node, mesh.root_node);
                assert_eq!(info.hierarchy_depth, mesh.hierarchy_depth);
                assert_eq!(
                    pool.vertices[&(key, ordinal)],
                    (info.range.vertex_base, mesh.vertices.len() as u32)
                );
                let bases = [
                    info.range.node_base,
                    info.range.cluster_base,
                    info.range.vertex_base,
                    info.range.triangle_base,
                ];
                let want = bytes_of(mesh);
                for i in 0..4 {
                    let start = (u64::from(bases[i]) * SIZES[i]) as usize;
                    let end = start + want[i].len();
                    assert_eq!(
                        &pool.bytes[i][start..end],
                        &want[i][..],
                        "stream {i} of {key}"
                    );
                    if end > start {
                        claimed[i].push((start, end));
                    }
                }
            }
        }
        assert_eq!(pool.sections.len(), expected_sections);
        assert_eq!(pool.vertices.len(), expected_sections);
        for (i, ranges) in claimed.iter_mut().enumerate() {
            ranges.sort_unstable();
            for pair in ranges.windows(2) {
                assert!(pair[0].1 <= pair[1].0, "stream {i} ranges overlap");
            }
            let live: u64 = ranges.iter().map(|(a, b)| (b - a) as u64).sum();
            assert_eq!(
                live + pool.dead[i] * SIZES[i],
                pool.bytes[i].len() as u64,
                "stream {i}: live + dead = total"
            );
        }
    }

    /// The bytes an update writes, with the GPU buffer modelled as a byte array.
    fn apply(gpu: &mut [Vec<u8>; 4], pool: &PackedPool, update: &PoolUpdate) -> u64 {
        let mut written = 0;
        for i in 0..4 {
            gpu[i].resize(gpu[i].len().max(pool.bytes[i].len()), 0xAA);
            for span in &update.spans {
                let range = span.start[i]..span.end[i];
                written += range.len() as u64;
                gpu[i][range.clone()].copy_from_slice(&pool.bytes[i][range]);
            }
        }
        written
    }

    fn assert_gpu_matches(gpu: &[Vec<u8>; 4], pool: &PackedPool) {
        for i in 0..4 {
            // Dead ranges may hold anything; every live section must match.
            for info in pool.sections.values() {
                let bases = [
                    info.range.node_base,
                    info.range.cluster_base,
                    info.range.vertex_base,
                    info.range.triangle_base,
                ];
                let start = (u64::from(bases[i]) * SIZES[i]) as usize;
                let end = start + (u64::from(info.counts[i]) * SIZES[i]) as usize;
                assert_eq!(gpu[i][start..end], pool.bytes[i][start..end], "stream {i}");
            }
        }
    }

    #[test]
    fn appends_keep_bases_and_write_only_the_new_section() {
        let mut store = ClusterStore::new();
        let keys = keys_for(&[2, 5, 9]);
        let mut pool = PackedPool::default();
        let mut gpu: [Vec<u8>; 4] = Default::default();

        let first = pool.sync(&store, &keys).expect("first sync");
        assert!(first.layout_changed);
        assert_eq!(apply(&mut gpu, &pool, &first), 0);
        assert!(pool.sync(&store, &keys).is_none(), "nothing changed");

        // Arrival order 9, 2, 5 differs from the key order.
        let mut written = 0;
        let mut ranges = Vec::new();
        for (key, counts) in [
            (9, [3, 5, 40, 60]),
            (2, [1, 2, 10, 12]),
            (5, [7, 9, 90, 150]),
        ] {
            store.attach(key, vec![section(key, counts)]);
            let before = pool.bytes.each_ref().map(Vec::len);
            let update = pool.sync(&store, &keys).expect("an attach uploads");
            assert!(update.layout_changed);
            assert_eq!(update.spans.len(), 1);
            assert_eq!(update.spans[0].start, before, "only the appended bytes");
            written += apply(&mut gpu, &pool, &update);
            assert_consistent(&pool, &store, &keys);
            assert_gpu_matches(&gpu, &pool);
            ranges.push(pool.sections[&(key, 0)].range);
            for (k, range) in [9, 2, 5].iter().zip(&ranges) {
                assert_eq!(
                    pool.sections[&(*k, 0)].range,
                    *range,
                    "a resident base moved"
                );
            }
        }
        assert_eq!(written, pool.total_bytes(), "each byte is written once");
        let full = full_pack(&store, &keys);
        assert_eq!(
            pool.total_bytes(),
            full.iter().map(|b| b.len() as u64).sum::<u64>()
        );
    }

    #[test]
    fn a_key_without_mesh_ids_is_not_uploaded_until_registered() {
        let mut store = ClusterStore::new();
        let mut keys = keys_for(&[1]);
        let mut pool = PackedPool::default();
        store.attach(1, vec![section(1, [1, 1, 3, 1])]);
        store.attach(4, vec![section(4, [2, 2, 6, 2])]);
        pool.sync(&store, &keys).expect("sync");
        assert!(!pool.residents.contains_key(&4));
        keys.set(4, &[MeshId(40)]);
        let update = pool.sync(&store, &keys).expect("the new key uploads");
        assert_eq!(update.spans.len(), 1);
        assert_consistent(&pool, &store, &keys);
    }

    /// The Windows activation order: a runtime bake of a key becomes Ready before the saved
    /// section of the same key arrives. Equal shapes overwrite in place, nothing else moves.
    #[test]
    fn runtime_then_attached_same_key_overwrites_in_place() {
        let mut store = ClusterStore::new();
        let keys = keys_for(&[1, 2, 3]);
        let mut pool = PackedPool::default();
        let mut gpu: [Vec<u8>; 4] = Default::default();
        store.attach(1, vec![section(1, [4, 6, 50, 70])]);
        store
            .start_bake(
                2,
                vec![OwnedClusterInput::displaced_sphere(2)],
                &BuildOptions::default(),
            )
            .expect("bake");
        let mut guard = 0;
        while store.state(2).state != ClusterState::Ready {
            store.step_all(&mut || true);
            guard += 1;
            assert!(guard < 10_000);
        }
        assert_eq!(store.state(2).source, Some(ClusterSource::Runtime));
        store.attach(3, vec![section(3, [2, 3, 20, 30])]);
        let update = pool.sync(&store, &keys).expect("sync");
        apply(&mut gpu, &pool, &update);
        assert_consistent(&pool, &store, &keys);
        let before: Vec<_> = [1, 2, 3].map(|k| pool.sections[&(k, 0)].range).to_vec();
        let total = pool.total_bytes();

        // The saved section: the same baked records (the sidecar's vertices and indices are
        // byte-identical to the runtime bake), with different cluster errors.
        let mut saved = store.sections(2).expect("runtime sections").to_vec();
        for cluster in &mut saved[0].clusters {
            cluster.lod_error += 0.25;
        }
        store.attach(2, saved.clone());
        assert_eq!(store.state(2).source, Some(ClusterSource::Attached));
        let update = pool.sync(&store, &keys).expect("the replacement uploads");
        assert!(!update.layout_changed, "no base moved");
        let written = apply(&mut gpu, &pool, &update);
        let section_bytes: u64 = bytes_of(&saved[0]).iter().map(|b| b.len() as u64).sum();
        assert_eq!(written, section_bytes, "only key 2 is rewritten");
        assert_eq!(pool.total_bytes(), total);
        assert_eq!(pool.dead_bytes(), 0);
        assert_eq!(pool.in_place, 1);
        let after: Vec<_> = [1, 2, 3].map(|k| pool.sections[&(k, 0)].range).to_vec();
        assert_eq!(before, after);
        assert_consistent(&pool, &store, &keys);
        assert_gpu_matches(&gpu, &pool);
    }

    #[test]
    fn a_differently_shaped_replacement_relocates_then_compacts() {
        let mut store = ClusterStore::new();
        let keys = keys_for(&[1, 2, 3]);
        let mut pool = PackedPool::default();
        let mut gpu: [Vec<u8>; 4] = Default::default();
        store.attach(1, vec![section(1, [10, 20, 400, 600])]);
        store.attach(2, vec![section(2, [2, 3, 30, 40])]);
        store.attach(3, vec![section(3, [10, 20, 400, 600])]);
        let update = pool.sync(&store, &keys).expect("sync");
        apply(&mut gpu, &pool, &update);
        let r1 = pool.sections[&(1, 0)].range;
        let r3 = pool.sections[&(3, 0)].range;

        store.attach(2, vec![section(2, [3, 4, 32, 44])]);
        let update = pool.sync(&store, &keys).expect("relocation");
        assert!(update.layout_changed);
        let written = apply(&mut gpu, &pool, &update);
        assert_eq!(
            written,
            bytes_of(&section(2, [3, 4, 32, 44]))
                .iter()
                .map(|b| b.len() as u64)
                .sum::<u64>()
        );
        assert_eq!(pool.relocations, 1);
        assert_eq!(pool.compactions, 0);
        assert!(pool.dead_bytes() > 0);
        assert_eq!(pool.sections[&(1, 0)].range, r1);
        assert_eq!(pool.sections[&(3, 0)].range, r3);
        assert_consistent(&pool, &store, &keys);
        assert_gpu_matches(&gpu, &pool);

        // Clearing the two large keys leaves dead bytes above the live ones: one repack.
        store.clear(1);
        let update = pool.sync(&store, &keys).expect("removal");
        apply(&mut gpu, &pool, &update);
        assert_eq!(pool.compactions, 0, "dead still below live");
        store.clear(3);
        let update = pool.sync(&store, &keys).expect("removal");
        assert_eq!(pool.compactions, 1);
        assert_eq!(pool.dead_bytes(), 0);
        assert_eq!(
            update.spans,
            vec![PoolSpan {
                start: [0; 4],
                end: pool.bytes.each_ref().map(Vec::len)
            }]
        );
        assert_eq!(
            pool.bytes,
            full_pack(&store, &keys),
            "a compaction is the key-order pack"
        );
        apply(&mut gpu, &pool, &update);
        assert_consistent(&pool, &store, &keys);
        assert_gpu_matches(&gpu, &pool);
    }

    #[test]
    fn removing_mesh_ids_or_store_entries_evicts() {
        let mut store = ClusterStore::new();
        let mut keys = keys_for(&[1, 2, 3, 4]);
        let mut pool = PackedPool::default();
        for key in 1..=4 {
            store.attach(key, vec![section(key, [5, 5, 50, 50])]);
        }
        pool.sync(&store, &keys).expect("sync");
        keys.remove(2);
        let update = pool.sync(&store, &keys).expect("eviction");
        assert!(update.layout_changed && update.spans.is_empty());
        assert_consistent(&pool, &store, &keys);
        store
            .start_bake(
                3,
                vec![OwnedClusterInput::single_triangle()],
                &BuildOptions::default(),
            )
            .expect("bake");
        pool.sync(&store, &keys)
            .expect("a key leaving Ready is evicted");
        assert!(!pool.residents.contains_key(&3));
        assert_consistent(&pool, &store, &keys);
    }

    #[test]
    fn a_new_store_repacks_everything() {
        let mut first = ClusterStore::new();
        let keys = keys_for(&[1, 2]);
        let mut pool = PackedPool::default();
        first.attach(2, vec![section(2, [1, 1, 8, 8])]);
        first.attach(1, vec![section(1, [1, 1, 8, 8])]);
        pool.sync(&first, &keys).expect("sync");
        let mut second = ClusterStore::new();
        second.attach(1, vec![section(11, [1, 1, 8, 8])]);
        second.attach(2, vec![section(12, [1, 1, 8, 8])]);
        assert_eq!(first.generation(), second.generation());
        let update = pool.sync(&second, &keys).expect("another store");
        assert_eq!(update.spans[0].start, [0; 4]);
        assert_eq!(pool.bytes, full_pack(&second, &keys));
        assert_consistent(&pool, &second, &keys);
    }

    #[test]
    fn pool_colour_texels_follow_appended_bases() {
        let mut store = ClusterStore::new();
        let keys = keys_for(&[1, 2]);
        let mut pool = PackedPool::default();
        store.attach(2, vec![section(2, [1, 1, 3, 1])]);
        pool.sync(&store, &keys).expect("sync");
        store.attach(1, vec![section(1, [1, 1, 2, 1])]);
        pool.sync(&store, &keys).expect("sync");
        // Key 2 arrived first, so it holds vertices 0..3 and key 1 holds 3..5.
        assert_eq!(pool.vertices[&(2, 0)], (0, 3));
        assert_eq!(pool.vertices[&(1, 0)], (3, 2));
        let mut colors = BTreeMap::new();
        colors.insert((1, 0), vec![[0.5, 0.25, 0.125, 1.0]; 2]);
        let mut shading = BTreeMap::new();
        shading.insert(
            (2, 0),
            VgShading {
                colors: None,
                uvs: Some(vec![[0.75, 0.5]; 3]),
            },
        );
        let texels = pool_color_texels(pool.end.vertex_base, &pool.vertices, &colors, &shading);
        assert_eq!(texels.len(), 10);
        for v in 0..3 {
            assert_eq!(texels[2 * v], VG_WHITE);
            assert_eq!(
                texels[2 * v + 1],
                [0.75f32.to_bits(), 0.5f32.to_bits(), 1, 0]
            );
        }
        for v in 3..5 {
            assert_eq!(texels[2 * v][0], 0.5f32.to_bits());
            assert_eq!(texels[2 * v + 1], VG_NO_UV);
        }
    }

    #[test]
    fn growth_is_geometric_page_aligned_and_capped() {
        assert_eq!(grown_capacity(0, 0, 64, 1 << 20), 64);
        assert_eq!(grown_capacity(0, 100, 64, 1 << 20), 128);
        assert_eq!(grown_capacity(128, 129, 64, 1 << 20), 192);
        assert_eq!(grown_capacity(1000, 1001, 256, 1 << 20), 1536);
        assert_eq!(
            grown_capacity(1000, 1001, 256, 1200),
            1200,
            "capped at the limit"
        );
        assert_eq!(grown_capacity(1000, 1300, 256, 1200), 1300, "required wins");
        // Over the 319 Sponza vertex appends the buffer is recreated a logarithmic number of times.
        let mut capacity = 0;
        let mut reallocations = 0;
        for required in (1..=319u32).map(|i| i * 15_600) {
            if required > capacity {
                capacity = grown_capacity(capacity, required, VERTEX_PAGE, u32::MAX);
                reallocations += 1;
            }
        }
        assert!(reallocations <= 16, "{reallocations} reallocations");
    }

    /// The 319 attachments of the Windows Sponza activation (native-final-f3458a90-a984c1e-v2-a)
    /// in their recorded order, with the exact record counts of their v2 sidecar sections. The
    /// old upload wrote every resident stream on each store change; the incremental pool writes
    /// each section once, plus one in-place rewrite for the runtime→attached key.
    #[test]
    fn sponza_activation_order_writes_each_section_once() {
        let order: Vec<(u32, [u32; 4])> = include_str!("../tests/data/sponza-attachments.txt")
            .lines()
            .map(|line| {
                let v: Vec<u32> = line.split(' ').map(|n| n.parse().expect("count")).collect();
                (v[0], [v[1], v[2], v[3], v[4]])
            })
            .collect();
        assert_eq!(order.len(), 319);
        let keys = keys_for(&order.iter().map(|(key, _)| *key).collect::<Vec<_>>());
        let mut store = ClusterStore::new();
        let mut pool = PackedPool::default();
        let mut gpu: [Vec<u8>; 4] = Default::default();
        pool.sync(&store, &keys).expect("first sync");
        let (mut old_bytes, mut new_bytes, mut resident) = (0u64, 0u64, 0u64);
        // A runtime bake of the 40th mesh is Ready before its saved section (same shape).
        let early = order[39].0;
        for (step, &(key, counts)) in order.iter().enumerate() {
            if step == 10 {
                store.attach(early, vec![section(early ^ 0x5a5a, order[39].1)]);
                resident += order[39]
                    .1
                    .iter()
                    .zip(SIZES)
                    .map(|(&n, s)| u64::from(n) * s)
                    .sum::<u64>();
                old_bytes += resident;
                let update = pool.sync(&store, &keys).expect("runtime ready");
                new_bytes += apply(&mut gpu, &pool, &update);
            }
            let mesh = section(key, counts);
            if key != early {
                resident += counts
                    .iter()
                    .zip(SIZES)
                    .map(|(&n, s)| u64::from(n) * s)
                    .sum::<u64>();
            }
            store.attach(key, vec![mesh]);
            old_bytes += resident;
            let update = pool.sync(&store, &keys).expect("attach");
            new_bytes += apply(&mut gpu, &pool, &update);
        }
        assert_consistent(&pool, &store, &keys);
        assert_gpu_matches(&gpu, &pool);
        let early_bytes: u64 = order[39]
            .1
            .iter()
            .zip(SIZES)
            .map(|(&n, s)| u64::from(n) * s)
            .sum();
        assert_eq!(pool.total_bytes(), 109_405_000, "the v2 sidecar payload");
        assert_eq!(resident, 109_405_000);
        assert_eq!(pool.in_place, 1);
        assert_eq!(pool.relocations + pool.compactions, 0);
        assert_eq!(new_bytes, 109_405_000 + early_bytes);
        // The old path: 19,509,389,224 bytes for the 319 attachments alone (reconstructed from
        // the same order), plus the early runtime upload.
        assert!(old_bytes > 19_509_389_224, "{old_bytes}");
        assert!(old_bytes / new_bytes > 170, "{old_bytes} vs {new_bytes}");
        eprintln!("sponza order: old path {old_bytes} bytes, incremental {new_bytes} bytes");
    }
}
