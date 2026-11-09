//! The engine state behind the JS `Engine` (host and wasm32): the scene, the lit renderer,
//! the pass pipeline and the JS handle tables. JS instance handles map to instance groups:
//! one scene instance per sub-mesh of a lit mesh, each tagged with the JS handle so the
//! per-instance extras (S10) of every sub-mesh draw are shared.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use raa_core::{GpuContext, MaterialId, MeshId, RaaError, wgpu};
use raa_csg::CsgResolver;
use raa_geom::{GeometryRegistry, MeshData};
use raa_lit::{LitMeshId, LitRenderer, RestMeshView};
use raa_loader::LoaderRegistry;
use raa_pipeline::{FeaturePass, LightingResources, Pipeline, RecordCtx, SubmissionState};
use raa_scene::{InstanceId, NodeId, OrbitController, RenderList, Scene};
use raa_skin::{SKIN_ORDER, SKIN_PASS, SkinMesh, SkinPass, SkinVertexInputs, validate_pose};
use raa_tonemap::TonemapControl;

use raa_scene::SceneError;

use crate::convert::{HandleMap, api_error, scene_error, skin_error};
use crate::feature_map::FeatureMap;

/// The material drawn by a sub-mesh without one (the lit renderer's default material).
pub(crate) const DEFAULT_MATERIAL: MaterialId = MaterialId(0);

/// One scene-frame index per Engine invocation (all viewports share it) and the one-shot
/// resize/camera-cut marks its prepares consume.
#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct SceneInvocation {
    index: u64,
    open: bool,
    discontinuity: u8,
}

impl SceneInvocation {
    pub(crate) fn begin(&mut self) -> u64 {
        self.open = true;
        self.index
    }
    pub(crate) fn discontinuity(&mut self, view: usize) {
        self.discontinuity |= view_bit(view);
    }
    pub(crate) fn discontinuity_all(&mut self) {
        self.discontinuity = 0b1111;
    }
    pub(crate) fn prepare(&mut self, view: usize) -> bool {
        let bit = view_bit(view);
        let dropped = self.discontinuity & bit != 0;
        self.discontinuity &= !bit;
        dropped
    }
    /// `(views to acknowledge, history complete)`: nothing submitted keeps history and
    /// index; anything submitted advances once; partial or failed drops history.
    pub(crate) fn finish(&mut self, runtime_ok: bool, s: SubmissionState) -> (u8, bool) {
        if !self.open {
            return (0, true);
        }
        let acknowledge = s.submitted_mask | if runtime_ok { s.pending_mask } else { 0 };
        let skipped = (s.submitted_mask | s.pending_mask) == 0 && !s.record_failed;
        let complete = skipped || (runtime_ok && !s.record_failed);
        if acknowledge != 0 {
            self.index = self.index.saturating_add(1);
        }
        self.open = false;
        (acknowledge, complete)
    }
}

fn view_bit(view: usize) -> u8 {
    1 << (view % 4)
}

/// Which API created a mesh handle.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum MeshOrigin {
    /// `primitiveMesh` or `loadPrimsAsset` (S10 default attributes).
    V0,
    /// `addMesh` (m07).
    M07,
}

/// A JS mesh handle: the lit mesh and where it came from.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct MeshEntry {
    pub lit: LitMeshId,
    pub origin: MeshOrigin,
}

/// The S18 skin retained for one lit mesh: the validated typed joints (`u16x4`) and
/// weights (`f32x4`) per vertex, with the rest geometry they were validated against.
pub(crate) struct SkinEntry {
    pub mesh: LitMeshId,
    pub skin: SkinMesh,
}

struct SkinnedMeshGpu {
    inputs: Rc<SkinVertexInputs>,
    rest: RestMeshView,
}

fn skin_busy() -> RaaError {
    RaaError::FrameGraph {
        code: "pipeline.busy",
        message: "the skin pass is borrowed (called while a frame executes)".into(),
    }
}

/// S18 pose joint count of `skin`: one past the highest joint index it references.
pub(crate) fn skin_pose_joints(skin: &SkinMesh) -> usize {
    skin.joints
        .iter()
        .copied()
        .max()
        .map_or(0, |j| usize::from(j) + 1)
}

/// Validates one mesh-space pose for `skin`: 16 finite, column-major floats per joint
/// (`skin.pose_length`, `skin.cap_joints`), every referenced joint present
/// (`skin.joint_range`) and exactly [`skin_pose_joints`] joints, the batch layout's
/// per-instance slice (`skin.pose_length`).
pub(crate) fn validate_skin_pose(skin: &SkinMesh, matrices: &[f32]) -> Result<(), RaaError> {
    let joints = validate_pose(matrices).map_err(|e| skin_error(&e))?;
    skin.validate_joint_count(joints)
        .map_err(|e| skin_error(&e))?;
    let expected = skin_pose_joints(skin);
    if joints != expected {
        return Err(api_error(
            "skin.pose_length",
            format!(
                "pose has {joints} joints; this mesh's skin pose has {expected} \
                 (16 column-major floats per joint)"
            ),
        ));
    }
    Ok(())
}

/// The `(start, end)` float ranges of the S18 pose batch layout: one slice of
/// 16 x [`skin_pose_joints`] per skin, concatenated in order (mixed joint counts
/// allowed). Every slice, and the total `floats`, is validated before any upload.
pub(crate) fn skin_pose_ranges(
    skins: &[&SkinMesh],
    matrices: &[f32],
) -> Result<Vec<(usize, usize)>, RaaError> {
    let floats = matrices.len();
    let total = skins
        .iter()
        .map(|skin| skin_pose_joints(skin) * 16)
        .sum::<usize>();
    let mismatch = || {
        api_error(
            "skin.pose_length",
            format!(
                "pose data has {floats} floats; {} instance(s) need {total} (16 floats per \
                 joint, one slice per mesh skin)",
                skins.len()
            ),
        )
    };
    let mut ranges = Vec::with_capacity(skins.len());
    let mut offset = 0usize;
    for skin in skins {
        let end = offset + skin_pose_joints(skin) * 16;
        let slice = matrices.get(offset..end).ok_or_else(mismatch)?;
        validate_skin_pose(skin, slice)?;
        ranges.push((offset, end));
        offset = end;
    }
    if offset != floats {
        return Err(mismatch());
    }
    Ok(ranges)
}

/// The `skin` feature (S18): the order-15 pass and per-mesh rest uploads.
#[derive(Clone)]
pub(crate) struct SkinFeature {
    pass: Rc<RefCell<SkinPass>>,
    meshes: Rc<RefCell<BTreeMap<LitMeshId, Rc<SkinnedMeshGpu>>>>,
    /// Tags whose view is built before the next lit prepare (replaced in place).
    pending: Rc<RefCell<Vec<(u32, LitMeshId)>>>,
}

impl SkinFeature {
    pub(crate) fn new(gpu: &GpuContext) -> Self {
        SkinFeature {
            pass: Rc::new(RefCell::new(SkinPass::new(gpu))),
            meshes: Rc::new(RefCell::new(BTreeMap::new())),
            pending: Rc::new(RefCell::new(Vec::new())),
        }
    }

    pub(crate) fn pipeline_pass(&self) -> SkinPipelinePass {
        SkinPipelinePass(Rc::clone(&self.pass))
    }

    pub(crate) fn connect(&self, lit: &mut LitRenderer) {
        lit.handle().set_skin_views(self.pass.borrow().views());
    }

    pub(crate) fn add_instance(
        &self,
        gpu: &GpuContext,
        instance: u32,
        mesh: LitMeshId,
        skin: &SkinMesh,
    ) -> Result<(), RaaError> {
        let data = self.mesh_data(gpu, mesh, skin)?;
        self.pass
            .try_borrow_mut()
            .map_err(|_| skin_busy())?
            .add_instance(instance, Rc::clone(&data.inputs), data.rest.clone())
    }

    pub(crate) fn queue(&self, tag: u32, mesh: LitMeshId) {
        let mut pending = self.pending.borrow_mut();
        match pending.iter_mut().find(|(t, _)| *t == tag) {
            Some(slot) => slot.1 = mesh,
            None => pending.push((tag, mesh)),
        }
    }

    pub(crate) fn flush(&self, gpu: &GpuContext, skins: &[SkinEntry]) -> Result<(), RaaError> {
        let pending = std::mem::take(&mut *self.pending.borrow_mut());
        for (tag, mesh) in pending {
            match skins.iter().find(|e| e.mesh == mesh) {
                Some(entry) => self.add_instance(gpu, tag, mesh, &entry.skin)?,
                None => self.remove_instance(tag),
            }
        }
        Ok(())
    }

    pub(crate) fn remove_instance(&self, tag: u32) {
        self.pending.borrow_mut().retain(|(t, _)| *t != tag);
        if let Ok(mut pass) = self.pass.try_borrow_mut() {
            pass.remove_instance(tag);
        }
    }

    pub(crate) fn forget_mesh(&self, mesh: LitMeshId) {
        self.meshes.borrow_mut().remove(&mesh);
    }

    fn mesh_data(
        &self,
        gpu: &GpuContext,
        mesh: LitMeshId,
        skin: &SkinMesh,
    ) -> Result<Rc<SkinnedMeshGpu>, RaaError> {
        if let Some(data) = self.meshes.borrow().get(&mesh) {
            return Ok(Rc::clone(data));
        }
        let inputs = Rc::new(SkinVertexInputs::new(gpu, skin)?);
        let data = Rc::new(SkinnedMeshGpu {
            rest: rest_view(gpu, skin, inputs.edges)?,
            inputs,
        });
        self.meshes.borrow_mut().insert(mesh, Rc::clone(&data));
        Ok(data)
    }
}

pub(crate) struct SkinPipelinePass(Rc<RefCell<SkinPass>>);

impl FeaturePass for SkinPipelinePass {
    fn name(&self) -> &str {
        SKIN_PASS
    }

    fn order(&self) -> u32 {
        SKIN_ORDER
    }

    fn set_enabled(&mut self, enabled: bool, lighting: &mut LightingResources) {
        if let Ok(mut pass) = self.0.try_borrow_mut() {
            pass.set_enabled(enabled, lighting);
        }
    }

    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        self.0
            .try_borrow_mut()
            .map_err(|_| skin_busy())?
            .record(ctx)
    }
}

/// The rest view sizes: S1 stream 0, S10 stream 1, indices and 24 B per edge.
fn rest_view(gpu: &GpuContext, skin: &SkinMesh, edges: u32) -> Result<RestMeshView, RaaError> {
    gpu.ensure_live()?;
    let mut rest = skin.rest.clone();
    rest.ensure_tangents();
    let (mut stream0, mut stream1, mut index) = (Vec::new(), Vec::new(), Vec::new());
    rest.write_stream0_bytes(&mut stream0);
    rest.write_stream1_bytes(&mut stream1);
    rest.write_index_bytes(&mut index);
    // A plain buffer plus a queue write: `create_buffer_init` maps at creation and panics on a
    // device the browser has already lost, before `with_state` can report `core.gpu`.
    let buffer = |label: &'static str, contents: &[u8], usage: wgpu::BufferUsages| {
        let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: contents.len() as u64,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !contents.is_empty() {
            gpu.queue().write_buffer(&buffer, 0, contents);
        }
        buffer
    };
    let edge_buffer = (edges > 0).then(|| {
        gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa.skin.rest.edges"),
            size: u64::from(edges) * 24,
            usage: wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        })
    });
    RestMeshView::new(
        buffer(
            "raa.skin.rest.stream0",
            &stream0,
            wgpu::BufferUsages::VERTEX,
        ),
        buffer(
            "raa.skin.rest.stream1",
            &stream1,
            wgpu::BufferUsages::VERTEX,
        ),
        buffer("raa.skin.rest.index", &index, wgpu::BufferUsages::INDEX),
        edge_buffer,
    )
}

/// A JS instance handle: one scene instance per sub-mesh of `mesh` at `node`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct InstanceGroup {
    pub node: NodeId,
    pub mesh: LitMeshId,
    /// One per sub-mesh, in sub-mesh order; empty until built.
    pub instances: Vec<InstanceId>,
    pub visible: bool,
    /// The JS instance handle, set as the tag of every scene instance.
    pub tag: u32,
    /// v0 `addInstance(node, mesh, material)`: every sub-mesh draws this material.
    pub material: Option<MaterialId>,
    /// `setInstanceMaterial`: the object's single material, drawn on every sub-mesh while
    /// `setSingleMaterial(true)` is on (ignored while it is off).
    pub object_material: Option<MaterialId>,
    /// `setSingleMaterial(enabled, instance)`: this instance's own setting; `None` follows the
    /// engine's global setting.
    pub single: Option<bool>,
    /// The node was removed (`removeNode`): the group is never rebuilt and using it
    /// throws `scene.stale_instance`, as a v0 instance of a removed node did.
    pub stale: bool,
}

impl InstanceGroup {
    pub(crate) fn new(
        node: NodeId,
        mesh: LitMeshId,
        tag: u32,
        material: Option<MaterialId>,
    ) -> Self {
        Self {
            node,
            mesh,
            instances: Vec::new(),
            visible: true,
            tag,
            material,
            object_material: None,
            single: None,
            stale: false,
        }
    }

    /// Replaces the scene instances of the group by one per `(sub-mesh id, sub-mesh
    /// material)` part, keeping node, visibility and tag. The override material wins,
    /// then the sub-mesh material, then [`DEFAULT_MATERIAL`]. Previous instances that are
    /// already stale (their node was removed) are skipped. A stale node refuses with
    /// `scene.stale_node` and marks the group stale; the group then lists only the
    /// instances that exist.
    pub(crate) fn rebuild(
        &mut self,
        scene: &mut Scene,
        parts: impl IntoIterator<Item = (MeshId, Option<MaterialId>)>,
    ) -> Result<(), RaaError> {
        for id in self.instances.drain(..) {
            // A stale instance is already gone from the scene: nothing to remove.
            let _ = scene.remove_instance(id);
        }
        for (mesh, sub_material) in parts {
            let material = self.material.or(sub_material).unwrap_or(DEFAULT_MATERIAL);
            let id = match scene.add_instance(self.node, mesh, material) {
                Ok(id) => id,
                Err(e) => {
                    self.stale |= e == SceneError::StaleNode;
                    return Err(scene_error(&e));
                }
            };
            self.instances.push(id);
            scene
                .set_instance_tag(id, self.tag)
                .map_err(|e| scene_error(&e))?;
            if !self.visible {
                scene
                    .set_instance_visible(id, false)
                    .map_err(|e| scene_error(&e))?;
            }
        }
        Ok(())
    }

    /// Removes every scene instance of the group. All are tried; the first error
    /// (`scene.stale_instance` when the node was removed, as in v0) is returned.
    pub(crate) fn remove_from(&mut self, scene: &mut Scene) -> Result<(), RaaError> {
        let mut first = self.check_live();
        for id in self.instances.drain(..) {
            if let Err(e) = scene.remove_instance(id)
                && first.is_ok()
            {
                first = Err(scene_error(&e));
            }
        }
        first
    }

    /// Shows or hides every scene instance of the group (kept across rebuilds).
    pub(crate) fn set_visible(&mut self, scene: &mut Scene, visible: bool) -> Result<(), RaaError> {
        self.check_live()?;
        self.visible = visible;
        for &id in &self.instances {
            scene
                .set_instance_visible(id, visible)
                .map_err(|e| scene_error(&e))?;
        }
        Ok(())
    }

    /// `scene.stale_instance` when the group's node was removed.
    pub(crate) fn check_live(&self) -> Result<(), RaaError> {
        if self.stale {
            Err(scene_error(&SceneError::StaleInstance))
        } else {
            Ok(())
        }
    }
}

/// Marks stale every live group whose node no longer exists in `scene`.
fn mark_stale_groups(scene: &Scene, groups: &mut HandleMap<InstanceGroup>) {
    for (_, group) in groups.iter_mut() {
        if !group.stale && scene.local(group.node).is_err() {
            group.stale = true;
        }
    }
}

/// Runs `rebuild` on every live (not stale) group whose mesh is in `dirty`, then clears
/// `dirty` (capacity kept). Every group is tried; the first error is returned.
fn rebuild_dirty(
    dirty: &mut Vec<LitMeshId>,
    groups: &mut HandleMap<InstanceGroup>,
    mut rebuild: impl FnMut(&mut InstanceGroup) -> Result<(), RaaError>,
) -> Result<(), RaaError> {
    if dirty.is_empty() {
        return Ok(());
    }
    let mut first = Ok(());
    for (_, group) in groups.iter_mut() {
        if group.stale || !dirty.contains(&group.mesh) {
            continue;
        }
        if let Err(e) = rebuild(group)
            && first.is_ok()
        {
            first = Err(e);
        }
    }
    dirty.clear();
    first
}

/// A loaded asset: one (mesh handle, material handle) per LoadedMesh, in mesh order.
pub(crate) struct AssetEntry {
    pub parts: Vec<(u32, u32)>,
}

/// Everything the engine owns once the GPU is up.
pub(crate) struct EngineState {
    pub scene: Scene,
    pub list: RenderList,
    /// `tags[i]` is the scene tag of `list.items[i]`.
    pub tags: Vec<u32>,
    /// `slots[i]` is the scene instance slot of `list.items[i]`.
    pub slots: Vec<u32>,
    pub orbit: OrbitController,
    /// The 8 S3 kinds; assets load with strict kinds.
    pub geometry: GeometryRegistry,
    pub loaders: LoaderRegistry,
    pub booleans: CsgResolver,
    /// Reused generation buffer for primitiveMesh.
    pub scratch: MeshData,
    pub lit: LitRenderer,
    pub pipeline: Rc<RefCell<Pipeline>>,
    pub tonemap: TonemapControl,
    pub features: FeatureMap,
    /// The open render/probe invocation and the scene-frame index advancing once per
    /// submitted invocation (M2-FRAME-LIFECYCLE handoff in `engine.rs`).
    pub scene_invocation: SceneInvocation,
    /// Last effective camera (eye3, target3, fov, near, far) of each viewport that has its
    /// own m21 camera: a change is a camera discontinuity for that viewport.
    pub view_cameras: [Option<[f32; 9]>; 4],
    pub max_materials: u32,
    pub meshes: HandleMap<MeshEntry>,
    /// The S18 skin of every mesh uploaded with joints and weights, keyed by lit mesh.
    pub skins: Vec<SkinEntry>,
    pub materials: HandleMap<MaterialId>,
    pub nodes: HandleMap<NodeId>,
    pub instances: HandleMap<InstanceGroup>,
    /// Lit meshes whose geometry was replaced since the last frame: their instance groups
    /// are rebuilt by [`EngineState::rebuild_dirty_groups`] in `update`, before
    /// `lit.prepare`.
    pub dirty_meshes: Vec<LitMeshId>,
    /// `setSingleMaterial`: every instance draws its object's single material on all its
    /// sub-meshes instead of the per sub-mesh materials. Default false.
    pub single_material: bool,
    /// Asset handle to an index into `assets`.
    pub asset_handles: HandleMap<u32>,
    pub assets: Vec<AssetEntry>,
    /// setTransforms copies the typed arrays here; grown only for a larger input.
    pub scratch_nodes: Vec<u32>,
    pub scratch_trs: Vec<f32>,
}

impl EngineState {
    /// The scene node behind JS handle `h` (`api.unknown_handle`).
    pub(crate) fn node(&self, h: u32) -> Result<NodeId, RaaError> {
        self.nodes.get(h)
    }

    pub(crate) fn material(&self, h: u32) -> Result<MaterialId, RaaError> {
        self.materials.get(h)
    }

    pub(crate) fn mesh(&self, h: u32) -> Result<MeshEntry, RaaError> {
        self.meshes.get(h)
    }

    /// Retains the S18 skin of `mesh`, replacing a previous one; `None` clears it.
    pub(crate) fn set_skin(&mut self, mesh: LitMeshId, skin: Option<SkinMesh>) {
        self.clear_skin(mesh);
        if let Some(skin) = skin {
            self.skins.push(SkinEntry { mesh, skin });
        }
        if let Some(feature) = self.features.get::<SkinFeature>() {
            feature.forget_mesh(mesh);
        }
    }

    /// The S18 skin of `mesh`, when it was uploaded with joints and weights.
    pub(crate) fn skin(&self, mesh: LitMeshId) -> Option<&SkinMesh> {
        self.skins.iter().find(|e| e.mesh == mesh).map(|e| &e.skin)
    }

    /// Drops the skin of `mesh`, if any.
    pub(crate) fn clear_skin(&mut self, mesh: LitMeshId) {
        self.skins.retain(|e| e.mesh != mesh);
    }

    /// Plans a pose batch: looks up every handle and its mesh skin, then validates the
    /// concatenated slices ([`skin_pose_ranges`]) before anything is uploaded.
    fn plan_skin_poses(
        &self,
        instances: &[u32],
        matrices: &[f32],
    ) -> Result<Vec<(u32, LitMeshId, usize, usize)>, RaaError> {
        let mut plan = Vec::with_capacity(instances.len());
        let mut skins = Vec::with_capacity(instances.len());
        for &h in instances {
            let group = self.instance_group(h)?;
            let skin = self.skin(group.mesh).ok_or_else(|| {
                api_error(
                    "skin.not_skinned",
                    format!("instance {h} draws a mesh without S18 joints and weights"),
                )
            })?;
            skins.push(skin);
            plan.push((group.tag, group.mesh));
        }
        let ranges = skin_pose_ranges(&skins, matrices)?;
        Ok(plan
            .into_iter()
            .zip(ranges)
            .map(|((tag, mesh), (start, end))| (tag, mesh, start, end))
            .collect())
    }

    /// Sets the mesh-space, column-major pose of JS instance `h` (16 floats per joint).
    /// Refusals: `api.unknown_handle`, `skin.not_skinned`, `skin.pose_length`,
    /// `skin.cap_joints`, `skin.joint_range`; a refusal changes nothing.
    pub(crate) fn set_skin_pose(
        &mut self,
        gpu: &GpuContext,
        h: u32,
        matrices: &[f32],
    ) -> Result<(), RaaError> {
        validate_pose(matrices).map_err(|e| skin_error(&e))?;
        let plan = self.plan_skin_poses(&[h], matrices)?;
        self.apply_skin_poses(gpu, &plan, matrices)
    }

    /// Sets many poses at once: `matrices` concatenates one slice per `instances` entry
    /// in order, each 16 x its mesh skin's joint count (mixed joint counts allowed). The
    /// whole batch is validated before any instance is updated.
    pub(crate) fn set_skin_poses(
        &mut self,
        gpu: &GpuContext,
        instances: &[u32],
        matrices: &[f32],
    ) -> Result<(), RaaError> {
        let plan = self.plan_skin_poses(instances, matrices)?;
        self.apply_skin_poses(gpu, &plan, matrices)
    }

    /// Structural counts of the skin pass's last frame (zeros without the feature).
    pub(crate) fn skin_stats(&self) -> raa_skin::SkinFrameStats {
        self.features
            .get::<SkinFeature>()
            .and_then(|f| f.pass.try_borrow().ok().map(|p| p.last_stats()))
            .unwrap_or_default()
    }

    fn apply_skin_poses(
        &mut self,
        gpu: &GpuContext,
        plan: &[(u32, LitMeshId, usize, usize)],
        matrices: &[f32],
    ) -> Result<(), RaaError> {
        self.rebuild_dirty_groups()?;
        self.flush_skins(gpu)?;
        let feature = self.features.get::<SkinFeature>().ok_or_else(|| {
            api_error(
                "api.feature_missing",
                "feature `skin` is not installed in this engine",
            )
        })?;
        for &(tag, _, start, end) in plan {
            feature
                .pass
                .try_borrow_mut()
                .map_err(|_| skin_busy())?
                .set_pose(gpu, tag, &matrices[start..end])?;
        }
        Ok(())
    }

    pub(crate) fn instance_group(&self, h: u32) -> Result<&InstanceGroup, RaaError> {
        self.instances.get_ref(h)
    }

    pub(crate) fn instance_group_mut(&mut self, h: u32) -> Result<&mut InstanceGroup, RaaError> {
        self.instances.get_mut(h)
    }

    /// Creates the instance group of JS handle `next_handle()` for `mesh` at `node` and
    /// returns that handle. `material` overrides every sub-mesh material (v0 addInstance).
    pub(crate) fn add_instance_group(
        &mut self,
        node: NodeId,
        mesh: LitMeshId,
        material: Option<MaterialId>,
    ) -> Result<u32, RaaError> {
        let tag = self.instances.next_handle();
        let mut group = InstanceGroup::new(node, mesh, tag, material);
        let skin = self.features.get::<SkinFeature>();
        let registration = skin.zip(Some(self.skins.as_slice()));
        let single = self.single_material;
        let r = rebuild_group(&mut self.scene, &self.lit, registration, single, &mut group);
        if let Err(e) = r {
            // Leave no half-built group in the scene.
            let _ = group.remove_from(&mut self.scene);
            return Err(e);
        }
        Ok(self.instances.insert(group))
    }

    /// Rebuilds the group of JS handle `h` from its mesh's current sub-meshes.
    pub(crate) fn rebuild_instance_group(&mut self, h: u32) -> Result<(), RaaError> {
        let skin = self.features.get::<SkinFeature>();
        let registration = skin.zip(Some(self.skins.as_slice()));
        let single = self.single_material;
        let group = self.instances.get_mut(h)?;
        rebuild_group(&mut self.scene, &self.lit, registration, single, group)
    }

    /// `setSingleMaterial(enabled, instance)`: the setting of one instance (`None` follows the
    /// global setting again); the group is rebuilt.
    pub(crate) fn set_instance_single_material(
        &mut self,
        h: u32,
        on: Option<bool>,
    ) -> Result<(), RaaError> {
        let group = self.instances.get_mut(h)?;
        group.check_live()?;
        if group.single == on {
            return Ok(());
        }
        group.single = on;
        self.rebuild_instance_group(h)
    }

    /// `setSingleMaterial`: switches the draw material of every instance group between the
    /// per sub-mesh materials (off, the default) and the object's single material (on), and
    /// rebuilds the live groups (an instance with its own setting keeps it). The meshes and
    /// their sub-mesh materials are not changed.
    pub(crate) fn set_single_material(&mut self, on: bool) -> Result<(), RaaError> {
        if self.single_material == on {
            return Ok(());
        }
        self.single_material = on;
        let (scene, lit, skins) = (&mut self.scene, &self.lit, &self.skins);
        let registration = self
            .features
            .get::<SkinFeature>()
            .zip(Some(skins.as_slice()));
        let mut first = Ok(());
        for (_, group) in self.instances.iter_mut() {
            if group.stale || lit.submesh_ids(group.mesh).is_none() {
                continue;
            }
            if let Err(e) = rebuild_group(scene, lit, registration, on, group)
                && first.is_ok()
            {
                first = Err(e);
            }
        }
        first
    }

    /// Removes the group of JS handle `h` (its handle is never reused) and its scene
    /// instances; the instance extras of its tag are cleared. A stale group (its node was
    /// removed) is unmapped too and throws `scene.stale_instance`, as in v0.
    pub(crate) fn remove_instance_group(&mut self, h: u32) -> Result<(), RaaError> {
        let mut group = self.instances.remove(h)?;
        if let Some(skin) = self.features.get::<SkinFeature>() {
            skin.remove_instance(group.tag);
        }
        self.lit.clear_instance_extra(group.tag);
        self.lit.handle().set_outline(group.tag, None);
        self.lit.handle().clear_wireframe(group.tag);
        group.remove_from(&mut self.scene)
    }

    /// Removes `node`, its subtree and their scene instances, and marks stale the groups
    /// whose node went away (later use throws `scene.stale_instance`). Their instance
    /// extras are cleared; their handles stay mapped, as in v0.
    pub(crate) fn remove_node(&mut self, node: NodeId) -> Result<(), RaaError> {
        self.scene.remove_node(node).map_err(|e| scene_error(&e))?;
        mark_stale_groups(&self.scene, &mut self.instances);
        let skin = self.features.get::<SkinFeature>();
        for (_, group) in self.instances.iter() {
            if group.stale {
                if let Some(skin) = skin {
                    skin.remove_instance(group.tag);
                }
                self.lit.clear_instance_extra(group.tag);
                self.lit.handle().set_outline(group.tag, None);
                self.lit.handle().clear_wireframe(group.tag);
            }
        }
        Ok(())
    }

    /// Records that the geometry of `mesh` was replaced: its instance groups are rebuilt
    /// at the next frame, keeping node, visibility and tag.
    pub(crate) fn mark_mesh_dirty(&mut self, mesh: LitMeshId) {
        if !self.dirty_meshes.contains(&mesh) {
            self.dirty_meshes.push(mesh);
        }
    }

    /// Frees the mesh of JS handle `h` (any origin) and drops it from the dirty list.
    pub(crate) fn remove_mesh(&mut self, h: u32) -> Result<(), RaaError> {
        let mesh = self.meshes.remove(h)?;
        self.clear_skin(mesh.lit);
        if let Some(skin) = self.features.get::<SkinFeature>() {
            skin.forget_mesh(mesh.lit);
            for (_, group) in self.instances.iter() {
                if group.mesh == mesh.lit {
                    skin.remove_instance(group.tag);
                }
            }
        }
        self.dirty_meshes.retain(|&m| m != mesh.lit);
        self.lit.remove(mesh.lit);
        Ok(())
    }

    /// Rebuilds the groups of the meshes marked dirty since the last frame (called by
    /// `update` before `lit.prepare`). Stale groups and groups whose mesh was removed
    /// since are skipped. Nothing is done, and nothing allocated, when the list is empty.
    pub(crate) fn rebuild_dirty_groups(&mut self) -> Result<(), RaaError> {
        let single = self.single_material;
        let (scene, lit, skins) = (&mut self.scene, &self.lit, &self.skins);
        let skin = self.features.get::<SkinFeature>();
        let registration = skin.zip(Some(skins.as_slice()));
        rebuild_dirty(&mut self.dirty_meshes, &mut self.instances, |group| {
            if lit.submesh_ids(group.mesh).is_none() {
                return Ok(());
            }
            rebuild_group(scene, lit, registration, single, group)
        })
    }

    /// Builds the skinned views queued by instance creation and rebuilds, before lit
    /// prepares the frame. Tags of meshes without a skin draw rest.
    pub(crate) fn flush_skins(&mut self, gpu: &GpuContext) -> Result<(), RaaError> {
        let Some(skin) = self.features.get::<SkinFeature>().cloned() else {
            return Ok(());
        };
        skin.flush(gpu, &self.skins)
    }
}

/// Rebuilds `group` from the sub-meshes of its lit mesh (`render.unknown_mesh` when the
/// mesh was removed) and re-registers the group's tag on the mesh's shared rest upload.
/// The group's scene instances share one lit mesh and tag, so one view (and one pose)
/// covers them; other groups of the same mesh keep their own views.
fn rebuild_group(
    scene: &mut Scene,
    lit: &LitRenderer,
    skin: Option<(&SkinFeature, &[SkinEntry])>,
    single: bool,
    group: &mut InstanceGroup,
) -> Result<(), RaaError> {
    let unknown = || {
        crate::convert::api_error(
            "render.unknown_mesh",
            format!("lit mesh {} does not exist", group.mesh.0),
        )
    };
    let info = lit.info(group.mesh).ok_or_else(unknown)?;
    let ids = lit.submesh_ids(group.mesh).ok_or_else(unknown)?;
    let parts: Vec<(MeshId, Option<MaterialId>, u32)> = ids
        .iter()
        .zip(&info.submeshes)
        .map(|(&id, sub)| (id, sub.material, sub.index_count))
        .collect();
    let draws = group_draw_materials(group, &parts, single);
    group.rebuild(scene, draws)?;
    if let Some((feature, skins)) = skin {
        feature.remove_instance(group.tag);
        if skins.iter().any(|e| e.mesh == group.mesh) {
            feature.queue(group.tag, group.mesh);
        }
    }
    Ok(())
}

/// The draw material of each sub-mesh `(id, material, index count)`. Off: the sub-mesh's
/// own material (the per-face assignment, unchanged). On (`setSingleMaterial(true)`): every
/// sub-mesh draws the object's single material, `object` (`setInstanceMaterial`), else the
/// material of the sub-mesh with the most indices (the first on a tie), else none (the
/// default material). A v0 group material still wins over both (InstanceGroup::rebuild).
/// [`draw_materials`] of `group`: its own `setSingleMaterial` setting, else the global one,
/// and its `setInstanceMaterial` material.
pub(crate) fn group_draw_materials(
    group: &InstanceGroup,
    parts: &[(MeshId, Option<MaterialId>, u32)],
    global: bool,
) -> Vec<(MeshId, Option<MaterialId>)> {
    let single = group.single.unwrap_or(global);
    draw_materials(
        parts,
        single,
        if single { group.object_material } else { None },
    )
}

pub(crate) fn draw_materials(
    parts: &[(MeshId, Option<MaterialId>, u32)],
    single: bool,
    object: Option<MaterialId>,
) -> Vec<(MeshId, Option<MaterialId>)> {
    if !single {
        return parts.iter().map(|&(id, m, _)| (id, m)).collect();
    }
    let mut largest: Option<(u32, Option<MaterialId>)> = None;
    for &(_, m, count) in parts {
        if largest.is_none_or(|(c, _)| count > c) {
            largest = Some((count, m));
        }
    }
    let one = object.or(largest.and_then(|(_, m)| m));
    parts.iter().map(|&(id, _, _)| (id, one)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_scene::Transform;

    fn scene_with_node() -> (Scene, NodeId) {
        let mut scene = Scene::new();
        let node = scene.add_node(None, Transform::IDENTITY).unwrap();
        (scene, node)
    }

    fn parts() -> [(MeshId, Option<MaterialId>); 2] {
        [(MeshId(3), Some(MaterialId(5))), (MeshId(4), None)]
    }

    #[test]
    fn rebuild_makes_one_tagged_instance_per_submesh() {
        let (mut scene, node) = scene_with_node();
        let mut g = InstanceGroup::new(node, LitMeshId(0), 7, None);
        g.rebuild(&mut scene, parts()).unwrap();
        assert_eq!(g.instances.len(), 2);
        assert_eq!(scene.instance_count(), 2);
        for &id in &g.instances {
            assert_eq!(scene.instance_tag(id).unwrap(), 7);
        }
        let mut list = RenderList::default();
        let mut tags = Vec::new();
        scene.update_world();
        scene.build_render_list_tagged(&mut list, &mut tags);
        let mut used: Vec<_> = list.items.iter().map(|i| (i.mesh, i.material)).collect();
        used.sort();
        assert_eq!(
            used,
            [(MeshId(3), MaterialId(5)), (MeshId(4), DEFAULT_MATERIAL)]
        );
        assert_eq!(tags, [7, 7]);
    }

    /// The (mesh, material) of every render-list item of a group rebuilt from `parts`, sorted.
    fn drawn(
        parts: &[(MeshId, Option<MaterialId>, u32)],
        single: bool,
        object: Option<MaterialId>,
        v0: Option<MaterialId>,
    ) -> Vec<(MeshId, MaterialId)> {
        let (mut scene, node) = scene_with_node();
        let mut g = InstanceGroup::new(node, LitMeshId(0), 1, v0);
        g.rebuild(&mut scene, draw_materials(parts, single, object))
            .unwrap();
        let (mut list, mut tags) = (RenderList::default(), Vec::new());
        scene.update_world();
        scene.build_render_list_tagged(&mut list, &mut tags);
        let mut used: Vec<_> = list.items.iter().map(|i| (i.mesh, i.material)).collect();
        used.sort();
        used
    }

    // A mesh of 3 sub-meshes: 900 indices with no material (the object's faces), 300 with
    // material 5 (a face assignment), 120 with material 6.
    const FACES: [(MeshId, Option<MaterialId>, u32); 3] = [
        (MeshId(3), None, 900),
        (MeshId(4), Some(MaterialId(5)), 300),
        (MeshId(5), Some(MaterialId(6)), 120),
    ];

    #[test]
    fn single_material_off_keeps_the_per_face_materials() {
        // Off is the behaviour before the option: each sub-mesh draws its own material.
        let before: Vec<_> = FACES.iter().map(|&(id, m, _)| (id, m)).collect();
        assert_eq!(draw_materials(&FACES, false, Some(MaterialId(9))), before);
        assert_eq!(
            drawn(&FACES, false, Some(MaterialId(9)), None),
            [
                (MeshId(3), DEFAULT_MATERIAL),
                (MeshId(4), MaterialId(5)),
                (MeshId(5), MaterialId(6))
            ]
        );
    }

    #[test]
    fn single_material_on_draws_one_material_on_every_face() {
        // The object's material (setInstanceMaterial) on all three sub-meshes.
        assert_eq!(
            drawn(&FACES, true, Some(MaterialId(9)), None),
            [
                (MeshId(3), MaterialId(9)),
                (MeshId(4), MaterialId(9)),
                (MeshId(5), MaterialId(9))
            ]
        );
        // None given: the largest sub-mesh decides; here it has no material, so the default.
        assert_eq!(
            drawn(&FACES, true, None, None),
            [
                (MeshId(3), DEFAULT_MATERIAL),
                (MeshId(4), DEFAULT_MATERIAL),
                (MeshId(5), DEFAULT_MATERIAL)
            ]
        );
        let painted = [
            (MeshId(3), Some(MaterialId(7)), 900),
            (MeshId(4), Some(MaterialId(5)), 300),
        ];
        assert_eq!(
            drawn(&painted, true, None, None),
            [(MeshId(3), MaterialId(7)), (MeshId(4), MaterialId(7))]
        );
        // A v0 addInstance material still wins.
        assert_eq!(
            drawn(&FACES, true, Some(MaterialId(9)), Some(MaterialId(2))),
            [
                (MeshId(3), MaterialId(2)),
                (MeshId(4), MaterialId(2)),
                (MeshId(5), MaterialId(2))
            ]
        );
    }

    #[test]
    fn single_material_instance_setting_wins_over_the_global_one() {
        let (_, node) = scene_with_node();
        let mut g = InstanceGroup::new(node, LitMeshId(0), 1, None);
        g.object_material = Some(MaterialId(9));
        let own: Vec<_> = FACES.iter().map(|&(id, m, _)| (id, m)).collect();
        let one: Vec<_> = FACES
            .iter()
            .map(|&(id, _, _)| (id, Some(MaterialId(9))))
            .collect();
        // Follows the global setting while it has none.
        assert_eq!(group_draw_materials(&g, &FACES, false), own);
        assert_eq!(group_draw_materials(&g, &FACES, true), one);
        // An instance turned off draws its per-face materials under a global on, and the reverse.
        g.single = Some(false);
        assert_eq!(group_draw_materials(&g, &FACES, true), own);
        g.single = Some(true);
        assert_eq!(group_draw_materials(&g, &FACES, false), one);
    }

    #[test]
    fn override_material_wins_and_rebuild_keeps_visibility() {
        let (mut scene, node) = scene_with_node();
        let mut g = InstanceGroup::new(node, LitMeshId(0), 2, Some(MaterialId(9)));
        g.rebuild(&mut scene, parts()).unwrap();
        g.set_visible(&mut scene, false).unwrap();
        g.rebuild(&mut scene, parts()[..1].iter().copied()).unwrap();
        assert_eq!(scene.instance_count(), 1);
        let mut list = RenderList::default();
        let mut tags = Vec::new();
        scene.update_world();
        scene.build_render_list_tagged(&mut list, &mut tags);
        assert!(list.items.is_empty());
        g.set_visible(&mut scene, true).unwrap();
        scene.build_render_list_tagged(&mut list, &mut tags);
        assert_eq!(list.items.len(), 1);
        assert_eq!(list.items[0].material, MaterialId(9));
    }

    #[test]
    fn removal_clears_the_group_and_reports_stale_instances() {
        let (mut scene, node) = scene_with_node();
        let mut g = InstanceGroup::new(node, LitMeshId(0), 1, None);
        g.rebuild(&mut scene, parts()).unwrap();
        g.remove_from(&mut scene).unwrap();
        assert!(g.instances.is_empty());
        assert_eq!(scene.instance_count(), 0);

        g.rebuild(&mut scene, parts()).unwrap();
        scene.remove_node(node).unwrap();
        let e = g.remove_from(&mut scene).unwrap_err();
        assert_eq!(e.code(), "scene.stale_instance");
        assert!(g.instances.is_empty());
        let e = g.rebuild(&mut scene, parts()).unwrap_err();
        assert_eq!(e.code(), "scene.stale_node");
    }

    #[test]
    fn removed_node_marks_groups_stale_and_use_throws_stale_instance() {
        let (mut scene, node) = scene_with_node();
        let other = scene.add_node(None, Transform::IDENTITY).unwrap();
        let mut groups: HandleMap<InstanceGroup> = HandleMap::new("instance");
        for n in [node, other] {
            let mut g = InstanceGroup::new(n, LitMeshId(0), groups.next_handle(), None);
            g.rebuild(&mut scene, parts()).unwrap();
            groups.insert(g);
        }
        scene.remove_node(node).unwrap();
        mark_stale_groups(&scene, &mut groups);
        assert!(groups.get_ref(1).unwrap().stale);
        assert!(!groups.get_ref(2).unwrap().stale);

        let g = groups.get_mut(1).unwrap();
        let e = g.set_visible(&mut scene, false).unwrap_err();
        assert_eq!(e.code(), "scene.stale_instance");
        let e = g.remove_from(&mut scene).unwrap_err();
        assert_eq!(e.code(), "scene.stale_instance");
        // Still stale with no instance left (e.g. after a failed rebuild).
        let e = g.remove_from(&mut scene).unwrap_err();
        assert_eq!(e.code(), "scene.stale_instance");
        assert_eq!(scene.instance_count(), 2);
    }

    #[test]
    fn dirty_meshes_rebuild_live_groups_keeping_node_visibility_and_tag() {
        let (mut scene, node) = scene_with_node();
        let gone = scene.add_node(None, Transform::IDENTITY).unwrap();
        let mut groups: HandleMap<InstanceGroup> = HandleMap::new("instance");
        for (n, mesh) in [(node, 0), (node, 1), (gone, 0)] {
            let mut g = InstanceGroup::new(n, LitMeshId(mesh), groups.next_handle(), None);
            g.rebuild(&mut scene, parts()).unwrap();
            groups.insert(g);
        }
        groups
            .get_mut(1)
            .unwrap()
            .set_visible(&mut scene, false)
            .unwrap();
        scene.remove_node(gone).unwrap();
        mark_stale_groups(&scene, &mut groups);

        let mut dirty = vec![LitMeshId(0)];
        let replaced = [(MeshId(8), None)];
        let mut rebuilt = Vec::new();
        rebuild_dirty(&mut dirty, &mut groups, |g| {
            rebuilt.push(g.tag);
            g.rebuild(&mut scene, replaced)
        })
        .unwrap();
        assert_eq!(rebuilt, [1]);
        assert!(dirty.is_empty());
        let g = groups.get_ref(1).unwrap();
        assert_eq!((g.node, g.tag, g.visible), (node, 1, false));
        assert_eq!(g.instances.len(), 1);
        assert_eq!(scene.instance_tag(g.instances[0]).unwrap(), 1);
        assert_eq!(groups.get_ref(2).unwrap().instances.len(), 2);
        assert_eq!(scene.instance_count(), 3);

        // An empty list does nothing; a failing rebuild is reported after all are tried.
        rebuild_dirty(&mut dirty, &mut groups, |_| unreachable!()).unwrap();
        let mut dirty = vec![LitMeshId(0), LitMeshId(1)];
        let mut tried = 0;
        let e = rebuild_dirty(&mut dirty, &mut groups, |_| {
            tried += 1;
            Err(scene_error(&SceneError::StaleNode))
        })
        .unwrap_err();
        assert_eq!((e.code(), tried), ("scene.stale_node", 2));
        assert!(dirty.is_empty());
    }

    #[test]
    fn rebuild_on_a_removed_node_marks_the_group_stale() {
        let (mut scene, node) = scene_with_node();
        let mut g = InstanceGroup::new(node, LitMeshId(0), 1, None);
        scene.remove_node(node).unwrap();
        assert!(g.rebuild(&mut scene, parts()).is_err());
        assert!(g.stale);
    }

    #[test]
    fn group_handles_refuse_unknown() {
        let mut groups: HandleMap<InstanceGroup> = HandleMap::new("instance");
        let (_, node) = scene_with_node();
        assert_eq!(groups.next_handle(), 1);
        let h = groups.insert(InstanceGroup::new(node, LitMeshId(0), 1, None));
        assert_eq!(groups.get_ref(h).unwrap().tag, h);
        groups.get_mut(h).unwrap().visible = false;
        assert!(!groups.remove(h).unwrap().visible);
        for h in [0, 1, 2] {
            assert_eq!(groups.get_ref(h).unwrap_err().code(), "api.unknown_handle");
        }
        assert_eq!(groups.next_handle(), 2);
    }

    fn test_skin(joints: &[u16]) -> SkinMesh {
        let vertices = joints.len() / 4;
        let positions: Vec<f32> = (0..vertices).flat_map(|i| [i as f32, 0.0, 0.0]).collect();
        let normals = [0.0, 0.0, 1.0].repeat(vertices);
        let mesh =
            raa_mesh::from_js_arrays(&positions, &normals, None, None, None, &[0, 0, 0], &[])
                .unwrap();
        let weights = [1.0, 0.0, 0.0, 0.0].repeat(vertices);
        SkinMesh::new(mesh, joints, &weights).unwrap()
    }

    #[test]
    fn skin_pose_joints_is_one_past_the_highest_referenced_joint() {
        assert_eq!(skin_pose_joints(&test_skin(&[0; 4])), 1);
        assert_eq!(skin_pose_joints(&test_skin(&[0, 0, 0, 0, 2, 0, 0, 0])), 3);
    }

    #[test]
    fn skin_pose_validation_uses_the_s18_codes() {
        let skin = test_skin(&[0, 0, 0, 0, 2, 0, 0, 0]);
        let code = |matrices: Vec<f32>| validate_skin_pose(&skin, &matrices).unwrap_err().code();
        validate_skin_pose(&skin, &[0.0; 48]).unwrap();
        assert_eq!(code(vec![]), "skin.pose_length");
        assert_eq!(code(vec![0.0; 15]), "skin.pose_length");
        assert_eq!(code(vec![f32::NAN; 48]), "skin.pose_length");
        assert_eq!(code(vec![0.0; 32]), "skin.joint_range");
        assert_eq!(code(vec![0.0; 64]), "skin.pose_length");
        assert_eq!(code(vec![0.0; 257 * 16]), "skin.cap_joints");
    }

    #[test]
    fn skin_pose_batch_slices_mixed_joint_counts_before_any_update() {
        let one = test_skin(&[0; 4]);
        let three = test_skin(&[0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0]);
        let skins = [&one, &three];
        let ok = vec![0.0; (1 + 3) * 16];
        assert_eq!(skin_pose_ranges(&skins, &ok).unwrap(), [(0, 16), (16, 64)]);
        for (floats, mentions) in [(63usize, "63"), (64 + 16, "80")] {
            let e = skin_pose_ranges(&skins, &vec![0.0; floats]).unwrap_err();
            assert_eq!(e.code(), "skin.pose_length");
            assert!(e.to_string().contains(mentions), "{e}");
        }
        let e = skin_pose_ranges(&skins, &[f32::NAN; 80]).unwrap_err();
        assert_eq!(e.code(), "skin.pose_length");
    }
}
