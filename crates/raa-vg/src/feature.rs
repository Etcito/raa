//! `VgFeature::install` and the `VgHandle` of RAA's virtualized geometry (S31, S32): the five
//! S12 entries (`vg.cull` 21, `vg.raster` 22, `vg.resolve` 23, `vg.gbuffer` 32 and the forward
//! contributor `vg.shade` 450), the settings, the per-instance requests and mesh keys, the
//! active-state table and the frame preparation that plans, uploads and switches the path.
//!
//! The handle shares one `SharedVg` frame state with every pass and contributor. An inactive
//! frame releases the frame state, so no vg buffer or texture survives and feature ids 27, 28
//! and 36 stay clear (P3-Q8, S32); feature id 36 follows the cascade availability.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::rc::{Rc, Weak};

use raa_cluster::ClusterStore;
use raa_core::{GpuContext, MeshId, RaaError, wgpu};
use raa_lit::LitRenderer;
use raa_pipeline::Pipeline;
use raa_scene::{Camera, Scene};
use raa_shadow::ShadowHandle;

use crate::culling::{CullPipelines, ViewSlots};
use crate::eligibility::InstancePath;
use crate::error::{
    DEFORMED_UNSUPPORTED, SKINNED_UNSUPPORTED, TRANSPARENT_MATERIAL, invalid, refusal,
};
use crate::frame_passes::{SharedVg, VgCullPass, VgFrameState, VgRasterPass, VgResolvePass};
use crate::gbuffer::VgGbufferPass;
use crate::scene::{
    InstanceRequests, ItemScratch, MeshKeys, VgPlan, VgScene, VgSyncReport, WithGiMembers,
    plan_sync,
};
use crate::settings::{VgSettings, VgSettingsPatch};
use crate::shade::VgShadeContributor;

/// The five pipeline entries this feature installs, in S12 order.
const VG_ENTRIES: [&str; 5] = [
    "vg.cull",
    "vg.raster",
    "vg.resolve",
    "vg.gbuffer",
    "vg.shade",
];

fn pipeline_busy() -> RaaError {
    refusal(
        "pipeline.busy",
        "the pipeline is borrowed (called while a frame executes)".to_string(),
    )
}

fn duplicate(name: &str) -> RaaError {
    refusal(
        "pipeline.duplicate_pass",
        format!("an entry named `{name}` is already in the pipeline"),
    )
}

/// The message of an instance refusal code.
fn ineligible_message(code: &'static str) -> &'static str {
    match code {
        SKINNED_UNSUPPORTED => "skinned instance: posed skins stay on the forward path",
        DEFORMED_UNSUPPORTED => {
            "deformed instance (squash or jiggle): deformed instances stay on the forward path"
        }
        TRANSPARENT_MATERIAL => "material class transparent: only opaque materials are virtualized",
        _ => "instance is not eligible for the virtualized path",
    }
}

/// The reason the path is inactive, or `None` when it is active (S31 rank table). The first
/// match wins: `disabled` (!enabled), `passDisabled` (a vg entry is disabled), `multiView`
/// (submitted views != 1), `noInstance` (no qualifying instance).
pub fn inactive_reason(
    enabled: bool,
    passes_enabled: bool,
    submitted_views: u8,
    qualifying_instances: u32,
) -> Option<&'static str> {
    if !enabled {
        Some("disabled")
    } else if !passes_enabled {
        Some("passDisabled")
    } else if submitted_views != 1 {
        Some("multiView")
    } else if qualifying_instances == 0 {
        Some("noInstance")
    } else {
        None
    }
}

/// The counters of the S31 milestone-2 table. The cull counters are `None` until the first
/// invocation is published; `pending` is true while a capture or its map is outstanding, and a
/// still-mapped buffer is never reused or waited on.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VgStats {
    pub active: bool,
    pub inactive_reason: Option<&'static str>,
    pub virtual_instances: u32,
    pub forward_instances: u32,
    pub refused: BTreeMap<&'static str, u32>,
    pub baking_meshes: u32,
    pub ready_meshes: u32,
    pub drawn_clusters: Option<u32>,
    pub drawn_triangles: Option<u32>,
    pub hardware_clusters: Option<u32>,
    pub software_clusters: Option<u32>,
    pub shadow_clusters: Option<u32>,
    pub dropped_clusters: Option<u32>,
    /// Main-view hierarchy entries the frustum rejected (instance roots, nodes, clusters).
    pub frustum_culled: Option<u32>,
    /// Main-view entries the first occlusion phase rejected against the previous pyramid.
    pub occlusion_rejected: Option<u32>,
    /// Main-view clusters the second occlusion phase recovered (drawn after all).
    pub occlusion_recovered: Option<u32>,
    pub stats_frame_index: Option<u64>,
    pub pending: bool,
}

/// The current frame's virtualized-geometry state lent by [`VgHandle::with_gi_frame`]: the
/// scene pool, the cull pipelines, the `PassView` slots and the cull scene group.
pub struct VgGiFrame<'a> {
    pub scene: &'a VgScene,
    pub cull: &'a CullPipelines,
    pub slots: &'a ViewSlots,
    /// `VgResources::cull_history`: the cull scene group with the `PassView` slots at binding 0,
    /// the scene instances, nodes and clusters at 1-3, and the previous-frame HZB at 4. The HZB
    /// is never read by an orthographic view, because its `occlusion_enabled` is 0.
    pub cull_scene: &'a wgpu::BindGroup,
}

#[derive(Default)]
struct Latest {
    report: Option<VgSyncReport>,
    reason: Option<&'static str>,
}

/// Everything a clone of [`VgHandle`] shares.
struct HandleState {
    vg: SharedVg,
    // A pipeline pass can retain this handle through the GI voxel driver.
    pipeline: Weak<RefCell<Pipeline>>,
    store: Rc<RefCell<ClusterStore>>,
    requests: RefCell<InstanceRequests>,
    keys: RefCell<MeshKeys>,
    scratch: RefCell<ItemScratch>,
    latest: RefCell<Latest>,
    submitted_views: Cell<u8>,
    /// The GI member tags, sorted and deduplicated; empty until `set_gi_members` changes it.
    gi_members: RefCell<BTreeSet<u32>>,
}

/// The installed virtualized-geometry feature; clones share one state.
#[derive(Clone)]
pub struct VgHandle {
    shared: Rc<HandleState>,
}

impl VgHandle {
    /// Merges `patch` over the current settings. `enabled` true needs a lit msaa of 1
    /// (`vg.invalid`, message `msaa: virtualized geometry needs msaa 1`); a refused patch
    /// changes nothing. A change of `enabled` clears the occlusion history.
    pub fn set(&self, patch: VgSettingsPatch) -> Result<(), RaaError> {
        let (candidate, msaa) = {
            let state = self.shared.vg.try_borrow().map_err(|_| pipeline_busy())?;
            (patch.apply(&state.settings)?, state.lit.msaa())
        };
        if candidate.enabled && msaa != 1 {
            return Err(invalid("msaa", "virtualized geometry needs msaa 1"));
        }
        let mut state = self
            .shared
            .vg
            .try_borrow_mut()
            .map_err(|_| pipeline_busy())?;
        if state.settings.enabled != candidate.enabled {
            state.occlusion_history = false;
        }
        state.settings = candidate;
        Ok(())
    }

    /// The current settings.
    pub fn get(&self) -> VgSettings {
        self.shared
            .vg
            .try_borrow()
            .map(|state| state.settings)
            .unwrap_or_default()
    }

    /// Sets the request of `tag`; `None` follows `instanceDefault`. `Some(true)` on a tag the
    /// latest [`prepare_frame`](Self::prepare_frame) found ineligible refuses with its code and
    /// message and changes nothing.
    pub fn set_instance(&self, tag: u32, request: Option<bool>) -> Result<(), RaaError> {
        if request == Some(true) {
            let latest = self
                .shared
                .latest
                .try_borrow()
                .map_err(|_| pipeline_busy())?;
            if let Some(report) = latest.report.as_ref()
                && let Some(&code) = report.ineligible.get(&tag)
            {
                let message = report
                    .paths
                    .get(&tag)
                    .and_then(|path| match path {
                        InstancePath::Forward { message, .. } => message.clone(),
                        InstancePath::Virtual => None,
                    })
                    .unwrap_or_else(|| ineligible_message(code).to_string());
                return Err(refusal(code, message));
            }
        }
        self.shared
            .requests
            .try_borrow_mut()
            .map_err(|_| pipeline_busy())?
            .set(tag, request);
        Ok(())
    }

    /// The path of `tag` from the latest report; an unknown tag is a non-pending forward path.
    pub fn instance(&self, tag: u32) -> InstancePath {
        self.shared
            .latest
            .try_borrow()
            .ok()
            .and_then(|latest| {
                latest
                    .report
                    .as_ref()
                    .and_then(|report| report.paths.get(&tag).cloned())
            })
            .unwrap_or(InstancePath::Forward {
                pending: false,
                code: None,
                message: None,
            })
    }

    /// The explicit request of `tag`, or `None` when it follows `instanceDefault`.
    pub fn request(&self, tag: u32) -> Option<bool> {
        self.shared
            .requests
            .try_borrow()
            .ok()
            .and_then(|requests| requests.get(tag))
    }

    /// Sets the camera and viewport of the next frame; a viewport change clears the occlusion
    /// history. `submitted_views` is the view count the invocation will submit.
    pub fn set_view(&self, camera: &Camera, viewport: (u32, u32), submitted_views: u8) {
        self.shared.submitted_views.set(submitted_views);
        if let Ok(mut state) = self.shared.vg.try_borrow_mut() {
            if state.viewport != viewport {
                state.occlusion_history = false;
            }
            state.camera = *camera;
            state.viewport = viewport;
        }
    }

    /// Drops the occlusion history after a camera discontinuity.
    pub fn invalidate_history(&self) {
        if let Ok(mut state) = self.shared.vg.try_borrow_mut() {
            state.occlusion_history = false;
            state.previous_view_proj = None;
        }
    }

    /// Maps the lit sub-mesh ids of cluster-store key `key`.
    pub fn set_mesh_key(&self, key: u32, submeshes: &[MeshId]) {
        if let Ok(mut keys) = self.shared.keys.try_borrow_mut() {
            keys.set(key, submeshes);
        }
    }

    /// Registers (`Some`) or removes (`None`) the per-vertex colours of section `ordinal` of
    /// cluster-store key `key` (linear rgba, 4 floats per `ClusterMesh` vertex): a virtualized
    /// pixel's base colour is multiplied by them, as the forward path multiplies by its vertex
    /// colours. A stream whose length differs from the section's vertex count draws white. A
    /// wrong length or a non-finite value is `vg.invalid`; a busy pipeline is `pipeline.busy`.
    pub fn set_cluster_vertex_colors(
        &self,
        key: u32,
        ordinal: usize,
        colors: Option<&[f32]>,
    ) -> Result<(), RaaError> {
        let mut state = self
            .shared
            .vg
            .try_borrow_mut()
            .map_err(|_| pipeline_busy())?;
        state
            .scene
            .set_cluster_colors(key, ordinal, colors)
            .map_err(|reason| invalid("clusterVertexColors", reason))
    }

    /// Registers or removes (both `None`) the shading attributes of section `ordinal` of
    /// cluster-store key `key`: rgba colours (4 floats) and uvs (2 floats) per `ClusterMesh`
    /// vertex, which a virtualized pixel hands its material as `MaterialInput.color` and `.uv`
    /// in place of the vertex colour and the cluster uv, so a material that keeps data in its
    /// colour and uv streams (raa-sdfm's field shading) reads it on the virtualized path too.
    /// A wrong length or a non-finite value is `vg.invalid`; a busy pipeline is `pipeline.busy`.
    pub fn set_cluster_shading(
        &self,
        key: u32,
        ordinal: usize,
        colors: Option<&[f32]>,
        uvs: Option<&[f32]>,
    ) -> Result<(), RaaError> {
        let mut state = self
            .shared
            .vg
            .try_borrow_mut()
            .map_err(|_| pipeline_busy())?;
        state
            .scene
            .set_cluster_shading(key, ordinal, colors, uvs)
            .map_err(|reason| invalid("clusterShading", reason))
    }

    /// Removes the mesh-key mapping of `key`.
    pub fn remove_mesh_key(&self, key: u32) {
        if let Ok(mut keys) = self.shared.keys.try_borrow_mut() {
            keys.remove(key);
        }
    }

    /// Lends the current frame's virtualized-geometry state to `f` for the duration of the
    /// closure only.
    ///
    /// The closure gets `None` while: VG is inactive (the frame state's `active` is false); the
    /// pipelines or `VgResources` are released; the scene is not uploaded (`scene_ready` is
    /// false); or the frame state is already borrowed (`try_borrow` fails). It never creates,
    /// recreates or drops state.
    ///
    /// LIFECYCLE: [`VgHandle::prepare_frame`] runs from the m29 api frame.
    /// crates/raa-js/build.rs:99-107 `frame_all` calls the api files in file-name order, so
    /// m29_virtual_geometry.rs runs before m30_global_illumination.rs. On an active frame it
    /// creates the pipelines and `VgResources` through `VgFrameState::ensure_resources`. On an
    /// inactive frame it drops them through `VgFrameState::release`. `GiHandle::prepare_frame`
    /// runs after it in the same invocation. `gi.volume` (order 25) records after `vg.resolve`
    /// (order 23), because the pipeline records passes in order, so a `with_gi_frame` call from
    /// either place sees this frame's resources.
    ///
    /// Handoff: `raa-gi: driver B part virtual_driver and the voxelize_virtual kernel` calls
    /// `VgHandle::with_gi_frame` and `VgHandle::set_gi_members` exported by
    /// crates/raa-vg/src/feature.rs to borrow this frame's `VgGiFrame` and drive the `Gi` cull
    /// filter from the member set. The `INSTANCE_GI` flag itself comes from
    /// `WithGiMembers::new` (crates/raa-vg/src/scene.rs:232), whose Handoff names this card.
    pub fn with_gi_frame<R>(&self, f: impl FnOnce(Option<VgGiFrame<'_>>) -> R) -> R {
        let Ok(state) = self.shared.vg.try_borrow() else {
            return f(None);
        };
        let (Some(pipelines), Some(resources)) =
            (state.pipelines.as_ref(), state.resources.as_ref())
        else {
            return f(None);
        };
        if !state.active || !state.scene_ready() {
            return f(None);
        }
        f(Some(VgGiFrame {
            scene: &state.scene,
            cull: &pipelines.cull,
            slots: &resources.slots,
            cull_scene: &resources.cull_history,
        }))
    }

    /// Replaces the GI member set with `tags`, held as a sorted, deduplicated
    /// `BTreeSet<u32>` in the handle state. An unchanged set is a no-op; otherwise the next
    /// [`VgHandle::prepare_frame`] plans with
    /// [`WithGiMembers::new`](crate::scene::WithGiMembers::new), so `ids.z & INSTANCE_GI` of
    /// each uploaded `VgInstanceRecord` follows the set from that frame on.
    /// `VgScene::upload_instances` writes only changed bytes. A busy handle state is skipped.
    pub fn set_gi_members(&self, tags: &[u32]) {
        let members: BTreeSet<u32> = tags.iter().copied().collect();
        let Ok(mut gi_members) = self.shared.gi_members.try_borrow_mut() else {
            return;
        };
        if *gi_members == members {
            return;
        }
        *gi_members = members;
    }

    /// Plans the frame's paths, switches the active state and updates the lit `VirtualSet` and
    /// feature ids 27, 28 and 36 (36 only when shadows are on and `shadow.csm` is enabled).
    /// Active frames create the resources,
    /// upload the plan and publish the material functions; inactive frames release every vg
    /// buffer and texture. The report is kept for `instance`, `stats` and `set_instance`.
    ///
    /// Handoff: `raa-vg: cascade integration into vg.cull and vg.raster, feature id 36` takes
    /// `VgHandle::prepare_frame(&self, gpu: &GpuContext, scene: &mut Scene, lit: &mut LitRenderer)
    /// -> VgSyncReport` exported by crates/raa-vg/src/feature.rs; it attaches by reading the
    /// `Option<ShadowHandle>` that `VgFeature::install` stored in the handle's state and setting
    /// feature id 36 from the cascade activity here; its frame_passes.rs part is cited at
    /// `VgCullPass::record` (crates/raa-vg/src/frame_passes.rs). State/keys: the `SharedVg`
    /// frame state (its `VgScene` pool and `settings.shadows`) and the lit feature ids 27, 28
    /// and 36.
    pub fn prepare_frame(
        &self,
        gpu: &GpuContext,
        scene: &mut Scene,
        lit: &mut LitRenderer,
    ) -> VgSyncReport {
        let (settings, shadow_handle) = self
            .shared
            .vg
            .try_borrow()
            .map(|state| (state.settings, state.shadow.is_some()))
            .unwrap_or_default();
        if let Ok(mut state) = self.shared.vg.try_borrow_mut() {
            state.recorded.clear();
        }
        let (passes_enabled, shadow_enabled) = self
            .shared
            .pipeline
            .upgrade()
            .and_then(|pipeline| {
                let pipeline = pipeline.try_borrow().ok()?;
                let entries = pipeline.passes();
                Some((
                    VG_ENTRIES.iter().all(|name| {
                        entries
                            .iter()
                            .any(|info| info.name == *name && info.enabled)
                    }),
                    entries
                        .iter()
                        .any(|info| info.name == "shadow.csm" && info.enabled),
                ))
            })
            .unwrap_or_default();
        scene.update_world();
        let mut plan = self.plan(scene, lit, &settings);
        if !settings.enabled {
            disabled_report(&mut plan.report);
        }
        let reason = inactive_reason(
            settings.enabled,
            passes_enabled,
            self.shared.submitted_views.get(),
            plan.report.virtual_instances,
        );
        let active = reason.is_none();
        if active {
            if let Ok(mut state) = self.shared.vg.try_borrow_mut() {
                state.active = true;
                state.ensure_resources(gpu);
            }
            self.upload(gpu, &plan);
        } else if let Ok(mut state) = self.shared.vg.try_borrow_mut() {
            state.active = false;
            state.release();
            state.previous_view_proj = None;
        }
        lit.virtual_instances_mut().clear();
        if active {
            for &instance in &plan.report.virtual_ids {
                lit.virtual_instances_mut().set(instance, true);
            }
        }
        lit.set_frame_feature(27, active);
        lit.set_frame_feature(28, active);
        lit.set_frame_feature(
            36,
            active && settings.shadows && shadow_handle && shadow_enabled,
        );
        if let Ok(mut latest) = self.shared.latest.try_borrow_mut() {
            latest.report = Some(plan.report.clone());
            latest.reason = reason;
        }
        plan.report
    }

    /// Forwards the invocation and whether it was submitted completely to the readback of the
    /// frame path. The newest completely read invocation is copied to `VgFrameState::published`,
    /// which outlives the resources. The invocation is the id `VgHandle` callers pass from
    /// `finish_scene_invocation`; no vg pass numbers invocations itself.
    pub fn observe_submission(&self, invocation: u64, complete: bool) {
        let Ok(mut state) = self.shared.vg.try_borrow_mut() else {
            return;
        };
        if let Some(resources) = state.resources.as_mut() {
            resources.stats.observe_submission(invocation, complete);
        }
        let latest = state.resources.as_ref().and_then(|r| r.stats.latest());
        if let Some(latest) = latest
            && state
                .published
                .is_none_or(|published| latest.invocation > published.invocation)
        {
            state.published = Some(latest);
        }
    }

    /// The registry counts and reason of the latest report, and the cull counters of the newest
    /// published invocation: `None` before the first publication and kept while inactive.
    pub fn stats(&self) -> VgStats {
        let (report, reason, known) = match self.shared.latest.try_borrow() {
            Ok(latest) => (
                latest.report.clone(),
                latest.reason,
                latest.report.is_some(),
            ),
            Err(_) => (None, None, false),
        };
        // The map callbacks run in the browser's own time, so the newest resolved sample may
        // have arrived since the last `observe_submission`. The getter moves it into the
        // published totals itself and reports whether a capture or its map is still outstanding.
        let (pending, published) = self
            .shared
            .vg
            .try_borrow_mut()
            .map(|mut state| {
                let latest = state
                    .resources
                    .as_ref()
                    .and_then(|resources| resources.stats.latest());
                if let Some(latest) = latest
                    && state
                        .published
                        .is_none_or(|published| latest.invocation > published.invocation)
                {
                    state.published = Some(latest);
                }
                let pending = state
                    .resources
                    .as_ref()
                    .is_some_and(|resources| resources.stats.pending());
                (pending, state.published)
            })
            .unwrap_or_default();
        let main = published.map(|published| published.main);
        VgStats {
            active: known && reason.is_none(),
            inactive_reason: if known { reason } else { Some("disabled") },
            virtual_instances: report.as_ref().map_or(0, |report| report.virtual_instances),
            forward_instances: report.as_ref().map_or(0, |report| report.forward_instances),
            refused: report
                .as_ref()
                .map(|report| report.refused.clone())
                .unwrap_or_default(),
            baking_meshes: report.as_ref().map_or(0, |report| report.baking_meshes),
            ready_meshes: report.as_ref().map_or(0, |report| report.ready_meshes),
            drawn_clusters: main.map(|main| main.clusters),
            drawn_triangles: main.map(|main| main.triangles),
            hardware_clusters: main.map(|main| main.clusters.saturating_sub(main.software)),
            software_clusters: main.map(|main| main.software),
            shadow_clusters: published.map(|published| published.shadow_clusters),
            dropped_clusters: published.map(|published| published.dropped),
            frustum_culled: main.map(|main| main.frustum_culled),
            occlusion_rejected: main.map(|main| main.occlusion_rejected),
            occlusion_recovered: main.map(|main| main.occlusion_recovered),
            stats_frame_index: published.map(|published| published.invocation),
            pending,
        }
    }

    /// The pool upload counters of the frame state's scene (bytes handed to the queue).
    pub fn pool_traffic(&self) -> crate::scene::VgPoolTraffic {
        self.shared
            .vg
            .try_borrow()
            .map(|state| state.scene.traffic())
            .unwrap_or_default()
    }

    /// Every vg buffer and texture the frame state owns; 0 means none exists (P3-Q8).
    pub fn allocated_bytes(&self) -> u64 {
        self.shared
            .vg
            .try_borrow()
            .map(|state| state.allocated_bytes())
            .unwrap_or(0)
    }

    /// The names recorded in the last frame, in order.
    pub fn recorded_passes(&self) -> Vec<&'static str> {
        self.shared
            .vg
            .try_borrow()
            .map(|state| state.recorded.clone())
            .unwrap_or_default()
    }

    fn plan(&self, scene: &mut Scene, lit: &LitRenderer, settings: &VgSettings) -> VgPlan {
        let Ok(mut scratch) = self.shared.scratch.try_borrow_mut() else {
            return VgPlan::default();
        };
        let Ok(store) = self.shared.store.try_borrow() else {
            return VgPlan::default();
        };
        let Ok(requests) = self.shared.requests.try_borrow() else {
            return VgPlan::default();
        };
        let Ok(keys) = self.shared.keys.try_borrow() else {
            return VgPlan::default();
        };
        let Ok(members) = self.shared.gi_members.try_borrow() else {
            return VgPlan::default();
        };
        let items = scratch.fill(scene);
        let lit_handle = lit.handle();
        let source = WithGiMembers::new(&lit_handle, &members);
        plan_sync(items, &source, &store, &requests, &keys, settings)
    }

    fn upload(&self, gpu: &GpuContext, plan: &VgPlan) {
        let Ok(store) = self.shared.store.try_borrow() else {
            return;
        };
        let Ok(keys) = self.shared.keys.try_borrow() else {
            return;
        };
        let Ok(mut state) = self.shared.vg.try_borrow_mut() else {
            return;
        };
        state.scene.upload_plan(gpu, &store, &keys, plan);
        state.material_fns = plan.report.material_fns.clone();
    }
}

/// Rewrites the plan of a frame whose feature is disabled (`enabled` false) into what that frame
/// submits: every instance draws on forward, so a `Virtual` path becomes a non-pending forward
/// path, a pending one is no longer pending, and no runtime bake is requested. Refusal codes and
/// `ineligible` are kept (`set_instance` still refuses an ineligible request). Without this the
/// disabled engine reported `virtual` for every baked default-requested instance and kept
/// starting runtime bakes it never drew (SDF Modeler's Game view, setting off, never ready).
fn disabled_report(report: &mut VgSyncReport) {
    for path in report.paths.values_mut() {
        match path {
            InstancePath::Virtual => {
                *path = InstancePath::Forward {
                    pending: false,
                    code: None,
                    message: None,
                }
            }
            InstancePath::Forward { pending, .. } => *pending = false,
        }
    }
    report.forward_instances += report.virtual_instances;
    report.virtual_instances = 0;
    report.virtual_ids.clear();
    report.material_fns.clear();
    report.bake_requests.clear();
}

/// Installs the five virtualized-geometry entries of S12 on a pipeline and its lit renderer:
/// the four feature passes `vg.cull`, `vg.raster`, `vg.resolve` and `vg.gbuffer` and the
/// forward contributor `vg.shade`, all left disabled.
///
/// Handoff: `raa-vg: feature tests (msaa refusal, disabled state, S12 recording order)` takes
/// `VgFeature::install(gpu: &GpuContext, pipeline: &Rc<RefCell<Pipeline>>, lit: &mut LitRenderer,
/// store: Rc<RefCell<ClusterStore>>, shadow: Option<ShadowHandle>) -> Result<VgHandle, RaaError>`
/// exported by crates/raa-vg/src/feature.rs; it attaches by calling it from
/// crates/raa-vg/tests/feature.rs and driving `VgHandle::set`, `VgHandle::set_view`,
/// `VgHandle::prepare_frame` and `VgHandle::recorded_passes`. State/keys: the `SharedVg` frame
/// state, the five `Pipeline` entries, and the `ClusterStore`/`InstanceRequests`/`MeshKeys`
/// registry the handle owns.
pub struct VgFeature;

impl VgFeature {
    /// Adds `vg.shade` through [`LitRenderer::add_contributor`](raa_lit::LitRenderer) and the
    /// four passes through [`Pipeline::add`](raa_pipeline::Pipeline), stores the optional
    /// `shadow.csm` handle in the shared frame state, then disables all five.
    /// Refusals leave nothing installed: `pipeline.busy` when the pipeline is borrowed,
    /// `pipeline.duplicate_pass` when one of the five names already exists.
    pub fn install(
        gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
        lit: &mut LitRenderer,
        store: Rc<RefCell<ClusterStore>>,
        shadow: Option<ShadowHandle>,
    ) -> Result<VgHandle, RaaError> {
        let _ = gpu;
        {
            let p = pipeline.try_borrow().map_err(|_| pipeline_busy())?;
            let entries = p.passes();
            for name in VG_ENTRIES {
                if entries.iter().any(|info| info.name == name) {
                    return Err(duplicate(name));
                }
            }
        }
        let vg: SharedVg = Rc::new(RefCell::new(VgFrameState::new(lit.handle())));
        if let Ok(mut state) = vg.try_borrow_mut() {
            state.shadow = shadow.clone();
        }
        lit.add_contributor(Box::new(VgShadeContributor::new(Rc::clone(&vg))))?;
        {
            let mut p = pipeline.try_borrow_mut().map_err(|_| pipeline_busy())?;
            p.add(Box::new(VgCullPass::new(Rc::clone(&vg))))?;
            p.add(Box::new(VgRasterPass::new(Rc::clone(&vg))))?;
            p.add(Box::new(VgResolvePass::new(Rc::clone(&vg))))?;
            p.add(Box::new(VgGbufferPass::new(Rc::clone(&vg))))?;
        }
        {
            let mut p = pipeline.try_borrow_mut().map_err(|_| pipeline_busy())?;
            for name in VG_ENTRIES {
                p.set_enabled(name, false)?;
            }
        }
        Ok(VgHandle {
            shared: Rc::new(HandleState {
                vg,
                pipeline: Rc::downgrade(pipeline),
                store,
                requests: RefCell::new(InstanceRequests::default()),
                keys: RefCell::new(MeshKeys::default()),
                scratch: RefCell::new(ItemScratch::default()),
                latest: RefCell::new(Latest::default()),
                submitted_views: Cell::new(1),
                gi_members: RefCell::new(BTreeSet::new()),
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The S31 rank table over all sixteen combinations of the four binary inputs: the first
    /// match wins (`disabled`, `passDisabled`, `multiView`, `noInstance`), otherwise active.
    #[test]
    fn inactive_reason_rank_table() {
        let rows: [(bool, bool, u8, u32, Option<&'static str>); 16] = [
            (false, false, 0, 0, Some("disabled")),
            (false, false, 0, 1, Some("disabled")),
            (false, false, 1, 0, Some("disabled")),
            (false, false, 1, 1, Some("disabled")),
            (false, true, 0, 0, Some("disabled")),
            (false, true, 0, 1, Some("disabled")),
            (false, true, 1, 0, Some("disabled")),
            (false, true, 1, 1, Some("disabled")),
            (true, false, 0, 0, Some("passDisabled")),
            (true, false, 0, 1, Some("passDisabled")),
            (true, false, 1, 0, Some("passDisabled")),
            (true, false, 1, 1, Some("passDisabled")),
            (true, true, 0, 0, Some("multiView")),
            (true, true, 0, 1, Some("multiView")),
            (true, true, 1, 0, Some("noInstance")),
            (true, true, 1, 1, None),
        ];
        for (enabled, passes, views, instances, expected) in rows {
            assert_eq!(
                inactive_reason(enabled, passes, views, instances),
                expected,
                "enabled={enabled} passes={passes} views={views} instances={instances}"
            );
        }
    }

    #[test]
    fn a_disabled_frame_reports_every_instance_on_forward_and_requests_no_bake() {
        use crate::scene::VgSyncReport;
        let mut report = VgSyncReport {
            virtual_instances: 1,
            forward_instances: 2,
            bake_requests: vec![4],
            ..VgSyncReport::default()
        };
        report.paths.insert(1, InstancePath::Virtual);
        report.paths.insert(
            2,
            InstancePath::Forward {
                pending: true,
                code: None,
                message: None,
            },
        );
        report.paths.insert(
            3,
            InstancePath::Forward {
                pending: false,
                code: Some("vg.skinned_unsupported"),
                message: Some("skinned".to_string()),
            },
        );
        disabled_report(&mut report);
        let forward = InstancePath::Forward {
            pending: false,
            code: None,
            message: None,
        };
        assert_eq!(
            report.paths[&1], forward,
            "a baked instance is not virtual while disabled"
        );
        assert_eq!(
            report.paths[&2], forward,
            "a disabled frame bakes nothing, so nothing is pending"
        );
        assert_eq!(
            report.paths[&3],
            InstancePath::Forward {
                pending: false,
                code: Some("vg.skinned_unsupported"),
                message: Some("skinned".to_string()),
            },
            "a refusal keeps its code"
        );
        assert_eq!((report.virtual_instances, report.forward_instances), (0, 3));
        assert!(report.bake_requests.is_empty() && report.virtual_ids.is_empty());
    }
}
