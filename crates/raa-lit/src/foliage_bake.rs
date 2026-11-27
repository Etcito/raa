//! The M7 foliage bake coordinator (`m7-bake-service`): routes one validated preparation to
//! its complete producer, initializes one immutable [`FoliageResources`] generation, waits for
//! every submission it depends on and only then hands the result to
//! [`crate::LitRenderer::prepare_foliage`] for atomic publication.
//!
//! Whole-tree impostor modes (spom, pom, offset, shell) upload the actual convex hull of
//! `build_foliage_hull` and initialize the arrays with the eight packed reference layers of
//! `pack_foliage_layers`; stochastic and differentiable modes upload the distilled mesh of
//! `build_stochastic_foliage` / `optimize_foliage` and initialize only the native volume.
//! Every mode writes the same native source-albedo/occlusion volume. Partial allocations are
//! plain owned values of the returned future, so a failure or a dropped future releases them
//! (wgpu keeps submitted resources alive until their work completes). Source draws are never
//! touched: they stay visible until the caller swaps to the published asset.

use std::cell::RefCell;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use raa_core::{GpuContext, MaterialId, RaaError};
use raa_geom::MeshData;
use raa_mesh::AttributedMesh;

use crate::LitHandle;
use crate::foliage_appearance::sample_foliage_albedo;
use crate::foliage_hull::build_foliage_hull;
use crate::foliage_optimize::{
    FoliageOptimizeSettings, optimize_foliage, validate_reference_count,
};
use crate::foliage_reference::{pack_foliage_layers, render_foliage_references};
use crate::foliage_resources::{
    FOLIAGE_MODE_DIFFERENTIABLE, FOLIAGE_MODE_SHELL, FOLIAGE_MODE_STOCHASTIC, FoliageParams,
    FoliageResources, FoliageSource,
};
use crate::foliage_sampling::{FoliageSample, sample_foliage};
use crate::foliage_stochastic::build_stochastic_foliage;
use crate::foliage_volume::FoliageVolumeAccumulator;
use crate::handle::LitShared;
use crate::meshes::LitMeshId;

/// Magic word 0 of the '97. M7 foliage material data' envelope.
pub const FOLIAGE_DATA_MAGIC: u32 = 0x4d37_464c;
/// Version word 1 of the envelope.
pub const FOLIAGE_DATA_VERSION: u32 = 1;
/// Header words before the compiled graph constants.
pub const FOLIAGE_DATA_HEADER_WORDS: usize = 4;
/// Defined authored-channel mask bits (base_color, normal, ao, opacity).
pub const FOLIAGE_DATA_MASK: u32 = 0xf;
/// Reference views of whole-tree impostor preparation.
pub const IMPOSTOR_VIEWS: u32 = 8;

/// The settings of one preparation; all fields are validated before any allocation.
///
/// `material` is the caller-owned compiled LOD1 surface material (FoliageDesc.material),
/// carried unchanged into [`FoliageAsset::material`]; it is never a source triangle material.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FoliageBakeSettings {
    pub params: FoliageParams,
    pub material: MaterialId,
    pub count: u32,
    pub seed: u32,
    pub iterations: u32,
    pub silhouette_weight: f32,
    pub normal_weight: f32,
    pub edge_weight: f32,
}

/// One published foliage generation. It owns its proxy mesh and the resource association of
/// `material`; the material and its data block stay borrowed caller leases.
pub struct FoliageAsset {
    pub resources: Rc<FoliageResources>,
    pub mesh: LitMeshId,
    pub material: MaterialId,
    pub generation: u32,
    pub triangle_count: u32,
    pub mode: u32,
    pub volume_resolution: u32,
}

/// The producer a mode rank routes to.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FoliageRoute {
    /// spom, pom, offset, shell: convex hull plus packed reference layers.
    Impostor,
    /// Stochastic area-compensated distillation.
    Stochastic,
    /// Differentiable G-buffer distillation.
    Differentiable,
}

pub(crate) fn invalid(message: impl Into<String>) -> RaaError {
    RaaError::FrameGraph {
        code: "foliage.invalid",
        message: message.into(),
    }
}

/// The route of mode rank `mode` (m7-foliage-resources ranks 0..=5), else `foliage.invalid`.
pub fn foliage_route(mode: u32) -> Result<FoliageRoute, RaaError> {
    match mode {
        m if m <= FOLIAGE_MODE_SHELL => Ok(FoliageRoute::Impostor),
        FOLIAGE_MODE_STOCHASTIC => Ok(FoliageRoute::Stochastic),
        FOLIAGE_MODE_DIFFERENTIABLE => Ok(FoliageRoute::Differentiable),
        m => Err(invalid(format!("foliage mode {m} is not a known rank"))),
    }
}

impl FoliageBakeSettings {
    /// Validates every field and returns the route. Impostors need exactly
    /// [`IMPOSTOR_VIEWS`] views, the differentiable method 64..=128.
    pub fn validate(&self) -> Result<FoliageRoute, RaaError> {
        self.params.validate()?;
        let route = foliage_route(self.params.mode)?;
        if self.count == 0 || self.iterations == 0 {
            return Err(invalid("foliage count and iterations must be positive"));
        }
        let weights = [self.silhouette_weight, self.normal_weight, self.edge_weight];
        if weights.iter().any(|w| !w.is_finite() || *w < 0.0) || weights.iter().all(|w| *w == 0.0) {
            return Err(invalid(
                "foliage weights must be finite, nonnegative and not all zero",
            ));
        }
        match route {
            FoliageRoute::Impostor if self.params.view_count != IMPOSTOR_VIEWS => Err(invalid(
                format!("impostor foliage needs {IMPOSTOR_VIEWS} reference views"),
            )),
            FoliageRoute::Differentiable => {
                validate_reference_count(self.params.view_count).map(|()| route)
            }
            _ => Ok(route),
        }
    }

    /// The optimizer settings of the differentiable route.
    pub fn optimize_settings(&self) -> FoliageOptimizeSettings {
        FoliageOptimizeSettings {
            count: self.count,
            seed: self.seed,
            resolution: self.params.resolution,
            iterations: self.iterations,
            silhouette_weight: self.silhouette_weight,
            normal_weight: self.normal_weight,
            edge_weight: self.edge_weight,
        }
    }
}

/// Validates the M7 material-data envelope of the supplied material's data block (`None`
/// when the block is not live) and returns its authored-channel mask.
pub fn validate_material_data(words: Option<&[u32]>) -> Result<u32, RaaError> {
    let words = words.ok_or_else(|| invalid("foliage material has no live material data"))?;
    if words.len() < FOLIAGE_DATA_HEADER_WORDS {
        return Err(invalid("foliage material data is shorter than its header"));
    }
    if words[0] != FOLIAGE_DATA_MAGIC || words[1] != FOLIAGE_DATA_VERSION {
        return Err(invalid(
            "foliage material data has a wrong magic or version",
        ));
    }
    if words[2] & !FOLIAGE_DATA_MASK != 0 {
        return Err(invalid("foliage material data sets reserved mask bits"));
    }
    if words[3] as usize > words.len() - FOLIAGE_DATA_HEADER_WORDS {
        return Err(invalid(
            "foliage material data graph count exceeds its block",
        ));
    }
    Ok(words[2])
}

/// Refuses an empty source and a source whose generation differs from `params.generation`.
pub fn validate_source(source: &FoliageSource, params: &FoliageParams) -> Result<(), RaaError> {
    if source.triangles.is_empty() {
        return Err(invalid("foliage source is empty"));
    }
    if source.generation != params.generation {
        return Err(invalid(format!(
            "foliage source generation {} differs from the requested {}",
            source.generation, params.generation
        )));
    }
    Ok(())
}

/// Host bytes the impostor route allocates per eligible source triangle besides the hull:
/// the sampling table of `FoliageSurface::new` (`eligible: u32` and `cumulative: f64`).
const SAMPLING_BYTES_PER_TRIANGLE: usize = 4 + 8;

/// The hull's distinct-position set for `triangles` source triangles in the worst case
/// (every corner a distinct position): the final table plus the previous one, live
/// together while the last doubling rehashes. `None` when the size overflows.
fn hull_worst_bytes(triangles: usize) -> Option<usize> {
    let set = crate::foliage_hull::hull_set_bytes(triangles.checked_mul(3)?)?;
    set.checked_add(set / 2)
}

/// Checks that the impostor route can allocate its source-proportional host buffers for
/// `triangles` source triangles and returns their byte total: the hull's distinct-position
/// set in the worst case ([`crate::foliage_hull::hull_set_bytes`] of every corner, plus
/// the previous table during its last doubling) and the sampling table (12 B per
/// triangle). The source snapshot itself is already allocated by the caller; the hull no
/// longer copies corners, and the reference views keep one instance record per draw run
/// and encode the vertex streams one window at a time. An overflowing size or a refused
/// allocation (probed with `Vec::try_reserve_exact` at the size of the largest one, the
/// hull's final table, never an abort) is `foliage.invalid` naming the byte count.
/// Successful bakes are unchanged by the check.
///
/// For the 2 609 340-triangle 41-part sample tree: hull set at most 67 108 864 B (plus
/// 33 554 432 B while it grows), sampling table 31 312 080 B, 131 975 376 B in total, next
/// to the 281 808 720 B snapshot (108 B per `SourceTriangle`). Before this card the hull
/// reserved 187 872 480 B of points and 218 103 812 B of position keys.
pub fn check_impostor_allocation(triangles: usize) -> Result<usize, RaaError> {
    let too_large = || {
        invalid(format!(
            "foliage impostor bake of {triangles} triangle(s) needs more than {} bytes",
            usize::MAX
        ))
    };
    let largest = triangles
        .checked_mul(3)
        .and_then(crate::foliage_hull::hull_set_bytes)
        .ok_or_else(too_large)?;
    let bytes = hull_worst_bytes(triangles)
        .zip(triangles.checked_mul(SAMPLING_BYTES_PER_TRIANGLE))
        .and_then(|(hull, sampling)| hull.checked_add(sampling))
        .ok_or_else(too_large)?;
    let mut probe: Vec<u8> = Vec::new();
    probe.try_reserve_exact(largest).map_err(|_| {
        invalid(format!(
            "foliage impostor bake could not allocate {largest} bytes ({bytes} in total)"
        ))
    })?;
    Ok(bytes)
}

/// One stage boundary of the foliage preparation path (`foliage:tree:prepare`), reported
/// before the stage allocates: its name, the host bytes it is about to request (computed
/// from the sizes the stage reserves; 0 for a GPU-only stage) and the source triangle
/// count it works on.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FoliageStage {
    pub stage: &'static str,
    pub requested_bytes: usize,
    pub source_triangles: usize,
}

static STAGE_REPORTER: std::sync::OnceLock<fn(&FoliageStage)> = std::sync::OnceLock::new();

/// Installs the receiver of every [`report_foliage_stage`] call; the first call wins and
/// later calls return `false`. raa-js installs one at module start that adds the live
/// allocated bytes and the wasm memory size and writes the line to `console.error`.
/// Without a receiver (native tests) reports are dropped.
pub fn set_foliage_stage_reporter(reporter: fn(&FoliageStage)) -> bool {
    STAGE_REPORTER.set(reporter).is_ok()
}

/// Reports one stage boundary to the installed receiver, if any; never allocates itself.
pub fn report_foliage_stage(stage: &'static str, requested_bytes: usize, source_triangles: usize) {
    if let Some(reporter) = STAGE_REPORTER.get() {
        reporter(&FoliageStage {
            stage,
            requested_bytes,
            source_triangles,
        });
    }
}

/// Host bytes the reference views request: the vertex stream windows of
/// `encode_reference_streams` (stream 0 and stream 1 of one window).
const REFERENCE_WINDOW_BYTES: usize = crate::foliage_reference::REFERENCE_STREAM_CHUNK_TRIANGLES
    * 3
    * (crate::foliage_reference::REFERENCE_STREAM0_CORNER_BYTES
        + crate::foliage_reference::REFERENCE_STREAM1_CORNER_BYTES);

/// Shared between the renderer and its pending preparations: the latest requested
/// generation per material and the proxy meshes published since the last frame.
#[derive(Default)]
pub struct FoliageBakeState {
    latest: HashMap<MaterialId, u32>,
    published: Vec<LitMeshId>,
}

impl FoliageBakeState {
    /// Records `generation` as the newest request of `material`; older ones become stale.
    pub fn begin(&mut self, material: MaterialId, generation: u32) {
        self.latest.insert(material, generation);
    }

    /// Whether `generation` is still the newest request of `material`.
    pub fn is_current(&self, material: MaterialId, generation: u32) -> bool {
        self.latest.get(&material) == Some(&generation)
    }

    /// Records a proxy mesh uploaded by a preparation.
    pub fn publish(&mut self, mesh: LitMeshId) {
        self.published.push(mesh);
    }

    /// The proxy meshes published since the last call.
    pub fn take_published(&mut self) -> Vec<LitMeshId> {
        std::mem::take(&mut self.published)
    }
}

/// Runs `release` when dropped unless [`Publication::complete`] ran first: the cleanup of a
/// partial publication on failure or cancellation.
pub struct Publication<F: FnOnce()> {
    release: Option<F>,
}

impl<F: FnOnce()> Publication<F> {
    pub fn new(release: F) -> Self {
        Publication {
            release: Some(release),
        }
    }

    /// Keeps the guarded allocation: publication succeeded.
    pub fn complete(mut self) {
        self.release = None;
    }
}

impl<F: FnOnce()> Drop for Publication<F> {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            release();
        }
    }
}

/// Retires a published generation in the required order: `detach` clears the material
/// association, `remove_mesh` frees the proxy mesh, then the last asset reference drops.
pub fn retire_foliage<R>(resources: Rc<R>, detach: impl FnOnce(), remove_mesh: impl FnOnce()) {
    detach();
    remove_mesh();
    drop(resources);
}

#[derive(Default)]
struct DoneState {
    done: bool,
    waker: Option<Waker>,
}

/// Resolves once every submission made before its creation has completed on the GPU.
struct SubmissionsDone {
    #[cfg(not(target_arch = "wasm32"))]
    gpu: GpuContext,
    state: Arc<Mutex<DoneState>>,
}

fn submissions_done(gpu: &GpuContext) -> SubmissionsDone {
    let state = Arc::new(Mutex::new(DoneState::default()));
    let signal = Arc::clone(&state);
    gpu.queue().on_submitted_work_done(move || {
        let mut s = signal.lock().unwrap_or_else(|e| e.into_inner());
        s.done = true;
        if let Some(waker) = s.waker.take() {
            waker.wake();
        }
    });
    SubmissionsDone {
        #[cfg(not(target_arch = "wasm32"))]
        gpu: gpu.clone(),
        state,
    }
}

impl Future for SubmissionsDone {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        let mut state = this.state.lock().unwrap_or_else(|e| e.into_inner());
        if !state.done {
            state.waker = Some(cx.waker().clone());
            drop(state);
            #[cfg(not(target_arch = "wasm32"))]
            let _ = this.gpu.device().poll(wgpu::PollType::wait_indefinitely());
            state = this.state.lock().unwrap_or_else(|e| e.into_inner());
            if !state.done {
                return Poll::Pending;
            }
        }
        Poll::Ready(())
    }
}

/// One completed, initialized generation that is not yet published.
pub(crate) struct CompletedFoliage {
    pub(crate) resources: FoliageResources,
    pub(crate) mesh: MeshData,
}

/// The samples of `source` and their unlit source albedo, reporting both stages.
async fn sampled_appearance(
    gpu: &GpuContext,
    lit: &LitHandle,
    source: &FoliageSource,
    settings: &FoliageBakeSettings,
) -> Result<(Vec<FoliageSample>, Vec<[f32; 3]>), RaaError> {
    let triangles = source.triangles.len();
    let count = settings.count as usize;
    report_foliage_stage(
        "sampling",
        triangles
            .saturating_mul(SAMPLING_BYTES_PER_TRIANGLE)
            .saturating_add(count.saturating_mul(std::mem::size_of::<FoliageSample>())),
        triangles,
    );
    let samples = sample_foliage(source, settings.count, settings.seed)?;
    report_foliage_stage(
        "source appearance",
        samples
            .len()
            .saturating_mul(crate::foliage_appearance::SAMPLE_INPUT_STRIDE),
        triangles,
    );
    let albedo = sample_foliage_albedo(gpu, lit, source, &samples).await?;
    Ok((samples, albedo))
}

/// Host bytes of the volume accumulator (`FoliageVolumeAccumulator`: one `f64` area and
/// three `f64` weighted albedo channels per cell) and its finished 4-byte cells.
fn volume_bytes(resolution: u32) -> usize {
    (resolution as usize)
        .saturating_pow(3)
        .saturating_mul(4 * 8 + 4)
}

/// The native source-albedo/occlusion volume and stochastic mesh of `source`.
async fn stochastic_bake(
    gpu: &GpuContext,
    lit: &LitHandle,
    source: &FoliageSource,
    settings: &FoliageBakeSettings,
) -> Result<crate::foliage_stochastic::FoliageMeshBake, RaaError> {
    let (samples, albedo) = sampled_appearance(gpu, lit, source, settings).await?;
    let wood = source.triangles.iter().filter(|t| !t.foliage).count();
    // `build_stochastic_foliage` reserves three corners (position 12 B, normal 12 B,
    // index 4 B) per wood triangle and per sample.
    report_foliage_stage(
        "stochastic assembly",
        wood.saturating_add(samples.len())
            .saturating_mul(3 * (12 + 12 + 4))
            .saturating_add(volume_bytes(settings.params.resolution)),
        source.triangles.len(),
    );
    build_stochastic_foliage(source, &samples, &albedo, settings.params.resolution)
}

/// The native source-albedo/occlusion volume alone, for the impostor routes: the same
/// samples, source albedo and volume accumulation as [`stochastic_bake`] (identical bytes),
/// without the distilled mesh `build_stochastic_foliage` also builds. That mesh copies
/// every wood triangle (three corners of 28 B, 84 B per triangle) and the impostor route
/// dropped it unused: 174 251 616 B if the sample tree's wood is its 2 074 424 non-leaf
/// triangles (2 609 340 in all, 534 916 leaf triangles per the procgen evaluation quoted
/// in `tests/foliage_memory.rs`).
async fn impostor_volume(
    gpu: &GpuContext,
    lit: &LitHandle,
    source: &FoliageSource,
    settings: &FoliageBakeSettings,
) -> Result<Vec<u8>, RaaError> {
    let (samples, albedo) = sampled_appearance(gpu, lit, source, settings).await?;
    report_foliage_stage(
        "impostor volume",
        volume_bytes(settings.params.resolution),
        source.triangles.len(),
    );
    if samples.len() != albedo.len() {
        return Err(invalid(format!(
            "foliage albedo length {} does not match sample count {}",
            albedo.len(),
            samples.len()
        )));
    }
    let mut volume =
        FoliageVolumeAccumulator::new(source.min, source.max, settings.params.resolution)?;
    for (sample, albedo) in samples.iter().zip(&albedo) {
        volume.add(sample.position, *albedo, sample.represented_area)?;
    }
    Ok(volume.finish())
}

/// Produces, initializes and awaits one generation for an already validated request. Holds
/// no renderer borrow across an await; every partial allocation is owned by this future.
pub(crate) async fn bake_foliage(
    gpu: GpuContext,
    lit: LitHandle,
    source: FoliageSource,
    settings: FoliageBakeSettings,
    route: FoliageRoute,
) -> Result<CompletedFoliage, RaaError> {
    let params = settings.params;
    let triangles = source.triangles.len();
    let (mesh, volume, references, layers) = match route {
        FoliageRoute::Impostor => {
            let probe = triangles
                .checked_mul(3)
                .and_then(crate::foliage_hull::hull_set_bytes)
                .unwrap_or(usize::MAX);
            report_foliage_stage("impostor pre-check", probe, triangles);
            check_impostor_allocation(triangles)?;
            report_foliage_stage(
                "hull",
                hull_worst_bytes(triangles).unwrap_or(usize::MAX),
                triangles,
            );
            let hull = build_foliage_hull(&source)?;
            report_foliage_stage("reference views", REFERENCE_WINDOW_BYTES, triangles);
            let references = render_foliage_references(
                &gpu,
                &lit,
                &source,
                params.view_count,
                params.texture_resolution,
            )?;
            report_foliage_stage("pack layers", 0, triangles);
            let layers = pack_foliage_layers(&gpu, &source, &references)?;
            let volume = impostor_volume(&gpu, &lit, &source, &settings).await?;
            (hull, volume, Some(references), Some(layers))
        }
        FoliageRoute::Stochastic => {
            let bake = stochastic_bake(&gpu, &lit, &source, &settings).await?;
            (bake.mesh, bake.volume, None, None)
        }
        FoliageRoute::Differentiable => {
            report_foliage_stage("reference views", REFERENCE_WINDOW_BYTES, triangles);
            let references = render_foliage_references(
                &gpu,
                &lit,
                &source,
                params.view_count,
                params.texture_resolution,
            )?;
            report_foliage_stage("optimization", 0, triangles);
            let bake =
                optimize_foliage(&gpu, &lit, source, references, settings.optimize_settings())
                    .await?;
            (bake.mesh, bake.volume, None, None)
        }
    };
    report_foliage_stage("resources", volume.len(), triangles);
    let mut resources = FoliageResources::new(&gpu, params)?;
    resources.initialize(&gpu, layers.as_ref(), &volume)?;
    // The packing and initialization submissions precede this marker.
    submissions_done(&gpu).await;
    drop((references, layers));
    report_foliage_stage("bake complete", 0, triangles);
    Ok(CompletedFoliage { resources, mesh })
}

/// Installs a completed generation for a current request: uploads the colourless proxy
/// mesh, attaches the immutable resources to `settings.material` and records the mesh for
/// the renderer's bounds refresh. A stale generation or any failure releases everything.
pub(crate) fn publish_foliage(
    gpu: &GpuContext,
    shared: &Rc<RefCell<LitShared>>,
    lit: &LitHandle,
    state: &Rc<RefCell<FoliageBakeState>>,
    settings: &FoliageBakeSettings,
    completed: CompletedFoliage,
) -> Result<FoliageAsset, RaaError> {
    let params = settings.params;
    let busy = || invalid("the foliage bake state is borrowed during publication");
    if !state
        .try_borrow()
        .map_err(|_| busy())?
        .is_current(settings.material, params.generation)
    {
        return Err(invalid(format!(
            "foliage generation {} was superseded before completion",
            params.generation
        )));
    }
    let triangle_count = u32::try_from(completed.mesh.indices.len() / 3)
        .map_err(|_| invalid("foliage proxy mesh is too large"))?;
    let mesh = {
        let mut s = shared
            .try_borrow_mut()
            .map_err(|_| crate::renderer::pipeline_busy())?;
        check_material(&s, settings.material)?;
        let id = s
            .meshes
            .upload(gpu, &AttributedMesh::from_mesh(completed.mesh))?;
        s.set_mesh_colors(id, false);
        id
    };
    let guard = Publication::new(|| {
        if let Ok(mut s) = shared.try_borrow_mut() {
            s.meshes.remove(mesh);
            s.remove_mesh_colors(mesh);
        }
    });
    let resources = Rc::new(completed.resources);
    let mut published = state.try_borrow_mut().map_err(|_| busy())?;
    lit.set_foliage(settings.material, Some(Rc::clone(&resources)))?;
    guard.complete();
    published.publish(mesh);
    drop(published);
    Ok(FoliageAsset {
        resources,
        mesh,
        material: settings.material,
        generation: params.generation,
        triangle_count,
        mode: params.mode,
        volume_resolution: params.resolution,
    })
}

/// Refuses a material that is not live or whose data block lacks the M7 envelope.
pub(crate) fn check_material(s: &LitShared, material: MaterialId) -> Result<(), RaaError> {
    let m = s
        .materials
        .get(material)
        .ok_or_else(|| invalid(format!("foliage material {} is not live", material.0)))?;
    validate_material_data(s.material_data().block(m.data)).map(|_| ())
}
