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
use crate::foliage_sampling::sample_foliage;
use crate::foliage_stochastic::build_stochastic_foliage;
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

/// Host bytes per source triangle on the impostor route that still scale with the source.
/// The reference views keep one instance record per draw run and encode the vertex streams
/// one window at a time, so the per-triangle share is the hull's working set: three corner
/// points (`3 x 24` B, `f64`) and three position keys (`3 x 12` B).
const IMPOSTOR_BYTES_PER_TRIANGLE: usize = 3 * 24 + 3 * 12;

/// Checks that the impostor route can allocate its source-proportional host buffers for
/// `triangles` source triangles and returns their byte total. An overflowing size or a
/// refused allocation (probed with `Vec::try_reserve_exact` at the size of the largest
/// one, the hull's corner points, never an abort) is `foliage.invalid` naming the byte
/// count. Successful bakes are unchanged by the check.
pub fn check_impostor_allocation(triangles: usize) -> Result<usize, RaaError> {
    let bytes = triangles
        .checked_mul(IMPOSTOR_BYTES_PER_TRIANGLE)
        .ok_or_else(|| {
            invalid(format!(
                "foliage impostor bake of {triangles} triangle(s) needs more than {} bytes",
                usize::MAX
            ))
        })?;
    let largest = triangles.saturating_mul(3 * 24);
    let mut probe: Vec<u8> = Vec::new();
    probe.try_reserve_exact(largest).map_err(|_| {
        invalid(format!(
            "foliage impostor bake could not allocate {largest} bytes ({bytes} in total)"
        ))
    })?;
    Ok(bytes)
}

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

/// The native source-albedo/occlusion volume (and stochastic mesh) of `source`.
async fn stochastic_bake(
    gpu: &GpuContext,
    lit: &LitHandle,
    source: &FoliageSource,
    settings: &FoliageBakeSettings,
) -> Result<crate::foliage_stochastic::FoliageMeshBake, RaaError> {
    let samples = sample_foliage(source, settings.count, settings.seed)?;
    let albedo = sample_foliage_albedo(gpu, lit, source, &samples).await?;
    build_stochastic_foliage(source, &samples, &albedo, settings.params.resolution)
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
    let (mesh, volume, references, layers) = match route {
        FoliageRoute::Impostor => {
            check_impostor_allocation(source.triangles.len())?;
            let hull = build_foliage_hull(&source)?;
            let references = render_foliage_references(
                &gpu,
                &lit,
                &source,
                params.view_count,
                params.texture_resolution,
            )?;
            let layers = pack_foliage_layers(&gpu, &source, &references)?;
            let volume = stochastic_bake(&gpu, &lit, &source, &settings)
                .await?
                .volume;
            (hull, volume, Some(references), Some(layers))
        }
        FoliageRoute::Stochastic => {
            let bake = stochastic_bake(&gpu, &lit, &source, &settings).await?;
            (bake.mesh, bake.volume, None, None)
        }
        FoliageRoute::Differentiable => {
            let references = render_foliage_references(
                &gpu,
                &lit,
                &source,
                params.view_count,
                params.texture_resolution,
            )?;
            let bake =
                optimize_foliage(&gpu, &lit, source, references, settings.optimize_settings())
                    .await?;
            (bake.mesh, bake.volume, None, None)
        }
    };
    let mut resources = FoliageResources::new(&gpu, params)?;
    resources.initialize(&gpu, layers.as_ref(), &volume)?;
    // The packing and initialization submissions precede this marker.
    submissions_done(&gpu).await;
    drop((references, layers));
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
