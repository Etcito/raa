//! The part seam of the GI volume (S37): the install context [`PartsCtx`], the two traits a part
//! or driver implements ([`GiPart`], [`VoxelDriver`]), the frame record handed to a part
//! ([`PartFrame`]), the per-level voxelize record ([`LevelVoxelize`]) and the shared frame state
//! handle ([`GiShared`]).
//!
//! The registry is generated: `build.rs` lists `src/parts/*.rs` in file-name order and writes
//! `$OUT_DIR/gi_parts.rs`, included at the end of this file, so a part module lives at
//! `raa_gi::parts::<stem>` beside these types and `install_parts` (crate-private) calls each
//! module's `install` in that order.

use std::cell::RefCell;
use std::num::NonZeroU64;
use std::rc::Rc;

use raa_core::{GpuContext, RaaError};
use raa_ibl::IblHandle;
use raa_lit::LitRenderer;
use raa_pipeline::{LightingResources, Pipeline, ResourceTable, Targets};
use raa_vg::{VgHandle, scene::SyncItem};

use crate::emissive::EmissiveSources;
use crate::membership::Member;
use crate::settings::{GiQuality, GiSettings};
use crate::tunables::GiTunables;
use crate::vertex_colors::GiVertexColors;

/// The four G-buffer views a GI part binds: the `depth`, `normal`, `material` and `motion`
/// inputs resolved through `Targets::depth_input`, `normal_input`, `material_input` and
/// `motion_input` (crates/raa-pipeline/src/targets.rs:323 `Targets::depth_input`); an unbound
/// input is today's target. A part keeps this in its bind-group staleness key and compares it
/// with `PartialEq`.
///
/// Handoff: card 'raa-gi: SSGI reads the G-buffer from bound inputs' calls
/// `GBufferViews::resolve(targets: &Targets, resources: &ResourceTable, pass: &str) ->
/// GBufferViews` (pub(crate), crates/raa-gi/src/parts.rs, reached as
/// `crate::parts::GBufferViews`) with pass `gi.ssgi`. The state passed: the four views
/// `GBufferViews { depth: wgpu::TextureView, normal: wgpu::TextureView, material:
/// wgpu::TextureView, motion: wgpu::TextureView }`, compared with PartialEq for bind-group
/// staleness.
/// Handoff: card 'raa-gi: area shadows read the G-buffer from bound inputs' calls the same
/// `GBufferViews::resolve` with pass `gi.area-shadow` and passes `&GBufferViews` to its
/// builders.
#[derive(Clone, PartialEq)]
pub(crate) struct GBufferViews {
    pub depth: wgpu::TextureView,
    pub normal: wgpu::TextureView,
    pub material: wgpu::TextureView,
    pub motion: wgpu::TextureView,
}

impl GBufferViews {
    /// Resolves the four G-buffer inputs of `pass`; unbound inputs give today's targets.
    pub(crate) fn resolve(
        targets: &Targets,
        resources: &ResourceTable,
        pass: &str,
    ) -> GBufferViews {
        GBufferViews {
            depth: targets.depth_input(resources, pass),
            normal: targets.normal_input(resources, pass),
            material: targets.material_input(resources, pass),
            motion: targets.motion_input(resources, pass),
        }
    }
}

/// The install context of [`install_parts`]: the frame's GPU objects, the pipeline and lit
/// renderer, the optional virtual-geometry and image-based-lighting handles, the shared GI frame
/// state, and the parts and drivers collected during installation.
pub struct PartsCtx<'a> {
    pub gpu: &'a GpuContext,
    pub pipeline: &'a Rc<RefCell<Pipeline>>,
    pub lit: &'a mut LitRenderer,
    pub vg: Option<VgHandle>,
    pub ibl: Option<IblHandle>,
    pub shared: GiShared,
    parts: Vec<Rc<dyn GiPart>>,
    drivers: Vec<Rc<dyn VoxelDriver>>,
}

impl<'a> PartsCtx<'a> {
    /// Handoff: the card "raa-gi: feature.rs share 2, passes, install and prepare_frame" calls
    /// `PartsCtx::new` with the frame's GPU context, pipeline, lit renderer, optional VG and IBL
    /// handles and the shared state, then `install_parts`, then `PartsCtx::take`; it also calls
    /// the `GiShared` setters. Today `new` builds the context with two empty collections,
    /// `add_part` and `add_driver` fill them, `take` returns them, and the generated
    /// `install_parts` calls each part module's `install` in file-name order. LIFECYCLE: as
    /// contract 'gi-frame-lifecycle' states, `GiHandle::prepare_frame` creates the frame buffer
    /// and atlas and publishes them through `set_frame`/`set_atlas` on an active allocating
    /// frame, and drops them through `clear` on an inactive frame; the shared state and its
    /// keys are contract 'gi-shared'.
    pub(crate) fn new(
        gpu: &'a GpuContext,
        pipeline: &'a Rc<RefCell<Pipeline>>,
        lit: &'a mut LitRenderer,
        vg: Option<VgHandle>,
        ibl: Option<IblHandle>,
        shared: GiShared,
    ) -> PartsCtx<'a> {
        PartsCtx {
            gpu,
            pipeline,
            lit,
            vg,
            ibl,
            shared,
            parts: Vec::new(),
            drivers: Vec::new(),
        }
    }

    /// Adds a part; its `prepare_frame` runs on every active frame.
    pub fn add_part(&mut self, part: Rc<dyn GiPart>) {
        self.parts.push(part);
    }

    /// Adds a voxel driver; it may claim members and records into the shared voxel buffer.
    pub fn add_driver(&mut self, driver: Rc<dyn VoxelDriver>) {
        self.drivers.push(driver);
    }

    /// The parts and drivers collected by `add_part` and `add_driver`, in insertion order.
    #[allow(clippy::type_complexity)]
    pub(crate) fn take(self) -> (Vec<Rc<dyn GiPart>>, Vec<Rc<dyn VoxelDriver>>) {
        (self.parts, self.drivers)
    }
}

/// A GI part: optional frame-level work beside the volume (screen probes, slot providers, debug
/// views). Every method after `name` has a no-op default.
pub trait GiPart {
    /// A stable name for diagnostics.
    fn name(&self) -> &'static str;

    /// Runs once per active frame, after the volume's own prepare.
    fn prepare_frame(&self, gpu: &GpuContext, frame: &PartFrame<'_>) {
        let _ = (gpu, frame);
    }

    /// The texture the part writes as a screen-space GI output, bound at group 1 binding 14.
    fn screen_output(&self) -> Option<wgpu::TextureView> {
        None
    }

    /// Whether the part reads the GI history texture, so the history pass must copy it.
    fn wants_history(&self) -> bool {
        false
    }

    /// Drops the part's history state.
    fn invalidate_history(&self) {}

    /// Releases the part's GPU resources.
    fn release(&self) {}

    /// The GPU bytes the part owns.
    fn allocated_bytes(&self) -> u64 {
        0
    }

    /// Runs after a submission completed (`GiHandle::observe_submission`): a part with a GPU
    /// readback starts mapping it here.
    fn observe_submission(&self) {}

    /// The screen accumulation statistics the part read back, if it has any.
    fn screen_stats(&self) -> Option<crate::parts::screen::ScreenStats> {
        None
    }
}

/// The frame record handed to [`GiPart::prepare_frame`].
pub struct PartFrame<'a> {
    pub active: bool,
    pub settings: GiSettings,
    pub quality: GiQuality,
    pub viewport: (u32, u32),
    pub members: &'a [Member],
    /// The previous frame's lit HDR was copied (`gi.history`) at this viewport.
    pub history_valid: bool,
    /// The previous frame was active at this viewport with no camera cut or reallocation in
    /// between: a part's own history of last frame may be reprojected, with or without the lit
    /// HDR copy.
    pub camera_valid: bool,
}

/// A voxel driver: writes voxelized members into the one shared voxel buffer (driver A forward,
/// driver B virtualized).
pub trait VoxelDriver {
    /// A stable name for diagnostics.
    fn name(&self) -> &'static str;

    /// Whether this driver voxelizes `item` instead of the built-in forward driver.
    fn claims(&self, item: &SyncItem, lit: &LitRenderer) -> bool;

    /// Uploads the driver's per-frame records for the prepared members and levels.
    fn prepare_frame(&self, gpu: &GpuContext, members: &[Member], levels: &[LevelVoxelize]) {
        let _ = (gpu, members, levels);
    }

    /// Records one rebuilt level into the command encoder.
    fn record_level(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        level: &LevelVoxelize,
        voxels: &wgpu::Buffer,
    );

    /// The scene instances this driver voxelized in the latest prepared frame.
    fn voxelized_instances(&self) -> u32;

    /// Releases the driver's GPU resources.
    fn release(&self);

    /// The GPU bytes the driver owns.
    fn allocated_bytes(&self) -> u64;
}

/// One clipmap level to voxelize. `params` is that level's `GiVoxelizeGpu` uniform, owned by the
/// volume.
#[derive(Clone, Debug)]
pub struct LevelVoxelize {
    pub level: u32,
    pub origin: [i32; 3],
    pub voxel_size: f32,
    pub resolution: u32,
    pub level_offset: u32,
    pub params: wgpu::Buffer,
}

/// The frame state the feature handle publishes for the GI compute passes: the 432 B
/// `GiFrameGpu` uniform, the atlas and its generation, the history view, the COMPUTE-visible
/// group-1 layout and its cached bind group, and the current quality and settings.
#[derive(Clone)]
pub struct GiShared(Rc<RefCell<SharedFrame>>);

/// The state behind [`GiShared`].
struct SharedFrame {
    device: wgpu::Device,
    frame: Option<wgpu::Buffer>,
    atlas: Option<wgpu::TextureView>,
    atlas_generation: u64,
    history: Option<wgpu::TextureView>,
    pyramid: Option<wgpu::TextureView>,
    probe_output: Option<wgpu::TextureView>,
    screen_debug: Option<wgpu::TextureView>,
    trace: Option<raa_sdf_trace::TraceHandle>,
    group1_layout: wgpu::BindGroupLayout,
    group1: Option<(u64, u64, wgpu::BindGroup)>,
    colors: GiVertexColors,
    quality: GiQuality,
    settings: GiSettings,
    tunables: GiTunables,
    emissive_sources: EmissiveSources,
    emissive_generation: u64,
    voxel_buffer: Option<wgpu::Buffer>,
    voxel_generation: u64,
}

impl GiShared {
    /// A handle with no frame, atlas or history yet, and the default quality, settings and
    /// tunables.
    pub fn new(gpu: &GpuContext) -> GiShared {
        let group1_layout =
            gpu.device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("raa-gi compute group 1"),
                    entries: &compute_group1_entries(),
                });
        GiShared(Rc::new(RefCell::new(SharedFrame {
            device: gpu.device().clone(),
            frame: None,
            atlas: None,
            atlas_generation: 0,
            history: None,
            pyramid: None,
            probe_output: None,
            screen_debug: None,
            trace: None,
            group1_layout,
            group1: None,
            colors: GiVertexColors::default(),
            quality: GiQuality::default(),
            settings: GiSettings::default(),
            tunables: GiTunables::new(),
            emissive_sources: EmissiveSources::ALL,
            emissive_generation: 0,
            voxel_buffer: None,
            voxel_generation: 0,
        })))
    }

    /// The optional per-vertex colour streams both voxelizers read (clones share them).
    pub fn vertex_colors(&self) -> GiVertexColors {
        self.0.borrow().colors.clone()
    }

    /// The `GiFrameGpu` uniform, or `None` while GI is inactive.
    pub fn frame_buffer(&self) -> Option<wgpu::Buffer> {
        self.0.borrow().frame.clone()
    }

    /// The volume atlas view, or `None` while GI is inactive.
    pub fn atlas_view(&self) -> Option<wgpu::TextureView> {
        self.0.borrow().atlas.clone()
    }

    /// The GI history view, or `None` before the history pass created it.
    pub fn history_view(&self) -> Option<wgpu::TextureView> {
        self.0.borrow().history.clone()
    }

    /// The all-mip R32Float closest-depth pyramid view that the gi.screen pass recorded in the
    /// current frame, or `None` when it did not record this frame and after `clear`.
    ///
    /// Handoff: the card "raa-gi: AreaShadowPart and the gi.area-shadow feature pass (order 43)
    /// publishing the mask" calls `GiShared::pyramid_view` and `GiShared::trace`. Today
    /// `pyramid_view` returns the view `GiShared::set_pyramid` stored, `clear` drops it, and
    /// `trace` returns the handle `GiShared::set_trace` stored, which `clear` keeps.
    pub fn pyramid_view(&self) -> Option<wgpu::TextureView> {
        self.0.borrow().pyramid.clone()
    }

    /// The Rgba16Float screen-probe output that the gi.screen pass recorded in the current frame:
    /// rgb is the probe irradiance and alpha the probe occlusion. It is None when gi.screen did
    /// not record this frame.
    ///
    /// Handoff: the card
    /// "raa-gi ssgi: SsgiPart, the gi.ssgi pass (order 44) and the binding-14 pick" calls
    /// `GiShared::probe_output_view`. Today it returns the view that
    /// `GiShared::set_probe_output` stored, and `clear` drops it. LIFECYCLE:
    /// `GiScreenPass::record` (crates/raa-gi/src/parts/screen.rs) clears the view at its start, in
    /// the same place where it clears the pyramid view. When `ScreenPipelines::record` returns
    /// true, it stores the output view of the current parity, the one written this frame, taken
    /// from `ScreenResources::output_view`. `GiScreenPass::set_enabled` with false and
    /// `ScreenPart::release` drop the view. The pipeline records passes in order
    /// (crates/raa-pipeline/src/pipeline.rs:917-919), so any pass of order > 42 reads this
    /// frame's view.
    pub fn probe_output_view(&self) -> Option<wgpu::TextureView> {
        self.0.borrow().probe_output.clone()
    }

    /// The SDF trace scene attached by `GiHandle::set_trace_scene`, or `None` by default.
    pub fn trace(&self) -> Option<raa_sdf_trace::TraceHandle> {
        self.0.borrow().trace.clone()
    }

    /// The COMPUTE-visible group-1 layout of the GI compute modules.
    pub fn compute_group1_layout(&self) -> wgpu::BindGroupLayout {
        self.0.borrow().group1_layout.clone()
    }

    /// The GI compute group 1: `None` without an atlas, otherwise the cached group, rebuilt when
    /// the lighting views or the atlas changed.
    pub fn compute_group1(&self, lighting: &LightingResources) -> Option<wgpu::BindGroup> {
        let views = lighting.views_generation();
        let mut shared = self.0.borrow_mut();
        let atlas = shared.atlas.clone()?;
        if let Some((cached_views, cached_atlas, group)) = &shared.group1
            && *cached_views == views
            && *cached_atlas == shared.atlas_generation
        {
            return Some(group.clone());
        }
        let device = shared.device.clone();
        let layout = shared.group1_layout.clone();
        let atlas_generation = shared.atlas_generation;
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-gi compute group 1"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(lighting.env_specular_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(lighting.linear_clamp()),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: lighting.local_lights_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: wgpu::BindingResource::TextureView(
                        lighting.cloud_transmittance_view(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: lighting
                        .atmospheric_transmission_buffer()
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: wgpu::BindingResource::TextureView(&atlas),
                },
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: lighting.gi_volume_params_buffer().as_entire_binding(),
                },
            ],
        });
        shared.group1 = Some((views, atlas_generation, group.clone()));
        Some(group)
    }

    /// The current quality.
    pub fn quality(&self) -> GiQuality {
        self.0.borrow().quality
    }

    /// The current settings.
    pub fn settings(&self) -> GiSettings {
        self.0.borrow().settings
    }

    /// Publishes the `GiFrameGpu` uniform, or drops it with `None`.
    pub(crate) fn set_frame(&self, frame: Option<wgpu::Buffer>) {
        self.0.borrow_mut().frame = frame;
    }

    /// Publishes the atlas, advancing its generation and dropping the cached compute group.
    pub(crate) fn set_atlas(&self, atlas: Option<wgpu::TextureView>) {
        let mut shared = self.0.borrow_mut();
        shared.atlas = atlas;
        shared.atlas_generation = shared.atlas_generation.wrapping_add(1);
        shared.group1 = None;
    }

    /// Publishes the history view, or drops it with `None`.
    #[allow(
        dead_code,
        reason = "no caller until a later milestone's history part publishes the history view"
    )]
    pub(crate) fn set_history(&self, history: Option<wgpu::TextureView>) {
        self.0.borrow_mut().history = history;
    }

    /// Publishes the gi.screen depth-pyramid view for this frame, or drops it with `None`.
    pub(crate) fn set_pyramid(&self, pyramid: Option<wgpu::TextureView>) {
        self.0.borrow_mut().pyramid = pyramid;
    }

    /// Publishes the gi.screen probe-output view for this frame, or drops it with `None`.
    pub(crate) fn set_probe_output(&self, view: Option<wgpu::TextureView>) {
        self.0.borrow_mut().probe_output = view;
    }

    /// The Rgba32Uint per-pixel debug record the gi.screen pass wrote this frame (raw irradiance,
    /// history length, accumulation state, reservoir confidence, hit code), or `None`.
    pub fn screen_debug_view(&self) -> Option<wgpu::TextureView> {
        self.0.borrow().screen_debug.clone()
    }

    /// Publishes the gi.screen debug record for this frame, or drops it with `None`.
    pub(crate) fn set_screen_debug(&self, view: Option<wgpu::TextureView>) {
        self.0.borrow_mut().screen_debug = view;
    }

    /// Attaches the SDF trace scene, or detaches it with `None`.
    pub(crate) fn set_trace(&self, trace: Option<raa_sdf_trace::TraceHandle>) {
        self.0.borrow_mut().trace = trace;
    }

    /// Publishes the quality.
    pub(crate) fn set_quality(&self, quality: GiQuality) {
        self.0.borrow_mut().quality = quality;
    }

    /// Publishes the settings.
    pub(crate) fn set_settings(&self, settings: GiSettings) {
        self.0.borrow_mut().settings = settings;
    }

    /// The value of tunable `key` in the published store, or `None` for an unknown key. It does
    /// not allocate, so a pass may read it every frame.
    pub fn tunable_value(&self, key: &str) -> Option<f64> {
        self.0.borrow().tunables.value(key)
    }

    /// The generation of the published store: it advances on every change. It does not allocate.
    pub fn tunables_generation(&self) -> u64 {
        self.0.borrow().tunables.generation()
    }

    /// A clone of the published store. The clone allocates, so a pass calls this only when
    /// [`GiShared::tunables_generation`] differs from the generation its pipelines were built with.
    pub fn tunables(&self) -> GiTunables {
        self.0.borrow().tunables.clone()
    }

    /// Publishes `tunables`, replacing the store: `GiHandle::set_tunable` calls it after its core
    /// copy changed.
    pub fn set_tunables(&self, tunables: GiTunables) {
        self.0.borrow_mut().tunables = tunables;
    }

    /// The emissive sources of the GI passes, `EmissiveSources::ALL` from construction. It does
    /// not allocate.
    pub fn emissive_sources(&self) -> EmissiveSources {
        self.0.borrow().emissive_sources
    }

    /// The generation of the emissive sources: 0 at construction, +1 on every change.
    pub fn emissive_generation(&self) -> u64 {
        self.0.borrow().emissive_generation
    }

    /// Publishes `sources`; the generation advances only when the value differs.
    pub(crate) fn set_emissive_sources(&self, sources: EmissiveSources) {
        let mut shared = self.0.borrow_mut();
        if shared.emissive_sources != sources {
            shared.emissive_sources = sources;
            shared.emissive_generation = shared.emissive_generation.wrapping_add(1);
        }
    }

    /// Handoff: 'raa-gi: screen probes and ReSTIR GI emissive gate at screen hits' (probe-trace
    /// and ReSTIR GI main pipelines and bind groups) and 'raa-gi: SSGI emissive gate at screen
    /// hits' (SSGI trace pipeline and bind group) call `GiShared::voxel_buffer`,
    /// `GiShared::emissive_sources` and `GiShared::emissive_generation`; 'raa-graph: emissive
    /// parameter on the GI pass nodes, executor extension and starters' calls
    /// `GiHandle::set_emissive_sources` (feature.rs), which stores the flags in the core and
    /// publishes them here. Today this returns the voxel word storage buffer of the current
    /// `GiVolume` (3 u32 per voxel, word 2 the emissive RGBM) and its generation (+1 on every
    /// `GiShared::set_voxel_buffer` call), or `None` while GI is inactive. LIFECYCLE:
    /// `GiHandle::prepare_frame` publishes the voxel buffer with `GiShared::set_voxel_buffer`
    /// with Some(buffer) right after it creates a GiVolume (next to `GiShared::set_atlas`), and
    /// `GiShared::clear` (called by prepare_frame when GI turns inactive, and by release) drops
    /// it together with the atlas; the emissive sources live in GiShared from `GiShared::new`
    /// for its whole life and are replaced only by `GiHandle::set_emissive_sources`. Every GI
    /// pass reads them in its `prepare`, which the pipeline runs after
    /// `GiHandle::prepare_frame`; no pass drops or replaces them.
    pub fn voxel_buffer(&self) -> Option<(wgpu::Buffer, u64)> {
        let shared = self.0.borrow();
        let generation = shared.voxel_generation;
        shared
            .voxel_buffer
            .clone()
            .map(|buffer| (buffer, generation))
    }

    /// Publishes the voxel word buffer, or drops it with `None`, advancing its generation.
    pub(crate) fn set_voxel_buffer(&self, buffer: Option<wgpu::Buffer>) {
        let mut shared = self.0.borrow_mut();
        shared.voxel_buffer = buffer;
        shared.voxel_generation = shared.voxel_generation.wrapping_add(1);
    }

    /// Drops the frame buffer, atlas, voxel buffer, history, pyramid and probe-output views and
    /// the cached compute group; the attached trace scene stays.
    pub(crate) fn clear(&self) {
        let mut shared = self.0.borrow_mut();
        shared.frame = None;
        shared.atlas = None;
        shared.voxel_buffer = None;
        shared.history = None;
        shared.pyramid = None;
        shared.probe_output = None;
        shared.screen_debug = None;
        shared.group1 = None;
    }
}

/// The COMPUTE-visible group-1 layout of the GI compute modules: bindings 3, 6, 12, 15, 16, 17
/// and 18 at their S10 numbers.
pub fn compute_group1_entries() -> [wgpu::BindGroupLayoutEntry; 7] {
    use wgpu::TextureSampleType as S;
    use wgpu::TextureViewDimension as D;
    let filterable = S::Float { filterable: true };
    [
        texture_entry(3, filterable, D::Cube),
        sampler_entry(6),
        uniform_entry(12, 16384),
        texture_entry(15, filterable, D::D2Array),
        uniform_entry(16, 64),
        texture_entry(17, filterable, D::D3),
        uniform_entry(18, 96),
    ]
}

/// A COMPUTE texture entry.
fn texture_entry(
    binding: u32,
    sample_type: wgpu::TextureSampleType,
    view_dimension: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled: false,
        },
        count: None,
    }
}

/// A COMPUTE filtering-sampler entry.
fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

/// A COMPUTE uniform entry of `size` bytes.
fn uniform_entry(binding: u32, size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(size),
        },
        count: None,
    }
}

include!(concat!(env!("OUT_DIR"), "/gi_parts.rs"));

/// The software device of one raa-gi unit test, held under the serial GPU lock of this test
/// binary.
///
/// Mesa lavapipe (llvmpipe 26.0.3, LLVM 21) segfaults when one thread destroys its device while
/// another thread of the same process creates compute pipelines on another lavapipe device;
/// `cargo test -p raa-gi --lib` hit it with `parts::tests` beside
/// `area_shadow::resources::tests` running in parallel. Every GPU unit test of the crate takes
/// its device here instead of `raa_core::testing::software_gpu`: the lock is held from before
/// the device exists until after it is dropped (`gpu` is declared before `_serial`, so it drops
/// first), and the CPU tests still run in parallel. The lock is a const-initialised test-only
/// `Mutex`, not a lazy global, and a poisoned lock (a failed GPU test) is taken over.
#[cfg(test)]
pub(crate) struct SerialGpu {
    gpu: GpuContext,
    _serial: std::sync::MutexGuard<'static, ()>,
}

#[cfg(test)]
impl std::ops::Deref for SerialGpu {
    type Target = GpuContext;

    fn deref(&self) -> &GpuContext {
        &self.gpu
    }
}

/// `raa_core::testing::software_gpu` under the serial GPU lock (see [`SerialGpu`]).
#[cfg(test)]
pub(crate) fn serial_software_gpu() -> Option<SerialGpu> {
    static GPU_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let serial = GPU_TESTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    raa_core::testing::software_gpu().map(|gpu| SerialGpu {
        gpu,
        _serial: serial,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_core::testing::skip;

    fn one_pixel_view(gpu: &GpuContext) -> wgpu::TextureView {
        let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("raa-gi pyramid test"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    #[test]
    fn pyramid_and_trace_follow_their_setters_and_clear() {
        let test = "pyramid_and_trace_follow_their_setters_and_clear";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let shared = GiShared::new(&gpu);
        assert!(shared.pyramid_view().is_none());
        assert!(shared.trace().is_none());

        shared.set_pyramid(Some(one_pixel_view(&gpu)));
        assert!(shared.pyramid_view().is_some());

        let scene = Rc::new(RefCell::new(raa_sdf_trace::TraceScene::default()));
        shared.set_trace(Some(Rc::clone(&scene)));
        let trace = shared.trace().expect("the trace is attached");
        assert!(Rc::ptr_eq(&trace, &scene));

        shared.clear();
        assert!(shared.pyramid_view().is_none());
        assert!(shared.trace().is_some());

        shared.set_trace(None);
        assert!(shared.trace().is_none());
    }

    #[test]
    fn probe_output_follows_its_setter_and_clear() {
        let test = "probe_output_follows_its_setter_and_clear";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let shared = GiShared::new(&gpu);
        assert!(shared.probe_output_view().is_none());

        shared.set_probe_output(Some(one_pixel_view(&gpu)));
        assert!(shared.probe_output_view().is_some());

        shared.clear();
        assert!(shared.probe_output_view().is_none());

        shared.set_probe_output(Some(one_pixel_view(&gpu)));
        shared.set_probe_output(None);
        assert!(shared.probe_output_view().is_none());
    }
}
