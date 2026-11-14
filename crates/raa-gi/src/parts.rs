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
use raa_pipeline::{LightingResources, Pipeline};
use raa_vg::{VgHandle, scene::SyncItem};

use crate::membership::Member;
use crate::settings::{GiQuality, GiSettings};

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
}

/// The frame record handed to [`GiPart::prepare_frame`].
pub struct PartFrame<'a> {
    pub active: bool,
    pub settings: GiSettings,
    pub quality: GiQuality,
    pub viewport: (u32, u32),
    pub members: &'a [Member],
    pub history_valid: bool,
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
    group1_layout: wgpu::BindGroupLayout,
    group1: Option<(u64, u64, wgpu::BindGroup)>,
    quality: GiQuality,
    settings: GiSettings,
}

impl GiShared {
    /// A handle with no frame, atlas or history yet, and the default quality and settings.
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
            group1_layout,
            group1: None,
            quality: GiQuality::default(),
            settings: GiSettings::default(),
        })))
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

    /// Publishes the quality.
    pub(crate) fn set_quality(&self, quality: GiQuality) {
        self.0.borrow_mut().quality = quality;
    }

    /// Publishes the settings.
    pub(crate) fn set_settings(&self, settings: GiSettings) {
        self.0.borrow_mut().settings = settings;
    }

    /// Drops the frame buffer, atlas, history view and cached compute group.
    pub(crate) fn clear(&self) {
        let mut shared = self.0.borrow_mut();
        shared.frame = None;
        shared.atlas = None;
        shared.history = None;
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
