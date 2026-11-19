//! The stages `gi.pyramid` and `gi.probes` (contract 'rg-gi-stage-api'; types and ports of
//! 'rg-gi-stage-types'): each is a [`StageImpl`] that `crate::stages::stage_impl` hands to the
//! stage pass. The pyramid publishes the R32Float depth mip chain and records its copy and reduce;
//! the probes publish the probe atlas and the two probe buffers and record the trace, project and
//! filter dispatches. Each stage builds its own [`ScreenPipelines`]. The whole-pass path
//! (`ScreenPart`, `GiScreenPass`) is unchanged.

use raa_core::{GpuContext, RaaError};
use raa_pipeline::{ImageDesc, ImageDim, PrepareCtx, RecordCtx, validate_tunable};

use super::cpu;
use super::passes::{PROBE_BYTES, SCREEN_PARAMS_SIZE, ScreenPipelines, pyramid_sizes};
use super::{
    FRAME_JITTER_X_OFFSET, FRAME_JITTER_Y_OFFSET, buffer, full_view, group, mip_view,
    screen_params_bytes, texture, texture_resource,
};
use crate::parts::GBufferViews;
use crate::settings::GiSettings;
use crate::stages::{StageCx, StageImpl};
use crate::tunables;

/// The keys of `gi.probes`: `screenTraces`, then the seven probe tunables.
const PROBE_KEYS: &[&str] = &[
    "screenTraces",
    "probeScreenDistance",
    "probeScreenSteps",
    "probeScreenThickness",
    "probeVolumeDistance",
    "probeVolumeSteps",
    "probeVolumeHitVoxels",
    "screenEdgeFade",
];

/// The usages of a stage texture: TEXTURE_BINDING | STORAGE_BINDING | COPY_SRC.
fn stage_texture_usages() -> wgpu::TextureUsages {
    wgpu::TextureUsages::TEXTURE_BINDING
        | wgpu::TextureUsages::STORAGE_BINDING
        | wgpu::TextureUsages::COPY_SRC
}

/// A stage's own screen pipelines, with the values and emissive generation they were built for.
struct Built {
    screen: ScreenPipelines,
    stamp: (u64, u64),
}

/// Builds the pipelines on first use, and rebuilds the tuned ones when the stage's values
/// (`cx.values_generation`) or the emissive sources (`GiShared::emissive_generation`) changed.
fn refresh(slot: &mut Option<Built>, gpu: &GpuContext, cx: &StageCx<'_>) {
    let emissive = (
        cx.shared.emissive_sources().screen,
        cx.shared.emissive_generation(),
    );
    let stamp = (cx.values_generation, emissive.1);
    if slot.as_ref().is_some_and(|built| built.stamp == stamp) {
        return;
    }
    let tunables = cx.tunables();
    let screen = match slot.take() {
        Some(built) => {
            let mut screen = built.screen;
            screen.rebuild_tuned(gpu, &tunables, emissive);
            screen
        }
        None => ScreenPipelines::new(gpu, &cx.shared.compute_group1_layout(), &tunables, emissive),
    };
    *slot = Some(Built { screen, stamp });
}

/// Writes this frame's TAA jitter into the two spare lanes of the GI frame record, as
/// `GiScreenPass::prepare` does (`FRAME_JITTER_X_OFFSET`, `FRAME_JITTER_Y_OFFSET`).
fn write_jitter(cx: &StageCx<'_>, gpu: &GpuContext) {
    let Some(frame) = cx.shared.frame_buffer() else {
        return;
    };
    let [x, y] = cx.lit.camera_jitter_ndc();
    gpu.queue()
        .write_buffer(&frame, FRAME_JITTER_X_OFFSET, &x.to_le_bytes());
    gpu.queue()
        .write_buffer(&frame, FRAME_JITTER_Y_OFFSET, &y.to_le_bytes());
}

/// The `gi.pyramid` stage: the mip chain of the depth pyramid for one viewport.
#[derive(Default)]
pub(crate) struct PyramidStage {
    built: Option<Built>,
    chain: Option<Chain>,
    /// The copy and reduce groups of `chain` over the G-buffer views they were built from.
    bindings: Option<(GBufferViews, Vec<wgpu::BindGroup>)>,
}

/// The R32Float mip chain of one viewport: one view per level and the full view.
struct Chain {
    viewport: (u32, u32),
    sizes: Vec<(u32, u32)>,
    mip_views: Vec<wgpu::TextureView>,
    full: wgpu::TextureView,
}

impl Chain {
    fn new(gpu: &GpuContext, viewport: (u32, u32)) -> Chain {
        let sizes = pyramid_sizes(viewport);
        let pyramid = texture(
            gpu,
            "raa-gi.stage.pyramid",
            viewport,
            sizes.len() as u32,
            wgpu::TextureFormat::R32Float,
            stage_texture_usages(),
        );
        Chain {
            viewport,
            mip_views: (0..sizes.len())
                .map(|mip| mip_view(&pyramid, mip as u32))
                .collect(),
            full: full_view(&pyramid),
            sizes,
        }
    }
}

impl PyramidStage {
    /// Rebuilds the copy over the depth and the reduce of each later level over `views`, unless
    /// the cached groups were built from the same views and chain.
    fn ensure_groups(&mut self, gpu: &GpuContext, views: &GBufferViews) {
        let (Some(built), Some(chain)) = (self.built.as_ref(), self.chain.as_ref()) else {
            return;
        };
        if self
            .bindings
            .as_ref()
            .is_some_and(|(cached, _)| cached == views)
        {
            return;
        }
        let device = gpu.device();
        let mut groups = Vec::with_capacity(chain.mip_views.len());
        if let Some(first) = chain.mip_views.first() {
            groups.push(group(
                device,
                "raa-gi.stage.pyramid-copy",
                &built.screen.copy_layout,
                &[
                    (0, texture_resource(&views.depth)),
                    (2, texture_resource(first)),
                ],
            ));
        }
        for index in 1..chain.mip_views.len() {
            groups.push(group(
                device,
                "raa-gi.stage.pyramid-reduce",
                &built.screen.reduce_layout,
                &[
                    (1, texture_resource(&chain.mip_views[index - 1])),
                    (2, texture_resource(&chain.mip_views[index])),
                ],
            ));
        }
        self.bindings = Some((views.clone(), groups));
    }
}

impl StageImpl for PyramidStage {
    fn param_keys(&self) -> &'static [&'static str] {
        &[]
    }

    fn check_value(&self, key: &str, _value: f64) -> Result<(), String> {
        Err(format!("`{key}` is not a key of gi.pyramid"))
    }

    /// Allocates the chain for `ctx.size` (clamped to 1), writes the jitter, and publishes
    /// `pyramid` with the level count as `mips`.
    fn prepare(
        &mut self,
        cx: &StageCx<'_>,
        ctx: &mut PrepareCtx<'_>,
        published: &mut Vec<(&'static str, ImageDesc)>,
    ) -> Result<(), RaaError> {
        refresh(&mut self.built, ctx.gpu, cx);
        let viewport = (ctx.size.0.max(1), ctx.size.1.max(1));
        if self
            .chain
            .as_ref()
            .is_none_or(|chain| chain.viewport != viewport)
        {
            self.chain = Some(Chain::new(ctx.gpu, viewport));
            self.bindings = None;
        }
        write_jitter(cx, ctx.gpu);
        if let Some(chain) = self.chain.as_ref() {
            let desc = ImageDesc::new(
                ImageDim::D2,
                wgpu::TextureFormat::R32Float,
                viewport.0,
                viewport.1,
                1,
                chain.sizes.len() as u32,
            );
            ctx.resources
                .publish(cx.entry, "pyramid", &chain.full, desc);
            published.push(("pyramid", desc));
        }
        Ok(())
    }

    /// Records the copy and the reduces in one compute pass with group 1 set; false (nothing
    /// recorded) without group 1, the frame buffer, a built chain or its groups.
    fn record(&mut self, cx: &StageCx<'_>, ctx: &mut RecordCtx<'_>) -> Result<bool, RaaError> {
        let Some(group1) = cx.shared.compute_group1(ctx.lighting) else {
            return Ok(false);
        };
        if cx.shared.frame_buffer().is_none() {
            return Ok(false);
        }
        let views = GBufferViews::resolve(ctx.targets, ctx.resources, cx.entry);
        self.ensure_groups(ctx.gpu, &views);
        let (Some(built), Some(chain), Some((_, groups))) = (
            self.built.as_ref(),
            self.chain.as_ref(),
            self.bindings.as_ref(),
        ) else {
            return Ok(false);
        };
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("raa-gi.stage.pyramid"),
                timestamp_writes: None,
            });
        pass.set_bind_group(1, &group1, &[]);
        Ok(built
            .screen
            .record_pyramid_groups(&mut pass, groups, &chain.sizes))
    }

    fn resize(&mut self) {
        self.bindings = None;
    }
}

/// The probe allocation of one (viewport, probe rays).
struct ProbeAlloc {
    key: ((u32, u32), u32),
    grid: (u32, u32),
    atlas: wgpu::TextureView,
    raw: wgpu::Buffer,
    filtered: wgpu::Buffer,
}

impl ProbeAlloc {
    fn new(gpu: &GpuContext, viewport: (u32, u32), rays: u32) -> ProbeAlloc {
        let found = cpu::probe_grid([viewport.0, viewport.1]);
        let grid = (found[0], found[1]);
        let atlas = texture(
            gpu,
            "raa-gi.stage.atlas",
            (grid.0 * rays, grid.1 * rays),
            1,
            wgpu::TextureFormat::Rgba16Float,
            stage_texture_usages(),
        );
        let bytes = PROBE_BYTES * u64::from(grid.0) * u64::from(grid.1);
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        ProbeAlloc {
            key: (viewport, rays),
            grid,
            atlas: full_view(&atlas),
            raw: buffer(gpu, "raa-gi.stage.probes", bytes, usage),
            filtered: buffer(gpu, "raa-gi.stage.probes-filtered", bytes, usage),
        }
    }
}

/// The 1x1 history and the 16-B voxel words the trace group binds while the lit history or the
/// GI voxel buffer is absent.
struct Dummies {
    history: wgpu::TextureView,
    voxel: wgpu::Buffer,
}

impl Dummies {
    fn new(gpu: &GpuContext) -> Dummies {
        let dummy = texture(
            gpu,
            "raa-gi.stage.history-dummy",
            (1, 1),
            1,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        gpu.queue().write_texture(
            dummy.as_image_copy(),
            &[0, 0, 0, 0, 0, 0, 0x00, 0x3C],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        Dummies {
            history: full_view(&dummy),
            voxel: buffer(
                gpu,
                "raa-gi.stage.voxel-dummy",
                16,
                wgpu::BufferUsages::STORAGE,
            ),
        }
    }
}

/// The per-frame resources the probe groups bind besides the stage's atlas and probe buffers;
/// the groups are rebuilt when any of them changes.
#[derive(PartialEq)]
struct GroupKey {
    frame: wgpu::Buffer,
    views: GBufferViews,
    pyramid: wgpu::TextureView,
    history: Option<wgpu::TextureView>,
    voxels: Option<wgpu::Buffer>,
}

/// The trace, project and filter groups of one [`GroupKey`].
struct ProbeGroups {
    key: GroupKey,
    trace: wgpu::BindGroup,
    project: wgpu::BindGroup,
    filter: wgpu::BindGroup,
}

impl ProbeGroups {
    fn new(
        gpu: &GpuContext,
        built: &Built,
        alloc: &ProbeAlloc,
        dummies: &Dummies,
        params: &wgpu::Buffer,
        key: GroupKey,
    ) -> ProbeGroups {
        let device = gpu.device();
        let pipelines = &built.screen;
        let history = key.history.as_ref().unwrap_or(&dummies.history);
        let voxels = key.voxels.as_ref().unwrap_or(&dummies.voxel);
        let (frame, views) = (&key.frame, &key.views);
        let trace = group(
            device,
            "raa-gi.stage.trace",
            &pipelines.trace_layout,
            &[
                (0, frame.as_entire_binding()),
                (1, texture_resource(&views.depth)),
                (2, texture_resource(&views.normal)),
                (3, texture_resource(history)),
                (4, texture_resource(&key.pyramid)),
                (5, texture_resource(&alloc.atlas)),
                (6, voxels.as_entire_binding()),
                (7, params.as_entire_binding()),
            ],
        );
        let project = group(
            device,
            "raa-gi.stage.project",
            &pipelines.project_layout,
            &[
                (0, frame.as_entire_binding()),
                (1, texture_resource(&views.depth)),
                (2, texture_resource(&views.normal)),
                (3, texture_resource(&alloc.atlas)),
                (5, alloc.raw.as_entire_binding()),
            ],
        );
        let filter = group(
            device,
            "raa-gi.stage.filter",
            &pipelines.filter_layout,
            &[
                (0, frame.as_entire_binding()),
                (4, alloc.raw.as_entire_binding()),
                (5, alloc.filtered.as_entire_binding()),
            ],
        );
        ProbeGroups {
            key,
            trace,
            project,
            filter,
        }
    }
}

/// The `gi.probes` stage: the probe atlas, the raw and filtered probe buffers and the stage's own
/// `GiScreenGpu` params.
#[derive(Default)]
pub(crate) struct ProbesStage {
    built: Option<Built>,
    alloc: Option<ProbeAlloc>,
    params: Option<wgpu::Buffer>,
    dummies: Option<Dummies>,
    groups: Option<ProbeGroups>,
}

impl StageImpl for ProbesStage {
    fn param_keys(&self) -> &'static [&'static str] {
        PROBE_KEYS
    }

    /// `screenTraces` takes 0 or 1; a tunable takes its `validate_tunable` range, and a step
    /// count (an integer tunable) is not negative.
    fn check_value(&self, key: &str, value: f64) -> Result<(), String> {
        if key == "screenTraces" {
            return if value == 0.0 || value == 1.0 {
                Ok(())
            } else {
                Err("screenTraces takes 0 or 1".to_string())
            };
        }
        let spec = validate_tunable(tunables(), key, value)?;
        if spec.integer && value < 0.0 {
            return Err(format!("`{key}` is a step count and is not negative"));
        }
        Ok(())
    }

    /// Allocates the atlas and the probe buffers for the viewport and the quality's probe rays,
    /// writes the params (screen traces from the value, history from the shared state, the other
    /// fields from `GiSettings::DEFAULT`), and publishes `atlas`, `probesRaw` and `probesFiltered`.
    fn prepare(
        &mut self,
        cx: &StageCx<'_>,
        ctx: &mut PrepareCtx<'_>,
        published: &mut Vec<(&'static str, ImageDesc)>,
    ) -> Result<(), RaaError> {
        refresh(&mut self.built, ctx.gpu, cx);
        let viewport = (ctx.size.0.max(1), ctx.size.1.max(1));
        let rays = cx.shared.quality().probe_rays;
        if self
            .alloc
            .as_ref()
            .is_none_or(|alloc| alloc.key != (viewport, rays))
        {
            self.alloc = Some(ProbeAlloc::new(ctx.gpu, viewport, rays));
            self.groups = None;
        }
        self.dummies.get_or_insert_with(|| Dummies::new(ctx.gpu));
        let params = self.params.get_or_insert_with(|| {
            buffer(
                ctx.gpu,
                "raa-gi.stage.params",
                SCREEN_PARAMS_SIZE,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            )
        });
        let default_traces = f64::from(u8::from(GiSettings::DEFAULT.screen_traces));
        let screen_traces = cx.value("screenTraces", default_traces) != 0.0;
        let bytes = screen_params_bytes(
            cx.shared.history_valid(),
            screen_traces,
            0,
            &GiSettings::DEFAULT,
        );
        ctx.gpu.queue().write_buffer(params, 0, &bytes);
        if let Some(alloc) = self.alloc.as_ref() {
            let (gx, gy) = alloc.grid;
            let desc = ImageDesc::new(
                ImageDim::D2,
                wgpu::TextureFormat::Rgba16Float,
                gx * rays,
                gy * rays,
                1,
                1,
            );
            ctx.resources.publish(cx.entry, "atlas", &alloc.atlas, desc);
            published.push(("atlas", desc));
            let size = PROBE_BYTES * u64::from(gx) * u64::from(gy);
            ctx.resources
                .publish_buffer(cx.entry, "probesRaw", &alloc.raw, size);
            ctx.resources
                .publish_buffer(cx.entry, "probesFiltered", &alloc.filtered, size);
        }
        Ok(())
    }

    /// Records the trace, project and filter dispatches in one compute pass with group 1 set;
    /// false (nothing recorded) without the `pyramid` input, group 1, the frame buffer, or when
    /// the probe rays are not in `PROBE_RAY_OPTIONS`.
    fn record(&mut self, cx: &StageCx<'_>, ctx: &mut RecordCtx<'_>) -> Result<bool, RaaError> {
        let Some(pyramid) = ctx.resources.input(cx.entry, "pyramid") else {
            return Ok(false);
        };
        let Some(group1) = cx.shared.compute_group1(ctx.lighting) else {
            return Ok(false);
        };
        let Some(frame) = cx.shared.frame_buffer() else {
            return Ok(false);
        };
        let ProbesStage {
            built,
            alloc,
            params,
            dummies,
            groups,
        } = self;
        let (Some(built), Some(alloc), Some(params), Some(dummies)) = (
            built.as_ref(),
            alloc.as_ref(),
            params.as_ref(),
            dummies.as_ref(),
        ) else {
            return Ok(false);
        };
        let key = GroupKey {
            frame,
            views: GBufferViews::resolve(ctx.targets, ctx.resources, cx.entry),
            pyramid: pyramid.view,
            history: cx.shared.history_view(),
            voxels: cx.shared.voxel_buffer().map(|(buffer, _)| buffer),
        };
        if groups.as_ref().is_none_or(|cached| cached.key != key) {
            *groups = Some(ProbeGroups::new(
                ctx.gpu, built, alloc, dummies, params, key,
            ));
        }
        let Some(groups) = groups.as_ref() else {
            return Ok(false);
        };
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("raa-gi.stage.probes"),
                timestamp_writes: None,
            });
        pass.set_bind_group(1, &group1, &[]);
        Ok(built.screen.record_probe_chain(
            &mut pass,
            &groups.trace,
            &groups.project,
            &groups.filter,
            alloc.key.1,
            alloc.grid,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use raa_core::testing::skip;
    use raa_lit::{LitOptions, LitRenderer};
    use raa_pipeline::{LightingResources, Pipeline, ResourceTable, Targets};

    use super::*;
    use crate::error::GI_INVALID;
    use crate::parts::GiShared;
    use crate::stages::{GiStageHandle, GiStageKind, install_stage};

    /// The 432-B `GiFrameGpu` buffer the stage tests attach.
    fn frame_buffer(gpu: &GpuContext) -> wgpu::Buffer {
        gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("stages.test.frame"),
            size: 432,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Installs `kind` on a fresh pipeline, attaches a shared state with a 432-B frame buffer and
    /// a lit handle, takes the pass back and runs its resize and prepare on an 8x8 viewport. No
    /// input is bound and no other pass ran: prepare reads no image, so nothing else is set up.
    fn prepared(gpu: &GpuContext, kind: GiStageKind) -> (GiStageHandle, ResourceTable) {
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let pipeline = Rc::new(RefCell::new(Pipeline::new(gpu, format, false)));
        let handle = install_stage(gpu, &pipeline, kind, "t").expect("install");
        let shared = GiShared::new(gpu);
        shared.set_frame(Some(frame_buffer(gpu)));
        let lit = LitRenderer::new(gpu, &pipeline, LitOptions::default()).expect("lit renderer");
        handle.attach_shared(shared, lit.handle());
        let mut pass = pipeline
            .borrow_mut()
            .remove(&handle.entry())
            .expect("the stage pass");
        let targets = Targets::new(gpu, (8, 8));
        let mut lighting = LightingResources::new(gpu);
        let table = ResourceTable::new();
        pass.resize(gpu, &targets);
        let mut ctx = PrepareCtx {
            gpu,
            size: (8, 8),
            frame: 0,
            time_s: 0.0,
            dt_s: 0.0,
            targets: &targets,
            lighting: &mut lighting,
            resources: &table,
        };
        pass.prepare(&mut ctx).expect("prepare");
        (handle, table)
    }

    #[test]
    fn pyramid_publishes_the_mip_chain_of_the_viewport() {
        let test = "pyramid_publishes_the_mip_chain_of_the_viewport";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let (handle, table) = prepared(&gpu, GiStageKind::Pyramid);
        let mips = pyramid_sizes((8, 8)).len() as u32;
        assert!(mips > 1, "the 8x8 pyramid has mips");
        let desc = ImageDesc::new(ImageDim::D2, wgpu::TextureFormat::R32Float, 8, 8, 1, mips);
        assert_eq!(handle.published(), vec![("pyramid", desc)]);
        let found = table.get("gi.pyramid#t", "pyramid").map(|image| image.desc);
        assert_eq!(found, Some(desc));
        assert_eq!(table.entries().len(), 1);
    }

    #[test]
    fn probes_publish_the_atlas_and_both_probe_buffers() {
        let test = "probes_publish_the_atlas_and_both_probe_buffers";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let (handle, table) = prepared(&gpu, GiStageKind::Probes);
        let rays = crate::settings::GiQuality::default().probe_rays;
        let [gx, gy] = cpu::probe_grid([8, 8]);
        let format = wgpu::TextureFormat::Rgba16Float;
        let atlas = ImageDesc::new(ImageDim::D2, format, gx * rays, gy * rays, 1, 1);
        assert_eq!(handle.published(), vec![("atlas", atlas)]);
        let found = table.get("gi.probes#t", "atlas").map(|image| image.desc);
        assert_eq!(found, Some(atlas));
        let bytes = PROBE_BYTES * u64::from(gx) * u64::from(gy);
        for port in ["probesRaw", "probesFiltered"] {
            let size = table.buffer("gi.probes#t", port).map(|(_, size)| size);
            assert_eq!(size, Some(bytes), "{port}");
        }
        assert_eq!(table.buffer_entries().len(), 2);
    }

    #[test]
    fn probes_refuse_a_negative_step_count_and_an_unknown_key() {
        let test = "probes_refuse_a_negative_step_count_and_an_unknown_key";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, format, false)));
        let handle = install_stage(&gpu, &pipeline, GiStageKind::Probes, "t").expect("install");
        let accepted = handle.set_values(&[("probeScreenDistance".to_string(), 6.0)]);
        assert!(accepted.is_ok());
        let steps = handle.set_values(&[("probeScreenSteps".to_string(), -1.0)]);
        assert!(matches!(&steps, Err(error) if error.code() == GI_INVALID));
        let unknown = handle.set_values(&[("notAKey".to_string(), 1.0)]);
        assert!(matches!(&unknown, Err(error) if error.code() == GI_INVALID));
        let traces = handle.set_values(&[("screenTraces".to_string(), 2.0)]);
        assert!(matches!(&traces, Err(error) if error.code() == GI_INVALID));
    }

    #[test]
    fn pyramid_and_probes_record_once_after_prepare() {
        let test = "pyramid_and_probes_record_once_after_prepare";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, format, false)));
        let pyramid = install_stage(&gpu, &pipeline, GiStageKind::Pyramid, "t").expect("install");
        let probes = install_stage(&gpu, &pipeline, GiStageKind::Probes, "t").expect("install");
        let shared = GiShared::new(&gpu);
        shared.set_frame(Some(frame_buffer(&gpu)));
        // The 3D volume atlas group 1 binds; the volume pass publishes it in a real frame.
        let volume = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("stages.test.volume"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let volume = volume.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });
        shared.set_atlas(Some(volume));
        let lit = LitRenderer::new(&gpu, &pipeline, LitOptions::default()).expect("lit renderer");
        pyramid.attach_shared(shared.clone(), lit.handle());
        probes.attach_shared(shared, lit.handle());
        let mut passes = [pyramid.entry(), probes.entry()].map(|entry| {
            pipeline
                .borrow_mut()
                .remove(&entry)
                .expect("the stage pass")
        });
        let targets = Targets::new(&gpu, (8, 8));
        let mut lighting = LightingResources::new(&gpu);
        let table = ResourceTable::new();
        for pass in &mut passes {
            pass.resize(&gpu, &targets);
            let mut ctx = PrepareCtx {
                gpu: &gpu,
                size: (8, 8),
                frame: 0,
                time_s: 0.0,
                dt_s: 0.0,
                targets: &targets,
                lighting: &mut lighting,
                resources: &table,
            };
            pass.prepare(&mut ctx).expect("prepare");
        }
        // A graph binds the pyramid as the probes' input; the stage reads it at record time.
        let published = table.get(&pyramid.entry(), "pyramid").expect("the pyramid");
        table.bind_input(&probes.entry(), "pyramid", published);
        let backbuffer = texture(
            &gpu,
            "stages.test.backbuffer",
            (8, 8),
            1,
            format,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let backbuffer = full_view(&backbuffer);
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for pass in &mut passes {
            let mut ctx = RecordCtx {
                gpu: &gpu,
                encoder: &mut encoder,
                size: (8, 8),
                frame: 0,
                time_s: 0.0,
                targets: &targets,
                lighting: &lighting,
                resources: &table,
                backbuffer: &backbuffer,
                backbuffer_format: format,
                surface_is_srgb: false,
            };
            pass.record(&mut ctx).expect("record");
        }
        gpu.queue().submit(Some(encoder.finish()));
        assert!(pyramid.ran(), "the pyramid recorded its copy and reduces");
        assert!(
            probes.ran(),
            "the probes recorded their trace, project and filter"
        );
    }
}
