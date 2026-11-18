//! The milestone-2 part `screen` of the GI volume (S39 screen probes): the module root of the
//! part, registered by the generated registry in `parts.rs`. Its CPU twins live in `screen/cpu.rs`
//! and `screen/accumulate_cpu.rs`.
//! Adapted from solstice@7e465b5 `crates/solstice-render/src/gi/screen.rs` and
//! `crates/solstice-render/src/gi/depth_pyramid.rs`. MIT licence, Copyright (c) 2026 SamG-Coder.
//!
//! Each frame the part computes one irradiance per pixel (screen probes or ReSTIR GI), then the
//! shared temporal accumulation (`wgsl/screen/accumulate.wgsl`: disocclusion test, clamp,
//! alpha = max(1 / n, 1 / historyMax)) writes the history of this parity, and `spatialPasses`
//! variance-guided a-trous passes (`wgsl/screen/atrous.wgsl`) write the screen output that
//! `raa_slot_indirect` reads. A debug record per pixel and four statistics counters (read back
//! a few frames later) feed the screen debug views and `GiStats`.

#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use self::passes::{
    ATROUS_MAX_PASSES, ATROUS_SLOT_BYTES, PROBE_BYTES, SCREEN_PARAMS_SIZE, ScreenPipelines,
    ScreenWork, atrous_record, pyramid_sizes,
};
use super::{GBufferViews, GiPart, GiShared, PartFrame, PartsCtx};
use raa_core::{GpuContext, RaaError};
use raa_lit::LitHandle;
use raa_pipeline::{
    FeaturePass, ImageDesc, ImageDim, LightingResources, PrepareCtx, RecordCtx, ResourceTable,
    Targets,
};

#[path = "screen/cpu.rs"]
pub mod cpu;

#[path = "screen/accumulate_cpu.rs"]
pub mod accumulate_cpu;

#[path = "screen/passes.rs"]
pub mod passes;

#[path = "screen/restir_gi.rs"]
pub mod restir_gi;

#[path = "screen/restir_gi_cpu.rs"]
pub mod restir_gi_cpu;

use self::restir_gi::{RestirGiInputs, RestirGiPipelines, RestirGiResources};
use crate::settings::{GiSettings, IndirectMode};

/// Byte offsets of the two `GiFrameGpu` lanes that carry the TAA jitter (sun_direction.w and
/// sun_color.w, otherwise unused): probe_common.wgsl `raa_gi_view_position` reads them.
const FRAME_JITTER_X_OFFSET: u64 = 348;
const FRAME_JITTER_Y_OFFSET: u64 = 364;

/// Handoff: `install` is the body that card 'raa-gi screen: ScreenPart, gi.screen pass and install'
/// replaces. The generated `install_parts` of `parts.rs` calls it once per GI feature install, in
/// file-name order. It builds the pipelines, the part (with the lit handle whose jitter the pass
/// writes into the frame record) and the `gi.screen` pass.
pub(crate) fn install(ctx: &mut PartsCtx<'_>) -> Result<(), RaaError> {
    let pipelines = ScreenPipelines::new(
        ctx.gpu,
        &ctx.shared.compute_group1_layout(),
        &ctx.shared.tunables(),
        (
            ctx.shared.emissive_sources().screen,
            ctx.shared.emissive_generation(),
        ),
    );
    let part = Rc::new(ScreenPart {
        shared: ctx.shared.clone(),
        lit: ctx.lit.handle(),
        state: RefCell::new(ScreenState {
            pipelines,
            resources: None,
            key: None,
            parity: 0,
            active: false,
            pass_enabled: true,
            output_valid: false,
            wants: false,
            mode: IndirectMode::Probes,
            settings: GiSettings::DEFAULT,
            restir_pipelines: None,
            restir: None,
            restir_valid: false,
            restir_frame: 0,
            stats: None,
        }),
    });
    {
        let mut pipeline = ctx
            .pipeline
            .try_borrow_mut()
            .map_err(|_| RaaError::FrameGraph {
                code: "pipeline.busy",
                message: "the pipeline is borrowed (called while a frame executes)".to_string(),
            })?;
        pipeline.add(Box::new(GiScreenPass {
            part: Rc::clone(&part),
        }))?;
    }
    ctx.add_part(part);
    Ok(())
}

/// The group-2 bind groups of one built `ScreenResources`, cached until the frame buffer, the
/// history view or `invalidate_bindings` changes them.
struct ScreenBindings {
    frame: wgpu::Buffer,
    history: Option<wgpu::TextureView>,
    /// The G-buffer views the cached groups were built from.
    views: GBufferViews,
    /// The voxel-buffer generation binding 6 of the trace group was built with (None: dummy).
    voxels: Option<u64>,
    pyramid: Vec<wgpu::BindGroup>,
    trace: wgpu::BindGroup,
    project: wgpu::BindGroup,
    filter: wgpu::BindGroup,
    integrate: [wgpu::BindGroup; 2],
    /// Per parity, `2 * ATROUS_MAX_PASSES` groups: index `2 i` for pass i into the other filter
    /// texture, `2 i + 1` for pass i into the screen output.
    atrous: [Vec<wgpu::BindGroup>; 2],
}

/// The per-frame views the screen groups bind: the frame record and the lit history, or none
/// before the first frame. [`ScreenResources::work`] rebuilds its cached groups when either
/// changes.
pub(crate) struct ScreenInputs<'a> {
    pub(crate) frame: &'a wgpu::Buffer,
    pub(crate) history: Option<&'a wgpu::TextureView>,
    /// `GiShared::voxel_buffer`: the GI voxel words and their generation, None while GI has none.
    pub(crate) voxels: Option<&'a (wgpu::Buffer, u64)>,
}

/// The readback states of the statistics counters.
const READBACK_IDLE: u8 = 0;
const READBACK_COPIED: u8 = 1;
const READBACK_MAPPING: u8 = 2;
const READBACK_MAPPED: u8 = 3;

/// The statistics of one read-back frame: the mean history length and the rejected share in
/// percent, over one pixel in 16 of the geometry.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct ScreenStats {
    pub history_length_mean: f32,
    pub rejected_percent: f32,
    pub clamped_percent: f32,
}

/// The `ScreenStats` of the four counters (sum of history lengths, rejected, counted, clamped);
/// `None` when no pixel was counted.
pub fn screen_stats(counters: [u32; 4]) -> Option<ScreenStats> {
    let [lengths, rejected, counted, clamped] = counters;
    if counted == 0 {
        return None;
    }
    let counted = counted as f32;
    Some(ScreenStats {
        history_length_mean: lengths as f32 / counted,
        rejected_percent: 100.0 * rejected as f32 / counted,
        clamped_percent: 100.0 * clamped as f32 / counted,
    })
}

/// Handoff: card 'raa-gi screen: ScreenPart, gi.screen pass and install' takes over
/// `ScreenResources`: it calls `ScreenResources::new` for one (viewport, probe_rays),
/// `ScreenResources::work` per frame to obtain the `ScreenWork` it records through
/// `ScreenPipelines::record`, `ScreenResources::write_params`, `ScreenResources::output_view`
/// for the group-1 screen output, `ScreenResources::invalidate_bindings` on the gi.screen resize
/// hook and `ScreenResources::allocated_bytes` for the stats. `ScreenResources` allocates the
/// depth pyramid, the ray atlas, the two probe buffers, the two parity histories and moments,
/// the two a-trous textures, the screen output, the debug record, the statistics counters and
/// their readback, the `GiScreenGpu` uniform, the a-trous records and the dummy history, and
/// caches the group-2 bind groups of contract 'gi-screen-wgsl' until the frame, the history or
/// `invalidate_bindings` changes them. LIFECYCLE: `ScreenPart::prepare_frame` (a GiPart hook, run
/// by `GiHandle::prepare_frame` before the pipeline executes) creates ScreenResources on an
/// active frame when needs_reallocation is true. `GiPart::release` (run by `release_resources`
/// in crates/raa-gi/src/feature.rs on every inactive frame and before a reallocation) drops them.
pub(crate) struct ScreenResources {
    viewport: (u32, u32),
    probe_rays: u32,
    grid: (u32, u32),
    pyramid_sizes: Vec<(u32, u32)>,
    mip_views: Vec<wgpu::TextureView>,
    pyramid_view: wgpu::TextureView,
    atlas_view: wgpu::TextureView,
    probes_raw: wgpu::Buffer,
    probes_filtered: wgpu::Buffer,
    /// The accumulated irradiance (rgb) and occlusion (a), one per parity.
    output_views: [wgpu::TextureView; 2],
    /// The moments records (Rgba32Uint), one per parity.
    meta_views: [wgpu::TextureView; 2],
    /// The a-trous ping-pong textures (irradiance rgb, variance a).
    filter_views: [wgpu::TextureView; 2],
    /// The a-trous result: what binding 14 reads while `spatialPasses` > 0.
    display_view: wgpu::TextureView,
    /// The per-pixel debug record (Rgba32Uint).
    debug_view: wgpu::TextureView,
    counters: wgpu::Buffer,
    readback: wgpu::Buffer,
    readback_state: Arc<AtomicU8>,
    params: wgpu::Buffer,
    atrous_params: wgpu::Buffer,
    dummy_view: wgpu::TextureView,
    /// The 16-B zeroed storage buffer binding 6 reads while `GiShared::voxel_buffer` is None.
    voxel_dummy: wgpu::Buffer,
    bindings: Option<ScreenBindings>,
}

fn texture(
    gpu: &GpuContext,
    label: &str,
    size: (u32, u32),
    mip_level_count: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

fn buffer(gpu: &GpuContext, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn mip_view(texture: &wgpu::Texture, mip: u32) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        base_mip_level: mip,
        mip_level_count: Some(1),
        dimension: Some(wgpu::TextureViewDimension::D2),
        ..Default::default()
    })
}

fn full_view(texture: &wgpu::Texture) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn texture_resource(view: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::TextureView(view)
}

fn group(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    bindings: &[(u32, wgpu::BindingResource<'_>)],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry<'_>> = bindings
        .iter()
        .map(|(binding, resource)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: resource.clone(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &entries,
    })
}

/// The 32 B of `GiScreenGpu`, little-endian: history_valid, screen_traces, parity & 1,
/// debug_view, history_max, clamp_gamma, m_cap, history_min.
pub(crate) fn screen_params_bytes(
    history_valid: bool,
    screen_traces: bool,
    parity: usize,
    settings: &GiSettings,
) -> [u8; SCREEN_PARAMS_SIZE as usize] {
    let words = [
        u32::from(history_valid),
        u32::from(screen_traces),
        (parity & 1) as u32,
        settings.debug_view.code(),
        (settings.history_max.max(1) as f32).to_bits(),
        settings.clamp_gamma.to_bits(),
        settings.restir_m_cap.to_bits(),
        settings.history_min.max(1),
    ];
    let mut bytes = [0u8; SCREEN_PARAMS_SIZE as usize];
    for (index, value) in words.into_iter().enumerate() {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

impl ScreenResources {
    /// Allocates the viewport-sized resources of one (viewport, probe_rays). The viewport axes
    /// are clamped to at least 1.
    pub(crate) fn new(gpu: &GpuContext, viewport: (u32, u32), probe_rays: u32) -> ScreenResources {
        let viewport = (viewport.0.max(1), viewport.1.max(1));
        let found = cpu::probe_grid([viewport.0, viewport.1]);
        let grid = (found[0], found[1]);
        let sizes = pyramid_sizes(viewport);

        let textured = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING;
        let pyramid = texture(
            gpu,
            "raa-gi.screen.pyramid",
            viewport,
            sizes.len() as u32,
            wgpu::TextureFormat::R32Float,
            textured,
        );
        let mip_views = (0..sizes.len())
            .map(|mip| mip_view(&pyramid, mip as u32))
            .collect();
        let pyramid_view = full_view(&pyramid);

        let atlas = texture(
            gpu,
            "raa-gi.screen.atlas",
            (grid.0 * probe_rays, grid.1 * probe_rays),
            1,
            wgpu::TextureFormat::Rgba16Float,
            textured,
        );
        let atlas_view = full_view(&atlas);

        let probe_bytes = PROBE_BYTES * u64::from(grid.0) * u64::from(grid.1);
        let storage = wgpu::BufferUsages::STORAGE;
        let probes_raw = buffer(gpu, "raa-gi.screen.probes", probe_bytes, storage);
        let probes_filtered = buffer(gpu, "raa-gi.screen.probes-filtered", probe_bytes, storage);

        let pair = |label: &str, format: wgpu::TextureFormat| -> [wgpu::TextureView; 2] {
            [
                full_view(&texture(gpu, label, viewport, 1, format, textured)),
                full_view(&texture(gpu, label, viewport, 1, format, textured)),
            ]
        };
        let half = wgpu::TextureFormat::Rgba16Float;
        let words = wgpu::TextureFormat::Rgba32Uint;
        let output_views = pair("raa-gi.screen.history", half);
        let meta_views = pair("raa-gi.screen.moments", words);
        let filter_views = pair("raa-gi.screen.atrous", half);
        let display_view = full_view(&texture(
            gpu,
            "raa-gi.screen.output",
            viewport,
            1,
            half,
            textured,
        ));
        let debug_view = full_view(&texture(
            gpu,
            "raa-gi.screen.debug",
            viewport,
            1,
            words,
            textured,
        ));
        let counters = buffer(
            gpu,
            "raa-gi.screen.counters",
            16,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let readback = buffer(
            gpu,
            "raa-gi.screen.counters-readback",
            16,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        let params = buffer(
            gpu,
            "raa-gi.screen.params",
            SCREEN_PARAMS_SIZE,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let atrous_params = buffer(
            gpu,
            "raa-gi.screen.atrous-params",
            ATROUS_SLOT_BYTES * u64::from(2 * ATROUS_MAX_PASSES),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        for index in 0..ATROUS_MAX_PASSES {
            for last in [false, true] {
                let slot = u64::from(index * 2 + u32::from(last));
                gpu.queue().write_buffer(
                    &atrous_params,
                    slot * ATROUS_SLOT_BYTES,
                    &atrous_record(index, last),
                );
            }
        }

        let dummy = texture(
            gpu,
            "raa-gi.screen.history-dummy",
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
        let dummy_view = full_view(&dummy);
        let voxel_dummy = buffer(
            gpu,
            "raa-gi.screen.voxel-dummy",
            16,
            wgpu::BufferUsages::STORAGE,
        );

        ScreenResources {
            viewport,
            probe_rays,
            grid,
            pyramid_sizes: sizes,
            mip_views,
            pyramid_view,
            atlas_view,
            probes_raw,
            probes_filtered,
            output_views,
            meta_views,
            filter_views,
            display_view,
            debug_view,
            counters,
            readback,
            readback_state: Arc::new(AtomicU8::new(READBACK_IDLE)),
            params,
            atrous_params,
            dummy_view,
            voxel_dummy,
            bindings: None,
        }
    }

    /// The viewport this resource set was allocated for.
    pub(crate) fn viewport(&self) -> (u32, u32) {
        self.viewport
    }

    /// The probe rays this resource set was allocated for.
    pub(crate) fn probe_rays(&self) -> u32 {
        self.probe_rays
    }

    /// The probe grid, `cpu::probe_grid` of the viewport.
    pub(crate) fn grid(&self) -> (u32, u32) {
        self.grid
    }

    /// A clone of the history view of `parity & 1`, the texture accumulated this frame.
    pub(crate) fn output_view(&self, parity: usize) -> wgpu::TextureView {
        self.output_views[parity & 1].clone()
    }

    /// The screen output of this frame: the a-trous result when `passes` > 0, else the history
    /// of `parity`.
    pub(crate) fn screen_view(&self, parity: usize, passes: u32) -> wgpu::TextureView {
        if passes > 0 {
            self.display_view.clone()
        } else {
            self.output_view(parity)
        }
    }

    /// A clone of the debug-record view.
    pub(crate) fn debug_view(&self) -> wgpu::TextureView {
        self.debug_view.clone()
    }

    /// A clone of the all-mip R32Float depth-pyramid view.
    pub(crate) fn pyramid_view(&self) -> wgpu::TextureView {
        self.pyramid_view.clone()
    }

    /// Publishes the `gi.screen` ports of the taps table into `table`: the screen output the
    /// forward pass samples, the internals, and the current parity half of each pair.
    pub(crate) fn publish(&self, table: &ResourceTable, parity: usize, passes: u32) {
        let (w, h) = self.viewport;
        let parity = parity & 1;
        let half = wgpu::TextureFormat::Rgba16Float;
        let words = wgpu::TextureFormat::Rgba32Uint;
        let screen = if passes > 0 {
            &self.display_view
        } else {
            &self.output_views[parity]
        };
        let image = |format, w, h, mips| ImageDesc::new(ImageDim::D2, format, w, h, 1, mips);
        let atlas = (self.grid.0 * self.probe_rays, self.grid.1 * self.probe_rays);
        let pass = "gi.screen";
        table.publish(pass, "gi", screen, image(half, w, h, 1));
        let mips = self.pyramid_sizes.len() as u32;
        let r32 = wgpu::TextureFormat::R32Float;
        table.publish(pass, "pyramid", &self.pyramid_view, image(r32, w, h, mips));
        table.publish(
            pass,
            "atlas",
            &self.atlas_view,
            image(half, atlas.0, atlas.1, 1),
        );
        table.publish_buffer(pass, "probesRaw", &self.probes_raw, self.probes_raw.size());
        let filtered = &self.probes_filtered;
        table.publish_buffer(pass, "probesFiltered", filtered, filtered.size());
        table.publish(
            pass,
            "output",
            &self.output_views[parity],
            image(half, w, h, 1),
        );
        table.publish(
            pass,
            "meta",
            &self.meta_views[parity],
            image(words, w, h, 1),
        );
        table.publish(
            pass,
            "filter",
            &self.filter_views[parity],
            image(half, w, h, 1),
        );
        table.publish(pass, "display", &self.display_view, image(half, w, h, 1));
        table.publish(pass, "debug", &self.debug_view, image(words, w, h, 1));
    }

    /// Writes the 32 B of `GiScreenGpu` (`screen_params_bytes`).
    pub(crate) fn write_params(
        &self,
        gpu: &GpuContext,
        history_valid: bool,
        screen_traces: bool,
        parity: usize,
        settings: &GiSettings,
    ) {
        let bytes = screen_params_bytes(history_valid, screen_traces, parity, settings);
        gpu.queue().write_buffer(&self.params, 0, &bytes);
    }

    /// `resource_bytes` of this resource set's viewport and probe rays.
    pub(crate) fn allocated_bytes(&self) -> u64 {
        resource_bytes(self.viewport, self.probe_rays)
    }

    /// Clears the counters before the pass that accumulates into them.
    pub(crate) fn clear_counters(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.counters, 0, None);
    }

    /// Copies the counters into the readback buffer when it is free.
    pub(crate) fn copy_counters(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.readback_state.load(Ordering::Acquire) == READBACK_IDLE {
            encoder.copy_buffer_to_buffer(&self.counters, 0, &self.readback, 0, 16);
            self.readback_state
                .store(READBACK_COPIED, Ordering::Release);
        }
    }

    /// After the submission that copied the counters: starts mapping the readback buffer.
    pub(crate) fn map_counters(&self) {
        if self.readback_state.load(Ordering::Acquire) != READBACK_COPIED {
            return;
        }
        self.readback_state
            .store(READBACK_MAPPING, Ordering::Release);
        let state = Arc::clone(&self.readback_state);
        self.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let next = if result.is_ok() {
                    READBACK_MAPPED
                } else {
                    READBACK_IDLE
                };
                state.store(next, Ordering::Release);
            });
    }

    /// The counters once mapped (then unmapped, so the next copy can go), else `None`.
    pub(crate) fn take_counters(&self) -> Option<[u32; 4]> {
        if self.readback_state.load(Ordering::Acquire) != READBACK_MAPPED {
            return None;
        }
        let counters = match self.readback.slice(..).get_mapped_range() {
            Ok(view) => std::array::from_fn(|i| {
                u32::from_le_bytes(view[i * 4..i * 4 + 4].try_into().unwrap_or([0; 4]))
            }),
            Err(_) => [0; 4],
        };
        self.readback.unmap();
        self.readback_state.store(READBACK_IDLE, Ordering::Release);
        Some(counters)
    }

    /// Builds the group-2 bind groups of contract 'gi-screen-wgsl' on first use and caches them;
    /// rebuilds them after `invalidate_bindings` or when `frame`, `history`, `voxels` or the
    /// G-buffer `views` differ from the cached one.
    pub(crate) fn work<'a>(
        &'a mut self,
        gpu: &GpuContext,
        pipelines: &ScreenPipelines,
        views: &GBufferViews,
        inputs: ScreenInputs<'_>,
        parity: usize,
        atrous_passes: u32,
    ) -> ScreenWork<'a> {
        let ScreenInputs {
            frame,
            history,
            voxels,
        } = inputs;
        let stale = self.bindings.as_ref().is_none_or(|bindings| {
            bindings.frame != *frame
                || bindings.history.as_ref() != history
                || bindings.views != *views
                || bindings.voxels != voxels.map(|v| v.1)
        });
        if stale {
            let bindings = self.build_bindings(gpu, pipelines, views, frame, history, voxels);
            self.bindings = Some(bindings);
        }
        let bindings = self.bindings.as_ref().expect("bindings built above");
        ScreenWork {
            pyramid: &bindings.pyramid,
            pyramid_sizes: &self.pyramid_sizes,
            trace: &bindings.trace,
            project: &bindings.project,
            filter: &bindings.filter,
            integrate: &bindings.integrate[parity & 1],
            atrous: &bindings.atrous[parity & 1],
            atrous_passes: atrous_passes.min(ATROUS_MAX_PASSES),
            probe_rays: self.probe_rays,
            grid: self.grid,
            viewport: self.viewport,
        }
    }

    /// The buffer binding 6 reads: the GI voxel words, or the zeroed dummy while there are none.
    pub(crate) fn voxel_buffer<'a>(
        &'a self,
        voxels: Option<&'a (wgpu::Buffer, u64)>,
    ) -> &'a wgpu::Buffer {
        voxels.map_or(&self.voxel_dummy, |v| &v.0)
    }

    /// Drops the cached bind groups, so the next `work` builds them again.
    pub(crate) fn invalidate_bindings(&mut self) {
        self.bindings = None;
    }

    fn build_bindings(
        &self,
        gpu: &GpuContext,
        pipelines: &ScreenPipelines,
        views: &GBufferViews,
        frame: &wgpu::Buffer,
        history: Option<&wgpu::TextureView>,
        voxels: Option<&(wgpu::Buffer, u64)>,
    ) -> ScreenBindings {
        let device = gpu.device();
        let mut pyramid = Vec::with_capacity(self.mip_views.len());
        if let Some(first) = self.mip_views.first() {
            pyramid.push(group(
                device,
                "raa-gi.screen.pyramid-copy",
                &pipelines.copy_layout,
                &[
                    (0, texture_resource(&views.depth)),
                    (2, texture_resource(first)),
                ],
            ));
        }
        for index in 1..self.mip_views.len() {
            pyramid.push(group(
                device,
                "raa-gi.screen.pyramid-reduce",
                &pipelines.reduce_layout,
                &[
                    (1, texture_resource(&self.mip_views[index - 1])),
                    (2, texture_resource(&self.mip_views[index])),
                ],
            ));
        }
        let cached_history = history.cloned();
        let history_view = history.unwrap_or(&self.dummy_view);
        let trace = group(
            device,
            "raa-gi.screen.trace",
            &pipelines.trace_layout,
            &[
                (0, frame.as_entire_binding()),
                (1, texture_resource(&views.depth)),
                (2, texture_resource(&views.normal)),
                (3, texture_resource(history_view)),
                (4, texture_resource(&self.pyramid_view)),
                (5, texture_resource(&self.atlas_view)),
                (6, self.voxel_buffer(voxels).as_entire_binding()),
                (7, self.params.as_entire_binding()),
            ],
        );
        let project = group(
            device,
            "raa-gi.screen.project",
            &pipelines.project_layout,
            &[
                (0, frame.as_entire_binding()),
                (1, texture_resource(&views.depth)),
                (2, texture_resource(&views.normal)),
                (3, texture_resource(&self.atlas_view)),
                (5, self.probes_raw.as_entire_binding()),
            ],
        );
        let filter = group(
            device,
            "raa-gi.screen.filter",
            &pipelines.filter_layout,
            &[
                (0, frame.as_entire_binding()),
                (4, self.probes_raw.as_entire_binding()),
                (5, self.probes_filtered.as_entire_binding()),
            ],
        );
        let integrate = std::array::from_fn(|parity| {
            let mut entries = vec![
                (0, frame.as_entire_binding()),
                (1, texture_resource(&views.depth)),
                (2, texture_resource(&views.normal)),
                (5, self.probes_filtered.as_entire_binding()),
            ];
            entries.extend(self.accumulate_entries(views, parity));
            group(
                device,
                "raa-gi.screen.integrate",
                &pipelines.integrate_layout,
                &entries,
            )
        });
        let atrous = std::array::from_fn(|parity| {
            let mut groups = Vec::with_capacity(2 * ATROUS_MAX_PASSES as usize);
            for index in 0..ATROUS_MAX_PASSES {
                for last in [false, true] {
                    let input = &self.filter_views[(index % 2) as usize];
                    let output = if last {
                        &self.display_view
                    } else {
                        &self.filter_views[((index + 1) % 2) as usize]
                    };
                    let slot = u64::from(index * 2 + u32::from(last));
                    groups.push(group(
                        device,
                        "raa-gi.screen.atrous",
                        &pipelines.atrous_layout,
                        &[
                            (0, frame.as_entire_binding()),
                            (1, texture_resource(&views.depth)),
                            (2, texture_resource(&views.normal)),
                            (15, texture_resource(input)),
                            (16, texture_resource(output)),
                            (17, texture_resource(&self.output_views[parity])),
                            (
                                18,
                                wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &self.atrous_params,
                                    offset: slot * ATROUS_SLOT_BYTES,
                                    size: wgpu::BufferSize::new(passes::ATROUS_PARAMS_SIZE),
                                }),
                            ),
                        ],
                    ));
                }
            }
            groups
        });
        ScreenBindings {
            frame: frame.clone(),
            history: cached_history,
            views: views.clone(),
            voxels: voxels.map(|v| v.1),
            pyramid,
            trace,
            project,
            filter,
            integrate,
            atrous,
        }
    }

    /// Clones of the accumulation views and buffers, for the ReSTIR GI denoise groups.
    pub(crate) fn accumulate_views(&self) -> AccumulateViews {
        AccumulateViews {
            outputs: self.output_views.clone(),
            metas: self.meta_views.clone(),
            filter: self.filter_views[0].clone(),
            debug: self.debug_view.clone(),
            counters: self.counters.clone(),
        }
    }

    /// The accumulation bindings 3, 4, 6 to 12 of `parity` (accumulate.wgsl): this parity's
    /// history and moments are written, the other parity's read.
    pub(crate) fn accumulate_entries<'a>(
        &'a self,
        views: &'a GBufferViews,
        parity: usize,
    ) -> Vec<(u32, wgpu::BindingResource<'a>)> {
        let parity = parity & 1;
        vec![
            (3, texture_resource(&views.motion)),
            (4, texture_resource(&self.output_views[1 - parity])),
            (6, texture_resource(&self.output_views[parity])),
            (7, self.params.as_entire_binding()),
            (8, texture_resource(&self.meta_views[1 - parity])),
            (9, texture_resource(&self.meta_views[parity])),
            (10, texture_resource(&self.filter_views[0])),
            (11, texture_resource(&self.debug_view)),
            (12, self.counters.as_entire_binding()),
        ]
    }
}

/// The accumulation targets of one `ScreenResources` the ReSTIR GI denoise binds (clones).
pub(crate) struct AccumulateViews {
    pub(crate) outputs: [wgpu::TextureView; 2],
    pub(crate) metas: [wgpu::TextureView; 2],
    pub(crate) filter: wgpu::TextureView,
    pub(crate) debug: wgpu::TextureView,
    pub(crate) counters: wgpu::Buffer,
}

/// The GPU bytes of one `ScreenResources` of `viewport` and `probe_rays`: the pyramid mips at
/// 4 B per texel, the atlas at 8 B per texel, the two probe buffers, per pixel the two
/// histories (8 B), the two moments records (16 B), the two a-trous textures (8 B), the output
/// (8 B) and the debug record (16 B), then the 16-B counters and readback, the 32-B params, the
/// a-trous records and the 8-B dummy.
pub(crate) fn resource_bytes(viewport: (u32, u32), probe_rays: u32) -> u64 {
    let viewport = (viewport.0.max(1), viewport.1.max(1));
    let pyramid: u64 = pyramid_sizes(viewport)
        .into_iter()
        .map(|(width, height)| u64::from(width) * u64::from(height) * 4)
        .sum();
    let found = cpu::probe_grid([viewport.0, viewport.1]);
    let atlas = u64::from(found[0] * probe_rays) * u64::from(found[1] * probe_rays) * 8;
    let probes = 2 * PROBE_BYTES * u64::from(found[0]) * u64::from(found[1]);
    let pixels = u64::from(viewport.0) * u64::from(viewport.1);
    let per_pixel = 2 * 8 + 2 * 16 + 2 * 8 + 8 + 16;
    let atrous = ATROUS_SLOT_BYTES * u64::from(2 * ATROUS_MAX_PASSES);
    pyramid + atlas + probes + pixels * per_pixel + 16 + 16 + SCREEN_PARAMS_SIZE + atrous + 8
}

/// True when `current` is absent or is not (viewport, probe_rays); the viewport axes are
/// clamped to at least 1 as `ScreenResources::new` does.
pub(crate) fn needs_reallocation(
    current: Option<((u32, u32), u32)>,
    viewport: (u32, u32),
    probe_rays: u32,
) -> bool {
    current != Some(((viewport.0.max(1), viewport.1.max(1)), probe_rays))
}

/// The GI part `screen`: the shared frame state, the lit handle (for the TAA jitter) and the
/// per-frame state of the screen resources.
///
/// Handoff: `ScreenPart` is taken over by card 'raa-gi feature: gi.history copy and binding-14
/// screen output (M1 repair)', which calls `ScreenPart::screen_output` and
/// `ScreenPart::wants_history` through `GiPart`. `ScreenPart::screen_output` returns the screen
/// output of this frame while the part is active, the pass is enabled and `ScreenResources`
/// exist; `ScreenPart::wants_history` is true while the part is active with `screen_traces`
/// requested. The other `GiPart` hooks of the part are `ScreenPart::prepare_frame`,
/// `ScreenPart::invalidate_history`, `ScreenPart::release`, `ScreenPart::allocated_bytes`,
/// `ScreenPart::observe_submission` and `ScreenPart::screen_stats`.
pub(crate) struct ScreenPart {
    shared: GiShared,
    lit: LitHandle,
    state: RefCell<ScreenState>,
}

/// The mutable state of [`ScreenPart`].
struct ScreenState {
    pipelines: ScreenPipelines,
    resources: Option<ScreenResources>,
    key: Option<((u32, u32), u32)>,
    parity: usize,
    active: bool,
    pass_enabled: bool,
    output_valid: bool,
    wants: bool,
    /// The irradiance source of this frame (`GiSettings::indirect`).
    mode: IndirectMode,
    /// The settings of this frame.
    settings: GiSettings,
    /// The ReSTIR GI pipelines, built on the first `restir` frame.
    restir_pipelines: Option<RestirGiPipelines>,
    /// The ReSTIR GI resources of the current viewport, while the mode is `restir`.
    restir: Option<RestirGiResources>,
    /// True when the reservoirs of the previous parity hold the previous frame's reuse.
    restir_valid: bool,
    restir_frame: u32,
    /// The latest statistics read back.
    stats: Option<ScreenStats>,
}

impl GiPart for ScreenPart {
    fn name(&self) -> &'static str {
        "screen"
    }

    /// Allocates `ScreenResources` on an active frame when the (viewport, probe_rays) key
    /// changed, flips the output parity, writes the `GiScreenGpu` params and records whether the
    /// part is active and wants the history. The history is usable only when this part's other
    /// parity holds the previous frame and the camera is continuous (`camera_valid`), whether or
    /// not screen traces copy the lit HDR.
    fn prepare_frame(&self, gpu: &GpuContext, frame: &PartFrame<'_>) {
        let mut state = self.state.borrow_mut();
        if let Some(counters) = state
            .resources
            .as_ref()
            .and_then(ScreenResources::take_counters)
        {
            state.stats = screen_stats(counters);
        }
        let (width, height) = frame.viewport;
        if !frame.active || width == 0 || height == 0 {
            state.active = false;
            state.wants = false;
            return;
        }
        let emissive_generation = self.shared.emissive_generation();
        if state.pipelines.tunables_generation() != self.shared.tunables_generation()
            || state.pipelines.emissive_generation() != emissive_generation
        {
            let tunables = self.shared.tunables();
            let emissive = (self.shared.emissive_sources().screen, emissive_generation);
            state.pipelines.rebuild_tuned(gpu, &tunables, emissive);
        }
        let rays = frame.quality.probe_rays;
        if needs_reallocation(state.key, frame.viewport, rays) {
            state.resources = Some(ScreenResources::new(gpu, frame.viewport, rays));
            state.key = Some(((width.max(1), height.max(1)), rays));
            state.output_valid = false;
        }
        state.parity ^= 1;
        state.settings = frame.settings;
        let history_valid = state.output_valid && frame.camera_valid;
        if let Some(resources) = &state.resources {
            resources.write_params(
                gpu,
                history_valid,
                frame.settings.screen_traces && frame.history_valid,
                state.parity,
                &frame.settings,
            );
        }
        state.mode = frame.settings.indirect;
        if state.mode == IndirectMode::Restir {
            if state.restir_pipelines.is_none() {
                let group1 = self.shared.compute_group1_layout();
                state.restir_pipelines = Some(RestirGiPipelines::new(gpu, &group1, &self.shared));
            }
            if let Some(pipelines) = state.restir_pipelines.as_mut() {
                pipelines.refresh(gpu, &self.shared);
            }
            let viewport = (width.max(1), height.max(1));
            if state
                .restir
                .as_ref()
                .is_none_or(|r| r.viewport() != viewport)
            {
                state.restir = Some(RestirGiResources::new(gpu, viewport));
                state.restir_valid = false;
            }
            state.restir_frame = state.restir_frame.wrapping_add(1);
            let usable = state.restir_valid && frame.camera_valid;
            if let Some(restir) = &state.restir {
                restir.write_params(
                    gpu,
                    state.restir_frame,
                    usable,
                    frame.settings.screen_traces && frame.history_valid,
                    frame.settings.restir_m_cap,
                    &self.shared,
                );
            }
        } else {
            state.restir = None;
            state.restir_valid = false;
        }
        state.active = true;
        state.wants = frame.settings.screen_traces;
    }

    /// The screen output of the current frame while the part is active, the pass is enabled and
    /// resources exist; otherwise `None`.
    fn screen_output(&self) -> Option<wgpu::TextureView> {
        let state = self.state.borrow();
        if !state.active || !state.pass_enabled {
            return None;
        }
        state
            .resources
            .as_ref()
            .map(|resources| resources.screen_view(state.parity, state.settings.spatial_passes))
    }

    /// True while the part is active with screen traces requested.
    fn wants_history(&self) -> bool {
        let state = self.state.borrow();
        state.active && state.wants
    }

    /// Marks the parity output invalid, so the next integrate ignores it.
    fn invalidate_history(&self) {
        let mut state = self.state.borrow_mut();
        state.output_valid = false;
        state.restir_valid = false;
    }

    /// Drops the resources and the key so the next active frame reallocates them, and drops the
    /// published depth-pyramid, probe-output and debug views.
    fn release(&self) {
        let mut state = self.state.borrow_mut();
        state.resources = None;
        state.key = None;
        state.active = false;
        state.output_valid = false;
        state.restir = None;
        state.restir_valid = false;
        state.stats = None;
        self.shared.set_pyramid(None);
        self.shared.set_probe_output(None);
        self.shared.set_screen_debug(None);
    }

    /// The GPU bytes of the current resources, or 0 without them.
    fn allocated_bytes(&self) -> u64 {
        let state = self.state.borrow();
        state
            .resources
            .as_ref()
            .map_or(0, ScreenResources::allocated_bytes)
            + state
                .restir
                .as_ref()
                .map_or(0, RestirGiResources::allocated_bytes)
    }

    /// Starts mapping the counters a completed submission copied.
    fn observe_submission(&self) {
        if let Some(resources) = self.state.borrow().resources.as_ref() {
            resources.map_counters();
        }
    }

    /// The latest statistics read back, while the part is active.
    fn screen_stats(&self) -> Option<ScreenStats> {
        let state = self.state.borrow();
        if state.active { state.stats } else { None }
    }
}

/// The `gi.screen` feature pass (order 42): records the screen compute pass of one
/// [`ScreenPart`].
struct GiScreenPass {
    part: Rc<ScreenPart>,
}

impl FeaturePass for GiScreenPass {
    fn name(&self) -> &str {
        "gi.screen"
    }

    fn order(&self) -> u32 {
        42
    }

    /// Stores the enable flag and, on disable, invalidates the output, drops the published
    /// views and clears the group-1 GI dummies (S38).
    fn set_enabled(&mut self, enabled: bool, lighting: &mut LightingResources) {
        let mut state = self.part.state.borrow_mut();
        state.pass_enabled = enabled;
        if !enabled {
            state.output_valid = false;
            self.part.shared.set_pyramid(None);
            self.part.shared.set_probe_output(None);
            self.part.shared.set_screen_debug(None);
            lighting.clear_gi();
        }
    }

    /// Drops the cached group-2 bind groups after the targets were recreated.
    fn resize(&mut self, _gpu: &GpuContext, _targets: &Targets) {
        let mut state = self.part.state.borrow_mut();
        if let Some(resources) = state.resources.as_mut() {
            resources.invalidate_bindings();
        }
        if let Some(restir) = state.restir.as_mut() {
            restir.invalidate_bindings();
        }
    }

    /// Writes this frame's TAA jitter (known once the lit renderer prepared the frame) into the
    /// two spare lanes of the GI frame record, after `GiHandle::prepare_frame` wrote the record.
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let state = self.part.state.borrow();
        if !state.active || !state.pass_enabled {
            return Ok(());
        }
        if let Some(frame) = self.part.shared.frame_buffer() {
            let [x, y] = self.part.lit.camera_jitter_ndc();
            ctx.gpu
                .queue()
                .write_buffer(&frame, FRAME_JITTER_X_OFFSET, &x.to_le_bytes());
            ctx.gpu
                .queue()
                .write_buffer(&frame, FRAME_JITTER_Y_OFFSET, &y.to_le_bytes());
        }
        if let Some(resources) = state.resources.as_ref() {
            let passes = state.settings.spatial_passes.min(ATROUS_MAX_PASSES);
            resources.publish(ctx.resources, state.parity, passes);
        }
        Ok(())
    }

    /// Drops the published views first. Then records the `raa-gi.screen` pass when the part is
    /// active, enabled and has resources; when it recorded, publishes the pyramid, the screen
    /// output and the debug record of this frame and marks the history valid.
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        self.part.shared.set_pyramid(None);
        self.part.shared.set_probe_output(None);
        self.part.shared.set_screen_debug(None);
        let mut state = self.part.state.borrow_mut();
        if !state.active || !state.pass_enabled {
            return Ok(());
        }
        let ScreenState {
            pipelines,
            resources,
            parity,
            output_valid,
            mode,
            settings,
            restir_pipelines,
            restir,
            restir_valid,
            ..
        } = &mut *state;
        let Some(resources) = resources.as_mut() else {
            return Ok(());
        };
        let Some(frame) = self.part.shared.frame_buffer() else {
            return Ok(());
        };
        let Some(group1) = self.part.shared.compute_group1(ctx.lighting) else {
            return Ok(());
        };
        let passes = settings.spatial_passes.min(ATROUS_MAX_PASSES);
        let history = self.part.shared.history_view();
        let voxels = self.part.shared.voxel_buffer();
        let views = GBufferViews::resolve(ctx.targets, ctx.resources, "gi.screen");
        resources.clear_counters(ctx.encoder);
        let recorded = if *mode == IndirectMode::Restir
            && let (Some(restir_pipelines), Some(restir)) =
                (restir_pipelines.as_ref(), restir.as_mut())
        {
            // ReSTIR GI: the pyramid, the three ReSTIR stages (the third accumulates into this
            // parity's history), then the a-trous passes.
            let history_view = history
                .clone()
                .unwrap_or_else(|| resources.dummy_view.clone());
            let pyramid = resources.pyramid_view.clone();
            let voxel_words = resources.voxel_buffer(voxels.as_ref()).clone();
            let screen_params = resources.params.clone();
            let inputs = RestirGiInputs {
                frame: &frame,
                history: &history_view,
                voxels: &voxel_words,
                pyramid: &pyramid,
                screen_params: &screen_params,
                accumulate: resources.accumulate_views(),
            };
            let restir_work = restir.work(ctx.gpu, restir_pipelines, &views, &inputs, *parity);
            let work = resources.work(
                ctx.gpu,
                pipelines,
                &views,
                ScreenInputs {
                    frame: &frame,
                    history: history.as_ref(),
                    voxels: voxels.as_ref(),
                },
                *parity,
                passes,
            );
            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("raa-gi.screen.restir-gi"),
                    timestamp_writes: None,
                });
            pass.set_bind_group(1, &group1, &[]);
            let recorded = pipelines.record_pyramid(&mut pass, &work);
            if recorded {
                restir_pipelines.record(&mut pass, &restir_work);
                pipelines.record_atrous(&mut pass, &work);
                *restir_valid = true;
            }
            recorded
        } else {
            let work = resources.work(
                ctx.gpu,
                pipelines,
                &views,
                ScreenInputs {
                    frame: &frame,
                    history: history.as_ref(),
                    voxels: voxels.as_ref(),
                },
                *parity,
                passes,
            );
            pipelines.record(ctx.encoder, &group1, &work)
        };
        if recorded {
            resources.copy_counters(ctx.encoder);
            *output_valid = true;
            self.part.shared.set_pyramid(Some(resources.pyramid_view()));
            self.part
                .shared
                .set_probe_output(Some(resources.screen_view(*parity, passes)));
            self.part
                .shared
                .set_screen_debug(Some(resources.debug_view()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_pipeline::PublishedImage;

    #[test]
    fn resource_bytes_matches_the_hand_derivation() {
        // 1280 x 720, 8 rays. Pyramid: sum over the 11 mips of w x h x 4 = 4,915,052 (the S39
        // figure before the accumulation records); atlas 80 x 8 by 45 x 8 texels x 8 B =
        // 1,843,200; probes 2 x 176 x 80 x 45 = 1,267,200. Per pixel 2 x 8 + 2 x 16 + 2 x 8 + 8
        // + 16 = 88 B over 921,600 pixels = 81,100,800. Then 16 + 16 + 32 + 10 x 256 + 8 = 2,632.
        let pyramid: u64 = pyramid_sizes((1280, 720))
            .into_iter()
            .map(|(w, h)| u64::from(w) * u64::from(h) * 4)
            .sum();
        assert_eq!(pyramid, 4_915_052);
        assert_eq!(
            resource_bytes((1280, 720), 8),
            4_915_052 + 1_843_200 + 1_267_200 + 81_100_800 + 2_632
        );
    }

    #[test]
    fn needs_reallocation_tracks_the_key() {
        assert!(needs_reallocation(None, (1280, 720), 8));
        let current = Some(((1280, 720), 8));
        assert!(!needs_reallocation(current, (1280, 720), 8));
        assert!(needs_reallocation(current, (1920, 1080), 8));
        assert!(needs_reallocation(current, (1280, 720), 6));
    }

    #[test]
    fn screen_stats_average_the_counted_pixels() {
        // 50 pixels counted with history lengths summing to 1600 (mean 32), 5 rejected (10 %),
        // 2 clamped (4 %).
        let stats = screen_stats([1600, 5, 50, 2]).expect("pixels were counted");
        assert_eq!(stats.history_length_mean, 32.0);
        assert_eq!(stats.rejected_percent, 10.0);
        assert_eq!(stats.clamped_percent, 4.0);
        assert_eq!(screen_stats([0, 0, 0, 0]), None);
    }

    #[test]
    fn screen_params_follow_the_wgsl_record() {
        let settings = GiSettings {
            history_max: 128,
            clamp_gamma: 3.0,
            restir_m_cap: 30.0,
            ..GiSettings::DEFAULT
        };
        let bytes = screen_params_bytes(true, false, 3, &settings);
        let word = |i: usize| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!([word(0), word(1), word(2), word(3)], [1, 0, 1, 0]);
        assert_eq!(f32::from_bits(word(4)), 128.0);
        assert_eq!(f32::from_bits(word(5)), 3.0);
        assert_eq!(f32::from_bits(word(6)), 30.0);
    }

    /// Builds `gi.screen` through `install` (the pipeline hands the pass back through
    /// `Pipeline::remove`), runs its `resize` then `prepare` against an 8x8 `Targets` and a fresh
    /// `ResourceTable` before and after the part allocated its resources, and checks every tap.
    #[test]
    fn taps_publish_the_screen_ports() {
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            println!("SKIP taps_publish_the_screen_ports: software_gpu returned None");
            return;
        };
        let surface = wgpu::TextureFormat::Rgba8Unorm;
        let pipeline = Rc::new(RefCell::new(raa_pipeline::Pipeline::new(
            &gpu, surface, false,
        )));
        let opts = raa_lit::LitOptions::default();
        let mut lit = raa_lit::LitRenderer::new(&gpu, &pipeline, opts).expect("lit renderer");
        let shared = GiShared::new(&gpu);
        let mut ctx = PartsCtx::new(&gpu, &pipeline, &mut lit, None, None, shared);
        install(&mut ctx).expect("install");
        let (parts, _) = ctx.take();
        let mut pass = pipeline
            .borrow_mut()
            .remove("gi.screen")
            .expect("gi.screen");
        let targets = Targets::new(&gpu, (8, 8));
        let mut lighting = LightingResources::new(&gpu);
        pass.set_enabled(true, &mut lighting);
        let mut run = |table: &ResourceTable| {
            pass.resize(&gpu, &targets);
            let mut prepare = PrepareCtx {
                gpu: &gpu,
                size: (8, 8),
                frame: 0,
                time_s: 0.0,
                dt_s: 0.0,
                targets: &targets,
                lighting: &mut lighting,
                resources: table,
            };
            pass.prepare(&mut prepare).expect("prepare");
        };
        let before = ResourceTable::new();
        run(&before);
        assert!(before.entries().is_empty() && before.buffer_entries().is_empty());
        let quality = crate::settings::GiQuality::default();
        let frame = PartFrame {
            active: true,
            settings: GiSettings::DEFAULT,
            quality,
            viewport: (8, 8),
            members: &[],
            history_valid: false,
            camera_valid: false,
        };
        parts[0].prepare_frame(&gpu, &frame);
        let table = ResourceTable::new();
        run(&table);
        let half = wgpu::TextureFormat::Rgba16Float;
        let words = wgpu::TextureFormat::Rgba32Uint;
        let image = |format, w, h, mips| ImageDesc::new(ImageDim::D2, format, w, h, 1, mips);
        let [gx, gy] = cpu::probe_grid([8, 8]);
        let rays = quality.probe_rays;
        let mips = pyramid_sizes((8, 8)).len() as u32;
        assert!(mips > 1, "the pyramid has mips");
        let expected = [
            ("gi", image(half, 8, 8, 1)),
            ("pyramid", image(wgpu::TextureFormat::R32Float, 8, 8, mips)),
            ("atlas", image(half, gx * rays, gy * rays, 1)),
            ("output", image(half, 8, 8, 1)),
            ("meta", image(words, 8, 8, 1)),
            ("filter", image(half, 8, 8, 1)),
            ("display", image(half, 8, 8, 1)),
            ("debug", image(words, 8, 8, 1)),
        ];
        for (port, desc) in expected {
            let found = table.get("gi.screen", port).map(|image| image.desc);
            assert_eq!(found, Some(desc), "{port}");
        }
        assert_eq!(table.entries().len(), expected.len());
        let probe_bytes = PROBE_BYTES * u64::from(gx) * u64::from(gy);
        for port in ["probesRaw", "probesFiltered"] {
            let size = table.buffer("gi.screen", port).map(|(_, size)| size);
            assert_eq!(size, Some(probe_bytes), "{port}");
        }
        assert_eq!(table.buffer_entries().len(), 2);
    }

    /// Resolves the four G-buffer views from a table: an empty one gives today's targets, a
    /// bound Depth32Float `depth` input replaces only `depth`, and `ScreenResources::work`
    /// rebuilds its cached bind groups (its key holds the new depth view).
    #[test]
    fn resolve_and_work_follow_the_bound_depth_input() {
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            println!(
                "SKIP resolve_and_work_follow_the_bound_depth_input: software_gpu returned None"
            );
            return;
        };
        let targets = Targets::new(&gpu, (8, 8));
        let empty = ResourceTable::new();
        let plain = GBufferViews::resolve(&targets, &empty, "gi.screen");
        assert_eq!(plain.depth, *targets.depth());
        assert_eq!(plain.normal, *targets.normal());
        assert_eq!(plain.material, *targets.material());
        assert_eq!(plain.motion, *targets.motion());

        let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("raa-gi.test-bound-depth"),
            size: wgpu::Extent3d {
                width: 8,
                height: 8,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let bound_depth = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let table = ResourceTable::new();
        table.bind_input(
            "gi.screen",
            "depth",
            PublishedImage {
                view: bound_depth.clone(),
                desc: ImageDesc::new(ImageDim::D2, wgpu::TextureFormat::Depth32Float, 8, 8, 1, 1),
            },
        );
        let bound = GBufferViews::resolve(&targets, &table, "gi.screen");
        assert_eq!(bound.depth, bound_depth);
        assert_eq!(bound.normal, plain.normal);
        assert_eq!(bound.material, plain.material);
        assert_eq!(bound.motion, plain.motion);

        let shared = GiShared::new(&gpu);
        let pipelines = ScreenPipelines::new(
            &gpu,
            &shared.compute_group1_layout(),
            &shared.tunables(),
            (false, 0),
        );
        let mut resources = ScreenResources::new(&gpu, (8, 8), 8);
        let frame = buffer(&gpu, "raa-gi.test-frame", 768, wgpu::BufferUsages::UNIFORM);
        let first = {
            let work = resources.work(
                &gpu,
                &pipelines,
                &plain,
                ScreenInputs {
                    frame: &frame,
                    history: None,
                    voxels: None,
                },
                0,
                1,
            );
            (*work.trace).clone()
        };
        let second = {
            let work = resources.work(
                &gpu,
                &pipelines,
                &bound,
                ScreenInputs {
                    frame: &frame,
                    history: None,
                    voxels: None,
                },
                0,
                1,
            );
            (*work.trace).clone()
        };
        assert_ne!(first, second);
        let cached = resources.bindings.as_ref().expect("bindings built above");
        assert!(cached.views == bound);
        assert_eq!(cached.views.depth, bound_depth);
    }
}
