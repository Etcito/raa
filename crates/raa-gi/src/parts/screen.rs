//! The milestone-2 part `screen` of the GI volume (S39 screen probes): the module root of the
//! part, registered by the generated registry in `parts.rs`. Its CPU twins live in `screen/cpu.rs`.
//! Adapted from solstice@7e465b5 `crates/solstice-render/src/gi/screen.rs` and
//! `crates/solstice-render/src/gi/depth_pyramid.rs`. MIT licence, Copyright (c) 2026 SamG-Coder.

#![allow(dead_code)]

use std::cell::RefCell;
use std::rc::Rc;

use self::passes::{PROBE_BYTES, SCREEN_PARAMS_SIZE, ScreenPipelines, ScreenWork, pyramid_sizes};
use super::{GiPart, GiShared, PartFrame, PartsCtx};
use raa_core::{GpuContext, RaaError};
use raa_pipeline::{FeaturePass, LightingResources, RecordCtx, Targets};

#[path = "screen/cpu.rs"]
pub mod cpu;

#[path = "screen/passes.rs"]
pub mod passes;

/// Handoff: `install` is the body that card 'raa-gi screen: ScreenPart, gi.screen pass and install'
/// replaces. Today `install` adds nothing to `ctx` and returns `Ok(())`; the generated
/// `install_parts` of `parts.rs` calls it once per GI feature install, in file-name order.
pub(crate) fn install(ctx: &mut PartsCtx<'_>) -> Result<(), RaaError> {
    let pipelines = ScreenPipelines::new(ctx.gpu, &ctx.shared.compute_group1_layout());
    let part = Rc::new(ScreenPart {
        shared: ctx.shared.clone(),
        state: RefCell::new(ScreenState {
            pipelines,
            resources: None,
            key: None,
            parity: 0,
            active: false,
            pass_enabled: true,
            output_valid: false,
            wants: false,
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
    pyramid: Vec<wgpu::BindGroup>,
    trace: wgpu::BindGroup,
    project: wgpu::BindGroup,
    filter: wgpu::BindGroup,
    integrate: [wgpu::BindGroup; 2],
}

/// Handoff: card 'raa-gi screen: ScreenPart, gi.screen pass and install' takes over
/// `ScreenResources`: it calls `ScreenResources::new` for one (viewport, probe_rays),
/// `ScreenResources::work` per frame to obtain the `ScreenWork` it records through
/// `ScreenPipelines::record`, `ScreenResources::write_params`, `ScreenResources::output_view`
/// for the group-1 screen output, `ScreenResources::invalidate_bindings` on the gi.screen resize
/// hook and `ScreenResources::allocated_bytes` for the stats. Today `ScreenResources` allocates
/// the depth pyramid, the ray atlas, the two probe buffers, the two parity outputs, the
/// `GiScreenGpu` uniform and the dummy history, and caches the group-2 bind groups of contract
/// 'gi-screen-wgsl' until the frame, the history or `invalidate_bindings` changes them. It
/// extends the `ScreenPipelines` Handoff at crates/raa-gi/src/parts/screen/passes.rs:286
/// `ScreenPipelines`. LIFECYCLE: `ScreenPart::prepare_frame` (a GiPart hook, run by
/// `GiHandle::prepare_frame` before the pipeline executes) creates ScreenResources on an active
/// frame when needs_reallocation is true. `GiPart::release` (run by `release_resources` in
/// crates/raa-gi/src/feature.rs on every inactive frame and before a reallocation) drops them.
/// The bind groups that reference pipeline targets are created lazily by `work` from the
/// gi.screen record and dropped by `invalidate_bindings` from the gi.screen resize hook. Per
/// enabled pass in order, the pipeline calls `resize` (only when the targets generation
/// changed) and then `prepare` (crates/raa-pipeline/src/pipeline.rs:881-887), then records every
/// enabled pass in order (pipeline.rs:917-919).
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
    output_views: [wgpu::TextureView; 2],
    params: wgpu::Buffer,
    dummy_view: wgpu::TextureView,
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

impl ScreenResources {
    /// Allocates the viewport-sized resources of one (viewport, probe_rays): the pyramid with
    /// one single-mip view per mip and one all-mip view, the ray atlas, the raw and filtered
    /// probe buffers, the two parity outputs, the `GiScreenGpu` uniform and the 1x1 dummy history
    /// written to (0, 0, 0, 1). The viewport axes are clamped to at least 1.
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

        let outputs = [
            texture(
                gpu,
                "raa-gi.screen.output-a",
                viewport,
                1,
                wgpu::TextureFormat::Rgba16Float,
                textured,
            ),
            texture(
                gpu,
                "raa-gi.screen.output-b",
                viewport,
                1,
                wgpu::TextureFormat::Rgba16Float,
                textured,
            ),
        ];
        let output_views = [full_view(&outputs[0]), full_view(&outputs[1])];

        let params = buffer(
            gpu,
            "raa-gi.screen.params",
            SCREEN_PARAMS_SIZE,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

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
            params,
            dummy_view,
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

    /// A clone of the view of output `parity & 1`, the texture written this frame.
    pub(crate) fn output_view(&self, parity: usize) -> wgpu::TextureView {
        self.output_views[parity & 1].clone()
    }

    /// Writes the 16 B of `GiScreenGpu` as little-endian u32: (history_valid, screen_traces,
    /// parity & 1, 0).
    pub(crate) fn write_params(
        &self,
        gpu: &GpuContext,
        history_valid: bool,
        screen_traces: bool,
        parity: usize,
    ) {
        let values = [
            history_valid as u32,
            screen_traces as u32,
            (parity & 1) as u32,
            0,
        ];
        let mut bytes = [0u8; SCREEN_PARAMS_SIZE as usize];
        for (index, value) in values.into_iter().enumerate() {
            bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        gpu.queue().write_buffer(&self.params, 0, &bytes);
    }

    /// `resource_bytes` of this resource set's viewport and probe rays.
    pub(crate) fn allocated_bytes(&self) -> u64 {
        resource_bytes(self.viewport, self.probe_rays)
    }

    /// Builds the group-2 bind groups of contract 'gi-screen-wgsl' on first use and caches them;
    /// rebuilds them after `invalidate_bindings` or when `frame` or `history` differs from the
    /// cached one. The returned `ScreenWork` carries this set's pyramid sizes, grid, viewport and
    /// probe rays and the integrate group of `parity`.
    pub(crate) fn work<'a>(
        &'a mut self,
        gpu: &GpuContext,
        pipelines: &ScreenPipelines,
        targets: &Targets,
        frame: &wgpu::Buffer,
        history: Option<&wgpu::TextureView>,
        parity: usize,
    ) -> ScreenWork<'a> {
        let stale = self.bindings.as_ref().is_none_or(|bindings| {
            bindings.frame != *frame || bindings.history.as_ref() != history
        });
        if stale {
            self.bindings = Some(self.build_bindings(gpu, pipelines, targets, frame, history));
        }
        let bindings = self.bindings.as_ref().expect("bindings built above");
        ScreenWork {
            pyramid: &bindings.pyramid,
            pyramid_sizes: &self.pyramid_sizes,
            trace: &bindings.trace,
            project: &bindings.project,
            filter: &bindings.filter,
            integrate: &bindings.integrate[parity & 1],
            probe_rays: self.probe_rays,
            grid: self.grid,
            viewport: self.viewport,
        }
    }

    /// Drops the cached bind groups, so the next `work` builds them again.
    pub(crate) fn invalidate_bindings(&mut self) {
        self.bindings = None;
    }

    fn build_bindings(
        &self,
        gpu: &GpuContext,
        pipelines: &ScreenPipelines,
        targets: &Targets,
        frame: &wgpu::Buffer,
        history: Option<&wgpu::TextureView>,
    ) -> ScreenBindings {
        let device = gpu.device();
        let mut pyramid = Vec::with_capacity(self.mip_views.len());
        if let Some(first) = self.mip_views.first() {
            pyramid.push(group(
                device,
                "raa-gi.screen.pyramid-copy",
                &pipelines.copy_layout,
                &[
                    (0, texture_resource(targets.depth())),
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
                (1, texture_resource(targets.depth())),
                (2, texture_resource(targets.normal())),
                (3, texture_resource(history_view)),
                (4, texture_resource(&self.pyramid_view)),
                (5, texture_resource(&self.atlas_view)),
                (7, self.params.as_entire_binding()),
            ],
        );
        let project = group(
            device,
            "raa-gi.screen.project",
            &pipelines.project_layout,
            &[
                (0, frame.as_entire_binding()),
                (1, texture_resource(targets.depth())),
                (2, texture_resource(targets.normal())),
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
            group(
                device,
                "raa-gi.screen.integrate",
                &pipelines.integrate_layout,
                &[
                    (0, frame.as_entire_binding()),
                    (1, texture_resource(targets.depth())),
                    (2, texture_resource(targets.normal())),
                    (3, texture_resource(targets.motion())),
                    (4, texture_resource(&self.output_views[1 - parity])),
                    (5, self.probes_filtered.as_entire_binding()),
                    (6, texture_resource(&self.output_views[parity])),
                    (7, self.params.as_entire_binding()),
                ],
            )
        });
        ScreenBindings {
            frame: frame.clone(),
            history: cached_history,
            pyramid,
            trace,
            project,
            filter,
            integrate,
        }
    }
}

/// The GPU bytes of one `ScreenResources` of `viewport` and `probe_rays`: the pyramid mips at
/// 4 B per texel, the atlas at 8 B per texel, the two probe buffers, the two outputs at 8 B per
/// texel each, the 16-B params and the 8-B dummy. For 1280x720 with 8 rays it is 22,771,076.
pub(crate) fn resource_bytes(viewport: (u32, u32), probe_rays: u32) -> u64 {
    let viewport = (viewport.0.max(1), viewport.1.max(1));
    let pyramid: u64 = pyramid_sizes(viewport)
        .into_iter()
        .map(|(width, height)| u64::from(width) * u64::from(height) * 4)
        .sum();
    let found = cpu::probe_grid([viewport.0, viewport.1]);
    let atlas = u64::from(found[0] * probe_rays) * u64::from(found[1] * probe_rays) * 8;
    let probes = 2 * PROBE_BYTES * u64::from(found[0]) * u64::from(found[1]);
    let outputs = 2 * u64::from(viewport.0) * u64::from(viewport.1) * 8;
    pyramid + atlas + probes + outputs + SCREEN_PARAMS_SIZE + 8
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

/// The GI part `screen`: the shared frame state and the per-frame state of the screen-probe
/// resources.
///
/// Handoff: `ScreenPart` is taken over by card 'raa-gi feature: gi.history copy and binding-14
/// screen output (M1 repair)', which calls `ScreenPart::screen_output` and
/// `ScreenPart::wants_history` through `GiPart` to bind the output written this frame at group 1
/// binding 14 and to copy the previous lit HDR into `GiShared::history_view`.
/// `ScreenPart::screen_output` returns a clone of the output view of the current parity while the
/// part is active, the pass is enabled and `ScreenResources` exist; `ScreenPart::wants_history`
/// is true while the part is active with `screen_traces` requested.
/// The other `GiPart` hooks of the part are `ScreenPart::prepare_frame`,
/// `ScreenPart::invalidate_history`, `ScreenPart::release` and `ScreenPart::allocated_bytes`.
/// It builds on the `ScreenResources` Handoff at crates/raa-gi/src/parts/screen.rs:68
/// `ScreenResources` and the `ScreenPipelines` Handoff at
/// crates/raa-gi/src/parts/screen/passes.rs:286 `ScreenPipelines`. LIFECYCLE:
/// `ScreenPart::prepare_frame` (a GiPart hook, run by `GiHandle::prepare_frame` before the
/// pipeline executes) creates ScreenResources on an active frame when needs_reallocation is
/// true. `GiPart::release` (run by `release_resources` in crates/raa-gi/src/feature.rs on every
/// inactive frame and before a reallocation) drops them. The bind groups that reference pipeline
/// targets are created lazily by `work` from the gi.screen record and dropped by
/// `invalidate_bindings` from the gi.screen resize hook. Per enabled pass in order, the pipeline
/// calls `resize` (only when the targets generation changed) and then `prepare`
/// (crates/raa-pipeline/src/pipeline.rs:881-887), then records every enabled pass in order
/// (pipeline.rs:917-919).
pub(crate) struct ScreenPart {
    shared: GiShared,
    state: RefCell<ScreenState>,
}

/// The mutable state of [`ScreenPart`]: the pipelines and the resources keyed by
/// (viewport, probe_rays), the parity of the output written this frame, and the activity,
/// enable, validity and history flags.
struct ScreenState {
    pipelines: ScreenPipelines,
    resources: Option<ScreenResources>,
    key: Option<((u32, u32), u32)>,
    parity: usize,
    active: bool,
    pass_enabled: bool,
    output_valid: bool,
    wants: bool,
}

impl GiPart for ScreenPart {
    fn name(&self) -> &'static str {
        "screen"
    }

    /// Allocates `ScreenResources` on an active frame when the (viewport, probe_rays) key
    /// changed, flips the output parity, writes the `GiScreenGpu` params and records whether the
    /// part is active and wants the history.
    fn prepare_frame(&self, gpu: &GpuContext, frame: &PartFrame<'_>) {
        let mut state = self.state.borrow_mut();
        let (width, height) = frame.viewport;
        if !frame.active || width == 0 || height == 0 {
            state.active = false;
            state.wants = false;
            return;
        }
        let rays = frame.quality.probe_rays;
        if needs_reallocation(state.key, frame.viewport, rays) {
            state.resources = Some(ScreenResources::new(gpu, frame.viewport, rays));
            state.key = Some(((width.max(1), height.max(1)), rays));
            state.output_valid = false;
        }
        state.parity ^= 1;
        if let Some(resources) = &state.resources {
            resources.write_params(
                gpu,
                state.output_valid,
                frame.settings.screen_traces && frame.history_valid,
                state.parity,
            );
        }
        state.active = true;
        state.wants = frame.settings.screen_traces;
    }

    /// A clone of the output view of the current parity while the part is active, the pass is
    /// enabled and resources exist; otherwise `None`.
    fn screen_output(&self) -> Option<wgpu::TextureView> {
        let state = self.state.borrow();
        if !state.active || !state.pass_enabled {
            return None;
        }
        state
            .resources
            .as_ref()
            .map(|resources| resources.output_view(state.parity))
    }

    /// True while the part is active with screen traces requested.
    fn wants_history(&self) -> bool {
        let state = self.state.borrow();
        state.active && state.wants
    }

    /// Marks the parity output invalid, so the next integrate ignores it.
    fn invalidate_history(&self) {
        self.state.borrow_mut().output_valid = false;
    }

    /// Drops the resources and the key so the next active frame reallocates them.
    fn release(&self) {
        let mut state = self.state.borrow_mut();
        state.resources = None;
        state.key = None;
        state.active = false;
        state.output_valid = false;
    }

    /// The GPU bytes of the current resources, or 0 without them.
    fn allocated_bytes(&self) -> u64 {
        self.state
            .borrow()
            .resources
            .as_ref()
            .map_or(0, ScreenResources::allocated_bytes)
    }
}

/// The `gi.screen` feature pass (order 42): records the screen-probe compute pass of one
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

    /// Stores the enable flag and, on disable, invalidates the output and clears the group-1 GI
    /// dummies (S38).
    fn set_enabled(&mut self, enabled: bool, lighting: &mut LightingResources) {
        let mut state = self.part.state.borrow_mut();
        state.pass_enabled = enabled;
        if !enabled {
            state.output_valid = false;
            lighting.clear_gi();
        }
    }

    /// Drops the cached group-2 bind groups after the targets were recreated.
    fn resize(&mut self, _gpu: &GpuContext, _targets: &Targets) {
        if let Some(resources) = self.part.state.borrow_mut().resources.as_mut() {
            resources.invalidate_bindings();
        }
    }

    /// Records the `raa-gi.screen` pass when the part is active, enabled and has resources;
    /// marks the output valid when `ScreenPipelines::record` recorded it.
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let mut state = self.part.state.borrow_mut();
        if !state.active || !state.pass_enabled {
            return Ok(());
        }
        let ScreenState {
            pipelines,
            resources,
            parity,
            output_valid,
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
        let history = self.part.shared.history_view();
        let work = resources.work(
            ctx.gpu,
            pipelines,
            ctx.targets,
            &frame,
            history.as_ref(),
            *parity,
        );
        if pipelines.record(ctx.encoder, &group1, &work) {
            *output_valid = true;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_bytes_matches_the_hand_derivation() {
        assert_eq!(resource_bytes((1280, 720), 8), 22_771_076);
    }

    #[test]
    fn needs_reallocation_tracks_the_key() {
        assert!(needs_reallocation(None, (1280, 720), 8));
        let current = Some(((1280, 720), 8));
        assert!(!needs_reallocation(current, (1280, 720), 8));
        assert!(needs_reallocation(current, (1920, 1080), 8));
        assert!(needs_reallocation(current, (1280, 720), 6));
    }
}
