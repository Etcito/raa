//! `AutoExposureFeature::install` and the `AutoExposureHandle`: the `exposure.auto` pass (S46,
//! order 125, installed disabled).
//!
//! Each prepared frame maps the readback slots copied by earlier frames, decides the snap with
//! the tracker and writes the 48 B `ExposureParams`. Each recorded frame runs `build_histogram`
//! and `resolve` over the HDR source, draws the source times the adapted exposure into
//! `hdr.dst()`, swaps, and copies the 16 B state into an idle readback slot. The pass and every
//! handle clone share one `Inner`: the description, the enabled mirror and the tracker.

use std::cell::RefCell;
use std::num::NonZeroU64;
use std::rc::Rc;
use std::sync::{Arc, Mutex, MutexGuard};

use raa_core::wgpu;
use raa_core::{GpuContext, Lazy, RaaError};
use raa_pipeline::{FeaturePass, HDR_FORMAT, Pipeline, PrepareCtx, RecordCtx, Targets};

use crate::readback::{AutoExposureState, ExposureTracker, ReadbackSlots};
use crate::{
    AUTO_EXPOSURE_ORDER, AUTO_EXPOSURE_PASS_NAME, AutoExposureDesc, BUILD_HISTOGRAM,
    EXPOSURE_APPLY_WGSL, EXPOSURE_COMPUTE_WGSL, FS_APPLY, HISTOGRAM_BINS, HISTOGRAM_WORKGROUP,
    PARAMS_SIZE, RESOLVE, RESULT_SIZE, SAMPLE_STRIDE, VS_MAIN, pack_params,
};

/// Bytes of the params uniform.
const PARAMS_BYTES: u64 = PARAMS_SIZE as u64;
/// Bytes of the state record, which the readback pool copies.
const STATE_BYTES: u64 = RESULT_SIZE as u64;
/// Bytes of the histogram storage buffer: one `u32` per bin.
const HISTOGRAM_BYTES: u64 = HISTOGRAM_BINS as u64 * 4;

fn refusal(code: &'static str, message: &str) -> RaaError {
    RaaError::FrameGraph {
        code,
        message: message.to_string(),
    }
}

/// The refusal for a pipeline borrowed while a frame executes (raa-lit's form).
fn pipeline_busy() -> RaaError {
    refusal(
        "pipeline.busy",
        "the pipeline is borrowed (called while a frame executes)",
    )
}

/// `pipeline.busy` for the shared exposure state (only reachable by a re-entrant call).
fn state_busy() -> RaaError {
    refusal(
        "pipeline.busy",
        "the exposure state is borrowed (re-entrant call while a frame executes)",
    )
}

/// Locks the tracker and recovers a poisoned lock, as light_stats.rs does.
fn locked(tracker: &Mutex<ExposureTracker>) -> MutexGuard<'_, ExposureTracker> {
    tracker.lock().unwrap_or_else(|e| e.into_inner())
}

/// State shared by the pass and every clone of an `AutoExposureHandle`.
struct Inner {
    desc: AutoExposureDesc,
    /// Mirror of the pipeline flag of `exposure.auto`, set by `set_enabled_flag`.
    enabled_flag: bool,
    /// Frame bookkeeping and the two latest results; the map callbacks push into it.
    tracker: Arc<Mutex<ExposureTracker>>,
}

type SharedState = Rc<RefCell<Inner>>;

/// A bind group layout entry for `binding`, visible to `visibility`.
fn layout_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::BindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty,
        count: None,
    }
}

/// A buffer binding that needs at least `min_size` bytes.
fn buffer(ty: wgpu::BufferBindingType, min_size: u64) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty,
        has_dynamic_offset: false,
        min_binding_size: NonZeroU64::new(min_size),
    }
}

/// The HDR source: a float 2D texture that is only loaded, so it is not filterable.
fn sampled() -> wgpu::BindingType {
    wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float { filterable: false },
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    }
}

/// A bind group entry binding `resource` at `binding`.
fn bind(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
}

/// Group-0 layout of the compute pair: params, HDR source, histogram, state.
fn create_compute_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let stage = wgpu::ShaderStages::COMPUTE;
    let read_write = wgpu::BufferBindingType::Storage { read_only: false };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("raa-exposure.compute_layout"),
        entries: &[
            layout_entry(
                0,
                stage,
                buffer(wgpu::BufferBindingType::Uniform, PARAMS_BYTES),
            ),
            layout_entry(1, stage, sampled()),
            layout_entry(2, stage, buffer(read_write, HISTOGRAM_BYTES)),
            layout_entry(3, stage, buffer(read_write, STATE_BYTES)),
        ],
    })
}

/// Group-0 layout of the apply draw: params, HDR source, state (read-only).
fn create_apply_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let stage = wgpu::ShaderStages::FRAGMENT;
    let read_only = wgpu::BufferBindingType::Storage { read_only: true };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("raa-exposure.apply_layout"),
        entries: &[
            layout_entry(
                0,
                stage,
                buffer(wgpu::BufferBindingType::Uniform, PARAMS_BYTES),
            ),
            layout_entry(1, stage, sampled()),
            layout_entry(2, stage, buffer(read_only, STATE_BYTES)),
        ],
    })
}

/// One compute pipeline of `module` at `entry`, on the compute pair's pipeline layout.
fn create_compute_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    entry: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(layout),
        module,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

/// The fullscreen apply pipeline: `vs_main` and `fs_apply` into `HDR_FORMAT`, no blend.
fn create_apply_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("raa-exposure.apply"),
        source: wgpu::ShaderSource::Wgsl(EXPOSURE_APPLY_WGSL.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("raa-exposure.apply"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(FS_APPLY),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some(VS_MAIN),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some(FS_APPLY),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// Cached bind groups: the two own HDR chain textures plus graph-bound chain images.
const MAX_GROUPS: usize = 8;

/// The cached bind groups of one HDR chain texture: the compute pair and the apply draw read it.
struct TextureGroups {
    texture: wgpu::Texture,
    compute: wgpu::BindGroup,
    apply: wgpu::BindGroup,
}

/// The `exposure.auto` feature pass (S46).
struct AutoExposurePass {
    state: SharedState,
    compute_layout: wgpu::BindGroupLayout,
    apply_layout: wgpu::BindGroupLayout,
    /// The three pipelines are built on the first auto-exposure frame (raa_core::Lazy says why).
    build: Lazy<wgpu::ComputePipeline>,
    resolve: Lazy<wgpu::ComputePipeline>,
    apply: Lazy<wgpu::RenderPipeline>,
    uniform: wgpu::Buffer,
    histogram: wgpu::Buffer,
    result: wgpu::Buffer,
    slots: ReadbackSlots,
    scratch: [u8; PARAMS_SIZE],
    /// One entry per HDR chain texture (A and B, rebuilt by `resize`) and per graph-bound
    /// source seen by `record` (at most MAX_GROUPS in all).
    groups: Vec<TextureGroups>,
}

impl AutoExposurePass {
    /// Builds the layouts, the three pipelines, the buffers (zero at creation) and the readback
    /// pool. The bind groups wait for `resize`.
    fn new(gpu: &GpuContext, state: SharedState) -> Self {
        let device = gpu.device();
        let compute_layout = create_compute_layout(device);
        let apply_layout = create_apply_layout(device);
        let compute_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa-exposure.compute"),
            source: wgpu::ShaderSource::Wgsl(EXPOSURE_COMPUTE_WGSL.into()),
        });
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("raa-exposure.compute"),
                bind_group_layouts: &[Some(&compute_layout)],
                immediate_size: 0,
            });
        let lazy_compute = |entry: &'static str| {
            let (device, layout, module) = (
                device.clone(),
                compute_pipeline_layout.clone(),
                compute_module.clone(),
            );
            Lazy::new(move || create_compute_pipeline(&device, &layout, &module, entry))
        };
        let build = lazy_compute(BUILD_HISTOGRAM);
        let resolve = lazy_compute(RESOLVE);
        let apply = {
            let (device, layout) = (device.clone(), apply_layout.clone());
            Lazy::new(move || create_apply_pipeline(&device, &layout))
        };
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-exposure.params"),
            size: PARAMS_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let histogram = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-exposure.histogram"),
            size: HISTOGRAM_BYTES,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let result = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-exposure.state"),
            size: STATE_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        AutoExposurePass {
            state,
            compute_layout,
            apply_layout,
            build,
            resolve,
            apply,
            uniform,
            histogram,
            result,
            slots: ReadbackSlots::new(device),
            scratch: [0; PARAMS_SIZE],
            groups: Vec::new(),
        }
    }

    /// The compute and apply groups that read `view`, the view of `texture`.
    fn texture_groups(
        &self,
        device: &wgpu::Device,
        texture: &wgpu::Texture,
        view: &wgpu::TextureView,
    ) -> TextureGroups {
        let compute = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-exposure.compute_group"),
            layout: &self.compute_layout,
            entries: &[
                bind(0, self.uniform.as_entire_binding()),
                bind(1, wgpu::BindingResource::TextureView(view)),
                bind(2, self.histogram.as_entire_binding()),
                bind(3, self.result.as_entire_binding()),
            ],
        });
        let apply = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-exposure.apply_group"),
            layout: &self.apply_layout,
            entries: &[
                bind(0, self.uniform.as_entire_binding()),
                bind(1, wgpu::BindingResource::TextureView(view)),
                bind(2, self.result.as_entire_binding()),
            ],
        });
        TextureGroups {
            texture: texture.clone(),
            compute,
            apply,
        }
    }
}

impl FeaturePass for AutoExposurePass {
    fn name(&self) -> &str {
        AUTO_EXPOSURE_PASS_NAME
    }

    fn order(&self) -> u32 {
        AUTO_EXPOSURE_ORDER
    }

    /// Rebuilds the groups only when the HDR chain's textures differ from the cached pair, so a
    /// viewport pass resized every frame allocates nothing after warm-up.
    fn resize(&mut self, gpu: &GpuContext, targets: &Targets) {
        let hdr = targets.hdr();
        let cached = |texture: &wgpu::Texture| self.groups.iter().any(|g| g.texture == *texture);
        if cached(hdr.src_texture()) && cached(hdr.dst_texture()) {
            return;
        }
        let device = gpu.device();
        self.groups = vec![
            self.texture_groups(device, hdr.src_texture(), hdr.src()),
            self.texture_groups(device, hdr.dst_texture(), hdr.dst()),
        ];
    }

    /// Maps the copied slots, decides the snap and writes the params.
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let (desc, tracker) = {
            let st = self.state.try_borrow().map_err(|_| state_busy())?;
            (st.desc, Arc::clone(&st.tracker))
        };
        // Before the tracker lock: the poll may run map callbacks that lock it.
        self.slots.map_copied(ctx.gpu, &tracker);
        let first = locked(&tracker).begin_frame(ctx.frame, ctx.size);
        pack_params(&mut self.scratch, &desc, ctx.dt_s, first, ctx.frame);
        ctx.gpu
            .queue()
            .write_buffer(&self.uniform, 0, &self.scratch);
        Ok(())
    }

    /// Histogram and resolve, the apply draw into hdr.dst(), the swap, then the state copy.
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let hdr = ctx.targets.hdr();
        // A render graph binds its own chain images (`HdrChain::bind`), possibly at another
        // extent after a temporal resolve: their groups are created on first use and kept in a
        // small cache, as raa-sharpen does.
        if !self.groups.iter().any(|g| g.texture == *hdr.src_texture()) {
            if self.groups.len() >= MAX_GROUPS {
                self.groups.remove(0);
            }
            let groups = self.texture_groups(ctx.gpu.device(), hdr.src_texture(), hdr.src());
            self.groups.push(groups);
        }
        let Some(groups) = self.groups.iter().find(|g| g.texture == *hdr.src_texture()) else {
            return Err(RaaError::Resource(
                "exposure.auto: no bind group for the HDR source".to_owned(),
            ));
        };
        // The source's own extent: a graph-bound chain image after a temporal resolve is display
        // sized while `ctx.size` is the scene extent.
        let extent = hdr.src_texture().size();
        let (width, height) = (extent.width, extent.height);
        let x = width.div_ceil(SAMPLE_STRIDE).div_ceil(HISTOGRAM_WORKGROUP);
        let y = height.div_ceil(SAMPLE_STRIDE).div_ceil(HISTOGRAM_WORKGROUP);
        {
            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("raa-exposure.histogram"),
                    timestamp_writes: None,
                });
            pass.set_bind_group(0, &groups.compute, &[]);
            pass.set_pipeline(self.build.get());
            pass.dispatch_workgroups(x, y, 1);
            pass.set_pipeline(self.resolve.get());
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("raa-exposure.apply"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: hdr.dst(),
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(self.apply.get());
            pass.set_bind_group(0, &groups.apply, &[]);
            pass.draw(0..3, 0..1);
        }
        hdr.swap();
        self.slots.copy(ctx.encoder, &self.result);
        Ok(())
    }
}

/// Installs automatic exposure on a pipeline.
pub struct AutoExposureFeature;

impl AutoExposureFeature {
    /// Builds the pipelines, the buffers and the readback pool, adds the pass `exposure.auto`
    /// (order 125) and disables it, so nothing changes until a setter enables it.
    ///
    /// Refusals, with nothing installed: `pipeline.busy` when the pipeline is borrowed, and
    /// `pipeline.duplicate_pass` when `exposure.auto` is already listed.
    pub fn install(
        gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
    ) -> Result<AutoExposureHandle, RaaError> {
        let mut p = pipeline.try_borrow_mut().map_err(|_| pipeline_busy())?;
        if p.entry_enabled(AUTO_EXPOSURE_PASS_NAME).is_some() {
            return Err(refusal(
                "pipeline.duplicate_pass",
                "the pass `exposure.auto` is already installed",
            ));
        }
        let state = Rc::new(RefCell::new(Inner {
            desc: AutoExposureDesc::default(),
            enabled_flag: false,
            tracker: Arc::new(Mutex::new(ExposureTracker::new())),
        }));
        p.add(Box::new(AutoExposurePass::new(gpu, Rc::clone(&state))))?;
        p.set_enabled(AUTO_EXPOSURE_PASS_NAME, false)?;
        Ok(AutoExposureHandle { state })
    }
}

/// Access to the installed automatic exposure. Clones share one state with the pass.
#[derive(Clone)]
pub struct AutoExposureHandle {
    state: SharedState,
}

impl AutoExposureHandle {
    /// The current description: the default until set, and the default on a re-entrant borrow.
    pub fn get(&self) -> AutoExposureDesc {
        self.state
            .try_borrow()
            .map(|st| st.desc)
            .unwrap_or_default()
    }

    /// Replaces the description after `AutoExposureDesc::validate`. A refusal (`post.invalid`,
    /// naming the JS key) changes nothing; `pipeline.busy` on a re-entrant call.
    pub fn set(&self, desc: AutoExposureDesc) -> Result<(), RaaError> {
        desc.validate()?;
        self.state.try_borrow_mut().map_err(|_| state_busy())?.desc = desc;
        Ok(())
    }

    /// Mirrors the pipeline flag of the pass without validation. A re-entrant call leaves the
    /// mirror unchanged.
    pub fn set_enabled_flag(&self, enabled: bool) {
        if let Ok(mut st) = self.state.try_borrow_mut() {
            st.enabled_flag = enabled;
        }
    }

    /// The next prepared frame snaps: it starts from the measured average, not a smoothed one.
    pub fn invalidate(&self) {
        if let Ok(st) = self.state.try_borrow() {
            locked(&st.tracker).invalidate();
        }
    }

    /// The state the tracker reports for the mirrored flag. While the state is borrowed it
    /// reports pending and unsettled, so a host that draws until settled keeps drawing.
    pub fn state(&self) -> AutoExposureState {
        match self.state.try_borrow() {
            Ok(st) => locked(&st.tracker).state(st.enabled_flag),
            Err(_) => AutoExposureState {
                exposure_ev: None,
                average_luminance: None,
                settled: false,
                pending: true,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A handle over a fresh state: the default description, the flag off, no frame seen.
    fn fresh_handle() -> AutoExposureHandle {
        AutoExposureHandle {
            state: Rc::new(RefCell::new(Inner {
                desc: AutoExposureDesc::default(),
                enabled_flag: false,
                tracker: Arc::new(Mutex::new(ExposureTracker::new())),
            })),
        }
    }

    #[test]
    fn set_refuses_an_inverted_band_and_keeps_the_stored_desc() {
        let handle = fresh_handle();
        let valid = AutoExposureDesc {
            compensation_ev: 1.0,
            ..AutoExposureDesc::default()
        };
        handle.set(valid).unwrap();
        let inverted = AutoExposureDesc {
            low_percent: 0.9,
            high_percent: 0.5,
            ..valid
        };
        let err = handle.set(inverted).err().unwrap();
        assert_eq!(err.code(), "post.invalid");
        assert_eq!(handle.get(), valid);
    }

    #[test]
    fn state_is_settled_while_disabled_and_pending_once_the_flag_is_set() {
        let handle = fresh_handle();
        assert_eq!(
            handle.state(),
            AutoExposureState {
                exposure_ev: None,
                average_luminance: None,
                settled: true,
                pending: false,
            }
        );
        handle.set_enabled_flag(true);
        assert_eq!(
            handle.state(),
            AutoExposureState {
                exposure_ev: None,
                average_luminance: None,
                settled: false,
                pending: true,
            }
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    mod gpu {
        use std::cell::RefCell;
        use std::rc::Rc;

        use raa_core::testing::{skip, software_gpu};
        use raa_core::wgpu;
        use raa_pipeline::Pipeline;

        use super::super::*;

        #[test]
        fn install_lists_the_pass_disabled_at_order_125_and_refuses_a_second_install() {
            let name = "install_lists_the_pass_disabled_at_order_125_and_refuses_a_second_install";
            let Some(gpu) = software_gpu() else {
                skip(
                    name,
                    "software_gpu returned None; see the software_gpu line above",
                );
                return;
            };
            let pipeline = Rc::new(RefCell::new(Pipeline::new(
                &gpu,
                wgpu::TextureFormat::Rgba8Unorm,
                false,
            )));
            let handle = AutoExposureFeature::install(&gpu, &pipeline).unwrap();
            let passes = pipeline.borrow().passes();
            let info = passes
                .iter()
                .find(|p| p.name == AUTO_EXPOSURE_PASS_NAME)
                .unwrap();
            assert_eq!(info.order, AUTO_EXPOSURE_ORDER);
            assert!(!info.enabled);
            assert!(handle.state().settled);
            let err = AutoExposureFeature::install(&gpu, &pipeline).err().unwrap();
            assert_eq!(err.code(), "pipeline.duplicate_pass");
        }
    }
}
