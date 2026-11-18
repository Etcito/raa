//! `TaaFeature::install` and the `TaaHandle`: the `taa` pass (S12, order 100).
//!
//! The pass creates its [`HistoryPair`] and [`TaaTargets`] in `resize` (every resize
//! invalidates the history), writes `RaaTaaParams`, resolves `hdr.src()` into `hdr.dst()` and
//! the history write view, then swaps. History advances on submission. It is installed disabled; the pass and
//! every handle clone share a `Cell<u32>` frame count and a `Cell<bool>` pending invalidate.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use raa_core::wgpu;
use raa_core::{GpuContext, RaaError};
use raa_pipeline::{
    FeaturePass, HDR_FORMAT, ImageDesc, ImageDim, LightingResources, Pipeline, PrepareCtx,
    RecordCtx, Targets, validate_tunable,
};

use crate::TAA_ORDER;
use crate::history::HistoryPair;
use crate::resources::{TaaPipelines, TaaTargets, dispatch_size};
use crate::shaders::TAA_PARAMS_SIZE;

const NAME: &str = "taa";
const HISTORY_LABEL: &str = "raa-taa.history";
const PARAMS_LEN: usize = TAA_PARAMS_SIZE as usize;

fn refusal(code: &'static str, message: &str) -> RaaError {
    RaaError::FrameGraph {
        code,
        message: message.to_string(),
    }
}

/// Same refusal as raa-dof's for a pipeline borrowed while a frame executes.
fn pipeline_busy() -> RaaError {
    refusal(
        "pipeline.busy",
        "the pipeline is borrowed (called while a frame executes)",
    )
}

/// `RaaTaaParams` (16 B): `history_usable` is 1 when the history is valid; the rest is zero.
fn pack_params(history_usable: bool) -> [u8; PARAMS_LEN] {
    let mut out = [0u8; PARAMS_LEN];
    out[0..4].copy_from_slice(&u32::from(history_usable).to_le_bytes());
    out
}

/// The key and groups of a bound chain: the HDR (src, dst) textures, the bound history
/// view, the resolved depth and motion views and the two parity bind groups.
type BoundGroups = (
    wgpu::Texture,
    wgpu::Texture,
    Option<wgpu::TextureView>,
    wgpu::TextureView,
    wgpu::TextureView,
    [wgpu::BindGroup; 2],
);

/// Handoff: `TaaPass` is extended by card 'raa-taa: history input read when bound';
/// record currently resolves bound HDR views using its own history pair and publishes hdr.src().
struct TaaPass {
    name: String,
    /// The bound-chain groups with the HDR textures, the bound history view, the depth and
    /// motion views they embed.
    bound: Option<BoundGroups>,
    accumulated: Rc<Cell<u32>>,
    pending_invalidate: Rc<Cell<bool>>,
    tunables: Rc<Cell<[f64; 6]>>,
    generation: Rc<Cell<u64>>,
    built_generation: u64,
    pipelines: TaaPipelines,
    frame: wgpu::Buffer,
    history: Option<HistoryPair>,
    recorded: bool,
    targets: Option<TaaTargets>,
}

impl TaaPass {
    fn invalidate(&mut self) {
        if let Some(history) = self.history.as_mut() {
            history.state_mut().invalidate();
        }
        self.accumulated.set(0);
        self.pending_invalidate.set(false);
    }
}

impl FeaturePass for TaaPass {
    fn name(&self) -> &str {
        &self.name
    }

    fn order(&self) -> u32 {
        TAA_ORDER
    }

    /// Creates or ensures the history, rebuilds the bind groups and invalidates.
    fn resize(&mut self, gpu: &GpuContext, targets: &Targets) {
        self.bound = None;
        let size = targets.size();
        {
            let history = match self.history.as_mut() {
                Some(history) => {
                    history.ensure(gpu, size);
                    history
                }
                None => self
                    .history
                    .insert(HistoryPair::new(gpu, HISTORY_LABEL, HDR_FORMAT, size)),
            };
            self.targets = Some(TaaTargets::new(
                gpu,
                &self.pipelines,
                targets,
                &self.frame,
                history,
            ));
        }
        self.invalidate();
    }

    /// Disabling invalidates the history; enabling does nothing.
    fn set_enabled(&mut self, enabled: bool, _lighting: &mut LightingResources) {
        if !enabled {
            self.invalidate();
        }
    }

    /// Applies a pending invalidate, then writes `RaaTaaParams` from the history validity.
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        self.recorded = false;
        if self.generation.get() != self.built_generation {
            self.built_generation = self.generation.get();
            self.pipelines.rebuild(ctx.gpu, &self.tunables.get());
        }
        if self.pending_invalidate.replace(false) {
            if let Some(history) = self.history.as_mut() {
                history.state_mut().invalidate();
            }
            self.accumulated.set(0);
        }
        let usable = self
            .history
            .as_ref()
            .is_some_and(|history| history.state().is_valid());
        ctx.gpu
            .queue()
            .write_buffer(&self.pipelines.params, 0, &pack_params(usable));
        if let Some(history) = self.history.as_ref() {
            let (w, h) = history.state().size();
            let desc = ImageDesc::new(ImageDim::D2, history.format(), w, h, 1, 1);
            ctx.resources
                .publish(&self.name, "history", history.write_view(), desc);
        }
        Ok(())
    }

    /// Dispatches the resolve for the HDR source and write parity, then swaps HDR. A bound
    /// `history` input supplies the read history (binding 5); the write view stays this pass's.
    /// The `depth` and `motion` inputs (bindings 3 and 4) come from `Targets::depth_input`
    /// and `Targets::motion_input`, and rebuilding keys on them too.
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if self.targets.is_none() {
            return Err(RaaError::Resource(
                "taa: record ran before a resize created the targets".to_owned(),
            ));
        }
        let hdr = ctx.targets.hdr();
        let depth = ctx.targets.depth_input(ctx.resources, &self.name);
        let motion = ctx.targets.motion_input(ctx.resources, &self.name);
        let source_size = (hdr.src_texture().width(), hdr.src_texture().height());
        if [&depth, &motion].iter().any(|guide| {
            let texture = guide.texture();
            (texture.width(), texture.height()) != source_size || texture.sample_count() != 1
        }) {
            return Err(RaaError::Resource(
                "taa: depth and motion must be single-sample guides on the HDR source grid"
                    .to_owned(),
            ));
        }
        let output_size = (hdr.dst_texture().width(), hdr.dst_texture().height());
        let history = self.history.as_mut().ok_or_else(|| {
            RaaError::Resource("taa: record ran before a resize created history".to_owned())
        })?;
        if history.ensure(ctx.gpu, output_size) {
            self.bound = None;
            if !hdr.is_bound() {
                self.targets = Some(TaaTargets::new(
                    ctx.gpu,
                    &self.pipelines,
                    ctx.targets,
                    &self.frame,
                    history,
                ));
            }
            self.accumulated.set(0);
            ctx.gpu
                .queue()
                .write_buffer(&self.pipelines.params, 0, &pack_params(false));
        }
        let write_index = history.state().write_index();
        let written_history = history.write_view().clone();
        if let Some(resolved) = ctx.resources.input(&self.name, crate::NEURAL_COLOR_INPUT)
            && resolved.view.texture() == hdr.dst_texture()
            && resolved.view.texture().usage().contains(wgpu::TextureUsages::COPY_SRC)
            && (resolved.desc.width, resolved.desc.height) == output_size
        {
            let (width, height) = output_size;
            ctx.encoder.copy_texture_to_texture(
                resolved.view.texture().as_image_copy(),
                history.texture(write_index).as_image_copy(),
                wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            );
            hdr.swap();
            self.recorded = true;
            ctx.resources.publish(&self.name, "color", &resolved.view, resolved.desc);
            ctx.resources.publish(&self.name, "history", &written_history, resolved.desc);
            return Ok(());
        }

        let bound_history = ctx
            .resources
            .input(&self.name, "history")
            .filter(|image| (image.desc.width, image.desc.height) == output_size)
            .map(|image| image.view);
        let use_bound = hdr.is_bound()
            || bound_history.is_some()
            || depth != *ctx.targets.depth()
            || motion != *ctx.targets.motion();
        if use_bound
            && self
                .bound
                .as_ref()
                .is_none_or(|(src, dst, history, held_depth, held_motion, _)| {
                    src != hdr.src_texture()
                        || dst != hdr.dst_texture()
                        || history != &bound_history
                        || held_depth != &depth
                        || held_motion != &motion
                })
            && let Some(history) = self.history.as_ref()
        {
            let groups = TaaTargets::bound_groups(
                ctx.gpu,
                &self.pipelines,
                ctx.targets,
                (&depth, &motion),
                &self.frame,
                history,
                bound_history.as_ref(),
            );
            self.bound = Some((
                hdr.src_texture().clone(),
                hdr.dst_texture().clone(),
                bound_history,
                depth,
                motion,
                groups,
            ));
        } else if !use_bound {
            self.bound = None;
        }
        let group = if use_bound {
            self.bound
                .as_ref()
                .map(|(_, _, _, _, _, groups)| &groups[write_index & 1])
        } else {
            self.targets
                .as_ref()
                .and_then(|targets| targets.group(hdr.src_texture(), write_index))
        };
        let Some(group) = group else {
            return Err(RaaError::Resource(
                "taa: the HDR source is not a texture the pass was resized for".to_owned(),
            ));
        };
        let groups = dispatch_size(output_size);
        {
            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("raa-taa"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(self.pipelines.pipeline.get());
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, 1);
        }
        let color = hdr.dst().clone();
        hdr.swap();
        self.recorded = true;
        let (w, h) = output_size;
        let desc = ImageDesc::new(ImageDim::D2, HDR_FORMAT, w, h, 1, 1);
        ctx.resources.publish(&self.name, "color", &color, desc);
        ctx.resources
            .publish(&self.name, "history", &written_history, desc);
        Ok(())
    }
    fn submitted(&mut self) {
        if std::mem::take(&mut self.recorded)
            && let Some(history) = self.history.as_mut()
        {
            history.state_mut().commit();
            self.accumulated.set(history.state().accumulated_frames());
        }
    }

    fn invalidate_history(&mut self) {
        self.recorded = false;
        self.invalidate();
    }
}

/// Installs the TAA resolve pass on a pipeline.
///
/// Handoff: `raa-js: install TAA (m31) and give each viewport its own TAA history` calls
/// `TaaFeature::install` and `TaaHandle::invalidate`; today the pass is installed disabled.
/// Handoff: `raa-taa: software-GPU frame test with the lit renderer` calls
/// `TaaFeature::install`, `TaaHandle::accumulated_frames`, `history_valid` and `invalidate`;
/// today they report and reset the pass's accumulated frames.
pub struct TaaFeature;

impl TaaFeature {
    /// Builds the resolve pipeline and params buffer and adds the FeaturePass `taa`
    /// (order 100) bound to `frame` (the 768-byte S10 FrameGpu uniform), then installs it
    /// disabled.
    ///
    /// Refusals (nothing installed): `pipeline.busy` when the pipeline is borrowed,
    /// `pipeline.duplicate_pass` when `taa` is already installed.
    pub fn install(
        gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
        frame: &wgpu::Buffer,
    ) -> Result<TaaHandle, RaaError> {
        Self::install_named(gpu, pipeline, frame, NAME.to_owned(), false)
    }

    /// Handoff: `TaaFeature::install_instance` is called by card 'Wiring 4/5: multi-instance post passes';
    /// it adds an enabled `taa#<instance>` pass and returns a handle with independent
    /// accumulation, invalidation, tunables, generation and history state.
    pub fn install_instance(
        gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
        frame: &wgpu::Buffer,
        instance: &str,
    ) -> Result<TaaHandle, RaaError> {
        Self::install_named(gpu, pipeline, frame, format!("{NAME}#{instance}"), true)
    }

    fn install_named(
        gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
        frame: &wgpu::Buffer,
        name: String,
        enabled: bool,
    ) -> Result<TaaHandle, RaaError> {
        let mut p = pipeline.try_borrow_mut().map_err(|_| pipeline_busy())?;
        if p.passes().iter().any(|info| info.name == name) {
            return Err(refusal(
                "pipeline.duplicate_pass",
                &format!("the pass `{name}` is already installed"),
            ));
        }
        let accumulated = Rc::new(Cell::new(0));
        let pending_invalidate = Rc::new(Cell::new(false));
        let tunables = Rc::new(Cell::new(crate::tunables::defaults()));
        let generation = Rc::new(Cell::new(0));
        p.add(Box::new(TaaPass {
            name: name.clone(),
            bound: None,
            accumulated: accumulated.clone(),
            pending_invalidate: pending_invalidate.clone(),
            tunables: tunables.clone(),
            generation: generation.clone(),
            built_generation: 0,
            pipelines: TaaPipelines::new(gpu, &tunables.get()),
            frame: frame.clone(),
            history: None,
            recorded: false,
            targets: None,
        }))?;
        if !enabled {
            p.set_enabled(&name, false)?;
        }
        Ok(TaaHandle {
            accumulated,
            pending_invalidate,
            tunables,
            generation,
        })
    }
}

/// Access to the installed TAA resolve (clones share one state).
#[derive(Clone)]
pub struct TaaHandle {
    accumulated: Rc<Cell<u32>>,
    pending_invalidate: Rc<Cell<bool>>,
    tunables: Rc<Cell<[f64; 6]>>,
    generation: Rc<Cell<u64>>,
}

impl TaaHandle {
    /// Sets one tunable of `crate::tunables()`. Today it validates through
    /// `raa_pipeline::validate_tunable`, refuses with `post.invalid` without changing
    /// anything, and otherwise stores the value so the pass rebuilds its pipeline at its
    /// next prepare. The executor reaches the handle through `HandleSet.taa` and passes a
    /// catalog key of pass.taa and its value as f64.
    pub fn set_tunable(&self, key: &str, value: f64) -> Result<(), RaaError> {
        let specs = crate::tunables::tunables();
        validate_tunable(specs, key, value).map_err(|message| refusal("post.invalid", &message))?;
        let Some(index) = specs.iter().position(|s| s.key == key) else {
            return Err(refusal("post.invalid", "unknown tunable"));
        };
        let mut values = self.tunables.get();
        if values[index] != value {
            values[index] = value;
            self.tunables.set(values);
            self.generation.set(self.generation.get() + 1);
        }
        Ok(())
    }

    /// Frames the pass submitted since its last invalidation (0 at install, after a resize,
    /// after disabling and after `invalidate`).
    pub fn accumulated_frames(&self) -> u32 {
        self.accumulated.get()
    }

    /// True when at least one frame was submitted since the last invalidation.
    pub fn history_valid(&self) -> bool {
        self.accumulated.get() > 0
    }

    /// Drops the accumulated frames to 0 at once; the pass invalidates its HistoryState at
    /// its next prepare.
    pub fn invalidate(&self) {
        self.accumulated.set(0);
        self.pending_invalidate.set(true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_core::testing::{skip, software_gpu};
    use raa_pipeline::{PublishedImage, ResourceTable};

    #[test]
    fn pack_params_sets_only_the_history_flag() {
        assert_eq!(
            pack_params(true),
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(pack_params(false), [0; PARAMS_LEN]);
    }

    #[test]
    fn handle_reports_and_resets_accumulation() {
        let handle = TaaHandle {
            accumulated: Rc::new(Cell::new(0)),
            pending_invalidate: Rc::new(Cell::new(false)),
            tunables: Rc::new(Cell::new(crate::tunables::defaults())),
            generation: Rc::new(Cell::new(0)),
        };
        assert_eq!(handle.accumulated_frames(), 0);
        assert!(!handle.history_valid());
        handle.accumulated.set(3);
        assert_eq!(handle.accumulated_frames(), 3);
        assert!(handle.history_valid());
        handle.invalidate();
        assert_eq!(handle.accumulated_frames(), 0);
        assert!(!handle.history_valid());
        assert!(handle.pending_invalidate.get());
    }

    /// A bound `motion` input enters the `bound` cache key; with no input bound the pass
    /// falls back to the resized groups. The test needs no submit: it inspects the key the
    /// pass built and that the unbound record succeeds.
    #[test]
    fn bound_motion_input_keys_the_groups_and_unbound_uses_the_resized_groups() {
        let test = "bound_motion_input_keys_the_groups_and_unbound_uses_the_resized_groups";
        let Some(gpu) = software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let size = (64, 64);
        let targets = Targets::new(&gpu, size);
        let pipelines = TaaPipelines::new(&gpu, &crate::tunables::defaults());
        let frame = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("taa bound key test frame"),
            size: TAA_PARAMS_SIZE.max(768),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let history = HistoryPair::new(&gpu, "taa bound key test", HDR_FORMAT, size);
        let resized = TaaTargets::new(&gpu, &pipelines, &targets, &frame, &history);
        let mut pass = TaaPass {
            name: "taa".to_owned(),
            bound: None,
            accumulated: Rc::new(Cell::new(0)),
            pending_invalidate: Rc::new(Cell::new(false)),
            tunables: Rc::new(Cell::new(crate::tunables::defaults())),
            generation: Rc::new(Cell::new(0)),
            built_generation: 0,
            pipelines,
            frame,
            history: Some(history),
            recorded: false,
            targets: Some(resized),
        };
        let resources = ResourceTable::new();
        let motion_texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("taa bound motion input"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let motion_view = motion_texture.create_view(&Default::default());
        resources.bind_input(
            "taa",
            "motion",
            PublishedImage {
                view: motion_view.clone(),
                desc: ImageDesc::new(
                    ImageDim::D2,
                    wgpu::TextureFormat::Rg16Float,
                    size.0,
                    size.1,
                    1,
                    1,
                ),
            },
        );
        let backbuffer_texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("taa bound key test backbuffer"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let backbuffer = backbuffer_texture.create_view(&Default::default());
        let lighting = LightingResources::new(&gpu);
        let mut encoder = gpu.device().create_command_encoder(&Default::default());
        {
            let mut ctx = RecordCtx {
                gpu: &gpu,
                encoder: &mut encoder,
                size,
                frame: 0,
                time_s: 0.0,
                targets: &targets,
                lighting: &lighting,
                resources: &resources,
                backbuffer: &backbuffer,
                backbuffer_format: wgpu::TextureFormat::Rgba8Unorm,
                surface_is_srgb: false,
            };
            pass.record(&mut ctx)
                .expect("record with a bound motion input");
        }
        let (_, _, _, held_depth, held_motion, _) = pass
            .bound
            .as_ref()
            .expect("the bound groups exist after a bound input");
        assert!(
            held_motion == &motion_view,
            "the bound motion view is in the bound key"
        );
        assert!(
            held_depth == targets.depth(),
            "the unbound depth input is today's target"
        );

        resources.clear();
        let mut encoder = gpu.device().create_command_encoder(&Default::default());
        {
            let mut ctx = RecordCtx {
                gpu: &gpu,
                encoder: &mut encoder,
                size,
                frame: 1,
                time_s: 0.0,
                targets: &targets,
                lighting: &lighting,
                resources: &resources,
                backbuffer: &backbuffer,
                backbuffer_format: wgpu::TextureFormat::Rgba8Unorm,
                surface_is_srgb: false,
            };
            pass.record(&mut ctx).expect("record with no bound input");
        }
        assert!(
            pass.bound.is_none(),
            "no bound input means the resized groups are used"
        );
    }
}
