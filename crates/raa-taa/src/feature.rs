//! `TaaFeature::install` and the `TaaHandle`: the `taa` pass (S12, order 100).
//!
//! The pass creates its [`HistoryPair`] and [`TaaTargets`] in `resize` (every resize
//! invalidates the history), writes `RaaTaaParams`, resolves `hdr.src()` into `hdr.dst()` and
//! the history write view, then swaps and commits. It is installed disabled; the pass and
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

/// Handoff: `TaaPass` is extended by card 'raa-taa: history input read when bound';
/// record currently resolves bound HDR views using its own history pair and publishes hdr.src().
struct TaaPass {
    name: String,
    bound: Option<(wgpu::Texture, wgpu::Texture, [wgpu::BindGroup; 2])>,
    accumulated: Rc<Cell<u32>>,
    pending_invalidate: Rc<Cell<bool>>,
    tunables: Rc<Cell<[f64; 6]>>,
    generation: Rc<Cell<u64>>,
    built_generation: u64,
    pipelines: TaaPipelines,
    frame: wgpu::Buffer,
    history: Option<HistoryPair>,
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

    /// Dispatches the resolve for the HDR source and write parity, swaps and commits.
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let Some(targets) = self.targets.as_ref() else {
            return Err(RaaError::Resource(
                "taa: record ran before a resize created the targets".to_owned(),
            ));
        };
        let Some(write_index) = self
            .history
            .as_ref()
            .map(|history| history.state().write_index())
        else {
            return Err(RaaError::Resource(
                "taa: record ran before a resize created the targets".to_owned(),
            ));
        };
        let hdr = ctx.targets.hdr();
        if hdr.is_bound()
            && self
                .bound
                .as_ref()
                .is_none_or(|(src, dst, _)| src != hdr.src_texture() || dst != hdr.dst_texture())
            && let Some(history) = self.history.as_ref()
        {
            self.bound = Some((
                hdr.src_texture().clone(),
                hdr.dst_texture().clone(),
                TaaTargets::bound_groups(
                    ctx.gpu,
                    &self.pipelines,
                    ctx.targets,
                    &self.frame,
                    history,
                ),
            ));
        }
        let group = if hdr.is_bound() {
            self.bound
                .as_ref()
                .map(|(_, _, groups)| &groups[write_index & 1])
        } else {
            targets.group(hdr.src_texture(), write_index)
        };
        let Some(group) = group else {
            return Err(RaaError::Resource(
                "taa: the HDR source is not a texture the pass was resized for".to_owned(),
            ));
        };
        let groups = dispatch_size(targets.size());
        {
            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("raa-taa"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(&self.pipelines.pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, 1);
        }
        hdr.swap();
        if let Some(history) = self.history.as_mut() {
            history.state_mut().commit();
            self.accumulated.set(history.state().accumulated_frames());
        }
        let (w, h) = targets.size();
        let desc = ImageDesc::new(ImageDim::D2, HDR_FORMAT, w, h, 1, 1);
        ctx.resources.publish(&self.name, "color", hdr.src(), desc);
        Ok(())
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

    /// Frames the pass recorded since its last invalidation (0 at install, after a resize,
    /// after disabling and after `invalidate`).
    pub fn accumulated_frames(&self) -> u32 {
        self.accumulated.get()
    }

    /// True when at least one frame was recorded since the last invalidation.
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
}
