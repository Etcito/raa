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
    FeaturePass, HDR_FORMAT, LightingResources, Pipeline, PrepareCtx, RecordCtx, Targets,
};

use crate::TAA_ORDER;
use crate::history::HistoryPair;
use crate::resources::{TaaPipelines, TaaTargets, dispatch_size};
use crate::shaders::TAA_PARAMS_SIZE;

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

struct TaaPass {
    accumulated: Rc<Cell<u32>>,
    pending_invalidate: Rc<Cell<bool>>,
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
        crate::TAA_PASS_NAME
    }

    fn order(&self) -> u32 {
        TAA_ORDER
    }

    /// Creates or ensures the history, rebuilds the bind groups and invalidates.
    fn resize(&mut self, gpu: &GpuContext, targets: &Targets) {
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
        let Some(group) = targets.group(hdr.src_texture(), write_index) else {
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
        let mut p = pipeline.try_borrow_mut().map_err(|_| pipeline_busy())?;
        if p.passes()
            .iter()
            .any(|info| info.name == crate::TAA_PASS_NAME)
        {
            return Err(refusal(
                "pipeline.duplicate_pass",
                "the pass `taa` is already installed",
            ));
        }
        let accumulated = Rc::new(Cell::new(0));
        let pending_invalidate = Rc::new(Cell::new(false));
        p.add(Box::new(TaaPass {
            accumulated: accumulated.clone(),
            pending_invalidate: pending_invalidate.clone(),
            pipelines: TaaPipelines::new(gpu),
            frame: frame.clone(),
            history: None,
            targets: None,
        }))?;
        p.set_enabled(crate::TAA_PASS_NAME, false)?;
        Ok(TaaHandle {
            accumulated,
            pending_invalidate,
        })
    }
}

/// Access to the installed TAA resolve (clones share one state).
#[derive(Clone)]
pub struct TaaHandle {
    accumulated: Rc<Cell<u32>>,
    pending_invalidate: Rc<Cell<bool>>,
}

impl TaaHandle {
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
