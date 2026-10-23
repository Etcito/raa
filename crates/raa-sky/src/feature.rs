//! `SkyFeature::install` and the `SkyHandle` setters: the shared sky state (state.rs) seen by
//! the `sky.env` pass (env_pass.rs) and the `sky.background` contributor (background.rs).
//!
//! The handle logic only touches the shared state, so it is tested without a device.

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::{GpuContext, RaaError};
use raa_ibl::IblHandle;
use raa_lit::LitRenderer;
use raa_pipeline::Pipeline;

use crate::background::SkyBackground;
use crate::desc::SkyDesc;
use crate::env_pass::SkyEnvPass;
use crate::state::{SharedSky, SkyState};

/// `pipeline.busy`: the pipeline or the sky state is borrowed (called while a frame executes).
fn busy(what: &str) -> RaaError {
    RaaError::FrameGraph {
        code: "pipeline.busy",
        message: format!("the {what} is borrowed (called while a frame executes)"),
    }
}

/// Installs the procedural sky on a pipeline and a lit renderer.
pub struct SkyFeature;

impl SkyFeature {
    /// Creates the shared sky state (preset `day`), adds the FeaturePass `sky.env` (order 10,
    /// renders the SKY_ENV cube and filters it through `ibl`) to `pipeline` and the forward
    /// contributor `sky.background` (order 900) to `lit`.
    ///
    /// Refusals: `pipeline.busy` when the pipeline is borrowed, `pipeline.duplicate_pass` when
    /// the sky is already installed (from the pipeline, unchanged).
    pub fn install(
        gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
        lit: &mut LitRenderer,
        ibl: IblHandle,
    ) -> Result<SkyHandle, RaaError> {
        let state = SkyState::shared();
        {
            let mut p = pipeline.try_borrow_mut().map_err(|_| busy("pipeline"))?;
            let pass = SkyEnvPass::new(
                gpu,
                Rc::clone(&state),
                ibl,
                lit.handle(),
                p.lighting().layout(),
            );
            p.add(Box::new(pass))?;
        }
        // `add_contributor` borrows the pipeline itself: the borrow above is released.
        lit.add_contributor(Box::new(SkyBackground::new(Rc::clone(&state))))?;
        Ok(SkyHandle { state })
    }
}

/// Access to the installed sky (clones share one state).
#[derive(Clone)]
pub struct SkyHandle {
    state: SharedSky,
}

impl SkyHandle {
    /// Validates and stores `d`; the env cube and the `EnvGpu` sky fields are rebuilt at the
    /// next frame. `sky.invalid` (naming the field) leaves the current sky unchanged;
    /// `pipeline.busy` when called while a frame executes.
    pub fn set(&self, d: SkyDesc) -> Result<(), RaaError> {
        d.validate()?;
        let mut s = self.state.try_borrow_mut().map_err(|_| busy("sky state"))?;
        s.desc = d;
        s.env_dirty = true;
        s.gpu_dirty = true;
        Ok(())
    }

    /// The current sky description.
    pub fn get(&self) -> SkyDesc {
        // Invariant: the state is only borrowed mutably inside a frame's prepare/record and
        // inside `set`/`mark_dirty`, none of which call back into this handle.
        self.state.borrow().desc.clone()
    }

    /// The sun changed: re-render (and re-filter) the env cube at the next frame.
    pub fn mark_dirty(&self) {
        if let Ok(mut s) = self.state.try_borrow_mut() {
            s.env_dirty = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn handle() -> SkyHandle {
        SkyHandle {
            state: SkyState::shared(),
        }
    }

    fn clean(h: &SkyHandle) {
        let mut s = h.state.borrow_mut();
        s.env_dirty = false;
        s.gpu_dirty = false;
    }

    #[test]
    fn set_get_and_refusal_over_device_free_state() {
        let h = handle();
        assert_eq!(Some(h.get()), SkyDesc::preset("day"));

        clean(&h);
        let night = SkyDesc::preset("night").expect("night preset");
        h.set(night.clone()).expect("valid sky");
        assert_eq!(h.get(), night);
        assert!(h.state.borrow().env_dirty && h.state.borrow().gpu_dirty);
        assert_eq!(h.clone().get(), night, "clones share the state");

        clean(&h);
        let mut bad = SkyDesc::preset("snow").expect("snow preset");
        bad.clouds.coverage = 2.0;
        let err = h.set(bad).expect_err("coverage out of range");
        assert_eq!(err.code(), "sky.invalid");
        assert!(err.to_string().contains("clouds.coverage"), "{err}");
        assert_eq!(h.get(), night, "a refusal changes nothing");
        let s = h.state.borrow();
        assert!(!s.env_dirty && !s.gpu_dirty, "a refusal marks nothing");
    }

    #[test]
    fn mark_dirty_marks_only_the_env() {
        let h = handle();
        clean(&h);
        h.mark_dirty();
        let s = h.state.borrow();
        assert!(s.env_dirty && !s.gpu_dirty);
    }

    #[test]
    fn set_while_borrowed_is_busy() {
        let h = handle();
        let _frame = h.state.borrow_mut();
        let day = SkyDesc::preset("day").expect("day preset");
        let err = h.set(day).expect_err("busy");
        assert_eq!(err.code(), "pipeline.busy");
    }
}
