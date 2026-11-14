//! `IblFeature::install` and the per-frame `IblHandle`: the BRDF LUT (lut.rs) set once on
//! the pipeline's lighting, the `ibl` Ambient provider (provider.rs) added to the lit
//! renderer, and the filter (filter.rs) driven by `sky.env`.
//!
//! Readback timing: `filter` records into the frame's encoder and submits nothing; the map
//! of the staging buffers it wrote must follow that submission, so it is deferred to the
//! next `apply_pending` (called by `sky.env` in its prepare, after the previous frame was
//! submitted). The coefficients land one or more frames later; until the first SH arrives
//! the env stays cleared (`EnvSh.info.w = 0`) and the provider falls back to the flat S1
//! ambient.
//!
//! Lifecycle: every successfully recorded filter request obtains a monotonically increasing
//! generation and the current invalidation epoch. `invalidate` bumps that epoch through a
//! gate outside the `RefCell`-borrowed filter state, clears the published environment and
//! cancels every earlier request, so callbacks that land later can never restore it.
//! `publication`/`publication_error` observe the lifecycle without advancing any work.

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::{GpuContext, RaaError};
use raa_lit::LitRenderer;
use raa_pipeline::{LightingResources, Pipeline};

use crate::filter::{ENV_MIPS, Filter, PublicationGate};
use crate::lut::{BrdfLut, create_brdf_lut};
use crate::provider::ambient_provider;

/// `EnvSh.info.z`: the last prefiltered mip (roughness 1).
const MAX_MIP: f32 = (ENV_MIPS - 1) as f32;

fn refusal(code: &'static str, message: &str) -> RaaError {
    RaaError::FrameGraph {
        code,
        message: message.to_string(),
    }
}

/// Same refusal as raa-lit's for a pipeline borrowed while a frame executes.
fn pipeline_busy() -> RaaError {
    refusal(
        "pipeline.busy",
        "the pipeline is borrowed (called while a frame executes)",
    )
}

/// `pipeline.busy` for the shared IBL state (only reachable by a re-entrant call).
fn state_busy() -> RaaError {
    refusal(
        "pipeline.busy",
        "the IBL state is borrowed (re-entrant call while filtering)",
    )
}

/// The observable lifecycle of the IBL environment.
///
/// `requested_generation` counts successfully recorded filter requests and keeps increasing
/// across invalidation. `published_generation` is the generation last given to
/// `LightingResources::set_env` (None while the environment is cleared). `pending` is true
/// while eligible work of the current epoch (recorded, or completed but not yet applied) is
/// outstanding; it never counts callbacks cancelled by `invalidate`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IblPublication {
    pub requested_generation: u64,
    pub published_generation: Option<u64>,
    pub pending: bool,
}

/// State shared by every clone of an `IblHandle`.
struct IblState {
    filter: Filter,
    /// Keeps the LUT texture of the view set on the lighting alive.
    _lut: BrdfLut,
    diffuse: f32,
    specular: f32,
    /// Scale of the diffuse term, finite and in [0, 1]; published as diffuse x scale.
    diffuse_scale: f32,
    /// The `info` last given to `set_env` by `apply_pending`; None before the first one.
    published_info: Option<[f32; 4]>,
    /// The intensities changed since the last `set_env`.
    changed: bool,
}

/// Installs image-based lighting on a pipeline and its lit renderer.
pub struct IblFeature;

impl IblFeature {
    /// Adds the `Ambient` slot provider `ibl` (priority 10) to `lit`, builds the BRDF LUT
    /// once (BRDF region of the 128 x 192 Rgba16Float atlas, split-sum scale and bias, multiscatter) and sets it on the
    /// pipeline's lighting, and creates the filter with its output cube and staging ring.
    /// The env stays cleared until `IblHandle::apply_pending` applies a first SH.
    ///
    /// Refusals (nothing installed): `pipeline.busy` when the pipeline is borrowed, the
    /// `add_slot` refusal with its code (`shader.slot_conflict` when installed twice).
    pub fn install(
        gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
        lit: &mut LitRenderer,
    ) -> Result<IblHandle, RaaError> {
        let mut p = pipeline.try_borrow_mut().map_err(|_| pipeline_busy())?;
        lit.add_slot(ambient_provider())?;
        let lut = create_brdf_lut(gpu);
        p.lighting_mut().set_brdf_lut(lut.view.clone());
        let state = IblState {
            filter: Filter::new(gpu),
            _lut: lut,
            diffuse: 1.0,
            specular: 1.0,
            diffuse_scale: 1.0,
            published_info: None,
            changed: false,
        };
        Ok(IblHandle {
            state: Rc::new(RefCell::new(state)),
            gate: PublicationGate::new(),
        })
    }
}

/// Per-frame access to the installed IBL (clones share one state).
#[derive(Clone)]
pub struct IblHandle {
    state: Rc<RefCell<IblState>>,
    /// Cancellation/publication gate, independent of the `RefCell` borrow of `state`.
    gate: PublicationGate,
}

impl IblHandle {
    /// Records the filter of `cube` into `encoder` (submitted by the caller): source mips
    /// 1..5 by box downsample, mip 0 copied into the IBL output cube, output mips 1..5 by
    /// GGX prefilter, the SH9 projection and its copy into a free staging buffer (all busy:
    /// the next `apply_pending` returns true). The readback is mapped by the next
    /// `apply_pending`, which must run after this encoder was submitted.
    ///
    /// `cube` must be SKY_ENV (256 px, Rgba16Float, 6 mips) with usage TEXTURE_BINDING |
    /// STORAGE_BINDING | COPY_SRC; otherwise nothing is recorded, the env is cleared (flat
    /// ambient) and `ibl.invalid_source` is returned. `pipeline.busy` on a re-entrant call.
    pub fn filter(
        &self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        cube: &wgpu::Texture,
        lighting: &mut LightingResources,
    ) -> Result<(), RaaError> {
        let mut st = self.state.try_borrow_mut().map_err(|_| state_busy())?;
        let result = st.filter.record(gpu, encoder, cube, &self.gate);
        if result.is_err() {
            st.changed = false;
            lighting.clear_env();
        }
        result
    }

    /// The per-frame hook (`sky.env` prepare): maps the staging buffers written by earlier
    /// `filter` calls (their encoders were submitted since), and when a completed accepted
    /// SH arrived or the intensities changed, calls
    /// `set_env(output cube, sh, [specular, diffuse, 5, 1])` and records that request's
    /// generation. Returns true when a readback was skipped because every staging buffer was
    /// busy, so the caller filters again. Native callers poll the device for the map
    /// callbacks; on wasm the browser drives them.
    pub fn apply_pending(&self, lighting: &mut LightingResources) -> bool {
        let Ok(mut st) = self.state.try_borrow_mut() else {
            return false;
        };
        st.filter.after_submit(&self.gate);
        if let Some((_, sh)) = self.gate.take_publication() {
            let info = [st.specular, st.diffuse * st.diffuse_scale, MAX_MIP, 1.0];
            lighting.set_env(st.filter.output_view().clone(), sh, info);
            st.published_info = Some(info);
            st.changed = false;
        } else if st.changed
            && let Some(sh) = self.gate.published_sh()
        {
            let info = [st.specular, st.diffuse * st.diffuse_scale, MAX_MIP, 1.0];
            lighting.set_env(st.filter.output_view().clone(), sh, info);
            st.published_info = Some(info);
            st.changed = false;
        }
        st.filter.take_retry()
    }

    /// Sets the diffuse and specular IBL intensities (default 1), applied at the next
    /// `apply_pending`. Each value is stored when finite and >= 0 and ignored otherwise
    /// (raa-js refuses such values before they reach here).
    pub fn set_intensity(&self, diffuse: f32, specular: f32) {
        let Ok(mut st) = self.state.try_borrow_mut() else {
            return;
        };
        let valid = |v: f32| v.is_finite() && v >= 0.0;
        if valid(diffuse) {
            st.diffuse = diffuse;
            st.changed = true;
        }
        if valid(specular) {
            st.specular = specular;
            st.changed = true;
        }
    }

    /// Sets the diffuse scale (default 1), applied as diffuse x scale at the next publication.
    /// The value is stored when it is finite and in [0, 1] and ignored otherwise; a borrowed
    /// state is ignored too, as for the intensity setter.
    ///
    /// Handoff: the card "raa-gi slots: Indirect, Ao and LocalVisibility providers and the IBL scale part" takes over this scale and calls `IblHandle::set_diffuse_scale` every time its target scale changes. Today the call only stores the value and marks the state changed; the next `apply_pending` republishes the environment with `info[1]` = the authored diffuse x the stored scale. No caller in `src/` exists yet.
    pub fn set_diffuse_scale(&self, scale: f32) {
        let Ok(mut st) = self.state.try_borrow_mut() else {
            return;
        };
        if scale.is_finite() && (0.0..=1.0).contains(&scale) {
            st.diffuse_scale = scale;
            st.changed = true;
        }
    }

    /// The diffuse scale last stored by the scale setter (1 by default).
    pub fn diffuse_scale(&self) -> f32 {
        self.state.borrow().diffuse_scale
    }

    /// The `info` last given to `LightingResources::set_env` by `apply_pending`
    /// (`[specular, diffuse x diffuse scale, max mip, 1]`), None before the first publication.
    /// A read accessor for tests and diagnostics; `invalidate` does not clear it.
    pub fn published_info(&self) -> Option<[f32; 4]> {
        self.state.borrow().published_info
    }

    /// The SH9 irradiance coefficients last applied by `apply_pending` (rgb, w = 0;
    /// cosine-convolved / pi), None while the env is cleared.
    pub fn sh(&self) -> Option<[[f32; 4]; 9]> {
        self.gate.published_sh()
    }

    /// The current lifecycle observation; the getter never advances filtering or readback.
    pub fn publication(&self) -> IblPublication {
        let snapshot = self.gate.snapshot();
        IblPublication {
            requested_generation: snapshot.requested_generation,
            published_generation: snapshot.published_generation,
            pending: snapshot.pending,
        }
    }

    /// The attributable readback failure of the current environment (code `ibl.readback`),
    /// distinct from pending work. Cleared by a successful publication or `invalidate`.
    pub fn publication_error(&self) -> Option<RaaError> {
        self.gate.error()
    }

    /// Cancels every in-flight request of the current epoch like [`IblFeature::invalidate`] but
    /// leaves the lighting environment as it is: the last published environment keeps lighting
    /// the scene until the replacing one is published, instead of falling back to the flat
    /// ambient term while the new one is filtered over several frames.
    pub fn supersede(&self) {
        self.gate.invalidate();
        if let Ok(mut st) = self.state.try_borrow_mut() {
            st.changed = false;
        }
    }

    /// Invalidates the published environment and every in-flight request of the current
    /// epoch, then clears the lighting environment. Infallible and allocation-free; the
    /// gate is cancelled even when the filter state is borrowed, so a later callback can
    /// never restore the cleared resources. Intensities, the BRDF LUT, the source/output
    /// textures and the monotonically increasing requested-generation counter are kept, and
    /// an outstanding staging slot is never freed before its own callback.
    pub fn invalidate(&self, lighting: &mut LightingResources) {
        self.gate.invalidate();
        if let Ok(mut st) = self.state.try_borrow_mut() {
            st.changed = false;
        }
        lighting.clear_env();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_mip_is_the_last_sky_env_mip() {
        assert_eq!(MAX_MIP, 5.0);
    }
}
