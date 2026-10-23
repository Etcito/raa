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

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::{GpuContext, RaaError};
use raa_lit::LitRenderer;
use raa_pipeline::{LightingResources, Pipeline};

use crate::filter::{ENV_MIPS, Filter, Sh9};
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

/// State shared by every clone of an `IblHandle`.
struct IblState {
    filter: Filter,
    /// Keeps the LUT texture of the view set on the lighting alive.
    _lut: BrdfLut,
    diffuse: f32,
    specular: f32,
    /// The SH9 last given to `set_env` (None while the env is cleared).
    sh: Option<Sh9>,
    /// The intensities changed since the last `set_env`.
    changed: bool,
}

/// Installs image-based lighting on a pipeline and its lit renderer.
pub struct IblFeature;

impl IblFeature {
    /// Adds the `Ambient` slot provider `ibl` (priority 10) to `lit`, builds the BRDF LUT
    /// once (128x128 Rg16Float split-sum scale and bias, multiscatter) and sets it on the
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
            sh: None,
            changed: false,
        };
        Ok(IblHandle {
            state: Rc::new(RefCell::new(state)),
        })
    }
}

/// Per-frame access to the installed IBL (clones share one state).
#[derive(Clone)]
pub struct IblHandle {
    state: Rc<RefCell<IblState>>,
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
        st.filter.record(gpu, encoder, cube);
        if let Some(e) = st.filter.invalid_source().cloned() {
            lighting.clear_env();
            st.sh = None;
            return Err(e);
        }
        Ok(())
    }

    /// The per-frame hook (`sky.env` prepare): maps the staging buffers written by earlier
    /// `filter` calls (their encoders were submitted since), and when a new SH arrived or
    /// the intensities changed, calls `set_env(output cube, sh, [specular, diffuse, 5, 1])`.
    /// Returns true when a readback was skipped because every staging buffer was busy, so
    /// the caller filters again. Native callers poll the device for the map callbacks; on
    /// wasm the browser drives them.
    pub fn apply_pending(&self, lighting: &mut LightingResources) -> bool {
        let Ok(mut st) = self.state.try_borrow_mut() else {
            return false;
        };
        st.filter.after_submit();
        if let Some(sh) = st.filter.take_sh() {
            st.sh = Some(sh);
            st.changed = true;
        }
        if st.changed
            && let Some(sh) = st.sh
        {
            let info = [st.specular, st.diffuse, MAX_MIP, 1.0];
            lighting.set_env(st.filter.output_view().clone(), sh, info);
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

    /// The SH9 irradiance coefficients last applied by `apply_pending` (rgb, w = 0;
    /// cosine-convolved / pi), None while the env is cleared.
    pub fn sh(&self) -> Option<[[f32; 4]; 9]> {
        self.state.try_borrow().ok().and_then(|st| st.sh)
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
