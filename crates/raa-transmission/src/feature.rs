//! `TransmissionFeature::install`: registers [`TRANSMISSIVE_BUILTINS`], the `Transmission`
//! slot provider `screen` (priority 10) and the FeaturePass `opaque.copy` (S12, order 70).
//!
//! Every refusal is checked before the first change (the slot on a copy of the composer, the
//! function names on the registry), so a refused install leaves the pipeline and the lit
//! renderer as they were.

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::{GpuContext, RaaError};
use raa_lit::LitRenderer;
use raa_pipeline::Pipeline;
use raa_shader::ShaderError;

use crate::builtins::TRANSMISSIVE_BUILTINS;
use crate::copy::{NAME, OpaqueCopyPass};
use crate::slot::screen_provider;

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

/// A shader refusal as a [`RaaError`] with the same code and message (raa-js `api_error`).
fn shader_error(e: ShaderError) -> RaaError {
    RaaError::FrameGraph {
        code: e.code,
        message: e.message,
    }
}

/// Installs transmission and refraction on a lit renderer.
pub struct TransmissionFeature;

impl TransmissionFeature {
    /// Registers every [`TRANSMISSIVE_BUILTINS`] row as a built-in material function, adds
    /// the `Transmission` slot provider `screen` (priority 10) to `lit` and the FeaturePass
    /// `opaque.copy` (order 70) to `pipeline`.
    ///
    /// Refusals (nothing installed): `pipeline.busy` when the pipeline is borrowed,
    /// `pipeline.duplicate_pass` when `opaque.copy` is already installed (so a second
    /// install), the `add_slot` refusal with its code (`shader.slot_conflict` when another
    /// provider holds priority 10 on the slot) and the registry refusal with its code
    /// (`material.fn_duplicate` when a function name is already registered).
    pub fn install(
        gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
        lit: &mut LitRenderer,
    ) -> Result<(), RaaError> {
        let mut p = pipeline.try_borrow_mut().map_err(|_| pipeline_busy())?;
        if p.passes().iter().any(|info| info.name == NAME) {
            return Err(refusal(
                "pipeline.duplicate_pass",
                "the pass `opaque.copy` is already installed",
            ));
        }
        let mut probe = lit.composer().clone();
        probe.add_slot(screen_provider()).map_err(shader_error)?;
        let mut fns = lit.fns().clone();
        for b in TRANSMISSIVE_BUILTINS {
            fns.register_builtin(b.name, b.wgsl).map_err(shader_error)?;
        }
        {
            let mut fns = lit.fns_mut();
            for b in TRANSMISSIVE_BUILTINS {
                fns.register_builtin(b.name, b.wgsl).map_err(shader_error)?;
            }
        }
        lit.add_slot(screen_provider())?;
        p.add(Box::new(OpaqueCopyPass::new(gpu, lit.handle())))
    }
}

// GPU tests: `raa_core::testing` is native-only.
#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::copy::ORDER;
    use crate::slot::SCREEN_PROVIDER;
    use raa_core::testing::{skip, software_gpu};
    use raa_core::wgpu;
    use raa_lit::LitOptions;
    use raa_shader::SlotId;

    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

    fn setup(gpu: &GpuContext) -> (Rc<RefCell<Pipeline>>, LitRenderer) {
        let pipeline = Rc::new(RefCell::new(Pipeline::new(gpu, FORMAT, false)));
        let lit = LitRenderer::new(gpu, &pipeline, LitOptions::default()).unwrap();
        (pipeline, lit)
    }

    #[test]
    fn install_registers_builtins_slot_and_pass() {
        let Some(gpu) = software_gpu() else {
            skip(
                "install_registers_builtins_slot_and_pass",
                "software_gpu returned None; see the software_gpu line above",
            );
            return;
        };
        let (pipeline, mut lit) = setup(&gpu);
        TransmissionFeature::install(&gpu, &pipeline, &mut lit).unwrap();
        for b in TRANSMISSIVE_BUILTINS {
            assert!(lit.fns().id_of(b.name).is_some(), "{}", b.name);
        }
        let winner = lit
            .composer()
            .winner(SlotId::Transmission)
            .map(|w| (w.provider, w.priority));
        assert_eq!(winner, Some((SCREEN_PROVIDER, 10)));
        let passes = pipeline.borrow().passes();
        let info = passes.iter().find(|i| i.name == NAME).expect("opaque.copy");
        assert_eq!((info.order, info.required), (ORDER, false));
    }

    #[test]
    fn second_install_refuses_duplicate_pass_and_registers_nothing() {
        let Some(gpu) = software_gpu() else {
            skip(
                "second_install_refuses_duplicate_pass_and_registers_nothing",
                "software_gpu returned None; see the software_gpu line above",
            );
            return;
        };
        let (pipeline, mut lit) = setup(&gpu);
        TransmissionFeature::install(&gpu, &pipeline, &mut lit).unwrap();
        let fns = lit.fns().len();
        let hash = lit.composer().slots_hash();
        let passes = pipeline.borrow().passes().len();
        let err = TransmissionFeature::install(&gpu, &pipeline, &mut lit)
            .err()
            .unwrap();
        assert_eq!(err.code(), "pipeline.duplicate_pass");
        assert!(!err.to_string().is_empty());
        assert_eq!(lit.fns().len(), fns);
        assert_eq!(lit.composer().slots_hash(), hash);
        assert_eq!(pipeline.borrow().passes().len(), passes);
    }

    #[test]
    fn install_while_pipeline_borrowed_refuses_busy() {
        let Some(gpu) = software_gpu() else {
            skip(
                "install_while_pipeline_borrowed_refuses_busy",
                "software_gpu returned None; see the software_gpu line above",
            );
            return;
        };
        let (pipeline, mut lit) = setup(&gpu);
        let fns = lit.fns().len();
        let guard = pipeline.borrow();
        let err = TransmissionFeature::install(&gpu, &pipeline, &mut lit)
            .err()
            .unwrap();
        drop(guard);
        assert_eq!(err.code(), "pipeline.busy");
        assert_eq!(lit.fns().len(), fns);
        assert!(pipeline.borrow().passes().iter().all(|i| i.name != NAME));
    }

    #[test]
    fn registered_name_refuses_fn_duplicate_and_installs_nothing() {
        let Some(gpu) = software_gpu() else {
            skip(
                "registered_name_refuses_fn_duplicate_and_installs_nothing",
                "software_gpu returned None; see the software_gpu line above",
            );
            return;
        };
        let (pipeline, mut lit) = setup(&gpu);
        let b = &TRANSMISSIVE_BUILTINS[0];
        lit.fns_mut().register_builtin(b.name, b.wgsl).unwrap();
        let fns = lit.fns().len();
        let hash = lit.composer().slots_hash();
        let err = TransmissionFeature::install(&gpu, &pipeline, &mut lit)
            .err()
            .unwrap();
        assert_eq!(err.code(), "material.fn_duplicate");
        assert_eq!(lit.fns().len(), fns);
        assert_eq!(lit.composer().slots_hash(), hash);
        assert!(pipeline.borrow().passes().iter().all(|i| i.name != NAME));
    }
}
