//! m20 player tint (S23): per-material tint mode (`setMaterialTintMode`) and the reset of an
//! instance tint (`clearInstanceTint`).
//!
//! A material in tint mode `graph` reads `mat_instance_tint` / `mat_instance_tint_set` in its
//! material graph, so its passes compose without the post-material tint colour multiply
//! (`raa_shader::TintMode::Graph`, `Composer::compose_with`); `multiply` (the default) is
//! today's shader. The mode is stored per material by `LitHandle::set_material_tint_mode`:
//! the draws of a `graph` material carry `raa_shader::VARIANT_FLAG_TINT_GRAPH` in their
//! variant key. setInstanceTint (m15) sets `InstanceExtra::tint_set` (the shader's
//! `mat_instance_tint_set`); clearInstanceTint resets it. `install` and `frame` are empty.
//! Refusals: `tint.material_range` (unknown material handle), `tint.mode_unknown` (a mode
//! other than `multiply` / `graph`), and the setInstanceTint instance codes
//! (`api.unknown_handle`) for clearInstanceTint.

use raa_core::RaaError;
use raa_lit::InstanceExtra;
use raa_shader::TintMode;

/// Unknown material handle of setMaterialTintMode.
pub(crate) const MATERIAL_RANGE: &str = "tint.material_range";
/// A tint mode other than `multiply` or `graph`.
pub(crate) const MODE_UNKNOWN: &str = TintMode::MODE_UNKNOWN;

fn tint_error(code: &'static str, message: String) -> RaaError {
    RaaError::FrameGraph { code, message }
}

/// The tint mode named `name` (`tint.mode_unknown` otherwise).
pub(crate) fn tint_mode(name: &str) -> Result<TintMode, RaaError> {
    TintMode::from_name(name).map_err(|e| tint_error(MODE_UNKNOWN, e.message))
}

/// The refusal of an unknown material handle `material`.
pub(crate) fn material_range(material: u32) -> RaaError {
    tint_error(
        MATERIAL_RANGE,
        format!("material handle {material} is not a live surface material"),
    )
}

/// `extra` with the tint reset to (1, 1, 1, 1) and `tint_set` to 0 (the shader's
/// `mat_instance_tint_set`); squash, jiggle and parameters are kept.
pub(crate) fn cleared(extra: InstanceExtra) -> InstanceExtra {
    InstanceExtra {
        tint: [1.0; 4],
        tint_set: 0,
        ..extra
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) use web::{frame, install};

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use crate::engine::{Engine, InstallCtx};
    use crate::state::EngineState;
    use raa_core::GpuContext;
    use raa_web::FrameInput;
    use wasm_bindgen::prelude::*;

    pub(crate) fn install(_ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
        Ok(())
    }

    pub(crate) fn frame(
        _st: &mut EngineState,
        _gpu: &GpuContext,
        _input: &FrameInput,
    ) -> Result<(), RaaError> {
        Ok(())
    }

    #[wasm_bindgen(js_class = Engine)]
    impl Engine {
        #[wasm_bindgen(js_name = setMaterialTintMode)]
        pub fn set_material_tint_mode(&mut self, material: u32, mode: &str) -> Result<(), JsValue> {
            self.with_state(|_, st| {
                let id = st
                    .material(material)
                    .map_err(|_| material_range(material))?;
                let mode = tint_mode(mode)?;
                st.lit.handle().set_material_tint_mode(id, mode);
                Ok(())
            })
        }

        #[wasm_bindgen(js_name = clearInstanceTint)]
        pub fn clear_instance_tint(&mut self, instance: u32) -> Result<(), JsValue> {
            self.with_state(|_, st| {
                let tag = st.instance_group(instance)?.tag;
                let extra = cleared(st.lit.instance_extra(tag));
                st.lit.set_instance_extra(tag, extra);
                Ok(())
            })
        }
    }
}
