//! m25 material data words: `addMaterialData(words)` stores a block of u32 words in the
//! raa-lit material data buffer (group 0 binding 6) and returns its base word offset;
//! a surface material carries it as `SurfaceMaterialDesc.data` and its material function
//! reads the words with `raa_material_word(raa_material_data_base(in.record) + i)` or
//! `raa_material_f32(...)` (raa-shader prelude). One function (one pipeline) can then shade
//! many materials whose differences are data, not WGSL text. `setMaterialData(base, words)`
//! overwrites a live block of the same length; `removeMaterialData(base)` frees it (its
//! words read zero; an unknown base does nothing).
//!
//! `words` is a Uint32Array (stored as is), a Float32Array (each f32 stored as its bits,
//! read back by `raa_material_f32`) or an Int32Array (two's complement bits).
//! Refusals: `material.data_invalid` (no word; setMaterialData of an unknown base or another
//! length), `material.data_full` (the buffer would pass the device's storage binding size);
//! a malformed argument throws `api.invalid_argument`. `install` and `frame` are empty.
//! The pure part below compiles on the host (tests/material_data.rs includes this file).

use raa_core::RaaError;

fn invalid(message: String) -> RaaError {
    RaaError::FrameGraph {
        code: "api.invalid_argument",
        message,
    }
}

/// The base word offset argument `v` of setMaterialData / removeMaterialData: a whole
/// number in 0..=u32::MAX.
pub(crate) fn data_base(v: Option<f64>) -> Result<u32, RaaError> {
    let v = v.ok_or_else(|| invalid("'base' must be a number".to_string()))?;
    if v.fract() != 0.0 || !(0.0..=f64::from(u32::MAX)).contains(&v) {
        return Err(invalid(format!(
            "'base' must be a whole number >= 0, got {v}"
        )));
    }
    Ok(v as u32)
}

/// f32 values as the u32 words that hold their bits.
pub(crate) fn float_words(f: &[f32]) -> Vec<u32> {
    f.iter().map(|x| x.to_bits()).collect()
}

#[cfg(target_arch = "wasm32")]
pub(crate) use web::{frame, install};

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use crate::engine::{Engine, InstallCtx};
    use crate::state::EngineState;
    use js_sys::{Float32Array, Int32Array, Uint32Array};
    use raa_core::GpuContext;
    use raa_web::FrameInput;
    use wasm_bindgen::JsCast;
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

    fn words(v: &JsValue) -> Result<Vec<u32>, RaaError> {
        if let Some(w) = v.dyn_ref::<Uint32Array>() {
            return Ok(w.to_vec());
        }
        if let Some(f) = v.dyn_ref::<Float32Array>() {
            return Ok(float_words(&f.to_vec()));
        }
        if let Some(i) = v.dyn_ref::<Int32Array>() {
            return Ok(i.to_vec().into_iter().map(|x| x as u32).collect());
        }
        Err(invalid(
            "'words' must be a Uint32Array, Float32Array or Int32Array".into(),
        ))
    }

    #[wasm_bindgen(js_class = Engine)]
    impl Engine {
        #[wasm_bindgen(js_name = addMaterialData)]
        pub fn add_material_data(&mut self, words: JsValue) -> Result<u32, JsValue> {
            self.with_state(|gpu, st| {
                let w = super::web::words(&words)?;
                st.lit.add_material_data(gpu, &w)
            })
        }

        #[wasm_bindgen(js_name = setMaterialData)]
        pub fn set_material_data(&mut self, base: JsValue, words: JsValue) -> Result<(), JsValue> {
            self.with_state(|gpu, st| {
                let base = data_base(base.as_f64())?;
                let w = super::web::words(&words)?;
                st.lit.set_material_data(gpu, base, &w)
            })
        }

        #[wasm_bindgen(js_name = removeMaterialData)]
        pub fn remove_material_data(&mut self, base: JsValue) -> Result<(), JsValue> {
            self.with_state(|gpu, st| {
                let base = data_base(base.as_f64())?;
                st.lit.remove_material_data(gpu, base).map(|_| ())
            })
        }
    }
}
