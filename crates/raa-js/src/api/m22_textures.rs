//! m22 material textures: `addTexture({ width, height, data })` uploads an rgba image into
//! the raa-lit texture atlas and returns its id, which material functions sample with
//! `raa_texture(id, uv)` (raa-shader prelude); `removeTexture(id)` frees it.
//!
//! `data` holds width * height * 4 rgba components: a Uint8Array or Uint8ClampedArray is
//! stored as is, a Float32Array is clamped to [0, 1] and quantized to 8 bits. Ids are reused
//! (the lowest free id, from 1); a removed or unknown id samples (0, 0, 0, 0), and
//! removeTexture of such an id does nothing. `install` and `frame` are empty.
//! Refusals: `texture.invalid` (a data length other than width * height * 4, a side of 0 or
//! over `raa_lit::ATLAS_SIZE`), `texture.atlas_full` (no room or no free id left);
//! a malformed argument throws `api.invalid_argument`.
//! The pure part below compiles on the host (tests/textures.rs includes this file).

use raa_core::RaaError;
use raa_lit::TexelData;

fn invalid(message: String) -> RaaError {
    RaaError::FrameGraph {
        code: "api.invalid_argument",
        message,
    }
}

/// The texture side `name` of value `v`: a whole number in 0..=u32::MAX (raa-lit refuses 0
/// and oversized sides as `texture.invalid`).
pub(crate) fn texture_side(name: &str, v: Option<f64>) -> Result<u32, RaaError> {
    let v = v.ok_or_else(|| invalid(format!("'{name}' must be a number")))?;
    if v.fract() != 0.0 || !(0.0..=f64::from(u32::MAX)).contains(&v) {
        return Err(invalid(format!(
            "'{name}' must be a whole number >= 0, got {v}"
        )));
    }
    Ok(v as u32)
}

/// The texels of addTexture, copied out of the JS typed array.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Texels {
    Rgba8(Vec<u8>),
    Float(Vec<f32>),
}

impl Texels {
    pub(crate) fn data(&self) -> TexelData<'_> {
        match self {
            Texels::Rgba8(b) => TexelData::Rgba8(b),
            Texels::Float(f) => TexelData::Float(f),
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) use web::{frame, install};

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use crate::engine::{Engine, InstallCtx};
    use crate::state::EngineState;
    use js_sys::{Float32Array, Reflect, Uint8Array, Uint8ClampedArray};
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

    fn field(desc: &JsValue, key: &str) -> Result<JsValue, RaaError> {
        if !desc.is_object() {
            return Err(invalid("addTexture expects { width, height, data }".into()));
        }
        Reflect::get(desc, &JsValue::from_str(key))
            .map_err(|_| invalid(format!("cannot read '{key}'")))
    }

    fn texels(v: &JsValue) -> Result<Texels, RaaError> {
        if let Some(b) = v.dyn_ref::<Uint8Array>() {
            return Ok(Texels::Rgba8(b.to_vec()));
        }
        if let Some(b) = v.dyn_ref::<Uint8ClampedArray>() {
            return Ok(Texels::Rgba8(b.to_vec()));
        }
        if let Some(f) = v.dyn_ref::<Float32Array>() {
            return Ok(Texels::Float(f.to_vec()));
        }
        Err(invalid(
            "'data' must be a Uint8Array, Uint8ClampedArray or Float32Array".into(),
        ))
    }

    #[wasm_bindgen(js_class = Engine)]
    impl Engine {
        #[wasm_bindgen(js_name = addTexture)]
        pub fn add_texture(&mut self, desc: JsValue) -> Result<u32, JsValue> {
            self.with_state(|gpu, st| {
                let width = texture_side("width", field(&desc, "width")?.as_f64())?;
                let height = texture_side("height", field(&desc, "height")?.as_f64())?;
                let data = texels(&field(&desc, "data")?)?;
                st.lit.add_texture(gpu, width, height, data.data())
            })
        }

        #[wasm_bindgen(js_name = removeTexture)]
        pub fn remove_texture(&mut self, id: u32) -> Result<(), JsValue> {
            self.with_state(|gpu, st| st.lit.remove_texture(gpu, id).map(|_| ()))
        }
    }
}
