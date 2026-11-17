//! m22 material textures: `addTexture({ width, height, data, mips?, srgb? })` uploads an rgba
//! image into the raa-lit texture atlas and returns its id, which material functions sample
//! with `raa_texture(id, uv)` (level 0) or `raa_texture_grad(id, uv, duv_dx, duv_dy)` (the mip
//! chain; raa-shader prelude); `removeTexture(id)` frees it.
//!
//! `data` holds width * height * 4 rgba components: a Uint8Array or Uint8ClampedArray is
//! stored as is, a Float32Array is clamped to [0, 1] and quantized to 8 bits. Ids are reused
//! (the lowest free id, from 1); a removed or unknown id samples (0, 0, 0, 0), and
//! removeTexture of such an id does nothing. `install` and `frame` are empty.
//!
//! Options. `mips` (default true) uploads the mip chain, `min(5, 1 + floor(log2(min side)))`
//! levels (at most `raa_lit::ATLAS_MIP_LEVELS`), each image's rect aligned to 16 texels;
//! `mips: false` uploads level 0 only and the image is sampled at level 0. `srgb` (default
//! false) averages the chain in linear light and re-encodes it, for sRGB colour images;
//! `srgb: false` averages the stored bytes. Level 0 is the same either way.
//!
//! Refusals: `texture.invalid` (a data length other than width * height * 4, a side of 0 or
//! over `raa_lit::ATLAS_SIZE`), `texture.atlas_full` (no room or no free id left); a malformed
//! argument, or a `mips` or `srgb` that is not a boolean, throws `api.invalid_argument`.
//! A refused call changes nothing.
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

/// `texture_options(mips, srgb)` gives the texture options of `addTexture`: `None` takes the
/// default (`mips` true, `srgb` false) and `Some(b)` is `b`.
///
/// Handoff: `texture_options(mips, srgb)` is called today by `addTexture` in the wasm `web`
/// module, with the two booleans it read from JS (`None` for `undefined`); its result is the
/// `options` argument of `renderer.rs:735 LitRenderer::add_texture_with`. The downstream cards
/// "raa-sdfm: baked material samples raa_texture_grad; colour bake added with srgb" and "post
/// page: mips-ground scene and its test" call JS `engine.addTexture({ width, height, data,
/// mips?, srgb? })` and reach this function through it.
pub(crate) fn texture_options(mips: Option<bool>, srgb: Option<bool>) -> raa_lit::TextureOptions {
    let defaults = raa_lit::TextureOptions::default();
    raa_lit::TextureOptions {
        mips: mips.unwrap_or(defaults.mips),
        srgb: srgb.unwrap_or(defaults.srgb),
    }
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
            return Err(invalid(
                "addTexture expects { width, height, data, mips?, srgb? }".into(),
            ));
        }
        Reflect::get(desc, &JsValue::from_str(key))
            .map_err(|_| invalid(format!("cannot read '{key}'")))
    }

    /// An optional boolean option: `undefined` is None, a JS boolean is Some(b), any other
    /// value refuses naming the key.
    fn flag(desc: &JsValue, key: &str) -> Result<Option<bool>, RaaError> {
        let v = field(desc, key)?;
        if v.is_undefined() {
            return Ok(None);
        }
        v.as_bool()
            .map(Some)
            .ok_or_else(|| invalid(format!("'{key}' must be a boolean")))
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
                let mips = flag(&desc, "mips")?;
                let srgb = flag(&desc, "srgb")?;
                let options = texture_options(mips, srgb);
                st.lit
                    .add_texture_with(gpu, width, height, data.data(), options)
            })
        }

        #[wasm_bindgen(js_name = removeTexture)]
        pub fn remove_texture(&mut self, id: u32) -> Result<(), JsValue> {
            self.with_state(|gpu, st| st.lit.remove_texture(gpu, id).map(|_| ()))
        }
    }
}
