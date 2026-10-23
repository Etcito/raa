//! m08 sky and image-based lighting API: the procedural sky (`setSky`, `sky`) and the
//! intensities of the sky-derived environment lighting (`setEnvironmentLighting`).
//!
//! `install` adds IBL first, then the sky (which filters its env cube through the IBL
//! handle); `frame` marks the sky dirty when the scene light changed, so the env cube and
//! its IBL follow the sun. Sky refusals (`sky.invalid`, naming the field) are thrown with
//! their own code; malformed JS arguments throw `api.invalid_argument`.

use js_sys::{Array, Float32Array, Object, Reflect};
use raa_core::{GpuContext, RaaError};
use raa_ibl::{IblFeature, IblHandle};
use raa_scene::DirectionalLight;
use raa_sky::{EnvUpdate, SkyDesc, SkyFeature, SkyHandle};
use raa_web::FrameInput;
use wasm_bindgen::{JsCast, prelude::*};

use crate::convert::{INVALID_ARGUMENT, api_error, js};
use crate::engine::{Engine, InstallCtx};
use crate::state::EngineState;

/// The IBL handle with the intensities last set from JS (`setEnvironmentLighting` keeps
/// the missing keys from here).
struct EnvLighting {
    ibl: IblHandle,
    diffuse: f32,
    specular: f32,
}

/// The scene light seen by the previous `frame` (None before the first frame).
struct LastLight(Option<DirectionalLight>);

pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
    let ibl = IblFeature::install(ctx.gpu, ctx.pipeline, ctx.lit)?;
    let sky = SkyFeature::install(ctx.gpu, ctx.pipeline, ctx.lit, ibl.clone())?;
    ctx.features.insert(EnvLighting {
        ibl,
        diffuse: 1.0,
        specular: 1.0,
    });
    ctx.features.insert(sky);
    ctx.features.insert(LastLight(None));
    Ok(())
}

/// Marks the sky dirty when the scene light changed (or on the first frame); no allocation.
pub(crate) fn frame(
    st: &mut EngineState,
    _gpu: &GpuContext,
    _input: &FrameInput,
) -> Result<(), RaaError> {
    let light = st.scene.light;
    let changed = match st.features.get_mut::<LastLight>() {
        Some(last) if last.0 == Some(light) => false,
        Some(last) => {
            last.0 = Some(light);
            true
        }
        None => false,
    };
    if changed && let Some(sky) = st.features.get::<SkyHandle>() {
        sky.mark_dirty();
    }
    Ok(())
}

fn invalid(message: String) -> RaaError {
    api_error(INVALID_ARGUMENT, message)
}

/// The value at `key`, or None when it is missing, undefined or null.
fn get(obj: &JsValue, key: &str) -> Result<Option<JsValue>, RaaError> {
    if !obj.is_object() {
        return Err(invalid(format!(
            "expected an object holding '{key}', got a non-object"
        )));
    }
    let v = Reflect::get(obj, &JsValue::from_str(key))
        .map_err(|_| invalid(format!("cannot read '{key}'")))?;
    Ok(if v.is_undefined() || v.is_null() {
        None
    } else {
        Some(v)
    })
}

/// Exactly `N` finite numbers from an Array or a Float32Array at `key`, if present.
fn optional_floats<const N: usize>(obj: &JsValue, key: &str) -> Result<Option<[f32; N]>, RaaError> {
    let Some(v) = get(obj, key)? else {
        return Ok(None);
    };
    let mut out = [0.0f32; N];
    let len = if let Some(fa) = v.dyn_ref::<Float32Array>() {
        if fa.length() as usize == N {
            fa.copy_to(&mut out);
        }
        fa.length() as usize
    } else if Array::is_array(&v) {
        let a: &Array = v.unchecked_ref();
        if a.length() as usize == N {
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = a
                    .get(i as u32)
                    .as_f64()
                    .ok_or_else(|| invalid(format!("'{key}[{i}]' must be a number")))?
                    as f32;
            }
        }
        a.length() as usize
    } else {
        return Err(invalid(format!(
            "'{key}' must be an Array or Float32Array of {N} numbers"
        )));
    };
    if len != N {
        return Err(invalid(format!(
            "'{key}' must have {N} components, got {len}"
        )));
    }
    if !out.iter().all(|x| x.is_finite()) {
        return Err(invalid(format!("'{key}' must hold finite numbers")));
    }
    Ok(Some(out))
}

fn optional_f32(obj: &JsValue, key: &str) -> Result<Option<f32>, RaaError> {
    Ok(js::optional_f64(obj, key)?.map(|x| x as f32))
}

/// Overwrites `*slot` with the number at `key` when present.
fn read_f32(obj: &JsValue, key: &str, slot: &mut f32) -> Result<(), RaaError> {
    if let Some(x) = optional_f32(obj, key)? {
        *slot = x;
    }
    Ok(())
}

/// Overwrites `*slot` with the `N` numbers at `key` when present.
fn read_floats<const N: usize>(
    obj: &JsValue,
    key: &str,
    slot: &mut [f32; N],
) -> Result<(), RaaError> {
    if let Some(v) = optional_floats::<N>(obj, key)? {
        *slot = v;
    }
    Ok(())
}

/// Applies the JS overrides of `desc` (camelCase names) over `d`.
fn apply_overrides(desc: &JsValue, d: &mut SkyDesc) -> Result<(), RaaError> {
    read_floats(desc, "zenith", &mut d.zenith)?;
    read_floats(desc, "horizon", &mut d.horizon)?;
    read_floats(desc, "ground", &mut d.ground)?;
    read_f32(desc, "intensity", &mut d.intensity)?;
    read_f32(desc, "horizonSoftness", &mut d.horizon_softness)?;
    read_f32(desc, "sunDiscSize", &mut d.sun_disc_angular_radius)?;
    read_f32(desc, "sunDiscIntensity", &mut d.sun_disc_intensity)?;
    if let Some(c) = get(desc, "clouds")? {
        if !c.is_object() {
            return Err(invalid("'clouds' must be an object".to_string()));
        }
        let cl = &mut d.clouds;
        read_f32(&c, "coverage", &mut cl.coverage)?;
        read_f32(&c, "density", &mut cl.density)?;
        read_f32(&c, "altitude", &mut cl.altitude_m)?;
        read_f32(&c, "scale", &mut cl.scale_m)?;
        read_floats(&c, "wind", &mut cl.wind_mps)?;
        read_floats(&c, "color", &mut cl.color)?;
        read_f32(&c, "silverLining", &mut cl.silver_lining)?;
    }
    if let Some(name) = js::optional_string(desc, "envUpdate")? {
        d.env_update = match name.as_str() {
            "onChange" => EnvUpdate::OnChange,
            "everyFrame" => EnvUpdate::EveryFrame,
            _ => {
                return Err(invalid(format!(
                    "'envUpdate' must be 'onChange' or 'everyFrame', got '{name}'"
                )));
            }
        };
    }
    Ok(())
}

/// Sets `key` on a fresh plain object; this cannot throw, so the result is ignored.
fn set(obj: &Object, key: &str, value: &JsValue) {
    let _ = Reflect::set(obj, &JsValue::from_str(key), value);
}

fn floats_to_js(v: &[f32]) -> JsValue {
    v.iter()
        .map(|x| JsValue::from(*x))
        .collect::<Array>()
        .into()
}

fn sky_to_js(d: &SkyDesc) -> JsValue {
    let c = &d.clouds;
    let clouds = Object::new();
    set(&clouds, "coverage", &JsValue::from(c.coverage));
    set(&clouds, "density", &JsValue::from(c.density));
    set(&clouds, "altitude", &JsValue::from(c.altitude_m));
    set(&clouds, "scale", &JsValue::from(c.scale_m));
    set(&clouds, "wind", &floats_to_js(&c.wind_mps));
    set(&clouds, "color", &floats_to_js(&c.color));
    set(&clouds, "silverLining", &JsValue::from(c.silver_lining));
    let o = Object::new();
    set(&o, "zenith", &floats_to_js(&d.zenith));
    set(&o, "horizon", &floats_to_js(&d.horizon));
    set(&o, "ground", &floats_to_js(&d.ground));
    set(&o, "intensity", &JsValue::from(d.intensity));
    set(&o, "horizonSoftness", &JsValue::from(d.horizon_softness));
    set(&o, "sunDiscSize", &JsValue::from(d.sun_disc_angular_radius));
    set(&o, "sunDiscIntensity", &JsValue::from(d.sun_disc_intensity));
    set(&o, "clouds", &clouds.into());
    let env = match d.env_update {
        EnvUpdate::OnChange => "onChange",
        EnvUpdate::EveryFrame => "everyFrame",
    };
    set(&o, "envUpdate", &JsValue::from_str(env));
    o.into()
}

/// A finite intensity >= 0 at `key`, if present.
fn intensity(obj: &JsValue, key: &str) -> Result<Option<f32>, RaaError> {
    match optional_f32(obj, key)? {
        Some(x) if !(x.is_finite() && x >= 0.0) => {
            Err(invalid(format!("'{key}' must be finite and >= 0, got {x}")))
        }
        v => Ok(v),
    }
}

#[wasm_bindgen(js_class = Engine)]
impl Engine {
    /// Sets the procedural sky: `preset` ('day' | 'snow' | 'jungle' | 'night') first when
    /// given (else the current sky), then the overrides. Out-of-range values throw
    /// `sky.invalid` naming the field; nothing changes on a refusal.
    #[wasm_bindgen(js_name = setSky)]
    pub fn set_sky(&mut self, desc: JsValue) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            let sky = st.features.require_mut::<SkyHandle>("sky")?;
            let mut d = match js::optional_string(&desc, "preset")? {
                Some(name) => SkyDesc::preset(&name).ok_or_else(|| {
                    api_error(
                        "sky.invalid",
                        format!(
                            "'preset' must be 'day', 'snow', 'jungle' or 'night', got '{name}'"
                        ),
                    )
                })?,
                None => sky.get(),
            };
            apply_overrides(&desc, &mut d)?;
            sky.set(d)
        })
    }

    /// The resolved sky description (camelCase names, `clouds.wind` as [x, z]; no preset).
    pub fn sky(&self) -> Result<JsValue, JsValue> {
        self.with_state(|_, st| {
            let sky = st.features.require_mut::<SkyHandle>("sky")?;
            Ok(sky_to_js(&sky.get()))
        })
    }

    /// Sets the intensities of the image-based lighting: `{diffuse?, specular?}`, finite
    /// and >= 0 (`api.invalid_argument`); missing keys keep their current value.
    #[wasm_bindgen(js_name = setEnvironmentLighting)]
    pub fn set_environment_lighting(&mut self, settings: JsValue) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            let env = st.features.require_mut::<EnvLighting>("ibl")?;
            let diffuse = intensity(&settings, "diffuse")?.unwrap_or(env.diffuse);
            let specular = intensity(&settings, "specular")?.unwrap_or(env.specular);
            env.diffuse = diffuse;
            env.specular = specular;
            env.ibl.set_intensity(diffuse, specular);
            Ok(())
        })
    }
}
