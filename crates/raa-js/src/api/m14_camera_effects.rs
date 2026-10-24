//! m14 camera effects API: screen-space reflections (`setSsr`), depth of field (`setDof`),
//! height fog and aerial perspective (`setFog`), volumetric sun shafts (`setSunShafts`),
//! colour grading (`setGrading`) and named environments (`applyEnvironment`).
//!
//! `install` adds the SSR, fog, sun shafts, DOF and grading features in that order; `frame`
//! pushes the scene camera (and the sun, for the shafts) to their handles every frame (the
//! passes have no scene access). Enabling and disabling stays `setPassEnabled('ssr' |
//! 'dof' | 'sunshafts' | 'fog' | 'grading', on)`. Refusals (`post.invalid`, naming the
//! field) are thrown with their own code; an unknown fog, grading or post preset throws
//! `post.unknown_preset` naming it and the valid names; malformed JS arguments throw
//! `api.invalid_argument`.
//!
//! `applyEnvironment` applies `{sky?, sun?, fog?, post?}` in the contract order sky, sun,
//! fog, post: the sky and fog strings are presets of their own setters, `sun` maps
//! `dir` -> `direction` for `setLight`, and a `post` string is one of the private
//! [`POST_LOOKS`] (grading preset of the same name + bloom + tone mapping) applied through
//! `setGrading`, `setBloom` and `setToneMapping`.

use js_sys::{Array, Object, Reflect};
use raa_core::{GpuContext, RaaError};
use raa_dof::{DofFeature, DofHandle};
use raa_fog::{FogDesc, FogFeature, FogHandle};
use raa_grading::{GradingDesc, GradingFeature, GradingHandle};
use raa_shafts::{ShaftsFeature, ShaftsHandle};
use raa_ssr::{SsrFeature, SsrHandle};
use raa_web::{FrameInput, error_to_js};
use wasm_bindgen::prelude::*;

use crate::convert::{INVALID_ARGUMENT, api_error};
use crate::engine::{Engine, InstallCtx};
use crate::state::EngineState;

pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
    let ssr = SsrFeature::install(ctx.gpu, ctx.pipeline)?;
    let fog = FogFeature::install(ctx.gpu, ctx.pipeline, ctx.lit)?;
    let shafts = ShaftsFeature::install(ctx.gpu, ctx.pipeline)?;
    let dof = DofFeature::install(ctx.gpu, ctx.pipeline)?;
    let grading = GradingFeature::install(ctx.gpu, ctx.pipeline)?;
    ctx.features.insert(ssr);
    ctx.features.insert(fog);
    ctx.features.insert(shafts);
    ctx.features.insert(dof);
    ctx.features.insert(grading);
    Ok(())
}

/// Pushes the scene camera to the SSR, DOF and shafts handles and the sun to the shafts
/// (the handles keep their previous camera on a singular matrix). No allocation.
pub(crate) fn frame(
    st: &mut EngineState,
    _gpu: &GpuContext,
    _input: &FrameInput,
) -> Result<(), RaaError> {
    let cam = &st.scene.camera;
    let view = cam.view();
    let proj = cam.projection();
    if let Some(ssr) = st.features.get::<SsrHandle>() {
        ssr.set_camera(&view, &proj);
    }
    if let Some(dof) = st.features.get::<DofHandle>() {
        dof.set_camera(&view, &proj);
    }
    if let Some(shafts) = st.features.get::<ShaftsHandle>() {
        shafts.set_camera(&view, &proj);
        let light = &st.scene.light;
        let c = light.color;
        let i = light.intensity;
        shafts.set_sun(light.direction, [c[0] * i, c[1] * i, c[2] * i]);
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

/// The number at `key`, if present (`api.invalid_argument` for a non-number). Range and
/// finiteness are left to the feature's own validation, so it refuses with its code.
fn optional_f64(obj: &JsValue, key: &str) -> Result<Option<f64>, RaaError> {
    match get(obj, key)? {
        None => Ok(None),
        Some(v) => v
            .as_f64()
            .map(Some)
            .ok_or_else(|| invalid(format!("'{key}' must be a number"))),
    }
}

/// Overwrites `*slot` with the number at `key` when present.
fn read_f32(obj: &JsValue, key: &str, slot: &mut f32) -> Result<(), RaaError> {
    if let Some(x) = optional_f64(obj, key)? {
        *slot = x as f32;
    }
    Ok(())
}

/// Checks that `settings` is an object (`api.invalid_argument` otherwise).
fn require_object(settings: &JsValue, method: &str) -> Result<(), RaaError> {
    if settings.is_object() {
        Ok(())
    } else {
        Err(invalid(format!("{method} expects an object")))
    }
}

/// The array of `N` numbers at `key`, if present (`api.invalid_argument` otherwise).
fn optional_array<const N: usize>(obj: &JsValue, key: &str) -> Result<Option<[f32; N]>, RaaError> {
    let Some(v) = get(obj, key)? else {
        return Ok(None);
    };
    let bad = || invalid(format!("'{key}' must be an array of {N} numbers"));
    if !Array::is_array(&v) {
        return Err(bad());
    }
    let a = Array::from(&v);
    if a.length() != N as u32 {
        return Err(bad());
    }
    let mut out = [0.0; N];
    for (i, dst) in out.iter_mut().enumerate() {
        *dst = a.get(i as u32).as_f64().ok_or_else(bad)? as f32;
    }
    Ok(Some(out))
}

/// The array of 3 numbers at `key`, if present (used by the fog and grading setters).
fn optional_vec3(obj: &JsValue, key: &str) -> Result<Option<[f32; 3]>, RaaError> {
    optional_array::<3>(obj, key)
}

/// The string at `key`, if present (`api.invalid_argument` for a non-string).
fn optional_string(obj: &JsValue, key: &str) -> Result<Option<String>, RaaError> {
    match get(obj, key)? {
        None => Ok(None),
        Some(v) => v
            .as_string()
            .map(Some)
            .ok_or_else(|| invalid(format!("'{key}' must be a string"))),
    }
}

/// Writes `value` at `key` of a plain JS object (never fails for an `Object`).
fn set(obj: &Object, key: &str, value: &JsValue) {
    let _ = Reflect::set(obj, &JsValue::from_str(key), value);
}

/// The JS error of an `RaaError` (the codes travel unchanged).
fn js_error(e: RaaError) -> JsValue {
    error_to_js(&e)
}

/// A `post.unknown_preset` refusal naming `name` and the valid preset names.
fn unknown_preset(kind: &str, name: &str, valid: &[&str]) -> RaaError {
    let quoted: Vec<String> = valid.iter().map(|n| format!("'{n}'")).collect();
    let list = match quoted.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("{} or {last}", rest.join(", ")),
        _ => quoted.concat(),
    };
    api_error(
        "post.unknown_preset",
        format!("'{kind}' preset must be {list}, got '{name}'"),
    )
}

const FOG_PRESETS: [&str; 5] = ["clear", "haze", "snow", "jungle", "night"];
const GRADING_PRESETS: [&str; 5] = ["neutral", "candy", "ice", "jungle", "neon"];
const POST_PRESETS: [&str; 5] = ["neutral", "candy", "ice", "jungle", "neon"];

/// Writes `settings` (`FogSettings`) over the current fog: `preset` first, then the
/// camelCase overrides. `post.invalid` on an out-of-range field, nothing changes on it.
fn apply_fog(st: &mut EngineState, settings: &JsValue) -> Result<(), RaaError> {
    require_object(settings, "setFog")?;
    let fog = st.features.require_mut::<FogHandle>("fog")?;
    let mut d = match optional_string(settings, "preset")? {
        Some(name) => {
            FogDesc::preset(&name).ok_or_else(|| unknown_preset("fog", &name, &FOG_PRESETS))?
        }
        None => fog.get(),
    };
    read_f32(settings, "density", &mut d.density)?;
    read_f32(settings, "heightFalloff", &mut d.height_falloff)?;
    read_f32(settings, "baseHeight", &mut d.base_height_m)?;
    read_f32(settings, "start", &mut d.start_m)?;
    if let Some(color) = optional_vec3(settings, "color")? {
        d.color = color;
    }
    read_f32(settings, "inscatter", &mut d.inscatter)?;
    read_f32(settings, "aerialStrength", &mut d.aerial_strength)?;
    read_f32(settings, "aerialDistance", &mut d.aerial_distance_m)?;
    fog.set(d)
}

/// Writes `settings` (`GradingSettings`) over the current grade: `preset` first, then the
/// camelCase overrides. `post.invalid` on an out-of-range field, nothing changes on it.
fn apply_grading(st: &mut EngineState, settings: &JsValue) -> Result<(), RaaError> {
    require_object(settings, "setGrading")?;
    let grading = st.features.require_mut::<GradingHandle>("grading")?;
    let mut d = match optional_string(settings, "preset")? {
        Some(name) => GradingDesc::preset(&name)
            .ok_or_else(|| unknown_preset("grading", &name, &GRADING_PRESETS))?,
        None => grading.get(),
    };
    read_f32(settings, "temperature", &mut d.temperature_k)?;
    read_f32(settings, "tint", &mut d.tint)?;
    read_f32(settings, "exposure", &mut d.exposure_ev)?;
    read_f32(settings, "contrast", &mut d.contrast)?;
    read_f32(settings, "saturation", &mut d.saturation)?;
    read_f32(settings, "vibrance", &mut d.vibrance)?;
    if let Some(v) = optional_vec3(settings, "lift")? {
        d.lift = v;
    }
    if let Some(v) = optional_vec3(settings, "gamma")? {
        d.gamma = v;
    }
    if let Some(v) = optional_vec3(settings, "gain")? {
        d.gain = v;
    }
    if let Some(v) = optional_vec3(settings, "shadowsTint")? {
        d.shadows_tint = v;
    }
    if let Some(v) = optional_vec3(settings, "highlightsTint")? {
        d.highlights_tint = v;
    }
    read_f32(settings, "splitBalance", &mut d.split_balance)?;
    read_f32(settings, "vignette", &mut d.vignette)?;
    grading.set(d)
}

/// The private post table: `(name, grading preset, bloom strength, bloom radius, tone
/// mapper, exposure EV, look)`. The grading preset has the same name as the post preset.
const POST_LOOKS: [(&str, &str, f32, f32, &str, f32, &str); 5] = [
    ("neutral", "neutral", 0.04, 0.85, "agx", 0.0, "none"),
    ("candy", "candy", 0.05, 0.85, "agx", 0.0, "punchy"),
    ("ice", "ice", 0.04, 0.85, "agx", -0.2, "none"),
    ("jungle", "jungle", 0.03, 0.85, "agx", 0.0, "golden"),
    ("neon", "neon", 0.12, 0.9, "agx", 0.0, "punchy"),
];

/// Applies the named post look through `setGrading`, `setBloom` and `setToneMapping`
/// (`post.unknown_preset` when `name` is not in [`POST_LOOKS`]).
fn apply_post_preset(engine: &mut Engine, name: &str) -> Result<(), JsValue> {
    let Some(&(_, grading, strength, radius, mapper, exposure_ev, look)) =
        POST_LOOKS.iter().find(|p| p.0 == name)
    else {
        return Err(js_error(unknown_preset("post", name, &POST_PRESETS)));
    };
    let grading = {
        let o = Object::new();
        set(&o, "preset", &JsValue::from_str(grading));
        o
    };
    engine.set_grading(grading.into())?;
    let bloom = {
        let o = Object::new();
        set(&o, "strength", &JsValue::from_f64(f64::from(strength)));
        set(&o, "radius", &JsValue::from_f64(f64::from(radius)));
        o
    };
    engine.set_bloom(bloom.into())?;
    let tone = {
        let o = Object::new();
        set(&o, "mapper", &JsValue::from_str(mapper));
        set(&o, "exposureEv", &JsValue::from_f64(f64::from(exposure_ev)));
        set(&o, "look", &JsValue::from_str(look));
        o
    };
    engine.set_tone_mapping(tone.into())
}

#[wasm_bindgen(js_class = Engine)]
impl Engine {
    /// Sets the screen-space reflections: `{intensity?, maxRoughness?, thickness?}` merged
    /// over the current settings. Out-of-range values throw `post.invalid` naming the
    /// field; nothing changes on a refusal.
    #[wasm_bindgen(js_name = setSsr)]
    pub fn set_ssr(&mut self, settings: JsValue) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            require_object(&settings, "setSsr")?;
            let ssr = st.features.require_mut::<SsrHandle>("ssr")?;
            let mut d = ssr.get();
            read_f32(&settings, "intensity", &mut d.intensity)?;
            read_f32(&settings, "maxRoughness", &mut d.max_roughness)?;
            read_f32(&settings, "thickness", &mut d.thickness_m)?;
            ssr.set(d)
        })
    }

    /// Sets the depth of field: `{focusDistance?, fStop?, focalLength?, autofocus?: [u, v]
    /// | null, maxCoc?}` merged over the current settings (`autofocus: null` turns it off,
    /// absent keeps it). Out-of-range values throw `post.invalid` naming the field; nothing
    /// changes on a refusal.
    #[wasm_bindgen(js_name = setDof)]
    pub fn set_dof(&mut self, settings: JsValue) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            require_object(&settings, "setDof")?;
            let dof = st.features.require_mut::<DofHandle>("dof")?;
            let mut d = dof.get();
            read_f32(&settings, "focusDistance", &mut d.focus_distance_m)?;
            read_f32(&settings, "fStop", &mut d.f_stop)?;
            read_f32(&settings, "focalLength", &mut d.focal_length_mm)?;
            read_f32(&settings, "maxCoc", &mut d.max_coc_px)?;
            let af = Reflect::get(&settings, &JsValue::from_str("autofocus"))
                .map_err(|_| invalid("cannot read 'autofocus'".to_string()))?;
            if af.is_null() {
                d.autofocus = None;
            } else if !af.is_undefined() {
                d.autofocus = optional_array::<2>(&settings, "autofocus")?;
            }
            dof.set(d)
        })
    }

    /// Sets the sun shafts: `{intensity?, density?, maxDistance?}` merged over the current
    /// settings. Out-of-range values throw `post.invalid` naming the field; nothing changes
    /// on a refusal.
    #[wasm_bindgen(js_name = setSunShafts)]
    pub fn set_sun_shafts(&mut self, settings: JsValue) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            require_object(&settings, "setSunShafts")?;
            let shafts = st.features.require_mut::<ShaftsHandle>("sun shafts")?;
            let mut d = shafts.get();
            read_f32(&settings, "intensity", &mut d.intensity)?;
            read_f32(&settings, "density", &mut d.density)?;
            read_f32(&settings, "maxDistance", &mut d.max_distance_m)?;
            shafts.set(d)
        })
    }

    /// Sets the height fog and aerial perspective: `preset` ('clear' | 'haze' | 'snow' |
    /// 'jungle' | 'night') first, then `{density?, heightFalloff?, baseHeight?, start?,
    /// color?, inscatter?, aerialStrength?, aerialDistance?}` over it (over the current
    /// description without a preset). An unknown preset throws `post.unknown_preset`;
    /// out-of-range values throw `post.invalid` naming the field; nothing changes on a
    /// refusal.
    #[wasm_bindgen(js_name = setFog)]
    pub fn set_fog(&mut self, settings: JsValue) -> Result<(), JsValue> {
        self.with_state(|_, st| apply_fog(st, &settings))
    }

    /// Sets the colour grading: `preset` ('neutral' | 'candy' | 'ice' | 'jungle' | 'neon')
    /// first, then `{temperature?, tint?, exposure?, contrast?, saturation?, vibrance?,
    /// lift?, gamma?, gain?, shadowsTint?, highlightsTint?, splitBalance?, vignette?}` over
    /// it (over the current description without a preset). An unknown preset throws
    /// `post.unknown_preset`; out-of-range values throw `post.invalid` naming the field;
    /// nothing changes on a refusal.
    #[wasm_bindgen(js_name = setGrading)]
    pub fn set_grading(&mut self, settings: JsValue) -> Result<(), JsValue> {
        self.with_state(|_, st| apply_grading(st, &settings))
    }

    /// Applies a whole environment `{sky?, sun?, fog?, post?}` in the contract order sky,
    /// sun, fog, post. A `sky` string is a sky preset applied as `setSky({preset})`; a sky
    /// object is passed to `setSky` unchanged. `sun` `{dir, color, intensity}` is applied as
    /// `setLight({direction, color, intensity})`. A `fog` string or object goes through
    /// `setFog`. A `post` string is a post preset (`neutral`, `candy`, `ice`, `jungle`,
    /// `neon`): its grading preset, bloom and tone mapping are applied through `setGrading`,
    /// `setBloom` and `setToneMapping`; a post object applies its optional `grading`,
    /// `bloom`, `dof`, `ssr`, `sunShafts` and `toneMapping` keys through their setters.
    #[wasm_bindgen(js_name = applyEnvironment)]
    pub fn apply_environment(&mut self, env: JsValue) -> Result<(), JsValue> {
        require_object(&env, "applyEnvironment").map_err(js_error)?;
        if let Some(sky) = get(&env, "sky").map_err(js_error)? {
            if let Some(name) = sky.as_string() {
                let desc = Object::new();
                set(&desc, "preset", &JsValue::from_str(&name));
                self.set_sky(desc.into())?;
            } else {
                self.set_sky(sky)?;
            }
        }
        if let Some(sun) = get(&env, "sun").map_err(js_error)? {
            require_object(&sun, "sun").map_err(js_error)?;
            let desc = Object::new();
            for (from, to) in [
                ("dir", "direction"),
                ("color", "color"),
                ("intensity", "intensity"),
            ] {
                if let Some(v) = get(&sun, from).map_err(js_error)? {
                    set(&desc, to, &v);
                }
            }
            self.set_light(desc.into())?;
        }
        if let Some(fog) = get(&env, "fog").map_err(js_error)? {
            if let Some(name) = fog.as_string() {
                let desc = Object::new();
                set(&desc, "preset", &JsValue::from_str(&name));
                self.set_fog(desc.into())?;
            } else {
                self.set_fog(fog)?;
            }
        }
        if let Some(post) = get(&env, "post").map_err(js_error)? {
            if let Some(name) = post.as_string() {
                apply_post_preset(self, &name)?;
            } else {
                require_object(&post, "post").map_err(js_error)?;
                if let Some(v) = get(&post, "grading").map_err(js_error)? {
                    self.set_grading(v)?;
                }
                if let Some(v) = get(&post, "bloom").map_err(js_error)? {
                    self.set_bloom(v)?;
                }
                if let Some(v) = get(&post, "dof").map_err(js_error)? {
                    self.set_dof(v)?;
                }
                if let Some(v) = get(&post, "ssr").map_err(js_error)? {
                    self.set_ssr(v)?;
                }
                if let Some(v) = get(&post, "sunShafts").map_err(js_error)? {
                    self.set_sun_shafts(v)?;
                }
                if let Some(v) = get(&post, "toneMapping").map_err(js_error)? {
                    self.set_tone_mapping(v)?;
                }
            }
        }
        Ok(())
    }
}
