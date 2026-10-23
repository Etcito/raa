//! m10 materials API: material functions (`addMaterialFunction`, `materialFunctions`) and
//! physically based bloom (`setBloom`), surface materials (`addSurfaceMaterial`,
//! `setSurfaceMaterial`).
//!
//! `install` registers the raa-matfn built-in functions, the layered BRDF slot providers
//! (raa-brdf), the fur shell contributor (raa-fur) and the bloom pass (raa-bloom, whose
//! handle is kept in the feature map); `frame` is empty. Enabling and disabling stays
//! `setPassEnabled('bloom' | 'fur.shells', on)`. Refusals (`material.fn_invalid` with the
//! naga message, `material.fn_duplicate`, `bloom.invalid`) are thrown with their own code;
//! malformed JS arguments throw `api.invalid_argument`. Surface material descs map through
//! `raa_matfn::to_surface_material` and its codes are thrown unchanged.

use js_sys::{Array, Object, Reflect};
use raa_bloom::{BloomFeature, BloomHandle};
use raa_core::{GpuContext, RaaError};
use raa_fur::FurContributor;
use raa_matfn::{BUILTINS, Blend, ParamKind, ParamSpec, StandardOverrides, SurfaceDesc};
use raa_shader::MaterialFnId;
use raa_web::FrameInput;
use wasm_bindgen::prelude::*;

use crate::convert::{INVALID_ARGUMENT, api_error};
use crate::engine::{Engine, InstallCtx};
use crate::state::EngineState;

pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
    raa_matfn::register_builtins(&mut ctx.lit.fns_mut())
        .map_err(|e| api_error(e.code, e.message))?;
    raa_brdf::install(ctx.lit)?;
    FurContributor::install(ctx.gpu, ctx.lit)?;
    let bloom = BloomFeature::install(ctx.gpu, ctx.pipeline)?;
    ctx.features.insert(bloom);
    Ok(())
}

pub(crate) fn frame(
    _st: &mut EngineState,
    _gpu: &GpuContext,
    _input: &FrameInput,
) -> Result<(), RaaError> {
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

/// Sets `key` on a fresh plain object; this cannot throw, so the result is ignored.
fn set(obj: &Object, key: &str, value: &JsValue) {
    let _ = Reflect::set(obj, &JsValue::from_str(key), value);
}

/// JS kind of a parameter. The built-in tables only use Float and Color; Vec2 and Vec4
/// keep their own names rather than pretending to be `vec3`.
fn kind_name(kind: ParamKind) -> &'static str {
    match kind {
        ParamKind::Float => "scalar",
        ParamKind::Vec2 => "vec2",
        ParamKind::Vec3 => "vec3",
        ParamKind::Vec4 => "vec4",
        ParamKind::Color => "color",
    }
}

/// `{name, kind, default}` with `default` holding the parameter's components only.
fn param_to_js(p: &ParamSpec) -> JsValue {
    let o = Object::new();
    let n = p.kind.components().min(p.default.len());
    let default: Array = p.default[..n]
        .iter()
        .map(|&x| JsValue::from_f64(f64::from(x)))
        .collect();
    set(&o, "name", &JsValue::from_str(p.name));
    set(&o, "kind", &JsValue::from_str(kind_name(p.kind)));
    set(&o, "default", &default);
    o.into()
}

/// The boolean at `key`, or `default` when absent.
fn optional_bool(obj: &JsValue, key: &str, default: bool) -> Result<bool, RaaError> {
    match get(obj, key)? {
        None => Ok(default),
        Some(v) => v
            .as_bool()
            .ok_or_else(|| invalid(format!("'{key}' must be a boolean"))),
    }
}

/// The string at `key`, if present.
fn optional_string(obj: &JsValue, key: &str) -> Result<Option<String>, RaaError> {
    match get(obj, key)? {
        None => Ok(None),
        Some(v) => v
            .as_string()
            .map(Some)
            .ok_or_else(|| invalid(format!("'{key}' must be a string"))),
    }
}

/// A number or an array of numbers (`what` names the value in the refusal).
fn numbers(v: &JsValue, what: &str) -> Result<Vec<f32>, RaaError> {
    if let Some(x) = v.as_f64() {
        return Ok(vec![x as f32]);
    }
    if !Array::is_array(v) {
        return Err(invalid(format!(
            "{what} must be a number or an array of numbers"
        )));
    }
    Array::from(v)
        .iter()
        .map(|x| {
            x.as_f64()
                .map(|x| x as f32)
                .ok_or_else(|| invalid(format!("{what} must contain only numbers")))
        })
        .collect()
}

fn optional_f32(obj: &JsValue, key: &str) -> Result<Option<f32>, RaaError> {
    Ok(optional_f64(obj, key)?.map(|x| x as f32))
}

/// The 3-number array at `key`, if present.
fn optional_vec3(obj: &JsValue, key: &str) -> Result<Option<[f32; 3]>, RaaError> {
    let Some(v) = get(obj, key)? else {
        return Ok(None);
    };
    let bad = || invalid(format!("'{key}' must be an array of 3 numbers"));
    if !Array::is_array(&v) {
        return Err(bad());
    }
    let a = Array::from(&v);
    if a.length() != 3 {
        return Err(bad());
    }
    let mut out = [0.0; 3];
    for (i, dst) in out.iter_mut().enumerate() {
        *dst = a.get(i as u32).as_f64().ok_or_else(bad)? as f32;
    }
    Ok(Some(out))
}

/// `params`: a plain object whose values are a number or an array of numbers.
fn surface_params(desc: &JsValue) -> Result<Vec<(String, Vec<f32>)>, RaaError> {
    let Some(p) = get(desc, "params")? else {
        return Ok(Vec::new());
    };
    if !p.is_object() || Array::is_array(&p) {
        return Err(invalid("'params' must be a plain object".to_string()));
    }
    let mut out = Vec::new();
    for key in Object::keys(&Object::from(p.clone())).iter() {
        let Some(name) = key.as_string() else {
            continue;
        };
        let v =
            Reflect::get(&p, &key).map_err(|_| invalid(format!("cannot read 'params.{name}'")))?;
        let values = numbers(&v, &format!("'params.{name}'"))?;
        out.push((name, values));
    }
    Ok(out)
}

/// Parses a JS `SurfaceMaterialDesc` (`api.invalid_argument` naming the malformed key).
fn surface_desc(desc: &JsValue) -> Result<SurfaceDesc, RaaError> {
    require_object(desc, "SurfaceMaterialDesc")?;
    let d = SurfaceDesc::default();
    let blend = match optional_string(desc, "blend")?.as_deref() {
        None | Some("opaque") => Blend::Opaque,
        Some("transparent") => Blend::Transparent,
        Some("transmissive") => Blend::Transmissive,
        Some(other) => {
            return Err(invalid(format!(
                "'blend' must be 'opaque', 'transparent' or 'transmissive' (got '{other}')"
            )));
        }
    };
    let overrides = StandardOverrides {
        color: optional_vec3(desc, "color")?,
        roughness: optional_f32(desc, "roughness")?,
        metallic: optional_f32(desc, "metallic")?,
        emissive: optional_vec3(desc, "emissive")?,
        opacity: optional_f32(desc, "opacity")?,
        clearcoat: optional_f32(desc, "clearcoat")?,
        clearcoat_roughness: optional_f32(desc, "clearcoatRoughness")?,
        sheen_color: optional_vec3(desc, "sheenColor")?,
        sheen_roughness: optional_f32(desc, "sheenRoughness")?,
        transmission: optional_f32(desc, "transmission")?,
        ior: optional_f32(desc, "ior")?,
        thickness: optional_f32(desc, "thickness")?,
        attenuation_color: optional_vec3(desc, "attenuationColor")?,
        attenuation_distance: optional_f32(desc, "attenuationDistance")?,
        fur_length: optional_f32(desc, "furLength")?,
        fur_density: optional_f32(desc, "furDensity")?,
        fur_droop: optional_f32(desc, "furDroop")?,
        fur_shells: optional_f32(desc, "furShells")?,
    };
    Ok(SurfaceDesc {
        function: optional_string(desc, "function")?.unwrap_or(d.function),
        params: surface_params(desc)?,
        blend,
        double_sided: optional_bool(desc, "doubleSided", d.double_sided)?,
        cast_shadow: optional_bool(desc, "castShadow", d.cast_shadow)?,
        overrides,
    })
}

#[wasm_bindgen(js_class = Engine)]
impl Engine {
    /// Adds a surface material from a `SurfaceMaterialDesc` and returns its material handle
    /// (usable with addInstance, addMeshInstance and sub-mesh materials). raa-matfn codes
    /// (`material.unknown_fn`, `material.unknown_param`, `material.param_type`,
    /// `material.invalid`, `material.blend_unavailable`) are thrown unchanged.
    #[wasm_bindgen(js_name = addSurfaceMaterial)]
    pub fn add_surface_material(&mut self, desc: JsValue) -> Result<u32, JsValue> {
        self.with_state(|_, st| {
            let d = surface_desc(&desc)?;
            let m = raa_matfn::to_surface_material(&d, &st.lit)?;
            let len = st.lit.materials().len();
            if len >= st.max_materials as usize {
                return Err(api_error(
                    "render.too_many_materials",
                    format!(
                        "maxMaterials is {}, {len} are in use and 1 more was requested",
                        st.max_materials
                    ),
                ));
            }
            let id = st.lit.materials_mut().add(m)?;
            Ok(st.materials.insert(id))
        })
    }

    /// Replaces the material behind `id` with a `SurfaceMaterialDesc` (`api.unknown_handle`
    /// for an unknown handle); nothing changes on a refusal.
    #[wasm_bindgen(js_name = setSurfaceMaterial)]
    pub fn set_surface_material(&mut self, id: u32, desc: JsValue) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            let target = st.material(id)?;
            let d = surface_desc(&desc)?;
            let m = raa_matfn::to_surface_material(&d, &st.lit)?;
            st.lit.materials_mut().set(target, m)
        })
    }

    /// Sets the bloom: `{strength?, radius?}` merged over the current settings. Out-of-range
    /// values throw `bloom.invalid` naming the field; nothing changes on a refusal.
    #[wasm_bindgen(js_name = setBloom)]
    pub fn set_bloom(&mut self, settings: JsValue) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            require_object(&settings, "setBloom")?;
            let bloom = st.features.require_mut::<BloomHandle>("bloom")?;
            let mut d = bloom.get();
            read_f32(&settings, "strength", &mut d.strength)?;
            read_f32(&settings, "radius", &mut d.radius)?;
            bloom.set(d)
        })
    }

    /// Registers user WGSL as the material function `name` (S11 rules): `material.fn_invalid`
    /// with the reason (naga message with line:col), `material.fn_duplicate` for a taken name.
    /// Its parameters are positional: `in.record.p[0..8]`.
    #[wasm_bindgen(js_name = addMaterialFunction)]
    pub fn add_material_function(&mut self, name: &str, wgsl: &str) -> Result<(), JsValue> {
        self.with_state(|_, st| st.lit.register_user_fn(name, wgsl).map(|_| ()))
    }

    /// Every registered material function in id order: `{name, builtin, params}`. Built-ins
    /// of the raa-matfn table list `{name, kind: 'scalar' | 'vec3' | 'color', default}`;
    /// `raa.standard` and user functions list no params.
    #[wasm_bindgen(js_name = materialFunctions)]
    pub fn material_functions(&self) -> Result<Array, JsValue> {
        self.with_state(|_, st| {
            let fns = st.lit.fns();
            let out = Array::new();
            for i in 0..fns.len() {
                let Some(e) = u32::try_from(i).ok().and_then(|i| fns.get(MaterialFnId(i))) else {
                    continue;
                };
                let params: Array = BUILTINS
                    .iter()
                    .find(|b| e.builtin && b.name == e.name)
                    .map(|b| b.params.iter().map(param_to_js).collect())
                    .unwrap_or_default();
                let o = Object::new();
                set(&o, "name", &JsValue::from_str(&e.name));
                set(&o, "builtin", &JsValue::from_bool(e.builtin));
                set(&o, "params", &params);
                out.push(&o);
            }
            Ok(out)
        })
    }
}
