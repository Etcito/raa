//! m13 liquids and particles API: animated liquid surfaces and GPU particle emitters.
//!
//! `install` registers the liquid built-ins (`raa.liquid.water`, `raa.liquid.slime`,
//! `raa.liquid.mud`), pushes `LIQUID_BUILTINS` onto the `ExtraBuiltins` of the feature map
//! (so `addSurfaceMaterial` accepts them and `materialFunctions()` lists them) and installs
//! the particle feature, whose handle is kept in the feature map. `frame` is empty.
//!
//! `addEmitter`, `setEmitter`, `removeEmitter` and `burst` convert the JS `EmitterDesc`
//! starting from `EmitterDesc::for_kind`; malformed JS (wrong types, unknown kind, wrong
//! array lengths) throws `api.invalid_argument`, range rules are `EmitterDesc::validate`'s
//! (`particles.invalid`). `setLiquidImpacts` writes up to 4 ripple impacts
//! `(x, z, time, strength)` into the instance params 0..3 (x, z in the liquid mesh's local
//! space, time in seconds on the engine clock) and zeroes the rest.

use js_sys::{Array, Reflect};
use raa_core::{GpuContext, RaaError};
use raa_material::srgb_to_linear;
use raa_matfn::ExtraBuiltins;
use raa_particles::{EmitterDesc, ParticleFeature, ParticleHandle, ParticleKind};
use raa_web::FrameInput;
use wasm_bindgen::prelude::*;

use crate::convert::{INVALID_ARGUMENT, api_error};
use crate::engine::{Engine, InstallCtx};
use crate::state::EngineState;

/// Most impacts one liquid instance carries (instance params 0..3).
const MAX_IMPACTS: u32 = 4;

pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
    raa_liquid::install(ctx.lit)?;
    match ctx.features.get_mut::<ExtraBuiltins>() {
        Some(extras) => extras.push(raa_liquid::LIQUID_BUILTINS),
        None => {
            let mut extras = ExtraBuiltins::default();
            extras.push(raa_liquid::LIQUID_BUILTINS);
            ctx.features.insert(extras);
        }
    }
    let particles = ParticleFeature::install(ctx.gpu, ctx.pipeline, ctx.lit)?;
    ctx.features.insert(particles);
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

/// `v` as a number (`api.invalid_argument` otherwise). Finiteness and ranges are left to
/// `EmitterDesc::validate`, so they refuse with `particles.invalid`.
fn number(key: &str, v: &JsValue) -> Result<f32, RaaError> {
    v.as_f64()
        .map(|x| x as f32)
        .ok_or_else(|| invalid(format!("'{key}' must be a number")))
}

/// Overwrites `*slot` with the number at `key` when present.
fn read_f32(obj: &JsValue, key: &str, slot: &mut f32) -> Result<(), RaaError> {
    if let Some(v) = get(obj, key)? {
        *slot = number(key, &v)?;
    }
    Ok(())
}

/// Exactly `N` numbers from an Array or a typed array.
fn numbers<const N: usize>(key: &str, v: &JsValue) -> Result<[f32; N], RaaError> {
    if !Array::is_array(v) && !v.is_instance_of::<js_sys::Float32Array>() {
        return Err(invalid(format!("'{key}' must be an array of {N} numbers")));
    }
    let a = Array::from(v);
    if a.length() as usize != N {
        return Err(invalid(format!(
            "'{key}' must have {N} components, got {}",
            a.length()
        )));
    }
    let mut out = [0.0f32; N];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = number(&format!("{key}[{i}]"), &a.get(i as u32))?;
    }
    Ok(out)
}

/// Overwrites `*slot` with the `N` numbers at `key` when present.
fn read_array<const N: usize>(
    obj: &JsValue,
    key: &str,
    slot: &mut [f32; N],
) -> Result<(), RaaError> {
    if let Some(v) = get(obj, key)? {
        *slot = numbers::<N>(key, &v)?;
    }
    Ok(())
}

/// A `[min, max]` range given as a pair, or as one number meaning `[x, x]`.
fn read_range(obj: &JsValue, key: &str, slot: &mut [f32; 2]) -> Result<(), RaaError> {
    if let Some(v) = get(obj, key)? {
        *slot = if v.as_f64().is_some() {
            let x = number(key, &v)?;
            [x, x]
        } else {
            numbers::<2>(key, &v)?
        };
    }
    Ok(())
}

/// A non-negative integer at `key` for a u32 field. Values above `u32::MAX` saturate, so
/// `validate` refuses them with `particles.invalid` where a range applies.
fn read_u32(obj: &JsValue, key: &str, slot: &mut u32) -> Result<(), RaaError> {
    if let Some(v) = get(obj, key)? {
        let x = v
            .as_f64()
            .ok_or_else(|| invalid(format!("'{key}' must be a number")))?;
        if !(x >= 0.0 && x.fract() == 0.0) {
            return Err(invalid(format!(
                "'{key}' must be a non-negative integer, got {x}"
            )));
        }
        *slot = if x > f64::from(u32::MAX) {
            u32::MAX
        } else {
            x as u32
        };
    }
    Ok(())
}

fn kind(name: &str) -> Result<ParticleKind, RaaError> {
    match name {
        "splash" => Ok(ParticleKind::Splash),
        "snow" => Ok(ParticleKind::Snow),
        "confetti" => Ok(ParticleKind::Confetti),
        "sparkles" => Ok(ParticleKind::Sparkles),
        other => Err(invalid(format!(
            "unknown emitter kind '{other}' (expected splash, snow, confetti or sparkles)"
        ))),
    }
}

/// Converts the JS `EmitterDesc`: `kind` and `position` are required, every other field
/// overrides the kind's defaults. `colors` are sRGB, converted to linear.
fn emitter_desc(v: &JsValue) -> Result<EmitterDesc, RaaError> {
    let name = get(v, "kind")?
        .ok_or_else(|| invalid("'kind' is required".to_string()))?
        .as_string()
        .ok_or_else(|| invalid("'kind' must be a string".to_string()))?;
    let mut d = EmitterDesc::for_kind(kind(&name)?);
    let position =
        get(v, "position")?.ok_or_else(|| invalid("'position' is required".to_string()))?;
    d.position = numbers::<3>("position", &position)?;
    read_array(v, "direction", &mut d.direction)?;
    read_f32(v, "spread", &mut d.spread_rad)?;
    read_f32(v, "rate", &mut d.rate_per_s)?;
    read_array(v, "lifetime", &mut d.lifetime_s)?;
    read_range(v, "speed", &mut d.speed)?;
    read_range(v, "size", &mut d.size)?;
    read_f32(v, "gravity", &mut d.gravity)?;
    read_f32(v, "drag", &mut d.drag)?;
    if let Some(c) = get(v, "colors")? {
        if !Array::is_array(&c) {
            return Err(invalid("'colors' must be an array of Vec3".to_string()));
        }
        let a = Array::from(&c);
        let mut colors = Vec::with_capacity(a.length() as usize);
        for i in 0..a.length() {
            let rgb = numbers::<3>(&format!("colors[{i}]"), &a.get(i))?;
            colors.push(rgb.map(srgb_to_linear));
        }
        d.colors = colors;
    }
    read_f32(v, "emissive", &mut d.emissive)?;
    read_f32(v, "opacity", &mut d.opacity)?;
    read_array(v, "volume", &mut d.volume_half)?;
    if let Some(g) = get(v, "groundY")? {
        d.ground_y = Some(number("groundY", &g)?);
    }
    read_u32(v, "maxParticles", &mut d.max_particles)?;
    read_u32(v, "seed", &mut d.seed)?;
    Ok(d)
}

/// The impacts as instance params rows `(x, z, time, strength)`, unused rows zero.
fn impacts(v: &JsValue) -> Result<[[f32; 4]; 4], RaaError> {
    if !Array::is_array(v) {
        return Err(invalid("impacts must be an array".to_string()));
    }
    let a = Array::from(v);
    if a.length() > MAX_IMPACTS {
        return Err(invalid(format!(
            "at most {MAX_IMPACTS} impacts, got {}",
            a.length()
        )));
    }
    let mut rows = [[0.0f32; 4]; 4];
    for (i, row) in rows.iter_mut().enumerate().take(a.length() as usize) {
        let item = a.get(i as u32);
        for (slot, key) in row.iter_mut().zip(["x", "z", "time", "strength"]) {
            let x = get(&item, key)?
                .ok_or_else(|| invalid(format!("impacts[{i}].{key} is required")))?
                .as_f64()
                .ok_or_else(|| invalid(format!("impacts[{i}].{key} must be a number")))?;
            if !x.is_finite() {
                return Err(invalid(format!(
                    "impacts[{i}].{key} must be finite, got {x}"
                )));
            }
            *slot = x as f32;
        }
    }
    Ok(rows)
}

#[wasm_bindgen(js_class = Engine)]
impl Engine {
    /// Adds a particle emitter and returns its id (dense, lowest free reused).
    #[wasm_bindgen(js_name = addEmitter)]
    pub fn add_emitter(&mut self, desc: JsValue) -> Result<u32, JsValue> {
        self.with_state(|_, st| {
            let d = emitter_desc(&desc)?;
            st.features
                .require_mut::<ParticleHandle>("particles")?
                .add(d)
        })
    }

    /// Replaces the description of emitter `id`.
    #[wasm_bindgen(js_name = setEmitter)]
    pub fn set_emitter(&mut self, id: u32, desc: JsValue) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            let d = emitter_desc(&desc)?;
            st.features
                .require_mut::<ParticleHandle>("particles")?
                .set(id, d)
        })
    }

    /// Removes emitter `id` and its particles.
    #[wasm_bindgen(js_name = removeEmitter)]
    pub fn remove_emitter(&mut self, id: u32) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            st.features
                .require_mut::<ParticleHandle>("particles")?
                .remove(id)
        })
    }

    /// Spawns `count` extra particles from emitter `id` at the next simulation step.
    pub fn burst(&mut self, id: u32, count: u32) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            st.features
                .require_mut::<ParticleHandle>("particles")?
                .burst(id, count)
        })
    }

    /// Sets up to 4 ripple impacts `{x, z, time, strength}` of a liquid instance (x, z in
    /// the mesh's local space, time in seconds on the engine clock): instance params 0..3,
    /// the unused ones zeroed.
    #[wasm_bindgen(js_name = setLiquidImpacts)]
    pub fn set_liquid_impacts(
        &mut self,
        instance: u32,
        impacts_js: JsValue,
    ) -> Result<(), JsValue> {
        self.with_state(|_, st| {
            let rows = impacts(&impacts_js)?;
            let tag = st.instance_group(instance)?.tag;
            let mut x = st.lit.instance_extra(tag);
            x.params = rows;
            st.lit.set_instance_extra(tag, x);
            Ok(())
        })
    }
}
