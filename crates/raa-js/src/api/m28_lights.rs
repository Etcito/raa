//! m28 local lights: the order-35 `light.tiles` installation and the S17 descriptor parser.
//!
//! `install` registers the order-35 `light.tiles` pass through
//! `raa_lit::install_local_lights` against the renderer-owned `LocalLights` registry
//! (`M2-TILES-INTEGRATION`), and `frame` is the shared api registry's no-op hook: the pass
//! prepares itself from the `M2-FRAME-LIFECYCLE` invocation/view state. The sun's
//! `Engine.setLight(LightDesc)` is untouched; local lights get their own names.
//!
//! The parser below turns a JS `LocalLightDesc`/`LocalLightChanges` object into a
//! [`LocalLightPatch`] and merges it onto a base descriptor. It is pure Rust, so
//! `tests/local_lights.rs` exercises the merged state natively. The wasm32 [`web`] module
//! implements [`Properties`] over `js_sys::Reflect` and distinguishes an absent key from
//! one whose value is `undefined` with `Reflect::has`, because `Reflect::get` returns
//! `undefined` for both; `tests/api_registry.rs` pins that distinction in the source.
//!
//! Handoff: the S17 mutators (`addLight`, `updateLight`, `removeLight`, `attachLight`,
//! `detachLight`, `lightStats`) land in the later api-m28 cards of this file. They read a
//! JS object with `web::read_patch`, merge it onto the current `LocalLight` with [`merge`]
//! (`add` starts from `LocalLight::default()`, `update` from
//! `LitRenderer::local_lights().get(handle)`), and hand the result to
//! `LocalLights::add`/`replace`, which validates with `LocalLight::validate` before it
//! mutates. Handles pass through [`handle_arg`] before any narrowing to `LightHandle`.

use raa_core::RaaError;
use raa_lit::{LightHandle, LocalLight, LocalLightKind, MAX_LIGHT_HANDLE};

/// The refusal code of every malformed local-light description (S17).
pub(crate) const INVALID_DESC: &str = "light.invalid_desc";

/// The refusal code of a malformed non-descriptor argument (a bad handle number).
pub(crate) const INVALID_ARGUMENT: &str = "api.invalid_argument";

fn invalid(message: impl Into<String>) -> RaaError {
    RaaError::FrameGraph {
        code: INVALID_DESC,
        message: message.into(),
    }
}

/// Exact piecewise sRGB EOTF, the S17 boundary conversion. It is applied once, by
/// [`merge`], and only to a colour that was actually set.
#[allow(dead_code)] // consumed by the S17 mutators of the later api-m28 cards
pub(crate) fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// A raw property value as read from a description object, before any narrowing.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Raw {
    /// The key is not a property: `Reflect::has` is false.
    Absent,
    /// The key is present with the value `undefined` (or `null`).
    Undefined,
    Bool(bool),
    Number(f64),
    String(String),
    Numbers(Vec<f64>),
    /// Present with a value of no accepted type.
    Other,
}

/// The property source of a description object. `read` reports absence separately from an
/// `undefined` value, which a plain `get` cannot. `tests/local_lights.rs` supplies a
/// map-backed source; wasm32 uses `Reflect`.
pub(crate) trait Properties {
    fn read(&self, key: &str) -> Result<Raw, RaaError>;
}

/// A tri-state field of a description: an absent key keeps the base, an explicit
/// `undefined` restores the S17 default, and a value replaces the field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Patch<T> {
    Keep,
    Reset,
    Set(T),
}

impl<T: Copy> Patch<T> {
    #[allow(dead_code)] // consumed by `merge`, which the later api-m28 mutators call
    fn resolve(self, base: T, default: T) -> T {
        match self {
            Patch::Keep => base,
            Patch::Reset => default,
            Patch::Set(value) => value,
        }
    }
}

/// One parsed description: every field is a [`Patch`], so an omitted key and an explicit
/// `undefined` stay distinct through the merge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LocalLightPatch {
    pub kind: Patch<LocalLightKind>,
    pub position: Patch<[f32; 3]>,
    pub direction: Patch<[f32; 3]>,
    /// sRGB at the JS boundary; [`merge`] converts it to linear exactly once.
    pub color_srgb: Patch<[f32; 3]>,
    pub intensity: Patch<f32>,
    pub range: Patch<f32>,
    pub angle_deg: Patch<f32>,
    /// `Reset` clears an authored inner angle back to the 0.8 x `angle_deg` default.
    pub inner_angle_deg: Patch<Option<f32>>,
    pub size: Patch<[f32; 2]>,
    pub radius: Patch<f32>,
    pub shadows: Patch<bool>,
}

impl LocalLightPatch {
    /// Every field `Keep`: [`merge`] with this patch returns the base unchanged.
    #[allow(dead_code)] // consumed by the S17 mutators of the later api-m28 cards
    pub(crate) fn empty() -> Self {
        Self {
            kind: Patch::Keep,
            position: Patch::Keep,
            direction: Patch::Keep,
            color_srgb: Patch::Keep,
            intensity: Patch::Keep,
            range: Patch::Keep,
            angle_deg: Patch::Keep,
            inner_angle_deg: Patch::Keep,
            size: Patch::Keep,
            radius: Patch::Keep,
            shadows: Patch::Keep,
        }
    }
}

/// Maps a raw value's presence to a patch: absent keeps, undefined resets, else `parse`.
fn present<T>(
    raw: Raw,
    parse: impl FnOnce(Raw) -> Result<T, RaaError>,
) -> Result<Patch<T>, RaaError> {
    match raw {
        Raw::Absent => Ok(Patch::Keep),
        Raw::Undefined => Ok(Patch::Reset),
        value => parse(value).map(Patch::Set),
    }
}

fn scalar(raw: Raw, key: &str) -> Result<Patch<f32>, RaaError> {
    present(raw, |raw| match raw {
        Raw::Number(value) => Ok(value as f32),
        _ => Err(invalid(format!("'{key}' must be a number"))),
    })
}

fn vector<const N: usize>(raw: Raw, key: &str) -> Result<Patch<[f32; N]>, RaaError> {
    present(raw, |raw| match raw {
        Raw::Numbers(values) if values.len() == N => {
            let mut out = [0.0f32; N];
            for (slot, value) in out.iter_mut().zip(values) {
                *slot = value as f32;
            }
            Ok(out)
        }
        Raw::Numbers(values) => Err(invalid(format!(
            "'{key}' must have {N} components, got {}",
            values.len()
        ))),
        _ => Err(invalid(format!("'{key}' must be an array of {N} numbers"))),
    })
}

fn kind_field(raw: Raw) -> Result<Patch<LocalLightKind>, RaaError> {
    present(raw, |raw| match raw {
        Raw::String(name) => kind_name(&name),
        _ => Err(invalid("'kind' must be a string")),
    })
}

fn kind_name(name: &str) -> Result<LocalLightKind, RaaError> {
    match name {
        "point" => Ok(LocalLightKind::Point),
        "spot" => Ok(LocalLightKind::Spot),
        "directional" => Ok(LocalLightKind::Directional),
        "area" => Ok(LocalLightKind::Area),
        other => Err(invalid(format!(
            "'kind' must be point, spot, directional or area, got '{other}'"
        ))),
    }
}

fn inner_angle(raw: Raw) -> Result<Patch<Option<f32>>, RaaError> {
    match raw {
        Raw::Absent => Ok(Patch::Keep),
        Raw::Undefined => Ok(Patch::Reset),
        Raw::Number(value) => Ok(Patch::Set(Some(value as f32))),
        _ => Err(invalid("'innerAngleDeg' must be a number")),
    }
}

fn bool_field(raw: Raw, key: &str) -> Result<Patch<bool>, RaaError> {
    present(raw, |raw| match raw {
        Raw::Bool(value) => Ok(value),
        _ => Err(invalid(format!("'{key}' must be a boolean"))),
    })
}

/// Parses one description object into a patch. Absent keys stay `Keep` and an explicit
/// `undefined` becomes `Reset`, so the caller can merge it onto the current descriptor.
pub(crate) fn parse(source: &impl Properties) -> Result<LocalLightPatch, RaaError> {
    Ok(LocalLightPatch {
        kind: kind_field(source.read("kind")?)?,
        position: vector::<3>(source.read("position")?, "position")?,
        direction: vector::<3>(source.read("direction")?, "direction")?,
        color_srgb: vector::<3>(source.read("color")?, "color")?,
        intensity: scalar(source.read("intensity")?, "intensity")?,
        range: scalar(source.read("range")?, "range")?,
        angle_deg: scalar(source.read("angleDeg")?, "angleDeg")?,
        inner_angle_deg: inner_angle(source.read("innerAngleDeg")?)?,
        size: vector::<2>(source.read("size")?, "size")?,
        radius: scalar(source.read("radius")?, "radius")?,
        shadows: bool_field(source.read("shadows")?, "shadows")?,
    })
}

/// Merges a parsed patch onto `base`: `Keep` retains the base field, `Reset` restores the
/// S17 default and `Set` replaces it. The colour is converted from sRGB to linear exactly
/// once, and only when it was set, so a retained or default colour is never reconverted.
#[allow(dead_code)] // consumed by the S17 mutators of the later api-m28 cards
pub(crate) fn merge(base: LocalLight, patch: &LocalLightPatch) -> LocalLight {
    let default = LocalLight::default();
    LocalLight {
        kind: patch.kind.resolve(base.kind, default.kind),
        position: patch.position.resolve(base.position, default.position),
        direction: patch.direction.resolve(base.direction, default.direction),
        color: match patch.color_srgb {
            Patch::Keep => base.color,
            Patch::Reset => default.color,
            Patch::Set(srgb) => srgb.map(srgb_to_linear),
        },
        intensity: patch.intensity.resolve(base.intensity, default.intensity),
        range: patch.range.resolve(base.range, default.range),
        angle_deg: patch.angle_deg.resolve(base.angle_deg, default.angle_deg),
        inner_angle_deg: patch
            .inner_angle_deg
            .resolve(base.inner_angle_deg, default.inner_angle_deg),
        size: patch.size.resolve(base.size, default.size),
        radius: patch.radius.resolve(base.radius, default.radius),
        shadows: patch.shadows.resolve(base.shadows, default.shadows),
    }
}

/// A `LightHandle` argument: a positive safe JS integer, checked before narrowing. A
/// non-integer, non-finite, non-positive or out-of-range number is refused with
/// `api.invalid_argument`; a well-formed handle that names no light is left to the
/// registry's `api.unknown_handle`.
#[allow(dead_code)] // consumed by the S17 mutators of the later api-m28 cards
pub(crate) fn handle_arg(name: &str, value: f64) -> Result<LightHandle, RaaError> {
    if value.is_finite() && value.fract() == 0.0 && value >= 1.0 && value <= MAX_LIGHT_HANDLE as f64
    {
        Ok(value as LightHandle)
    } else {
        Err(RaaError::FrameGraph {
            code: INVALID_ARGUMENT,
            message: format!("'{name}' must be a positive safe light handle, got {value}"),
        })
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) use web::{frame, install};

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use crate::engine::InstallCtx;
    use crate::state::EngineState;
    use js_sys::{Array, Float32Array, Reflect};
    use raa_core::GpuContext;
    use raa_web::FrameInput;
    use wasm_bindgen::{JsCast, JsValue};

    /// Registers the order-35 `light.tiles` pass against the renderer-owned `LocalLights`.
    pub(crate) fn install(ctx: &mut InstallCtx<'_>) -> Result<(), RaaError> {
        raa_lit::install_local_lights(ctx.gpu, ctx.pipeline, ctx.lit)
    }

    /// The shared frame hook: tiling prepares itself from `M2-FRAME-LIFECYCLE` state.
    pub(crate) fn frame(
        _st: &mut EngineState,
        _gpu: &GpuContext,
        _input: &FrameInput,
    ) -> Result<(), RaaError> {
        Ok(())
    }

    /// A [`Properties`] view of a JS object: `Reflect::has` separates an absent key from
    /// one whose value is `undefined`, which `Reflect::get` alone cannot.
    struct ReflectSource<'a>(&'a JsValue);

    impl Properties for ReflectSource<'_> {
        fn read(&self, key: &str) -> Result<Raw, RaaError> {
            let object = self.0;
            if !object.is_object() || object.is_null() || Array::is_array(object) {
                return Err(invalid("a light description must be an object"));
            }
            let name = JsValue::from_str(key);
            let present =
                Reflect::has(object, &name).map_err(|_| invalid(format!("cannot read '{key}'")))?;
            if !present {
                return Ok(Raw::Absent);
            }
            let value =
                Reflect::get(object, &name).map_err(|_| invalid(format!("cannot read '{key}'")))?;
            Ok(raw_of(&value))
        }
    }

    /// Classifies a JS value without narrowing it.
    fn raw_of(value: &JsValue) -> Raw {
        if value.is_undefined() || value.is_null() {
            return Raw::Undefined;
        }
        if let Some(b) = value.as_bool() {
            return Raw::Bool(b);
        }
        if let Some(n) = value.as_f64() {
            return Raw::Number(n);
        }
        if let Some(s) = value.as_string() {
            return Raw::String(s);
        }
        if let Some(f) = value.dyn_ref::<Float32Array>() {
            return Raw::Numbers(f.to_vec().into_iter().map(f64::from).collect());
        }
        if let Some(a) = value.dyn_ref::<Array>() {
            let mut out = Vec::with_capacity(a.length() as usize);
            for i in 0..a.length() {
                match a.get(i).as_f64() {
                    Some(n) => out.push(n),
                    None => return Raw::Other,
                }
            }
            return Raw::Numbers(out);
        }
        Raw::Other
    }

    /// Parses a JS light description or change object into a patch. The S17 mutators call
    /// this and merge the result onto the current descriptor with [`merge`].
    #[allow(dead_code)] // used by the S17 mutators of the later api-m28 cards
    pub(crate) fn read_patch(object: &JsValue) -> Result<LocalLightPatch, RaaError> {
        parse(&ReflectSource(object))
    }
}
