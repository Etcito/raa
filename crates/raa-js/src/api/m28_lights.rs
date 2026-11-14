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
//! This share implements the additive S17 mutators `addLight`, `updateLight` and
//! `removeLight`. They read a JS object with `web::read_patch`, merge it onto the current
//! `LocalLight` with [`merge`] (`add` starts from `LocalLight::default()`, `update` from
//! `LitRenderer::local_lights().get(handle)`), and hand the result to
//! `LocalLights::add`/`replace`, which validates with `LocalLight::validate` before it
//! mutates. Handles pass through [`handle_arg`] before any narrowing to `LightHandle`.
//!
//! S17 lifecycle methods share the renderer-owned `LocalLights` registry and stable handles
//! with the add/update/remove methods. Parent keys are JS instance-group handles resolved by
//! `EngineState::instance_parent_world`; their submesh instance slots are never used.

use raa_core::RaaError;
use raa_lit::{LightHandle, LocalLight, LocalLightKind, LocalLights, MAX_LIGHT_HANDLE};

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

/// The logical `FrameGpu` feature id of the diffuse-only area-light form factor: id 39,
/// which `raa-shader`'s generated mapping places in `flags` word 3, bit 7. Setting it makes
/// the area-light branch of `local_lighting.wgsl` use the diffuse form factor instead of LTC.
pub(crate) const AREA_FORM_FACTOR_FEATURE_ID: u32 = 39;

/// `mode` values of the engine-wide area-light shading setting (contract
/// `area-light-shading-api`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AreaLightShading {
    /// Diffuse and GGX specular integrated over the rectangle with the LTC tables.
    Ltc,
    /// The diffuse polygon form factor only, without a table or specular.
    FormFactor,
}

impl AreaLightShading {
    /// The JS spelling of the mode.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            AreaLightShading::Ltc => "ltc",
            AreaLightShading::FormFactor => "formFactor",
        }
    }
}

/// The exact `setAreaLightShading` refusal (contract `area-light-shading-api`).
fn area_shading_invalid() -> RaaError {
    RaaError::FrameGraph {
        code: INVALID_ARGUMENT,
        message: "setAreaLightShading: 'mode' must be 'ltc' or 'formFactor'".into(),
    }
}

/// Parses the `mode` key of an area-light shading description. Only the strings `ltc` and
/// `formFactor` are accepted; an absent, `undefined`, non-string or other string value is
/// refused with `api.invalid_argument` and the contract message.
pub(crate) fn parse_area_shading(source: &impl Properties) -> Result<AreaLightShading, RaaError> {
    match source.read("mode")? {
        Raw::String(mode) if mode == "ltc" => Ok(AreaLightShading::Ltc),
        Raw::String(mode) if mode == "formFactor" => Ok(AreaLightShading::FormFactor),
        _ => Err(area_shading_invalid()),
    }
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

/// S17 `addLight`: merges the parsed description onto the S17 defaults, then validates and
/// registers it. The 257th light refuses with `lights.invalid` and a capacity message, and
/// handle exhaustion refuses instead of reusing a value; neither mutates the registry.
#[allow(dead_code)] // called from the wasm32 Engine methods below
pub(crate) fn add(
    lights: &mut LocalLights,
    patch: &LocalLightPatch,
) -> Result<LightHandle, RaaError> {
    lights.add(merge(LocalLight::default(), patch))
}

/// S17 `updateLight`: resolves the handle, merges the changes onto the stored descriptor and
/// validates the merged result before [`LocalLights::replace`] mutates anything, so a refusal
/// leaves the light unchanged. Fields the update omits (metadata included) are retained, and
/// an explicit inner angle stays subject to the merged outer angle. Only a real position
/// value counts as an authored position, which an attached light refuses.
#[allow(dead_code)] // called from the wasm32 Engine methods below
pub(crate) fn update(
    lights: &mut LocalLights,
    handle: LightHandle,
    patch: &LocalLightPatch,
) -> Result<(), RaaError> {
    let base = *lights.get(handle)?;
    let position_authored = matches!(patch.position, Patch::Set(_));
    lights.replace(handle, merge(base, patch), position_authored)
}

/// S17 `removeLight`: removes a live light. A stale or unknown handle refuses with
/// `api.unknown_handle`; a removed handle is never handed out again.
#[allow(dead_code)] // called from the wasm32 Engine methods below
pub(crate) fn remove(lights: &mut LocalLights, handle: LightHandle) -> Result<(), RaaError> {
    lights.remove(handle)
}

/// S17 `attachLight`: the caller resolves `parent` as an Engine instance-group handle and
/// supplies its current world transform. Registry validation completes before either the
/// attachment or the previous world-space light state changes.
pub(crate) fn attach(
    lights: &mut LocalLights,
    handle: LightHandle,
    parent: u32,
    offset: [f32; 3],
    world: &raa_math::Mat4,
) -> Result<(), RaaError> {
    lights.attach(handle, parent, offset, world)
}

/// S17 `detachLight`: preserve the latest world-space position and direction.
pub(crate) fn detach(lights: &mut LocalLights, handle: LightHandle) -> Result<(), RaaError> {
    lights.detach(handle)
}

#[cfg(target_arch = "wasm32")]
pub(crate) use web::{frame, install};

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use crate::engine::{Engine, InstallCtx};
    use crate::state::EngineState;
    use js_sys::{Array, Float32Array, Object, Reflect};
    use raa_core::GpuContext;
    use raa_web::FrameInput;
    use wasm_bindgen::{JsCast, prelude::*};

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

    fn attachment_offset(value: &JsValue) -> Result<[f32; 3], RaaError> {
        let values = if let Some(array) = value.dyn_ref::<Float32Array>() {
            array
                .to_vec()
                .into_iter()
                .map(f64::from)
                .collect::<Vec<_>>()
        } else if let Some(array) = value.dyn_ref::<Array>() {
            (0..array.length())
                .map(|index| array.get(index).as_f64().unwrap_or(f64::NAN))
                .collect()
        } else {
            return Err(RaaError::FrameGraph {
                code: "light.invalid_attachment",
                message: "'localOffset' must be an array of three finite numbers".into(),
            });
        };
        if values.len() != 3 {
            return Err(RaaError::FrameGraph {
                code: "light.invalid_attachment",
                message: "'localOffset' must have three components".into(),
            });
        }
        let mut offset = [0.0; 3];
        for (out, value) in offset.iter_mut().zip(values) {
            *out = value as f32;
            if !value.is_finite() || !out.is_finite() {
                return Err(RaaError::FrameGraph {
                    code: "light.invalid_attachment",
                    message: "'localOffset' components must be finite f32 values".into(),
                });
            }
        }
        Ok(offset)
    }

    fn set_stats_value(object: &Object, key: &str, value: &JsValue) -> Result<(), RaaError> {
        Reflect::set(object, &JsValue::from_str(key), value)
            .map(|_| ())
            .map_err(|_| RaaError::FrameGraph {
                code: INVALID_ARGUMENT,
                message: format!("could not construct lightStats field '{key}'"),
            })
    }

    fn stats_value(st: &EngineState) -> Result<JsValue, RaaError> {
        let stats = st.lit.light_stats();
        let out = Object::new();
        set_stats_value(
            &out,
            "activeLights",
            &JsValue::from_f64(stats.active_lights as f64),
        )?;
        set_stats_value(
            &out,
            "attachedLights",
            &JsValue::from_f64(stats.attached_lights as f64),
        )?;
        set_stats_value(
            &out,
            "tileColumns",
            &JsValue::from_f64(stats.tile_columns as f64),
        )?;
        set_stats_value(&out, "tileRows", &JsValue::from_f64(stats.tile_rows as f64))?;
        set_stats_value(
            &out,
            "tileCapacity",
            &JsValue::from_f64(stats.tile_capacity as f64),
        )?;
        set_stats_value(
            &out,
            "droppedLightReferences",
            &stats
                .dropped_light_references
                .map_or(JsValue::NULL, |v| JsValue::from_f64(v as f64)),
        )?;
        set_stats_value(
            &out,
            "statsFrameIndex",
            &stats
                .stats_frame_index
                .map_or(JsValue::NULL, |v| JsValue::from_f64(v as f64)),
        )?;
        set_stats_value(&out, "pending", &JsValue::from_bool(stats.pending))?;
        Ok(out.into())
    }

    /// Parses a JS light description or change object into a patch. The S17 mutators call
    /// this and merge the result onto the current descriptor with [`merge`].
    pub(crate) fn read_patch(object: &JsValue) -> Result<LocalLightPatch, RaaError> {
        parse(&ReflectSource(object))
    }

    #[wasm_bindgen(js_class = Engine)]
    impl Engine {
        /// Adds a local light from a `LocalLightDesc` and returns its stable handle
        /// (a positive safe JS integer). Omitted fields take the S17 defaults; a refusal
        /// (`light.invalid_desc`, or `lights.invalid` at the 256-light capacity) adds
        /// nothing. The sun's `setLight` is a separate, unchanged call.
        #[wasm_bindgen(js_name = addLight)]
        pub fn add_light(&mut self, desc: JsValue) -> Result<f64, JsValue> {
            self.with_state(|_, st| {
                let patch = read_patch(&desc)?;
                add(st.lit.local_lights_mut(), &patch).map(|handle| handle as f64)
            })
        }

        /// Merges `changes` onto the light behind `handle` and validates the result before
        /// applying it, so a refusal leaves the light unchanged. Omitted fields, including
        /// `radius` and `shadows`, are retained across kind changes; an explicit retained
        /// inner angle must not exceed the merged outer angle. Unknown or removed handles
        /// throw `api.unknown_handle`, an attached light's authored position throws
        /// `light.attached_position`.
        #[wasm_bindgen(js_name = updateLight)]
        pub fn update_light(&mut self, handle: f64, changes: JsValue) -> Result<(), JsValue> {
            self.with_state(|_, st| {
                let handle = handle_arg("handle", handle)?;
                let patch = read_patch(&changes)?;
                update(st.lit.local_lights_mut(), handle, &patch)
            })
        }

        /// Removes the light behind `handle`; its stable handle is never reused. An unknown
        /// or already removed handle throws `api.unknown_handle`.
        #[wasm_bindgen(js_name = removeLight)]
        pub fn remove_light(&mut self, handle: f64) -> Result<(), JsValue> {
            self.with_state(|_, st| {
                let handle = handle_arg("handle", handle)?;
                remove(st.lit.local_lights_mut(), handle)
            })
        }

        /// Attaches a light to an instance group, using the group's complete world transform.
        /// Reattachment is atomic; invalid handles, offsets, parents or transformed directions
        /// leave the current attachment unchanged.
        #[wasm_bindgen(js_name = attachLight)]
        pub fn attach_light(
            &mut self,
            handle: f64,
            instance_id: u32,
            local_offset: JsValue,
        ) -> Result<(), JsValue> {
            self.with_state(|_, st| {
                let handle = handle_arg("handle", handle)?;
                let offset = attachment_offset(&local_offset)?;
                let world = st.instance_parent_world(instance_id)?;
                attach(
                    st.lit.local_lights_mut(),
                    handle,
                    instance_id,
                    offset,
                    &world,
                )
            })
        }

        /// Detaches a light while preserving its latest world position and direction.
        #[wasm_bindgen(js_name = detachLight)]
        pub fn detach_light(&mut self, handle: f64) -> Result<(), JsValue> {
            self.with_state(|_, st| {
                let handle = handle_arg("handle", handle)?;
                detach(st.lit.local_lights_mut(), handle)
            })
        }

        /// Returns current registry counts and the latest fully completed asynchronous
        /// overflow result. This synchronous call never waits for GPU readback.
        #[wasm_bindgen(js_name = lightStats)]
        pub fn light_stats(&mut self) -> Result<JsValue, JsValue> {
            self.with_state(|_, st| stats_value(st))
        }

        /// Selects the engine-wide area-light shading mode: `ltc` integrates diffuse and GGX
        /// specular over the rectangle with the LTC tables, `formFactor` uses the diffuse
        /// polygon form factor only. The setting persists until changed and applies to every
        /// view. A non-object argument, a missing `mode` or any other value throws
        /// `api.invalid_argument` and changes nothing.
        #[wasm_bindgen(js_name = setAreaLightShading)]
        pub fn set_area_light_shading(&mut self, settings: JsValue) -> Result<(), JsValue> {
            self.with_state(|_, st| {
                if !settings.is_object() || settings.is_null() || Array::is_array(&settings) {
                    return Err(area_shading_invalid());
                }
                let mode = parse_area_shading(&ReflectSource(&settings))
                    .map_err(|_| area_shading_invalid())?;
                st.lit.set_frame_feature(
                    AREA_FORM_FACTOR_FEATURE_ID,
                    mode == AreaLightShading::FormFactor,
                );
                Ok(())
            })
        }

        /// Returns the current area-light shading mode as `{ mode }`. A new Engine reports
        /// `ltc` until `setAreaLightShading` changes it.
        #[wasm_bindgen(js_name = areaLightShading)]
        pub fn area_light_shading(&mut self) -> Result<JsValue, JsValue> {
            self.with_state(|_, st| {
                let mode = if st.lit.frame_feature(AREA_FORM_FACTOR_FEATURE_ID) {
                    AreaLightShading::FormFactor
                } else {
                    AreaLightShading::Ltc
                };
                let out = Object::new();
                Reflect::set(
                    &out,
                    &JsValue::from_str("mode"),
                    &JsValue::from_str(mode.as_str()),
                )
                .map_err(|_| area_shading_invalid())?;
                Ok(out.into())
            })
        }
    }
}
