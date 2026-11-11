//! m08 sky and image-based lighting API: the procedural sky (`setSky`, `sky`), the celestial
//! clock (`setTimeOfDay`, `timeOfDay`), the typed sky/weather observation
//! (`skyWeatherDiagnostics`) and the intensities of the sky-derived environment lighting
//! (`setEnvironmentLighting`).
//!
//! `install` adds IBL first, then the sky (which filters its env cube through the IBL
//! handle); `frame` marks the sky dirty when the accepted scene light changed, so the env
//! cube and its IBL follow the sun. The frame hook never advances the clock: `EngineApp`
//! already advanced it once from `FrameInput.dt_ms / 1000.0` under
//! `EngineState::sky_invocation_id()` before `api::frame_all`, and no submitted-frame
//! counter is used here.
//!
//! `setSky` is the S24 transaction boundary: it merges the JS patch over the selected
//! preset (or over the current sky when no preset is given), validates the complete
//! descriptor and hands `raa_sky::SkyTransaction` to `SkyHandle::set_transaction` exactly
//! once. The transaction keeps the caller's explicit intent separate from resolved
//! equality: `preset` records an explicit preset selection (which resets the preset moment
//! in `raa-sky`) even when every resolved `SkyDesc` value already equals the accepted one,
//! and `cloud_fields` records which shared legacy cloud values were supplied. An empty
//! patch therefore retains the authored moment while `{ preset: 'day' }` resets it, even
//! though both resolve to the same descriptor.
//!
//! The parser below is pure Rust: `tests/sky_weather.rs` exercises the merge and its
//! validation natively, and the wasm32 [`web`] module reads JS objects through
//! `js_sys::Reflect`. Sky refusals use `sky.invalid` and name the complete camelCase field
//! path; a non-object argument keeps `api.invalid_argument`.

use std::fmt::Display;
use std::rc::Rc;

use raa_core::RaaError;
use raa_ibl::IblPublication;
use raa_sky::{
    EnvUpdate, LegacyCloudFields, SkyDesc, SkyDiagnostics, SkyModel, SkyTransaction,
    TimeOfDaySettings,
};

/// The refusal code of an invalid sky/time/weather candidate (S24).
pub(crate) const SKY_INVALID: &str = "sky.invalid";

/// The largest integer JavaScript represents exactly (`Number.MAX_SAFE_INTEGER`).
pub(crate) const MAX_SAFE_COUNTER: u64 = (1 << 53) - 1;

/// A raw property value read from a description object.
#[derive(Clone)]
pub(crate) enum Raw {
    /// The key is not a property.
    Absent,
    /// The key is present with `undefined`.
    Undefined,
    /// The key is present with `null`.
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Numbers(Vec<f64>),
    /// A nested description object.
    Object(Rc<dyn Properties>),
    /// Present with a value of no accepted shape.
    Other,
}

/// The property source of a description object. `keys` enumerates the supplied keys (for
/// the S24 unknown-key refusal) and `read` separates absence, `undefined` and `null`, which
/// `Reflect::get` alone cannot.
pub(crate) trait Properties {
    fn keys(&self) -> Result<Vec<String>, RaaError>;
    fn read(&self, key: &str) -> Result<Raw, RaaError>;
}

fn invalid(path: &str, message: impl Display) -> RaaError {
    RaaError::FrameGraph {
        code: SKY_INVALID,
        message: format!("'{path}' {message}"),
    }
}

fn join(prefix: &str, key: &str) -> String {
    if prefix.is_empty() {
        key.to_string()
    } else {
        format!("{prefix}.{key}")
    }
}

const SKY_FIELDS: &[&str] = &[
    "preset",
    "model",
    "zenith",
    "horizon",
    "ground",
    "intensity",
    "horizonSoftness",
    "sunDiscSize",
    "sunDiscIntensity",
    "clouds",
    "envUpdate",
    "atmosphere",
];

const CLOUD_FIELDS: &[&str] = &[
    "coverage",
    "density",
    "altitude",
    "scale",
    "wind",
    "color",
    "silverLining",
];

const ATMOSPHERE_FIELDS: &[&str] = &["groundAlbedo", "mieMultiplier"];

/// Refuses the first supplied key that is not in `allowed`, with its complete path.
fn reject_unknown(obj: &dyn Properties, prefix: &str, allowed: &[&str]) -> Result<(), RaaError> {
    for key in obj.keys()? {
        if !allowed.contains(&key.as_str()) {
            return Err(invalid(&join(prefix, &key), "is not a known field"));
        }
    }
    Ok(())
}

/// Narrows a JS number to the f32 the GPU boundary stores, refusing overflow.
fn narrow(path: &str, value: f64) -> Result<f32, RaaError> {
    if !value.is_finite() {
        return Err(invalid(path, format!("must be finite, got {value}")));
    }
    let narrowed = value as f32;
    if !narrowed.is_finite() {
        return Err(invalid(
            path,
            format!("must fit in a 32-bit float, got {value}"),
        ));
    }
    Ok(narrowed)
}

fn at_least(path: &str, value: f32, min: f32) -> Result<(), RaaError> {
    if value >= min {
        Ok(())
    } else {
        Err(invalid(path, format!("must be >= {min}, got {value}")))
    }
}

fn positive(path: &str, value: f32) -> Result<(), RaaError> {
    if value > 0.0 {
        Ok(())
    } else {
        Err(invalid(path, format!("must be > 0, got {value}")))
    }
}

fn within(path: &str, value: f32, min: f32, max: f32) -> Result<(), RaaError> {
    if (min..=max).contains(&value) {
        Ok(())
    } else {
        Err(invalid(
            path,
            format!("must be in [{min}, {max}], got {value}"),
        ))
    }
}

fn color(path: &str, c: [f32; 3]) -> Result<(), RaaError> {
    for (i, v) in c.into_iter().enumerate() {
        at_least(&format!("{path}[{i}]"), v, 0.0)?;
    }
    Ok(())
}

/// An old numeric field: missing, `undefined` and `null` are omissions (v0 compatibility).
fn old_number(obj: &dyn Properties, key: &str, path: &str) -> Result<Option<f32>, RaaError> {
    match obj.read(key)? {
        Raw::Absent | Raw::Undefined | Raw::Null => Ok(None),
        Raw::Number(value) => narrow(path, value).map(Some),
        _ => Err(invalid(path, "must be a number")),
    }
}

/// A new numeric field: `null` is refused, `undefined` is an omission (S24 strict rule).
fn strict_number(obj: &dyn Properties, key: &str, path: &str) -> Result<Option<f32>, RaaError> {
    match obj.read(key)? {
        Raw::Absent | Raw::Undefined => Ok(None),
        Raw::Null => Err(invalid(path, "must not be null")),
        Raw::Number(value) => narrow(path, value).map(Some),
        _ => Err(invalid(path, "must be a number")),
    }
}

/// A new `f64` field (the clock stores `hours`/`speed` as f64): only finiteness applies.
fn strict_f64(obj: &dyn Properties, key: &str, path: &str) -> Result<Option<f64>, RaaError> {
    match obj.read(key)? {
        Raw::Absent | Raw::Undefined => Ok(None),
        Raw::Null => Err(invalid(path, "must not be null")),
        Raw::Number(value) if value.is_finite() => Ok(Some(value)),
        Raw::Number(value) => Err(invalid(path, format!("must be finite, got {value}"))),
        _ => Err(invalid(path, "must be a number")),
    }
}

/// A new boolean field: `null` is refused, `undefined` is an omission.
fn strict_bool(obj: &dyn Properties, key: &str, path: &str) -> Result<Option<bool>, RaaError> {
    match obj.read(key)? {
        Raw::Absent | Raw::Undefined => Ok(None),
        Raw::Null => Err(invalid(path, "must not be null")),
        Raw::Bool(value) => Ok(Some(value)),
        _ => Err(invalid(path, "must be a boolean")),
    }
}

fn numbers<const N: usize>(path: &str, values: Vec<f64>) -> Result<[f32; N], RaaError> {
    if values.len() != N {
        return Err(invalid(
            path,
            format!("must have {N} components, got {}", values.len()),
        ));
    }
    let mut out = [0.0f32; N];
    for (i, value) in values.into_iter().enumerate() {
        out[i] = narrow(&format!("{path}[{i}]"), value)?;
    }
    Ok(out)
}

/// An old numeric array field: missing, `undefined` and `null` are omissions.
fn old_floats<const N: usize>(
    obj: &dyn Properties,
    key: &str,
    path: &str,
) -> Result<Option<[f32; N]>, RaaError> {
    match obj.read(key)? {
        Raw::Absent | Raw::Undefined | Raw::Null => Ok(None),
        Raw::Numbers(values) => numbers(path, values).map(Some),
        _ => Err(invalid(
            path,
            format!("must be an Array or Float32Array of {N} numbers"),
        )),
    }
}

/// A new numeric array field: `null` is refused, `undefined` is an omission.
fn strict_floats<const N: usize>(
    obj: &dyn Properties,
    key: &str,
    path: &str,
) -> Result<Option<[f32; N]>, RaaError> {
    match obj.read(key)? {
        Raw::Absent | Raw::Undefined => Ok(None),
        Raw::Null => Err(invalid(path, "must not be null")),
        Raw::Numbers(values) => numbers(path, values).map(Some),
        _ => Err(invalid(
            path,
            format!("must be an Array or Float32Array of {N} numbers"),
        )),
    }
}

/// An old string field: missing, `undefined` and `null` are omissions.
fn old_string(obj: &dyn Properties, key: &str, path: &str) -> Result<Option<String>, RaaError> {
    match obj.read(key)? {
        Raw::Absent | Raw::Undefined | Raw::Null => Ok(None),
        Raw::String(value) => Ok(Some(value)),
        _ => Err(invalid(path, "must be a string")),
    }
}

/// A new string field: `null` is refused, `undefined` is an omission.
fn strict_string(obj: &dyn Properties, key: &str, path: &str) -> Result<Option<String>, RaaError> {
    match obj.read(key)? {
        Raw::Absent | Raw::Undefined => Ok(None),
        Raw::Null => Err(invalid(path, "must not be null")),
        Raw::String(value) => Ok(Some(value)),
        _ => Err(invalid(path, "must be a string")),
    }
}

/// An old nested object field: missing, `undefined` and `null` are omissions.
fn old_object(
    obj: &dyn Properties,
    key: &str,
    path: &str,
) -> Result<Option<Rc<dyn Properties>>, RaaError> {
    match obj.read(key)? {
        Raw::Absent | Raw::Undefined | Raw::Null => Ok(None),
        Raw::Object(value) => Ok(Some(value)),
        _ => Err(invalid(path, "must be an object")),
    }
}

/// A new nested object field: `null` is refused, `undefined` is an omission.
fn strict_object(
    obj: &dyn Properties,
    key: &str,
    path: &str,
) -> Result<Option<Rc<dyn Properties>>, RaaError> {
    match obj.read(key)? {
        Raw::Absent | Raw::Undefined => Ok(None),
        Raw::Null => Err(invalid(path, "must not be null")),
        Raw::Object(value) => Ok(Some(value)),
        _ => Err(invalid(path, "must be an object")),
    }
}

/// Parses a JS `SkyDesc` patch over `current` into the exact S24 transaction.
///
/// The preset (when present) supplies the base descriptor, then every other supplied field
/// overrides it. `preset` is `Some` only for an explicit preset selection and
/// `cloud_fields` marks only the shared cloud values (coverage, density, altitude, wind)
/// that the patch actually supplied, so neither can be inferred from equal values. The
/// complete merged descriptor is validated before the transaction is returned; the caller
/// then invokes `SkyHandle::set_transaction` once.
pub(crate) fn parse_sky_transaction(
    current: &SkyDesc,
    obj: &dyn Properties,
) -> Result<SkyTransaction, RaaError> {
    reject_unknown(obj, "", SKY_FIELDS)?;
    let (preset, mut desc) = match obj.read("preset")? {
        Raw::Absent | Raw::Undefined | Raw::Null => (None, current.clone()),
        Raw::String(name) => {
            let base = SkyDesc::preset(&name).ok_or_else(|| {
                invalid(
                    "preset",
                    format!("must be 'day', 'snow', 'jungle' or 'night', got '{name}'"),
                )
            })?;
            (Some(base.profile), base)
        }
        _ => {
            return Err(invalid(
                "preset",
                "must be 'day', 'snow', 'jungle' or 'night'",
            ));
        }
    };

    if let Some(model) = strict_string(obj, "model", "model")? {
        desc.model = match model.as_str() {
            "physical" => SkyModel::Physical,
            "legacy" => SkyModel::Legacy,
            other => {
                return Err(invalid(
                    "model",
                    format!("must be 'physical' or 'legacy', got '{other}'"),
                ));
            }
        };
    }
    if let Some(v) = old_floats::<3>(obj, "zenith", "zenith")? {
        color("zenith", v)?;
        desc.zenith = v;
    }
    if let Some(v) = old_floats::<3>(obj, "horizon", "horizon")? {
        color("horizon", v)?;
        desc.horizon = v;
    }
    if let Some(v) = old_floats::<3>(obj, "ground", "ground")? {
        color("ground", v)?;
        desc.ground = v;
    }
    if let Some(v) = old_number(obj, "intensity", "intensity")? {
        at_least("intensity", v, 0.0)?;
        desc.intensity = v;
    }
    if let Some(v) = old_number(obj, "horizonSoftness", "horizonSoftness")? {
        positive("horizonSoftness", v)?;
        desc.horizon_softness = v;
    }
    if let Some(v) = old_number(obj, "sunDiscSize", "sunDiscSize")? {
        within("sunDiscSize", v, 0.0, 0.2)?;
        desc.sun_disc_angular_radius = v;
    }
    if let Some(v) = old_number(obj, "sunDiscIntensity", "sunDiscIntensity")? {
        at_least("sunDiscIntensity", v, 0.0)?;
        desc.sun_disc_intensity = v;
    }

    let mut cloud_fields = LegacyCloudFields::NONE;
    if let Some(clouds) = old_object(obj, "clouds", "clouds")? {
        let clouds: &dyn Properties = &*clouds;
        reject_unknown(clouds, "clouds", CLOUD_FIELDS)?;
        let c = &mut desc.clouds;
        if let Some(v) = old_number(clouds, "coverage", "clouds.coverage")? {
            within("clouds.coverage", v, 0.0, 1.0)?;
            c.coverage = v;
            cloud_fields.0 |= LegacyCloudFields::COVERAGE.0;
        }
        if let Some(v) = old_number(clouds, "density", "clouds.density")? {
            at_least("clouds.density", v, 0.0)?;
            c.density = v;
            cloud_fields.0 |= LegacyCloudFields::DENSITY.0;
        }
        if let Some(v) = old_number(clouds, "altitude", "clouds.altitude")? {
            positive("clouds.altitude", v)?;
            c.altitude_m = v;
            cloud_fields.0 |= LegacyCloudFields::ALTITUDE.0;
        }
        if let Some(v) = old_number(clouds, "scale", "clouds.scale")? {
            positive("clouds.scale", v)?;
            c.scale_m = v;
        }
        if let Some(v) = old_floats::<2>(clouds, "wind", "clouds.wind")? {
            c.wind_mps = v;
            cloud_fields.0 |= LegacyCloudFields::WIND.0;
        }
        if let Some(v) = old_floats::<3>(clouds, "color", "clouds.color")? {
            color("clouds.color", v)?;
            c.color = v;
        }
        if let Some(v) = old_number(clouds, "silverLining", "clouds.silverLining")? {
            within("clouds.silverLining", v, 0.0, 1.0)?;
            c.silver_lining = v;
        }
    }
    if let Some(name) = old_string(obj, "envUpdate", "envUpdate")? {
        desc.env_update = match name.as_str() {
            "onChange" => EnvUpdate::OnChange,
            "everyFrame" => EnvUpdate::EveryFrame,
            _ => {
                return Err(invalid(
                    "envUpdate",
                    format!("must be 'onChange' or 'everyFrame', got '{name}'"),
                ));
            }
        };
    }
    if let Some(atmosphere) = strict_object(obj, "atmosphere", "atmosphere")? {
        let atmosphere: &dyn Properties = &*atmosphere;
        reject_unknown(atmosphere, "atmosphere", ATMOSPHERE_FIELDS)?;
        if let Some(v) = strict_floats::<3>(atmosphere, "groundAlbedo", "atmosphere.groundAlbedo")?
        {
            for (i, value) in v.iter().enumerate() {
                within(&format!("atmosphere.groundAlbedo[{i}]"), *value, 0.0, 1.0)?;
            }
            desc.atmosphere.ground_albedo = v;
        }
        if let Some(v) = strict_number(atmosphere, "mieMultiplier", "atmosphere.mieMultiplier")? {
            positive("atmosphere.mieMultiplier", v)?;
            desc.atmosphere.mie_multiplier = v;
        }
    }

    if preset.is_some_and(|p| p != desc.profile) {
        return Err(invalid("preset", "must agree with the selected profile"));
    }
    // The per-field checks above cover the patch; this verifies the complete merged result
    // before the one accepted transaction (and never after a partial mutation).
    desc.validate()?;
    Ok(SkyTransaction {
        desc,
        preset,
        cloud_fields,
    })
}

/// The S24 `TimeOfDayDesc` fields; `undefined` is the only omission.
const TIME_OF_DAY_FIELDS: &[&str] = &[
    "day", "hours", "latitude", "north", "enabled", "playing", "speed",
];

/// Merges a JS `TimeOfDayDesc` over the accepted authored settings and validates the
/// complete result (S24: day 1..=365, hours [0, 24), latitude [-90, 90], finite north,
/// finite speed >= 0). `null`, unknown keys, wrong types and f32 overflow refuse with
/// `sky.invalid` naming the complete field path before anything is applied.
pub(crate) fn parse_time_of_day(
    current: &TimeOfDaySettings,
    obj: &dyn Properties,
) -> Result<TimeOfDaySettings, RaaError> {
    reject_unknown(obj, "", TIME_OF_DAY_FIELDS)?;
    let mut settings = *current;
    if let Some(value) = strict_f64(obj, "day", "day")? {
        if value.fract() != 0.0 || !(1.0..=365.0).contains(&value) {
            return Err(invalid(
                "day",
                format!("must be an integer in [1, 365], got {value}"),
            ));
        }
        settings.time.day = value as u16;
    }
    if let Some(value) = strict_f64(obj, "hours", "hours")? {
        if !(0.0..24.0).contains(&value) {
            return Err(invalid("hours", format!("must be in [0, 24), got {value}")));
        }
        settings.time.hours = value;
    }
    if let Some(value) = strict_number(obj, "latitude", "latitude")? {
        if !(-90.0..=90.0).contains(&value) {
            return Err(invalid(
                "latitude",
                format!("must be in [-90, 90], got {value}"),
            ));
        }
        settings.place.latitude = value;
    }
    if let Some(value) = strict_number(obj, "north", "north")? {
        settings.place.north = value;
    }
    if let Some(value) = strict_bool(obj, "enabled", "enabled")? {
        settings.enabled = value;
    }
    if let Some(value) = strict_bool(obj, "playing", "playing")? {
        settings.playing = value;
    }
    if let Some(value) = strict_f64(obj, "speed", "speed")? {
        if value < 0.0 {
            return Err(invalid("speed", format!("must be >= 0, got {value}")));
        }
        settings.speed = value;
    }
    settings.validate()?;
    Ok(settings)
}

/// The S24 `TimeOfDayState` flattened into its JS fields (fresh on every getter call).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TimeOfDayView {
    pub day: u16,
    pub hours: f64,
    pub latitude: f32,
    pub north: f32,
    pub enabled: bool,
    pub playing: bool,
    pub speed: f64,
    pub current_day: u16,
    pub current_hours: f64,
    pub sun_direction: [f32; 3],
    pub moon_direction: [f32; 3],
    pub moon_lit: f32,
    pub star_pole: [f32; 3],
    pub star_equinox: [f32; 3],
    pub dominant: &'static str,
}

fn xyz(v: raa_math::Vec3) -> [f32; 3] {
    [v.x, v.y, v.z]
}

/// Flattens the copied `TimeOfDayState`: authored settings, runtime cursor, celestial
/// snapshot and the exact `manual`/`sun`/`moon` dominant spelling.
pub(crate) fn serialize_time_of_day(state: &raa_sky::TimeOfDayState) -> TimeOfDayView {
    let s = &state.settings;
    TimeOfDayView {
        day: s.time.day,
        hours: s.time.hours,
        latitude: s.place.latitude,
        north: s.place.north,
        enabled: s.enabled,
        playing: s.playing,
        speed: s.speed,
        current_day: state.current.day,
        current_hours: state.current.hours,
        sun_direction: xyz(state.sky.sun),
        moon_direction: xyz(state.sky.moon),
        moon_lit: state.sky.moon_lit,
        star_pole: xyz(state.sky.pole),
        star_equinox: xyz(state.sky.equinox),
        dominant: match state.dominant {
            raa_sky::DominantLight::Manual => "manual",
            raa_sky::DominantLight::Sun => "sun",
            raa_sky::DominantLight::Moon => "moon",
        },
    }
}

/// The JS-shaped `SkyWeatherDiagnostics`: the exact 7434 records with ordered views,
/// option-to-null mapping, readiness strings and the ordered reset reason array.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SkyWeatherDiagnostics {
    pub invocation: Option<u64>,
    pub submitted_invocation: Option<u64>,
    pub weather_generation: u64,
    pub atmosphere_generation: u64,
    pub environment: IblPublication,
    pub views: Vec<SkyWeatherView>,
    pub readiness: SkyWeatherReadiness,
}

/// One prepared view's serialized diagnostics record.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SkyWeatherView {
    pub view: u32,
    pub width: u32,
    pub height: u32,
    pub invocation: Option<u64>,
    pub submitted_invocation: Option<u64>,
    pub sky_view_generation: u64,
    pub history_valid: bool,
    pub reset_reasons: Vec<&'static str>,
    pub cloud_screen_valid: bool,
    pub cloud_panorama_valid: bool,
}

/// The serialized readiness record; `error` is null unless the status is failed.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SkyWeatherReadiness {
    pub status: &'static str,
    pub stage: &'static str,
    pub error: Option<String>,
}

/// A counter that must stay an exactly representable nonnegative JS integer; a larger value
/// refuses rather than rounding or wrapping.
fn safe_counter(path: &str, value: u64) -> Result<u64, RaaError> {
    if value > MAX_SAFE_COUNTER {
        return Err(invalid(
            path,
            format!("must stay a safe nonnegative JS integer, got {value}"),
        ));
    }
    Ok(value)
}

fn optional_counter(path: &str, value: Option<u64>) -> Result<Option<u64>, RaaError> {
    value.map(|v| safe_counter(path, v)).transpose()
}

/// Serializes the exact `SkyDiagnostics`: absent view slots are skipped in slot order,
/// options become `None` (null at the JS boundary) and failures keep their originating
/// message. Reading it never mutates the renderer state.
pub(crate) fn serialize_diagnostics(d: &SkyDiagnostics) -> Result<SkyWeatherDiagnostics, RaaError> {
    let environment = IblPublication {
        requested_generation: safe_counter(
            "environment.requestedGeneration",
            d.environment.requested_generation,
        )?,
        published_generation: optional_counter(
            "environment.publishedGeneration",
            d.environment.published_generation,
        )?,
        pending: d.environment.pending,
    };
    let mut views = Vec::new();
    for view in d.views.iter().flatten() {
        views.push(SkyWeatherView {
            view: view.view,
            width: view.width,
            height: view.height,
            invocation: optional_counter("views.invocation", view.invocation)?,
            submitted_invocation: optional_counter(
                "views.submittedInvocation",
                view.submitted_invocation,
            )?,
            sky_view_generation: safe_counter("views.skyViewGeneration", view.sky_view_generation)?,
            history_valid: view.history_valid,
            reset_reasons: view.reset_reasons.names(),
            cloud_screen_valid: view.cloud_screen_valid,
            cloud_panorama_valid: view.cloud_panorama_valid,
        });
    }
    Ok(SkyWeatherDiagnostics {
        invocation: optional_counter("invocation", d.invocation)?,
        submitted_invocation: optional_counter("submittedInvocation", d.submitted_invocation)?,
        weather_generation: safe_counter("weatherGeneration", d.weather_generation)?,
        atmosphere_generation: safe_counter("atmosphereGeneration", d.atmosphere_generation)?,
        environment,
        views,
        readiness: SkyWeatherReadiness {
            status: d.readiness.status.as_str(),
            stage: d.readiness.stage.as_str(),
            error: d.readiness.error.clone(),
        },
    })
}

#[cfg(target_arch = "wasm32")]
pub(crate) use web::{frame, install};

#[cfg(target_arch = "wasm32")]
mod web {
    use super::*;
    use crate::convert::{INVALID_ARGUMENT, api_error, js};
    use crate::engine::{Engine, InstallCtx};
    use crate::state::EngineState;
    use js_sys::{Array, Float32Array, Object, Reflect};
    use raa_core::GpuContext;
    use raa_ibl::{IblFeature, IblHandle};
    use raa_scene::DirectionalLight;
    use raa_sky::{SkyFeature, SkyHandle};
    use raa_web::FrameInput;
    use wasm_bindgen::{JsCast, prelude::*};

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

    fn arg_invalid(message: impl Into<String>) -> RaaError {
        api_error(INVALID_ARGUMENT, message)
    }

    /// A [`Properties`] view of a JS object. `Reflect::has` separates an absent key from one
    /// whose value is `undefined`, and an explicit `null` stays distinct from both.
    struct ReflectSource(JsValue);

    impl Properties for ReflectSource {
        fn keys(&self) -> Result<Vec<String>, RaaError> {
            let object = self
                .0
                .dyn_ref::<Object>()
                .ok_or_else(|| invalid("", "must be an object"))?;
            let mut keys = Vec::new();
            for key in Object::keys(object).iter() {
                let key = key
                    .as_string()
                    .ok_or_else(|| arg_invalid("description keys must be strings"))?;
                keys.push(key);
            }
            Ok(keys)
        }

        fn read(&self, key: &str) -> Result<Raw, RaaError> {
            if !self.0.is_object() || self.0.is_null() {
                return Err(arg_invalid(format!(
                    "expected an object holding '{key}', got a non-object"
                )));
            }
            let name = JsValue::from_str(key);
            let present = Reflect::has(&self.0, &name)
                .map_err(|_| arg_invalid(format!("cannot read '{key}'")))?;
            if !present {
                return Ok(Raw::Absent);
            }
            let value = Reflect::get(&self.0, &name)
                .map_err(|_| arg_invalid(format!("cannot read '{key}'")))?;
            Ok(Raw::from_js(&value))
        }
    }

    impl Raw {
        fn from_js(value: &JsValue) -> Raw {
            if value.is_undefined() {
                return Raw::Undefined;
            }
            if value.is_null() {
                return Raw::Null;
            }
            if let Some(value) = value.as_bool() {
                return Raw::Bool(value);
            }
            if let Some(value) = value.as_f64() {
                return Raw::Number(value);
            }
            if let Some(value) = value.as_string() {
                return Raw::String(value);
            }
            if let Some(array) = value.dyn_ref::<Float32Array>() {
                let mut values = vec![0.0f32; array.length() as usize];
                array.copy_to(&mut values);
                return Raw::Numbers(values.into_iter().map(f64::from).collect());
            }
            if Array::is_array(value) {
                let array: &Array = value.unchecked_ref();
                let mut values = Vec::with_capacity(array.length() as usize);
                for i in 0..array.length() {
                    let Some(value) = array.get(i).as_f64() else {
                        return Raw::Other;
                    };
                    values.push(value);
                }
                return Raw::Numbers(values);
            }
            if value.is_object() {
                return Raw::Object(Rc::new(ReflectSource(value.clone())));
            }
            Raw::Other
        }
    }

    fn model_str(model: SkyModel) -> &'static str {
        match model {
            SkyModel::Physical => "physical",
            SkyModel::Legacy => "legacy",
        }
    }

    fn env_update_str(update: EnvUpdate) -> &'static str {
        match update {
            EnvUpdate::OnChange => "onChange",
            EnvUpdate::EveryFrame => "everyFrame",
        }
    }

    /// Sets `key` on a fresh plain object; this cannot throw, so the result is ignored.
    fn set(obj: &Object, key: &str, value: &JsValue) {
        let _ = Reflect::set(obj, &JsValue::from_str(key), value);
    }

    fn floats_to_js(values: &[f32]) -> JsValue {
        values
            .iter()
            .map(|x| JsValue::from(*x))
            .collect::<Array>()
            .into()
    }

    /// The S24 `ResolvedSkyDesc` of `d`: the merged authored descriptor with its profile
    /// identity and physical atmosphere controls, and no `preset` intent field.
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
        let atmosphere = Object::new();
        set(
            &atmosphere,
            "groundAlbedo",
            &floats_to_js(&d.atmosphere.ground_albedo),
        );
        set(
            &atmosphere,
            "mieMultiplier",
            &JsValue::from(d.atmosphere.mie_multiplier),
        );
        let o = Object::new();
        set(&o, "model", &JsValue::from_str(model_str(d.model)));
        set(&o, "profile", &JsValue::from_str(d.profile.as_str()));
        set(&o, "zenith", &floats_to_js(&d.zenith));
        set(&o, "horizon", &floats_to_js(&d.horizon));
        set(&o, "ground", &floats_to_js(&d.ground));
        set(&o, "intensity", &JsValue::from(d.intensity));
        set(&o, "horizonSoftness", &JsValue::from(d.horizon_softness));
        set(&o, "sunDiscSize", &JsValue::from(d.sun_disc_angular_radius));
        set(&o, "sunDiscIntensity", &JsValue::from(d.sun_disc_intensity));
        set(&o, "clouds", &clouds.into());
        set(
            &o,
            "envUpdate",
            &JsValue::from_str(env_update_str(d.env_update)),
        );
        set(&o, "atmosphere", &atmosphere.into());
        o.into()
    }

    fn optional_number(value: Option<u64>) -> JsValue {
        value.map_or(JsValue::NULL, |v| JsValue::from_f64(v as f64))
    }

    fn strings_to_js(values: &[&'static str]) -> JsValue {
        values
            .iter()
            .map(|s| JsValue::from_str(s))
            .collect::<Array>()
            .into()
    }

    /// The S24 `TimeOfDayState` JS object: authored settings, cursor, celestial snapshot.
    fn time_of_day_to_js(v: &TimeOfDayView) -> JsValue {
        let o = Object::new();
        set(&o, "day", &JsValue::from(v.day));
        set(&o, "hours", &JsValue::from(v.hours));
        set(&o, "latitude", &JsValue::from(v.latitude));
        set(&o, "north", &JsValue::from(v.north));
        set(&o, "enabled", &JsValue::from_bool(v.enabled));
        set(&o, "playing", &JsValue::from_bool(v.playing));
        set(&o, "speed", &JsValue::from(v.speed));
        set(&o, "currentDay", &JsValue::from(v.current_day));
        set(&o, "currentHours", &JsValue::from(v.current_hours));
        set(&o, "sunDirection", &floats_to_js(&v.sun_direction));
        set(&o, "moonDirection", &floats_to_js(&v.moon_direction));
        set(&o, "moonLit", &JsValue::from(v.moon_lit));
        set(&o, "starPole", &floats_to_js(&v.star_pole));
        set(&o, "starEquinox", &floats_to_js(&v.star_equinox));
        set(&o, "dominant", &JsValue::from_str(v.dominant));
        o.into()
    }

    fn view_diagnostics_to_js(v: &SkyWeatherView) -> JsValue {
        let o = Object::new();
        set(&o, "view", &JsValue::from(v.view));
        set(&o, "width", &JsValue::from(v.width));
        set(&o, "height", &JsValue::from(v.height));
        set(&o, "invocation", &optional_number(v.invocation));
        set(
            &o,
            "submittedInvocation",
            &optional_number(v.submitted_invocation),
        );
        set(
            &o,
            "skyViewGeneration",
            &JsValue::from_f64(v.sky_view_generation as f64),
        );
        set(&o, "historyValid", &JsValue::from_bool(v.history_valid));
        set(&o, "resetReasons", &strings_to_js(&v.reset_reasons));
        set(
            &o,
            "cloudScreenValid",
            &JsValue::from_bool(v.cloud_screen_valid),
        );
        set(
            &o,
            "cloudPanoramaValid",
            &JsValue::from_bool(v.cloud_panorama_valid),
        );
        o.into()
    }

    /// The exact fresh `SkyWeatherDiagnostics` object; absent views are an empty array and
    /// options are null.
    fn diagnostics_to_js(d: &SkyWeatherDiagnostics) -> JsValue {
        let environment = Object::new();
        set(
            &environment,
            "requestedGeneration",
            &JsValue::from_f64(d.environment.requested_generation as f64),
        );
        set(
            &environment,
            "publishedGeneration",
            &optional_number(d.environment.published_generation),
        );
        set(
            &environment,
            "pending",
            &JsValue::from_bool(d.environment.pending),
        );
        let views: Array = d.views.iter().map(view_diagnostics_to_js).collect();
        let readiness = Object::new();
        set(&readiness, "status", &JsValue::from_str(d.readiness.status));
        set(&readiness, "stage", &JsValue::from_str(d.readiness.stage));
        set(
            &readiness,
            "error",
            &d.readiness
                .error
                .as_deref()
                .map_or(JsValue::NULL, JsValue::from_str),
        );
        let o = Object::new();
        set(&o, "invocation", &optional_number(d.invocation));
        set(
            &o,
            "submittedInvocation",
            &optional_number(d.submitted_invocation),
        );
        set(
            &o,
            "weatherGeneration",
            &JsValue::from_f64(d.weather_generation as f64),
        );
        set(
            &o,
            "atmosphereGeneration",
            &JsValue::from_f64(d.atmosphere_generation as f64),
        );
        set(&o, "environment", &environment.into());
        set(&o, "views", &views.into());
        set(&o, "readiness", &readiness.into());
        o.into()
    }

    /// A finite intensity >= 0 at `key`, if present (existing `setEnvironmentLighting` rule).
    fn intensity(obj: &JsValue, key: &str) -> Result<Option<f32>, RaaError> {
        match js::optional_f64(obj, key)?.map(|x| x as f32) {
            Some(x) if !(x.is_finite() && x >= 0.0) => Err(arg_invalid(format!(
                "'{key}' must be finite and >= 0, got {x}"
            ))),
            v => Ok(v),
        }
    }

    #[wasm_bindgen(js_class = Engine)]
    impl Engine {
        /// Parses the S24 sky patch, then applies exactly one `SkyTransaction`. A preset
        /// supplies the base before every override; an explicit preset stays explicit even
        /// when the resolved descriptor equals the current one, and an empty patch keeps the
        /// accepted sky and moment. Nothing changes on a refusal.
        #[wasm_bindgen(js_name = setSky)]
        pub fn set_sky(&mut self, desc: JsValue) -> Result<(), JsValue> {
            self.with_state(|_, st| {
                if !desc.is_object() || desc.is_null() || Array::is_array(&desc) {
                    return Err(arg_invalid("setSky expects a sky description object"));
                }
                let sky = st.features.require_mut::<SkyHandle>("sky")?;
                let current = sky.get();
                let source = ReflectSource(desc.clone());
                let transaction = parse_sky_transaction(&current, &source)?;
                sky.set_transaction(transaction)
            })
        }

        /// The resolved S24 sky: model, profile identity, artistic values, clouds, env
        /// update policy and physical atmosphere controls.
        pub fn sky(&self) -> Result<JsValue, JsValue> {
            self.with_state(|_, st| {
                let sky = st.features.require_mut::<SkyHandle>("sky")?;
                Ok(sky_to_js(&sky.get()))
            })
        }

        /// Merges a validated S24 time-of-day patch over the authored settings and applies
        /// it once; a changed authored day or hour resets the runtime cursor in raa-sky.
        /// Manual lighting and the daylight baseline stay owned by raa-sky.
        #[wasm_bindgen(js_name = setTimeOfDay)]
        pub fn set_time_of_day(&mut self, desc: JsValue) -> Result<(), JsValue> {
            self.with_state(|_, st| {
                if !desc.is_object() || desc.is_null() || Array::is_array(&desc) {
                    return Err(arg_invalid(
                        "setTimeOfDay expects a time-of-day description object",
                    ));
                }
                let sky = st.features.require_mut::<SkyHandle>("sky")?;
                let current = sky.time_of_day().settings;
                let settings = parse_time_of_day(&current, &ReflectSource(desc.clone()))?;
                sky.set_time_of_day(settings)
            })
        }

        /// The authored time settings, the runtime playback moment and its celestial
        /// snapshot, as a fresh object; it is never persisted.
        #[wasm_bindgen(js_name = timeOfDay)]
        pub fn time_of_day(&self) -> Result<JsValue, JsValue> {
            self.with_state(|_, st| {
                let sky = st.features.require_mut::<SkyHandle>("sky")?;
                Ok(time_of_day_to_js(&serialize_time_of_day(
                    &sky.time_of_day(),
                )))
            })
        }

        /// The exact typed sky/weather lifecycle observation: invocations, generations,
        /// environment publication, ordered prepared views and readiness. Allocates fresh
        /// objects only on request and changes no clock, resource or authored setting.
        #[wasm_bindgen(js_name = skyWeatherDiagnostics)]
        pub fn sky_weather_diagnostics(&self) -> Result<JsValue, JsValue> {
            self.with_state(|_, st| {
                let sky = st.features.require_mut::<SkyHandle>("sky")?;
                Ok(diagnostics_to_js(&serialize_diagnostics(
                    &sky.diagnostics(),
                )?))
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
}
