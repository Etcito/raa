//! Native-testable conversion helpers for the JS API.

use raa_core::RaaError;
use raa_geom::{GeomError, Quality};
use raa_loader::LoadError;
use raa_material::{Material, MaterialError, ShadingModel, srgb_to_linear};
use raa_math::{Quat, Vec3};
use raa_scene::{SceneError, Transform};
use raa_tonemap::ToneMapper;

pub const LENGTH_MISMATCH: &str = "api.length_mismatch";
pub const UNKNOWN_HANDLE: &str = "api.unknown_handle";
pub const INVALID_ARGUMENT: &str = "api.invalid_argument";
pub const UNKNOWN_PARAM: &str = "api.unknown_param";
pub const DISPOSED: &str = "api.disposed";

/// Builds an error with a non-core code. `RaaError` has no generic variant, so
/// the `FrameGraph { code, message }` variant (whose `code()` returns the given
/// code verbatim) is used as the carrier.
pub fn api_error(code: &'static str, message: impl Into<String>) -> RaaError {
    RaaError::FrameGraph {
        code,
        message: message.into(),
    }
}

pub fn geom_error(e: &GeomError) -> RaaError {
    api_error(e.code(), e.to_string())
}

pub fn load_error(e: &LoadError) -> RaaError {
    api_error(e.code, e.to_string())
}

pub fn scene_error(e: &SceneError) -> RaaError {
    api_error(e.code(), e.to_string())
}

pub fn material_error(e: &MaterialError) -> RaaError {
    api_error(e.code, e.to_string())
}

/// Monotonic, never-reused u32 handles (from 1) mapped to internal values.
pub struct HandleMap<T> {
    kind: &'static str,
    slots: Vec<Option<T>>,
    live: usize,
}

impl<T> HandleMap<T> {
    pub fn new(kind: &'static str) -> Self {
        Self {
            kind,
            slots: Vec::new(),
            live: 0,
        }
    }

    pub fn insert(&mut self, v: T) -> u32 {
        self.slots.push(Some(v));
        self.live += 1;
        self.slots.len() as u32
    }

    /// The handle the next `insert` returns.
    pub fn next_handle(&self) -> u32 {
        self.slots.len() as u32 + 1
    }

    fn unknown(kind: &str, h: u32) -> RaaError {
        api_error(UNKNOWN_HANDLE, format!("unknown {kind} handle {h}"))
    }

    pub fn get_ref(&self, h: u32) -> Result<&T, RaaError> {
        (h as usize)
            .checked_sub(1)
            .and_then(|i| self.slots.get(i))
            .and_then(|s| s.as_ref())
            .ok_or_else(|| Self::unknown(self.kind, h))
    }

    pub fn get_mut(&mut self, h: u32) -> Result<&mut T, RaaError> {
        let kind = self.kind;
        (h as usize)
            .checked_sub(1)
            .and_then(|i| self.slots.get_mut(i))
            .and_then(|s| s.as_mut())
            .ok_or_else(|| Self::unknown(kind, h))
    }

    pub fn remove(&mut self, h: u32) -> Result<T, RaaError> {
        let v = (h as usize)
            .checked_sub(1)
            .and_then(|i| self.slots.get_mut(i))
            .and_then(Option::take)
            .ok_or_else(|| Self::unknown(self.kind, h))?;
        self.live -= 1;
        Ok(v)
    }

    /// Live (handle, value) pairs in handle order.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &T)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.as_ref().map(|v| (i as u32 + 1, v)))
    }

    /// Live (handle, value) pairs in handle order, mutable.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (u32, &mut T)> {
        self.slots
            .iter_mut()
            .enumerate()
            .filter_map(|(i, s)| s.as_mut().map(|v| (i as u32 + 1, v)))
    }

    pub fn len(&self) -> usize {
        self.live
    }

    pub fn is_empty(&self) -> bool {
        self.live == 0
    }
}

impl<T: Copy> HandleMap<T> {
    pub fn get(&self, h: u32) -> Result<T, RaaError> {
        self.get_ref(h).copied()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelName {
    Flat,
    Pbr,
}

impl ModelName {
    pub fn parse(s: &str) -> Result<Self, RaaError> {
        match s {
            "flat" => Ok(Self::Flat),
            "pbr" => Ok(Self::Pbr),
            other => Err(api_error(
                INVALID_ARGUMENT,
                format!("model must be 'flat' or 'pbr', got '{other}'"),
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialDesc {
    pub color: [f32; 3],
    pub roughness: f32,
    pub metallic: f32,
    pub emissive: [f32; 3],
    pub model: ModelName,
}

impl MaterialDesc {
    pub fn new(color: [f32; 3]) -> Self {
        Self {
            color,
            roughness: 0.6,
            metallic: 0.0,
            emissive: [0.0; 3],
            model: ModelName::Pbr,
        }
    }

    pub fn to_material(&self) -> Result<Material, RaaError> {
        let m = Material {
            base_color: self.color.map(srgb_to_linear),
            roughness: self.roughness,
            metallic: self.metallic,
            emissive: self.emissive,
            model: match self.model {
                ModelName::Flat => ShadingModel::Flat,
                ModelName::Pbr => ShadingModel::PbrLite,
            },
        };
        m.validate().map_err(|e| material_error(&e))?;
        Ok(m)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraDesc {
    pub eye: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub fov_y_deg: f32,
    pub ortho_height: Option<f32>,
    pub near: f32,
    pub far: f32,
}

impl CameraDesc {
    pub fn new(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> Self {
        Self {
            eye,
            target,
            up,
            fov_y_deg: 60.0,
            ortho_height: None,
            near: 0.1,
            far: 200.0,
        }
    }
}

/// Reads node `i` from 10 floats per node: t3, r4 as [x,y,z,w], s3.
/// The caller checks the length first (`check_trs_len`); out of range reads
/// give the identity transform.
pub fn decode_trs(trs: &[f32], i: usize) -> Transform {
    let Some(c) = trs.get(i * 10..i * 10 + 10) else {
        return Transform::IDENTITY;
    };
    Transform {
        translation: Vec3::new(c[0], c[1], c[2]),
        rotation: Quat::from_xyzw(c[3], c[4], c[5], c[6]),
        scale: Vec3::new(c[7], c[8], c[9]),
    }
}

pub fn check_trs_len(nodes: usize, floats: usize) -> Result<(), RaaError> {
    if floats == nodes.saturating_mul(10) {
        Ok(())
    } else {
        Err(api_error(
            LENGTH_MISMATCH,
            format!("{nodes} nodes need {} floats, got {floats}", nodes * 10),
        ))
    }
}

pub fn quality(segments: Option<f64>) -> Result<Quality, RaaError> {
    match segments {
        None => Ok(Quality::LOD0),
        Some(s) if s.is_finite() => Ok(Quality::new(s.round().clamp(0.0, 1e6) as u32)),
        Some(s) => Err(api_error(
            INVALID_ARGUMENT,
            format!("segments must be finite, got {s}"),
        )),
    }
}

/// Engine options as read from JS, before validation: numbers raw (possibly non-finite),
/// strings as given; `None` = absent, undefined or null.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EngineOptionsInput {
    pub max_instances: Option<f64>,
    pub max_materials: Option<f64>,
    pub power_preference: Option<String>,
    pub msaa: Option<f64>,
    pub tone_mapper: Option<String>,
    pub exposure_ev: Option<f64>,
}

/// Validated engine options (`createEngine(canvas, options)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EngineOptions {
    pub max_instances: u32,
    pub max_materials: u32,
    pub power_preference: wgpu::PowerPreference,
    /// MSAA samples of the forward pass: 1 or 4.
    pub msaa: u32,
    pub tone_mapper: ToneMapper,
    pub exposure_ev: f32,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            max_instances: 16384,
            max_materials: 256,
            power_preference: wgpu::PowerPreference::HighPerformance,
            msaa: 4,
            tone_mapper: ToneMapper::Agx,
            exposure_ev: 0.0,
        }
    }
}

/// A positive integer that fits in u32, or `default` when absent.
fn count(key: &str, v: Option<f64>, default: u32) -> Result<u32, RaaError> {
    match v {
        None => Ok(default),
        Some(x) if x >= 1.0 && x <= f64::from(u32::MAX) && x.fract() == 0.0 => Ok(x as u32),
        Some(x) => Err(api_error(
            INVALID_ARGUMENT,
            format!("'{key}' must be a positive integer, got {x}"),
        )),
    }
}

/// Validates engine options; every refusal is `api.invalid_argument` naming the key.
pub fn engine_options(input: &EngineOptionsInput) -> Result<EngineOptions, RaaError> {
    let d = EngineOptions::default();
    let power_preference = match input.power_preference.as_deref() {
        None | Some("high-performance") => wgpu::PowerPreference::HighPerformance,
        Some("low-power") => wgpu::PowerPreference::LowPower,
        Some(other) => {
            return Err(api_error(
                INVALID_ARGUMENT,
                format!(
                    "'powerPreference' must be 'high-performance' or 'low-power', got '{other}'"
                ),
            ));
        }
    };
    let msaa = match input.msaa {
        None => d.msaa,
        Some(1.0) => 1,
        Some(4.0) => 4,
        Some(x) => {
            return Err(api_error(
                INVALID_ARGUMENT,
                format!("'msaa' must be 1 or 4, got {x}"),
            ));
        }
    };
    let tone_mapper = match input.tone_mapper.as_deref() {
        None => d.tone_mapper,
        Some(name) => tone_mapper(name).ok_or_else(|| {
            api_error(
                INVALID_ARGUMENT,
                format!("'toneMapper' must be 'agx', 'aces' or 'linear', got '{name}'"),
            )
        })?,
    };
    let exposure_ev = match input.exposure_ev {
        None => d.exposure_ev,
        Some(x) if x.is_finite() && (x as f32).is_finite() => x as f32,
        Some(x) => {
            return Err(api_error(
                INVALID_ARGUMENT,
                format!("'exposureEv' must be finite, got {x}"),
            ));
        }
    };
    Ok(EngineOptions {
        max_instances: count("maxInstances", input.max_instances, d.max_instances)?,
        max_materials: count("maxMaterials", input.max_materials, d.max_materials)?,
        power_preference,
        msaa,
        tone_mapper,
        exposure_ev,
    })
}

/// The JS name of a tone mapper: 'agx', 'aces' or 'linear'.
pub fn tone_mapper(name: &str) -> Option<ToneMapper> {
    match name {
        "agx" => Some(ToneMapper::Agx),
        "aces" => Some(ToneMapper::Aces),
        "linear" => Some(ToneMapper::Linear),
        _ => None,
    }
}

/// JS value readers (wasm32 only), built on `js_sys::Reflect`; no serde.
/// Every reader names the key in its message and never panics on user data.
#[cfg(target_arch = "wasm32")]
// The Engine (engine.rs) is the only consumer; it lands in a later task.
#[allow(dead_code)]
pub(crate) mod js {
    use super::{
        CameraDesc, EngineOptions, EngineOptionsInput, INVALID_ARGUMENT, MaterialDesc, ModelName,
        UNKNOWN_PARAM, api_error, geom_error,
    };
    use js_sys::{Array, Float32Array, Object, Reflect};
    use raa_core::RaaError;
    use raa_geom::{GeomError, ParamValue, Params};
    use raa_math::{Mat4, Vec3};
    use raa_prims::PrimKind;
    use raa_scene::{Ambient, DirectionalLight};
    use wasm_bindgen::{JsCast, JsValue};

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

    fn require<T>(key: &str, v: Option<T>) -> Result<T, RaaError> {
        v.ok_or_else(|| invalid(format!("'{key}' is required")))
    }

    fn finite(key: &str, x: f64) -> Result<f64, RaaError> {
        if x.is_finite() {
            Ok(x)
        } else {
            Err(invalid(format!("'{key}' must be finite, got {x}")))
        }
    }

    pub fn optional_f64(obj: &JsValue, key: &str) -> Result<Option<f64>, RaaError> {
        match get(obj, key)? {
            None => Ok(None),
            Some(v) => match v.as_f64() {
                Some(x) => finite(key, x).map(Some),
                None => Err(invalid(format!("'{key}' must be a number"))),
            },
        }
    }

    pub fn required_f64(obj: &JsValue, key: &str) -> Result<f64, RaaError> {
        require(key, optional_f64(obj, key)?)
    }

    fn optional_f32(obj: &JsValue, key: &str) -> Result<Option<f32>, RaaError> {
        Ok(optional_f64(obj, key)?.map(|x| x as f32))
    }

    /// Reads exactly `N` finite numbers from an Array or a Float32Array.
    fn floats<const N: usize>(key: &str, v: &JsValue) -> Result<[f32; N], RaaError> {
        let mut out = [0.0f32; N];
        if let Some(fa) = v.dyn_ref::<Float32Array>() {
            if fa.length() as usize != N {
                return Err(invalid(format!(
                    "'{key}' must have {N} components, got {}",
                    fa.length()
                )));
            }
            fa.copy_to(&mut out);
            for x in out {
                finite(key, f64::from(x))?;
            }
            return Ok(out);
        }
        if Array::is_array(v) {
            let a: &Array = v.unchecked_ref();
            if a.length() as usize != N {
                return Err(invalid(format!(
                    "'{key}' must have {N} components, got {}",
                    a.length()
                )));
            }
            for (i, slot) in out.iter_mut().enumerate() {
                let x = a
                    .get(i as u32)
                    .as_f64()
                    .ok_or_else(|| invalid(format!("'{key}[{i}]' must be a number")))?;
                *slot = finite(key, x)? as f32;
            }
            return Ok(out);
        }
        Err(invalid(format!(
            "'{key}' must be an Array or Float32Array of {N} numbers"
        )))
    }

    pub(crate) fn optional_vec3(obj: &JsValue, key: &str) -> Result<Option<[f32; 3]>, RaaError> {
        get(obj, key)?.map(|v| floats::<3>(key, &v)).transpose()
    }

    pub fn vec3(obj: &JsValue, key: &str) -> Result<[f32; 3], RaaError> {
        require(key, optional_vec3(obj, key)?)
    }

    pub fn quat(obj: &JsValue, key: &str) -> Result<[f32; 4], RaaError> {
        let v = require(key, get(obj, key)?)?;
        floats::<4>(key, &v)
    }

    pub fn optional_string(obj: &JsValue, key: &str) -> Result<Option<String>, RaaError> {
        match get(obj, key)? {
            None => Ok(None),
            Some(v) => v
                .as_string()
                .map(Some)
                .ok_or_else(|| invalid(format!("'{key}' must be a string"))),
        }
    }

    pub fn string(obj: &JsValue, key: &str) -> Result<String, RaaError> {
        require(key, optional_string(obj, key)?)
    }

    pub fn optional_bool(obj: &JsValue, key: &str) -> Result<Option<bool>, RaaError> {
        match get(obj, key)? {
            None => Ok(None),
            Some(v) => v
                .as_bool()
                .map(Some)
                .ok_or_else(|| invalid(format!("'{key}' must be a boolean"))),
        }
    }

    pub fn bool(obj: &JsValue, key: &str) -> Result<bool, RaaError> {
        require(key, optional_bool(obj, key)?)
    }

    /// A number at `key` without the finiteness check (the validator decides).
    fn optional_number(obj: &JsValue, key: &str) -> Result<Option<f64>, RaaError> {
        match get(obj, key)? {
            None => Ok(None),
            Some(v) => v
                .as_f64()
                .map(Some)
                .ok_or_else(|| invalid(format!("'{key}' must be a number"))),
        }
    }

    /// Engine options; undefined or null gives every default.
    pub fn engine_options(v: &JsValue) -> Result<EngineOptions, RaaError> {
        if v.is_undefined() || v.is_null() {
            return Ok(EngineOptions::default());
        }
        super::engine_options(&EngineOptionsInput {
            max_instances: optional_number(v, "maxInstances")?,
            max_materials: optional_number(v, "maxMaterials")?,
            power_preference: optional_string(v, "powerPreference")?,
            msaa: optional_number(v, "msaa")?,
            tone_mapper: optional_string(v, "toneMapper")?,
            exposure_ev: optional_number(v, "exposureEv")?,
        })
    }

    /// Material description: `color` (sRGB) is required, the rest default.
    pub fn material_desc(v: &JsValue) -> Result<MaterialDesc, RaaError> {
        let mut d = MaterialDesc::new(vec3(v, "color")?);
        if let Some(r) = optional_f32(v, "roughness")? {
            d.roughness = r;
        }
        if let Some(m) = optional_f32(v, "metallic")? {
            d.metallic = m;
        }
        if let Some(e) = optional_vec3(v, "emissive")? {
            d.emissive = e;
        }
        if let Some(model) = optional_string(v, "model")? {
            d.model = ModelName::parse(&model)?;
        }
        Ok(d)
    }

    /// Camera description: `eye` and `target` required; `up` defaults to +Y.
    pub fn camera_desc(v: &JsValue) -> Result<CameraDesc, RaaError> {
        let up = optional_vec3(v, "up")?.unwrap_or([0.0, 1.0, 0.0]);
        let eye = optional_vec3(v, "position")?
            .or(optional_vec3(v, "eye")?)
            .ok_or_else(|| invalid("'eye' or 'position' is required".into()))?;
        let mut d = CameraDesc::new(eye, vec3(v, "target")?, up);
        match optional_string(v, "projection")?
            .as_deref()
            .unwrap_or("perspective")
        {
            "perspective" => {}
            "orthographic" => {
                let height = optional_f32(v, "orthoHeight")?.unwrap_or(10.0);
                if height <= 0.0 {
                    return Err(invalid("'orthoHeight' must be positive".into()));
                }
                d.ortho_height = Some(height);
            }
            _ => {
                return Err(invalid(
                    "'projection' must be perspective or orthographic".into(),
                ));
            }
        }
        if let Some(f) = optional_f32(v, "fovYDeg")? {
            d.fov_y_deg = f;
        }
        if let Some(n) = optional_f32(v, "near")? {
            d.near = n;
        }
        if let Some(f) = optional_f32(v, "far")? {
            d.far = f;
        }
        Ok(d)
    }

    /// Directional light: `direction` (normalised here, non-zero), linear
    /// `color`, `intensity` (>= 0).
    pub fn light_desc(v: &JsValue) -> Result<DirectionalLight, RaaError> {
        let direction = Vec3::from_array(vec3(v, "direction")?);
        if direction.length_squared() <= 1e-12 {
            return Err(invalid("'direction' must be non-zero".to_string()));
        }
        Ok(DirectionalLight {
            direction: direction.normalize_or_zero(),
            color: vec3(v, "color")?,
            intensity: intensity(v)?,
        })
    }

    pub fn ambient_desc(v: &JsValue) -> Result<Ambient, RaaError> {
        Ok(Ambient {
            color: vec3(v, "color")?,
            intensity: intensity(v)?,
        })
    }

    fn intensity(v: &JsValue) -> Result<f32, RaaError> {
        let i = required_f64(v, "intensity")?;
        if i < 0.0 {
            return Err(invalid(format!("'intensity' must be >= 0, got {i}")));
        }
        Ok(i as f32)
    }

    /// Reads exactly the S3 parameters of `kind`, in S3 order.
    pub fn prim_params(kind: &str, obj: &JsValue) -> Result<Params, RaaError> {
        let k = PrimKind::parse(kind)
            .ok_or_else(|| geom_error(&GeomError::UnknownKind(kind.to_string())))?;
        let names = k.param_names();
        if !obj.is_object() {
            return Err(invalid(format!("params of '{kind}' must be an object")));
        }
        let obj_ref: &Object = obj.unchecked_ref();
        for key in Object::keys(obj_ref).iter() {
            let key = key.as_string().unwrap_or_default();
            if !names.contains(&key.as_str()) {
                return Err(api_error(
                    UNKNOWN_PARAM,
                    format!("unknown param '{key}' for kind '{kind}'"),
                ));
            }
        }
        let mut params = Params::new();
        for &name in names {
            let Some(v) = get(obj, name)? else {
                return Err(geom_error(&GeomError::MissingParam {
                    kind: kind.to_string(),
                    name: name.to_string(),
                }));
            };
            let value = if matches!(name, "halfExtents" | "radii") {
                ParamValue::Vec3(floats::<3>(name, &v)?)
            } else {
                let x = v
                    .as_f64()
                    .ok_or_else(|| invalid(format!("'{name}' must be a number")))?;
                ParamValue::Scalar(finite(name, x)? as f32)
            };
            params.set(name, value);
        }
        Ok(params)
    }

    /// 16 floats, column-major.
    pub fn mat4_to_f32array(m: &Mat4) -> Float32Array {
        Float32Array::from(&m.to_cols_array()[..])
    }
}
