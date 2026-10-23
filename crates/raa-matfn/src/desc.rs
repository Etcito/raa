//! [`SurfaceDesc`]: the JS surface material description and its mapping into a
//! [`SurfaceMaterial`].

use crate::{BUILTINS, BuiltinFn, DrawClass, MaterialFnRegistry, ParamKind};
use raa_core::RaaError;
use raa_lit::{
    FLAG_DOUBLE_SIDED, FLAG_FUR, FLAG_NO_SHADOW, FLAG_TRANSMISSIVE, FLAG_TRANSPARENT, LitRenderer,
    SurfaceMaterial, SurfaceModel,
};
use raa_shader::MaterialFnId;

/// Shell count used when a fur field is set without `furShells` (FUR value).
pub const FUR_SHELLS_DEFAULT: f32 = 32.0;
/// Strand density (per cm2) used when a fur field is set without `furDensity`.
pub const FUR_DENSITY_DEFAULT: f32 = 2000.0;
/// Highest accepted `furShells`.
pub const FUR_SHELLS_MAX: f32 = 64.0;

/// How a surface is composited.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Blend {
    /// Opaque (prepass and forward).
    #[default]
    Opaque,
    /// Blended: sets `FLAG_TRANSPARENT`.
    Transparent,
    /// Refractive: sets `FLAG_TRANSMISSIVE`.
    Transmissive,
}

/// Optional overrides of the standard surface fields; `None` keeps the mapped value.
/// `color`, `sheen_color` and `attenuation_color` are sRGB [0, 1]; `emissive` is linear.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StandardOverrides {
    pub color: Option<[f32; 3]>,
    pub roughness: Option<f32>,
    pub metallic: Option<f32>,
    pub emissive: Option<[f32; 3]>,
    pub opacity: Option<f32>,
    pub clearcoat: Option<f32>,
    pub clearcoat_roughness: Option<f32>,
    pub sheen_color: Option<[f32; 3]>,
    pub sheen_roughness: Option<f32>,
    pub transmission: Option<f32>,
    pub ior: Option<f32>,
    pub thickness: Option<f32>,
    pub attenuation_color: Option<[f32; 3]>,
    pub attenuation_distance: Option<f32>,
    /// Shell extrusion length in metres, >= 0.
    pub fur_length: Option<f32>,
    /// Strands per cm2, > 0.
    pub fur_density: Option<f32>,
    /// Gravity droop in [0, 1].
    pub fur_droop: Option<f32>,
    /// Shell count, an integer in [1, 64].
    pub fur_shells: Option<f32>,
}

impl StandardOverrides {
    fn has_fur(&self) -> bool {
        self.fur_length.is_some()
            || self.fur_density.is_some()
            || self.fur_droop.is_some()
            || self.fur_shells.is_some()
    }
}

/// A surface material description, as JS sends it (`SurfaceMaterialDesc`).
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceDesc {
    /// Material function name (default `raa.standard`).
    pub function: String,
    /// Function parameters by name, in the unit JS sends (colours sRGB [0, 1]).
    pub params: Vec<(String, Vec<f32>)>,
    pub blend: Blend,
    pub double_sided: bool,
    /// Default true.
    pub cast_shadow: bool,
    pub overrides: StandardOverrides,
}

impl Default for SurfaceDesc {
    fn default() -> Self {
        SurfaceDesc {
            function: "raa.standard".to_string(),
            params: Vec::new(),
            blend: Blend::Opaque,
            double_sided: false,
            cast_shadow: true,
            overrides: StandardOverrides::default(),
        }
    }
}

fn refuse(code: &'static str, message: String) -> RaaError {
    RaaError::FrameGraph { code, message }
}

/// Exact piecewise sRGB EOTF, the conversion of v0 materials (`raa_material::srgb_to_linear`).
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Refuses a non-finite colour, then converts it from sRGB to linear.
fn srgb3(field: &str, c: [f32; 3]) -> Result<[f32; 3], RaaError> {
    if let Some(v) = c.iter().find(|v| !v.is_finite()) {
        return Err(refuse(
            "material.invalid",
            format!("{field} must be finite (got {v})"),
        ));
    }
    Ok(c.map(srgb_to_linear))
}

/// Writes `values` (already in record units) into `params[slot.0][slot.1..]`.
fn write_slot(params: &mut [[f32; 4]; 8], slot: (usize, usize), values: &[f32]) {
    if let Some(row) = params.get_mut(slot.0) {
        for (dst, v) in row.iter_mut().skip(slot.1).zip(values) {
            *dst = *v;
        }
    }
}

/// Converts one built-in parameter value to record units (colours to linear).
fn record_value(kind: ParamKind, values: &[f32]) -> [f32; 4] {
    let mut out = [0.0; 4];
    for (dst, v) in out.iter_mut().zip(values) {
        *dst = if kind == ParamKind::Color {
            srgb_to_linear(*v)
        } else {
            *v
        };
    }
    out
}

fn check_param_finite(fn_name: &str, name: &str, values: &[f32]) -> Result<(), RaaError> {
    match values.iter().find(|v| !v.is_finite()) {
        Some(v) => Err(refuse(
            "material.invalid",
            format!("param `{name}` of `{fn_name}` must be finite (got {v})"),
        )),
        None => Ok(()),
    }
}

fn apply_builtin_params(
    m: &mut SurfaceMaterial,
    row: &BuiltinFn,
    params: &[(String, Vec<f32>)],
) -> Result<(), RaaError> {
    for spec in row.params {
        let n = spec.kind.components();
        let v = record_value(spec.kind, spec.default.get(..n).unwrap_or(&[]));
        write_slot(&mut m.params, spec.slot, v.get(..n).unwrap_or(&[]));
    }
    for (name, values) in params {
        let Some(spec) = row.params.iter().find(|s| s.name == name) else {
            return Err(refuse(
                "material.unknown_param",
                format!("`{}` has no parameter `{name}`", row.name),
            ));
        };
        let n = spec.kind.components();
        if values.len() != n {
            return Err(refuse(
                "material.param_type",
                format!(
                    "param `{name}` of `{}` takes {n} number(s) (got {})",
                    row.name,
                    values.len()
                ),
            ));
        }
        check_param_finite(row.name, name, values)?;
        let v = record_value(spec.kind, values);
        write_slot(&mut m.params, spec.slot, v.get(..n).unwrap_or(&[]));
    }
    Ok(())
}

/// User functions take positional `p0`..`p7`, each 1..4 numbers written from `.x`.
fn apply_user_params(
    m: &mut SurfaceMaterial,
    fn_name: &str,
    params: &[(String, Vec<f32>)],
) -> Result<(), RaaError> {
    for (name, values) in params {
        let index = name
            .strip_prefix('p')
            .filter(|d| d.len() == 1)
            .and_then(|d| d.parse::<usize>().ok())
            .filter(|i| *i < m.params.len());
        let Some(index) = index else {
            return Err(refuse(
                "material.unknown_param",
                format!("user function `{fn_name}` accepts only p0..p7 (got `{name}`)"),
            ));
        };
        if values.is_empty() || values.len() > 4 {
            return Err(refuse(
                "material.param_type",
                format!(
                    "param `{name}` of `{fn_name}` takes 1 to 4 numbers (got {})",
                    values.len()
                ),
            ));
        }
        check_param_finite(fn_name, name, values)?;
        write_slot(&mut m.params, (index, 0), values);
    }
    Ok(())
}

fn check_range(field: &str, v: f32, ok: bool, range: &str) -> Result<(), RaaError> {
    if v.is_finite() && ok {
        Ok(())
    } else {
        Err(refuse(
            "material.invalid",
            format!("{field} must be {range} (got {v})"),
        ))
    }
}

fn apply_fur(m: &mut SurfaceMaterial, o: &StandardOverrides) -> Result<(), RaaError> {
    if !o.has_fur() {
        return Ok(());
    }
    let length = o.fur_length.unwrap_or(0.0);
    check_range("furLength", length, length >= 0.0, ">= 0")?;
    let density = o.fur_density.unwrap_or(FUR_DENSITY_DEFAULT);
    check_range("furDensity", density, density > 0.0, "> 0")?;
    let droop = o.fur_droop.unwrap_or(0.0);
    check_range("furDroop", droop, (0.0..=1.0).contains(&droop), "in [0, 1]")?;
    let shells = o.fur_shells.unwrap_or(FUR_SHELLS_DEFAULT);
    check_range(
        "furShells",
        shells,
        shells.fract() == 0.0 && (1.0..=FUR_SHELLS_MAX).contains(&shells),
        "an integer in [1, 64]",
    )?;
    m.fur = [length, density, droop, shells];
    m.flags |= FLAG_FUR;
    Ok(())
}

fn apply_overrides(m: &mut SurfaceMaterial, o: &StandardOverrides) -> Result<(), RaaError> {
    if let Some(c) = o.color {
        m.base_color = srgb3("color", c)?;
    }
    if let Some(c) = o.sheen_color {
        m.sheen_color = srgb3("sheenColor", c)?;
    }
    if let Some(c) = o.attenuation_color {
        m.attenuation_color = srgb3("attenuationColor", c)?;
    }
    if let Some(c) = o.emissive {
        m.emissive = c;
    }
    for (dst, v) in [
        (&mut m.roughness, o.roughness),
        (&mut m.metallic, o.metallic),
        (&mut m.opacity, o.opacity),
        (&mut m.clearcoat, o.clearcoat),
        (&mut m.clearcoat_roughness, o.clearcoat_roughness),
        (&mut m.sheen_roughness, o.sheen_roughness),
        (&mut m.transmission, o.transmission),
        (&mut m.ior, o.ior),
        (&mut m.thickness, o.thickness),
        (&mut m.attenuation_distance, o.attenuation_distance),
    ] {
        if let Some(v) = v {
            *dst = v;
        }
    }
    apply_fur(m, o)
}

/// Maps `desc` into a [`SurfaceMaterial`] with the functions of `reg`; `class_handled`
/// says whether a draw class has a handler (see [`LitRenderer::class_handled`]).
///
/// Order: function lookup (`material.unknown_fn`), function params (`material.unknown_param`,
/// `material.param_type`, non-finite `material.invalid`), overrides (`material.invalid`),
/// blend and flags, [`SurfaceMaterial::validate`] (`material.invalid`), then
/// `material.blend_unavailable` when the material's class has no handler.
#[doc(hidden)]
pub fn map_desc(
    desc: &SurfaceDesc,
    reg: &MaterialFnRegistry,
    class_handled: impl Fn(DrawClass) -> bool,
) -> Result<SurfaceMaterial, RaaError> {
    let Some(id) = reg.id_of(&desc.function) else {
        return Err(refuse(
            "material.unknown_fn",
            format!(
                "function `{}` is not a registered material function",
                desc.function
            ),
        ));
    };
    let builtin = reg.get(id).is_none_or(|e| e.builtin);
    let mut m = SurfaceMaterial::default();
    if id == MaterialFnId::STANDARD {
        m.model = SurfaceModel::Standard;
    } else {
        m.model = SurfaceModel::Function(id);
    }
    match BUILTINS.iter().find(|b| b.name == desc.function) {
        Some(row) => {
            m.flags = row.flags;
            apply_builtin_params(&mut m, row, &desc.params)?;
        }
        None if builtin => {
            if let Some((name, _)) = desc.params.first() {
                return Err(refuse(
                    "material.unknown_param",
                    format!("`{}` has no parameter `{name}`", desc.function),
                ));
            }
        }
        None => apply_user_params(&mut m, &desc.function, &desc.params)?,
    }
    apply_overrides(&mut m, &desc.overrides)?;
    match desc.blend {
        Blend::Opaque => {}
        Blend::Transparent => m.flags |= FLAG_TRANSPARENT,
        Blend::Transmissive => m.flags |= FLAG_TRANSMISSIVE,
    }
    if desc.double_sided {
        m.flags |= FLAG_DOUBLE_SIDED;
    }
    if !desc.cast_shadow {
        m.flags |= FLAG_NO_SHADOW;
    }
    if desc.blend == Blend::Opaque && m.opacity < 1.0 {
        return Err(refuse(
            "material.invalid",
            format!("opacity {} < 1 needs blend transparent", m.opacity),
        ));
    }
    if desc.blend != Blend::Transmissive && m.transmission > 0.0 {
        return Err(refuse(
            "material.invalid",
            format!(
                "transmission {} > 0 needs blend transmissive",
                m.transmission
            ),
        ));
    }
    m.validate()?;
    let class = m.class();
    if !class_handled(class) {
        return Err(refuse(
            "material.blend_unavailable",
            format!(
                "blend {:?} (draw class {class:?}) has no handler installed",
                desc.blend
            ),
        ));
    }
    Ok(m)
}

/// Maps `desc` with the functions and class handlers of `lit` (see [`map_desc`]).
pub fn to_surface_material(
    desc: &SurfaceDesc,
    lit: &LitRenderer,
) -> Result<SurfaceMaterial, RaaError> {
    let reg = lit.fns();
    map_desc(desc, &reg, |c| lit.class_handled(c))
}
