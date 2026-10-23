//! [`SurfaceMaterial`], its 256-byte S10 `SurfaceMaterialGpu` layout and its [`DrawClass`].

use raa_core::RaaError;
use raa_material::{Material, ShadingModel};
use raa_shader::MaterialFnId;

/// Size in bytes of one S10 `SurfaceMaterialGpu` element.
pub const SURFACE_MATERIAL_GPU_SIZE: usize = 256;

/// Material flag: blended, drawn by the Transparent class handler.
pub const FLAG_TRANSPARENT: u32 = 1;
/// Material flag: transmission (refraction), drawn by the Transparent class handler.
pub const FLAG_TRANSMISSIVE: u32 = 2;
/// Material flag: fur shells, drawn by the Fur class handler over an opaque base.
pub const FLAG_FUR: u32 = 4;
/// Material flag: no back-face culling, the normal flips on back faces.
pub const FLAG_DOUBLE_SIDED: u32 = 8;
/// Material flag: casts no shadow.
pub const FLAG_NO_SHADOW: u32 = 16;
/// Material flag: the material function defines `material_vertex`.
pub const FLAG_VERTEX_FN: u32 = 32;
/// Material flag: shaded as `base + emissive`.
pub const FLAG_UNLIT: u32 = 64;

/// The value written for an infinite `attenuation_distance`.
const INFINITE_DISTANCE: f32 = 1e30;

/// How a surface is shaded.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SurfaceModel {
    /// Unlit: `base_color + emissive` (header model 0).
    Flat,
    /// The built-in `raa.standard` function (header model 1).
    Standard,
    /// A registered material function (header model 2).
    Function(MaterialFnId),
}

impl SurfaceModel {
    /// Header `x` (model) and `y` (material function id) of `SurfaceMaterialGpu`.
    fn header(self) -> (u32, u32) {
        match self {
            SurfaceModel::Flat => (0, MaterialFnId::STANDARD.0),
            SurfaceModel::Standard => (1, MaterialFnId::STANDARD.0),
            SurfaceModel::Function(id) => (2, id.0),
        }
    }
}

/// The class of draws a material belongs to.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DrawClass {
    /// Prepass and forward.
    Opaque,
    /// TRANSPARENT or TRANSMISSIVE materials.
    Transparent,
    /// FUR materials, also drawn as an opaque base.
    Fur,
}

/// A lit surface material. Colours are linear RGB.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceMaterial {
    pub model: SurfaceModel,
    /// Linear base colour, each channel >= 0.
    pub base_color: [f32; 3],
    /// Opacity in [0, 1].
    pub opacity: f32,
    /// Linear emissive colour x strength, each channel >= 0.
    pub emissive: [f32; 3],
    /// Roughness in [0, 1]; clamped to [0.03, 1] on the GPU.
    pub roughness: f32,
    /// Metallic in [0, 1].
    pub metallic: f32,
    /// Clear coat weight in [0, 1].
    pub clearcoat: f32,
    /// Clear coat roughness in [0, 1].
    pub clearcoat_roughness: f32,
    /// Linear sheen colour, each channel >= 0.
    pub sheen_color: [f32; 3],
    /// Sheen roughness in [0, 1].
    pub sheen_roughness: f32,
    /// Transmission in [0, 1].
    pub transmission: f32,
    /// Index of refraction in [1, 3] (default 1.5).
    pub ior: f32,
    /// Thickness in metres, >= 0.
    pub thickness: f32,
    /// Linear attenuation colour, each channel >= 0.
    pub attenuation_color: [f32; 3],
    /// Attenuation distance in metres, > 0; `f32::INFINITY` (default) is written as 1e30.
    pub attenuation_distance: f32,
    /// `FLAG_*` bits.
    pub flags: u32,
    /// Material function parameters (`SurfaceMaterialGpu.p`), finite.
    pub params: [[f32; 4]; 8],
    /// Fur: length m, density strands per cm2, droop, shell count; finite.
    pub fur: [f32; 4],
}

impl Default for SurfaceMaterial {
    /// Standard, linear grey 0.5, roughness 0.6, opaque, ior 1.5, no attenuation.
    fn default() -> Self {
        SurfaceMaterial {
            model: SurfaceModel::Standard,
            base_color: [0.5; 3],
            opacity: 1.0,
            emissive: [0.0; 3],
            roughness: 0.6,
            metallic: 0.0,
            clearcoat: 0.0,
            clearcoat_roughness: 0.0,
            sheen_color: [0.0; 3],
            sheen_roughness: 0.0,
            transmission: 0.0,
            ior: 1.5,
            thickness: 0.0,
            attenuation_color: [1.0; 3],
            attenuation_distance: f32::INFINITY,
            flags: 0,
            params: [[0.0; 4]; 8],
            fur: [0.0; 4],
        }
    }
}

fn invalid(message: String) -> RaaError {
    RaaError::FrameGraph {
        code: "material.invalid",
        message,
    }
}

fn check_finite(field: &str, v: f32) -> Result<(), RaaError> {
    if v.is_finite() {
        Ok(())
    } else {
        Err(invalid(format!("{field} must be finite (got {v})")))
    }
}

fn check_unit(field: &str, v: f32) -> Result<(), RaaError> {
    check_finite(field, v)?;
    if (0.0..=1.0).contains(&v) {
        Ok(())
    } else {
        Err(invalid(format!("{field} must be in [0, 1] (got {v})")))
    }
}

fn check_color(field: &str, c: [f32; 3]) -> Result<(), RaaError> {
    for (i, v) in c.iter().enumerate() {
        check_finite(&format!("{field}[{i}]"), *v)?;
        if *v < 0.0 {
            return Err(invalid(format!("{field}[{i}] must be >= 0 (got {v})")));
        }
    }
    Ok(())
}

impl SurfaceMaterial {
    /// Maps a v0 material: Flat -> model flat, PbrLite -> standard; colour, roughness,
    /// metallic and emissive copied, every other field at its default.
    pub fn from_v0(m: &Material) -> Self {
        SurfaceMaterial {
            model: match m.model {
                ShadingModel::Flat => SurfaceModel::Flat,
                ShadingModel::PbrLite => SurfaceModel::Standard,
            },
            base_color: m.base_color,
            emissive: m.emissive,
            roughness: m.roughness,
            metallic: m.metallic,
            ..SurfaceMaterial::default()
        }
    }

    /// Checks every value: finite, colours >= 0, opacity, roughness, metallic, clearcoat,
    /// clearcoat roughness, sheen roughness and transmission in [0, 1], ior in [1, 3],
    /// thickness >= 0, attenuation distance > 0 (infinity allowed), params and fur finite.
    /// Refusals have code `material.invalid` and name the field.
    pub fn validate(&self) -> Result<(), RaaError> {
        check_color("base_color", self.base_color)?;
        check_color("emissive", self.emissive)?;
        check_color("sheen_color", self.sheen_color)?;
        check_color("attenuation_color", self.attenuation_color)?;
        for (field, v) in [
            ("opacity", self.opacity),
            ("roughness", self.roughness),
            ("metallic", self.metallic),
            ("clearcoat", self.clearcoat),
            ("clearcoat_roughness", self.clearcoat_roughness),
            ("sheen_roughness", self.sheen_roughness),
            ("transmission", self.transmission),
        ] {
            check_unit(field, v)?;
        }
        check_finite("ior", self.ior)?;
        if !(1.0..=3.0).contains(&self.ior) {
            return Err(invalid(format!("ior must be in [1, 3] (got {})", self.ior)));
        }
        check_finite("thickness", self.thickness)?;
        if self.thickness < 0.0 {
            return Err(invalid(format!(
                "thickness must be >= 0 (got {})",
                self.thickness
            )));
        }
        let d = self.attenuation_distance;
        if d.is_nan() || d <= 0.0 {
            return Err(invalid(format!(
                "attenuation_distance must be > 0 or infinite (got {d})"
            )));
        }
        for (i, p) in self.params.iter().enumerate() {
            for (j, v) in p.iter().enumerate() {
                check_finite(&format!("params[{i}][{j}]"), *v)?;
            }
        }
        for (i, v) in self.fur.iter().enumerate() {
            check_finite(&format!("fur[{i}]"), *v)?;
        }
        Ok(())
    }

    /// Appends the 256-byte S10 `SurfaceMaterialGpu` little-endian to `out`: base (a =
    /// opacity) @0, emissive @16, surface @32 (roughness clamped to [0.03, 1], metallic,
    /// clearcoat, clearcoat roughness), sheen @48, transmission @64 (transmission, ior,
    /// thickness, 0), attenuation @80 (rgb, distance, infinity as 1e30), header @96 (model,
    /// fn id, flags, 0), p @112, fur @240.
    pub fn write_gpu(&self, out: &mut Vec<u8>) {
        let f = |out: &mut Vec<u8>, v: [f32; 4]| {
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        };
        let rgb = |c: [f32; 3], w: f32| [c[0], c[1], c[2], w];
        let distance = if self.attenuation_distance == f32::INFINITY {
            INFINITE_DISTANCE
        } else {
            self.attenuation_distance
        };
        f(out, rgb(self.base_color, self.opacity));
        f(out, rgb(self.emissive, 0.0));
        f(
            out,
            [
                self.roughness.clamp(0.03, 1.0),
                self.metallic,
                self.clearcoat,
                self.clearcoat_roughness,
            ],
        );
        f(out, rgb(self.sheen_color, self.sheen_roughness));
        f(out, [self.transmission, self.ior, self.thickness, 0.0]);
        f(out, rgb(self.attenuation_color, distance));
        let (model, fn_id) = self.model.header();
        for x in [model, fn_id, self.flags, 0] {
            out.extend_from_slice(&x.to_le_bytes());
        }
        for p in self.params {
            f(out, p);
        }
        f(out, self.fur);
    }

    /// TRANSPARENT or TRANSMISSIVE -> Transparent (wins over FUR); FUR -> Fur; else Opaque.
    pub fn class(&self) -> DrawClass {
        if self.flags & (FLAG_TRANSPARENT | FLAG_TRANSMISSIVE) != 0 {
            DrawClass::Transparent
        } else if self.flags & FLAG_FUR != 0 {
            DrawClass::Fur
        } else {
            DrawClass::Opaque
        }
    }
}
