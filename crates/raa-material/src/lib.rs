//! Simple materials: a linear-colour [`Material`], its 48-byte GPU layout (S1 `MaterialGpu`)
//! and a [`MaterialLibrary`] with dense [`MaterialId`]s.

use std::fmt;

pub use raa_core::MaterialId;

/// Size in bytes of one S1 `MaterialGpu` element.
pub const MATERIAL_GPU_SIZE: usize = 48;

/// WGSL source of the S1 shading: `struct MaterialGpu`, `struct FrameUniform` (144 B),
/// `fn raa_shade(m, n, v, f) -> vec3<f32>` (linear rgb) and `fn raa_encode_srgb`
/// (piecewise sRGB OETF, applied by the forward shader when `flags.x == 1`).
/// It declares no bindings and no entry points; raa-render prepends it to its shaders.
pub const WGSL_SHADING: &str = include_str!("shading.wgsl");

/// How a material is shaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadingModel {
    /// Unlit: `base_color + emissive`.
    Flat,
    /// One directional light, Lambert diffuse plus GGX specular, ambient term.
    PbrLite,
}

impl ShadingModel {
    /// The `params.z` value of `MaterialGpu`: 0 Flat, 1 PbrLite.
    pub fn gpu_value(self) -> f32 {
        match self {
            ShadingModel::Flat => 0.0,
            ShadingModel::PbrLite => 1.0,
        }
    }
}

/// The one error type of raa-material.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterialError {
    /// Stable code: `material.invalid` or `material.unknown_id`.
    pub code: &'static str,
    /// Human-readable message.
    pub message: String,
}

impl MaterialError {
    /// Stable machine-readable code of the form `<area>.<snake_case>`.
    pub fn code(&self) -> &'static str {
        self.code
    }

    fn invalid(message: String) -> Self {
        MaterialError {
            code: "material.invalid",
            message,
        }
    }
}

impl fmt::Display for MaterialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for MaterialError {}

/// Exact piecewise sRGB EOTF: sRGB-encoded `c` in [0, 1] to linear.
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// A simple material. All colours are linear RGB.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    /// Linear base colour, each channel >= 0.
    pub base_color: [f32; 3],
    /// Roughness in [0, 1]; clamped to [0.04, 1] on the GPU.
    pub roughness: f32,
    /// Metallic in [0, 1].
    pub metallic: f32,
    /// Linear emissive colour, each channel >= 0.
    pub emissive: [f32; 3],
    /// Shading model.
    pub model: ShadingModel,
}

impl Default for Material {
    /// Linear grey 0.5, roughness 0.6, metallic 0, PbrLite.
    fn default() -> Self {
        Material::pbr([0.5; 3], 0.6, 0.0)
    }
}

impl Material {
    /// An unlit material of the given linear colour.
    pub fn flat(color_linear: [f32; 3]) -> Material {
        Material {
            base_color: color_linear,
            roughness: 1.0,
            metallic: 0.0,
            emissive: [0.0; 3],
            model: ShadingModel::Flat,
        }
    }

    /// A PbrLite material of the given linear colour, without emission.
    pub fn pbr(color_linear: [f32; 3], roughness: f32, metallic: f32) -> Material {
        Material {
            base_color: color_linear,
            roughness,
            metallic,
            emissive: [0.0; 3],
            model: ShadingModel::PbrLite,
        }
    }

    /// A PbrLite material from an sRGB colour (converted with [`srgb_to_linear`]) and a
    /// linear emissive colour.
    pub fn from_srgb(
        color_srgb: [f32; 3],
        roughness: f32,
        metallic: f32,
        emissive_linear: [f32; 3],
    ) -> Material {
        Material {
            base_color: color_srgb.map(srgb_to_linear),
            roughness,
            metallic,
            emissive: emissive_linear,
            model: ShadingModel::PbrLite,
        }
    }

    /// Checks that every value is finite, colour channels are >= 0 and roughness and
    /// metallic are in [0, 1]. Refusals have code `material.invalid` and name the field.
    pub fn validate(&self) -> Result<(), MaterialError> {
        for (field, color) in [("base_color", self.base_color), ("emissive", self.emissive)] {
            for (i, c) in color.iter().enumerate() {
                if !c.is_finite() {
                    return Err(MaterialError::invalid(format!(
                        "{field}[{i}] must be finite (got {c})"
                    )));
                }
                if *c < 0.0 {
                    return Err(MaterialError::invalid(format!(
                        "{field}[{i}] must be >= 0 (got {c})"
                    )));
                }
            }
        }
        for (field, v) in [("roughness", self.roughness), ("metallic", self.metallic)] {
            if !v.is_finite() {
                return Err(MaterialError::invalid(format!(
                    "{field} must be finite (got {v})"
                )));
            }
            if !(0.0..=1.0).contains(&v) {
                return Err(MaterialError::invalid(format!(
                    "{field} must be in [0, 1] (got {v})"
                )));
            }
        }
        Ok(())
    }

    /// Writes the 48-byte S1 `MaterialGpu` little-endian into `out[..48]`:
    /// `base_color` (a = 1) @0, `emissive` (w = 0) @16, `params` @32 =
    /// (roughness clamped to [0.04, 1], metallic, model 0 Flat / 1 PbrLite, 0).
    ///
    /// Invariant: `out.len() >= MATERIAL_GPU_SIZE`. A shorter slice is a caller bug: it
    /// trips a debug assertion, and in release nothing is written (no panic).
    pub fn write_gpu(&self, out: &mut [u8]) {
        debug_assert!(
            out.len() >= MATERIAL_GPU_SIZE,
            "write_gpu needs {MATERIAL_GPU_SIZE} bytes, got {}",
            out.len()
        );
        let Some(out) = out.get_mut(..MATERIAL_GPU_SIZE) else {
            return;
        };
        let [r, g, b] = self.base_color;
        let [er, eg, eb] = self.emissive;
        let words: [f32; 12] = [
            r,
            g,
            b,
            1.0,
            er,
            eg,
            eb,
            0.0,
            self.roughness.clamp(0.04, 1.0),
            self.metallic,
            self.model.gpu_value(),
            0.0,
        ];
        for (dst, w) in out.chunks_exact_mut(4).zip(words) {
            dst.copy_from_slice(&w.to_le_bytes());
        }
    }
}

/// Materials addressed by dense [`MaterialId`]s from 0, with a dirty flag for GPU upload.
#[derive(Debug, Clone, Default)]
pub struct MaterialLibrary {
    materials: Vec<Material>,
    dirty: bool,
}

impl MaterialLibrary {
    /// An empty, clean library.
    pub fn new() -> Self {
        Self::default()
    }

    /// Validates `m` and appends it; ids are dense from 0. Marks the library dirty.
    pub fn add(&mut self, m: Material) -> Result<MaterialId, MaterialError> {
        m.validate()?;
        let id = u32::try_from(self.materials.len()).map_err(|_| {
            MaterialError::invalid("material library is full (u32 ids)".to_string())
        })?;
        self.materials.push(m);
        self.dirty = true;
        Ok(MaterialId(id))
    }

    /// Validates `m` and replaces material `id`. Marks the library dirty.
    /// An id never returned by [`add`](Self::add) gives `material.unknown_id`.
    pub fn set(&mut self, id: MaterialId, m: Material) -> Result<(), MaterialError> {
        let len = self.materials.len();
        let slot = self
            .materials
            .get_mut(id.0 as usize)
            .ok_or_else(|| MaterialError {
                code: "material.unknown_id",
                message: format!("material id {} is unknown (library has {len})", id.0),
            })?;
        m.validate()?;
        *slot = m;
        self.dirty = true;
        Ok(())
    }

    /// The material `id`, if it exists.
    pub fn get(&self, id: MaterialId) -> Option<&Material> {
        self.materials.get(id.0 as usize)
    }

    /// Number of materials.
    pub fn len(&self) -> usize {
        self.materials.len()
    }

    /// True when the library holds no material.
    pub fn is_empty(&self) -> bool {
        self.materials.is_empty()
    }

    /// True when a material was added or set since the last [`write_gpu_all`](Self::write_gpu_all).
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Clears `out`, appends 48 bytes per material in id order and clears the dirty flag.
    /// Reuses the capacity of `out`.
    pub fn write_gpu_all(&mut self, out: &mut Vec<u8>) {
        out.clear();
        out.resize(self.materials.len() * MATERIAL_GPU_SIZE, 0);
        for (m, dst) in self
            .materials
            .iter()
            .zip(out.chunks_exact_mut(MATERIAL_GPU_SIZE))
        {
            m.write_gpu(dst);
        }
        self.dirty = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_at(bytes: &[u8], offset: usize) -> f32 {
        f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    }

    #[test]
    fn srgb_to_linear_matches_the_eotf() {
        assert_eq!(srgb_to_linear(0.0), 0.0);
        assert!((srgb_to_linear(1.0) - 1.0).abs() <= 1e-6);
        assert!((srgb_to_linear(0.5) - 0.21404).abs() <= 1e-5);
        assert!((srgb_to_linear(0.04045) - 0.04045 / 12.92).abs() <= 1e-9);
    }

    #[test]
    fn default_is_linear_grey_pbr_lite() {
        let m = Material::default();
        assert_eq!(m.base_color, [0.5; 3]);
        assert_eq!(m.roughness, 0.6);
        assert_eq!(m.metallic, 0.0);
        assert_eq!(m.model, ShadingModel::PbrLite);
        assert!(m.validate().is_ok());
    }

    #[test]
    fn from_srgb_converts_colour_only() {
        let m = Material::from_srgb([0.5, 0.0, 1.0], 0.3, 0.2, [0.5, 0.0, 0.0]);
        assert!((m.base_color[0] - 0.21404).abs() <= 1e-5);
        assert_eq!(m.base_color[1], 0.0);
        assert!((m.base_color[2] - 1.0).abs() <= 1e-6);
        assert_eq!(m.emissive, [0.5, 0.0, 0.0]);
        assert_eq!(m.model, ShadingModel::PbrLite);
    }

    fn assert_invalid(m: Material, field: &str) {
        let err = m.validate().unwrap_err();
        assert_eq!(err.code(), "material.invalid");
        assert!(err.message.contains(field), "{}", err.message);
    }

    #[test]
    fn validate_refuses_each_bad_field() {
        let base = Material::default();
        assert_invalid(
            Material {
                base_color: [-0.1, 0.0, 0.0],
                ..base
            },
            "base_color",
        );
        assert_invalid(
            Material {
                base_color: [f32::NAN, 0.0, 0.0],
                ..base
            },
            "base_color",
        );
        assert_invalid(
            Material {
                emissive: [0.0, -1.0, 0.0],
                ..base
            },
            "emissive",
        );
        assert_invalid(
            Material {
                emissive: [0.0, 0.0, f32::INFINITY],
                ..base
            },
            "emissive",
        );
        assert_invalid(
            Material {
                roughness: 1.5,
                ..base
            },
            "roughness",
        );
        assert_invalid(
            Material {
                roughness: f32::NAN,
                ..base
            },
            "roughness",
        );
        assert_invalid(
            Material {
                metallic: -0.1,
                ..base
            },
            "metallic",
        );
        assert_invalid(
            Material {
                metallic: f32::INFINITY,
                ..base
            },
            "metallic",
        );
    }

    #[test]
    fn write_gpu_lays_out_s1_material_gpu() {
        let m = Material {
            base_color: [0.1, 0.2, 0.3],
            roughness: 0.5,
            metallic: 0.25,
            emissive: [0.4, 0.5, 0.6],
            model: ShadingModel::PbrLite,
        };
        let mut out = [0xAAu8; 52];
        m.write_gpu(&mut out);
        let at = |o| f32_at(&out, o);
        assert_eq!([at(0), at(4), at(8), at(12)], [0.1, 0.2, 0.3, 1.0]);
        assert_eq!([at(16), at(20), at(24), at(28)], [0.4, 0.5, 0.6, 0.0]);
        assert_eq!([at(32), at(36), at(40), at(44)], [0.5, 0.25, 1.0, 0.0]);
        assert_eq!(&out[48..], &[0xAA; 4]);
    }

    #[test]
    fn write_gpu_clamps_roughness_and_encodes_flat() {
        let m = Material {
            roughness: 0.0,
            ..Material::flat([1.0, 0.0, 0.0])
        };
        let mut out = [0u8; 48];
        m.write_gpu(&mut out);
        assert_eq!(f32_at(&out, 32), 0.04);
        assert_eq!(f32_at(&out, 40), 0.0);
    }

    #[test]
    fn library_ids_are_dense_from_zero() {
        let mut lib = MaterialLibrary::new();
        assert!(lib.is_empty());
        for i in 0..3 {
            assert_eq!(lib.add(Material::default()).unwrap(), MaterialId(i));
        }
        assert_eq!(lib.len(), 3);
        assert!(lib.get(MaterialId(2)).is_some());
        assert!(lib.get(MaterialId(3)).is_none());
    }

    #[test]
    fn library_add_validates() {
        let mut lib = MaterialLibrary::new();
        let bad = Material {
            metallic: 2.0,
            ..Material::default()
        };
        assert_eq!(lib.add(bad).unwrap_err().code(), "material.invalid");
        assert_eq!(lib.len(), 0);
        assert!(!lib.is_dirty());
    }

    #[test]
    fn library_dirty_flag_lifecycle() {
        let mut lib = MaterialLibrary::new();
        assert!(!lib.is_dirty());
        let id = lib.add(Material::default()).unwrap();
        assert!(lib.is_dirty());
        let mut out = vec![1u8; 7];
        lib.write_gpu_all(&mut out);
        assert!(!lib.is_dirty());
        assert_eq!(out.len(), 48);
        lib.set(id, Material::flat([0.2; 3])).unwrap();
        assert!(lib.is_dirty());
        lib.add(Material::pbr([0.9, 0.1, 0.1], 0.0, 1.0)).unwrap();
        lib.write_gpu_all(&mut out);
        assert!(!lib.is_dirty());
        assert_eq!(out.len(), 96);
        assert_eq!(f32_at(&out, 40), 0.0);
        assert_eq!(f32_at(&out, 48), 0.9);
        assert_eq!(f32_at(&out, 48 + 32), 0.04);
    }

    #[test]
    fn library_set_unknown_id_is_refused() {
        let mut lib = MaterialLibrary::new();
        lib.add(Material::default()).unwrap();
        let mut out = Vec::new();
        lib.write_gpu_all(&mut out);
        let err = lib.set(MaterialId(1), Material::default()).unwrap_err();
        assert_eq!(err.code(), "material.unknown_id");
        assert!(!lib.is_dirty());
    }
}
