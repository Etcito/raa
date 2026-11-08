//! Validated controls, independent of a device or host application.
use raa_core::RaaError;

/// Deterministic GPU threshold sources.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum Dither {
    Bayer2,
    #[default]
    Bayer4,
    Bayer8,
    Bayer16,
    Ign,
    White,
    Blue,
}
impl Dither {
    pub fn name(self) -> &'static str {
        [
            "bayer2", "bayer4", "bayer8", "bayer16", "ign", "white", "blue",
        ][self as usize]
    }
    pub fn from_name(name: &str) -> Result<Self, RaaError> {
        [
            Self::Bayer2,
            Self::Bayer4,
            Self::Bayer8,
            Self::Bayer16,
            Self::Ign,
            Self::White,
            Self::Blue,
        ]
        .into_iter()
        .find(|m| m.name() == name)
        .ok_or_else(|| invalid(format!("unknown dither '{name}'")))
    }
}

/// Full-resolution cel controls. Integer palette counts are validated before upload.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CelSettings {
    pub bands: u32,
    pub hues: u32,
    pub chroma: u32,
    pub dither: Dither,
    pub dither_amount: f32,
    pub outline_width: f32,
    pub outline_sharpness: f32,
    pub interior_edges: bool,
    pub edge_retention: f32,
    pub world_space: bool,
    pub world_scale: f32,
    pub softness: f32,
    pub specular: f32,
    pub toon: bool,
    pub palette: bool,
    pub outlines: bool,
    pub dither_outlines: bool,
}
impl Default for CelSettings {
    fn default() -> Self {
        Self {
            bands: 10,
            hues: 16,
            chroma: 6,
            dither: Dither::Bayer4,
            dither_amount: 0.65,
            outline_width: 1.0,
            outline_sharpness: 0.85,
            interior_edges: true,
            edge_retention: 0.85,
            world_space: true,
            world_scale: 48.0,
            softness: 0.12,
            specular: 0.7,
            toon: true,
            palette: true,
            outlines: true,
            dither_outlines: false,
        }
    }
}
pub(crate) fn invalid(message: String) -> RaaError {
    RaaError::FrameGraph {
        code: "cel.invalid",
        message,
    }
}
impl CelSettings {
    pub fn validate(&self) -> Result<(), RaaError> {
        for (name, v, lo, hi) in [
            ("bands", self.bands, 2, 32),
            ("hues", self.hues, 4, 32),
            ("chroma", self.chroma, 2, 12),
        ] {
            if !(lo..=hi).contains(&v) {
                return Err(invalid(format!("{name} must be in [{lo}, {hi}]")));
            }
        }
        for (name, v, lo, hi) in [
            ("ditherAmount", self.dither_amount, 0.0, 1.0),
            ("outlineWidth", self.outline_width, 0.5, 6.0),
            ("outlineSharpness", self.outline_sharpness, 0.0, 1.0),
            ("edgeRetention", self.edge_retention, 0.0, 1.0),
            ("worldScale", self.world_scale, 1.0, 512.0),
            ("softness", self.softness, 0.0, 1.0),
            ("specular", self.specular, 0.0, 2.0),
        ] {
            if !v.is_finite() || !(lo..=hi).contains(&v) {
                return Err(invalid(format!(
                    "{name} must be finite and in [{lo}, {hi}]"
                )));
            }
        }
        Ok(())
    }
    /// The exact 112-byte CelPost ABI: five float vectors, image u32, grid i32.
    pub fn to_bytes(&self, size: (u32, u32)) -> [u8; 112] {
        let flag = |v: bool| u32::from(v) as f32;
        let values = [
            [1.0, self.bands as f32, 1.0, 1.0],
            [
                flag(self.palette),
                self.hues as f32,
                self.chroma as f32,
                self.dither as u32 as f32,
            ],
            [
                flag(self.toon),
                self.softness,
                self.specular,
                self.dither_amount,
            ],
            [
                if self.outlines {
                    self.outline_width
                } else {
                    0.0
                },
                self.outline_sharpness,
                flag(self.interior_edges),
                flag(self.dither_outlines),
            ],
            [
                self.edge_retention,
                flag(self.world_space),
                self.world_scale,
                0.0,
            ],
        ];
        let mut bytes = [0; 112];
        for (out, v) in bytes[..80]
            .chunks_exact_mut(4)
            .zip(values.into_iter().flatten())
        {
            out.copy_from_slice(&v.to_le_bytes());
        }
        for (out, v) in bytes[80..96]
            .chunks_exact_mut(4)
            .zip([size.0, size.1, 1, 0])
        {
            out.copy_from_slice(&v.to_le_bytes());
        }
        bytes
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ranges_and_nonfinite_are_refused() {
        CelSettings::default().validate().unwrap();
        for bands in [0, 1, 33, u32::MAX] {
            assert_eq!(
                CelSettings {
                    bands,
                    ..Default::default()
                }
                .validate()
                .unwrap_err()
                .code(),
                "cel.invalid"
            );
        }
        for softness in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
            assert!(
                CelSettings {
                    softness,
                    ..Default::default()
                }
                .validate()
                .is_err()
            );
        }
        assert!(Dither::from_name("unknown").is_err());
    }
}
