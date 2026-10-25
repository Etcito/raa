//! Height fog description, presets, validation (`post.invalid`) and `EnvGpu` packing.

use raa_core::RaaError;
use raa_pipeline::EnvGpu;

/// Exponential height fog plus aerial perspective. Density follows
/// `density * exp(-height_falloff * (y - base_height_m))` from `start_m` along the view ray.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FogDesc {
    /// Fog density at the base height, in `[0, 1]` (0 = fog off).
    pub density: f32,
    /// Height falloff in `(0, 10]` per metre; larger values confine the fog nearer the base.
    pub height_falloff: f32,
    /// Height in metres where the fog has full `density` (finite).
    pub base_height_m: f32,
    /// Distance in metres along the view ray where the fog starts (>= 0).
    pub start_m: f32,
    /// Linear rgb fog colour (each component >= 0).
    pub color: [f32; 3],
    /// Sun inscatter strength in `[0, 4]`.
    pub inscatter: f32,
    /// Aerial perspective strength in `[0, 2]` (0 = off; above 1 the aerial fade saturates closer).
    pub aerial_strength: f32,
    /// Aerial perspective distance scale in metres (> 0).
    pub aerial_distance_m: f32,
}

impl Default for FogDesc {
    /// `clear`: density 0, so fog and aerial perspective are off.
    fn default() -> Self {
        FogDesc {
            density: 0.0,
            height_falloff: 0.1,
            base_height_m: 0.0,
            start_m: 0.0,
            color: [1.0, 1.0, 1.0],
            inscatter: 0.0,
            aerial_strength: 0.0,
            aerial_distance_m: 1000.0,
        }
    }
}

impl FogDesc {
    /// Built-in presets `clear`, `haze`, `snow`, `jungle`, `night`; `None` for any other
    /// name. The values are starting points for the look tuning.
    pub fn preset(name: &str) -> Option<FogDesc> {
        match name {
            // Fog off; the default description.
            "clear" => Some(FogDesc::default()),
            // Light ground haze: a warm-white veil that thickens toward the horizon and a
            // moderate aerial perspective over 400 m.
            "haze" => Some(FogDesc {
                density: 0.012,
                height_falloff: 0.12,
                base_height_m: 0.0,
                start_m: 0.0,
                color: [0.84, 0.80, 0.72],
                inscatter: 0.6,
                aerial_strength: 0.5,
                aerial_distance_m: 400.0,
            }),
            // Cold blue-white snow haze, nearly uniform with height, aerial over 250 m.
            "snow" => Some(FogDesc {
                density: 0.02,
                height_falloff: 0.08,
                base_height_m: 0.0,
                start_m: 0.0,
                color: [0.78, 0.84, 0.92],
                inscatter: 0.4,
                aerial_strength: 0.6,
                aerial_distance_m: 250.0,
            }),
            // Humid tropics: dense green-grey fog in a low layer over the ground.
            "jungle" => Some(FogDesc {
                density: 0.03,
                height_falloff: 0.3,
                base_height_m: 2.0,
                start_m: 0.0,
                color: [0.42, 0.50, 0.38],
                inscatter: 0.35,
                aerial_strength: 0.4,
                aerial_distance_m: 300.0,
            }),
            // Moonlit night: thin deep blue-violet haze with little inscatter.
            "night" => Some(FogDesc {
                density: 0.015,
                height_falloff: 0.1,
                base_height_m: 0.0,
                start_m: 0.0,
                color: [0.08, 0.07, 0.20],
                inscatter: 0.2,
                aerial_strength: 0.3,
                aerial_distance_m: 500.0,
            }),
            _ => None,
        }
    }

    /// Refuses with `post.invalid` and a message naming the first offending field:
    /// every number finite, `density` in [0, 1], `height_falloff` in (0, 10], `start_m` >= 0,
    /// every `color` component >= 0, `inscatter` in [0, 4], `aerial_strength` in [0, 2],
    /// `aerial_distance_m` > 0.
    pub fn validate(&self) -> Result<(), RaaError> {
        within("density", self.density, 0.0, 1.0)?;
        open_closed("height_falloff", self.height_falloff, 0.0, 10.0)?;
        finite("base_height_m", self.base_height_m)?;
        at_least("start_m", self.start_m, 0.0)?;
        for (i, value) in self.color.iter().copied().enumerate() {
            at_least(&format!("color[{i}]"), value, 0.0)?;
        }
        within("inscatter", self.inscatter, 0.0, 4.0)?;
        within("aerial_strength", self.aerial_strength, 0.0, 2.0)?;
        positive("aerial_distance_m", self.aerial_distance_m)
    }

    /// Writes the S10 `EnvGpu` fog fields: `fog` = (density, falloff, base, start),
    /// `fog_color` = (rgb, inscatter), `aerial` = (strength, distance, 0, 0). Every other
    /// `EnvGpu` field is left untouched. The caller validates and uploads the env.
    pub fn write_env(&self, env: &mut EnvGpu) {
        env.fog = [
            self.density,
            self.height_falloff,
            self.base_height_m,
            self.start_m,
        ];
        let [r, g, b] = self.color;
        env.fog_color = [r, g, b, self.inscatter];
        env.aerial = [self.aerial_strength, self.aerial_distance_m, 0.0, 0.0];
    }
}

fn invalid(message: String) -> RaaError {
    RaaError::FrameGraph {
        code: "post.invalid",
        message,
    }
}

fn finite(field: &str, v: f32) -> Result<(), RaaError> {
    if v.is_finite() {
        Ok(())
    } else {
        Err(invalid(format!("`{field}` must be finite, got {v}")))
    }
}

fn within(field: &str, v: f32, min: f32, max: f32) -> Result<(), RaaError> {
    finite(field, v)?;
    if (min..=max).contains(&v) {
        Ok(())
    } else {
        Err(invalid(format!(
            "`{field}` must be in [{min}, {max}], got {v}"
        )))
    }
}

fn open_closed(field: &str, v: f32, min: f32, max: f32) -> Result<(), RaaError> {
    finite(field, v)?;
    if v > min && v <= max {
        Ok(())
    } else {
        Err(invalid(format!(
            "`{field}` must be in ({min}, {max}], got {v}"
        )))
    }
}

fn at_least(field: &str, v: f32, min: f32) -> Result<(), RaaError> {
    finite(field, v)?;
    if v >= min {
        Ok(())
    } else {
        Err(invalid(format!("`{field}` must be >= {min}, got {v}")))
    }
}

fn positive(field: &str, v: f32) -> Result<(), RaaError> {
    finite(field, v)?;
    if v > 0.0 {
        Ok(())
    } else {
        Err(invalid(format!("`{field}` must be > 0, got {v}")))
    }
}
