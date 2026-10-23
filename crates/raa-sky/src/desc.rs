//! Sky and cloud descriptions, presets and validation (`sky.invalid`).

use raa_core::RaaError;
use raa_pipeline::EnvGpu;

/// Procedural cloud layer: a plane at `altitude_m` with fbm density over a coverage threshold.
#[derive(Clone, Debug, PartialEq)]
pub struct CloudDesc {
    /// Fraction of the sky covered, in [0, 1].
    pub coverage: f32,
    /// Optical density multiplier (>= 0).
    pub density: f32,
    /// Height of the cloud plane in metres (> 0).
    pub altitude_m: f32,
    /// Horizontal size of the noise features in metres (> 0).
    pub scale_m: f32,
    /// Wind velocity in m/s along world x and z.
    pub wind_mps: [f32; 2],
    /// Linear rgb cloud albedo (>= 0).
    pub color: [f32; 3],
    /// Strength of the forward-scattering rim, in [0, 1].
    pub silver_lining: f32,
}

impl Default for CloudDesc {
    /// The `day` clouds: altitude 1500 m, scale 2500 m.
    fn default() -> Self {
        CloudDesc {
            coverage: 0.45,
            density: 1.0,
            altitude_m: 1500.0,
            scale_m: 2500.0,
            wind_mps: [4.0, 1.5],
            color: [0.95, 0.95, 0.97],
            silver_lining: 0.8,
        }
    }
}

/// When the sky env cube (and its IBL filtering) is re-rendered.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EnvUpdate {
    /// Only after `SkyHandle::set` or `mark_dirty` (sun changed).
    #[default]
    OnChange,
    /// Every frame (animated clouds in reflections and ambient).
    EveryFrame,
}

/// Procedural sky: zenith/horizon/ground gradient (linear rgb, scaled by `intensity`), sun
/// disc and cloud layer. Scene-referred linear HDR.
#[derive(Clone, Debug, PartialEq)]
pub struct SkyDesc {
    /// Linear rgb at the zenith (>= 0).
    pub zenith: [f32; 3],
    /// Linear rgb at the horizon (>= 0).
    pub horizon: [f32; 3],
    /// Linear rgb below the horizon (>= 0).
    pub ground: [f32; 3],
    /// Multiplier of the gradient (>= 0).
    pub intensity: f32,
    /// Exponent of the horizon blend `pow(1 - max(dir.y, 0), softness)` (> 0).
    pub horizon_softness: f32,
    /// Angular radius of the sun disc in radians, in [0, 0.2].
    pub sun_disc_angular_radius: f32,
    /// Disc radiance multiplier of the sun colour (>= 0).
    pub sun_disc_intensity: f32,
    /// Cloud layer.
    pub clouds: CloudDesc,
    /// When the env cube is re-rendered.
    pub env_update: EnvUpdate,
}

impl SkyDesc {
    /// Built-in presets `day`, `snow`, `jungle`, `night`; `None` for any other name.
    pub fn preset(name: &str) -> Option<SkyDesc> {
        match name {
            // Bright midday: deep blue zenith fading into a pale warm horizon, scattered
            // fluffy white clouds with a strong silver lining.
            "day" => Some(SkyDesc {
                zenith: [0.15, 0.35, 0.85],
                horizon: [0.85, 0.80, 0.70],
                ground: [0.35, 0.30, 0.25],
                intensity: 1.6,
                horizon_softness: 4.0,
                sun_disc_angular_radius: 0.0047,
                sun_disc_intensity: 40.0,
                clouds: CloudDesc::default(),
                env_update: EnvUpdate::OnChange,
            }),
            // Cold overcast: pale blue-grey sky mostly hidden by thick grey-white clouds,
            // bright white snowy ground bouncing light back up.
            "snow" => Some(SkyDesc {
                zenith: [0.55, 0.62, 0.72],
                horizon: [0.82, 0.85, 0.90],
                ground: [0.95, 0.96, 0.98],
                intensity: 1.4,
                horizon_softness: 2.5,
                sun_disc_angular_radius: 0.0047,
                sun_disc_intensity: 12.0,
                clouds: CloudDesc {
                    coverage: 0.75,
                    density: 1.4,
                    wind_mps: [2.0, 0.5],
                    color: [0.85, 0.87, 0.90],
                    silver_lining: 0.4,
                    ..CloudDesc::default()
                },
                env_update: EnvUpdate::OnChange,
            }),
            // Humid tropics: teal-green haze at the horizon, warm sun, many puffy clouds over
            // a dark green canopy.
            "jungle" => Some(SkyDesc {
                zenith: [0.20, 0.42, 0.62],
                horizon: [0.55, 0.78, 0.68],
                ground: [0.18, 0.26, 0.14],
                intensity: 1.5,
                horizon_softness: 3.0,
                sun_disc_angular_radius: 0.0055,
                sun_disc_intensity: 35.0,
                clouds: CloudDesc {
                    coverage: 0.55,
                    density: 1.2,
                    wind_mps: [2.0, 0.5],
                    color: [0.92, 0.94, 0.90],
                    silver_lining: 0.7,
                    ..CloudDesc::default()
                },
                env_update: EnvUpdate::OnChange,
            }),
            // Moonlit night: dark navy zenith, faint horizon glow, a small cool moon-like
            // disc and a few thin dim clouds.
            "night" => Some(SkyDesc {
                zenith: [0.01, 0.02, 0.06],
                horizon: [0.04, 0.06, 0.10],
                ground: [0.02, 0.02, 0.025],
                intensity: 0.05,
                horizon_softness: 3.0,
                sun_disc_angular_radius: 0.003,
                sun_disc_intensity: 8.0,
                clouds: CloudDesc {
                    coverage: 0.2,
                    density: 0.5,
                    wind_mps: [1.0, 0.3],
                    color: [0.50, 0.55, 0.65],
                    silver_lining: 0.3,
                    ..CloudDesc::default()
                },
                env_update: EnvUpdate::OnChange,
            }),
            _ => None,
        }
    }

    /// Refuses with `sky.invalid` and a message naming the first offending field.
    pub fn validate(&self) -> Result<(), RaaError> {
        color("zenith", self.zenith)?;
        color("horizon", self.horizon)?;
        color("ground", self.ground)?;
        at_least("intensity", self.intensity, 0.0)?;
        positive("horizon_softness", self.horizon_softness)?;
        within(
            "sun_disc_angular_radius",
            self.sun_disc_angular_radius,
            0.0,
            0.2,
        )?;
        at_least("sun_disc_intensity", self.sun_disc_intensity, 0.0)?;
        let c = &self.clouds;
        within("clouds.coverage", c.coverage, 0.0, 1.0)?;
        at_least("clouds.density", c.density, 0.0)?;
        positive("clouds.altitude_m", c.altitude_m)?;
        positive("clouds.scale_m", c.scale_m)?;
        finite("clouds.wind_mps[0]", c.wind_mps[0])?;
        finite("clouds.wind_mps[1]", c.wind_mps[1])?;
        color("clouds.color", c.color)?;
        within("clouds.silver_lining", c.silver_lining, 0.0, 1.0)?;
        Ok(())
    }

    /// Writes the S10 `EnvGpu` sky fields: `sun_disc`, `zenith`, `horizon`, `ground`,
    /// `sky_params.x/y` (intensity, softness), `clouds`, `clouds2`, `cloud_color`. The
    /// background mode (`sky_params.z`), `background`, `fog` and `aerial` are left untouched.
    // The `sky.env` pass (a later task) is the caller.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn write_env(&self, env: &mut EnvGpu) {
        let rgb = |c: [f32; 3]| [c[0], c[1], c[2], 0.0];
        env.sun_disc = [
            self.sun_disc_angular_radius,
            self.sun_disc_intensity,
            0.0,
            0.0,
        ];
        env.zenith = rgb(self.zenith);
        env.horizon = rgb(self.horizon);
        env.ground = rgb(self.ground);
        env.sky_params[0] = self.intensity;
        env.sky_params[1] = self.horizon_softness;
        let c = &self.clouds;
        env.clouds = [c.coverage, c.density, c.altitude_m, c.scale_m];
        env.clouds2 = [c.wind_mps[0], c.wind_mps[1], c.silver_lining, 0.0];
        env.cloud_color = rgb(c.color);
    }
}

fn invalid(message: String) -> RaaError {
    RaaError::FrameGraph {
        code: "sky.invalid",
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

fn color(field: &str, c: [f32; 3]) -> Result<(), RaaError> {
    for (i, v) in c.into_iter().enumerate() {
        at_least(&format!("{field}[{i}]"), v, 0.0)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_env_places_fields_and_keeps_background() {
        let sky = SkyDesc::preset("day").unwrap();
        let mut env = EnvGpu {
            sky_params: [9.0, 9.0, 1.0, 7.0],
            fog: [1.0, 2.0, 3.0, 4.0],
            fog_color: [5.0, 6.0, 7.0, 8.0],
            aerial: [0.5, 0.25, 0.0, 0.0],
            ..EnvGpu::default()
        };
        let before = env.clone();
        sky.write_env(&mut env);
        assert_eq!(env.sun_disc, [0.0047, 40.0, 0.0, 0.0]);
        assert_eq!(env.zenith, [0.15, 0.35, 0.85, 0.0]);
        assert_eq!(env.horizon, [0.85, 0.80, 0.70, 0.0]);
        assert_eq!(env.ground, [0.35, 0.30, 0.25, 0.0]);
        assert_eq!(env.sky_params, [1.6, 4.0, 1.0, 7.0]);
        assert_eq!(env.clouds, [0.45, 1.0, 1500.0, 2500.0]);
        assert_eq!(env.clouds2, [4.0, 1.5, 0.8, 0.0]);
        assert_eq!(env.cloud_color, [0.95, 0.95, 0.97, 0.0]);
        assert_eq!(env.background, before.background);
        assert_eq!(env.fog, before.fog);
        assert_eq!(env.fog_color, before.fog_color);
        assert_eq!(env.aerial, before.aerial);
    }

    #[test]
    fn cloud_defaults_match_contract() {
        let c = CloudDesc::default();
        assert_eq!(c.altitude_m, 1500.0);
        assert_eq!(c.scale_m, 2500.0);
        assert_eq!(EnvUpdate::default(), EnvUpdate::OnChange);
    }
}
