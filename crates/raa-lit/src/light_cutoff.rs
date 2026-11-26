//! Effective influence radius of a local light: a light whose contribution stays below what the
//! display can show is shortened or switched off for every consumer of the registry (tile
//! lists, forward shading, DDGI probe rays, hero and stochastic area shadows, GI epochs).
//!
//! Formula. Let `T` be [`LightCutoff::min_radiance`], the scene-linear radiance (exposure not
//! yet applied) a neutral grey needs to reach the display code `threshold` (default 2/255) under
//! the scene's fixed exposure, mapper and look (`raa_tonemap::curves::visible_radiance`;
//! auto-exposure is deliberately ignored, exposure is an artistic choice). Let `L = max(r, g, b)
//! * intensity` (times `max(1, width * height)` for an area light, whose form factor tends to
//! `area / d^2`), the largest radiance a unit-albedo surface can receive at distance 1.
//! Shading is `L * windowed_falloff(d, range)`, bounded by `L / d^2`. Replacing `range` by
//! `r = sqrt(2 L / T)` leaves a difference of at most `L / d^2 * (1 - (1 - (d/r)^4)^2)
//! <= 2 (L / r^2) = T` inside `r` and below `T` outside it, so the cut is at most `T`, under
//! the display threshold. The window `(1 - (d/r)^4)^2` reaches zero with zero slope, so the
//! falloff stays continuous (no pop). The effective range is `min(authored range, r)`. When
//! even the nearest clamp (`windowed_falloff` peaks at 25) stays below `T / 2` (`25 L <= T/2`),
//! the light is off (intensity 0) and every consumer skips it. A directional light has no range:
//! it is off when `L < T`.

use crate::local_light::{LocalLight, LocalLightKind};

/// Default display threshold: two codes of an 8 bit display.
pub const DEFAULT_DISPLAY_THRESHOLD: f32 = 2.0 / 255.0;

/// Peak of `windowed_falloff` (distance clamped to 0.2 m).
const FALLOFF_PEAK: f32 = 25.0;

/// The light cutoff setting resolved to scene radiance.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LightCutoff {
    /// Display threshold as a fraction of white (default 2/255); 0 disables the cutoff.
    pub threshold: f32,
    /// Scene-linear radiance (before exposure) at which a grey reaches `threshold` on screen.
    pub min_radiance: f32,
}

impl Default for LightCutoff {
    /// 2/255 under 0 EV and a linear display transform (the sRGB EOTF of 2/255).
    fn default() -> Self {
        LightCutoff {
            threshold: DEFAULT_DISPLAY_THRESHOLD,
            min_radiance: 6.07e-4,
        }
    }
}

impl LightCutoff {
    /// A cutoff that never changes a light.
    pub const OFF: LightCutoff = LightCutoff {
        threshold: 0.0,
        min_radiance: 0.0,
    };

    /// Whether the cutoff changes lights at all.
    pub fn enabled(self) -> bool {
        self.threshold > 0.0 && self.min_radiance > 0.0 && self.min_radiance.is_finite()
    }
}

impl LocalLight {
    /// Largest radiance this light delivers at distance 1 (see the module formula).
    pub fn peak_radiance(&self) -> f32 {
        let color = self.color[0].max(self.color[1]).max(self.color[2]);
        let area = match self.kind {
            LocalLightKind::Area => (self.size[0] * self.size[1]).max(1.0),
            _ => 1.0,
        };
        color * self.intensity * area
    }

    /// Whether the light contributes nothing (zero radiance).
    pub fn is_off(&self) -> bool {
        self.intensity <= 0.0 || self.color.iter().all(|c| *c <= 0.0)
    }

    /// The light as every consumer must see it under `cutoff`: range shortened to the
    /// influence radius, or intensity 0 when it cannot reach the display threshold.
    pub fn with_cutoff(mut self, cutoff: LightCutoff) -> LocalLight {
        if self.is_off() {
            self.intensity = 0.0;
            return self;
        }
        if !cutoff.enabled() {
            return self;
        }
        let peak = self.peak_radiance();
        let t = cutoff.min_radiance;
        match self.kind {
            LocalLightKind::Directional => {
                if peak < t {
                    self.intensity = 0.0;
                }
            }
            _ => {
                if FALLOFF_PEAK * peak <= 0.5 * t {
                    self.intensity = 0.0;
                } else {
                    let radius = (2.0 * peak / t).sqrt();
                    self.range = self.range.min(radius);
                }
            }
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::light_math::windowed_falloff;

    fn cutoff() -> LightCutoff {
        LightCutoff::default()
    }

    fn lit(intensity: f32, range: f32) -> LocalLight {
        LocalLight {
            intensity,
            range,
            ..LocalLight::default()
        }
    }

    #[test]
    fn off_lights_stay_off_and_dim_lights_are_switched_off() {
        assert!(lit(0.0, 6.0).with_cutoff(cutoff()).is_off());
        assert!(lit(1e-6, 6.0).with_cutoff(cutoff()).is_off());
        assert!(!lit(0.01, 6.0).with_cutoff(cutoff()).is_off());
    }

    #[test]
    fn range_shrinks_for_dim_lights_and_never_grows() {
        let dim = lit(0.05, 50.0).with_cutoff(cutoff());
        let expected = (2.0 * 0.05 / 6.07e-4_f32).sqrt();
        assert!((dim.range - expected).abs() < 1e-4, "{}", dim.range);
        assert!(dim.range < 50.0);
        let bright = lit(2.0, 6.0).with_cutoff(cutoff());
        assert_eq!(bright.range, 6.0, "a bright light keeps its authored range");
        assert_eq!(lit(2.0, 6.0).with_cutoff(LightCutoff::OFF).range, 6.0);
    }

    #[test]
    fn cut_light_differs_from_authored_by_at_most_the_threshold() {
        let t = cutoff().min_radiance;
        for (intensity, range) in [(0.05, 50.0), (0.3, 20.0), (1.0, 100.0), (0.01, 6.0)] {
            let authored = lit(intensity, range);
            let cut = authored.with_cutoff(cutoff());
            for step in 1..4000 {
                let d = step as f32 * 0.025;
                let before = intensity * windowed_falloff(d, authored.range);
                let after = if cut.is_off() {
                    0.0
                } else {
                    cut.intensity * windowed_falloff(d, cut.range)
                };
                assert!(
                    (before - after).abs() <= t * 1.001,
                    "i={intensity} r={range} d={d}: {before} vs {after}"
                );
            }
        }
    }

    #[test]
    fn falloff_reaches_zero_at_the_effective_range_without_a_step() {
        let cut = lit(0.2, 80.0).with_cutoff(cutoff());
        let inside = cut.intensity * windowed_falloff(cut.range * 0.999, cut.range);
        assert!(inside < 1e-3 * cut.intensity, "{inside}");
        assert_eq!(windowed_falloff(cut.range, cut.range), 0.0);
    }

    #[test]
    fn area_lights_count_their_rectangle_and_directional_lights_have_no_range() {
        let small = LocalLight {
            kind: LocalLightKind::Area,
            size: [0.5, 0.5],
            ..lit(0.05, 50.0)
        };
        let large = LocalLight {
            size: [4.0, 4.0],
            ..small
        };
        assert!(large.with_cutoff(cutoff()).range > small.with_cutoff(cutoff()).range);
        let sun = LocalLight {
            kind: LocalLightKind::Directional,
            intensity: 1e-4,
            ..LocalLight::default()
        };
        assert!(sun.with_cutoff(cutoff()).is_off());
        let day = LocalLight {
            intensity: 3.0,
            ..sun
        };
        assert_eq!(day.with_cutoff(cutoff()), day);
    }
}
