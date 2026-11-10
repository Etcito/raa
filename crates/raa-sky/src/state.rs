//! Sky state shared by the `sky.env` pass, the `sky.background` contributor and the sky
//! handle setters.
//!
//! The automatic light's fades, moon handoff and stepped direction are adapted from
//! Solstice's `crates/solstice-render/src/time_of_day.rs:176-232` at revision
//! 7e465b599c13da165948ebaa44df824c36c3ae07.
/*
MIT License

Copyright (c) 2026 SamG-Coder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::RaaError;
use raa_math::Vec3;
use raa_scene::DirectionalLight;

use crate::desc::SkyDesc;
use crate::time_of_day::{
    CelestialSky, DominantLight, TimeOfDay, TimeOfDaySettings, TimeOfDayState,
};

/// The shared sky state; clones of the `Rc` see the same description and flags.
#[allow(dead_code)] // consumed by SkyFeature (sky.env, sky.background)
pub(crate) type SharedSky = Rc<RefCell<SkyState>>;

/// Current sky description and the work it implies.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SkyState {
    /// The current description (default: preset `day`).
    pub desc: SkyDesc,
    /// The env cube must be re-rendered (and re-filtered).
    pub env_dirty: bool,
    /// The `EnvGpu` sky fields must be rewritten from `desc` (`SkyDesc::write_env`).
    pub gpu_dirty: bool,
    /// Set by the `sky.background` contributor in each of its prepares (it only prepares
    /// while enabled).
    pub background_on: bool,
    /// Authored time-of-day settings, the runtime cursor and the directional-light
    /// playback state.
    pub time: TimePlayback,
}

impl SkyState {
    /// A new shared state over [`SkyState::default`].
    #[allow(dead_code)] // consumed by SkyFeature (sky.env, sky.background)
    pub(crate) fn shared() -> SharedSky {
        Rc::new(RefCell::new(SkyState::default()))
    }
}

impl Default for SkyState {
    /// Preset `day`, everything dirty (cube and uniform not written yet), background off.
    fn default() -> Self {
        SkyState {
            // Invariant: `day` is a built-in preset (desc.rs tests check every preset).
            desc: SkyDesc::preset("day").expect("the `day` sky preset exists"),
            env_dirty: true,
            gpu_dirty: true,
            background_on: false,
            time: TimePlayback::default(),
        }
    }
}

/// Moonlight against sunlight, as in Solstice: night is shown brighter than it is so the
/// scene stays readable.
const MOONLIGHT: f32 = 1.0 / 2500.0;
/// The accepted light direction moves on a grid of this many degrees, so a slowly moving
/// sun redraws its shadows only now and then.
const LIGHT_STEP_DEGREES: f32 = 0.05;

/// Authored time-of-day settings and their runtime playback cursor.
///
/// The authored moment ([`TimeOfDayState::settings`]) is what persistence stores; the
/// cursor ([`TimeOfDayState::current`]) advances without touching it. The manual light is
/// the stored descriptor: it is returned verbatim while automatic lighting is disabled and
/// supplies the undimmed colour and intensity baseline while automatic lighting is enabled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TimePlayback {
    /// Authored settings plus the current evaluated time and sky.
    pub state: TimeOfDayState,
    /// The stored manual directional-light descriptor (travel direction, linear colour,
    /// intensity).
    pub manual: DirectionalLight,
    /// Invocation ID of the last accepted [`Self::advance`].
    pub last_invocation: Option<u64>,
    /// The directional light accepted for `last_invocation`, returned again for a repeated
    /// invocation without advancing the clock a second time.
    pub last_light: DirectionalLight,
}

impl Default for TimePlayback {
    fn default() -> Self {
        let state = TimeOfDayState::default();
        let manual = DirectionalLight::default();
        TimePlayback {
            state,
            manual,
            last_invocation: None,
            last_light: manual,
        }
    }
}

impl TimePlayback {
    /// Stores validated settings. A changed authored day or hour resets the runtime cursor
    /// to the authored moment; pause/resume and speed/place/orientation edits keep it.
    /// Returns whether the evaluated sky or accepted light changed.
    pub(crate) fn apply_settings(&mut self, settings: TimeOfDaySettings) -> bool {
        if settings == self.state.settings {
            return false;
        }
        let before = self.visible();
        if settings.time != self.state.settings.time {
            self.state.current = settings.time;
        }
        self.state.settings = settings;
        self.evaluate();
        self.visible() != before
    }

    /// Stores a validated manual light descriptor. While automatic lighting is enabled its
    /// direction is ignored (the astronomical direction wins) but its colour and intensity
    /// are the undimmed daylight baseline. Returns whether the accepted light changed.
    pub(crate) fn apply_manual(&mut self, light: DirectionalLight) -> bool {
        if light == self.manual {
            return false;
        }
        let before = self.last_light;
        self.manual = light;
        self.evaluate();
        self.last_light != before
    }

    /// Advances the runtime cursor at most once for `invocation`, by the real frame delta
    /// scaled by the authored speed. The first accepted invocation and a nonpositive delta
    /// never advance; a repeated invocation returns the accepted snapshot unchanged.
    /// Returns the accepted light and whether the clock advanced.
    pub(crate) fn advance(
        &mut self,
        invocation: u64,
        dt_seconds: f64,
    ) -> Result<(DirectionalLight, bool), RaaError> {
        if !dt_seconds.is_finite() {
            return Err(invalid("advance_frame.dt_seconds must be finite"));
        }
        if self.last_invocation == Some(invocation) {
            return Ok((self.last_light, false));
        }
        let first = self.last_invocation.is_none();
        let hours = if !first
            && dt_seconds > 0.0
            && self.state.settings.enabled
            && self.state.settings.playing
        {
            let hours = dt_seconds * self.state.settings.speed / 3600.0;
            if !hours.is_finite() {
                return Err(invalid(
                    "advance_frame.dt_seconds is too large for the authored speed",
                ));
            }
            Some(hours)
        } else {
            None
        };
        // Advance a copy first so a refusal leaves the cursor and the invocation untouched.
        let mut current = self.state.current;
        if let Some(hours) = hours {
            current.advance(hours)?;
        }
        let advanced = current != self.state.current;
        self.state.current = current;
        self.last_invocation = Some(invocation);
        let light = self.evaluate();
        Ok((light, advanced))
    }

    /// Recomputes the sky and the accepted light from the current time, place and manual
    /// baseline, updating [`TimeOfDayState::sky`] and [`TimeOfDayState::dominant`].
    fn evaluate(&mut self) -> DirectionalLight {
        let sky = self.state.current.sky(&self.state.settings.place);
        self.state.sky = sky;
        if self.state.settings.enabled {
            let (dominant, light) = automatic_light(&sky, &self.manual);
            self.state.dominant = dominant;
            self.last_light = light;
        } else {
            self.state.dominant = DominantLight::Manual;
            self.last_light = self.manual;
        }
        self.last_light
    }

    /// The values a settings edit can change: current time, sky, dominant source and light.
    fn visible(&self) -> (TimeOfDay, CelestialSky, DominantLight, DirectionalLight) {
        (
            self.state.current,
            self.state.sky,
            self.state.dominant,
            self.last_light,
        )
    }
}

/// The automatic directional light: the sun while it lights the scene, the moon after the
/// source's handoff, with the source's fades and stepped direction. Directions are
/// converted from towards-body to RAA's light travel direction.
fn automatic_light(
    sky: &CelestialSky,
    daylight: &DirectionalLight,
) -> (DominantLight, DirectionalLight) {
    let sun_height = sky.sun.y.clamp(-1.0, 1.0).asin().to_degrees();
    let moon_height = sky.moon.y.clamp(-1.0, 1.0).asin().to_degrees();
    let sun_fade = smoothstep(-10.0, -5.0, sun_height);
    let (dominant, toward, color, intensity) = if sun_height > -10.0 {
        (
            DominantLight::Sun,
            sky.sun,
            daylight.color,
            daylight.intensity * sun_fade,
        )
    } else {
        let rise = smoothstep(-4.0, 6.0, moon_height);
        let handoff = smoothstep(-10.0, -14.0, sun_height);
        let warm = [1.0, 0.97, 0.92];
        let color = [
            daylight.color[0] * warm[0],
            daylight.color[1] * warm[1],
            daylight.color[2] * warm[2],
        ];
        (
            DominantLight::Moon,
            sky.moon,
            color,
            daylight.intensity * MOONLIGHT * sky.moon_lit * rise * handoff,
        )
    };
    let light = DirectionalLight {
        direction: -stepped(toward),
        color,
        intensity,
    };
    (dominant, light)
}

/// `direction`, moved to the nearest of a fine set of directions (source `stepped`).
fn stepped(direction: Vec3) -> Vec3 {
    let step = LIGHT_STEP_DEGREES.to_radians();
    let altitude = (direction.y.clamp(-1.0, 1.0).asin() / step).round() * step;
    let azimuth = (direction.x.atan2(-direction.z) / step).round() * step;
    let (sin_alt, cos_alt) = altitude.sin_cos();
    Vec3::new(cos_alt * azimuth.sin(), sin_alt, -cos_alt * azimuth.cos())
}

fn smoothstep(low: f32, high: f32, x: f32) -> f32 {
    let t = ((x - low) / (high - low)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// `sky.invalid`: an invalid time/light edit, naming the field.
pub(crate) fn invalid(message: impl Into<String>) -> RaaError {
    RaaError::FrameGraph {
        code: "sky.invalid",
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_day_and_dirty() {
        let s = SkyState::shared();
        let s = s.borrow();
        assert_eq!(Some(s.desc.clone()), SkyDesc::preset("day"));
        assert!(s.env_dirty && s.gpu_dirty && !s.background_on);
        assert_eq!(s.time.state, TimeOfDayState::default());
        assert_eq!(s.time.manual, DirectionalLight::default());
        assert_eq!(s.time.last_invocation, None);
        assert_eq!(s.time.last_light, DirectionalLight::default());
    }

    #[test]
    fn automatic_light_hands_over_to_a_stepped_moon_below_twilight() {
        let mut playback = TimePlayback::default();
        playback.state.settings.enabled = true;
        playback.state.current = TimeOfDay {
            day: 172,
            hours: 0.0,
        };
        let light = playback.evaluate();
        assert_eq!(playback.state.dominant, DominantLight::Moon);
        assert!((light.direction - -stepped(playback.state.sky.moon)).length() < 1e-6);
        assert!(light.intensity > 0.0 && light.intensity.is_finite());
        assert_eq!(light.color, [1.0, 0.97, 0.92]);
    }

    #[test]
    fn sun_light_uses_the_undimmed_baseline_and_a_stepped_travel_direction() {
        let mut playback = TimePlayback::default();
        playback.state.settings.enabled = true;
        playback.state.current = TimeOfDay {
            day: 172,
            hours: 9.0,
        };
        let manual = DirectionalLight {
            color: [0.9, 0.8, 0.7],
            intensity: 2.5,
            ..DirectionalLight::default()
        };
        playback.manual = manual;
        let light = playback.evaluate();
        assert_eq!(playback.state.dominant, DominantLight::Sun);
        assert_eq!(light.color, manual.color);
        assert_eq!(light.intensity, manual.intensity);
        assert!((light.direction - -stepped(playback.state.sky.sun)).length() < 1e-6);
        assert_eq!(playback.manual, manual, "the raw baseline is not replaced");
    }
}
