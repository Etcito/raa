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
use raa_math::{Mat4, Vec3};
use raa_scene::{Camera, DirectionalLight};

use crate::desc::SkyDesc;
use crate::observations::{CloudResetReasons, SkyReadiness, SkyReadinessStage, SkyViewDiagnostics};
use crate::time_of_day::{
    CelestialSky, DominantLight, TimeOfDay, TimeOfDaySettings, TimeOfDayState,
};
use crate::weather::{
    ResolvedWeatherState, Weather, WeatherCycle, WeatherPreset, WeatherSettings, WeatherUpdate,
};

/// View slots the sky feature tracks; matches the engine's four viewports.
pub(crate) const VIEW_COUNT: usize = 4;

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
    /// Authored weather, its evolving cycle target and the animation cursor.
    pub weather: WeatherPlayback,
    /// Prepared view lifecycles, indexed by view slot; `None` for an absent view.
    pub views: [Option<ViewSlot>; VIEW_COUNT],
    /// Invocation whose recording was last accepted.
    pub invocation: Option<u64>,
    /// Invocation the GPU resources were recorded for (the pipeline frame identity).
    pub recorded_invocation: Option<u64>,
    /// Last invocation with at least one actual submitted view.
    pub submitted_invocation: Option<u64>,
    /// Accepted weather changes and explicit cycle restarts.
    pub weather_generation: u64,
    /// Accepted explicit source replacements (preset reselection and weather-cycle
    /// restarts) that may leave every authored value equal but reset the runtime source.
    pub source_generation: u64,
    /// Acknowledged atmosphere (Milky Way / LUT) work.
    pub atmosphere_generation: u64,
    /// The sky pipeline's readiness observation.
    pub readiness: SkyReadiness,
    /// The view the sky passes prepare and record.
    pub active_view: Option<u32>,
    /// Atmosphere work was recorded and awaits an actual submission.
    pub pending_atmosphere: bool,
    /// The Milky Way bake was recorded and awaits an actual submission.
    pub milky_way_pending: bool,
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
            weather: WeatherPlayback::default(),
            views: std::array::from_fn(|_| None),
            invocation: None,
            recorded_invocation: None,
            submitted_invocation: None,
            weather_generation: 0,
            source_generation: 0,
            atmosphere_generation: 0,
            readiness: SkyReadiness::default(),
            active_view: None,
            pending_atmosphere: false,
            milky_way_pending: false,
        }
    }
}

/// One prepared view's lifecycle state (device-free).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ViewSlot {
    /// The view's real index.
    pub view: u32,
    /// Physical viewport size in pixels; `(0, 0)` until the first frame prepares it.
    pub size: (u32, u32),
    /// The view's camera at the last `set_view`.
    pub camera: Camera,
    /// The previous camera's unjittered view-projection for cloud history.
    pub previous_view_proj: Mat4,
    /// Whether the cloud history may be reprojected.
    pub history_valid: bool,
    /// Invocation whose work this view last recorded.
    pub invocation: Option<u64>,
    /// Last invocation actually submitted for this view.
    pub submitted_invocation: Option<u64>,
    /// Acknowledged sky-view LUT work.
    pub sky_view_generation: u64,
    /// Invalidation bits accepted for this view.
    pub reset_reasons: CloudResetReasons,
    /// The atmosphere pass recorded this view's sky view.
    pub pending_sky_view: bool,
}

impl ViewSlot {
    pub(crate) fn new(view: u32, camera: Camera, history_valid: bool) -> Self {
        ViewSlot {
            view,
            size: (0, 0),
            camera,
            previous_view_proj: camera.view_proj(),
            history_valid,
            invocation: None,
            submitted_invocation: None,
            sky_view_generation: 0,
            reset_reasons: CloudResetReasons::NONE,
            pending_sky_view: false,
        }
    }

    /// The public per-view observation; cloud validity comes from the backend's views.
    pub(crate) fn diagnostics(
        &self,
        cloud_screen_valid: bool,
        cloud_panorama_valid: bool,
        cloud: crate::observations::CloudAccumulationDiagnostics,
    ) -> SkyViewDiagnostics {
        SkyViewDiagnostics {
            view: self.view,
            width: self.size.0,
            height: self.size.1,
            invocation: self.invocation,
            submitted_invocation: self.submitted_invocation,
            sky_view_generation: self.sky_view_generation,
            history_valid: self.history_valid
                && self.submitted_invocation.is_some()
                && (cloud_screen_valid || cloud_panorama_valid),
            reset_reasons: self.reset_reasons,
            cloud_screen_valid,
            cloud_panorama_valid,
            cloud,
        }
    }
}

impl SkyState {
    /// Adds invalidation `reason` to one view or (`None`) every prepared view.
    pub(crate) fn reset_reasons(&mut self, view: Option<u32>, reason: CloudResetReasons) {
        match view {
            Some(view) => {
                if let Some(slot) = self.views.get_mut(view as usize).and_then(Option::as_mut) {
                    slot.reset_reasons = slot.reset_reasons.union(reason);
                }
            }
            None => {
                for slot in self.views.iter_mut().flatten() {
                    slot.reset_reasons = slot.reset_reasons.union(reason);
                }
            }
        }
    }

    /// Counts one accepted explicit source replacement even when the authored values are
    /// unchanged; the env pass cancels older IBL completions before publishing. Exhaustion
    /// refuses rather than reusing an identity.
    pub(crate) fn bump_source(&mut self) -> Result<(), RaaError> {
        self.source_generation = self
            .source_generation
            .checked_add(1)
            .ok_or_else(|| invalid("source_generation is exhausted"))?;
        Ok(())
    }

    /// Counts one accepted effective weather change; exhaustion refuses rather than
    /// rounding or reusing an identity.
    pub(crate) fn bump_weather(&mut self) -> Result<(), RaaError> {
        self.weather_generation = self
            .weather_generation
            .checked_add(1)
            .ok_or_else(|| invalid("weather_generation is exhausted"))?;
        Ok(())
    }

    /// Counts one acknowledged atmosphere submission; exhaustion fails readiness and
    /// returns false instead of reusing an identity.
    pub(crate) fn bump_atmosphere(&mut self) -> bool {
        match self.atmosphere_generation.checked_add(1) {
            Some(next) => {
                self.atmosphere_generation = next;
                true
            }
            None => {
                self.readiness = SkyReadiness::failed(
                    SkyReadinessStage::Submission,
                    "atmosphere_generation is exhausted",
                );
                false
            }
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

    /// Explicit preset selection resets playback even when authored values are equal.
    pub(crate) fn apply_preset_time(&mut self, settings: TimeOfDaySettings) {
        self.state.current = settings.time;
        self.state.settings = settings;
        self.evaluate();
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
            // The manual light is the only sun: the sky draws (and scatters, and lights its
            // clouds from) the direction the scene is lit from, instead of a second astronomical
            // sun at the authored day and hour.
            let toward = (-self.manual.direction).normalize_or_zero();
            if toward != Vec3::ZERO {
                self.state.sky.sun = toward;
            }
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

/// Every noise repeat divides this distance, so wrapping the drift by it changes nothing while
/// keeping the numbers small enough for single precision.
const DRIFT_PERIOD: f32 = 252_000.0;
/// The cloud shapes boil over this repeating range, in metres.
const EVOLUTION_PERIOD: f32 = 7_000.0;
/// How fast cloud shapes boil upwards at evolution 1, in metres per second.
const BOIL_SPEED: f32 = 4.0;
/// A long pause, such as a hidden tab, must not make the clouds leap.
const FRAME_CAP: f64 = 0.25;
/// Explicit CPU advancement is taken a second at a time so the wind follows the weather.
const STEP_CAP: f32 = 1.0;

/// Authored weather, its evolving target and the animation cursor.
///
/// The authored settings (including [`WeatherSettings::target`]) are what persistence stores.
/// The runtime `target` and `current` evolve by themselves: a manual target edit eases toward
/// the authored target, while an active [`WeatherCycle`] moves the runtime target from one
/// built-in state to the next. `start` is the cycle-start identity retained independently of
/// numeric target equality; it is used only when a cycle starts and is kept while changing is
/// off for a later enable.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct WeatherPlayback {
    /// Authored settings.
    pub settings: WeatherSettings,
    /// The weather as rendered now, part way to its target while it changes.
    pub current: Weather,
    /// The evolving target: the authored target or the active cycle's current weather.
    pub target: Weather,
    /// The cycle-start identity, initially fair.
    pub start: WeatherPreset,
    /// The active cycle, `None` while the weather does not change by itself.
    pub cycle: Option<WeatherCycle>,
    /// Easing time for the evolving target, in seconds; `0` means immediate.
    pub transition_seconds: f32,
    /// Wind drift over the ground, wrapped to [`DRIFT_PERIOD`], in metres (east, south).
    pub wind_offset: [f32; 2],
    /// Cloud-boil phase, wrapped to [`EVOLUTION_PERIOD`], in metres.
    pub evolution: f32,
    /// Invocation ID of the last frame that advanced the animation.
    pub last_invocation: Option<u64>,
}

impl Default for WeatherPlayback {
    /// Fair weather, no cycle, and the animation cursor at rest.
    fn default() -> Self {
        let settings = WeatherSettings::default();
        WeatherPlayback {
            settings,
            current: settings.target,
            target: settings.target,
            start: WeatherPreset::Fair,
            cycle: None,
            transition_seconds: settings.transition_seconds,
            wind_offset: [0.0; 2],
            evolution: 0.0,
            last_invocation: None,
        }
    }
}

impl WeatherPlayback {
    /// The externally visible snapshot. Text is generated by the serializing caller, never here.
    pub(crate) fn resolved(&self) -> ResolvedWeatherState {
        ResolvedWeatherState {
            settings: self.settings,
            current: self.current,
            cycle_preset: self.cycle.as_ref().map(|_| self.start),
        }
    }

    /// Applies a complete validated candidate. Returns whether an accepted visible change
    /// happened. A supplied start identity is retained even when it only takes effect at the
    /// next enable; a new supplied start identity, a changed seed, a changing false-to-true
    /// edit or an explicit restart intent restarts an enabled cycle, while an unchanged
    /// `changing: true` alone keeps it running.
    pub(crate) fn apply_update(&mut self, update: WeatherUpdate) -> Result<bool, RaaError> {
        update.validate()?;
        let before = self.visible();
        let previous = self.settings;
        let candidate = update.settings;
        let target_changed = candidate.target != previous.target;
        let transition_changed = candidate.transition_seconds != previous.transition_seconds;
        let seed_changed = candidate.seed != previous.seed;
        let enabling = !previous.changing && candidate.changing;
        let stopping = previous.changing && !candidate.changing;

        let start_changed = update.cycle_start.is_some_and(|start| start != self.start);
        if let Some(start) = update.cycle_start {
            self.start = start;
        }
        self.settings = candidate;

        let restart = candidate.changing
            && (update.restart_cycle || seed_changed || enabling || start_changed);

        if restart {
            self.cycle = Some(WeatherCycle::new(self.start.as_str(), candidate.seed));
            self.target = candidate.target;
            self.transition_seconds = candidate.transition_seconds;
        } else if stopping {
            self.cycle = None;
            self.target = candidate.target;
            self.transition_seconds = candidate.transition_seconds;
        } else {
            if target_changed {
                self.target = candidate.target;
                self.transition_seconds = candidate.transition_seconds;
            } else if transition_changed {
                self.transition_seconds = candidate.transition_seconds;
            }
            if !candidate.changing {
                self.cycle = None;
            }
        }

        if self.transition_seconds <= 0.0 {
            self.current = self.target;
        }
        Ok(restart || self.visible() != before)
    }

    /// Advances the animation at most once for `invocation`, by the real frame delta capped at
    /// [`FRAME_CAP`]. The first accepted invocation and a nonpositive delta advance nothing; a
    /// repeated invocation returns without advancing again. `north` rotates the compass wind
    /// from north into the scene's frame. Returns whether the animation advanced.
    pub(crate) fn advance(&mut self, invocation: u64, dt_seconds: f64, north: f32) -> bool {
        if self.last_invocation == Some(invocation) {
            return false;
        }
        let first = self.last_invocation.is_none();
        self.last_invocation = Some(invocation);
        if first || dt_seconds <= 0.0 {
            return false;
        }
        let seconds = dt_seconds.min(FRAME_CAP) as f32;
        self.step(seconds, north);
        true
    }

    /// Eases toward the target, drifts with the wind and boils the shapes, in steps of at most
    /// one second.
    fn step(&mut self, seconds: f32, north: f32) {
        let mut left = seconds.max(0.0);
        while left > 0.0 {
            let step = left.min(STEP_CAP);
            left -= step;
            if let Some(next) = self.cycle.as_mut().and_then(|cycle| cycle.advance(step))
                && let Some(weather) = Weather::by_name(next)
            {
                self.target = weather;
                self.transition_seconds = WeatherCycle::TRANSITION;
            }
            let ease = if self.transition_seconds <= 0.0 {
                1.0
            } else {
                1.0 - (-step / self.transition_seconds).exp()
            };
            self.current = self.current.lerp(&self.target, ease);
            let [east, south] = self.current.wind_velocity();
            let (sin, cos) = north.sin_cos();
            let wind = [east * cos + south * sin, -east * sin + south * cos];
            self.wind_offset = [
                (self.wind_offset[0] + wind[0] * step).rem_euclid(DRIFT_PERIOD),
                (self.wind_offset[1] + wind[1] * step).rem_euclid(DRIFT_PERIOD),
            ];
            self.evolution = (self.evolution + step * BOIL_SPEED * self.current.evolution)
                .rem_euclid(EVOLUTION_PERIOD);
        }
    }

    /// The values an edit can change: authored settings, current weather and cycle identity.
    fn visible(&self) -> (WeatherSettings, Weather, Option<WeatherPreset>) {
        (
            self.settings,
            self.current,
            self.cycle.as_ref().map(|_| self.start),
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
        assert_eq!(s.weather, WeatherPlayback::default());
        assert_eq!(s.weather.settings, WeatherSettings::default());
        assert_eq!(s.weather.current, Weather::default());
        assert_eq!(s.weather.resolved().cycle_preset, None);
    }

    #[test]
    fn manual_direction_never_changes_the_automatic_light_or_dirties_the_environment() {
        let mut playback = TimePlayback::default();
        playback.state.settings.enabled = true;
        playback.state.current = TimeOfDay {
            day: 172,
            hours: 9.0,
        };
        playback.evaluate();
        let before = (playback.last_light, playback.state.sky, playback.state.dominant);
        for azimuth in [0.3_f32, 1.1, 2.9] {
            let turned = DirectionalLight {
                direction: Vec3::new(azimuth.cos(), -0.8, azimuth.sin()).normalize_or_zero(),
                ..playback.manual
            };
            assert!(!playback.apply_manual(turned), "azimuth {azimuth}");
            assert_eq!(
                (playback.last_light, playback.state.sky, playback.state.dominant),
                before
            );
        }
    }

    #[test]
    fn manual_light_is_the_only_sun_the_sky_draws() {
        let mut playback = TimePlayback::default();
        playback.state.settings.enabled = false;
        playback.state.current = TimeOfDay {
            day: 172,
            hours: 9.0,
        };
        let manual = DirectionalLight {
            direction: Vec3::new(0.3, -0.8, 0.5).normalize_or_zero(),
            ..DirectionalLight::default()
        };
        playback.manual = manual;
        playback.evaluate();
        assert_eq!(playback.state.dominant, DominantLight::Manual);
        assert!((playback.state.sky.sun - -manual.direction).length() < 1e-6);
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
