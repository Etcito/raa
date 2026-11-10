//! `SkyFeature::install` and the `SkyHandle` setters: the shared sky state (state.rs) seen by
//! the `sky.env` pass (env_pass.rs) and the `sky.background` contributor (background.rs).
//!
//! The handle logic only touches the shared state, so it is tested without a device.

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::{GpuContext, RaaError};
use raa_ibl::IblHandle;
use raa_lit::LitRenderer;
use raa_pipeline::Pipeline;
use raa_scene::DirectionalLight;

use crate::background::SkyBackground;
use crate::desc::SkyDesc;
use crate::env_pass::SkyEnvPass;
use crate::state::{SharedSky, SkyState, invalid};
use crate::time_of_day::{TimeOfDaySettings, TimeOfDayState};

/// `pipeline.busy`: the pipeline or the sky state is borrowed (called while a frame executes).
fn busy(what: &str) -> RaaError {
    RaaError::FrameGraph {
        code: "pipeline.busy",
        message: format!("the {what} is borrowed (called while a frame executes)"),
    }
}

/// Installs the procedural sky on a pipeline and a lit renderer.
pub struct SkyFeature;

impl SkyFeature {
    /// Creates the shared sky state (preset `day`), adds the FeaturePass `sky.env` (order 10,
    /// renders the SKY_ENV cube and filters it through `ibl`) to `pipeline` and the forward
    /// contributor `sky.background` (order 900) to `lit`.
    ///
    /// Refusals: `pipeline.busy` when the pipeline is borrowed, `pipeline.duplicate_pass` when
    /// the sky is already installed (from the pipeline, unchanged).
    pub fn install(
        gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
        lit: &mut LitRenderer,
        ibl: IblHandle,
    ) -> Result<SkyHandle, RaaError> {
        let state = SkyState::shared();
        {
            let mut p = pipeline.try_borrow_mut().map_err(|_| busy("pipeline"))?;
            let pass = SkyEnvPass::new(
                gpu,
                Rc::clone(&state),
                ibl,
                lit.handle(),
                p.lighting().layout(),
            );
            p.add(Box::new(pass))?;
        }
        // `add_contributor` borrows the pipeline itself: the borrow above is released.
        lit.add_contributor(Box::new(SkyBackground::new(Rc::clone(&state))))?;
        Ok(SkyHandle { state })
    }
}

/// Access to the installed sky (clones share one state).
#[derive(Clone)]
pub struct SkyHandle {
    state: SharedSky,
}

impl SkyHandle {
    /// Validates and stores `d`; the env cube and the `EnvGpu` sky fields are rebuilt at the
    /// next frame. `sky.invalid` (naming the field) leaves the current sky unchanged;
    /// `pipeline.busy` when called while a frame executes.
    pub fn set(&self, d: SkyDesc) -> Result<(), RaaError> {
        d.validate()?;
        let mut s = self.state.try_borrow_mut().map_err(|_| busy("sky state"))?;
        s.desc = d;
        s.env_dirty = true;
        s.gpu_dirty = true;
        Ok(())
    }

    /// The current sky description.
    pub fn get(&self) -> SkyDesc {
        // Invariant: the state is only borrowed mutably inside a frame's prepare/record and
        // inside `set`/`mark_dirty`, none of which call back into this handle.
        self.state.borrow().desc.clone()
    }

    /// The sun changed: re-render (and re-filter) the env cube at the next frame.
    pub fn mark_dirty(&self) {
        if let Ok(mut s) = self.state.try_borrow_mut() {
            s.env_dirty = true;
        }
    }

    /// Validates and stores the authored time-of-day settings. A changed authored day or
    /// hour resets the runtime cursor to that moment; pause/resume and speed, place or
    /// orientation edits keep it. An accepted change that moves the sky or light marks the
    /// env cube dirty. `sky.invalid` (naming the field) leaves the state unchanged;
    /// `pipeline.busy` when called while a frame executes.
    pub fn set_time_of_day(&self, settings: TimeOfDaySettings) -> Result<(), RaaError> {
        settings.validate()?;
        let mut s = self.state.try_borrow_mut().map_err(|_| busy("sky state"))?;
        if s.time.apply_settings(settings) {
            s.env_dirty = true;
        }
        Ok(())
    }

    /// The authored settings, the current playback moment and its celestial snapshot.
    /// Getters return a copy; changing the result never changes the stored state.
    pub fn time_of_day(&self) -> TimeOfDayState {
        self.state.borrow().time.state
    }

    /// Validates and stores the manual directional light. While automatic lighting is
    /// disabled this exact descriptor is the rendered light; while enabled it supplies the
    /// undimmed daylight colour and intensity baseline and the astronomical direction wins.
    /// Disabling automatic control restores the stored descriptor. `sky.invalid` (naming
    /// the field) leaves the state unchanged; `pipeline.busy` when called while a frame
    /// executes.
    pub fn set_manual_light(&self, light: DirectionalLight) -> Result<(), RaaError> {
        validate_light(&light)?;
        let mut s = self.state.try_borrow_mut().map_err(|_| busy("sky state"))?;
        if s.time.apply_manual(light) {
            s.env_dirty = true;
        }
        Ok(())
    }

    /// Advances the runtime clock by the real frame delta once per `invocation` and returns
    /// the accepted directional light. The first accepted invocation and a nonpositive
    /// delta do not advance; a repeated invocation returns the accepted snapshot without
    /// advancing again; a nonfinite delta refuses with `sky.invalid` before any mutation.
    /// `pipeline.busy` when called while a frame executes.
    pub fn advance_frame(
        &self,
        invocation: u64,
        dt_seconds: f64,
    ) -> Result<DirectionalLight, RaaError> {
        if !dt_seconds.is_finite() {
            return Err(invalid("advance_frame.dt_seconds must be finite"));
        }
        let mut s = self.state.try_borrow_mut().map_err(|_| busy("sky state"))?;
        let (light, advanced) = s.time.advance(invocation, dt_seconds)?;
        if advanced {
            s.env_dirty = true;
        }
        Ok(light)
    }
}

/// `sky.invalid` for a directional light that is not finite, has a zero direction or a
/// negative colour/intensity. Called before the shared state is touched.
fn validate_light(light: &DirectionalLight) -> Result<(), RaaError> {
    let d = light.direction;
    let length = d.length();
    let finite = d.x.is_finite() && d.y.is_finite() && d.z.is_finite() && length.is_finite();
    if !finite || length <= 0.0 {
        return Err(invalid("light.direction must be a finite nonzero vector"));
    }
    for (i, c) in light.color.into_iter().enumerate() {
        if !c.is_finite() || c < 0.0 {
            return Err(invalid(format!(
                "light.color[{i}] must be finite and nonnegative, got {c}"
            )));
        }
    }
    if !light.intensity.is_finite() || light.intensity < 0.0 {
        return Err(invalid(format!(
            "light.intensity must be finite and nonnegative, got {}",
            light.intensity
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time_of_day::{DominantLight, Place, TimeOfDay};
    use raa_math::Vec3;

    fn handle() -> SkyHandle {
        SkyHandle {
            state: SkyState::shared(),
        }
    }

    fn clean(h: &SkyHandle) {
        let mut s = h.state.borrow_mut();
        s.env_dirty = false;
        s.gpu_dirty = false;
    }

    fn near(a: Vec3, b: Vec3, tolerance: f32) {
        assert!((a - b).length() < tolerance, "{a:?} != {b:?}");
    }

    /// Enabled, playing at one simulated hour per real second (speed 3600 s/s).
    fn settings(hours: f64) -> TimeOfDaySettings {
        TimeOfDaySettings {
            time: TimeOfDay { day: 172, hours },
            enabled: true,
            playing: true,
            speed: 3600.0,
            ..Default::default()
        }
    }

    #[test]
    fn set_get_and_refusal_over_device_free_state() {
        let h = handle();
        assert_eq!(Some(h.get()), SkyDesc::preset("day"));

        clean(&h);
        let night = SkyDesc::preset("night").expect("night preset");
        h.set(night.clone()).expect("valid sky");
        assert_eq!(h.get(), night);
        assert!(h.state.borrow().env_dirty && h.state.borrow().gpu_dirty);
        assert_eq!(h.clone().get(), night, "clones share the state");

        clean(&h);
        let mut bad = SkyDesc::preset("snow").expect("snow preset");
        bad.clouds.coverage = 2.0;
        let err = h.set(bad).expect_err("coverage out of range");
        assert_eq!(err.code(), "sky.invalid");
        assert!(err.to_string().contains("clouds.coverage"), "{err}");
        assert_eq!(h.get(), night, "a refusal changes nothing");
        let s = h.state.borrow();
        assert!(!s.env_dirty && !s.gpu_dirty, "a refusal marks nothing");
    }

    #[test]
    fn mark_dirty_marks_only_the_env() {
        let h = handle();
        clean(&h);
        h.mark_dirty();
        let s = h.state.borrow();
        assert!(s.env_dirty && !s.gpu_dirty);
    }

    #[test]
    fn set_while_borrowed_is_busy() {
        let h = handle();
        let _frame = h.state.borrow_mut();
        let day = SkyDesc::preset("day").expect("day preset");
        assert_eq!(h.set(day).expect_err("busy").code(), "pipeline.busy");
        assert_eq!(
            h.set_time_of_day(settings(9.0)).expect_err("busy").code(),
            "pipeline.busy"
        );
        assert_eq!(
            h.set_manual_light(DirectionalLight::default())
                .expect_err("busy")
                .code(),
            "pipeline.busy"
        );
        assert_eq!(
            h.advance_frame(1, 1.0).expect_err("busy").code(),
            "pipeline.busy"
        );
    }

    #[test]
    fn authored_time_change_resets_cursor_but_pause_and_resume_keep_it() {
        let h = handle();
        let mut s = settings(9.0);
        h.set_time_of_day(s).expect("valid time");
        let first = h.time_of_day();
        assert_eq!(first.current, first.settings.time);

        h.advance_frame(1, 1.0).expect("first invocation");
        h.advance_frame(2, 1.0).expect("advance one hour");
        assert_eq!(h.time_of_day().current.hours, 10.0);

        s.playing = false;
        h.set_time_of_day(s).expect("pause");
        assert_eq!(
            h.time_of_day().current.hours,
            10.0,
            "pause keeps the cursor"
        );
        h.advance_frame(3, 1.0).expect("paused frame");
        assert_eq!(
            h.time_of_day().current.hours,
            10.0,
            "paused does not advance"
        );

        s.playing = true;
        h.set_time_of_day(s).expect("resume");
        assert_eq!(
            h.time_of_day().current.hours,
            10.0,
            "resume keeps the cursor"
        );
        h.advance_frame(4, 1.0).expect("resumed frame");
        assert_eq!(h.time_of_day().current.hours, 11.0);

        s.speed = 7200.0;
        h.set_time_of_day(s).expect("speed change");
        assert_eq!(
            h.time_of_day().current.hours,
            11.0,
            "speed keeps the cursor"
        );

        s.time = TimeOfDay {
            day: 173,
            hours: 8.0,
        };
        s.speed = 3600.0;
        h.set_time_of_day(s).expect("authored edit");
        let reset = h.time_of_day();
        assert_eq!(reset.current, s.time, "an authored edit resets the cursor");
        assert_eq!(reset.settings.time, s.time);
        h.advance_frame(5, 1.0)
            .expect("advance from the reset moment");
        assert_eq!(h.time_of_day().current.hours, 9.0);
        assert_eq!(
            h.time_of_day().settings.time,
            s.time,
            "authored moment is kept"
        );
    }

    #[test]
    fn first_nonpositive_and_repeated_invocations_do_not_over_advance() {
        let h = handle();
        h.set_time_of_day(settings(9.0)).expect("valid time");
        let start = h.time_of_day().current;

        let first = h.advance_frame(10, 1.0).expect("first frame");
        assert_eq!(
            h.time_of_day().current,
            start,
            "the first frame does not advance"
        );
        assert_eq!(h.advance_frame(10, 1.0).expect("repeat"), first);
        assert_eq!(
            h.time_of_day().current,
            start,
            "a repeated ID does not advance"
        );

        h.advance_frame(11, 0.0).expect("zero delta");
        h.advance_frame(12, -5.0).expect("negative delta");
        assert_eq!(
            h.time_of_day().current,
            start,
            "nonpositive deltas do not advance"
        );

        assert_eq!(
            h.advance_frame(13, f64::NAN).expect_err("NaN").code(),
            "sky.invalid"
        );
        assert_eq!(
            h.advance_frame(14, f64::INFINITY)
                .expect_err("infinite")
                .code(),
            "sky.invalid"
        );
        assert_eq!(h.time_of_day().current, start, "refusals leave the cursor");

        h.advance_frame(13, 1.0)
            .expect("the refused ID was not consumed");
        assert_eq!(h.time_of_day().current.hours, 10.0);
    }

    #[test]
    fn manual_light_is_the_baseline_and_is_restored_on_disable() {
        let h = handle();
        let manual = DirectionalLight {
            direction: Vec3::new(0.0, -1.0, 0.0),
            color: [1.0, 0.8, 0.6],
            intensity: 2.0,
        };
        h.set_manual_light(manual).expect("valid manual light");
        assert_eq!(h.advance_frame(1, 1.0).expect("manual frame"), manual);
        assert_eq!(h.time_of_day().dominant, DominantLight::Manual);

        let mut s = settings(9.0);
        h.set_time_of_day(s).expect("enable automatic lighting");
        let sun = h.advance_frame(2, 1.0).expect("sun frame");
        assert_eq!(h.time_of_day().dominant, DominantLight::Sun);
        assert_eq!(sun.color, manual.color, "the baseline colour is kept");
        assert!((sun.intensity - manual.intensity).abs() < 1e-6);
        assert_ne!(
            sun.direction, manual.direction,
            "the astronomical direction wins"
        );
        near(sun.direction, -h.time_of_day().sky.sun, 0.002);

        let raised = DirectionalLight {
            direction: Vec3::new(1.0, 0.0, 0.0),
            color: [0.5, 0.4, 0.3],
            intensity: 1.0,
        };
        h.set_manual_light(raised).expect("new baseline");
        let auto = h.advance_frame(3, 1.0).expect("new baseline frame");
        assert_eq!(auto.color, raised.color);
        assert!((auto.intensity - raised.intensity).abs() < 1e-6);
        near(auto.direction, -h.time_of_day().sky.sun, 0.002);

        s.enabled = false;
        h.set_time_of_day(s).expect("disable automatic lighting");
        assert_eq!(h.advance_frame(4, 1.0).expect("manual again"), raised);
        assert_eq!(h.time_of_day().dominant, DominantLight::Manual);
    }

    #[test]
    fn invalid_time_and_light_edits_refuse_without_mutation() {
        let h = handle();
        clean(&h);
        let before = h.time_of_day();
        for bad in [
            TimeOfDaySettings {
                speed: -1.0,
                ..Default::default()
            },
            TimeOfDaySettings {
                time: TimeOfDay { day: 0, hours: 9.0 },
                ..Default::default()
            },
            TimeOfDaySettings {
                time: TimeOfDay {
                    day: 1,
                    hours: 24.0,
                },
                ..Default::default()
            },
            TimeOfDaySettings {
                place: Place {
                    latitude: 91.0,
                    north: 0.0,
                },
                ..Default::default()
            },
        ] {
            assert_eq!(
                h.set_time_of_day(bad).expect_err("invalid time").code(),
                "sky.invalid"
            );
            assert_eq!(h.time_of_day(), before, "a refusal changes nothing");
        }

        let before_light = h.state.borrow().time.manual;
        for bad in [
            DirectionalLight {
                direction: Vec3::new(f32::NAN, 0.0, 0.0),
                ..DirectionalLight::default()
            },
            DirectionalLight {
                direction: Vec3::ZERO,
                ..DirectionalLight::default()
            },
            DirectionalLight {
                color: [1.0, -1.0, 1.0],
                ..DirectionalLight::default()
            },
            DirectionalLight {
                intensity: f32::NAN,
                ..DirectionalLight::default()
            },
        ] {
            let error = h.set_manual_light(bad).expect_err("invalid light");
            assert_eq!(error.code(), "sky.invalid");
            assert!(error.message().contains("light."), "{error}");
        }
        assert_eq!(h.state.borrow().time.manual, before_light);
        assert_eq!(h.time_of_day(), before, "a refusal changes nothing");
        assert!(!h.state.borrow().env_dirty, "a refusal marks nothing");
    }
}
