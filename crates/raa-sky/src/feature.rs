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
use raa_scene::{Camera, DirectionalLight};

use crate::atmosphere::{AtmosphereResources, SharedAtmosphere, SkyAtmospherePass};
use crate::background::SkyBackground;
use crate::cloud_backend::{
    CloudInvalidation, SharedCloud, SkyCloudLightingPass, SkyCloudScreenPass, create_cloud_backend,
};
use crate::desc::{LegacyCloudFields, SkyDesc, SkyModel, SkyPreset, SkyTransaction};
use crate::env_pass::SkyEnvPass;
use crate::observations::{
    CloudResetReasons, SkyDiagnostics, SkyReadiness, SkyReadinessStage, SkyReadinessStatus,
};
use crate::state::{SharedSky, SkyState, VIEW_COUNT, ViewSlot, invalid};
use crate::time_of_day::{TimeOfDaySettings, TimeOfDayState};
use crate::weather::{ResolvedWeatherState, WeatherPreset, WeatherSettings, WeatherUpdate};

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
        let atmosphere: SharedAtmosphere = Rc::new(RefCell::new(AtmosphereResources::new(gpu)?));
        let cloud: SharedCloud = Rc::new(RefCell::new(create_cloud_backend(gpu)?));
        {
            let mut p = pipeline.try_borrow_mut().map_err(|_| busy("pipeline"))?;
            p.add(Box::new(SkyAtmospherePass::new(
                Rc::clone(&state),
                Rc::clone(&atmosphere),
            )))?;
            p.add(Box::new(SkyCloudLightingPass::new(
                Rc::clone(&state),
                Rc::clone(&cloud),
                Rc::clone(&atmosphere),
            )))?;
            let pass = SkyEnvPass::new(
                gpu,
                Rc::clone(&state),
                ibl.clone(),
                lit.handle(),
                p.lighting().layout(),
                Rc::clone(&atmosphere),
                Rc::clone(&cloud),
            );
            p.add(Box::new(pass))?;
            p.add(Box::new(SkyCloudScreenPass::new(
                Rc::clone(&state),
                Rc::clone(&cloud),
                Rc::clone(&atmosphere),
            )))?;
        }
        // `add_contributor` borrows the pipeline itself: the borrow above is released.
        lit.add_contributor(Box::new(
            SkyBackground::new(Rc::clone(&state))
                .with_resources(Rc::clone(&atmosphere), Rc::clone(&cloud)),
        ))?;
        Ok(SkyHandle {
            state,
            ibl: Some(ibl),
            cloud: Some(cloud),
            atmosphere: Some(atmosphere),
        })
    }
}

/// Access to the installed sky (clones share one state).
#[derive(Clone)]
pub struct SkyHandle {
    state: SharedSky,
    ibl: Option<IblHandle>,
    cloud: Option<SharedCloud>,
    atmosphere: Option<SharedAtmosphere>,
}

impl SkyHandle {
    /// Validates and stores `d`; the env cube and the `EnvGpu` sky fields are rebuilt at the
    /// next frame. `sky.invalid` (naming the field) leaves the current sky unchanged;
    /// `pipeline.busy` when called while a frame executes.
    pub fn set(&self, d: SkyDesc) -> Result<(), RaaError> {
        self.set_transaction(SkyTransaction {
            desc: d,
            preset: None,
            cloud_fields: LegacyCloudFields::ALL,
        })
    }

    /// Atomically accepts a merged sky description and its explicit preset and cloud intent.
    pub fn set_transaction(&self, transaction: SkyTransaction) -> Result<(), RaaError> {
        let SkyTransaction {
            desc,
            preset,
            cloud_fields,
        } = transaction;
        desc.validate()?;
        if cloud_fields.0 & !LegacyCloudFields::ALL.0 != 0 {
            return Err(invalid("cloud_fields contains unknown bits"));
        }
        if preset.is_some_and(|p| p != desc.profile) {
            return Err(invalid("preset must agree with profile"));
        }
        let mut s = self.state.try_borrow_mut().map_err(|_| busy("sky state"))?;
        let model_changed = s.desc.model != desc.model;
        let mut weather_changed = false;
        let mut source_reset = false;
        let mut time = s.time;
        let mut weather = s.weather.clone();
        if desc.model == SkyModel::Physical {
            let mut settings = weather.settings;
            let mut start = None;
            if let Some(p) = preset {
                let (weather_preset, hours) = match p {
                    SkyPreset::Day => (WeatherPreset::Fair, 9.0),
                    SkyPreset::Snow => (WeatherPreset::Overcast, 12.0),
                    SkyPreset::Jungle => (WeatherPreset::Broken, 15.0),
                    SkyPreset::Night => (WeatherPreset::Clear, 0.0),
                };
                let mut time_settings = time.state.settings;
                time_settings.time.day = 172;
                time_settings.time.hours = hours;
                time_settings.validate()?;
                time.apply_preset_time(time_settings);
                settings.target = crate::weather::Weather::by_name(weather_preset.as_str())
                    .expect("built-in weather preset");
                start = Some(weather_preset);
                // An explicit preset selection is an accepted source replacement even when
                // the authored values equal the previous ones: it resets the runtime moment.
                source_reset = true;
            }
            let fields = cloud_fields.0;
            if fields & LegacyCloudFields::COVERAGE.0 != 0 {
                settings.target.coverage = desc.clouds.coverage;
            }
            if fields & LegacyCloudFields::DENSITY.0 != 0 {
                settings.target.density = desc.clouds.density;
            }
            if fields & LegacyCloudFields::ALTITUDE.0 != 0 {
                settings.target.base = desc.clouds.altitude_m;
            }
            if fields & LegacyCloudFields::WIND.0 != 0 {
                let [x, z] = desc.clouds.wind_mps;
                settings.target.wind_speed = x.hypot(z);
                settings.target.wind_from = (-x).atan2(z).to_degrees().rem_euclid(360.0);
            }
            if preset.is_none() && fields != 0 && settings.target != weather.settings.target {
                settings.changing = false;
            }
            let update = WeatherUpdate {
                settings,
                cycle_start: start,
                restart_cycle: preset.is_some(),
            };
            update.validate()?;
            weather_changed = weather.apply_update(update)?;
        }
        if weather_changed {
            s.bump_weather()?;
        }
        if source_reset {
            s.bump_source()?;
        }
        s.desc = desc;
        s.time = time;
        s.weather = weather;
        if model_changed {
            s.reset_reasons(None, CloudResetReasons::MODEL);
            self.invalidate_cloud(None, CloudInvalidation::Model);
        }
        if weather_changed {
            s.reset_reasons(None, CloudResetReasons::WEATHER);
            self.invalidate_cloud(None, CloudInvalidation::Weather);
        }
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

    /// Advances the runtime clock and the weather animation by the real frame delta once per
    /// `invocation` and returns the accepted directional light. The first accepted invocation
    /// and a nonpositive delta advance neither; a repeated invocation returns the accepted
    /// snapshot without advancing again; a nonfinite delta refuses with `sky.invalid` before
    /// any mutation. `pipeline.busy` when called while a frame executes.
    pub fn advance_frame(
        &self,
        invocation: u64,
        dt_seconds: f64,
    ) -> Result<DirectionalLight, RaaError> {
        if !dt_seconds.is_finite() {
            return Err(invalid("advance_frame.dt_seconds must be finite"));
        }
        let mut s = self.state.try_borrow_mut().map_err(|_| busy("sky state"))?;
        let weather_advances = s.weather.last_invocation != Some(invocation)
            && s.weather.last_invocation.is_some()
            && dt_seconds > 0.0;
        if weather_advances && s.weather_generation == u64::MAX {
            return Err(invalid("weather_generation is exhausted"));
        }
        let (light, advanced) = s.time.advance(invocation, dt_seconds)?;
        let north = s.time.state.settings.place.north;
        // Drifting clouds refresh reflections only under `EnvUpdate::EveryFrame`; `OnChange`
        // keeps the authored snapshot, so only an accepted settings edit marks the env dirty.
        if s.weather.advance(invocation, dt_seconds, north) {
            s.weather_generation += 1;
        }
        if advanced {
            s.env_dirty = true;
        }
        Ok(light)
    }

    /// Validates and stores the complete authored weather settings without an explicit
    /// cycle-start identity. A changed seed or a changing false-to-true edit restarts an
    /// enabled cycle; an unchanged `changing: true` alone keeps it running. `sky.invalid`
    /// (naming the field) leaves the state unchanged; `pipeline.busy` when called while a
    /// frame executes.
    pub fn set_weather(&self, settings: WeatherSettings) -> Result<(), RaaError> {
        self.set_weather_update(WeatherUpdate {
            settings,
            cycle_start: None,
            restart_cycle: false,
        })
    }

    /// Validates the complete weather candidate before mutation, then stores the authored
    /// settings, the retained cycle-start identity and the explicit restart intent. An
    /// new start identity, explicit restart intent, a changed seed or a changing false-to-true
    /// edit restarts an enabled cycle; `restart_cycle` never enables changing by itself. `sky.invalid` (naming the
    /// field) leaves the state unchanged; `pipeline.busy` when called while a frame executes.
    pub fn set_weather_update(&self, update: WeatherUpdate) -> Result<(), RaaError> {
        update.validate()?;
        let mut s = self.state.try_borrow_mut().map_err(|_| busy("sky state"))?;
        let mut weather = s.weather.clone();
        let accepted = weather.apply_update(update)?;
        if accepted {
            s.bump_weather()?;
            // A restart (or any accepted authored edit) replaces the accepted source even
            // when an unchanged cycle restarts to equal values.
            s.bump_source()?;
            s.weather = weather;
            s.reset_reasons(None, CloudResetReasons::WEATHER);
            self.invalidate_cloud(None, CloudInvalidation::Weather);
            s.env_dirty = true;
        }
        Ok(())
    }

    /// The authored weather settings, the currently rendered weather and the active cycle's
    /// preset identity (`None` whenever no cycle is active). The getter returns a copy and
    /// never generates text.
    pub fn weather(&self) -> ResolvedWeatherState {
        self.state.borrow().weather.resolved()
    }

    /// Records (or updates) one view's real index, camera and history eligibility. The next
    /// sky pass prepares this view at the frame's physical size; a `history_valid == false`
    /// camera cut records the `CameraDiscontinuity` reset reason and discards the view's
    /// cloud history. `sky.invalid` for an index outside `0..4` or a non-finite camera.
    pub fn set_view(
        &self,
        view: u32,
        camera: &Camera,
        history_valid: bool,
    ) -> Result<(), RaaError> {
        if view as usize >= VIEW_COUNT {
            return Err(invalid(format!("view {view} is outside 0..{VIEW_COUNT}")));
        }
        if !camera_is_finite(camera) {
            return Err(invalid(format!("view {view} camera must be finite")));
        }
        let mut s = self.state.try_borrow_mut().map_err(|_| busy("sky state"))?;
        let index = view as usize;
        let mut cut = false;
        match s.views[index].as_mut() {
            Some(slot) => {
                if slot.camera != *camera {
                    slot.previous_view_proj = slot.camera.view_proj();
                    slot.camera = *camera;
                }
                cut = slot.history_valid && !history_valid;
                slot.history_valid = history_valid;
            }
            None => {
                s.views[index] = Some(ViewSlot::new(view, *camera, history_valid));
            }
        }
        if cut {
            if let Some(slot) = s.views[index].as_mut() {
                slot.reset_reasons = slot
                    .reset_reasons
                    .union(CloudResetReasons::CAMERA_DISCONTINUITY);
            }
            self.invalidate_cloud(Some(view), CloudInvalidation::CameraDiscontinuity);
        }
        s.active_view = Some(view);
        s.readiness = SkyReadiness::pending(SkyReadinessStage::Atmosphere);
        Ok(())
    }

    /// Acknowledges exactly the views an actual pipeline submission carried. Nothing is
    /// accepted for an empty mask; an incomplete submission acknowledges its submitted views
    /// but invalidates every view's history and stays pending. Only acknowledged work advances
    /// the atmosphere/sky-view generations, and no recording alone becomes ready.
    pub fn observe_submission(&self, invocation: u64, submitted_views: u8, complete: bool) {
        let recorded_invocation = {
            let Ok(s) = self.state.try_borrow() else {
                return;
            };
            s.recorded_invocation.unwrap_or(invocation)
        };
        let bake_ready = if let Some(atmosphere) = self.atmosphere() {
            match atmosphere.try_borrow_mut() {
                Ok(mut resources) => {
                    resources.observe_submission(recorded_invocation, submitted_views, complete);
                    resources.milky_way_ready()
                }
                Err(_) => false,
            }
        } else {
            true
        };
        {
            let Ok(mut s) = self.state.try_borrow_mut() else {
                return;
            };
            let mask = submitted_views & 0x0f;
            let prior_failed = s.readiness.status == SkyReadinessStatus::Failed;
            let mut ack_failed = false;
            if mask != 0 {
                s.invocation = Some(invocation);
                s.submitted_invocation = Some(invocation);
                let mut acknowledged_sky = false;
                for view in 0..VIEW_COUNT as u32 {
                    if mask & (1 << view) == 0 {
                        continue;
                    }
                    let Some(slot) = s.views[view as usize].as_mut() else {
                        continue;
                    };
                    slot.invocation = Some(invocation);
                    slot.submitted_invocation = Some(invocation);
                    if slot.pending_sky_view {
                        let next = slot.sky_view_generation.checked_add(1);
                        slot.pending_sky_view = false;
                        match next {
                            Some(next) => {
                                slot.sky_view_generation = next;
                                acknowledged_sky = true;
                            }
                            None => {
                                ack_failed = true;
                                s.readiness = SkyReadiness::failed(
                                    SkyReadinessStage::Submission,
                                    "sky_view_generation is exhausted",
                                );
                            }
                        }
                    }
                }
                if (acknowledged_sky || s.pending_atmosphere) && !s.bump_atmosphere() {
                    ack_failed = true;
                }
                s.pending_atmosphere = false;
                if complete && bake_ready {
                    s.milky_way_pending = false;
                }
                if !complete {
                    for slot in s.views.iter_mut().flatten() {
                        slot.history_valid = false;
                    }
                }
                let pending = s.milky_way_pending
                    || s.views.iter().flatten().any(|slot| slot.pending_sky_view);
                if !ack_failed {
                    s.readiness = if complete && !pending {
                        SkyReadiness {
                            status: SkyReadinessStatus::Ready,
                            stage: SkyReadinessStage::Submission,
                            error: None,
                        }
                    } else if prior_failed {
                        s.readiness.clone()
                    } else {
                        SkyReadiness::pending(SkyReadinessStage::Submission)
                    };
                }
            } else if !complete {
                for slot in s.views.iter_mut().flatten() {
                    slot.history_valid = false;
                }
                if !prior_failed {
                    s.readiness = SkyReadiness::pending(SkyReadinessStage::Submission);
                }
            }
        }
        if let Some(cloud) = &self.cloud
            && let Ok(mut backend) = cloud.try_borrow_mut()
        {
            for view in 0..VIEW_COUNT as u32 {
                if submitted_views & (1 << view) != 0 {
                    backend.submitted(view, invocation);
                }
            }
        }
    }

    /// The exact typed lifecycle observation; reading it changes no clock, resource or
    /// authored setting.
    pub fn diagnostics(&self) -> SkyDiagnostics {
        let state = self.state.borrow();
        let environment = self.ibl.as_ref().map_or_else(
            || raa_ibl::IblPublication {
                requested_generation: 0,
                published_generation: None,
                pending: false,
            },
            IblHandle::publication,
        );
        let views = std::array::from_fn(|index| {
            state.views[index].as_ref().map(|slot| {
                let (screen, panorama) = self
                    .cloud_views(slot.view)
                    .map(|views| (views.screen_valid, views.panorama_valid))
                    .unwrap_or((false, false));
                slot.diagnostics(screen, panorama)
            })
        });
        let mut readiness = state.readiness.clone();
        if readiness.status != SkyReadinessStatus::Failed
            && let Some(error) = self.ibl.as_ref().and_then(IblHandle::publication_error)
        {
            readiness = SkyReadiness::failed(SkyReadinessStage::Environment, error.to_string());
        }
        SkyDiagnostics {
            invocation: state.invocation,
            submitted_invocation: state.submitted_invocation,
            weather_generation: state.weather_generation,
            atmosphere_generation: state.atmosphere_generation,
            environment,
            views,
            readiness,
        }
    }

    /// Invalidates a removed view (`ViewRemoved`) and then releases its atmosphere and cloud
    /// resources; the slot becomes `None`. Valid indexes without a prepared view are a no-op.
    pub fn remove_view(&self, view: u32) -> Result<(), RaaError> {
        if view as usize >= VIEW_COUNT {
            return Err(invalid(format!("view {view} is outside 0..{VIEW_COUNT}")));
        }
        let mut s = self.state.try_borrow_mut().map_err(|_| busy("sky state"))?;
        let index = view as usize;
        if s.views[index].is_none() {
            return Ok(());
        }
        if let Some(slot) = s.views[index].as_mut() {
            slot.reset_reasons = slot.reset_reasons.union(CloudResetReasons::VIEW_REMOVED);
        }
        self.invalidate_cloud(Some(view), CloudInvalidation::ViewRemoved);
        if let Some(atmosphere) = &self.atmosphere
            && let Ok(mut resources) = atmosphere.try_borrow_mut()
        {
            resources.remove_view(view);
        }
        s.views[index] = None;
        if s.active_view == Some(view) {
            s.active_view = None;
        }
        s.readiness = SkyReadiness::pending(SkyReadinessStage::Atmosphere);
        Ok(())
    }

    /// The active view's cloned cloud views (crate seam for the physical sky consumers).
    pub(crate) fn cloud_views(&self, view: u32) -> Option<crate::cloud_backend::CloudViews> {
        let cloud = self.cloud.as_ref()?;
        let backend = cloud.try_borrow().ok()?;
        Some(backend.views(view))
    }

    /// The shared opaque atmosphere resources (crate seam for the physical sky consumers).
    pub(crate) fn atmosphere(&self) -> Option<&SharedAtmosphere> {
        self.atmosphere.as_ref()
    }

    /// Records an invalidation on the completed backend, if one is installed.
    fn invalidate_cloud(&self, view: Option<u32>, reason: CloudInvalidation) {
        if let Some(cloud) = &self.cloud
            && let Ok(mut backend) = cloud.try_borrow_mut()
        {
            backend.invalidate(view, reason);
        }
    }
}

/// True when every camera field is finite and its clip/viewport domains are usable.
fn camera_is_finite(camera: &Camera) -> bool {
    let fields = [
        camera.eye.x,
        camera.eye.y,
        camera.eye.z,
        camera.target.x,
        camera.target.y,
        camera.target.z,
        camera.up.x,
        camera.up.y,
        camera.up.z,
        camera.fov_y_rad,
        camera.near,
        camera.far,
        camera.aspect,
    ];
    fields.iter().all(|value| value.is_finite())
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
    use crate::weather::{Weather, WeatherCycle, WeatherPreset};
    use raa_math::Vec3;

    fn handle() -> SkyHandle {
        SkyHandle {
            state: SkyState::shared(),
            ibl: None,
            cloud: None,
            atmosphere: None,
        }
    }

    #[test]
    fn transactions_preserve_presence_and_are_atomic() {
        let h = handle();
        for (preset, weather, hours) in [
            (SkyPreset::Day, WeatherPreset::Fair, 9.0),
            (SkyPreset::Snow, WeatherPreset::Overcast, 12.0),
            (SkyPreset::Jungle, WeatherPreset::Broken, 15.0),
            (SkyPreset::Night, WeatherPreset::Clear, 0.0),
        ] {
            let desc = SkyDesc::preset(preset.as_str()).unwrap();
            h.set_transaction(SkyTransaction {
                desc,
                preset: Some(preset),
                cloud_fields: LegacyCloudFields::NONE,
            })
            .unwrap();
            assert_eq!(h.time_of_day().settings.time.hours, hours);
            assert_eq!(
                h.weather().settings.target,
                Weather::by_name(weather.as_str()).unwrap()
            );
        }
        let desc = SkyDesc::preset("day").unwrap();
        h.set_transaction(SkyTransaction {
            desc: desc.clone(),
            preset: Some(SkyPreset::Day),
            cloud_fields: LegacyCloudFields::NONE,
        })
        .unwrap();
        let mut time = h.time_of_day().settings;
        time.time.hours = 18.0;
        h.set_time_of_day(time).unwrap();
        h.set_transaction(SkyTransaction {
            desc: desc.clone(),
            preset: None,
            cloud_fields: LegacyCloudFields::NONE,
        })
        .unwrap();
        assert_eq!(h.time_of_day().settings.time.hours, 18.0);
        h.set_transaction(SkyTransaction {
            desc: desc.clone(),
            preset: Some(SkyPreset::Day),
            cloud_fields: LegacyCloudFields::COVERAGE,
        })
        .unwrap();
        assert_eq!(h.time_of_day().settings.time.hours, 9.0);
        assert_eq!(h.weather().settings.target.coverage, desc.clouds.coverage);
        let before = h.state.borrow().clone();
        let mut bad = desc.clone();
        bad.clouds.wind_mps = [f32::MAX, f32::MAX];
        assert!(
            h.set_transaction(SkyTransaction {
                desc: bad,
                preset: Some(SkyPreset::Day),
                cloud_fields: LegacyCloudFields::WIND
            })
            .is_err()
        );
        assert_eq!(*h.state.borrow(), before);
        assert!(
            h.set_transaction(SkyTransaction {
                desc,
                preset: None,
                cloud_fields: LegacyCloudFields(16)
            })
            .is_err()
        );
        assert_eq!(*h.state.borrow(), before);
    }

    #[test]
    fn shared_cloud_edits_and_later_weather_setters_win() {
        let h = handle();
        let mut desc = h.get();
        desc.clouds.density = 2.5;
        desc.clouds.wind_mps = [3.0, 4.0];
        h.set_transaction(SkyTransaction {
            desc: desc.clone(),
            preset: Some(SkyPreset::Day),
            cloud_fields: LegacyCloudFields(
                LegacyCloudFields::DENSITY.0 | LegacyCloudFields::WIND.0,
            ),
        })
        .unwrap();
        let mut settings = h.weather().settings;
        assert_eq!(settings.target.density, 2.5);
        assert_eq!(settings.target.wind_speed, 5.0);
        settings.target.coverage = 0.7;
        h.set_weather(settings).unwrap();
        desc.model = SkyModel::Legacy;
        h.set_transaction(SkyTransaction {
            desc: desc.clone(),
            preset: None,
            cloud_fields: LegacyCloudFields::NONE,
        })
        .unwrap();
        desc.model = SkyModel::Physical;
        h.set_transaction(SkyTransaction {
            desc,
            preset: None,
            cloud_fields: LegacyCloudFields::NONE,
        })
        .unwrap();
        assert_eq!(h.weather().settings, settings);
    }

    fn clean(h: &SkyHandle) {
        let mut s = h.state.borrow_mut();
        s.env_dirty = false;
        s.gpu_dirty = false;
    }

    #[test]
    fn explicit_resets_advance_the_source_generation_independently_of_values() {
        let h = handle();
        let day = SkyDesc::preset("day").expect("day preset");
        h.set_transaction(SkyTransaction {
            desc: day.clone(),
            preset: Some(SkyPreset::Day),
            cloud_fields: LegacyCloudFields::NONE,
        })
        .unwrap();
        let after_preset = h.state.borrow().source_generation;
        assert_eq!(
            after_preset, 1,
            "preset selection is an accepted source event"
        );

        // Playback moves the runtime moment but keeps every authored value.
        h.set_time_of_day(settings(9.0)).unwrap();
        h.advance_frame(1, 1.0).unwrap();
        h.advance_frame(2, 1.0).unwrap();
        assert_eq!(h.time_of_day().current.hours, 10.0);
        assert_eq!(h.time_of_day().settings.time.hours, 9.0);
        assert_eq!(
            h.state.borrow().source_generation,
            after_preset,
            "playback is evolution within the accepted source"
        );

        // Reselecting the same preset resets the runtime moment without changing any
        // authored value; it is still an accepted source replacement.
        h.set_transaction(SkyTransaction {
            desc: day,
            preset: Some(SkyPreset::Day),
            cloud_fields: LegacyCloudFields::NONE,
        })
        .unwrap();
        assert_eq!(h.time_of_day().current.hours, 9.0);
        assert_eq!(h.state.borrow().source_generation, after_preset + 1);

        // An unchanged weather-cycle restart is the same kind of accepted event.
        let mut weather = h.weather().settings;
        weather.changing = true;
        weather.transition_seconds = 0.0;
        h.set_weather_update(WeatherUpdate {
            settings: weather,
            cycle_start: Some(WeatherPreset::Fair),
            restart_cycle: true,
        })
        .unwrap();
        let after_start = h.state.borrow().source_generation;
        h.set_weather_update(WeatherUpdate {
            settings: weather,
            cycle_start: Some(WeatherPreset::Fair),
            restart_cycle: true,
        })
        .unwrap();
        assert_eq!(h.weather().settings, weather, "authored values are equal");
        assert_eq!(
            h.state.borrow().source_generation,
            after_start + 1,
            "an unchanged explicit cycle restart replaces the source"
        );

        // Advancing playback still does not replace the source.
        h.advance_frame(3, 1.0).unwrap();
        assert_eq!(h.state.borrow().source_generation, after_start + 1);
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

    fn weather_settings(changing: bool, target: Weather) -> WeatherSettings {
        WeatherSettings {
            changing,
            target,
            ..WeatherSettings::default()
        }
    }

    fn cycle_name(h: &SkyHandle) -> Option<&'static str> {
        h.state
            .borrow()
            .weather
            .cycle
            .as_ref()
            .map(WeatherCycle::current)
    }

    #[test]
    fn weather_starts_fair_and_a_full_edit_stores_authored_values() {
        let h = handle();
        let initial = h.weather();
        assert_eq!(initial.settings, WeatherSettings::default());
        assert_eq!(initial.current, Weather::default());
        assert_eq!(initial.cycle_preset, None);

        clean(&h);
        let storm = Weather::by_name("storm").expect("storm preset");
        let settings = weather_settings(false, storm);
        h.set_weather(settings).expect("valid weather");
        let resolved = h.weather();
        assert_eq!(resolved.settings, settings);
        assert_eq!(resolved.current, storm, "transition 0 is immediate");
        assert_eq!(resolved.cycle_preset, None);
        assert!(h.state.borrow().env_dirty);
    }

    #[test]
    fn invalid_weather_edits_refuse_without_mutation() {
        let h = handle();
        let storm = Weather::by_name("storm").expect("storm preset");
        h.set_weather(weather_settings(false, storm))
            .expect("valid weather");
        let before = h.weather();
        clean(&h);
        for bad in [
            WeatherSettings {
                target: Weather { rain: 2.0, ..storm },
                ..WeatherSettings::default()
            },
            WeatherSettings {
                transition_seconds: -1.0,
                ..WeatherSettings::default()
            },
            WeatherSettings {
                target: Weather {
                    wind_from: f32::NAN,
                    ..storm
                },
                ..WeatherSettings::default()
            },
        ] {
            assert_eq!(
                h.set_weather(bad).expect_err("invalid weather").code(),
                "sky.invalid"
            );
            let update = WeatherUpdate {
                settings: bad,
                cycle_start: Some(WeatherPreset::Storm),
                restart_cycle: true,
            };
            assert_eq!(
                h.set_weather_update(update)
                    .expect_err("invalid update")
                    .code(),
                "sky.invalid"
            );
            assert_eq!(h.weather(), before, "a refusal changes nothing");
        }
        assert!(!h.state.borrow().env_dirty, "a refusal marks nothing");

        let _frame = h.state.borrow_mut();
        assert_eq!(
            h.set_weather(WeatherSettings::default())
                .expect_err("busy")
                .code(),
            "pipeline.busy"
        );
    }

    #[test]
    fn preset_plus_overrides_keeps_the_named_cycle_start() {
        let h = handle();
        let mut target = Weather::by_name("storm").expect("storm preset");
        target.coverage = 0.55;
        let mut settings = weather_settings(true, target);
        settings.seed = 9;
        h.set_weather_update(WeatherUpdate {
            settings,
            cycle_start: Some(WeatherPreset::Storm),
            restart_cycle: true,
        })
        .expect("valid update");
        let resolved = h.weather();
        assert_eq!(resolved.cycle_preset, Some(WeatherPreset::Storm));
        assert_eq!(resolved.settings.target, target, "overrides are authored");
        assert_eq!(cycle_name(&h), Some("storm"), "the cycle starts at storm");
    }

    #[test]
    fn seed_and_enable_edits_restart_while_an_unchanged_enable_does_not() {
        let h = handle();
        let fair = Weather::default();
        h.set_weather_update(WeatherUpdate {
            settings: weather_settings(true, fair),
            cycle_start: Some(WeatherPreset::Fair),
            restart_cycle: true,
        })
        .expect("start cycle");
        assert_eq!(cycle_name(&h), Some("fair"));

        let mut settings = weather_settings(true, fair);
        h.advance_frame(1, 0.0).expect("first invocation");
        for invocation in 2..=(4 * (WeatherCycle::DWELL.1 as u64 + 2)) {
            h.advance_frame(invocation, 1.0).expect("advance cycle");
        }
        assert_ne!(cycle_name(&h), Some("fair"), "the cycle moved on");

        h.set_weather(settings)
            .expect("unchanged enable keeps the cycle");
        assert_ne!(
            cycle_name(&h),
            Some("fair"),
            "an unchanged changing:true alone does not restart"
        );

        settings.seed = 42;
        h.set_weather(settings).expect("seed change");
        assert_eq!(cycle_name(&h), Some("fair"), "a seed change restarts");

        h.set_weather(weather_settings(false, fair))
            .expect("stop changing");
        assert_eq!(h.weather().cycle_preset, None);
        h.set_weather(weather_settings(true, fair))
            .expect("enable changing");
        assert_eq!(
            h.weather().cycle_preset,
            Some(WeatherPreset::Fair),
            "false-to-true restarts"
        );
    }

    #[test]
    fn explicit_selection_restarts_even_when_values_are_unchanged() {
        let h = handle();
        let broken = Weather::by_name("broken").expect("broken preset");
        let settings = weather_settings(true, broken);
        let select = WeatherUpdate {
            settings,
            cycle_start: Some(WeatherPreset::Broken),
            restart_cycle: true,
        };
        h.set_weather_update(select).expect("select broken");
        h.advance_frame(1, 0.0).expect("first invocation");
        for invocation in 2..=(4 * (WeatherCycle::DWELL.1 as u64 + 2)) {
            h.advance_frame(invocation, 1.0).expect("advance cycle");
        }
        assert_ne!(cycle_name(&h), Some("broken"));

        h.set_weather_update(WeatherUpdate {
            restart_cycle: false,
            ..select
        })
        .expect("identity only");
        assert_ne!(
            cycle_name(&h),
            Some("broken"),
            "reapplying an identity without intent does not reset playback"
        );

        h.set_weather_update(select).expect("reselect broken");
        assert_eq!(
            cycle_name(&h),
            Some("broken"),
            "the explicit selection restarts the cycle"
        );
    }

    #[test]
    fn manual_target_edit_stops_changing_until_enabled_again() {
        let h = handle();
        let fair = Weather::default();
        h.set_weather_update(WeatherUpdate {
            settings: weather_settings(true, fair),
            cycle_start: Some(WeatherPreset::Broken),
            restart_cycle: true,
        })
        .expect("cycle from broken");
        assert_eq!(h.weather().cycle_preset, Some(WeatherPreset::Broken));

        let storm = Weather::by_name("storm").expect("storm preset");
        h.set_weather(weather_settings(false, storm))
            .expect("manual target edit");
        let stopped = h.weather();
        assert_eq!(stopped.settings.target, storm);
        assert_eq!(stopped.cycle_preset, None, "a manual edit stops changing");
        assert_eq!(stopped.current, storm);

        h.set_weather(weather_settings(true, storm))
            .expect("enable changing");
        assert_eq!(
            h.weather().cycle_preset,
            Some(WeatherPreset::Broken),
            "the retained start is used at the next enable"
        );
    }

    #[test]
    fn weather_advances_once_per_invocation_and_eases_exponentially() {
        let h = handle();
        let storm = Weather::by_name("storm").expect("storm preset");
        h.set_weather(WeatherSettings {
            transition_seconds: 2.0,
            ..weather_settings(false, storm)
        })
        .expect("easing target");

        h.advance_frame(1, 1.0).expect("first invocation");
        assert_eq!(
            h.weather().current,
            Weather::default(),
            "the first frame does not advance"
        );
        h.advance_frame(2, 1.0).expect("advance");
        let eased = h.weather().current;
        assert_ne!(eased, Weather::default(), "the weather moved toward storm");
        assert_ne!(eased, storm, "the transition is exponential, not immediate");
        let offset = h.state.borrow().weather.wind_offset;
        assert_ne!(offset, [0.0; 2], "the wind drifted");

        h.advance_frame(2, 1.0).expect("repeated invocation");
        assert_eq!(h.weather().current, eased, "a repeated ID does not advance");
        assert_eq!(h.state.borrow().weather.wind_offset, offset);
    }

    #[test]
    fn view_lifecycle_records_invalidations_and_only_submitted_masks() {
        let h = handle();
        let mut camera = Camera::default();
        h.set_view(0, &camera, true).expect("first view");
        let d = h.diagnostics();
        assert_eq!(d.readiness.status, SkyReadinessStatus::Pending);
        assert_eq!(d.readiness.stage, SkyReadinessStage::Atmosphere);
        assert_eq!(d.submitted_invocation, None);
        let view = d.views[0].expect("view 0 record");
        assert_eq!((view.view, view.width, view.height), (0, 0, 0));
        assert_eq!(view.invocation, None);
        assert_eq!(view.submitted_invocation, None);
        assert_eq!(view.sky_view_generation, 0);
        assert_eq!(
            view.reset_reasons,
            CloudResetReasons::NONE,
            "creation is not an invalidation"
        );
        assert!(!view.history_valid, "no submitted cloud history yet");
        assert!(!view.cloud_screen_valid && !view.cloud_panorama_valid);
        assert_eq!(d.views[1], None);

        assert_eq!(
            h.set_view(4, &camera, true).expect_err("index").code(),
            "sky.invalid"
        );
        camera.eye.x = 2.0;
        h.set_view(0, &camera, false).expect("camera cut");
        assert!(
            h.state.borrow().views[0]
                .as_ref()
                .expect("slot")
                .reset_reasons
                .contains(CloudResetReasons::CAMERA_DISCONTINUITY)
        );
        h.set_view(0, &camera, true).expect("history eligible");
        assert!(
            h.state.borrow().views[0]
                .as_ref()
                .expect("slot")
                .history_valid
        );

        let mut legacy = h.get();
        legacy.model = SkyModel::Legacy;
        h.set(legacy).expect("model switch");
        let mut storm = h.weather().settings;
        storm.target = Weather::by_name("storm").expect("storm");
        h.set_weather(storm).expect("weather change");
        let d = h.diagnostics();
        let reasons = d.views[0].expect("view 0").reset_reasons;
        assert!(reasons.contains(CloudResetReasons::MODEL));
        assert!(reasons.contains(CloudResetReasons::WEATHER));
        assert_eq!(d.weather_generation, 1, "one accepted weather change");

        h.observe_submission(7, 0, false);
        let d = h.diagnostics();
        assert_eq!(d.submitted_invocation, None, "no view was submitted");
        assert_eq!(d.readiness.status, SkyReadinessStatus::Pending);

        h.observe_submission(8, 0b0001, false);
        let d = h.diagnostics();
        assert_eq!(d.submitted_invocation, Some(8));
        assert_eq!(d.views[0].expect("view 0").submitted_invocation, Some(8));
        assert!(!d.views[0].expect("view 0").history_valid);
        assert_eq!(d.readiness.status, SkyReadinessStatus::Pending);
        assert_eq!(d.readiness.stage, SkyReadinessStage::Submission);

        h.observe_submission(9, 0b0001, true);
        let d = h.diagnostics();
        assert_eq!(d.readiness.status, SkyReadinessStatus::Ready);
        let reasons = d.views[0].expect("view 0").reset_reasons;
        assert!(reasons.contains(CloudResetReasons::CAMERA_DISCONTINUITY));
        assert!(reasons.contains(CloudResetReasons::MODEL));
        assert!(reasons.contains(CloudResetReasons::WEATHER));

        h.remove_view(0).expect("remove view");
        assert_eq!(h.diagnostics().views[0], None);
        assert_eq!(h.remove_view(4).expect_err("index").code(), "sky.invalid");
    }

    #[test]
    fn acknowledged_generations_require_recorded_and_submitted_work() {
        let h = handle();
        let camera = Camera::default();
        h.set_view(0, &camera, true).expect("view 0");
        {
            let mut s = h.state.borrow_mut();
            s.recorded_invocation = Some(3);
            s.invocation = Some(3);
            s.pending_atmosphere = true;
            s.milky_way_pending = true;
            let slot = s.views[0].as_mut().expect("slot");
            slot.pending_sky_view = true;
            slot.size = (1280, 720);
        }
        h.observe_submission(3, 0b0001, true);
        let d = h.diagnostics();
        assert_eq!(d.atmosphere_generation, 1);
        assert_eq!(d.views[0].expect("view 0").sky_view_generation, 1);
        assert_eq!(d.readiness.status, SkyReadinessStatus::Ready);
        assert_eq!(d.environment.requested_generation, 0, "no IBL handle");

        {
            let mut s = h.state.borrow_mut();
            s.pending_atmosphere = true;
            s.milky_way_pending = true;
            s.views[0].as_mut().expect("slot").pending_sky_view = true;
        }
        h.observe_submission(4, 0b0001, false);
        let d = h.diagnostics();
        assert_eq!(
            d.atmosphere_generation, 2,
            "an actually submitted view is acknowledged even when the invocation is partial"
        );
        assert_eq!(d.views[0].expect("view 0").sky_view_generation, 2);
        assert_eq!(d.readiness.status, SkyReadinessStatus::Pending);
        assert!(!d.views[0].expect("view 0").history_valid);

        h.observe_submission(5, 0b0001, true);
        let d = h.diagnostics();
        assert_eq!(d.atmosphere_generation, 2, "no double acknowledgement");
        assert_eq!(d.views[0].expect("view 0").sky_view_generation, 2);
        assert_eq!(d.readiness.status, SkyReadinessStatus::Ready);
    }

    #[test]
    fn a_failure_is_retained_until_a_new_preparation_or_submission() {
        let h = handle();
        {
            let mut s = h.state.borrow_mut();
            s.readiness = SkyReadiness::failed(SkyReadinessStage::CloudScreen, "gpu.device_lost");
        }
        let d = h.diagnostics();
        assert_eq!(d.readiness.status, SkyReadinessStatus::Failed);
        assert_eq!(d.readiness.stage, SkyReadinessStage::CloudScreen);
        assert_eq!(d.readiness.error.as_deref(), Some("gpu.device_lost"));

        h.set_view(0, &Camera::default(), true)
            .expect("new preparation");
        assert_eq!(
            h.diagnostics().readiness.status,
            SkyReadinessStatus::Pending
        );
        h.observe_submission(1, 0b0001, true);
        assert_eq!(h.diagnostics().readiness.status, SkyReadinessStatus::Ready);
    }

    #[test]
    fn wind_drift_is_rotated_by_the_place_north() {
        let h = handle();
        let mut settings = weather_settings(false, Weather::default());
        settings.target.wind_from = 270.0;
        settings.target.wind_speed = 10.0;
        h.set_weather(settings).expect("windy weather");
        let mut time = crate::time_of_day::TimeOfDaySettings {
            enabled: false,
            playing: false,
            ..Default::default()
        };
        time.place.north = std::f32::consts::FRAC_PI_2;
        h.set_time_of_day(time).expect("turn the place");
        h.advance_frame(1, 0.0).expect("first invocation");
        h.advance_frame(2, 1.0).expect("advance");
        let [east, south] = h.state.borrow().weather.wind_offset;
        let east = if east > 126_000.0 {
            east - 252_000.0
        } else {
            east
        };
        assert!(east.abs() < 1.0e-3, "{east}");
        let expected = 252_000.0 - 10.0 * 0.25;
        assert!(
            (south - expected).abs() < 1.0e-3,
            "the wind turned into +Z: {south}"
        );
    }
}
