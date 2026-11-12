//! The neutral cloud backend and the M2 backend seam (S25).
//!
//! [`CloudFrame`] is the copied per-view snapshot the sky feature hands a backend;
//! [`CloudViews`] are the panorama, screen and two-layer transmittance views the
//! consumers sample; [`CloudInvalidation`] names the events that discard a view's
//! cloud history. The object-safe, crate-private [`CloudRenderer`] is the seam:
//! M1 registers [`NeutralCloudRenderer`] and the M2 producer file `src/clouds/mod.rs`
//! contributes its own `create` through [`create_cloud_backend`] when it exists.
//!
//! Panorama uses the atmosphere sky-view direction mapping; screen uses viewport UV.
//! Panorama and screen are filterable D2 Rgba16Float views whose RGB is linear HDR
//! cloud radiance and whose A is the scalar transmittance; the composite is
//! `clear_sky * A + RGB`, so the neutral texel `(0, 0, 0, 1)` leaves the clear sky
//! untouched. Transmittance is the S10 two-layer Rgba16Float array (layer 0 the
//! ground map, layer 1 the mean at texel (0, 0)); the neutral array is
//! `(1, 1, 1, 1)` in both layers. The M1 backend always reports both validity flags
//! false and keeps no volumetric history, so initialized dummies never assert
//! completed volume work.
//!
//! The screen shader's `-1.0` alpha is reserved for texels M2 did not march
//! (`UNKNOWN` in Solstice `shaders/clouds/screen.wgsl:24-26` at revision
//! 7e465b599c13da165948ebaa44df824c36c3ae07); it must never enter composition as
//! a negative transmittance. Consumers fall back to a valid panorama or the
//! neutral cloud contribution. [`CLOUD_SCREEN_UNKNOWN_ALPHA`] keeps that sentinel
//! reserved for M2.

/* Adapted channel/mapping contract: solstice@7e465b5.
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

use raa_core::{GpuContext, RaaError};
use raa_math::Mat4;
use raa_pipeline::{FeaturePass, LightingResources, PrepareCtx, RecordCtx};
use raa_scene::{Camera, DirectionalLight};

use crate::atmosphere::{AtmosphereResources, SharedAtmosphere};
use crate::desc::SkyModel;
use crate::observations::{CloudResetReasons, SkyReadiness, SkyReadinessStage};
use crate::state::{SharedSky, SkyState};
use crate::time_of_day::CelestialSky;
use crate::weather::Weather;

/// The shared backend handle (the feature and both cloud passes share one backend).
pub(crate) type SharedCloud = Rc<RefCell<Box<dyn CloudRenderer>>>;

/// S12 name of the cloud lighting pass.
pub(crate) const CLOUD_LIGHTING_NAME: &str = "sky.cloud-lighting";
/// S12 order of the cloud lighting pass.
pub(crate) const CLOUD_LIGHTING_ORDER: u32 = 8;
/// S12 name of the cloud screen pass.
pub(crate) const CLOUD_SCREEN_NAME: &str = "sky.cloud-screen";
/// S12 order of the cloud screen pass.
pub(crate) const CLOUD_SCREEN_ORDER: u32 = 55;

/// View slots the sky feature tracks; matches the atmosphere resource set.
const MAX_VIEWS: usize = 4;

/// The reserved negative screen-alpha sentinel for un-marched M2 texels.
pub const CLOUD_SCREEN_UNKNOWN_ALPHA: f32 = -1.0;

/// Build-time result of the `build.rs` discovery of the declared M2 source.
///
/// True only when `src/clouds/mod.rs` existed when this crate was compiled.
pub const M2_CLOUD_SOURCE_PRESENT: bool = cfg!(raa_m2_clouds);

/// Registry name of the completed M1 neutral/legacy backend.
pub(crate) const NEUTRAL_BACKEND: &str = "neutral";
/// Registry name of the declared M2 producer backend.
pub(crate) const M2_BACKEND: &str = "m2";

/// The copied per-view snapshot a cloud backend prepares and records from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudFrame {
    /// Invocation ID shared by every view of one Engine invocation.
    pub invocation: u64,
    /// View index, `0..MAX_VIEWS`.
    pub view: u32,
    /// Physical view size in pixels.
    pub size: (u32, u32),
    /// The view's camera.
    pub camera: Camera,
    /// Previous invocation's unjittered view-projection, for history reprojection.
    pub previous_view_proj: Mat4,
    /// Whether the previous invocation's history may be reprojected.
    pub history_valid: bool,
    /// The authored/evolving weather snapshot.
    pub weather: Weather,
    /// The accepted celestial snapshot.
    pub sky: CelestialSky,
    /// The accepted dominant directional light.
    pub dominant: DirectionalLight,
    /// Animation seconds since the feature started, for evolution and jitter.
    pub animation_seconds: f64,
    /// Accepted wind drift in metres (x = east, z = south, rotated by `Place.north`,
    /// wrapped to 252000).
    pub wind_offset: [f32; 2],
    /// Current weather wind velocity in metres per second, same axes, rotated by the
    /// accepted `Place.north`.
    pub wind_velocity: [f32; 2],
    /// Accepted cloud-boil phase in metres, wrapped to 7000.
    pub evolution: f32,
    /// The accepted `WeatherSettings.enabled` flag.
    pub weather_enabled: bool,
    /// The accepted `WeatherSettings.shadows` flag.
    pub cloud_shadows: bool,
    /// True when the accepted sky model is `SkyModel::Physical`.
    pub physical: bool,
}

/// The panorama, screen and transmittance views a cloud backend exposes.
///
/// Clone, not Copy: the fields own `wgpu::TextureView` handles. `screen_valid` and
/// `panorama_valid` are false whenever the backend holds no completed volume work,
/// so a disabled, model-switched or invalidated resource never exposes stale
/// validity.
#[derive(Clone, Debug)]
pub struct CloudViews {
    /// Filterable D2 Rgba16Float panorama: RGB radiance, A transmittance.
    pub panorama: wgpu::TextureView,
    /// Filterable D2 Rgba16Float screen result: RGB radiance, A transmittance.
    pub screen: wgpu::TextureView,
    /// Filterable D2Array Rgba16Float transmittance: layer 0 map, layer 1 mean.
    pub transmittance: wgpu::TextureView,
    /// True only when `screen` holds a completed march for this view.
    pub screen_valid: bool,
    /// True only when `panorama` holds a completed march for this view.
    pub panorama_valid: bool,
}

impl CloudViews {
    /// The neutral 1x1 views: panorama and screen radiance `(0, 0, 0, 1)`, and a
    /// two-layer transmittance array of `(1, 1, 1, 1)`. Both validity flags are false.
    pub fn neutral(gpu: &GpuContext) -> Result<CloudViews, RaaError> {
        gpu.ensure_live()?;
        Ok(CloudViews {
            panorama: neutral_radiance(gpu, "raa-sky.cloud.panorama")?,
            screen: neutral_radiance(gpu, "raa-sky.cloud.screen")?,
            transmittance: neutral_transmittance(gpu)?,
            screen_valid: false,
            panorama_valid: false,
        })
    }
}

/// An event that discards a view's cloud history.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CloudInvalidation {
    /// The view was resized.
    Resize,
    /// The camera jumped (cut, teleport, projection discontinuity).
    CameraDiscontinuity,
    /// The weather state changed.
    Weather,
    /// The sky model switched between physical and legacy.
    Model,
    /// The cloud feature was enabled after being disabled.
    Enable,
    /// The view was removed.
    ViewRemoved,
}

/// The crate-private object-safe backend seam.
///
/// `prepare` runs before recording and may publish resources or upload uniforms;
/// `record_lighting` and `record_screen` append their passes to the frame encoder;
/// `views` hands out the sampled views; `invalidate` discards a view's history (or
/// every view's for `None`); `submitted` acknowledges an actual pipeline submission.
/// No backend owns the Engine clock or the P1 lights.
pub(crate) trait CloudRenderer {
    fn prepare(
        &mut self,
        gpu: &GpuContext,
        frame: &CloudFrame,
        atmosphere: &AtmosphereResources,
        lighting: &mut LightingResources,
    ) -> Result<(), RaaError>;

    fn record_lighting(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        frame: &CloudFrame,
        atmosphere: &AtmosphereResources,
    ) -> Result<(), RaaError>;

    fn record_screen(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        frame: &CloudFrame,
        depth: &wgpu::TextureView,
        atmosphere: &AtmosphereResources,
    ) -> Result<(), RaaError>;

    fn views(&self, view: u32) -> CloudViews;

    fn invalidate(&mut self, view: Option<u32>, reason: CloudInvalidation);

    fn submitted(&mut self, view: u32, invocation: u64);
}

/// The copied snapshot of the active view, or `sky.invalid` when it is not prepared.
fn cloud_frame(
    state: &SkyState,
    invocation: u64,
    view: u32,
    size: (u32, u32),
    time_s: f32,
) -> Result<CloudFrame, RaaError> {
    let slot = state
        .views
        .get(view as usize)
        .and_then(Option::as_ref)
        .ok_or_else(|| invalid(format!("cloud view {view} is not prepared")))?;
    let [east, south] = state.weather.current.wind_velocity();
    let (sin, cos) = state.time.state.settings.place.north.sin_cos();
    Ok(CloudFrame {
        invocation,
        view,
        size,
        camera: slot.camera,
        previous_view_proj: slot.previous_view_proj,
        history_valid: slot.history_valid,
        weather: state.weather.current,
        sky: state.time.state.sky,
        dominant: state.time.last_light,
        animation_seconds: f64::from(time_s),
        wind_offset: state.weather.wind_offset,
        wind_velocity: [east * cos + south * sin, -east * sin + south * cos],
        evolution: state.weather.evolution,
        weather_enabled: state.weather.settings.enabled,
        cloud_shadows: state.weather.settings.shadows,
        physical: state.desc.model == SkyModel::Physical,
    })
}

fn state_busy(name: &str) -> RaaError {
    RaaError::Resource(format!("{name}: sky state is busy"))
}

fn resources_busy(name: &str) -> RaaError {
    RaaError::Resource(format!("{name}: resources are borrowed"))
}

/// The `sky.cloud-lighting` pass (S12 order 8): prepares the backend for the active view and
/// records its panorama/transmittance work after the atmosphere LUTs are available.
pub(crate) struct SkyCloudLightingPass {
    state: SharedSky,
    cloud: SharedCloud,
    atmosphere: SharedAtmosphere,
    enabled: bool,
    /// Last physical size prepared per view, for backend resize invalidation.
    sizes: [Option<(u32, u32)>; MAX_VIEWS],
}

impl SkyCloudLightingPass {
    pub(crate) fn new(state: SharedSky, cloud: SharedCloud, atmosphere: SharedAtmosphere) -> Self {
        SkyCloudLightingPass {
            state,
            cloud,
            atmosphere,
            enabled: true,
            sizes: [None; MAX_VIEWS],
        }
    }
}

impl FeaturePass for SkyCloudLightingPass {
    fn name(&self) -> &str {
        CLOUD_LIGHTING_NAME
    }

    fn order(&self) -> u32 {
        CLOUD_LIGHTING_ORDER
    }

    /// Disabling publishes the neutral transmittance again and, only on the enabled-to-disabled
    /// transition, invalidates the backend; re-enabling discards the cloud history and
    /// records the `Enable` reset reason.
    fn set_enabled(&mut self, enabled: bool, lighting: &mut LightingResources) {
        if enabled && !self.enabled {
            if let Ok(mut cloud) = self.cloud.try_borrow_mut() {
                cloud.invalidate(None, CloudInvalidation::Enable);
            }
            if let Ok(mut s) = self.state.try_borrow_mut() {
                s.reset_reasons(None, CloudResetReasons::ENABLE);
                s.readiness = SkyReadiness::pending(SkyReadinessStage::CloudLighting);
            }
        }
        if !enabled
            && self.enabled
            && let Ok(mut cloud) = self.cloud.try_borrow_mut()
        {
            cloud.invalidate(None, CloudInvalidation::Enable);
        }
        self.enabled = enabled;
        if !enabled {
            lighting.clear_cloud_transmittance();
        }
    }

    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let frame = {
            let s = self
                .state
                .try_borrow()
                .map_err(|_| state_busy(CLOUD_LIGHTING_NAME))?;
            let Some(view) = s.active_view else {
                return Ok(());
            };
            cloud_frame(&s, ctx.frame, view, ctx.size, ctx.time_s)?
        };
        let index = frame.view as usize;
        let resized = self.sizes[index].is_some_and(|size| size != ctx.size);
        self.sizes[index] = Some(ctx.size);
        let atmosphere = self
            .atmosphere
            .try_borrow()
            .map_err(|_| resources_busy(CLOUD_LIGHTING_NAME))?;
        let mut backend = self
            .cloud
            .try_borrow_mut()
            .map_err(|_| resources_busy(CLOUD_LIGHTING_NAME))?;
        if resized {
            backend.invalidate(Some(frame.view), CloudInvalidation::Resize);
        }
        let prepared = backend.prepare(ctx.gpu, &frame, &atmosphere, ctx.lighting);
        drop(backend);
        drop(atmosphere);
        match prepared {
            Ok(()) => {
                if let Ok(mut s) = self.state.try_borrow_mut() {
                    s.readiness = SkyReadiness::pending(SkyReadinessStage::CloudLighting);
                }
                Ok(())
            }
            Err(error) => {
                if let Ok(mut s) = self.state.try_borrow_mut() {
                    s.readiness =
                        SkyReadiness::failed(SkyReadinessStage::CloudLighting, error.to_string());
                }
                Err(error)
            }
        }
    }

    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let frame = {
            let s = self
                .state
                .try_borrow()
                .map_err(|_| state_busy(CLOUD_LIGHTING_NAME))?;
            let Some(view) = s.active_view else {
                return Ok(());
            };
            cloud_frame(&s, ctx.frame, view, ctx.size, ctx.time_s)?
        };
        let atmosphere = self
            .atmosphere
            .try_borrow()
            .map_err(|_| resources_busy(CLOUD_LIGHTING_NAME))?;
        let recorded = self
            .cloud
            .try_borrow_mut()
            .map_err(|_| resources_busy(CLOUD_LIGHTING_NAME))?
            .record_lighting(ctx.gpu, ctx.encoder, &frame, &atmosphere);
        drop(atmosphere);
        match recorded {
            Ok(()) => {
                if let Ok(mut s) = self.state.try_borrow_mut() {
                    s.readiness = SkyReadiness::pending(SkyReadinessStage::CloudLighting);
                }
                Ok(())
            }
            Err(error) => {
                if let Ok(mut s) = self.state.try_borrow_mut() {
                    s.readiness =
                        SkyReadiness::failed(SkyReadinessStage::CloudLighting, error.to_string());
                }
                Err(error)
            }
        }
    }
}

/// The `sky.cloud-screen` pass (S12 order 55): records the backend's screen march after the
/// opaque prepass and before the forward pass samples it.
pub(crate) struct SkyCloudScreenPass {
    state: SharedSky,
    cloud: SharedCloud,
    atmosphere: SharedAtmosphere,
}

impl SkyCloudScreenPass {
    pub(crate) fn new(state: SharedSky, cloud: SharedCloud, atmosphere: SharedAtmosphere) -> Self {
        SkyCloudScreenPass {
            state,
            cloud,
            atmosphere,
        }
    }
}

impl FeaturePass for SkyCloudScreenPass {
    fn name(&self) -> &str {
        CLOUD_SCREEN_NAME
    }

    fn order(&self) -> u32 {
        CLOUD_SCREEN_ORDER
    }

    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let frame = {
            let s = self
                .state
                .try_borrow()
                .map_err(|_| state_busy(CLOUD_SCREEN_NAME))?;
            let Some(view) = s.active_view else {
                return Ok(());
            };
            cloud_frame(&s, ctx.frame, view, ctx.size, ctx.time_s)?
        };
        let atmosphere = self
            .atmosphere
            .try_borrow()
            .map_err(|_| resources_busy(CLOUD_SCREEN_NAME))?;
        let recorded = self
            .cloud
            .try_borrow_mut()
            .map_err(|_| resources_busy(CLOUD_SCREEN_NAME))?
            .record_screen(
                ctx.gpu,
                ctx.encoder,
                &frame,
                ctx.targets.depth(),
                &atmosphere,
            );
        drop(atmosphere);
        match recorded {
            Ok(()) => {
                if let Ok(mut s) = self.state.try_borrow_mut() {
                    s.readiness = SkyReadiness::pending(SkyReadinessStage::CloudScreen);
                }
                Ok(())
            }
            Err(error) => {
                if let Ok(mut s) = self.state.try_borrow_mut() {
                    s.readiness =
                        SkyReadiness::failed(SkyReadinessStage::CloudScreen, error.to_string());
                }
                Err(error)
            }
        }
    }
}

/// The completed M1 neutral/legacy backend: valid dummies, no volume work.
pub(crate) struct NeutralCloudRenderer {
    views: CloudViews,
    /// Published the neutral transmittance/map/light once per enabled resource set.
    neutral_published: bool,
    /// Last invalidation accepted per view, for diagnostics and tests.
    invalidations: [Option<CloudInvalidation>; MAX_VIEWS],
    /// Last submitted invocation acknowledged per view.
    submitted: [Option<u64>; MAX_VIEWS],
}

impl NeutralCloudRenderer {
    /// The backend factory registered under [`NEUTRAL_BACKEND`].
    pub(crate) fn create(gpu: &GpuContext) -> Result<Box<dyn CloudRenderer>, RaaError> {
        Ok(Box::new(NeutralCloudRenderer {
            views: CloudViews::neutral(gpu)?,
            neutral_published: false,
            invalidations: [None; MAX_VIEWS],
            submitted: [None; MAX_VIEWS],
        }))
    }

    /// The last invalidation accepted for a prepared view.
    pub(crate) fn last_invalidation(&self, view: u32) -> Option<CloudInvalidation> {
        self.invalidations.get(view as usize).copied().flatten()
    }

    /// The last submission acknowledged for a prepared view.
    pub(crate) fn last_submitted(&self, view: u32) -> Option<u64> {
        self.submitted.get(view as usize).copied().flatten()
    }

    /// Validates the frame's view index and physical size.
    fn validate(frame: &CloudFrame) -> Result<(), RaaError> {
        if frame.view as usize >= MAX_VIEWS {
            return Err(invalid(format!(
                "cloud view {} is outside 0..{MAX_VIEWS}",
                frame.view
            )));
        }
        if frame.size.0 == 0 || frame.size.1 == 0 {
            return Err(invalid("cloud view size must be non-zero"));
        }
        Ok(())
    }
}

impl CloudRenderer for NeutralCloudRenderer {
    /// Validates, then publishes the neutral transmittance/map/light once per enabled
    /// resource set. The M1 neutral backend allocates nothing per steady frame.
    fn prepare(
        &mut self,
        gpu: &GpuContext,
        frame: &CloudFrame,
        _atmosphere: &AtmosphereResources,
        lighting: &mut LightingResources,
    ) -> Result<(), RaaError> {
        gpu.ensure_live()?;
        Self::validate(frame)?;
        if !self.neutral_published {
            lighting.set_cloud_transmittance(
                self.views.transmittance.clone(),
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            )?;
            self.neutral_published = true;
        }
        Ok(())
    }

    /// The neutral backend marches no volume, so there is no lighting work to record.
    fn record_lighting(
        &mut self,
        gpu: &GpuContext,
        _encoder: &mut wgpu::CommandEncoder,
        frame: &CloudFrame,
        _atmosphere: &AtmosphereResources,
    ) -> Result<(), RaaError> {
        gpu.ensure_live()?;
        Self::validate(frame)
    }

    /// The neutral backend marches no volume, so there is no screen work to record.
    fn record_screen(
        &mut self,
        gpu: &GpuContext,
        _encoder: &mut wgpu::CommandEncoder,
        frame: &CloudFrame,
        _depth: &wgpu::TextureView,
        _atmosphere: &AtmosphereResources,
    ) -> Result<(), RaaError> {
        gpu.ensure_live()?;
        Self::validate(frame)
    }

    /// Cloned neutral views with false flags for every index, present or not.
    fn views(&self, _view: u32) -> CloudViews {
        self.views.clone()
    }

    /// Records the invalidation and republishes the neutral resources on a model or
    /// enable change. The M1 backend has no history to discard.
    fn invalidate(&mut self, view: Option<u32>, reason: CloudInvalidation) {
        match view {
            Some(view) => {
                if let Some(slot) = self.invalidations.get_mut(view as usize) {
                    *slot = Some(reason);
                }
            }
            None => self.invalidations = [Some(reason); MAX_VIEWS],
        }
        if matches!(reason, CloudInvalidation::Model | CloudInvalidation::Enable) {
            self.neutral_published = false;
        }
    }

    /// Acknowledges an actual submission; recording alone establishes nothing.
    fn submitted(&mut self, view: u32, invocation: u64) {
        if let Some(slot) = self.submitted.get_mut(view as usize) {
            *slot = Some(invocation);
        }
    }
}

/// A backend factory: builds a backend or refuses with `sky.invalid`.
pub(crate) type CloudBackendFactory = fn(&GpuContext) -> Result<Box<dyn CloudRenderer>, RaaError>;

/// The registration list. A duplicate name is refused so two producers cannot
/// silently claim the same backend.
pub(crate) struct CloudBackendRegistry {
    backends: Vec<(&'static str, CloudBackendFactory)>,
}

impl CloudBackendRegistry {
    pub(crate) fn new() -> Self {
        CloudBackendRegistry {
            backends: Vec::new(),
        }
    }

    /// Adds a backend, refusing a name already registered.
    pub(crate) fn register(
        &mut self,
        name: &'static str,
        factory: CloudBackendFactory,
    ) -> Result<(), RaaError> {
        if self.backends.iter().any(|(existing, _)| *existing == name) {
            return Err(invalid(format!(
                "cloud backend `{name}` is already registered"
            )));
        }
        self.backends.push((name, factory));
        Ok(())
    }

    /// The registered names in registration order.
    pub(crate) fn names(&self) -> Vec<&'static str> {
        self.backends.iter().map(|(name, _)| *name).collect()
    }

    /// Builds the preferred backend: the M2 producer when registered, else neutral.
    pub(crate) fn create(&self, gpu: &GpuContext) -> Result<Box<dyn CloudRenderer>, RaaError> {
        let factory = self
            .backends
            .iter()
            .find(|(name, _)| *name == M2_BACKEND)
            .or_else(|| {
                self.backends
                    .iter()
                    .find(|(name, _)| *name == NEUTRAL_BACKEND)
            })
            .map(|(_, factory)| factory)
            .ok_or_else(|| invalid("no cloud backend is registered"))?;
        factory(gpu)
    }
}

// Registers neutral always and M2 only when its declared source exists.
include!(concat!(env!("OUT_DIR"), "/cloud_backend_registry.rs"));

/// Builds the backend for this build: the M2 producer when present, else neutral.
pub(crate) fn create_cloud_backend(gpu: &GpuContext) -> Result<Box<dyn CloudRenderer>, RaaError> {
    let mut registry = CloudBackendRegistry::new();
    register_builtin(&mut registry)?;
    registry.create(gpu)
}

/// The backend names a build with the given M2 discovery result would register.
///
/// Pure decision helper: it tests both the absent and present outcomes without
/// writing or reading any M2 source file.
pub fn cloud_backend_names(m2_source_present: bool) -> Vec<&'static str> {
    let mut names = vec![NEUTRAL_BACKEND];
    if m2_source_present {
        names.push(M2_BACKEND);
    }
    names
}

/// The backend names actually registered in this build.
pub fn registered_cloud_backend_names() -> Vec<&'static str> {
    let mut registry = CloudBackendRegistry::new();
    register_builtin(&mut registry).expect("built-in cloud backend registration is unique");
    registry.names()
}

/// A 1x1 Rgba16Float texture of `texel` in every one of `layers` layers.
fn neutral_texture(
    gpu: &GpuContext,
    label: &str,
    layers: u32,
    texel: &[f32; 4],
) -> Result<wgpu::Texture, RaaError> {
    gpu.ensure_live()?;
    let size = wgpu::Extent3d {
        width: 1,
        height: 1,
        depth_or_array_layers: layers,
    };
    let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let bytes = f16_texel(texel).repeat(layers as usize);
    gpu.queue().write_texture(
        texture.as_image_copy(),
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some((texel.len() * 2) as u32),
            rows_per_image: Some(1),
        },
        size,
    );
    Ok(texture)
}

/// A neutral D2 radiance/transmittance view with texel `(0, 0, 0, 1)`.
fn neutral_radiance(gpu: &GpuContext, label: &str) -> Result<wgpu::TextureView, RaaError> {
    let texture = neutral_texture(gpu, label, 1, &[0.0, 0.0, 0.0, 1.0])?;
    Ok(texture.create_view(&wgpu::TextureViewDescriptor::default()))
}

/// The neutral two-layer transmittance array with `(1, 1, 1, 1)` in every texel.
fn neutral_transmittance(gpu: &GpuContext) -> Result<wgpu::TextureView, RaaError> {
    let texture = neutral_texture(gpu, "raa-sky.cloud.transmittance", 2, &[1.0, 1.0, 1.0, 1.0])?;
    Ok(texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    }))
}

/// f16 bits of a finite f32 (round to nearest even; values below the f16 normal range
/// flush to zero, values above it become infinity). Used only to fill the dummies.
fn f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    if exponent <= 0 {
        return sign;
    }
    if exponent >= 31 {
        return sign | 0x7c00;
    }
    let mut half = ((exponent as u32) << 10) | (mantissa >> 13);
    let rest = mantissa & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && half & 1 == 1) {
        half += 1;
    }
    sign | half as u16
}

/// Little-endian f16 texel bytes.
fn f16_texel(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| f16_bits(*v).to_le_bytes())
        .collect()
}

/// `sky.invalid`: an invalid cloud frame or registration, naming the field.
fn invalid(message: impl Into<String>) -> RaaError {
    RaaError::FrameGraph {
        code: "sky.invalid",
        message: message.into(),
    }
}

/// Compile-time contract of the M2 backend seam.
///
/// Card 7434 (sky passes) calls exactly these items. Pinning them keeps the
/// crate-private seam reachable for the `dead_code` lint until that consumer lands
/// and refuses an accidental interface drift in between.
const _: () = {
    type Factory = fn(&GpuContext) -> Result<Box<dyn CloudRenderer>, RaaError>;
    type Prepare = fn(
        &mut NeutralCloudRenderer,
        &GpuContext,
        &CloudFrame,
        &AtmosphereResources,
        &mut LightingResources,
    ) -> Result<(), RaaError>;
    type RecordLighting = fn(
        &mut NeutralCloudRenderer,
        &GpuContext,
        &mut wgpu::CommandEncoder,
        &CloudFrame,
        &AtmosphereResources,
    ) -> Result<(), RaaError>;
    type RecordScreen = fn(
        &mut NeutralCloudRenderer,
        &GpuContext,
        &mut wgpu::CommandEncoder,
        &CloudFrame,
        &wgpu::TextureView,
        &AtmosphereResources,
    ) -> Result<(), RaaError>;
    type Views = fn(&NeutralCloudRenderer, u32) -> CloudViews;
    type Invalidate = fn(&mut NeutralCloudRenderer, Option<u32>, CloudInvalidation);
    type Submitted = fn(&mut NeutralCloudRenderer, u32, u64);
    type LastInvalidation = fn(&NeutralCloudRenderer, u32) -> Option<CloudInvalidation>;
    type LastSubmitted = fn(&NeutralCloudRenderer, u32) -> Option<u64>;
    let _: Factory = create_cloud_backend;
    let _: Factory = NeutralCloudRenderer::create;
    let _: Prepare = NeutralCloudRenderer::prepare;
    let _: RecordLighting = NeutralCloudRenderer::record_lighting;
    let _: RecordScreen = NeutralCloudRenderer::record_screen;
    let _: Views = NeutralCloudRenderer::views;
    let _: Invalidate = NeutralCloudRenderer::invalidate;
    let _: Submitted = NeutralCloudRenderer::submitted;
    let _: LastInvalidation = NeutralCloudRenderer::last_invalidation;
    let _: LastSubmitted = NeutralCloudRenderer::last_submitted;
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time_of_day::TimeOfDayState;

    fn frame(view: u32) -> CloudFrame {
        CloudFrame {
            invocation: 7,
            view,
            size: (1280, 720),
            camera: Camera::default(),
            previous_view_proj: Mat4::IDENTITY,
            history_valid: true,
            weather: Weather::default(),
            sky: TimeOfDayState::default().sky,
            dominant: DirectionalLight::default(),
            animation_seconds: 3.5,
            wind_offset: [0.0; 2],
            wind_velocity: [0.0; 2],
            evolution: 0.0,
            weather_enabled: true,
            cloud_shadows: true,
            physical: true,
        }
    }

    fn state_with_view() -> SkyState {
        let mut state = SkyState::default();
        state.views[0] = Some(crate::state::ViewSlot::new(0, Camera::default(), true));
        state
    }

    #[test]
    fn cloud_frame_copies_drift_evolution_flags_and_model() {
        let mut state = state_with_view();
        state.weather.wind_offset = [12.0, -3.0];
        state.weather.evolution = 42.0;
        state.weather.settings.enabled = false;
        state.weather.settings.shadows = false;
        state.desc.model = SkyModel::Legacy;
        let frame = cloud_frame(&state, 1, 0, (64, 64), 0.0).expect("frame");
        assert_eq!(frame.wind_offset, [12.0, -3.0]);
        assert_eq!(frame.evolution, 42.0);
        assert!(!frame.weather_enabled && !frame.cloud_shadows && !frame.physical);
    }

    #[test]
    fn cloud_frame_rotates_wind_velocity_by_north() {
        let mut state = state_with_view();
        state.weather.current.wind_speed = 9.0;
        state.weather.current.wind_from = 250.0;
        state.time.state.settings.place.north = std::f32::consts::FRAC_PI_2;
        let [east, south] = state.weather.current.wind_velocity();
        let frame = cloud_frame(&state, 1, 0, (64, 64), 0.0).expect("frame");
        assert!((frame.wind_velocity[0] - south).abs() < 1e-5);
        assert!((frame.wind_velocity[1] + east).abs() < 1e-5);
    }

    type InvalidationLog = Rc<RefCell<Vec<(Option<u32>, CloudInvalidation)>>>;

    /// Records the invalidations it receives; every other method is inert.
    struct Recording {
        views: CloudViews,
        log: InvalidationLog,
    }

    impl CloudRenderer for Recording {
        fn prepare(
            &mut self,
            _: &GpuContext,
            _: &CloudFrame,
            _: &AtmosphereResources,
            _: &mut LightingResources,
        ) -> Result<(), RaaError> {
            Ok(())
        }

        fn record_lighting(
            &mut self,
            _: &GpuContext,
            _: &mut wgpu::CommandEncoder,
            _: &CloudFrame,
            _: &AtmosphereResources,
        ) -> Result<(), RaaError> {
            Ok(())
        }

        fn record_screen(
            &mut self,
            _: &GpuContext,
            _: &mut wgpu::CommandEncoder,
            _: &CloudFrame,
            _: &wgpu::TextureView,
            _: &AtmosphereResources,
        ) -> Result<(), RaaError> {
            Ok(())
        }

        fn views(&self, _: u32) -> CloudViews {
            self.views.clone()
        }

        fn invalidate(&mut self, view: Option<u32>, reason: CloudInvalidation) {
            self.log.borrow_mut().push((view, reason));
        }

        fn submitted(&mut self, _: u32, _: u64) {}
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn disabling_cloud_lighting_invalidates_the_backend() {
        let _gpu = crate::gpu_test_guard();
        let Some(gpu) = raa_core::testing::software_gpu() else {
            raa_core::testing::skip(
                "disabling_cloud_lighting_invalidates_the_backend",
                "software_gpu returned None; see the software_gpu line above",
            );
            return;
        };
        let log = Rc::new(RefCell::new(Vec::new()));
        let backend: Box<dyn CloudRenderer> = Box::new(Recording {
            views: CloudViews::neutral(&gpu).expect("neutral views"),
            log: log.clone(),
        });
        let state: SharedSky = Rc::new(RefCell::new(SkyState::default()));
        let atmosphere = Rc::new(RefCell::new(
            AtmosphereResources::new(&gpu).expect("atmosphere resources"),
        ));
        let mut pass =
            SkyCloudLightingPass::new(state.clone(), Rc::new(RefCell::new(backend)), atmosphere);
        let mut lighting = LightingResources::new(&gpu);
        let readiness = state.borrow().readiness.clone();
        let once = vec![(None, CloudInvalidation::Enable)];
        pass.set_enabled(false, &mut lighting);
        assert_eq!(*log.borrow(), once);
        assert_eq!(state.borrow().readiness, readiness);
        pass.set_enabled(false, &mut lighting);
        assert_eq!(*log.borrow(), once, "a second disable adds nothing");
        pass.set_enabled(true, &mut lighting);
        assert_eq!(log.borrow().len(), 2);
        assert_eq!(log.borrow()[1], (None, CloudInvalidation::Enable));
    }

    #[test]
    fn f16_neutral_texels_match_known_values() {
        assert_eq!(f16_bits(0.0), 0x0000);
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(-1.0), 0xbc00);
        assert_eq!(
            f16_texel(&[0.0, 0.0, 0.0, 1.0]),
            [0, 0, 0, 0, 0, 0, 0, 0x3c]
        );
    }

    #[test]
    fn registry_decisions_match_discovery() {
        assert_eq!(cloud_backend_names(false), vec![NEUTRAL_BACKEND]);
        assert_eq!(cloud_backend_names(true), vec![NEUTRAL_BACKEND, M2_BACKEND]);
        assert_eq!(
            registered_cloud_backend_names(),
            cloud_backend_names(M2_CLOUD_SOURCE_PRESENT),
            "the registered set follows the build-time discovery"
        );
    }

    #[test]
    fn duplicate_registration_rejects() {
        let mut registry = CloudBackendRegistry::new();
        registry
            .register(NEUTRAL_BACKEND, NeutralCloudRenderer::create)
            .expect("first registration");
        let error = registry
            .register(NEUTRAL_BACKEND, NeutralCloudRenderer::create)
            .expect_err("a duplicate name is refused");
        assert_eq!(error.code(), "sky.invalid");
        assert_eq!(registry.names(), vec![NEUTRAL_BACKEND]);
    }

    #[test]
    fn empty_registry_has_no_names() {
        let registry = CloudBackendRegistry::new();
        assert_eq!(registry.names(), Vec::<&'static str>::new());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn neutral_views_are_initialized_and_invalid() {
        let _gpu = crate::gpu_test_guard();
        let Some(gpu) = raa_core::testing::software_gpu() else {
            raa_core::testing::skip(
                "neutral_views_are_initialized_and_invalid",
                "software_gpu returned None; see the software_gpu line above",
            );
            return;
        };
        let views = CloudViews::neutral(&gpu).expect("neutral views");
        assert!(!views.screen_valid && !views.panorama_valid);
        for (label, view) in [("panorama", &views.panorama), ("screen", &views.screen)] {
            let texture = view.texture();
            assert_eq!(
                texture.format(),
                wgpu::TextureFormat::Rgba16Float,
                "{label}"
            );
            assert_eq!((texture.width(), texture.height()), (1, 1), "{label}");
            assert_eq!(texture.dimension(), wgpu::TextureDimension::D2, "{label}");
        }
        let transmittance = views.transmittance.texture();
        assert_eq!(transmittance.format(), wgpu::TextureFormat::Rgba16Float);
        assert_eq!(
            transmittance.depth_or_array_layers(),
            2,
            "the S10 transmittance is a two-layer array"
        );
        assert_eq!(CLOUD_SCREEN_UNKNOWN_ALPHA, -1.0);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn missing_views_return_cloned_neutral_views() {
        let _gpu = crate::gpu_test_guard();
        let Some(gpu) = raa_core::testing::software_gpu() else {
            raa_core::testing::skip(
                "missing_views_return_cloned_neutral_views",
                "software_gpu returned None; see the software_gpu line above",
            );
            return;
        };
        let backend = NeutralCloudRenderer {
            views: CloudViews::neutral(&gpu).expect("neutral views"),
            neutral_published: false,
            invalidations: [None; MAX_VIEWS],
            submitted: [None; MAX_VIEWS],
        };
        let neutral = backend.views(0);
        assert!(!neutral.screen_valid);
        assert!(!neutral.panorama_valid);
        for view in [0, 1, 3, 4, 99] {
            let clone = backend.views(view);
            assert!(!clone.screen_valid && !clone.panorama_valid, "view {view}");
            assert!(
                clone.screen == neutral.screen,
                "view {view} reuses the one neutral screen texture"
            );
            assert!(
                clone.transmittance == neutral.transmittance,
                "view {view} reuses the one neutral transmittance array"
            );
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn prepare_and_recording_reject_invalid_indices() {
        let _gpu = crate::gpu_test_guard();
        let Some(gpu) = raa_core::testing::software_gpu() else {
            raa_core::testing::skip(
                "prepare_and_recording_reject_invalid_indices",
                "software_gpu returned None; see the software_gpu line above",
            );
            return;
        };
        let atmosphere = AtmosphereResources::new(&gpu).expect("atmosphere resources");
        let mut lighting = LightingResources::new(&gpu);
        let mut backend = NeutralCloudRenderer::create(&gpu).expect("neutral backend");
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let depth = CloudViews::neutral(&gpu).expect("views").screen;
        for (bad, name) in [
            (frame(MAX_VIEWS as u32), "out of range"),
            (
                CloudFrame {
                    size: (0, 720),
                    ..frame(0)
                },
                "zero size",
            ),
        ] {
            let error = backend
                .prepare(&gpu, &bad, &atmosphere, &mut lighting)
                .expect_err(name);
            assert_eq!(error.code(), "sky.invalid", "{name}");
            let error = backend
                .record_lighting(&gpu, &mut encoder, &bad, &atmosphere)
                .expect_err(name);
            assert_eq!(error.code(), "sky.invalid", "{name}");
            let error = backend
                .record_screen(&gpu, &mut encoder, &bad, &depth, &atmosphere)
                .expect_err(name);
            assert_eq!(error.code(), "sky.invalid", "{name}");
        }
        // A valid frame prepares (publishing the neutral transmittance once) and
        // records nothing, so repeating it is a steady-frame no-op.
        let valid = frame(0);
        for _ in 0..2 {
            backend
                .prepare(&gpu, &valid, &atmosphere, &mut lighting)
                .expect("valid prepare");
            backend
                .record_lighting(&gpu, &mut encoder, &valid, &atmosphere)
                .expect("valid lighting record");
            backend
                .record_screen(&gpu, &mut encoder, &valid, &depth, &atmosphere)
                .expect("valid screen record");
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn all_invalidations_are_accepted_and_leave_views_neutral() {
        let _gpu = crate::gpu_test_guard();
        let Some(gpu) = raa_core::testing::software_gpu() else {
            raa_core::testing::skip(
                "all_invalidations_are_accepted_and_leave_views_neutral",
                "software_gpu returned None; see the software_gpu line above",
            );
            return;
        };
        let mut backend = NeutralCloudRenderer {
            views: CloudViews::neutral(&gpu).expect("neutral views"),
            neutral_published: false,
            invalidations: [None; MAX_VIEWS],
            submitted: [None; MAX_VIEWS],
        };
        let reasons = [
            CloudInvalidation::Resize,
            CloudInvalidation::CameraDiscontinuity,
            CloudInvalidation::Weather,
            CloudInvalidation::Model,
            CloudInvalidation::Enable,
            CloudInvalidation::ViewRemoved,
        ];
        for reason in reasons {
            backend.invalidate(Some(0), reason);
            assert_eq!(backend.last_invalidation(0), Some(reason));
            backend.invalidate(None, reason);
            for view in 0..MAX_VIEWS {
                assert_eq!(backend.last_invalidation(view as u32), Some(reason));
                let views = backend.views(view as u32);
                assert!(!views.screen_valid && !views.panorama_valid);
            }
        }
        backend.submitted(2, 41);
        assert_eq!(backend.last_submitted(2), Some(41));
        let views = backend.views(2);
        assert!(!views.screen_valid && !views.panorama_valid);
        backend.submitted(99, 42);
        assert_eq!(backend.last_submitted(99), None);
    }
}
