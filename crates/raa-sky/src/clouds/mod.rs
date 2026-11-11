//! The M2 volumetric cloud backend `VolumetricClouds`, registered by build.rs as
//! the preferred producer of the cloud seam (`crate::cloud_backend`).
//!
//! It adapts Solstice `crates/solstice-render/src/clouds/mod.rs` at revision
//! 7e465b599c13da165948ebaa44df824c36c3ae07: weather-driven volumetric clouds
//! rendered into a panorama for the environment, a half-resolution screen march
//! with cloud-only history, and a ground transmittance map. It validates frames,
//! tracks the per-view lifecycle of contract `P2-CLOUD-VALIDITY` in
//! [`history::CloudHistory`], marches the ground transmittance map of marchable frames
//! and publishes it to `LightingResources` (S10 map/light), or the neutral
//! transmittance (map [0,0,1,0], cloudEnabled 0) when shadows cannot apply. Each view's
//! half-resolution screen march blends with that view's own committed history, which only
//! an acknowledged submission commits and every invalidation or resize resets.

/* Adapted from solstice@7e465b5 crates/solstice-render/src/clouds/mod.rs.
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

mod history;
mod noise;
mod passes;
mod resources;
mod shaders;
#[cfg(test)]
mod tests;

use raa_core::{GpuContext, RaaError};
use raa_pipeline::LightingResources;

use super::{CloudFrame, CloudInvalidation, CloudRenderer, CloudViews, MAX_VIEWS, invalid};
use crate::atmosphere::AtmosphereResources;
use history::CloudHistory;
use noise::CloudNoise;
use passes::CloudPasses;
use resources::{CloudGpu, CloudTargets, SHADOW_EXTENT, cloud_sky_parameters};

/// The S25 factory registered under `M2_BACKEND`.
pub(crate) fn create(gpu: &GpuContext) -> Result<Box<dyn CloudRenderer>, RaaError> {
    Ok(Box::new(VolumetricClouds::new(gpu)?))
}

/// One cloud transmittance publication: the published view (the ground map when
/// `cloud`, else the neutral dummy) with its S10 map and light lanes.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Publication {
    cloud: bool,
    map: [f32; 4],
    light: [f32; 4],
}

impl Publication {
    /// The neutral transmittance: map [0,0,1,0] (cloudEnabled 0), default light.
    const NEUTRAL: Publication = Publication {
        cloud: false,
        map: [0.0, 0.0, 1.0, 0.0],
        light: [0.0, 1.0, 0.0, 0.0],
    };

    /// What a prepared frame publishes: its ground map (origin and extent of the cloud
    /// record, light toward the dominant body) when it is marchable with cloud shadows
    /// on, the neutral transmittance otherwise.
    fn for_frame(frame: &CloudFrame, marchable: bool) -> Self {
        if !(marchable && frame.cloud_shadows) {
            return Self::NEUTRAL;
        }
        let [x, z] = CloudGpu::from_frame(frame, None).shadow_origin;
        let toward = cloud_sky_parameters(frame).dominant;
        Publication {
            cloud: true,
            map: [x, z, SHADOW_EXTENT, 1.0],
            light: [toward[0], toward[1], toward[2], 0.0],
        }
    }
}

/// MARCHABLE: the physical model renders weather and the view has a sky-view LUT.
fn marchable(frame: &CloudFrame, sky_view: bool) -> bool {
    frame.physical && frame.weather_enabled && sky_view
}

/// Whether `next` must be published over the last published value.
fn needs_publication(last: Option<Publication>, next: Publication) -> bool {
    last != Some(next)
}

/// The M2 volumetric cloud backend.
struct VolumetricClouds {
    views: CloudViews,
    history: CloudHistory,
    /// One-time noise and coverage generation.
    noise: CloudNoise,
    /// Per-view uniforms, the noise sampler and the ground transmittance texture.
    targets: CloudTargets,
    /// The ground shadow pipelines and per-view bind groups.
    passes: CloudPasses,
    /// The last published transmittance; `None` forces the next publication.
    published: Option<Publication>,
}

impl VolumetricClouds {
    /// The backend with its shared resources and pipelines; no view prepared.
    fn new(gpu: &GpuContext) -> Result<VolumetricClouds, RaaError> {
        Ok(VolumetricClouds {
            views: CloudViews::neutral(gpu)?,
            history: CloudHistory::new(),
            noise: CloudNoise::new(gpu)?,
            targets: CloudTargets::new(gpu)?,
            passes: CloudPasses::new(gpu)?,
            published: None,
        })
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

impl CloudRenderer for VolumetricClouds {
    fn prepare(
        &mut self,
        gpu: &GpuContext,
        frame: &CloudFrame,
        atmosphere: &AtmosphereResources,
        lighting: &mut LightingResources,
    ) -> Result<(), RaaError> {
        gpu.ensure_live()?;
        Self::validate(frame)?;
        let marchable = marchable(frame, atmosphere.sky_view(frame.view).is_some());
        self.history
            .prepare(frame.view, frame.size, frame.invocation, marchable, true);
        if marchable {
            self.passes
                .bind_view(gpu, frame.view, &self.noise, &self.targets, atmosphere)?;
            self.targets.ensure_screen(gpu, frame.view, frame.size)?;
            let history = self.history.history(frame.view, frame.history_valid);
            let record = CloudGpu::from_frame(frame, history).to_bytes();
            let sky = cloud_sky_parameters(frame).to_bytes();
            let queue = gpu.queue();
            queue.write_buffer(self.targets.params(frame.view), 0, &record);
            queue.write_buffer(self.targets.sky(frame.view), 0, &sky);
        }
        let next = Publication::for_frame(frame, marchable);
        if needs_publication(self.published, next) {
            let view = if next.cloud {
                self.targets.transmittance()
            } else {
                &self.views.transmittance
            };
            lighting.set_cloud_transmittance(view.clone(), next.map, next.light)?;
            self.published = Some(next);
        }
        Ok(())
    }

    /// Records the noise generation of a marchable frame until it is acknowledged,
    /// then its ground shadow map and mean when cloud shadows are on, then the panorama.
    fn record_lighting(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        frame: &CloudFrame,
        _atmosphere: &AtmosphereResources,
    ) -> Result<(), RaaError> {
        gpu.ensure_live()?;
        Self::validate(frame)?;
        if self.history.marchable(frame.view, frame.invocation) {
            self.noise.record(encoder, frame.invocation, frame.view);
            if frame.cloud_shadows {
                self.passes.record_ground_shadow(encoder, frame.view);
            }
            self.passes.record_panorama(encoder, frame.view);
        }
        Ok(())
    }

    /// Marches the screen clouds of a marchable frame and notes the camera it used;
    /// records nothing (and notes that) otherwise.
    fn record_screen(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        frame: &CloudFrame,
        depth: &wgpu::TextureView,
        atmosphere: &AtmosphereResources,
    ) -> Result<(), RaaError> {
        gpu.ensure_live()?;
        Self::validate(frame)?;
        if !self.history.marchable(frame.view, frame.invocation) {
            self.history
                .record_screen(frame.view, frame.invocation, None);
            return Ok(());
        }
        self.passes.record_screen(
            gpu,
            encoder,
            frame.view,
            &self.noise,
            &self.targets,
            atmosphere,
            depth,
        )?;
        self.history
            .record_screen(frame.view, frame.invocation, Some(frame.camera.view_proj()));
        Ok(())
    }

    /// The view's lifecycle flags, the shared panorama while it is valid, the view's screen
    /// target while it exists, and neutral views otherwise; nothing is allocated.
    fn views(&self, view: u32) -> CloudViews {
        let (panorama_valid, screen_valid) = self.history.flags(view);
        let panorama = if panorama_valid {
            self.targets.panorama().clone()
        } else {
            self.views.panorama.clone()
        };
        let screen = self
            .targets
            .screen_current(view)
            .unwrap_or(&self.views.screen)
            .clone();
        CloudViews {
            panorama,
            screen,
            panorama_valid,
            screen_valid,
            ..self.views.clone()
        }
    }

    fn invalidate(&mut self, view: Option<u32>, reason: CloudInvalidation) {
        self.history.invalidate(view, reason);
        match reason {
            CloudInvalidation::Model | CloudInvalidation::Enable => self.published = None,
            CloudInvalidation::ViewRemoved => {
                let views = match view {
                    Some(view) => view..view.saturating_add(1),
                    None => 0..MAX_VIEWS as u32,
                };
                for view in views {
                    self.passes.release_view(view);
                    self.targets.release_screen(view);
                }
            }
            _ => {}
        }
    }

    fn submitted(&mut self, view: u32, invocation: u64) {
        self.history.submitted(view, invocation);
        self.noise.acknowledge(view, invocation);
    }
}
