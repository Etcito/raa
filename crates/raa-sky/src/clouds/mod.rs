//! The M2 volumetric cloud backend `VolumetricClouds`, registered by build.rs as
//! the preferred producer of the cloud seam (`crate::cloud_backend`).
//!
//! It adapts Solstice `crates/solstice-render/src/clouds/mod.rs` at revision
//! 7e465b599c13da165948ebaa44df824c36c3ae07: weather-driven volumetric clouds
//! rendered into a panorama for the environment, a half-resolution screen march
//! with cloud-only history, and a ground transmittance map. Until those passes
//! land it validates frames, tracks the per-view lifecycle of contract
//! `P2-CLOUD-VALIDITY` in [`history::CloudHistory`] and publishes the neutral
//! transmittance (map [0,0,1,0], cloudEnabled 0).

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
mod shaders;
#[cfg(test)]
mod tests;

use raa_core::{GpuContext, RaaError};
use raa_math::Mat4;
use raa_pipeline::LightingResources;

use super::{CloudFrame, CloudInvalidation, CloudRenderer, CloudViews, MAX_VIEWS, invalid};
use crate::atmosphere::AtmosphereResources;
use history::CloudHistory;
use noise::CloudNoise;

// Contract checks keeping the lifecycle entry points of later cards reachable.
const _: fn(&history::CloudHistory, u32, bool) -> Option<Mat4> = history::CloudHistory::history;

/// The S25 factory registered under `M2_BACKEND`.
pub(crate) fn create(gpu: &GpuContext) -> Result<Box<dyn CloudRenderer>, RaaError> {
    Ok(Box::new(VolumetricClouds {
        views: CloudViews::neutral(gpu)?,
        history: CloudHistory::new(),
        noise: CloudNoise::new(gpu)?,
        neutral_published: false,
    }))
}

/// The M2 volumetric cloud backend.
struct VolumetricClouds {
    views: CloudViews,
    history: CloudHistory,
    /// One-time noise and coverage generation.
    noise: CloudNoise,
    /// Published the neutral transmittance once per enabled resource set.
    neutral_published: bool,
}

impl VolumetricClouds {
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
        let marchable =
            frame.physical && frame.weather_enabled && atmosphere.sky_view(frame.view).is_some();
        self.history
            .prepare(frame.view, frame.size, frame.invocation, marchable, false);
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

    /// Records the noise generation of a marchable frame until it is acknowledged.
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
        }
        Ok(())
    }

    /// No screen march exists yet; notes that none was recorded.
    fn record_screen(
        &mut self,
        gpu: &GpuContext,
        _encoder: &mut wgpu::CommandEncoder,
        frame: &CloudFrame,
        _depth: &wgpu::TextureView,
        _atmosphere: &AtmosphereResources,
    ) -> Result<(), RaaError> {
        gpu.ensure_live()?;
        Self::validate(frame)?;
        self.history
            .record_screen(frame.view, frame.invocation, None);
        Ok(())
    }

    /// Neutral views carrying the view's lifecycle flags; nothing is allocated.
    fn views(&self, view: u32) -> CloudViews {
        let (panorama_valid, screen_valid) = self.history.flags(view);
        CloudViews {
            panorama_valid,
            screen_valid,
            ..self.views.clone()
        }
    }

    fn invalidate(&mut self, view: Option<u32>, reason: CloudInvalidation) {
        self.history.invalidate(view, reason);
        if matches!(reason, CloudInvalidation::Model | CloudInvalidation::Enable) {
            self.neutral_published = false;
        }
    }

    fn submitted(&mut self, view: u32, invocation: u64) {
        self.history.submitted(view, invocation);
        self.noise.acknowledge(view, invocation);
    }
}
