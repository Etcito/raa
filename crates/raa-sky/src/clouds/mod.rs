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

mod dome;
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
use dome::{DomeGpu, DomeState, dome_record, dome_sky};
use history::{
    Accumulation, BURST_FRAMES, BURST_STRIDE, CloudHistory, FREEZE_FRAMES, FrameNote, Marked,
    REFRESH_PERIOD,
};
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
        let [x, z] = CloudGpu::from_frame(frame, None, false).shadow_origin;
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

/// What a frame does with its screen march (see `VolumetricClouds::freeze`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Freeze {
    /// The full march with history blend.
    March,
    /// Nothing is dispatched; the screen texture keeps its converged result.
    Skip,
    /// The refresh pass of a frozen view.
    Refresh,
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
    /// The world-space cloud dome (shared by every view), allocated while `cloudDome` is on.
    dome: Option<DomeGpu>,
    /// The dome's lifecycle: committed march, claim and note of the invocation.
    dome_state: DomeState,
    /// The last prepared marchable frame asked for the dome.
    dome_active: bool,
    /// `CloudFrame::signature` of the last prepared dome frame.
    dome_signature: u64,
    /// Marches of the prepared invocation's screen convergence burst, per view (1 = no burst).
    screen_burst: [u32; MAX_VIEWS],
    /// Frames that ran a screen convergence burst, per view.
    screen_burst_frames: [u32; MAX_VIEWS],
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
            dome: None,
            dome_state: DomeState::default(),
            dome_active: false,
            dome_signature: 0,
            screen_burst: [1; MAX_VIEWS],
            screen_burst_frames: [0; MAX_VIEWS],
        })
    }

    /// Allocates (or resizes, which resets) the dome and claims this invocation's march for
    /// `frame`'s view unless another view already did; uploads its uniforms when it did.
    fn prepare_dome(&mut self, gpu: &GpuContext, frame: &CloudFrame) -> Result<(), RaaError> {
        if self
            .dome
            .as_ref()
            .is_none_or(|dome| dome.size != frame.cloud_dome_size)
        {
            self.dome = Some(DomeGpu::new(gpu, frame.cloud_dome_size)?);
            self.dome_state.reset();
        }
        self.dome_signature = frame.signature;
        let claim = self.dome_state.claim(
            frame.invocation,
            frame.view,
            frame.signature,
            frame.animated,
            frame.cloud_dither,
        );
        if let (Some(plan), Some(dome)) = (claim, self.dome.as_ref()) {
            let queue = gpu.queue();
            for slot in 0..plan.burst {
                queue.write_buffer(
                    &dome.params,
                    u64::from(slot) * BURST_STRIDE,
                    &dome_record(frame, &plan, slot),
                );
            }
            queue.write_buffer(&dome.sky, 0, &dome_sky(frame));
        }
        Ok(())
    }

    /// The accumulation of `frame`'s view: its committed history and frame count.
    fn accumulation(&self, frame: &CloudFrame) -> Option<Accumulation> {
        self.history.accumulation(
            frame.view,
            frame.history_valid,
            frame.camera.view_proj(),
            frame.signature,
        )
    }

    /// What a frame does with its screen march: a view whose camera and sky signature have
    /// not changed for [`FREEZE_FRAMES`] frames and whose sky is not animated keeps its
    /// converged screen texture and records nothing, except on every [`REFRESH_PERIOD`]th
    /// invocation, a cheap pass that keeps the converged texels and marches only those that
    /// have none (geometry that uncovered sky while the camera stood still).
    fn freeze(frame: &CloudFrame, accumulation: Option<Accumulation>) -> Freeze {
        match accumulation {
            Some(a) if a.count >= FREEZE_FRAMES && !frame.animated => {
                if frame.invocation % REFRESH_PERIOD == 0 {
                    Freeze::Refresh
                } else {
                    Freeze::Skip
                }
            }
            _ => Freeze::March,
        }
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
            // With the dome on no screen march exists: the screen target is not even made.
            let accumulation = if frame.cloud_dome {
                None
            } else {
                // A recreated screen target holds no march: its history restarts, whatever
                // the size or `cloud_scale` change that caused it.
                if self
                    .targets
                    .ensure_screen(gpu, frame.view, frame.size, frame.cloud_scale)?
                {
                    self.history.reset_screen(frame.view);
                }
                self.accumulation(frame)
            };
            let refresh = matches!(
                Self::freeze(frame, accumulation),
                Freeze::Refresh
            );
            // A changed signature or no history at all: converge inside this frame (see
            // `BURST_FRAMES`); a still dither-off march is final after one.
            let burst = if !frame.cloud_dome
                && frame.cloud_dither
                && self
                    .history
                    .signature_changed(frame.view, frame.signature)
            {
                BURST_FRAMES
            } else {
                1
            };
            self.screen_burst[frame.view as usize] = burst;
            let record = CloudGpu::from_frame(frame, accumulation, refresh).to_bytes();
            let sky = cloud_sky_parameters(frame).to_bytes();
            let queue = gpu.queue();
            queue.write_buffer(self.targets.params(frame.view), 0, &record);
            if burst > 1 {
                // Slot 0 restarts the march (no history), slots 1.. blend by running mean from
                // the same camera, so each reads exactly the texel the last one wrote.
                let still = frame.camera.view_proj();
                for slot in 0..burst {
                    let acc = (slot > 0).then_some(Accumulation {
                        previous: still,
                        count: slot + 1,
                    });
                    let record = CloudGpu::from_frame(frame, acc, false).to_bytes();
                    queue.write_buffer(
                        self.targets.params(frame.view),
                        u64::from(slot) * BURST_STRIDE,
                        &record,
                    );
                }
            }
            queue.write_buffer(self.targets.sky(frame.view), 0, &sky);
        }
        self.dome_active = marchable && frame.cloud_dome;
        if self.dome_active {
            self.prepare_dome(gpu, frame)?;
        } else if !frame.cloud_dome && self.dome.is_some() {
            self.dome = None;
            self.dome_state.reset();
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
        if frame.cloud_dome {
            // No screen-space march: the dome (once per invocation, by its claiming view) is
            // the clouds, and the screen result stays invalid.
            self.history.record_frame(
                frame.view,
                frame.invocation,
                FrameNote {
                    animated: frame.animated,
                    frozen: true,
                    history_used: false,
                },
            );
            self.history
                .record_screen(frame.view, frame.invocation, None);
            if let Some(plan) = self.dome_state.record(frame.invocation, frame.view)
                && !plan.frozen
                && let Some(dome) = self.dome.as_ref()
            {
                self.passes.record_dome(
                    gpu,
                    encoder,
                    frame.view,
                    &self.noise,
                    &self.targets,
                    atmosphere,
                    dome,
                    plan.burst,
                )?;
            }
            return Ok(());
        }
        let accumulation = self.accumulation(frame);
        let freeze = Self::freeze(frame, accumulation);
        let burst = if freeze == Freeze::March {
            self.screen_burst[frame.view as usize]
        } else {
            1
        };
        if burst > 1 {
            let frames = &mut self.screen_burst_frames[frame.view as usize];
            *frames = frames.saturating_add(1);
        }
        self.history.record_frame(
            frame.view,
            frame.invocation,
            FrameNote {
                animated: frame.animated,
                frozen: freeze != Freeze::March,
                history_used: accumulation.is_some() && burst == 1,
            },
        );
        if freeze != Freeze::Skip {
            self.passes.record_screen(
                gpu,
                encoder,
                frame.view,
                &self.noise,
                &self.targets,
                atmosphere,
                depth,
                burst,
            )?;
        }
        // A skipped march still commits: the screen texture stays valid and keeps counting.
        self.history
            .record_screen(frame.view, frame.invocation, Some(frame.camera.view_proj()));
        self.history.record_marked(
            frame.view,
            frame.invocation,
            Marked {
                count: if burst > 1 {
                    burst
                } else {
                    accumulation.map_or(1, |a| a.count)
                },
                signature: frame.signature,
            },
        );
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
        let dome_valid = self.dome_active && self.dome_state.has_committed();
        let dome = self
            .dome
            .as_ref()
            .map_or_else(|| self.views.dome.clone(), |dome| dome.current.clone());
        CloudViews {
            panorama,
            screen,
            panorama_valid,
            screen_valid,
            dome,
            dome_valid,
            ..self.views.clone()
        }
    }

    fn invalidate(&mut self, view: Option<u32>, reason: CloudInvalidation) {
        self.history.invalidate(view, reason);
        // The dome follows the sky, not a camera: only a model or enable event drops it. A
        // weather edit keeps the committed dome displayable until the changed signature's
        // march replaces it in the same frame.
        if matches!(reason, CloudInvalidation::Model | CloudInvalidation::Enable) {
            self.dome_state.reset();
        }
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

    fn accumulation_observation(&self, view: u32) -> crate::CloudAccumulationDiagnostics {
        let (count, note) = self.history.observation(view);
        let (dome_count, dome_frozen, dome_burst_frames) = self.dome_state.observation();
        crate::CloudAccumulationDiagnostics {
            dome: crate::CloudDomeDiagnostics {
                active: self.dome_active,
                count: dome_count,
                frozen: self.dome_active && dome_frozen,
                burst_frames: dome_burst_frames,
            },
            burst_frames: self
                .screen_burst_frames
                .get(view as usize)
                .copied()
                .unwrap_or(0),
            count,
            animated: note.is_some_and(|n| n.animated),
            frozen: note.is_some_and(|n| n.frozen),
            history_used: note.is_some_and(|n| n.history_used),
        }
    }

    fn submitted(&mut self, view: u32, invocation: u64) {
        self.history.submitted(view, invocation);
        self.dome_state.submitted(view, invocation);
        self.noise.acknowledge(view, invocation);
    }
}
