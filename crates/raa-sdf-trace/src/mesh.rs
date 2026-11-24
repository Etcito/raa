//! Hybrid mesh reflections and an explicit full-frame path-traced reference.

mod geometry;
mod hybrid;
mod pass;
pub use geometry::MeshInput;
use hybrid::PathCapture;
pub use hybrid::PathMode;
pub use pass::{MESH_TRACE_NAME, MeshTraceFeature, MeshTraceHandle};

use std::sync::mpsc::Receiver;

use raa_core::RaaError;
use wgpu::Buffer;

use self::geometry::Geometry;

pub const MAX_BOUNCES: u32 = 6;
pub const REFERENCE_SPP: u32 = 4;
const COUNTER_COUNT: usize = 20;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshTraceSettings {
    pub reflections: bool,
    pub reference: bool,
    pub confidence_threshold: f32,
    pub reflection_bounces: u32,
    pub path_bounces: u32,
    pub path_mode: PathMode,
}

impl Default for MeshTraceSettings {
    fn default() -> Self {
        Self {
            reflections: false,
            reference: false,
            confidence_threshold: 0.55,
            reflection_bounces: 2,
            path_bounces: 2,
            path_mode: PathMode::Off,
        }
    }
}

impl MeshTraceSettings {
    pub fn active(self) -> bool {
        self.reflections || self.reference || self.path_mode != PathMode::Off
    }

    pub fn validate(self) -> Result<Self, RaaError> {
        if !(0.0..=1.0).contains(&self.confidence_threshold)
            || !(1..=MAX_BOUNCES).contains(&self.reflection_bounces)
            || self.path_bounces > MAX_BOUNCES
        {
            return Err(RaaError::FrameGraph {
                code: "trace.invalid_settings",
                message: format!(
                    "expected confidenceThreshold in [0, 1], reflectionBounces in [1, {MAX_BOUNCES}] and pathBounces in [0, {MAX_BOUNCES}]"
                ),
            });
        }
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshTraceMaterial {
    pub base: [f32; 3],
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub metallic: f32,
    pub mirror: bool,
    pub unlit: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct MeshTraceStats {
    pub frame: u64,
    pub size: (u32, u32),
    pub settings: MeshTraceSettings,
    pub counts: [u32; COUNTER_COUNT],
}

#[derive(Default)]
pub struct MeshTraceScene {
    pub settings: MeshTraceSettings,
    path: hybrid::PathState,
    geometry: Geometry,
    materials: Vec<MeshTraceMaterial>,
    material_bytes: Vec<u8>,
    material_revision: u64,
    frame: Option<Buffer>,
    sky_diffuse: f32,
    latest: Option<MeshTraceStats>,
    pending: Option<Receiver<Option<MeshTraceStats>>>,
    epoch: u64,
}

impl MeshTraceScene {
    pub fn set_path_model(&mut self, bytes: &[u8]) -> Result<(), RaaError> {
        self.path.load(bytes)?;
        self.epoch = self.epoch.wrapping_add(1);
        self.latest = None;
        Ok(())
    }

    pub fn path_model_ready(&self) -> bool {
        self.path.weights.is_some()
    }

    pub fn begin_path_capture(&mut self, seed: u32) {
        self.path.capture = Some(PathCapture { seed, result: None });
    }

    pub fn end_path_capture(&mut self) -> Option<((u32, u32), Buffer)> {
        self.path.capture.take()?.result
    }

    pub fn configure(&mut self, settings: MeshTraceSettings) {
        if settings != self.settings {
            self.settings = settings;
            self.epoch = self.epoch.wrapping_add(1);
            self.latest = None;
        }
    }

    pub fn update(
        &mut self,
        inputs: &[MeshInput<'_>],
        materials: &[MeshTraceMaterial],
        frame: Buffer,
        sky_diffuse: f32,
    ) -> Result<(), RaaError> {
        self.geometry.update(inputs)?;
        self.frame = Some(frame);
        self.sky_diffuse = sky_diffuse;
        if materials != self.materials {
            self.material_revision = self.material_revision.wrapping_add(1);
            self.materials.clear();
            self.materials.extend_from_slice(materials);
            self.material_bytes.clear();
            for material in materials {
                let flags = u32::from(material.mirror) | (u32::from(material.unlit) << 1);
                let lanes = material
                    .base
                    .into_iter()
                    .chain([material.roughness])
                    .chain(material.emissive)
                    .chain([material.metallic]);
                self.material_bytes.extend(lanes.flat_map(f32::to_le_bytes));
                self.material_bytes
                    .extend([flags, 0, 0, 0].into_iter().flat_map(u32::to_le_bytes));
            }
        }
        Ok(())
    }

    pub fn stats(&mut self) -> Option<MeshTraceStats> {
        self.poll();
        self.latest
    }

    fn poll(&mut self) {
        if let Some(pending) = &self.pending {
            match pending.try_recv() {
                Ok(Some(report)) => {
                    if report.settings == self.settings {
                        self.latest = Some(report);
                    }
                    self.pending = None;
                }
                Ok(None) | Err(std::sync::mpsc::TryRecvError::Disconnected) => self.pending = None,
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
    }
}
