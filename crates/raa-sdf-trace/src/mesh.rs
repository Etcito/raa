//! Hybrid mesh reflections and an explicit full-frame path-traced reference.

mod geometry;
pub mod spherical;
mod hybrid;
mod pass;
mod spherical_gpu;
pub use geometry::MeshInput;
use hybrid::PathCapture;
pub use hybrid::PathMode;
pub use pass::{MESH_TRACE_NAME, MeshTraceFeature, MeshTraceHandle};
pub use spherical_gpu::{
    SPHERICAL_GI_NAME, SphericalGiDebug, SphericalGiFeature, SphericalGiHandle, SphericalGiProbe,
    SphericalGiSettings, SphericalGiStats,
};

use std::sync::mpsc::Receiver;

use raa_core::RaaError;
use wgpu::Buffer;

use self::geometry::Geometry;

pub const MAX_BOUNCES: u32 = 6;
/// A complete reference path can use both the diffuse/specular and mirror budgets.
pub const MAX_LIGHTING_DEPTH: u32 = 2 * MAX_BOUNCES;
pub const REFERENCE_SPP: u32 = 4;
const COUNTER_COUNT: usize = 20;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum DiffuseOutput {
    #[default]
    Off,
    Direct,
    DirectAndIndirect,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LightingOutput {
    #[default]
    Combined,
    Direct,
    Indirect,
    Bounce(u32),
}

impl LightingOutput {
    /// Shared with select_lighting_depth in mesh_trace.wgsl.
    fn code(self) -> u32 {
        match self {
            Self::Combined => 0,
            Self::Direct => 1,
            Self::Indirect => 2,
            Self::Bounce(depth) => 3 + depth,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum PanelSampling {
    #[default]
    Stable,
    Single,
    Independent,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshTraceSettings {
    pub reflections: bool,
    pub reference: bool,
    pub confidence_threshold: f32,
    pub reflection_bounces: u32,
    pub path_bounces: u32,
    pub path_mode: PathMode,
    pub diffuse_output: DiffuseOutput,
    pub panel_sampling: PanelSampling,
    pub lighting_output: LightingOutput,
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
            diffuse_output: DiffuseOutput::Off,
            panel_sampling: PanelSampling::Stable,
            lighting_output: LightingOutput::Combined,
        }
    }
}

impl MeshTraceSettings {
    pub fn active(self) -> bool {
        self.reflections || self.reference || self.path_mode != PathMode::Off
    }

    pub fn validate(self) -> Result<Self, RaaError> {
        if self.diffuse_output != DiffuseOutput::Off
            && self.lighting_output != LightingOutput::Combined
        {
            return Err(RaaError::FrameGraph {
                code: "trace.invalid_settings",
                message: "diffuseOutput diagnostics require combined lightingOutput".into(),
            });
        }
        if let LightingOutput::Bounce(depth) = self.lighting_output
            && (!self.reference || depth > MAX_LIGHTING_DEPTH)
        {
            return Err(RaaError::FrameGraph {
                code: "trace.invalid_settings",
                message: format!("lightingDepth requires the path reference and a depth in 0..={MAX_LIGHTING_DEPTH}"),
            });
        }
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
    gpu_geometry: Option<pass::GeometryBuffers>,
    spherical_active: bool,
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
            self.release_gpu_geometry_if_idle();
            self.epoch = self.epoch.wrapping_add(1);
            self.latest = None;
        }
    }

    fn set_spherical_active(&mut self, active: bool) {
        self.spherical_active = active;
        self.release_gpu_geometry_if_idle();
    }

    fn release_gpu_geometry_if_idle(&mut self) {
        if !self.spherical_active && !self.settings.active() {
            self.gpu_geometry = None;
        }
    }

    fn geometry_buffers(&mut self, gpu: &raa_core::GpuContext) -> [Buffer; 3] {
        self.gpu_geometry
            .get_or_insert_with(|| pass::GeometryBuffers::new(gpu.device()))
            .upload(
                gpu,
                &self.geometry,
                self.material_revision,
                &self.material_bytes,
            )
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
