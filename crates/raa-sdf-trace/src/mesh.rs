//! Hybrid mesh reflections and an explicit full-frame path-traced reference.

mod geometry;
mod hybrid;
mod pass;
pub mod spherical;
mod spherical_gpu;
pub use geometry::MeshInput;
use hybrid::PathCapture;
pub use hybrid::PathMode;
pub use pass::{GraphMeshTrace, MESH_TRACE_NAME, MeshTraceFeature, MeshTraceHandle};
pub use spherical_gpu::{
    SPHERICAL_GI_NAME, SphericalGiDebug, SphericalGiFeature, SphericalGiHandle, SphericalGiProbe,
    SphericalGiSettings, SphericalGiStats,
};

use std::{cell::RefCell, collections::BTreeMap, rc::Rc, sync::mpsc::Receiver};

use raa_core::RaaError;
use wgpu::Buffer;

use self::geometry::Geometry;

pub const MAX_BOUNCES: u32 = 6;
/// A complete reference path can use both the diffuse/specular and mirror budgets.
pub const MAX_LIGHTING_DEPTH: u32 = 2 * MAX_BOUNCES;
pub const REFERENCE_SPP: u32 = 4;
const COUNTER_COUNT: usize = 29;

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
    IndirectDiffuse,
    IndirectSpecular,
    RoughGrid,
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
            Self::IndirectDiffuse => 16,
            Self::IndirectSpecular => 17,
            Self::RoughGrid => 18,
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
        if matches!(
            self.lighting_output,
            LightingOutput::IndirectDiffuse
                | LightingOutput::IndirectSpecular
                | LightingOutput::RoughGrid
        ) && (self.reference || self.path_mode != PathMode::Learned)
        {
            return Err(RaaError::FrameGraph {
                code: "trace.invalid_settings",
                message: "these lightingOutput views require the fitted path".into(),
            });
        }
        if let LightingOutput::Bounce(depth) = self.lighting_output
            && (!self.reference || depth > MAX_LIGHTING_DEPTH)
        {
            return Err(RaaError::FrameGraph {
                code: "trace.invalid_settings",
                message: format!(
                    "lightingDepth requires the path reference and a depth in 0..={MAX_LIGHTING_DEPTH}"
                ),
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
    pub reference_spp: u32,
    pub counts: [u32; COUNTER_COUNT],
}

/// Live GI route for the legacy mesh tracing pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MeshIndirectGrid {
    #[default]
    Auto,
    Off,
    Volume,
    Spherical,
}

#[derive(Default)]
pub struct MeshTraceScene {
    pub settings: MeshTraceSettings,
    pub indirect_grid: MeshIndirectGrid,
    reference_sample: Option<(u32, u32)>,
    path: hybrid::PathState,
    geometry: Geometry,
    gpu_geometry: Option<pass::GeometryBuffers>,
    spherical_active: bool,
    graph_reports: BTreeMap<String, Rc<RefCell<TraceReport>>>,
    materials: Vec<MeshTraceMaterial>,
    material_bytes: Vec<u8>,
    material_revision: u64,
    frame: Option<Buffer>,
    sky_diffuse: f32,
    report: TraceReport,
    epoch: u64,
}

impl MeshTraceScene {
    /// Seed and sample count for an explicitly requested offline reference invocation.
    pub fn set_reference_sample(&mut self, sample: Option<(u32, u32)>) {
        self.reference_sample = sample;
    }

    pub fn set_path_model(&mut self, bytes: &[u8]) -> Result<(), RaaError> {
        self.path.load(bytes)?;
        self.epoch = self.epoch.wrapping_add(1);
        self.report.latest = None;
        Ok(())
    }

    pub fn path_model_ready(&self) -> bool {
        self.path.weights.is_some()
    }

    pub fn begin_path_capture(&mut self, seed: u32) {
        self.path.capture = Some(PathCapture::training(seed));
    }

    pub fn end_path_capture(&mut self) -> Option<((u32, u32), Buffer)> {
        self.path
            .capture
            .take()?
            .result
            .map(|(_, size, buffer)| (size, buffer))
    }

    pub fn begin_rough_evidence(&mut self, seed: u32) {
        self.path.capture = Some(PathCapture::rough_evidence(seed));
    }

    pub fn end_rough_evidence(&mut self) -> Option<(u64, (u32, u32), Buffer)> {
        self.path.capture.take()?.result
    }

    pub fn configure(&mut self, settings: MeshTraceSettings) {
        if settings != self.settings {
            self.settings = settings;
            self.release_gpu_geometry_if_idle();
            self.epoch = self.epoch.wrapping_add(1);
            self.report.latest = None;
        }
    }

    fn set_spherical_active(&mut self, active: bool) {
        self.spherical_active = active;
        self.release_gpu_geometry_if_idle();
    }

    /// Whether a legacy or graph trace needs scene geometry this frame.
    pub fn active(&self) -> bool {
        self.settings.active() || !self.graph_reports.is_empty()
    }

    fn release_gpu_geometry_if_idle(&mut self) {
        if !self.spherical_active && !self.active() {
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
        self.report.poll(self.settings);
        self.report.latest
    }

    pub fn graph_stats(&self, node: &str) -> Option<MeshTraceStats> {
        let mut report = self
            .graph_reports
            .iter()
            .find(|(entry, _)| entry.rsplit_once('#').is_some_and(|(_, id)| id == node))?
            .1
            .borrow_mut();
        report.poll_pending();
        report.latest
    }
}

#[derive(Default)]
struct TraceReport {
    latest: Option<MeshTraceStats>,
    pending: Option<Receiver<Option<MeshTraceStats>>>,
}

impl TraceReport {
    fn poll(&mut self, settings: MeshTraceSettings) {
        self.poll_pending();
        self.latest = self.latest.filter(|stats| stats.settings == settings);
    }

    fn poll_pending(&mut self) {
        let Some(pending) = &self.pending else {
            return;
        };
        match pending.try_recv() {
            Ok(Some(report)) => {
                self.latest = Some(report);
                self.pending = None;
            }
            Ok(None) | Err(std::sync::mpsc::TryRecvError::Disconnected) => self.pending = None,
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
        }
    }
}
