//! Hybrid mesh reflections and an explicit full-frame path-traced reference.

mod ddgi;
mod geometry;
mod hybrid;
mod nrc;
mod pass;
pub mod spherical;
mod spherical_gpu;
pub use ddgi::{
    DDGI_FRAME_FEATURE, DDGI_GI_NAME, DdgiDebug, DdgiFeature, DdgiHandle, DdgiInvalidation,
    DdgiSettings, DdgiStats, DdgiView, DdgiVisibility,
};
pub use geometry::MeshInput;
pub use nrc::{
    NRC_FRAME_FEATURE, NRC_GI_NAME, NrcFeature, NrcHandle, NrcSettings, NrcState, NrcStats,
};
use hybrid::PathCapture;
pub use hybrid::PathMode;
pub use pass::{GraphMeshTrace, MESH_TRACE_NAME, MeshTraceFeature, MeshTraceHandle};
pub use spherical_gpu::{
    SPHERICAL_GI_NAME, SphericalGiDebug, SphericalGiFeature, SphericalGiHandle, SphericalGiProbe,
    SphericalGiSettings, SphericalGiStats, SphericalSunRefresh,
};

use std::{cell::RefCell, collections::BTreeMap, rc::Rc, sync::mpsc::Receiver};

use raa_core::RaaError;
use wgpu::Buffer;

use self::geometry::Geometry;

pub const MAX_BOUNCES: u32 = 6;
/// A complete reference path can use both the diffuse/specular and mirror budgets.
pub const MAX_LIGHTING_DEPTH: u32 = 2 * MAX_BOUNCES;
pub const REFERENCE_SPP: u32 = 4;
const COUNTER_COUNT: usize = 36;
pub const GRID_DIFFUSE_WEIGHT_COUNT: usize = 81;
const GRID_DIFFUSE_CAPTURE_WORDS: usize = 128;
const GRID_DIFFUSE_MAX_CAPTURE_ROWS: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GridDiffuseCaptureRequest {
    seed: u32,
    stride: u32,
}

#[derive(Clone, Debug)]
pub struct GridDiffuseCapture {
    pub frame: u64,
    pub size: (u32, u32),
    pub candidates: u32,
    pub words: Vec<u32>,
    pub corner_diagnostics: Vec<u32>,
    request: GridDiffuseCaptureRequest,
}

impl GridDiffuseCapture {
    pub fn seed(&self) -> u32 {
        self.request.seed
    }
    pub fn stride(&self) -> u32 {
        self.request.stride
    }
}

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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GridDiffuseMode {
    #[default]
    Legacy,
    LegacyExperiment,
    AnalyticDirectional,
    Learned,
}

impl GridDiffuseMode {
    pub fn name(self) -> &'static str {
        match self {
            Self::Legacy | Self::LegacyExperiment => "legacy",
            Self::AnalyticDirectional => "analytic-directional",
            Self::Learned => "learned",
        }
    }
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
    pub grid_diffuse: GridDiffuseMode,
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
            grid_diffuse: GridDiffuseMode::Legacy,
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

pub(super) const TRACE_MATERIAL_BYTES: u64 = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshTraceTexture {
    /// Atlas x, y, width and height in mip-0 texels.
    pub rect: [u32; 4],
    /// Distinguishes new contents at a recycled atlas slot for cached captures.
    pub generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshTraceMaterial {
    pub base: [f32; 3],
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub metallic: f32,
    pub mirror: bool,
    pub unlit: bool,
    pub base_texture: Option<MeshTraceTexture>,
}

/// Opt-in legacy sharp-mirror residency experiment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum MirrorPriority {
    #[default]
    Off = 0,
    Measure = 1,
    On = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ReflectedSnapshot {
    grid_revision: u64,
    origin_epoch: u64,
    geometry_revision: u64,
    configuration: u64,
}

struct ReflectedFeedback {
    snapshot: ReflectedSnapshot,
    receivers: Vec<spherical::ReflectedReceiver>,
}

#[derive(Clone, Copy, Debug)]
pub struct MeshTraceStats {
    pub frame: u64,
    pub size: (u32, u32),
    pub settings: MeshTraceSettings,
    pub reference_spp: u32,
    pub mirror_priority: MirrorPriority,
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
    mirror_priority: MirrorPriority,
    mirror_priority_configuration: u64,
    reflected_snapshot: Option<ReflectedSnapshot>,
    reference_sample: Option<(u32, u32)>,
    path: hybrid::PathState,
    grid_diffuse_weights: Option<[f32; GRID_DIFFUSE_WEIGHT_COUNT]>,
    grid_diffuse_capture: Option<GridDiffuseCaptureRequest>,
    geometry: Geometry,
    gpu_geometry: Option<pass::GeometryBuffers>,
    spherical_active: bool,
    ddgi_active: bool,
    nrc_active: bool,
    graph_reports: BTreeMap<String, Rc<RefCell<TraceReport>>>,
    materials: Vec<MeshTraceMaterial>,
    material_bytes: Vec<u8>,
    material_revision: u64,
    frame: Option<Buffer>,
    texture_atlas: Option<wgpu::TextureView>,
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

    /// A frozen local gate in row-major W1, b1, W2, b2 order.
    pub fn set_grid_diffuse_model(&mut self, bytes: &[u8]) -> Result<(), RaaError> {
        let error = || RaaError::FrameGraph {
            code: "trace.grid_diffuse_model",
            message: "Expected 81 finite little-endian f32 grid diffuse weights bounded by 16."
                .into(),
        };
        if bytes.len() != GRID_DIFFUSE_WEIGHT_COUNT * 4 {
            return Err(error());
        }
        let weights = std::array::from_fn(|i| {
            let lane = i * 4;
            f32::from_le_bytes(bytes[lane..lane + 4].try_into().expect("one weight"))
        });
        if weights
            .iter()
            .any(|weight| !(-16.0..=16.0).contains(weight))
        {
            return Err(error());
        }
        if self
            .grid_diffuse_weights
            .is_some_and(|loaded| loaded != weights)
        {
            return Err(RaaError::FrameGraph {
                code: "trace.grid_diffuse_model_frozen",
                message: "This renderer already owns a different frozen grid diffuse checkpoint."
                    .into(),
            });
        }
        self.grid_diffuse_weights = Some(weights);
        Ok(())
    }

    pub fn request_grid_diffuse_capture(&mut self, seed: u32, stride: u32) {
        let request = (stride != 0).then_some(GridDiffuseCaptureRequest { seed, stride });
        if request != self.grid_diffuse_capture {
            self.grid_diffuse_capture = request;
            for report in self.graph_reports.values() {
                report.borrow_mut().grid_diffuse_capture = None;
            }
        }
    }

    pub fn grid_diffuse_capture(&self, node: &str) -> Option<GridDiffuseCapture> {
        let mut report = self.graph_report(node)?.borrow_mut();
        report.poll_pending();
        report
            .grid_diffuse_capture
            .as_ref()
            .filter(|capture| Some(capture.request) == self.grid_diffuse_capture)
            .cloned()
    }

    pub fn grid_diffuse_model_ready(&self) -> bool {
        self.grid_diffuse_weights.is_some()
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

    pub fn mirror_priority(&self) -> MirrorPriority {
        self.mirror_priority
    }

    pub fn set_mirror_priority(&mut self, mode: MirrorPriority) {
        if mode != self.mirror_priority {
            self.mirror_priority = mode;
            self.mirror_priority_configuration = self.mirror_priority_configuration.wrapping_add(1);
            self.report.latest = None;
            self.report.reflected_feedback = None;
        }
    }

    fn active_mirror_priority(&self) -> MirrorPriority {
        if self.settings.reflections
            && !self.settings.reference
            && self.settings.path_mode == PathMode::Off
            && matches!(
                self.indirect_grid,
                MeshIndirectGrid::Auto | MeshIndirectGrid::Spherical
            )
            && self.spherical_active
        {
            self.mirror_priority
        } else {
            MirrorPriority::Off
        }
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

    fn set_ddgi_active(&mut self, active: bool) {
        self.ddgi_active = active;
        self.release_gpu_geometry_if_idle();
    }

    fn set_nrc_active(&mut self, active: bool) {
        self.nrc_active = active;
        self.release_gpu_geometry_if_idle();
    }

    /// Whether a legacy or graph trace is configured for this scene.
    pub fn active(&self) -> bool {
        self.needs_trace_geometry(true)
    }

    /// Graph passes own their installation. The legacy pass can be disabled
    /// by an active graph while its standalone settings remain configured.
    pub fn needs_trace_geometry(&self, legacy_pass_enabled: bool) -> bool {
        (legacy_pass_enabled && self.settings.active()) || !self.graph_reports.is_empty()
    }

    fn release_gpu_geometry_if_idle(&mut self) {
        if !self.spherical_active && !self.ddgi_active && !self.nrc_active && !self.active() {
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
        gpu: &raa_core::GpuContext,
        inputs: &[MeshInput<'_>],
        materials: &[MeshTraceMaterial],
        frame: Buffer,
        texture_atlas: wgpu::TextureView,
        sky_diffuse: f32,
    ) -> Result<(), RaaError> {
        let storage_limit = geometry::storage_limit(gpu.limits());
        let material_size = materials.len() as u64 * TRACE_MATERIAL_BYTES;
        self.geometry.update(inputs, storage_limit, material_size)?;
        self.frame = Some(frame);
        self.texture_atlas = Some(texture_atlas);
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
                let rect = material.base_texture.map_or([0; 4], |tex| tex.rect);
                self.material_bytes
                    .extend(rect.into_iter().flat_map(u32::to_le_bytes));
            }
        }
        Ok(())
    }

    pub fn stats(&mut self) -> Option<MeshTraceStats> {
        self.report.poll(self.settings);
        self.report
            .latest
            .filter(|stats| stats.mirror_priority == self.active_mirror_priority())
    }

    pub fn graph_stats(&self, node: &str) -> Option<MeshTraceStats> {
        let mut report = self.graph_report(node)?.borrow_mut();
        report.poll_pending();
        report.latest
    }

    fn graph_report(&self, node: &str) -> Option<&Rc<RefCell<TraceReport>>> {
        self.graph_reports.iter().find_map(|(entry, report)| {
            entry
                .rsplit_once('#')
                .is_some_and(|(_, id)| id == node)
                .then_some(report)
        })
    }
}

struct TraceReadback {
    stats: MeshTraceStats,
    reflected_feedback: Option<ReflectedFeedback>,
    grid_diffuse_capture: Option<GridDiffuseCapture>,
}

#[derive(Default)]
struct TraceReport {
    latest: Option<MeshTraceStats>,
    pending: Option<Receiver<Option<TraceReadback>>>,
    reflected_feedback: Option<ReflectedFeedback>,
    grid_diffuse_capture: Option<GridDiffuseCapture>,
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
            Ok(Some(readback)) => {
                self.latest = Some(readback.stats);
                self.reflected_feedback = readback.reflected_feedback;
                self.grid_diffuse_capture = readback.grid_diffuse_capture;
                self.pending = None;
            }
            Ok(None) | Err(std::sync::mpsc::TryRecvError::Disconnected) => self.pending = None,
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
        }
    }
}

#[cfg(test)]
mod trace_routing_tests {
    use super::*;

    #[test]
    fn retained_legacy_settings_do_not_prepare_disabled_mesh_tracing() {
        let mut scene = MeshTraceScene::default();
        scene.configure(MeshTraceSettings {
            reflections: true,
            ..Default::default()
        });
        assert!(scene.active());
        assert!(scene.needs_trace_geometry(true));
        assert!(!scene.needs_trace_geometry(false));

        scene
            .graph_reports
            .insert("trace.mesh#reflections".into(), Default::default());
        assert!(scene.needs_trace_geometry(false));
        scene.graph_reports.clear();
        assert!(!scene.needs_trace_geometry(false));
        assert!(scene.needs_trace_geometry(true));
    }
}
