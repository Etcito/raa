//! The real-frame inference extension uses the live graph's scene and display textures.

use std::{cell::RefCell, rc::Rc};

use raa_core::{GpuContext, RaaError};
use raa_lit::{LitHandle, StochasticMode};
use raa_pipeline::{
    GraphFrame, HDR_FORMAT, ImageDesc, ImageDim, PublishedImage, RecordCtx, ResourceTable,
};
use raa_taa::HistoryState;
use serde_json::Value;
use sha2::{Digest, Sha256};
use wgpu::{
    BindingResource as Resource, Buffer, BufferUsages, TextureFormat as Format, TextureUsages,
    TextureView,
};

use super::area_model::AreaRoute;
use crate::exec::FeatureHandles;
use crate::plan::{GraphPlan, InputSource, PlanNode};

const WGSL: &str = include_str!("real_model.wgsl");
const ID: &str = "real-frame-v1";
const WEIGHT_BYTES: usize = 6560;
const GUIDE_BYTES: u64 = 128;
const STATE_BYTES: u64 = 64;
const STEM_BYTES: u64 = 48;
const COEFF_BYTES: u64 = 32;
const AREA_LAYERS: u32 = 4;
const GUIDES: [&str; 12] = [
    "normalX",
    "normalY",
    "normalZ",
    "logDepthOver8",
    "roughness",
    "motionU",
    "motionV",
    "historyValid",
    "reactive",
    "jitterX",
    "jitterY",
    "geometryValid",
];
const TENSORS: [(&str, &[u32], usize); 14] = [
    ("stem.weight", &[12, 12], 0),
    ("stem.bias", &[12], 576),
    ("spatial.weight", &[12, 3, 3], 624),
    ("spatial.bias", &[12], 1056),
    ("mix.weight", &[12, 12], 1104),
    ("mix.bias", &[12], 1680),
    ("area.weight", &[9, 26], 1728),
    ("area.bias", &[9], 2672),
    ("recurrent.weight", &[24, 33], 2720),
    ("recurrent.bias", &[24], 5888),
    ("reflection.weight", &[3, 12], 5984),
    ("reflection.bias", &[3], 6128),
    ("sr.weight", &[8, 12], 6144),
    ("sr.bias", &[8], 6528),
];

fn refusal(code: &'static str, message: impl Into<String>) -> RaaError {
    RaaError::FrameGraph {
        code,
        message: message.into(),
    }
}

#[derive(Clone)]
pub struct RealModelPackage {
    bytes: Rc<[u8]>,
    digest: String,
    kind: String,
    trained_heads: Vec<String>,
}

impl RealModelPackage {
    pub fn validate(manifest: &str, weights: Vec<u8>) -> Result<Self, RaaError> {
        let invalid = |message| refusal("graph.neural_package", message);
        let m: Value = serde_json::from_str(manifest)
            .map_err(|_| invalid("expected a JSON model manifest"))?;
        let digest = format!("{:x}", Sha256::digest(&weights));
        let kind = m["kind"]
            .as_str()
            .filter(|v| !v.is_empty())
            .ok_or_else(|| invalid("model provenance is missing"))?;
        if m["format"] != "raa.real-frame-model/1"
            || m["modelId"] != ID
            || m["precision"] != "fp32"
            || m["tensorLayout"] != "NHWC"
            || m["tensorAlignment"] != 16
            || m["guideChannels"] != serde_json::json!(GUIDES)
            || m["cadence"] != "successfully submitted real frame per view"
            || m["colorDomain"] != "nonnegative scene-linear HDR at fixed exposure 1"
            || weights.len() != WEIGHT_BYTES
            || m["weightsByteLength"] != WEIGHT_BYTES
            || m["weightsSha256"] != digest
        {
            return Err(invalid(
                "model ABI, guide order, byte count or SHA-256 differs",
            ));
        }
        let tensors = m["tensors"]
            .as_array()
            .filter(|list| list.len() == TENSORS.len())
            .ok_or_else(|| invalid("model must supply every tensor in ABI order"))?;
        let mut end = 0;
        for (tensor, (name, shape, offset)) in tensors.iter().zip(TENSORS) {
            let bytes = 4 * shape.iter().map(|&n| n as usize).product::<usize>();
            if tensor["name"] != name
                || tensor["offset"] != offset
                || tensor["shape"] != serde_json::json!(shape)
                || tensor["dtype"] != "f32le"
                || tensor["byteLength"] != bytes
                || weights[end..offset].iter().any(|&byte| byte != 0)
                || weights[offset..offset + bytes]
                    .chunks_exact(4)
                    .map(|word| f32::from_le_bytes(word.try_into().expect("FP32 word")))
                    .any(|f| !f.is_finite() || f.abs() > 4096.0)
            {
                return Err(invalid(
                    "tensor layout, zero padding or finite bounds differ",
                ));
            }
            end = offset + bytes;
        }
        let mut trained_heads = Vec::new();
        for head in m["trainedHeads"].as_array().ok_or_else(|| invalid("trainedHeads must be an array"))? {
            let head = head.as_str().filter(|head| matches!(*head, "area" | "reflection" | "display"))
                .ok_or_else(|| invalid("unknown trained head in manifest"))?;
            if trained_heads.iter().any(|existing| existing == head) {
                return Err(invalid("duplicate trained head in manifest"));
            }
            trained_heads.push(head.to_owned());
        }
        Ok(Self {
            bytes: weights.into(),
            digest,
            kind: kind.to_owned(),
            trained_heads,
        })
    }

    pub fn trained_heads(&self) -> &[String] {
        &self.trained_heads
    }

    pub fn digest(&self) -> &str {
        &self.digest
    }
    pub fn kind(&self) -> &str {
        &self.kind
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RealModelHeads {
    pub area: bool,
    pub reflection: bool,
    pub sr: bool,
}

impl RealModelHeads {
    pub fn any(self) -> bool {
        self.area || self.reflection || self.sr
    }
}

pub struct RealModelSelection {
    pub package: RealModelPackage,
    pub heads: RealModelHeads,
    pub scene_size: (u32, u32),
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RealModelStats {
    pub submitted_frame: Option<u64>,
    pub scene_size: (u32, u32),
    pub display_size: (u32, u32),
    pub bytes: u64,
    pub dispatches: u32,
    pub ran: RealModelHeads,
    pub history_frames: u32,
}

struct ReflectionRoute {
    replace: String,
    trace: String,
    incident: String,
    alias: Option<String>,
}

fn linked<'a>(
    plan: &'a GraphPlan,
    node: &PlanNode,
    key: &str,
    output: &str,
    kind: &str,
) -> Option<&'a PlanNode> {
    let InputSource::Link {
        node,
        port,
        history: false,
    } = &node.inputs.iter().find(|p| p.port == key)?.source
    else {
        return None;
    };
    if port != output {
        return None;
    }
    plan.nodes
        .iter()
        .find(|n| n.id == *node && n.type_id == kind)
}

impl ReflectionRoute {
    fn find(plan: &GraphPlan) -> Option<Self> {
        let replace = plan.nodes.iter().find(|n| n.type_id == "ssr.replace")?;
        let neural = linked(plan, replace, "incident", "incident", "neural.reflection-reconstruction");
        let incident = linked(plan, neural.unwrap_or(replace), "incident", "incident", "ssr.incident")?;
        let trace = linked(plan, incident, "trace", "trace", "ssr.trace")?;
        Some(Self {
            replace: format!("ssr.replace#{}", replace.id),
            trace: format!("ssr.trace#{}", trace.id),
            incident: format!("ssr.incident#{}", incident.id),
            alias: neural.map(|n| format!("neural.reflection-reconstruction#{}", n.id)),
        })
    }
}

pub(super) struct NeuralRoutes {
    area: Option<AreaRoute>,
    reflection: Option<ReflectionRoute>,
}

impl NeuralRoutes {
    pub(super) fn from_plan(plan: &GraphPlan) -> Result<Self, RaaError> {
        let has = |name| plan.nodes.iter().any(|node| node.type_id == name);
        let area = AreaRoute::from_plan(plan).filter(|route| route.alias.is_some());
        let reflection = ReflectionRoute::find(plan).filter(|route| route.alias.is_some());
        if has("neural.area-reconstruction") && area.is_none() {
            return Err(refusal("graph.neural_route", "wire area.trace.raw → neural area pairs → area.mask.src → forward"));
        }
        if has("neural.reflection-reconstruction") && reflection.is_none() {
            return Err(refusal("graph.neural_route", "wire SSR trace → incident → neural incident → SSR replace"));
        }
        if has("pass.taa") && has("neural.super-resolution") {
            return Err(refusal("graph.neural_route", "use one temporal resolve owner per HDR branch"));
        }
        if let Some(generation) = plan.nodes.iter().find(|node| node.type_id == "neural.frame-generation") {
            let color = |node: &PlanNode| node.inputs.iter().find(|input| input.port == "color").map(|input| input.source.clone());
            let endpoint = color(generation);
            let display = plan.nodes.iter().find(|node| node.type_id == "pass.tonemap").and_then(color);
            if endpoint.is_none() || endpoint != display {
                return Err(refusal("graph.neural_route", "frame generation must read the same real HDR endpoint as tonemap"));
            }
        }
        Ok(Self { area, reflection })
    }

    pub(super) fn active(&self) -> bool { self.area.is_some() || self.reflection.is_some() }
}

pub(super) struct SharedNeuralRoutes(pub Rc<RefCell<NeuralRoutes>>);

impl GraphFrame for SharedNeuralRoutes {
    fn before_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let routes = self.0.borrow();
        if let Some(route) = &routes.area
            && route.mask == pass
            && let Some(image) = ctx.resources.get(&route.trace, "raw")
        {
            if let Some(alias) = &route.alias {
                ctx.resources.publish(alias, "pairs", &image.view, image.desc);
            }
            ctx.resources.bind_input(pass, "src", image);
        }
        if let Some(route) = &routes.reflection
            && route.replace == pass
            && let Some(image) = ctx.resources.get(&route.incident, "incident")
        {
            if let Some(alias) = &route.alias {
                ctx.resources.publish(alias, "incident", &image.view, image.desc);
            }
            ctx.resources.bind_input(pass, "incident", image);
        }
        Ok(())
    }
}

pub(super) struct RealModelFrame {
    model: RealModelSelection,
    lit: LitHandle,
    area: Option<AreaRoute>,
    reflection: Option<ReflectionRoute>,
    reactive: bool,
    kernels: Kernels,
    resources: Option<Tensors>,
    attempted_size: (u32, u32),
    history: HistoryState,
    display_size: (u32, u32),
    encoded: bool,
    recurrent: bool,
    pending: RealModelStats,
    submitted: RealModelStats,
}

impl RealModelFrame {
    pub(super) fn prepare(
        model: RealModelSelection,
        plan: &GraphPlan,
        h: &FeatureHandles<'_>,
    ) -> Result<Self, RaaError> {
        let fail = |message| refusal("graph.neural_route", message);
        let heads = model.heads;
        let area = if heads.area {
            Some(AreaRoute::from_plan(plan).ok_or_else(|| {
                fail("area reconstruction requires stratified area.trace → area.mask → forward")
            })?)
        } else {
            None
        };
        let reflection = ReflectionRoute::find(plan);
        if heads.reflection && reflection.is_none() {
            return Err(fail(
                "reflection reconstruction requires SSR trace → incident → replace",
            ));
        }
        if heads.sr && !plan.nodes.iter().any(|node| matches!(node.type_id.as_str(), "pass.taa" | "neural.super-resolution")) {
            return Err(fail("SR requires the typed TAA resolve boundary"));
        }
        if !heads.any() || h.viewports != 1 || !h.gpu.summary().is_hardware() {
            return Err(fail("choose one or more heads on a single hardware view"));
        }
        let lit = h
            .lit
            .as_ref()
            .ok_or_else(|| fail("lit frame guides are required"))?
            .handle();
        let size = model.scene_size;
        let resources = Tensors::new(h.gpu, size, heads)?;
        let reactive = plan.nodes.iter().any(|node| {
            matches!(
                node.type_id.as_str(),
                "pass.transparent.oit" | "pass.particles.blend" | "pass.fur.shell"
            )
        });
        Ok(Self {
            kernels: Kernels::new(h.gpu, &model.package, heads),
            model,
            lit,
            area,
            reflection,
            reactive,
            resources: Some(resources),
            attempted_size: size,
            history: HistoryState::new(size),
            display_size: (0, 0),
            encoded: false,
            recurrent: false,
            pending: RealModelStats::default(),
            submitted: RealModelStats::default(),
        })
    }

    pub(super) fn stats(&self) -> RealModelStats {
        self.submitted
    }
    pub(super) fn heads(&self) -> RealModelHeads {
        self.model.heads
    }

    fn ensure_size(&mut self, ctx: &RecordCtx<'_>) {
        if self.attempted_size != ctx.size {
            self.attempted_size = ctx.size;
            self.resources = Tensors::new(ctx.gpu, ctx.size, self.model.heads).ok();
            self.history.resize(ctx.size);
            self.history.invalidate();
        }
    }

    fn encode(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> bool {
        self.ensure_size(ctx);
        let display = ctx.targets.display_rect().size;
        if self.display_size != display {
            self.display_size = display;
            self.history.invalidate();
        }
        if self.encoded {
            return true;
        }
        let Some(t) = &self.resources else {
            return false;
        };
        let depth = ctx.targets.depth_input(ctx.resources, pass);
        let normal = ctx.targets.normal_input(ctx.resources, pass);
        let motion = ctx.targets.motion_input(ctx.resources, pass);
        if [&depth, &normal, &motion]
            .iter()
            .any(|view| !matches_size(view, ctx.size))
        {
            return false;
        }
        let k = &self.kernels;
        k.params(
            ctx.gpu,
            &k.encode_params,
            ctx.size,
            [
                u32::from(self.history.is_valid()),
                u32::from(self.reactive),
                0,
                0,
                0,
                0,
            ],
        );
        let frame = self.lit.frame_buffer();
        k.stem.run(
            ctx,
            &[
                b(0, &k.encode_params),
                b(1, &k.weights),
                b(2, &frame),
                v(3, &depth),
                v(4, &normal),
                v(5, &motion),
                b(6, &t.state[self.history.read_index()]),
                b(7, &t.guide),
                b(8, &t.stem),
            ],
            ctx.size,
            1,
        );
        k.spatial.run(
            ctx,
            &[
                b(0, &k.encode_params),
                b(1, &k.weights),
                b(7, &t.guide),
                b(8, &t.stem),
            ],
            ctx.size,
            1,
        );
        self.pending.dispatches += 2;
        self.pending.scene_size = ctx.size;
        self.pending.bytes = t.bytes + k.bytes();
        self.encoded = true;
        true
    }

    fn run_area(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) {
        let stochastic = self.lit.area_stochastic();
        if stochastic.mode != StochasticMode::Stratified || stochastic.active == 0 {
            return;
        }
        let route = self.area.as_ref().expect("validated area route");
        let Some(raw) = ctx.resources.get(&route.trace, "raw") else {
            return;
        };
        if !matches_size(&raw.view, ctx.size) || !(1..=AREA_LAYERS).contains(&raw.desc.layers) {
            return;
        }
        if !self.encode(pass, ctx) {
            return;
        }
        let t = self.resources.as_ref().expect("encoded tensors");
        let output = t.area.as_ref().expect("selected area");
        let k = &self.kernels;
        let active = (0..8)
            .filter(|&slot| raa_lit::table_slot(&stochastic.table, slot).is_some())
            .fold(0, |mask, slot| mask | (1 << slot));
        k.params(ctx.gpu, &k.area_params, ctx.size, [0, 0, 0, 0, active, 0]);
        k.area.as_ref().expect("selected pipeline").run(
            ctx,
            &[
                b(0, &k.area_params),
                b(1, &k.weights),
                b(7, &t.guide),
                v(9, &raw.view),
                v(10, output),
            ],
            ctx.size,
            raw.desc.layers,
        );
        let view = if raw.desc.layers == AREA_LAYERS { output.clone() } else {
            output.texture().create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                array_layer_count: Some(raw.desc.layers),
                ..Default::default()
            })
        };
        if let Some(alias) = self.area.as_ref().and_then(|route| route.alias.as_ref()) {
            ctx.resources.publish(alias, "pairs", &view, raw.desc);
        }
        ctx.resources.bind_input(pass, "src", PublishedImage { view, desc: raw.desc });
        self.pending.ran.area = true;
        self.pending.dispatches += 1;
    }

    fn run_recurrent(
        &mut self,
        pass: &str,
        ctx: &mut RecordCtx<'_>,
        incident: Option<&PublishedImage>,
    ) -> bool {
        if self.recurrent {
            return true;
        }
        let color = ctx.targets.hdr().src().clone();
        if !matches_size(&color, ctx.size) || !self.encode(pass, ctx) {
            return false;
        }
        let t = self.resources.as_ref().expect("encoded tensors");
        let meta = self
            .reflection
            .as_ref()
            .and_then(|route| ctx.resources.get(&route.trace, "hitMeta"))
            .filter(|image| matches_size(&image.view, ctx.size));
        let incident = incident.filter(|image| matches_size(&image.view, ctx.size));
        let k = &self.kernels;
        k.params(
            ctx.gpu,
            &k.late_params,
            ctx.size,
            [
                0,
                0,
                u32::from(incident.is_some()),
                u32::from(meta.is_some()),
                0,
                0,
            ],
        );
        k.recurrent.run(
            ctx,
            &[
                b(0, &k.late_params),
                b(1, &k.weights),
                b(7, &t.guide),
                v(11, &color),
                v(12, incident.map_or(&k.zero, |image| &image.view)),
                v(13, meta.as_ref().map_or(&k.zero, |image| &image.view)),
                b(14, &t.state[self.history.write_index()]),
            ],
            ctx.size,
            1,
        );
        self.recurrent = true;
        self.pending.dispatches += 1;
        true
    }

    fn run_reflection(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) {
        let incident = self.reflection.as_ref()
            .and_then(|route| ctx.resources.get(&route.incident, "incident"))
            .or_else(|| ctx.resources.input(pass, "incident"))
            .filter(|i| matches_size(&i.view, ctx.size));
        if !self.run_recurrent(pass, ctx, incident.as_ref()) || !self.model.heads.reflection {
            return;
        }
        let Some(incident) = incident else {
            return;
        };
        let t = self.resources.as_ref().expect("encoded tensors");
        let k = &self.kernels;
        let output = t.reflection.as_ref().expect("selected reflection");
        k.reflection.as_ref().expect("selected pipeline").run(
            ctx,
            &[
                b(0, &k.late_params),
                b(1, &k.weights),
                v(12, &incident.view),
                b(14, &t.state[self.history.write_index()]),
                v(15, output),
            ],
            ctx.size,
            1,
        );
        ctx.resources.bind_input(
            pass,
            "incident",
            PublishedImage {
                view: output.clone(),
                desc: incident.desc,
            },
        );
        if let Some(alias) = self.reflection.as_ref().and_then(|route| route.alias.as_ref()) {
            ctx.resources.publish(alias, "incident", output, incident.desc);
        }
        self.pending.ran.reflection = true;
        self.pending.dispatches += 1;
    }

    fn run_sr(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) {
        let source = ctx.targets.hdr().src().clone();
        let destination = ctx.targets.hdr().dst().clone();
        let display = size_of(&destination);
        if !matches_size(&source, ctx.size)
            || source.texture() == destination.texture()
            || destination.texture().format() != HDR_FORMAT
            || !destination
                .texture()
                .usage()
                .contains(TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC)
            || display.0 < ctx.size.0
            || display.1 < ctx.size.1
            || !self.run_recurrent(pass, ctx, None)
        {
            return;
        }
        let k = &self.kernels;
        let t = self.resources.as_ref().expect("encoded tensors");
        let coeffs = t.coefficients.as_ref().expect("selected SR");
        let frame = self.lit.frame_buffer();
        k.sr_head.as_ref().expect("selected pipeline").run(
            ctx,
            &[
                b(0, &k.late_params),
                b(1, &k.weights),
                b(14, &t.state[self.history.write_index()]),
                b(16, coeffs),
            ],
            ctx.size,
            1,
        );
        k.sr_resolve.as_ref().expect("selected pipeline").run(
            ctx,
            &[
                b(0, &k.late_params),
                b(2, &frame),
                v(11, &source),
                b(16, coeffs),
                v(17, &destination),
            ],
            display,
            1,
        );
        let desc = ImageDesc::new(ImageDim::D2, HDR_FORMAT, display.0, display.1, 1, 1);
        ctx.resources.bind_input(
            pass,
            raa_taa::NEURAL_COLOR_INPUT,
            PublishedImage {
                view: destination,
                desc,
            },
        );
        self.pending.display_size = display;
        self.pending.ran.sr = true;
        self.pending.dispatches += 2;
    }
}

pub(super) struct SharedRealFrame(pub Rc<RefCell<RealModelFrame>>);

impl GraphFrame for SharedRealFrame {
    fn begin(&mut self, _: &ResourceTable) {
        let mut model = self.0.borrow_mut();
        model.pending = RealModelStats::default();
        model.encoded = false;
        model.recurrent = false;
    }

    fn before_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let mut model = self.0.borrow_mut();
        model.pending.submitted_frame = Some(ctx.frame);
        if model.area.as_ref().is_some_and(|route| route.mask == pass) {
            model.run_area(pass, ctx);
        }
        if model
            .reflection
            .as_ref()
            .is_some_and(|route| route.replace == pass)
        {
            model.run_reflection(pass, ctx);
        }
        if pass == raa_taa::TAA_PASS_NAME && model.model.heads.sr {
            model.run_sr(pass, ctx);
        }
        Ok(())
    }

    fn after_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let mut model = self.0.borrow_mut();
        if pass == "forward" && model.reflection.is_none() && !model.model.heads.sr {
            model.run_recurrent(pass, ctx, None);
        }
        Ok(())
    }

    fn submitted(&mut self) {
        let mut model = self.0.borrow_mut();
        if model.recurrent {
            model.history.commit();
        } else {
            model.history.invalidate();
        }
        model.pending.history_frames = model.history.accumulated_frames();
        model.submitted = model.pending;
    }

    fn invalidate_history(&mut self) {
        self.0.borrow_mut().history.invalidate();
    }
}

fn size_of(view: &TextureView) -> (u32, u32) {
    (view.texture().width(), view.texture().height())
}

fn matches_size(view: &TextureView, size: (u32, u32)) -> bool {
    size_of(view) == size && view.texture().sample_count() == 1
}

fn buffer(gpu: &GpuContext, label: &str, bytes: u64, usage: BufferUsages) -> Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage,
        mapped_at_creation: false,
    })
}

fn texture(
    gpu: &GpuContext,
    label: &str,
    size: (u32, u32),
    layers: u32,
    format: Format,
    usage: TextureUsages,
) -> TextureView {
    gpu.device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            format,
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: layers,
            },
            dimension: wgpu::TextureDimension::D2,
            mip_level_count: 1,
            sample_count: 1,
            usage,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

struct Tensors {
    guide: Buffer,
    stem: Buffer,
    state: [Buffer; 2],
    coefficients: Option<Buffer>,
    area: Option<TextureView>,
    reflection: Option<TextureView>,
    bytes: u64,
}

impl Tensors {
    fn new(gpu: &GpuContext, size: (u32, u32), heads: RealModelHeads) -> Result<Self, RaaError> {
        let pixels = u64::from(size.0) * u64::from(size.1);
        gpu.check_compute_limits(|limits| {
            let max = limits
                .max_storage_buffer_binding_size
                .min(limits.max_buffer_size);
            if size.0 == 0
                || size.1 == 0
                || size.0.max(size.1) > limits.max_texture_dimension_2d
                || size.0.max(size.1).div_ceil(8) > limits.max_compute_workgroups_per_dimension
                || limits
                    .max_compute_workgroup_size_x
                    .min(limits.max_compute_workgroup_size_y)
                    < 8
                || limits.max_compute_invocations_per_workgroup < 64
                || limits.max_bindings_per_bind_group < 18
                || limits.max_sampled_textures_per_shader_stage < 3
                || limits.max_storage_buffers_per_shader_stage < 4
                || limits.max_storage_textures_per_shader_stage < 1
                || limits.max_uniform_buffers_per_shader_stage < 2
                || limits.max_texture_array_layers < AREA_LAYERS
                || pixels.checked_mul(GUIDE_BYTES).is_none_or(|n| n > max)
            {
                return Err(refusal(
                    "graph.neural_limits",
                    "real-frame extent or compute bindings exceed device limits",
                ));
            }
            Ok(())
        })?;
        let storage = |label, stride| {
            buffer(
                gpu,
                label,
                pixels * stride,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            )
        };
        let sampled_storage = TextureUsages::TEXTURE_BINDING
            | TextureUsages::STORAGE_BINDING
            | TextureUsages::COPY_SRC;
        let area = heads.area.then(|| {
            texture(
                gpu,
                "neural.area",
                size,
                AREA_LAYERS,
                HDR_FORMAT,
                sampled_storage,
            )
        });
        let reflection = heads.reflection.then(|| {
            texture(
                gpu,
                "neural.incident",
                size,
                1,
                Format::Rgba32Float,
                sampled_storage,
            )
        });
        let coefficients = heads
            .sr
            .then(|| storage("neural.coefficients", COEFF_BYTES));
        let stride = GUIDE_BYTES
            + STEM_BYTES
            + 2 * STATE_BYTES
            + if heads.sr { COEFF_BYTES } else { 0 }
            + if heads.area {
                u64::from(AREA_LAYERS) * 8
            } else {
                0
            }
            + if heads.reflection { 16 } else { 0 };
        Ok(Self {
            guide: storage("neural.guide", GUIDE_BYTES),
            stem: storage("neural.stem", STEM_BYTES),
            state: [
                storage("neural.history.0", STATE_BYTES),
                storage("neural.history.1", STATE_BYTES),
            ],
            coefficients,
            area,
            reflection,
            bytes: pixels * stride,
        })
    }
}

fn b(binding: u32, buffer: &Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn v(binding: u32, view: &TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: Resource::TextureView(view),
    }
}

struct Kernel {
    name: &'static str,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

impl Kernel {
    fn run(
        &self,
        ctx: &mut RecordCtx<'_>,
        entries: &[wgpu::BindGroupEntry<'_>],
        size: (u32, u32),
        layers: u32,
    ) {
        let group = ctx
            .gpu
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(self.name),
                layout: &self.layout,
                entries,
            });
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(self.name),
                timestamp_writes: None,
            });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(size.0.div_ceil(8), size.1.div_ceil(8), layers);
    }
}

struct Kernels {
    weights: Buffer,
    encode_params: Buffer,
    area_params: Buffer,
    late_params: Buffer,
    zero: TextureView,
    stem: Kernel,
    spatial: Kernel,
    area: Option<Kernel>,
    recurrent: Kernel,
    reflection: Option<Kernel>,
    sr_head: Option<Kernel>,
    sr_resolve: Option<Kernel>,
}

impl Kernels {
    fn new(gpu: &GpuContext, package: &RealModelPackage, heads: RealModelHeads) -> Self {
        let device = gpu.device();
        let source = format!("{}\n{WGSL}", raa_taa::REPROJECT_WGSL);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("neural.real-frame"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        use wgpu::{
            BindingType as T, BufferBindingType as B, TextureSampleType as S,
            TextureViewDimension as D,
        };
        let buffer_type = |ty| T::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let uniform = buffer_type(B::Uniform);
        let read = buffer_type(B::Storage { read_only: true });
        let write = buffer_type(B::Storage { read_only: false });
        let sampled = |sample_type, view_dimension| T::Texture {
            sample_type,
            view_dimension,
            multisampled: false,
        };
        let float = S::Float { filterable: false };
        let image = sampled(float, D::D2);
        let storage = |format, view_dimension| T::StorageTexture {
            format,
            view_dimension,
            access: wgpu::StorageTextureAccess::WriteOnly,
        };
        let kernel = |name, bindings: &[(u32, T)]| {
            let entries = bindings
                .iter()
                .map(|&(binding, ty)| wgpu::BindGroupLayoutEntry {
                    binding,
                    ty,
                    count: None,
                    visibility: wgpu::ShaderStages::COMPUTE,
                })
                .collect::<Vec<_>>();
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(name),
                entries: &entries,
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(name),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            });
            Kernel {
                name,
                layout,
                pipeline,
            }
        };
        let params = |label| {
            buffer(
                gpu,
                label,
                32,
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            )
        };
        let weights = buffer(
            gpu,
            "neural.weights",
            WEIGHT_BYTES as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        gpu.queue().write_buffer(&weights, 0, &package.bytes);
        Self {
            weights,
            encode_params: params("neural.guide.params"),
            area_params: params("neural.area.params"),
            late_params: params("neural.late.params"),
            zero: texture(
                gpu,
                "neural.incident.empty",
                (1, 1),
                1,
                Format::Rgba32Float,
                TextureUsages::TEXTURE_BINDING,
            ),
            stem: kernel(
                "encode_stem",
                &[
                    (0, uniform),
                    (1, read),
                    (2, uniform),
                    (3, sampled(S::Depth, D::D2)),
                    (4, image),
                    (5, image),
                    (6, read),
                    (7, write),
                    (8, write),
                ],
            ),
            spatial: kernel(
                "encode_spatial",
                &[(0, uniform), (1, read), (7, write), (8, write)],
            ),
            area: heads.area.then(|| {
                kernel(
                    "reconstruct_area",
                    &[
                        (0, uniform),
                        (1, read),
                        (7, write),
                        (9, sampled(float, D::D2Array)),
                        (10, storage(HDR_FORMAT, D::D2Array)),
                    ],
                )
            }),
            recurrent: kernel(
                "recurrent",
                &[
                    (0, uniform),
                    (1, read),
                    (7, write),
                    (11, image),
                    (12, image),
                    (13, image),
                    (14, write),
                ],
            ),
            reflection: heads.reflection.then(|| {
                kernel(
                    "reconstruct_reflection",
                    &[
                        (0, uniform),
                        (1, read),
                        (12, image),
                        (14, write),
                        (15, storage(Format::Rgba32Float, D::D2)),
                    ],
                )
            }),
            sr_head: heads.sr.then(|| {
                kernel(
                    "sr_head",
                    &[(0, uniform), (1, read), (14, write), (16, write)],
                )
            }),
            sr_resolve: heads.sr.then(|| {
                kernel(
                    "resolve_display",
                    &[
                        (0, uniform),
                        (2, uniform),
                        (11, image),
                        (16, write),
                        (17, storage(HDR_FORMAT, D::D2)),
                    ],
                )
            }),
        }
    }

    fn params(&self, gpu: &GpuContext, buffer: &Buffer, size: (u32, u32), fields: [u32; 6]) {
        let bytes = [size.0, size.1]
            .into_iter()
            .chain(fields)
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        gpu.queue().write_buffer(buffer, 0, &bytes);
    }

    fn bytes(&self) -> u64 {
        self.weights.size() + 3 * 32 + 16
    }
}
