//! Executes the `denoise.*` nodes (SIGMA-style visibility, ReLAX-style radiance).
//!
//! Each node is routed `producer.port → denoise.signal → consumer.port`. A graph frame runs the
//! denoiser's compute passes right before the consumer pass records, publishes its output under
//! `denoise.<kind>#<id>` / `signal` and rebinds the consumer input to it, so the GPU time is
//! measured inside the consumer pass. Clean-room implementation (see denoise.wgsl).

use std::{cell::RefCell, rc::Rc};

use raa_core::{GpuContext, RaaError};
use raa_gi::GiHandle;
use raa_lit::{LitHandle, StochasticMode};
use raa_pipeline::{GraphFrame, PrepareCtx, PublishedImage, RecordCtx, ResourceTable};

use super::area_model::{AreaFrame, SLOTS};
use crate::catalog::{Catalog, ParamValue};
use crate::exec::FeatureHandles;
use crate::plan::{GraphPlan, InputSource, PlanNode};

const SIGMA_WGSL: &str = include_str!("denoise.wgsl");
const RELAX_WGSL: &str = include_str!("denoise_relax.wgsl");
const PARAM_BYTES: u64 = 64;
const MAX_ITERATIONS: u32 = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    Sigma,
    Relax,
}

#[derive(Clone, Debug, PartialEq)]
struct Settings {
    history: f32,
    radius: f32,
    normal_power: f32,
    depth_sigma: f32,
    clamp_sigma: f32,
    phi_luminance: f32,
    iterations: u32,
}

impl Settings {
    fn from_node(kind: Kind, node: &PlanNode) -> Self {
        let get = |key: &str, default: f64| {
            node.params
                .iter()
                .find(|(k, _)| k == key)
                .and_then(|(_, v)| match v {
                    ParamValue::Number(n) => Some(*n),
                    ParamValue::Int(n) => Some(*n as f64),
                    _ => None,
                })
                .unwrap_or(default) as f32
        };
        match kind {
            Kind::Sigma => Self {
                history: get("historyFrames", 8.0),
                radius: get("radius", 4.0),
                normal_power: get("normalPower", 32.0),
                depth_sigma: get("depthSigma", 1.0),
                clamp_sigma: get("clampSigma", 1.5),
                phi_luminance: 0.0,
                iterations: 0,
            },
            Kind::Relax => Self {
                history: get("historyFrames", 16.0),
                radius: 0.0,
                normal_power: get("normalPower", 64.0),
                depth_sigma: get("depthSigma", 1.0),
                clamp_sigma: get("clampSigma", 2.0),
                phi_luminance: get("phiLuminance", 4.0),
                iterations: (get("iterations", 2.0).max(0.0) as u32).min(MAX_ITERATIONS),
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Route {
    pub(super) kind: Kind,
    pub(super) alias: String,
    pub(super) producer: (String, String),
    pub(super) consumer: (String, String),
    pub(super) hit_meta: Option<(String, String)>,
    pub(super) trace: Option<(String, String)>,
    settings: Settings,
}

/// The pipeline entry of a planned node: its pass name, or `type#id` for a stage node.
fn entry_of(plan: &GraphPlan, catalog: &Catalog, id: &str) -> Option<String> {
    let node = plan.nodes.iter().find(|n| n.id == id)?;
    match catalog.node(&node.type_id).and_then(|spec| spec.pass) {
        Some(pass) => {
            let first = plan.nodes.iter().find(|n| n.type_id == node.type_id)?;
            Some(if first.id == node.id { pass.to_owned() } else { format!("{pass}#{}", node.id) })
        }
        None => Some(format!("{}#{}", node.type_id, node.id)),
    }
}

fn linked_source(node: &PlanNode, port: &str) -> Option<(String, String)> {
    match &node.inputs.iter().find(|i| i.port == port)?.source {
        InputSource::Link { node, port, history: false } => Some((node.clone(), port.clone())),
        _ => None,
    }
}

pub(super) fn routes_from_plan(plan: &GraphPlan, catalog: &Catalog) -> Result<Vec<Route>, RaaError> {
    let refuse = |message: String| RaaError::FrameGraph { code: "graph.denoise_route", message };
    let mut routes = Vec::new();
    for node in &plan.nodes {
        let kind = match node.type_id.as_str() {
            "denoise.sigma" => Kind::Sigma,
            "denoise.relax" => Kind::Relax,
            _ => continue,
        };
        let (source, source_port) = linked_source(node, "signal")
            .ok_or_else(|| refuse(format!("link a signal into `{}`", node.id)))?;
        let consumers: Vec<_> = plan
            .nodes
            .iter()
            .flat_map(|consumer| {
                consumer.inputs.iter().filter_map(move |input| match &input.source {
                    InputSource::Link { node: from, port, history: false }
                        if from == &node.id && port == "signal" =>
                    {
                        Some((consumer, input.port.clone()))
                    }
                    _ => None,
                })
            })
            .collect();
        let [(consumer, consumer_port)] = consumers.as_slice() else {
            return Err(refuse(format!("`{}` must feed exactly one consumer input", node.id)));
        };
        let producer = plan.nodes.iter().find(|n| n.id == source);
        if producer.is_none_or(|p| p.order_index >= consumer.order_index) {
            return Err(refuse(format!("`{}`: the producer must run before the consumer", node.id)));
        }
        let entry = |id: &str| {
            entry_of(plan, catalog, id).ok_or_else(|| refuse(format!("unknown node `{id}`")))
        };
        let optional = |port: &str| -> Result<Option<(String, String)>, RaaError> {
            linked_source(node, port).map(|(n, p)| Ok((entry(&n)?, p))).transpose()
        };
        routes.push(Route {
            kind,
            alias: format!("{}#{}", node.type_id, node.id),
            producer: (entry(&source)?, source_port),
            consumer: (entry(&consumer.id)?, consumer_port.clone()),
            hit_meta: optional("hitMeta")?,
            trace: optional("trace")?,
            settings: Settings::from_node(kind, node),
        });
    }
    Ok(routes)
}

struct RouteState {
    route: Route,
    resources: Option<Resources>,
    parity: usize,
    history: bool,
    pending: bool,
    jitter: [f32; 2],
    pending_jitter: [f32; 2],
    area: Option<AreaFrame>,
    pending_area: Option<AreaFrame>,
}

pub(super) struct DenoiseFrame {
    routes: Vec<RouteState>,
    lit: Option<LitHandle>,
    gi: Option<GiHandle>,
}

impl DenoiseFrame {
    pub(super) fn new(routes: Vec<Route>, h: &FeatureHandles<'_>) -> Self {
        Self {
            routes: routes
                .into_iter()
                .map(|route| RouteState {
                    route,
                    resources: None,
                    parity: 0,
                    history: false,
                    pending: false,
                    jitter: [0.0; 2],
                    pending_jitter: [0.0; 2],
                    area: None,
                    pending_area: None,
                })
                .collect(),
            lit: h.lit.as_ref().map(|lit| lit.handle()),
            gi: h.handles.gi.clone(),
        }
    }

    fn record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let jitter = self.lit.as_ref().map_or([0.0; 2], |lit| lit.camera_jitter_ndc());
        let camera_valid = self.gi.as_ref().is_none_or(|gi| gi.shared().camera_valid());
        for state in self.routes.iter_mut().filter(|s| s.route.consumer.0 == pass) {
            let route = &state.route;
            let Some(input) = ctx.resources.get(&route.producer.0, &route.producer.1) else {
                continue;
            };
            let format = input.desc.format;
            let layers = input.desc.layers.max(1);
            let supported = match route.kind {
                Kind::Sigma => format == wgpu::TextureFormat::Rgba16Float,
                Kind::Relax => matches!(
                    format,
                    wgpu::TextureFormat::Rgba16Float | wgpu::TextureFormat::Rgba32Float
                ),
            };
            if !supported || (input.desc.width, input.desc.height) != ctx.size {
                continue;
            }
            let key = (ctx.size, layers, format);
            if state.resources.as_ref().is_none_or(|r| r.key != key) {
                state.resources = Resources::new(ctx.gpu, route.kind, key).ok();
                state.history = false;
                state.area = None;
            }
            let Some(resources) = &state.resources else { continue };
            let jitter_delta = if state.history {
                [(state.jitter[0] - jitter[0]) * 0.5, (state.jitter[1] - jitter[1]) * -0.5]
            } else {
                [0.0; 2]
            };
            let mut reset_mask = 0u32;
            if route.kind == Kind::Sigma
                && let Some(lit) = &self.lit
            {
                let stochastic = lit.area_stochastic();
                if stochastic.mode == StochasticMode::Stratified && stochastic.active != 0 {
                    let current = AreaFrame::capture(stochastic, jitter, ctx);
                    for slot in 0..SLOTS {
                        if !state.area.as_ref().is_some_and(|old| current.same_light(old, slot)) {
                            reset_mask |= 1 << slot;
                        }
                    }
                    state.pending_area = Some(current);
                }
            }
            let fetch = |entry: &Option<(String, String)>| {
                entry.as_ref().and_then(|(e, p)| ctx.resources.get(e, p)).filter(|image| {
                    (image.desc.width, image.desc.height) == ctx.size
                        && image.desc.dim == raa_pipeline::ImageDim::D2
                })
            };
            let meta = fetch(&route.hit_meta);
            let trace = fetch(&route.trace);
            let flags = u32::from(state.history && camera_valid)
                | if meta.is_some() { 2 } else { 0 }
                | if trace.is_some() { 4 } else { 0 };
            let frame = Frame {
                settings: &route.settings,
                parity: state.parity,
                flags,
                jitter_delta,
                reset_mask,
                frame: ctx.frame as u32,
            };
            let normal = ctx.targets.normal_input(ctx.resources, pass);
            let motion = ctx.targets.motion_input(ctx.resources, pass);
            let depth = ctx.targets.depth_input(ctx.resources, pass);
            let output = resources.record(
                ctx,
                &frame,
                &input,
                &depth,
                &normal,
                &motion,
                meta.as_ref(),
                trace.as_ref(),
            );
            let desc = input.desc;
            ctx.resources.publish(&route.alias, "signal", &output, desc);
            ctx.resources
                .bind_input(pass, &route.consumer.1, PublishedImage { view: output, desc });
            state.pending = true;
            state.pending_jitter = jitter;
        }
        Ok(())
    }
}

pub(super) struct SharedDenoiseFrame(pub Rc<RefCell<DenoiseFrame>>);

impl SharedDenoiseFrame {
    fn bind_source(&self, pass: &str, resources: &ResourceTable) {
        for state in self.0.borrow().routes.iter().filter(|s| s.route.consumer.0 == pass) {
            let route = &state.route;
            if let Some(image) = resources.get(&route.producer.0, &route.producer.1) {
                resources.publish(&route.alias, "signal", &image.view, image.desc);
                resources.bind_input(pass, &route.consumer.1, image);
            }
        }
    }
}

impl GraphFrame for SharedDenoiseFrame {
    fn begin(&mut self, _table: &ResourceTable) {
        for state in &mut self.0.borrow_mut().routes {
            state.pending = false;
            state.pending_area = None;
        }
    }

    fn before_prepare(&mut self, pass: &str, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        // Consumers that publish bindings at prepare see the typed producer image.
        self.bind_source(pass, ctx.resources);
        Ok(())
    }

    fn before_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        self.bind_source(pass, ctx.resources);
        self.0.borrow_mut().record(pass, ctx)
    }

    fn submitted(&mut self) {
        for state in &mut self.0.borrow_mut().routes {
            if state.pending {
                state.pending = false;
                state.parity ^= 1;
                state.history = true;
                state.jitter = state.pending_jitter;
                if let Some(area) = state.pending_area.take() {
                    state.area = Some(area);
                }
            }
        }
    }

    fn invalidate_history(&mut self) {
        for state in &mut self.0.borrow_mut().routes {
            state.pending = false;
            state.history = false;
            state.area = None;
            state.pending_area = None;
        }
    }
}

struct Frame<'a> {
    settings: &'a Settings,
    parity: usize,
    flags: u32,
    jitter_delta: [f32; 2],
    reset_mask: u32,
    frame: u32,
}

impl Frame<'_> {
    fn params(&self, size: (u32, u32), layers: u32, step: u32, final_pass: bool) -> Vec<u8> {
        let s = self.settings;
        let mut bytes = Vec::with_capacity(PARAM_BYTES as usize);
        for v in [size.0, size.1, layers, self.flags] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for v in [
            self.jitter_delta[0],
            self.jitter_delta[1],
            s.history,
            s.radius,
            s.normal_power,
            s.depth_sigma,
            s.clamp_sigma,
            s.phi_luminance,
        ] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for v in [self.reset_mask, step, u32::from(final_pass), self.frame] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }
}

type Key = ((u32, u32), u32, wgpu::TextureFormat);

struct Resources {
    key: Key,
    kind: Kind,
    params: Vec<wgpu::Buffer>,
    history: [wgpu::TextureView; 2],
    moments: [wgpu::TextureView; 2],
    guide: [wgpu::TextureView; 2],
    acc: wgpu::TextureView,
    tmp: [wgpu::TextureView; 2],
    output: Option<wgpu::TextureView>,
    dummy: wgpu::TextureView,
    pipelines: Vec<wgpu::ComputePipeline>,
}

const SIGMA_TEMPORAL: usize = 0;
const SIGMA_SPATIAL: usize = 1;
const RELAX_TEMPORAL: usize = 0;
const RELAX_HISTORY: usize = 1;
const RELAX_ATROUS: usize = 2;
const RELAX_FINAL: usize = 3;

impl Resources {
    fn new(gpu: &GpuContext, kind: Kind, key: Key) -> Result<Self, RaaError> {
        let ((width, height), layers, format) = key;
        gpu.check_compute_limits(|limits| {
            if limits.max_storage_textures_per_shader_stage < 3
                || limits.max_texture_array_layers < layers
            {
                return Err(RaaError::Unsupported("denoiser bindings exceed device limits".into()));
            }
            Ok(())
        })?;
        let device = gpu.device();
        let texture = |label: &str, format, layers: u32, array: bool, extra| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width, height, depth_or_array_layers: layers },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | extra,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(if array {
                        wgpu::TextureViewDimension::D2Array
                    } else {
                        wgpu::TextureViewDimension::D2
                    }),
                    ..Default::default()
                })
        };
        let copy = wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST;
        let none = wgpu::TextureUsages::empty();
        let (state_format, array, state_layers) = match kind {
            Kind::Sigma => (wgpu::TextureFormat::Rgba16Float, true, layers),
            Kind::Relax => (wgpu::TextureFormat::Rgba32Float, false, 1),
        };
        let pair = |label: &str, extra| {
            [
                texture(label, state_format, state_layers, array, extra),
                texture(label, state_format, state_layers, array, extra),
            ]
        };
        let guide_format = wgpu::TextureFormat::Rgba32Float;
        let dummy = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("denoise absent input"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let source = match kind {
            Kind::Sigma => SIGMA_WGSL.to_owned(),
            Kind::Relax => RELAX_WGSL.replace(
                "OUT_FORMAT",
                if format == wgpu::TextureFormat::Rgba16Float { "rgba16float" } else { "rgba32float" },
            ),
        };
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("denoise"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        use wgpu::TextureViewDimension as V;
        let sampled = |view_dimension| wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension,
            multisampled: false,
        };
        let depth = wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: V::D2,
            multisampled: false,
        };
        let uniform = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let storage = |format, view_dimension| wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension,
        };
        let pipelines = match kind {
            Kind::Sigma => {
                let half = wgpu::TextureFormat::Rgba16Float;
                vec![
                    pipeline(device, &module, "sigma_temporal", &[
                        (0, sampled(V::D2Array)),
                        (1, depth),
                        (2, sampled(V::D2)),
                        (3, sampled(V::D2)),
                        (4, sampled(V::D2Array)),
                        (5, sampled(V::D2Array)),
                        (6, sampled(V::D2)),
                        (7, uniform),
                        (8, storage(half, V::D2Array)),
                        (9, storage(half, V::D2Array)),
                        (10, storage(guide_format, V::D2)),
                    ]),
                    pipeline(device, &module, "sigma_spatial", &[
                        (0, sampled(V::D2Array)),
                        (5, sampled(V::D2Array)),
                        (6, sampled(V::D2)),
                        (7, uniform),
                        (8, storage(half, V::D2Array)),
                    ]),
                ]
            }
            Kind::Relax => {
                let full = wgpu::TextureFormat::Rgba32Float;
                vec![
                    pipeline(device, &module, "relax_temporal", &[
                        (0, sampled(V::D2)),
                        (1, depth),
                        (2, sampled(V::D2)),
                        (3, sampled(V::D2)),
                        (4, sampled(V::D2)),
                        (5, sampled(V::D2)),
                        (6, sampled(V::D2)),
                        (7, uniform),
                        (8, storage(full, V::D2)),
                        (9, storage(full, V::D2)),
                        (10, storage(guide_format, V::D2)),
                        (11, sampled(V::D2)),
                        (12, sampled(V::D2)),
                    ]),
                    pipeline(device, &module, "relax_history", &[
                        (2, sampled(V::D2)),
                        (6, sampled(V::D2)),
                        (7, uniform),
                        (8, storage(full, V::D2)),
                        (9, storage(full, V::D2)),
                        (12, sampled(V::D2)),
                        (13, sampled(V::D2)),
                    ]),
                    pipeline(device, &module, "relax_atrous", &[
                        (2, sampled(V::D2)),
                        (6, sampled(V::D2)),
                        (7, uniform),
                        (8, storage(full, V::D2)),
                        (12, sampled(V::D2)),
                        (13, sampled(V::D2)),
                    ]),
                    pipeline(device, &module, "relax_final", &[
                        (0, sampled(V::D2)),
                        (1, depth),
                        (2, sampled(V::D2)),
                        (6, sampled(V::D2)),
                        (7, uniform),
                        (11, sampled(V::D2)),
                        (12, sampled(V::D2)),
                        (13, sampled(V::D2)),
                        (14, storage(format, V::D2)),
                    ]),
                ]
            }
        };
        let params = (0..(MAX_ITERATIONS as usize + 2))
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("denoise parameters"),
                    size: PARAM_BYTES,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        Ok(Self {
            key,
            kind,
            params,
            history: pair("denoise history", copy),
            moments: pair("denoise moments", none),
            guide: [
                texture("denoise guide", guide_format, 1, false, none),
                texture("denoise guide", guide_format, 1, false, none),
            ],
            acc: texture("denoise accumulation", state_format, state_layers, array, copy),
            tmp: pair("denoise a-trous", none),
            output: (kind == Kind::Relax)
                .then(|| texture("denoise output", format, 1, false, copy)),
            dummy,
            pipelines,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn record(
        &self,
        ctx: &mut RecordCtx<'_>,
        frame: &Frame<'_>,
        input: &PublishedImage,
        depth: &wgpu::TextureView,
        normal: &wgpu::TextureView,
        motion: &wgpu::TextureView,
        meta: Option<&PublishedImage>,
        trace: Option<&PublishedImage>,
    ) -> wgpu::TextureView {
        let ((width, height), layers, _) = self.key;
        let groups = [width.div_ceil(8), height.div_ceil(8)];
        let (cur, prev) = (frame.parity, frame.parity ^ 1);
        fn view(v: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
            wgpu::BindingResource::TextureView(v)
        }
        let gpu = ctx.gpu;
        let write = |index: usize, step: u32, final_pass: bool| {
            gpu.queue().write_buffer(
                &self.params[index],
                0,
                &frame.params((width, height), layers, step, final_pass),
            );
            self.params[index].as_entire_binding()
        };
        match self.kind {
            Kind::Sigma => {
                let p0 = write(0, 0, false);
                let p1 = write(1, 0, true);
                dispatch(ctx, &self.pipelines[SIGMA_TEMPORAL], &[
                    (0, view(&input.view)),
                    (1, view(depth)),
                    (2, view(normal)),
                    (3, view(motion)),
                    (4, view(&self.history[prev])),
                    (5, view(&self.moments[prev])),
                    (6, view(&self.guide[prev])),
                    (7, p0),
                    (8, view(&self.acc)),
                    (9, view(&self.moments[cur])),
                    (10, view(&self.guide[cur])),
                ], [groups[0], groups[1], layers]);
                dispatch(ctx, &self.pipelines[SIGMA_SPATIAL], &[
                    (0, view(&self.acc)),
                    (5, view(&self.moments[cur])),
                    (6, view(&self.guide[cur])),
                    (7, p1),
                    (8, view(&self.history[cur])),
                ], [groups[0], groups[1], layers]);
                self.history[cur].clone()
            }
            Kind::Relax => {
                let iterations = frame.settings.iterations;
                let meta_view = meta.map_or(&self.dummy, |m| &m.view);
                let trace_view = trace.map_or(&self.dummy, |t| &t.view);
                let p_temporal = write(0, 0, false);
                dispatch(ctx, &self.pipelines[RELAX_TEMPORAL], &[
                    (0, view(&input.view)),
                    (1, view(depth)),
                    (2, view(normal)),
                    (3, view(motion)),
                    (4, view(&self.history[prev])),
                    (5, view(&self.moments[prev])),
                    (6, view(&self.guide[prev])),
                    (7, p_temporal),
                    (8, view(&self.acc)),
                    (9, view(&self.moments[cur])),
                    (10, view(&self.guide[cur])),
                    (11, view(meta_view)),
                    (12, view(trace_view)),
                ], [groups[0], groups[1], 1]);
                // Iteration 0 writes the colour history (step 0 = unfiltered copy).
                let p_history = write(1, u32::from(iterations > 0), false);
                dispatch(ctx, &self.pipelines[RELAX_HISTORY], &[
                    (2, view(normal)),
                    (6, view(&self.guide[cur])),
                    (7, p_history),
                    (8, view(&self.history[cur])),
                    (9, view(&self.tmp[0])),
                    (12, view(trace_view)),
                    (13, view(&self.acc)),
                ], [groups[0], groups[1], 1]);
                let mut source = 0usize;
                for i in 1..iterations.saturating_sub(1) {
                    let p = write(1 + i as usize, 1 << i, false);
                    dispatch(ctx, &self.pipelines[RELAX_ATROUS], &[
                        (2, view(normal)),
                        (6, view(&self.guide[cur])),
                        (7, p),
                        (8, view(&self.tmp[source ^ 1])),
                        (12, view(trace_view)),
                        (13, view(&self.tmp[source])),
                    ], [groups[0], groups[1], 1]);
                    source ^= 1;
                }
                let step = if iterations >= 2 { 1 << (iterations - 1) } else { 0 };
                let output = self.output.as_ref().expect("relax output");
                let p_final = write(MAX_ITERATIONS as usize + 1, step, true);
                dispatch(ctx, &self.pipelines[RELAX_FINAL], &[
                    (0, view(&input.view)),
                    (1, view(depth)),
                    (2, view(normal)),
                    (6, view(&self.guide[cur])),
                    (7, p_final),
                    (11, view(meta_view)),
                    (12, view(trace_view)),
                    (13, view(&self.tmp[source])),
                    (14, view(output)),
                ], [groups[0], groups[1], 1]);
                output.clone()
            }
        }
    }
}

fn pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    entry: &str,
    bindings: &[(u32, wgpu::BindingType)],
) -> wgpu::ComputePipeline {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(entry),
        entries: &bindings
            .iter()
            .map(|&(binding, ty)| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty,
                count: None,
            })
            .collect::<Vec<_>>(),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(entry),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(&layout),
        module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn dispatch(
    ctx: &mut RecordCtx<'_>,
    pipeline: &wgpu::ComputePipeline,
    bindings: &[(u32, wgpu::BindingResource<'_>)],
    [x, y, z]: [u32; 3],
) {
    let group = ctx.gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("denoise"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &bindings
            .iter()
            .map(|(binding, resource)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: resource.clone(),
            })
            .collect::<Vec<_>>(),
    });
    let mut pass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("denoise"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &group, &[]);
    pass.dispatch_workgroups(x, y, z);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn denoise_routes_follow_the_signal_links() {
        let mut doc: serde_json::Value =
            serde_json::from_str(include_str!("../../graphs/real-frame-model.json")).unwrap();
        let nodes = doc["nodes"].as_array_mut().unwrap();
        nodes.retain(|n| !n["type"].as_str().unwrap().starts_with("neural.frame"));
        for node in nodes.iter_mut() {
            match node["type"].as_str().unwrap() {
                "neural.area-reconstruction" => {
                    node["id"] = "areaDenoise".into();
                    node["type"] = "denoise.sigma".into();
                }
                "neural.reflection-reconstruction" => {
                    node["id"] = "reflectionDenoise".into();
                    node["type"] = "denoise.relax".into();
                }
                _ => {}
            }
        }
        let links = doc["links"].as_array_mut().unwrap();
        links.retain(|l| !l["to"].as_str().unwrap().starts_with("frameGeneration"));
        for link in links.iter_mut() {
            for end in ["from", "to"] {
                let text = link[end].as_str().unwrap().to_owned();
                let text = text
                    .replace("areaModel.pairs", "areaDenoise.signal")
                    .replace("reflectionModel.incident", "reflectionDenoise.signal");
                link[end] = text.into();
            }
        }
        links.push(serde_json::json!({"from": "ssrTrace.hitMeta", "to": "reflectionDenoise.hitMeta"}));
        links.push(serde_json::json!({"from": "ssrTrace.trace", "to": "reflectionDenoise.trace"}));
        let plan = crate::plan_from_json(&doc.to_string()).expect("denoise graph plans");
        let routes = routes_from_plan(&plan, &Catalog::build()).expect("denoise routes");
        assert_eq!(routes.len(), 2);
        let sigma = routes.iter().find(|r| r.kind == Kind::Sigma).unwrap();
        assert_eq!(sigma.alias, "denoise.sigma#areaDenoise");
        assert_eq!(sigma.producer, ("area.trace#trace".into(), "raw".into()));
        assert_eq!(sigma.consumer, ("area.mask#mask".into(), "src".into()));
        let relax = routes.iter().find(|r| r.kind == Kind::Relax).unwrap();
        assert_eq!(relax.producer, ("ssr.incident#incident".into(), "incident".into()));
        assert_eq!(relax.consumer, ("ssr.replace#replace".into(), "incident".into()));
        assert_eq!(relax.hit_meta, Some(("ssr.trace#ssrTrace".into(), "hitMeta".into())));
        assert_eq!(relax.trace, Some(("ssr.trace#ssrTrace".into(), "trace".into())));
    }
}
