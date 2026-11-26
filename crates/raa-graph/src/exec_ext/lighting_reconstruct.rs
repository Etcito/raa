//! Executor extension `lighting_reconstruct`: installs one `pass.lighting.reconstruct#<id>`
//! pipeline pass per node (catalog family `lighting_reconstruct`).
//!
//! The pass is a member of the HDR colour chain (wiring `chain_type`): the wiring frame binds the
//! chain source (the forward's `color` = d0, a = s) and a pooled destination before it records,
//! and publishes the destination as `(entry, color)` after it. The pass reads the forward layers
//! from the resource table, keeps its own ping-pong history pair (H0 = E rgb, a; H1 = s, linear
//! z, confidence, 0) and runs `lighting_reconstruct.wgsl` once per frame.
//!
//! History lifecycle: textures created at the first record and recreated with the scene size
//! (history invalid); the pair flips only in `submitted` after a frame that wrote it (then
//! valid); `invalidate_history` (camera cut / view discontinuity), a frame whose forward is not
//! in "layers-interlaced" or whose layers are missing (the pass then copies the chain source)
//! make it invalid. An invalid history reads as confidence 0 everywhere.

use crate::catalog::ParamValue;
use crate::error::{Issue, Severity, Stage};
use crate::exec::{ExecExtension, FeatureHandles, NodeStats, ResourceStat};
use crate::plan::{GraphPlan, InputSource, PlanNode};
use crate::registry::exec_wiring::plan_fns::node_entries;
use crate::registry::lighting_reconstruct::{DEBUG_VIEWS, TYPE};
use raa_core::{GpuContext, RaaError};
use raa_lit::LitHandle;
use raa_pipeline::{FeaturePass, PrepareCtx, RecordCtx};
use std::{cell::RefCell, rc::Rc};

const WGSL: &str = include_str!("lighting_reconstruct.wgsl");
const ORDER: u32 = 61;
const PARAM_BYTES: usize = 64;
const HISTORY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// The layer inputs read from the forward, in binding order 1..=4.
const LAYERS: [&str; 4] = ["layerKd", "layerS0", "layerEm", "layerE"];
/// The forward lightingOutput the node accepts on its `color` producer.
const INTERLACED: &str = "layers-interlaced";

pub fn extension() -> Box<dyn ExecExtension> {
    Box::new(LightingReconstruct::default())
}

/// The per-node values of the node's parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Settings {
    /// s hit, s miss, (E,a) hit, (E,a) miss.
    alpha: [f32; 4],
    q_miss: f32,
    decay: f32,
    jitter_correct: bool,
    debug: u32,
}

impl Settings {
    fn from_node(node: &PlanNode) -> Settings {
        let value = |key: &str| node.params.iter().find(|(k, _)| k == key).map(|(_, v)| v);
        let number = |key: &str, default: f64| match value(key) {
            Some(ParamValue::Number(n)) => *n as f32,
            Some(ParamValue::Int(n)) => *n as f32,
            _ => default as f32,
        };
        Settings {
            alpha: [
                number("alphaSHit", 0.985_325_634_479_522_7),
                number("alphaSMiss", 0.980_347_156_524_658_2),
                number("alphaEHit", 0.988_787_829_875_946),
                number("alphaEMiss", 0.979_948_699_474_334_7),
            ],
            q_miss: number("qMiss", 0.3),
            decay: number("decay", 0.98),
            jitter_correct: matches!(value("jitterCorrect"), Some(ParamValue::Bool(true))),
            debug: match value("debug") {
                Some(ParamValue::Enum(view)) => DEBUG_VIEWS
                    .iter()
                    .position(|v| v == view)
                    .unwrap_or(0) as u32,
                _ => 0,
            },
        }
    }
}

/// What the extension reports in `renderGraphStats`.
#[derive(Clone, Copy, Debug, Default)]
struct Report {
    ran: bool,
    history_valid: bool,
    size: (u32, u32),
}

#[derive(Default)]
struct LightingReconstruct {
    /// `(node id, entry, report)` of the installed passes.
    nodes: Vec<(String, String, Rc<RefCell<Report>>)>,
}

fn issue(node: &PlanNode, code: &'static str, port: Option<&str>, message: &str, fix: &str) -> Issue {
    Issue {
        code,
        severity: Severity::Error,
        stage: Stage::Apply,
        node: Some(node.id.clone()),
        port: port.map(str::to_owned),
        param: None,
        link: None,
        message: message.into(),
        fix: fix.into(),
    }
}

/// The producer `(node id, port)` feeding `port` of `node` (a link or a default source).
fn source<'a>(node: &'a PlanNode, port: &str) -> Option<(&'a str, &'a str)> {
    match &node.inputs.iter().find(|i| i.port == port)?.source {
        InputSource::Link {
            node,
            port,
            history: false,
        }
        | InputSource::Default { node, port } => Some((node, port)),
        _ => None,
    }
}

/// The refusal of a reconstruct node whose `color` producer is not a `pass.forward` in
/// lightingOutput "layers-interlaced", or whose layers do not come from that forward.
pub(crate) fn source_refusal(plan: &GraphPlan, node: &PlanNode) -> Option<Issue> {
    let forward = source(node, "color").and_then(|(id, port)| {
        (port == "color")
            .then(|| plan.nodes.iter().find(|n| n.id == id))
            .flatten()
    });
    let interlaced = forward.is_some_and(|forward| {
        forward.type_id == "pass.forward"
            && forward
                .params
                .iter()
                .any(|(k, v)| k == "lightingOutput" && *v == ParamValue::Enum(INTERLACED.into()))
    });
    if !interlaced {
        return Some(issue(
            node,
            "graph.reconstruct_source",
            Some("color"),
            "The colour input must be the `color` output of a pass.forward in lightingOutput \
             \"layers-interlaced\".",
            "Link pass.forward.color (lightingOutput \"layers-interlaced\") into color.",
        ));
    }
    let forward = forward.expect("checked above");
    for layer in LAYERS {
        if source(node, layer) != Some((forward.id.as_str(), layer)) {
            return Some(issue(
                node,
                "graph.reconstruct_source",
                Some(layer),
                &format!("`{layer}` must be linked from the same forward's `{layer}` output."),
                &format!("Link {}.{layer} into {layer}.", forward.id),
            ));
        }
    }
    None
}

impl ExecExtension for LightingReconstruct {
    fn name(&self) -> &'static str {
        "lighting_reconstruct"
    }

    fn apply(
        &mut self,
        plan: &GraphPlan,
        _: Option<&GraphPlan>,
        h: &mut FeatureHandles<'_>,
    ) -> Result<(), Vec<Issue>> {
        let nodes: Vec<&PlanNode> = plan.nodes.iter().filter(|n| n.type_id == TYPE).collect();
        if nodes.is_empty() {
            self.clear(h);
            return Ok(());
        }
        let Some(lit) = h.lit.as_ref().map(|lit| lit.handle()) else {
            return Err(vec![issue(
                nodes[0],
                "graph.unavailable",
                None,
                "pass.lighting.reconstruct requires an installed lit renderer.",
                "Enable the lit renderer for this graph.",
            )]);
        };
        if h.viewports != 1 {
            return Err(vec![issue(
                nodes[0],
                "graph.reconstruct_view",
                None,
                "pass.lighting.reconstruct keeps one history and requires one viewport.",
                "Use a single viewport.",
            )]);
        }
        let catalog = crate::catalog::Catalog::build();
        let entries = node_entries(plan, &catalog);
        let entry_of = |id: &str| {
            entries
                .iter()
                .find(|(node, _)| node == id)
                .map(|(_, entry)| entry.clone())
        };
        let mut prepared = Vec::new();
        for node in nodes {
            if let Some(refusal) = source_refusal(plan, node) {
                return Err(vec![refusal]);
            }
            let mut sources = Vec::new();
            for port in LAYERS.iter().chain(&["depth", "normal", "motion"]) {
                let resolved = source(node, port)
                    .and_then(|(id, out)| Some((entry_of(id)?, out.to_owned())));
                sources.push(resolved);
            }
            let entry = entry_of(&node.id).expect("reconstruct node has a pipeline entry");
            prepared.push((node, entry, sources, Settings::from_node(node)));
        }
        self.clear(h);
        let mut pipeline = h.pipeline.borrow_mut();
        for (node, entry, sources, settings) in prepared {
            let report = Rc::new(RefCell::new(Report::default()));
            let pass = ReconstructPass::new(
                h.gpu,
                entry.clone(),
                lit.clone(),
                sources,
                settings,
                report.clone(),
            );
            pipeline.add(Box::new(pass)).map_err(|error| {
                vec![issue(
                    node,
                    "graph.apply",
                    None,
                    &error.to_string(),
                    "Check the reconstruct pipeline entry.",
                )]
            })?;
            self.nodes.push((node.id.clone(), entry, report));
        }
        Ok(())
    }

    fn clear(&mut self, h: &mut FeatureHandles<'_>) {
        let mut pipeline = h.pipeline.borrow_mut();
        for (_, entry, _) in self.nodes.drain(..) {
            let _ = pipeline.remove(&entry);
        }
    }

    fn stats(&self, out: &mut Vec<NodeStats>) {
        for (id, _, report) in &self.nodes {
            let report = *report.borrow();
            let Some(node) = out.iter_mut().find(|node| node.node == *id) else {
                continue;
            };
            node.ran = report.ran;
            node.history_valid = Some(report.history_valid);
            if report.ran {
                let (width, height) = report.size;
                for port in ["color", "history0", "history1"] {
                    node.resources.push(ResourceStat {
                        port: port.into(),
                        width,
                        height,
                        layers: 1,
                        mips: 1,
                        format: "rgba16float".into(),
                        bytes: u64::from(width) * u64::from(height) * 8,
                    });
                }
            }
        }
    }
}

/// The history pair textures of one size: `[H0, H1]` for each parity.
struct HistoryPair {
    size: (u32, u32),
    views: [[wgpu::TextureView; 2]; 2],
}

impl HistoryPair {
    fn new(gpu: &GpuContext, size: (u32, u32)) -> HistoryPair {
        let make = || {
            gpu.device()
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("raa.lighting.reconstruct.history"),
                    size: wgpu::Extent3d {
                        width: size.0,
                        height: size.1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: HISTORY_FORMAT,
                    usage: wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        HistoryPair {
            size,
            views: [[make(), make()], [make(), make()]],
        }
    }
}

struct ReconstructPass {
    entry: String,
    lit: LitHandle,
    /// `(entry, port)` of layerKd, layerS0, layerEm, layerE, depth, normal, motion.
    sources: Vec<Option<(String, String)>>,
    settings: Settings,
    report: Rc<RefCell<Report>>,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    params: wgpu::Buffer,
    history: Option<HistoryPair>,
    /// The pair index written this frame; the other one is the previous frame's.
    parity: usize,
    valid: bool,
    pending: bool,
    /// The camera jitter (NDC) of the frame that wrote the valid history, and of this frame.
    jitter: [f32; 2],
    pending_jitter: [f32; 2],
}

impl ReconstructPass {
    fn new(
        gpu: &GpuContext,
        entry: String,
        lit: LitHandle,
        sources: Vec<Option<(String, String)>>,
        settings: Settings,
        report: Rc<RefCell<Report>>,
    ) -> ReconstructPass {
        let device = gpu.device();
        use wgpu::TextureViewDimension as V;
        let sampled = |filterable| wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: V::D2,
            multisampled: false,
        };
        let uniform = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let storage = wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: HISTORY_FORMAT,
            view_dimension: V::D2,
        };
        let mut bindings = vec![
            sampled(false),
            sampled(false),
            sampled(false),
            sampled(false),
            sampled(false),
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: V::D2,
                multisampled: false,
            },
            sampled(false),
            sampled(false),
            sampled(true),
            sampled(true),
            wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            uniform,
            uniform,
        ];
        bindings.extend([storage; 3]);
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa.lighting.reconstruct"),
            entries: &bindings
                .iter()
                .enumerate()
                .map(|(binding, ty)| wgpu::BindGroupLayoutEntry {
                    binding: binding as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: *ty,
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raa.lighting.reconstruct"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa.lighting.reconstruct"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("raa.lighting.reconstruct"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("reconstruct"),
            compilation_options: Default::default(),
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("raa.lighting.reconstruct.history"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa.lighting.reconstruct.params"),
            size: PARAM_BYTES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ReconstructPass {
            entry,
            lit,
            sources,
            settings,
            report,
            pipeline,
            layout,
            sampler,
            params,
            history: None,
            parity: 0,
            valid: false,
            pending: false,
            jitter: [0.0; 2],
            pending_jitter: [0.0; 2],
        }
    }

    /// The image published under source `index`, at the frame size.
    fn source(&self, ctx: &RecordCtx<'_>, index: usize) -> Option<wgpu::TextureView> {
        let (entry, port) = self.sources.get(index)?.as_ref()?;
        let image = ctx.resources.get(entry, port)?;
        ((image.desc.width, image.desc.height) == ctx.size).then_some(image.view)
    }

    /// The params uniform (layout of `Params` in lighting_reconstruct.wgsl).
    fn param_bytes(&self, size: (u32, u32), real: bool, phase: [u32; 2], jitter_px: [f32; 2]) -> [u8; PARAM_BYTES] {
        let s = &self.settings;
        let flags = u32::from(self.valid) | (u32::from(real) << 1);
        let words: [u32; 16] = [
            size.0,
            size.1,
            flags,
            s.debug,
            phase[0],
            phase[1],
            0,
            0,
            jitter_px[0].to_bits(),
            jitter_px[1].to_bits(),
            s.q_miss.to_bits(),
            s.decay.to_bits(),
            s.alpha[0].to_bits(),
            s.alpha[1].to_bits(),
            s.alpha[2].to_bits(),
            s.alpha[3].to_bits(),
        ];
        let mut out = [0u8; PARAM_BYTES];
        for (chunk, word) in out.chunks_exact_mut(4).zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        out
    }
}

/// The history reprojection offset in pixels for a jittered G-buffer: the engine jitter is the
/// NDC translation of the projection (`raa_frame.camera_jitter`, NDC y up), so a surface at the
/// unjittered UV u is seen at u + (jx / 2, -jy / 2). The pixel p of frame n was rendered at
/// unjittered p - j_n, so its frame n-1 texel is p - j_n - motion + j_{n-1}:
/// `(prev - cur) x (0.5 W, -0.5 H)` (raa-taa `raa_temporal_raster_uv` uses the same sign).
pub(crate) fn jitter_pixels(previous: [f32; 2], current: [f32; 2], size: (u32, u32)) -> [f32; 2] {
    [
        (previous[0] - current[0]) * 0.5 * size.0 as f32,
        (previous[1] - current[1]) * -0.5 * size.1 as f32,
    ]
}

impl FeaturePass for ReconstructPass {
    fn name(&self) -> &str {
        &self.entry
    }

    fn order(&self) -> u32 {
        ORDER
    }

    fn prepare(&mut self, _ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        self.report.borrow_mut().ran = false;
        Ok(())
    }

    fn submitted(&mut self) {
        if self.pending {
            self.pending = false;
            self.parity ^= 1;
            self.valid = true;
            self.jitter = self.pending_jitter;
        }
    }

    fn invalidate_history(&mut self) {
        self.pending = false;
        self.valid = false;
    }

    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let hdr = ctx.targets.hdr();
        if !hdr.is_bound() {
            return Err(RaaError::Resource(format!(
                "{}: the HDR chain is not bound (the node runs only inside a render graph)",
                self.entry
            )));
        }
        let (src, dst) = (hdr.src().clone(), hdr.dst().clone());
        let interlace = ctx.lighting.interlace();
        let layers: Vec<_> = (0..LAYERS.len()).map(|i| self.source(ctx, i)).collect();
        if !interlace.enabled || layers.iter().any(Option::is_none) {
            // Not composable: pass the chain colour through and drop the history.
            self.pending = false;
            self.valid = false;
            ctx.encoder.copy_texture_to_texture(
                src.texture().as_image_copy(),
                dst.texture().as_image_copy(),
                wgpu::Extent3d {
                    width: src.texture().width().min(dst.texture().width()),
                    height: src.texture().height().min(dst.texture().height()),
                    depth_or_array_layers: 1,
                },
            );
            hdr.swap();
            return Ok(());
        }
        let size = ctx.size;
        if self.history.as_ref().is_none_or(|h| h.size != size) {
            self.history = Some(HistoryPair::new(ctx.gpu, size));
            self.valid = false;
            self.pending = false;
        }
        let jitter = self.lit.camera_jitter_ndc();
        let jitter_px = if self.settings.jitter_correct && self.valid {
            jitter_pixels(self.jitter, jitter, size)
        } else {
            [0.0; 2]
        };
        let bytes = self.param_bytes(size, interlace.real(), interlace.phase(), jitter_px);
        ctx.gpu.queue().write_buffer(&self.params, 0, &bytes);
        let depth = self
            .source(ctx, 4)
            .unwrap_or_else(|| ctx.targets.depth().clone());
        let normal = self
            .source(ctx, 5)
            .unwrap_or_else(|| ctx.targets.normal().clone());
        let motion = self
            .source(ctx, 6)
            .unwrap_or_else(|| ctx.targets.motion().clone());
        let frame = self.lit.frame_buffer();
        let history = self.history.as_ref().expect("history created above");
        let (cur, prev) = (&history.views[self.parity], &history.views[self.parity ^ 1]);
        let layers: Vec<wgpu::TextureView> = layers.into_iter().flatten().collect();
        let view = wgpu::BindingResource::TextureView;
        let resources = [
            view(&src),
            view(&layers[0]),
            view(&layers[1]),
            view(&layers[2]),
            view(&layers[3]),
            view(&depth),
            view(&normal),
            view(&motion),
            view(&prev[0]),
            view(&prev[1]),
            wgpu::BindingResource::Sampler(&self.sampler),
            self.params.as_entire_binding(),
            frame.as_entire_binding(),
            view(&cur[0]),
            view(&cur[1]),
            view(&dst),
        ];
        let group = ctx.gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa.lighting.reconstruct"),
            layout: &self.layout,
            entries: &resources
                .into_iter()
                .enumerate()
                .map(|(binding, resource)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource,
                })
                .collect::<Vec<_>>(),
        });
        {
            let mut pass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("raa.lighting.reconstruct"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(size.0.div_ceil(8), size.1.div_ceil(8), 1);
        }
        hdr.swap();
        self.pending = true;
        self.pending_jitter = jitter;
        let mut report = self.report.borrow_mut();
        report.ran = true;
        report.history_valid = self.valid;
        report.size = size;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jitter_correction_is_previous_minus_current_in_pixels() {
        assert_eq!(jitter_pixels([0.002, -0.004], [0.0, 0.0], (1000, 500)), [1.0, 1.0]);
        assert_eq!(jitter_pixels([0.0; 2], [0.0; 2], (1920, 1080)), [0.0, 0.0]);
    }
}
