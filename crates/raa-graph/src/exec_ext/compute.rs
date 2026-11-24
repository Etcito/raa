//! Executor extension `compute`: per-node passes and device-free plan helpers.

use crate as raa_graph;
use crate::catalog::{Catalog, GraphFormat, ParamValue, PortType, SizeSpec};
use crate::error::{Issue, Severity, Stage};
use crate::exec::{ExecExtension, FeatureHandles, NodeStats, ResourceStat};
use crate::format::{ComputeDecl, ExtentBasis, NodeDoc};
use raa_loader::json::JsonValue;
use std::{cell::RefCell, rc::Rc};
#[path = "compute/pass.rs"]
mod pass;
use crate::plan::{GraphPlan, InputSource, PlanHistory, PlanNode, PlanPortRef, PlanResource};
use raa_core::RaaError;
use raa_pipeline::{
    GraphFrame, ImageDesc, ImageDim, PrepareCtx, PublishedImage, RecordCtx, ResourceTable,
};

#[allow(dead_code, clippy::duplicate_mod)]
#[path = "wiring/plan_fns.rs"]
mod plan_fns;

#[allow(dead_code)]
#[path = "compute/plan.rs"]
mod plan;

/// The extension, discovered by build.rs.
pub fn extension() -> Box<dyn ExecExtension> {
    Box::new(ComputeExtension::default())
}

#[derive(Default)]
pub(crate) struct ComputeExtension {
    nodes: Vec<Rc<RefCell<pass::ComputeNodeState>>>,
    histories: Rc<RefCell<Vec<ComputeHistory>>>,
    frame_added: bool,
}

struct ComputeHistory {
    consumer: String,
    input: String,
    producer: String,
    output: String,
    images: Vec<(wgpu::Texture, wgpu::TextureView)>,
    desc: Option<ImageDesc>,
    current: usize,
    valid: bool,
    copied: bool,
}

impl ComputeHistory {
    fn flip(&mut self) {
        if self.copied {
            self.current = 1 - self.current;
            self.valid = true;
            self.copied = false;
        }
    }
}

impl ExecExtension for ComputeExtension {
    fn name(&self) -> &'static str {
        "compute"
    }

    fn apply(
        &mut self,
        graph: &GraphPlan,
        _previous: Option<&GraphPlan>,
        handles: &mut FeatureHandles<'_>,
    ) -> Result<(), Vec<Issue>> {
        let nodes = plan::compute_nodes(graph);
        if !nodes.is_empty() && handles.lit.is_none() {
            let node = &nodes[0].id;
            return Err(vec![Issue {
                code: "graph.unavailable",
                severity: Severity::Error,
                stage: Stage::Structure,
                node: Some(node.clone()),
                port: None,
                param: None,
                link: None,
                message: "compute nodes require an installed lit renderer".to_owned(),
                fix: "Enable the lit renderer for this graph.".to_owned(),
            }]);
        }
        let catalog = Catalog::build();
        let lit = handles.lit.as_ref().map(|lit| lit.handle().frame_buffer());
        let mut prepared = Vec::new();
        for node in nodes {
            let Some(compute) = &node.compute else {
                continue;
            };
            for output in &compute.outputs {
                let Some(extent) = &output.extent else {
                    continue;
                };
                let valid_basis = match &extent.basis {
                    ExtentBasis::Parent(key) => {
                        compute.inputs.iter().any(|input| {
                            input.key == *key
                                && matches!(input.ty, PortType::Image2d | PortType::Depth2d)
                        }) && node.inputs.iter().any(|input| {
                            input.port == *key
                                && matches!(input.source, InputSource::Link { history: false, .. })
                        })
                    }
                    ExtentBasis::Display | ExtentBasis::Scene => true,
                };
                if extent.numerator == 0 || output.size.is_some() || !valid_basis {
                    return Err(vec![Issue {
                        code: "graph.apply",
                        severity: Severity::Error,
                        stage: Stage::Apply,
                        node: Some(node.id.clone()),
                        port: Some(output.key.clone()),
                        param: None,
                        link: None,
                        message: "Invalid compute output extent".into(),
                        fix: "Use a positive scale, omit legacy size and link a current-frame image input when using the parent basis.".into(),
                    }]);
                }
            }
            let params = node
                .params
                .iter()
                .map(|(key, value)| (key.clone(), json_value(value)))
                .collect();
            let doc = NodeDoc {
                id: node.id.clone(),
                type_id: node.type_id.clone(),
                version: None,
                params,
                compute: Some(ComputeDecl {
                    inputs: compute.inputs.clone(),
                    outputs: compute.outputs.clone(),
                    math: JsonValue::Null,
                    wgsl: compute.wgsl.clone(),
                }),
            };
            let module = crate::registry::compute::codegen::generate_module(&doc)
                .map_err(|mut issue| {
                    issue.code = "graph.apply";
                    issue.stage = Stage::Apply;
                    issue.node = Some(node.id.clone());
                    vec![issue]
                })?
                .0;
            prepared.push((node, module));
        }
        let mut next = Vec::new();
        let mut installed: Vec<String> = Vec::new();
        let mut removed = Vec::new();
        for (node, module) in &prepared {
            if let Some(old) = self.nodes.iter().find(|old| {
                let old = old.borrow();
                old.node == node.id && old.module == *module
            }) {
                next.push(old.clone());
                continue;
            }
            let Some(frame) = lit.as_ref() else { continue };
            let entry = plan::compute_entry(&node.id);
            if self.nodes.iter().any(|old| old.borrow().node == node.id)
                && let Ok(old) = handles.pipeline.borrow_mut().remove(&entry)
            {
                removed.push(old);
            }
            let pass = pass::ComputeNodePass::new(
                handles.gpu,
                node,
                module.clone(),
                plan::compute_inputs(graph, &catalog, node),
                frame,
                handles.pipeline.borrow().surface_format(),
            );
            let state = pass.state.clone();
            let result = handles.pipeline.borrow_mut().add(Box::new(pass));
            if let Err(error) = result {
                for name in installed {
                    let _ = handles.pipeline.borrow_mut().remove(&name);
                }
                for old in removed {
                    let _ = handles.pipeline.borrow_mut().add(old);
                }
                return Err(vec![Issue {
                    code: "graph.apply",
                    severity: Severity::Error,
                    stage: Stage::Apply,
                    node: Some(node.id.clone()),
                    port: None,
                    param: None,
                    link: None,
                    message: error.to_string(),
                    fix: "Install the compute pass.".into(),
                }]);
            }
            installed.push(entry);
            next.push(state);
        }
        for old in &self.nodes {
            if !next.iter().any(|new| Rc::ptr_eq(old, new))
                && !next
                    .iter()
                    .any(|new| new.borrow().node == old.borrow().node)
            {
                let _ = handles
                    .pipeline
                    .borrow_mut()
                    .remove(&plan::compute_entry(&old.borrow().node));
            }
        }
        for (node, _) in &prepared {
            if let Some(state) = next.iter().find(|state| {
                state.borrow().node == node.id
                    && self.nodes.iter().any(|old| Rc::ptr_eq(old, state))
            }) {
                state.borrow_mut().update(
                    handles.gpu,
                    node,
                    plan::compute_inputs(graph, &catalog, node),
                );
            }
        }
        self.nodes = next;
        let resets = plan_fns::history_resets(_previous, graph);
        let mut old_histories = std::mem::take(&mut *self.histories.borrow_mut());
        let mut histories = Vec::new();
        for (consumer, input, producer, output) in plan::history_inputs(graph) {
            let key = format!("{consumer}.{input}");
            let carry = if !resets.contains(&key) {
                old_histories.iter().position(|old| {
                    old.consumer == consumer
                        && old.input == input
                        && old.producer == producer
                        && old.output == output
                })
            } else {
                None
            };
            histories.push(carry.map_or_else(
                || ComputeHistory {
                    consumer,
                    input,
                    producer,
                    output,
                    images: Vec::new(),
                    desc: None,
                    current: 0,
                    valid: false,
                    copied: false,
                },
                |index| old_histories.remove(index),
            ));
        }
        *self.histories.borrow_mut() = histories;
        let has_histories = !self.histories.borrow().is_empty();
        let mut pipeline = handles.pipeline.borrow_mut();
        if self.frame_added {
            pipeline.remove_graph_frame("compute");
            self.frame_added = false;
        }
        if has_histories {
            pipeline
                .add_graph_frame(
                    "compute",
                    Box::new(ComputeHistoryFrame {
                        histories: self.histories.clone(),
                        states: self.nodes.clone(),
                        gi: handles.handles.gi.clone(),
                        dummy: zero_history(handles.gpu),
                    }),
                )
                .map_err(|error| vec![compute_apply_issue(&error)])?;
            self.frame_added = true;
        }
        Ok(())
    }

    fn clear(&mut self, handles: &mut FeatureHandles<'_>) {
        handles.pipeline.borrow_mut().remove_graph_frame("compute");
        self.frame_added = false;
        self.histories.borrow_mut().clear();
        for node in self.nodes.drain(..) {
            let _ = handles
                .pipeline
                .borrow_mut()
                .remove(&plan::compute_entry(&node.borrow().node));
        }
    }

    fn frame(&mut self, _handles: &mut FeatureHandles<'_>) -> Result<(), RaaError> {
        Ok(())
    }

    fn stats(&self, out: &mut Vec<NodeStats>) {
        for node in &self.nodes {
            let stats = node.borrow().stats();
            if let Some(existing) = out.iter_mut().find(|existing| existing.node == stats.node) {
                *existing = stats;
            } else {
                out.push(stats);
            }
        }
    }
}

fn zero_history(gpu: &raa_core::GpuContext) -> PublishedImage {
    let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("raa.compute.history.zero"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    gpu.queue().write_texture(
        texture.as_image_copy(),
        &[0; 8],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(8),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&Default::default());
    PublishedImage {
        view,
        desc: ImageDesc::new(ImageDim::D2, wgpu::TextureFormat::Rgba16Float, 1, 1, 1, 1),
    }
}

fn compute_apply_issue(error: &RaaError) -> Issue {
    Issue {
        code: "graph.apply",
        severity: Severity::Error,
        stage: Stage::Apply,
        node: None,
        port: None,
        param: None,
        link: None,
        message: error.to_string(),
        fix: "Install the compute history frame.".into(),
    }
}

struct ComputeHistoryFrame {
    histories: Rc<RefCell<Vec<ComputeHistory>>>,
    states: Vec<Rc<RefCell<pass::ComputeNodeState>>>,
    gi: Option<raa_gi::GiHandle>,
    dummy: PublishedImage,
}

impl GraphFrame for ComputeHistoryFrame {
    fn begin(&mut self, _table: &ResourceTable) {
        if self
            .gi
            .as_ref()
            .is_some_and(|gi| !gi.shared().camera_valid())
        {
            for history in self.histories.borrow_mut().iter_mut() {
                history.valid = false;
            }
        }
        for state in &self.states {
            state.borrow_mut().history_pairs_valid = false;
        }
    }

    fn before_prepare(&mut self, pass: &str, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let histories = self.histories.borrow();
        let pairs: Vec<_> = histories.iter().filter(|h| h.consumer == pass).collect();
        for history in &pairs {
            if history.valid
                && history.images.len() == 2
                && let Some((_, view)) = history.images.get(1 - history.current)
            {
                let desc = history.desc.expect("allocated history descriptor");
                ctx.resources.bind_input(
                    pass,
                    &history.input,
                    PublishedImage {
                        view: view.clone(),
                        desc,
                    },
                );
            } else {
                ctx.resources
                    .bind_input(pass, &history.input, self.dummy.clone());
            }
        }
        if !pairs.is_empty() {
            let mut state = self
                .states
                .iter()
                .find(|s| plan::compute_entry(&s.borrow().node) == pass)
                .map(|s| s.borrow_mut());
            if let Some(state) = state.as_mut() {
                state.history_pairs_valid = pairs.iter().all(|h| h.valid && h.images.len() == 2);
            }
        }
        Ok(())
    }

    fn after_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        for history in self
            .histories
            .borrow_mut()
            .iter_mut()
            .filter(|h| h.producer == pass)
        {
            let Some(image) = ctx.resources.get(pass, &history.output) else {
                continue;
            };
            if history.desc != Some(image.desc) {
                history.images.clear();
                for _ in 0..2 {
                    let texture = ctx.gpu.device().create_texture(&wgpu::TextureDescriptor {
                        label: Some("raa.compute.history"),
                        size: wgpu::Extent3d {
                            width: image.desc.width,
                            height: image.desc.height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: image.desc.format,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    });
                    let view = texture.create_view(&Default::default());
                    history.images.push((texture, view));
                }
                history.desc = Some(image.desc);
                history.current = 0;
                history.valid = false;
            }
            let (texture, _) = &history.images[history.current];
            ctx.encoder.copy_texture_to_texture(
                image.view.texture().as_image_copy(),
                texture.as_image_copy(),
                wgpu::Extent3d {
                    width: image.desc.width,
                    height: image.desc.height,
                    depth_or_array_layers: 1,
                },
            );
            history.copied = true;
        }
        Ok(())
    }

    fn end(&mut self, _table: &ResourceTable) {
        for history in self.histories.borrow_mut().iter_mut() {
            history.flip();
        }
    }
}

fn json_value(value: &ParamValue) -> JsonValue {
    match value {
        ParamValue::Number(v) => JsonValue::Number(*v),
        ParamValue::Bool(v) => JsonValue::Bool(*v),
        ParamValue::Vec2(v) => JsonValue::Array(v.iter().copied().map(JsonValue::Number).collect()),
        ParamValue::Vec3(v) => JsonValue::Array(v.iter().copied().map(JsonValue::Number).collect()),
        ParamValue::Vec4(v) => JsonValue::Array(v.iter().copied().map(JsonValue::Number).collect()),
        _ => JsonValue::Null,
    }
}
