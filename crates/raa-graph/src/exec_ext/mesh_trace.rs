//! Instanced, graph-wired use of the renderer's shared live mesh trace scene.

use crate::catalog::{Catalog, ParamValue};
use crate::error::{Issue, Severity, Stage};
use crate::exec::{ExecExtension, FeatureHandles, NodeStats, ResourceStat};
use crate::plan::{GraphPlan, InputSource, PlanNode};
use crate::registry::exec_wiring::plan_fns::{mesh_trace_type, node_entries};
use raa_core::RaaError;
use raa_pipeline::{GraphFrame, ImageDesc, RecordCtx};
use raa_sdf_trace::mesh::{
    GraphMeshTrace, LightingOutput, MeshTraceFeature, MeshTraceSettings, PanelSampling, PathMode,
};
use std::{cell::RefCell, rc::Rc};

const NAME: &str = "mesh_trace";

pub fn extension() -> Box<dyn ExecExtension> {
    Box::new(MeshTrace::default())
}

#[derive(Default)]
struct MeshTrace {
    nodes: Vec<(String, String)>,
    stats: Rc<RefCell<Vec<(String, ImageDesc, &'static str)>>>,
}

struct Binding {
    consumer: String,
    input: String,
    producer: String,
    output: String,
}

struct MeshFrame {
    bindings: Vec<Binding>,
    nodes: Vec<(String, String)>,
    stats: Rc<RefCell<Vec<(String, ImageDesc, &'static str)>>>,
}

impl GraphFrame for MeshFrame {
    fn begin(&mut self, _: &raa_pipeline::ResourceTable) {
        self.stats.borrow_mut().clear();
    }

    fn before_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        for binding in self
            .bindings
            .iter()
            .filter(|binding| binding.consumer == pass)
        {
            let image = ctx
                .resources
                .get(&binding.producer, &binding.output)
                .ok_or_else(|| {
                    RaaError::Resource(format!(
                        "{pass}.{}: {}.{} did not publish an image",
                        binding.input, binding.producer, binding.output
                    ))
                })?;
            ctx.resources.bind_input(pass, &binding.input, image);
        }
        Ok(())
    }

    fn after_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if let Some((id, _)) = self.nodes.iter().find(|(_, entry)| entry == pass) {
            let mut stats = self.stats.borrow_mut();
            for port in ["color", "radiance"] {
                if let Some(image) = ctx.resources.get(pass, port) {
                    stats.push((id.clone(), image.desc, port));
                }
            }
        }
        Ok(())
    }
}

impl ExecExtension for MeshTrace {
    fn name(&self) -> &'static str {
        NAME
    }

    fn apply(
        &mut self,
        plan: &GraphPlan,
        _: Option<&GraphPlan>,
        h: &mut FeatureHandles<'_>,
    ) -> Result<(), Vec<Issue>> {
        let nodes: Vec<_> = plan
            .nodes
            .iter()
            .filter(|n| mesh_trace_type(&n.type_id))
            .collect();
        if nodes.is_empty() {
            self.clear(h);
            return Ok(());
        }
        if h.viewports != 1 {
            return Err(issue(
                nodes[0],
                "graph.mesh_view",
                "Mesh trace nodes require one viewport.",
                "Use a single viewport.",
            ));
        }
        let scene = h.handles.mesh_trace.as_ref().cloned().ok_or_else(|| {
            issue(
                nodes[0],
                "graph.mesh_scene",
                "The renderer has no mesh trace scene.",
                "Install the live mesh trace scene.",
            )
        })?;
        let catalog = Catalog::build();
        let entries = node_entries(plan, &catalog);
        let entry = |id: &str| {
            entries
                .iter()
                .find(|(node, _)| node == id)
                .map(|(_, entry)| entry)
        };
        let mut prepared = Vec::new();
        let mut bindings = Vec::new();
        for node in nodes {
            let options = options(node);
            if options.settings.path_mode == PathMode::Learned && !scene.borrow().path_model_ready()
            {
                return Err(issue(
                    node,
                    "trace.path_model_missing",
                    "Fitted specular weights are missing.",
                    "Load the fitted weights with setMeshPathModel before applying the graph.",
                ));
            }
            let target = entry(&node.id)
                .expect("mesh trace node has a pipeline entry")
                .clone();
            for input in node.inputs.iter().filter(|input| input.port != "color") {
                let (InputSource::Link {
                    node: producer,
                    port,
                    history: false,
                }
                | InputSource::Default {
                    node: producer,
                    port,
                }) = &input.source
                else {
                    return Err(issue(
                        node,
                        "graph.mesh_input",
                        "A runtime image input is missing.",
                        "Connect each mesh trace input to a live image output.",
                    ));
                };
                let producer = entry(producer).ok_or_else(|| {
                    issue(
                        node,
                        "graph.mesh_input",
                        "The image producer has no pipeline entry.",
                        "Connect the input to an executed image producer.",
                    )
                })?;
                bindings.push(Binding {
                    consumer: target.clone(),
                    input: input.port.clone(),
                    producer: producer.clone(),
                    output: port.clone(),
                });
            }
            prepared.push((node, target, options));
        }

        self.clear(h);
        let mut pipeline = h.pipeline.borrow_mut();
        for (node, entry, options) in prepared {
            MeshTraceFeature::install_graph(&mut pipeline, entry.clone(), scene.clone(), options)
                .map_err(|error| {
                issue(
                    node,
                    "graph.mesh_install",
                    error.message(),
                    "Check mesh trace settings.",
                )
            })?;
            self.nodes.push((node.id.clone(), entry));
        }
        pipeline
            .add_graph_frame(
                NAME,
                Box::new(MeshFrame {
                    bindings,
                    nodes: self.nodes.clone(),
                    stats: self.stats.clone(),
                }),
            )
            .map_err(|error| {
                issue(
                    plan.nodes.first().expect("nonempty plan"),
                    "graph.mesh_install",
                    error.message(),
                    "Check the mesh trace pipeline.",
                )
            })?;
        Ok(())
    }

    fn clear(&mut self, h: &mut FeatureHandles<'_>) {
        let mut pipeline = h.pipeline.borrow_mut();
        pipeline.remove_graph_frame(NAME);
        for (_, entry) in self.nodes.drain(..) {
            let _ = pipeline.remove(&entry);
        }
        self.stats.borrow_mut().clear();
    }

    fn stats(&self, out: &mut Vec<NodeStats>) {
        for (id, desc, port) in self.stats.borrow().iter() {
            if let Some(node) = out.iter_mut().find(|node| node.node == *id) {
                node.ran = true;
                node.resources.push(ResourceStat {
                    port: (*port).into(),
                    width: desc.width,
                    height: desc.height,
                    layers: desc.layers,
                    mips: desc.mips,
                    format: "rgba16float".into(),
                    bytes: u64::from(desc.width) * u64::from(desc.height) * 8,
                });
            }
        }
    }
}

fn options(node: &PlanNode) -> GraphMeshTrace {
    let value = |key| {
        node.params
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
    };
    let int = |key, default| match value(key) {
        Some(ParamValue::Int(value)) => *value as u32,
        _ => default,
    };
    let choice = |key| match value(key) {
        Some(ParamValue::Enum(value)) => value.as_str(),
        _ => "",
    };
    let default = MeshTraceSettings::default();
    let settings = MeshTraceSettings {
        reflections: true,
        reference: node.type_id == "mesh.reference",
        path_mode: if node.type_id == "mesh.fitted-specular" {
            PathMode::Learned
        } else {
            PathMode::Off
        },
        confidence_threshold: match value("confidenceThreshold") {
            Some(ParamValue::Number(value)) => *value as f32,
            _ => default.confidence_threshold,
        },
        reflection_bounces: int("reflectionBounces", default.reflection_bounces),
        path_bounces: int("pathBounces", default.path_bounces),
        lighting_output: match choice("lightingOutput") {
            "direct" => LightingOutput::Direct,
            "indirect" => LightingOutput::Indirect,
            "bounce" => LightingOutput::Bounce(int("lightingDepth", 0)),
            _ => LightingOutput::Combined,
        },
        panel_sampling: match choice("panelSampling") {
            "single" => PanelSampling::Single,
            "independent" => PanelSampling::Independent,
            _ => PanelSampling::Stable,
        },
        ..default
    };
    GraphMeshTrace {
        settings,
        reference_sample: (
            int("seed", 0),
            int("samples", raa_sdf_trace::mesh::REFERENCE_SPP),
        ),
    }
}

fn issue(node: &PlanNode, code: &'static str, message: &str, fix: &str) -> Vec<Issue> {
    vec![Issue {
        code,
        severity: Severity::Error,
        stage: Stage::Apply,
        node: Some(node.id.clone()),
        port: None,
        param: None,
        link: None,
        message: message.into(),
        fix: fix.into(),
    }]
}
