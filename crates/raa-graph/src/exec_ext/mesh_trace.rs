//! Instanced, graph-wired use of the renderer's shared live mesh trace scene.

use crate::catalog::{Catalog, ParamValue};
use crate::error::{Issue, Severity, Stage};
use crate::exec::{ExecExtension, FeatureHandles, NodeStats, ResourceStat};
use crate::mesh_indirect::{
    SPHERICAL_INPUT, SPHERICAL_OUTPUT, VOLUME_PORT, VOLUME_TYPE, mesh_grid_consumer, mesh_grid_port,
};
use crate::plan::{GraphPlan, InputSource, PlanNode};
use crate::registry::exec_wiring::plan_fns::{mesh_trace_type, node_entries};
use crate::runtime_lighting::SPHERICAL_TYPE;
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
            let grids = indirect_grids(plan, node)?;
            for input in node
                .inputs
                .iter()
                .filter(|input| input.port != "color" && !mesh_grid_port(&input.port))
            {
                if input.port == "material" && matches!(input.source, InputSource::Absent) {
                    continue;
                }
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
            prepared.push((node, target, options, grids));
        }

        self.clear(h);
        let mut pipeline = h.pipeline.borrow_mut();
        for (node, entry, options, grids) in prepared {
            MeshTraceFeature::install_graph_with_indirect(
                &mut pipeline,
                entry.clone(),
                scene.clone(),
                options,
                grids.spherical,
                grids.volume,
            )
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

#[derive(Debug, PartialEq, Eq)]
struct IndirectGrids {
    spherical: bool,
    volume: bool,
}

fn indirect_grids(plan: &GraphPlan, node: &PlanNode) -> Result<IndirectGrids, Vec<Issue>> {
    if !mesh_grid_consumer(&node.type_id) {
        // The reference follows its existing integrator and has no graph grid controls.
        return Ok(IndirectGrids {
            spherical: false,
            volume: true,
        });
    }
    let source_before = |id: Option<&str>, ty: &str| {
        plan.nodes.iter().any(|source| {
            id.is_none_or(|id| source.id == id)
                && source.type_id == ty
                && source.order_index < node.order_index
        })
    };
    let connected = |input: &str, output: &str, ty: &str| {
        let Some(input) = node.inputs.iter().find(|candidate| candidate.port == input) else {
            return Ok(false);
        };
        match &input.source {
            InputSource::Absent => Ok(false),
            InputSource::Link {
                node,
                port,
                history: false,
            }
            | InputSource::Default { node, port }
                if port == output && source_before(Some(node), ty) =>
            {
                Ok(true)
            }
            _ => Err(issue(
                node,
                "graph.mesh_grid",
                "A mesh grid input has no matching producer before the trace.",
                "Connect the input to its grid producer and regenerate any frozen graph.",
            )),
        }
    };
    // Frozen literals created before these optional inputs used the live volume pass only.
    let legacy = !node.inputs.iter().any(|input| mesh_grid_port(&input.port));
    Ok(IndirectGrids {
        spherical: connected(SPHERICAL_INPUT, SPHERICAL_OUTPUT, SPHERICAL_TYPE)?,
        volume: connected(VOLUME_PORT, VOLUME_PORT, VOLUME_TYPE)?
            || legacy && source_before(None, VOLUME_TYPE),
    })
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
            "diffuse-indirect" => LightingOutput::IndirectDiffuse,
            "rough-specular" => LightingOutput::IndirectSpecular,
            "rough-grid" => LightingOutput::RoughGrid,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_mesh_literals_keep_volume_and_invalid_grid_producers_are_refused() {
        let fixture = include_str!("../../tests/fixtures/geometry-reflections.json");
        let plan = crate::plan_from_json(fixture).unwrap();
        let mut mesh = plan
            .nodes
            .iter()
            .find(|node| node.id == "mirrors")
            .unwrap()
            .clone();
        mesh.inputs.retain(|input| !mesh_grid_port(&input.port));
        assert_eq!(
            indirect_grids(&plan, &mesh).unwrap(),
            IndirectGrids {
                spherical: false,
                volume: true,
            }
        );

        mesh.inputs.push(crate::plan::PlanInput {
            port: SPHERICAL_INPUT.into(),
            source: InputSource::Link {
                node: "volume".into(),
                port: SPHERICAL_OUTPUT.into(),
                history: false,
            },
        });
        let issues = indirect_grids(&plan, &mesh).unwrap_err();
        assert_eq!(issues[0].code, "graph.mesh_grid");
        assert_eq!(issues[0].node.as_deref(), Some("mirrors"));
    }
}
