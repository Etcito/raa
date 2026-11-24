//! Executor extension `wiring` (S57, module raa-graph-wiring): the colour chain with its pooled
//! resources, the link order, and the multi-instance post passes `<entry>#<nodeId>` that the
//! second and later nodes of a post type drive. The device-free functions live in
//! `wiring/plan_fns.rs`.

use crate as raa_graph;
use crate::catalog::{Catalog, FeatureId, NodeTypeSpec, ParamBinding, ParamValue};
use crate::error::{Issue, Severity, Stage};
use crate::exec::{
    ExecExtension, FeatureHandles, HandleSet, NodeStats, ResourceStat, default_values,
    merge_values, write_values,
};
use crate::plan::InputSource;
use crate::plan::{GraphPlan, PlanHistory, PlanNode, PlanPortRef, PlanResource};
use raa_core::RaaError;
use raa_pipeline::{
    FeaturePass, GraphFrame, ImageDesc, ImageDim, PrepareCtx, PublishedImage, RecordCtx,
    ResourceTable,
};
use std::{cell::RefCell, rc::Rc};

#[allow(dead_code)] // the wiring cards call these; the tests compile the same file for themselves
#[path = "wiring/plan_fns.rs"]
pub(crate) mod plan_fns;

// Re-exported for the wiring cards; `apply` and `histories` call them.
#[allow(unused_imports)]
pub use plan_fns::{history_resets, pool_resources};

/// The post node types that the extension installs as instance passes (S57).
const INSTANCE_TYPES: [&str; 7] = [
    "pass.ssr",
    "pass.dof",
    "pass.sunshafts",
    "pass.bloom",
    "pass.grading",
    "pass.cel.resolve",
    "pass.taa",
];

/// The extension, discovered by build.rs (stem `wiring`, registered after `views`).
pub fn extension() -> Box<dyn ExecExtension> {
    Box::new(WiringExtension::new())
}

/// The handle that drives one instance pass, by feature.
#[derive(Clone)]
pub(crate) enum InstanceHandle {
    Ssr(raa_ssr::SsrHandle),
    Dof(raa_dof::DofHandle),
    Shafts(raa_shafts::ShaftsHandle),
    Bloom(raa_bloom::BloomHandle),
    Grading(raa_grading::GradingHandle),
    Cel(raa_cel::CelHandle),
    Taa(raa_taa::TaaHandle),
}

impl InstanceHandle {
    /// The node type whose pass this handle drives.
    fn type_id(&self) -> &'static str {
        match self {
            Self::Ssr(_) => "pass.ssr",
            Self::Dof(_) => "pass.dof",
            Self::Shafts(_) => "pass.sunshafts",
            Self::Bloom(_) => "pass.bloom",
            Self::Grading(_) => "pass.grading",
            Self::Cel(_) => "pass.cel.resolve",
            Self::Taa(_) => "pass.taa",
        }
    }

    /// Sets a tunable on the instance; bloom, grading and cel have no tunable handle.
    fn set_tunable(&self, key: &str, value: f64) -> Result<(), RaaError> {
        match self {
            Self::Ssr(h) => h.set_tunable(key, value),
            Self::Dof(h) => h.set_tunable(key, value),
            Self::Shafts(h) => h.set_tunable(key, value),
            Self::Taa(h) => h.set_tunable(key, value),
            Self::Bloom(_) | Self::Grading(_) | Self::Cel(_) => Ok(()),
        }
    }
}

/// Handoff: `WiringExtension` is taken over by card 'Wiring 5/5: executor-owned histories', which
/// reads the instance table `WiringExtension.instances` and calls `TaaHandle::invalidate` on its
/// `InstanceHandle::Taa` values. Today `apply` rebuilds that table (node id, pipeline entry,
/// handle) and `clear` empties it; nothing invalidates a TAA instance.
///
/// Handoff: `WiringExtension` serves 'Wiring 4/5: multi-instance post passes'; today it
/// applies colour bindings and link order using the shared frame state. The upstream skeleton
/// handoff is at `exec_ext/wiring.rs` `WiringExtension` (Wiring 2/5).
pub(crate) struct WiringExtension {
    catalog: Catalog,
    state: Rc<RefCell<State>>,
    instances: Vec<(String, String, InstanceHandle)>,
    instance_stats: Vec<(String, bool, Vec<ResourceStat>)>,
}

impl WiringExtension {
    /// The extension with its own catalog.
    fn new() -> WiringExtension {
        WiringExtension {
            catalog: Catalog::build(),
            state: Rc::new(RefCell::new(State::default())),
            instances: Vec::new(),
            instance_stats: Vec::new(),
        }
    }

    /// The pipeline entry of the stored instance of node `id`.
    fn entry_of_instance(&self, id: &str) -> Option<String> {
        self.instances
            .iter()
            .find(|(node, _, _)| node.as_str() == id)
            .map(|(_, entry, _)| entry.clone())
    }

    /// Installs, removes, enables and writes the instance passes of `plan`, then sets the order and
    /// the frame. Any refusal is returned; `rollback` undoes what this call did.
    fn commit(
        &mut self,
        plan: &GraphPlan,
        entries: &[(String, String)],
        instances: &[(String, String)],
        layout: Layout,
        h: &mut FeatureHandles<'_>,
        undo: &mut Undo,
    ) -> Result<(), Vec<Issue>> {
        let kept: Vec<(String, String)> = self
            .instances
            .iter()
            .map(|(id, _, handle)| (id.clone(), handle.type_id().to_owned()))
            .collect();
        let (install, keep, remove) = plan_fns::instance_plan(&kept, instances);
        for id in &remove {
            let Some(entry) = self.entry_of_instance(id) else {
                continue;
            };
            let pass = h
                .pipeline
                .borrow_mut()
                .remove(&entry)
                .map_err(|e| apply_issue(Some(id.as_str()), None, e.message()))?;
            undo.removed.push((entry, pass));
        }
        self.instances.retain(|(id, _, _)| !remove.contains(id));
        for (id, type_id) in &install {
            let Some(entry) = entry_of(entries, id) else {
                continue;
            };
            let instance = install_instance(type_id, id, h)
                .map_err(|e| apply_issue(Some(id.as_str()), None, e.message()))?;
            undo.installed.push(entry.clone());
            self.instances.push((id.clone(), entry, instance));
        }
        for (id, _) in &keep {
            if let Some(entry) = self.entry_of_instance(id) {
                h.pipeline
                    .borrow_mut()
                    .set_enabled(&entry, true)
                    .map_err(|e| apply_issue(Some(id.as_str()), None, e.message()))?;
            }
        }
        write_instances(plan, &self.catalog, &self.instances, h)?;
        write_installed_tunables(plan, &self.catalog, h)?;
        let mut pipeline = h.pipeline.borrow_mut();
        let order = plan_fns::link_order(plan, entries, &pipeline.passes());
        let names: Vec<&str> = order.iter().map(String::as_str).collect();
        pipeline
            .set_order_links(&names)
            .map_err(|e| vec![issue("graph.order", None, None, e.message().to_owned())])?;
        let old = pipeline.remove_graph_frame("wiring");
        let frame = WiringFrame {
            state: self.state.clone(),
            prepared: false,
        };
        if let Err(e) = pipeline.add_graph_frame("wiring", Box::new(frame)) {
            if let Some(frame) = old {
                let _ = pipeline.add_graph_frame("wiring", frame);
            }
            return Err(vec![issue(
                "graph.order",
                None,
                None,
                e.message().to_owned(),
            )]);
        }
        drop(pipeline);
        let mut state = self.state.borrow_mut();
        state.bindings = layout.bindings;
        state.view_inputs = layout.view_inputs;
        state.slot_count = layout.slot_count;
        self.instance_stats.clear();
        Ok(())
    }

    /// The executor-owned histories of `plan`. A history keeps its images, parity and validity
    /// when its consumer, producer and format are unchanged; one that `history_resets` restarts
    /// becomes invalid and its TAA consumer is invalidated once (a history without images is
    /// invalidated when the frame allocates them).
    fn histories(
        &self,
        plan: &GraphPlan,
        previous: Option<&GraphPlan>,
        entries: &[(String, String)],
        h: &mut FeatureHandles<'_>,
    ) -> Vec<History> {
        let resets = history_resets(previous, plan);
        let mut old = std::mem::take(&mut self.state.borrow_mut().histories);
        let mut out = Vec::new();
        for history in &plan.histories {
            let Some(resource) = plan.resources.iter().find(|r| r.id == history.resource) else {
                continue;
            };
            let consumer = &history.consumer;
            let (Some(producer_entry), Some(consumer_entry)) = (
                entry_of(entries, &resource.producer.node),
                entry_of(entries, &consumer.node),
            ) else {
                continue;
            };
            let type_of = |id: &str| {
                plan.nodes
                    .iter()
                    .find(|n| n.id == id)
                    .and_then(|n| self.catalog.node(&n.type_id))
            };
            let Some(consumer_spec) = type_of(&consumer.node) else {
                continue;
            };
            let Some(port) = consumer_spec.inputs.iter().find(|i| i.key == consumer.port) else {
                continue;
            };
            let format = type_of(&resource.producer.node)
                .and_then(|s| s.outputs.iter().find(|o| o.key == resource.producer.port))
                .and_then(|o| o.format.clone())
                .map_or(wgpu::TextureFormat::Rgba16Float, |f| {
                    f.to_wgpu(wgpu::TextureFormat::Rgba16Float)
                });
            let taa = if consumer_spec.id == "pass.taa" {
                self.instances
                    .iter()
                    .find_map(|(id, _, handle)| match handle {
                        InstanceHandle::Taa(t) if *id == consumer.node => Some(t.clone()),
                        _ => None,
                    })
                    .or_else(|| h.handles.taa.clone())
            } else {
                None
            };
            let producer = (producer_entry, resource.producer.port.clone());
            let key = format!("{}.{}", consumer.node, consumer.port);
            let display = self
                .state
                .borrow()
                .bindings
                .iter()
                .any(|binding| binding.entry == producer.0 && binding.resolve_display);
            let mut next = History {
                producer,
                consumer: (consumer_entry, port.key),
                format,
                display,
                taa,
                images: Vec::new(),
                current: 0,
                valid: false,
                copied: false,
                key,
            };
            if let Some(i) = old.iter().position(|o| {
                o.key == next.key && o.producer == next.producer && o.format == next.format
            }) {
                let kept = old.swap_remove(i);
                next.images = kept.images;
                next.current = kept.current;
                next.valid = kept.valid;
            }
            if resets.contains(&next.key) && !next.images.is_empty() {
                next.valid = false;
                if let Some(taa) = &next.taa {
                    taa.invalidate();
                }
            }
            out.push(next);
        }
        out
    }

    /// Restores the instance passes and their values after a refused `commit`: the passes it
    /// installed are removed, the passes it removed are added back, and the previous plan's values
    /// are written again.
    fn rollback(
        &mut self,
        snapshot: Vec<(String, String, InstanceHandle)>,
        undo: Undo,
        previous: Option<&GraphPlan>,
        h: &mut FeatureHandles<'_>,
    ) {
        {
            let mut pipeline = h.pipeline.borrow_mut();
            for entry in undo.installed.iter().rev() {
                let _ = pipeline.remove(entry);
            }
            for (entry, pass) in undo.removed {
                if pipeline.add(pass).is_ok() {
                    let _ = pipeline.set_enabled(&entry, true);
                }
            }
        }
        self.instances = snapshot;
        if let Some(previous) = previous {
            let _ = write_instances(previous, &self.catalog, &self.instances, h);
        }
    }
}

impl ExecExtension for WiringExtension {
    fn name(&self) -> &'static str {
        "wiring"
    }

    fn apply(
        &mut self,
        plan: &GraphPlan,
        previous: Option<&GraphPlan>,
        h: &mut FeatureHandles<'_>,
    ) -> Result<(), Vec<Issue>> {
        let resources = plan_fns::chain_resources(plan, &self.catalog);
        let entries = plan_fns::node_entries(plan, &self.catalog);
        let slots = pool_resources(&resources);
        let mut bindings = Vec::new();
        for node in &plan.nodes {
            if !plan_fns::chain_type(&node.type_id) && node.type_id != "pass.tonemap" {
                continue;
            }
            let Some((_, entry)) = entries.iter().find(|(id, _)| *id == node.id) else {
                continue;
            };
            let source =
                node.inputs
                    .iter()
                    .find(|i| i.port == "color")
                    .and_then(|i| match &i.source {
                        InputSource::Link { node, port, .. }
                        | InputSource::Default { node, port } => Some((node, port)),
                        InputSource::Absent => None,
                    });
            if node.type_id != "pass.forward"
                && node.type_id != "pass.tonemap"
                && !resources.iter().any(|r| {
                    source.is_some_and(|(id, port)| {
                        *id == r.producer.node && *port == r.producer.port
                    })
                })
            {
                return Err(vec![issue(
                    "graph.link_type",
                    Some(&node.id),
                    Some("color"),
                    "The colour source must be a chain colour output.".into(),
                )]);
            }
            let source = source.and_then(|(id, port)| {
                entries
                    .iter()
                    .find(|(node, _)| node == id)
                    .map(|(_, entry)| (entry.clone(), port.clone()))
            });
            let dst = resources
                .iter()
                .position(|r| r.producer.node == node.id)
                .map(|i| slots[i] as usize);
            if dst.is_none() && node.type_id != "pass.tonemap" {
                continue;
            }
            bindings.push(Binding {
                entry: entry.clone(),
                source,
                dst,
                forward: node.type_id == "pass.forward",
                tonemap: node.type_id == "pass.tonemap",
                resolve_display: node.type_id == "pass.taa",
                display_post: matches!(node.type_id.as_str(), "pass.bloom" | "pass.grading"),
            });
        }
        resolve_display_branches(&mut bindings);
        if plan_fns::bound_view_count(plan, &self.catalog) > 32 {
            let node = plan
                .nodes
                .iter()
                .find(|n| n.type_id == "pass.tonemap")
                .map(|n| n.id.as_str());
            return Err(vec![issue(
                "graph.budget",
                node,
                Some("color"),
                "The HDR chain exceeds 32 bound views.".into(),
            )]);
        }
        let refusals = plan_fns::history_refusals(plan, &self.catalog);
        if !refusals.is_empty() {
            return Err(refusals
                .iter()
                .map(|(node, port)| {
                    issue(
                        "graph.history",
                        Some(node),
                        Some(port),
                        format!(
                            "{node}.{port} needs a history link from pass.taa's history or a chain \
                             colour output; link it as a history link."
                        ),
                    )
                })
                .collect());
        }
        let instances = wanted_instances(plan, &entries);
        let layout = Layout {
            bindings,
            view_inputs: view_inputs(plan, &self.catalog, &entries, &instances),
            slot_count: slots.iter().max().map_or(0, |s| *s as usize + 1),
        };
        let snapshot = self.instances.clone();
        let mut undo = Undo::default();
        match self.commit(plan, &entries, &instances, layout, h, &mut undo) {
            Ok(()) => {
                let histories = self.histories(plan, previous, &entries, h);
                self.state.borrow_mut().histories = histories;
                Ok(())
            }
            Err(issues) => {
                self.rollback(snapshot, undo, previous, h);
                Err(issues)
            }
        }
    }

    fn clear(&mut self, h: &mut FeatureHandles<'_>) {
        let mut pipeline = h.pipeline.borrow_mut();
        pipeline.remove_graph_frame("wiring");
        for (_, entry, _) in self.instances.drain(..) {
            let _ = pipeline.remove(&entry);
        }
        self.instance_stats.clear();
        *self.state.borrow_mut() = State::default();
    }

    fn reorder(&mut self, plan: &GraphPlan, h: &mut FeatureHandles<'_>) -> Result<(), RaaError> {
        let entries = plan_fns::node_entries(plan, &self.catalog);
        let mut pipeline = h.pipeline.borrow_mut();
        for (_, entry, _) in &self.instances {
            pipeline.set_enabled(entry, true)?;
        }
        let order = plan_fns::link_order(plan, &entries, &pipeline.passes());
        let names: Vec<&str> = order.iter().map(String::as_str).collect();
        pipeline.set_order_links(&names)
    }

    fn frame(&mut self, h: &mut FeatureHandles<'_>) -> Result<(), RaaError> {
        let pipeline = h.pipeline.borrow();
        let table = pipeline.resources();
        let images = table.entries();
        let buffers = table.buffer_entries();
        self.instance_stats = self
            .instances
            .iter()
            .map(|(node, entry, _)| {
                let mut resources: Vec<ResourceStat> = images
                    .iter()
                    .filter(|(pass, _, _)| pass == entry)
                    .map(|(_, port, desc)| image_stat(port, desc))
                    .collect();
                resources.extend(
                    buffers
                        .iter()
                        .filter(|(pass, _, _)| pass == entry)
                        .map(|(_, port, bytes)| buffer_stat(port, *bytes)),
                );
                (
                    node.clone(),
                    pipeline.entry_enabled(entry) == Some(true),
                    resources,
                )
            })
            .collect();
        Ok(())
    }

    fn stats(&self, out: &mut Vec<NodeStats>) {
        for node in out.iter_mut() {
            if let Some((_, ran, resources)) = self
                .instance_stats
                .iter()
                .find(|(id, _, _)| *id == node.node)
            {
                node.ran = *ran;
                node.resources = resources.iter().map(copy_stat).collect();
            }
        }
    }
}

fn issue(code: &'static str, node: Option<&str>, port: Option<&str>, message: String) -> Issue {
    Issue {
        code,
        severity: Severity::Error,
        stage: Stage::Apply,
        node: node.map(str::to_owned),
        port: port.map(str::to_owned),
        param: None,
        link: None,
        message,
        fix: "Choose a compatible colour source and pass order.".into(),
    }
}

/// A `graph.apply` issue on `node` (and `param` when known), with the feature's message.
fn apply_issue(node: Option<&str>, param: Option<&str>, message: &str) -> Vec<Issue> {
    vec![Issue {
        code: "graph.apply",
        severity: Severity::Error,
        stage: Stage::Apply,
        node: node.map(str::to_owned),
        port: None,
        param: param.map(str::to_owned),
        link: None,
        message: message.to_owned(),
        fix: "Use a value the feature accepts.".into(),
    }]
}

/// The pipeline entry of node `id` in `entries`.
fn entry_of(entries: &[(String, String)], id: &str) -> Option<String> {
    entries
        .iter()
        .find(|(node, _)| node.as_str() == id)
        .map(|(_, entry)| entry.clone())
}

/// The plan's instance nodes: nodes of an instance type whose entry names an instance pass.
fn wanted_instances(plan: &GraphPlan, entries: &[(String, String)]) -> Vec<(String, String)> {
    entries
        .iter()
        .filter(|(_, entry)| entry.contains('#'))
        .filter_map(|(id, _)| {
            let node = plan.nodes.iter().find(|node| node.id == *id)?;
            INSTANCE_TYPES
                .contains(&node.type_id.as_str())
                .then(|| (id.clone(), node.type_id.clone()))
        })
        .collect()
}

/// Installs the instance pass `<entry>#<node>` of post node type `type_id` through its feature.
fn install_instance(
    type_id: &str,
    node: &str,
    h: &FeatureHandles<'_>,
) -> Result<InstanceHandle, RaaError> {
    let (gpu, pipeline) = (h.gpu, h.pipeline);
    Ok(match type_id {
        "pass.ssr" => {
            InstanceHandle::Ssr(raa_ssr::SsrFeature::install_instance(gpu, pipeline, node)?)
        }
        "pass.dof" => {
            InstanceHandle::Dof(raa_dof::DofFeature::install_instance(gpu, pipeline, node)?)
        }
        "pass.sunshafts" => InstanceHandle::Shafts(raa_shafts::ShaftsFeature::install_instance(
            gpu, pipeline, node,
        )?),
        "pass.bloom" => InstanceHandle::Bloom(raa_bloom::BloomFeature::install_instance(
            gpu, pipeline, node,
        )?),
        "pass.grading" => InstanceHandle::Grading(raa_grading::GradingFeature::install_instance(
            gpu, pipeline, node,
        )?),
        "pass.cel.resolve" => {
            InstanceHandle::Cel(raa_cel::CelFeature::install_instance(gpu, pipeline, node)?)
        }
        _ => {
            let lit = h.lit.as_deref().ok_or_else(|| {
                RaaError::Resource(format!("{node}: no lit renderer for the TAA frame buffer"))
            })?;
            let frame = lit.handle().frame_buffer();
            InstanceHandle::Taa(raa_taa::TaaFeature::install_instance(
                gpu, pipeline, &frame, node,
            )?)
        }
    })
}

/// The feature a post node type belongs to.
fn feature_of(type_id: &str) -> Option<FeatureId> {
    Some(match type_id {
        "pass.ssr" => FeatureId::Ssr,
        "pass.dof" => FeatureId::Dof,
        "pass.sunshafts" => FeatureId::Shafts,
        "pass.bloom" => FeatureId::Bloom,
        "pass.grading" => FeatureId::Grading,
        "pass.cel.resolve" => FeatureId::Cel,
        "pass.taa" => FeatureId::Taa,
        _ => return None,
    })
}

/// Puts `instance` into the field of `handles` that its feature's values are written to.
fn stand_in(handles: &mut HandleSet, instance: &InstanceHandle) {
    match instance {
        InstanceHandle::Ssr(h) => handles.ssr = Some(h.clone()),
        InstanceHandle::Dof(h) => handles.dof = Some(h.clone()),
        InstanceHandle::Shafts(h) => handles.shafts = Some(h.clone()),
        InstanceHandle::Bloom(h) => handles.bloom = Some(h.clone()),
        InstanceHandle::Grading(h) => handles.grading = Some(h.clone()),
        InstanceHandle::Cel(h) => handles.cel = Some(h.clone()),
        InstanceHandle::Taa(h) => handles.taa = Some(h.clone()),
    }
}

/// The value the plan holds for parameter `key` of `node`.
fn node_value<'a>(node: &'a PlanNode, key: &str) -> Option<&'a ParamValue> {
    node.params
        .iter()
        .find(|(name, _)| name.as_str() == key)
        .map(|(_, value)| value)
}

/// The Setting values `node` writes to `feature`, under the key each parameter binds.
fn setting_changes(
    node: &PlanNode,
    spec: &NodeTypeSpec,
    feature: &FeatureId,
) -> Vec<(String, ParamValue)> {
    spec.params
        .iter()
        .filter_map(|param| match &param.binding {
            ParamBinding::Setting {
                feature: bound,
                key,
            } if bound == feature => {
                node_value(node, param.key).map(|value| ((*key).to_owned(), value.clone()))
            }
            _ => None,
        })
        .collect()
}

/// The Tunable values of `node` as f64, with their feature and key.
fn tunable_values(node: &PlanNode, spec: &NodeTypeSpec) -> Vec<(FeatureId, &'static str, f64)> {
    spec.params
        .iter()
        .filter_map(|param| match &param.binding {
            ParamBinding::Tunable { feature, key } => match node_value(node, param.key)? {
                ParamValue::Number(x) => Some((feature.clone(), *key, *x)),
                ParamValue::Int(i) => Some((feature.clone(), *key, *i as f64)),
                _ => None,
            },
            _ => None,
        })
        .collect()
}

/// Writes every instance node's values into its instance handle: the Setting values merged over
/// the feature's defaults, then the Tunable values. The installed handle is set aside meanwhile,
/// so the feature's writer reads and writes the instance only.
fn write_instances(
    plan: &GraphPlan,
    catalog: &Catalog,
    instances: &[(String, String, InstanceHandle)],
    h: &mut FeatureHandles<'_>,
) -> Result<(), Vec<Issue>> {
    for node in &plan.nodes {
        let Some((_, _, instance)) = instances.iter().find(|(id, _, _)| *id == node.id) else {
            continue;
        };
        let (Some(spec), Some(feature)) = (catalog.node(&node.type_id), feature_of(&node.type_id))
        else {
            continue;
        };
        let changes = setting_changes(node, spec, &feature);
        let merged = merge_values(&feature, &default_values(&feature), &changes).map_err(
            |(key, message)| apply_issue(Some(node.id.as_str()), Some(key.as_str()), &message),
        )?;
        let installed = std::mem::take(&mut *h.handles);
        stand_in(h.handles, instance);
        let written = write_values(h, &feature, &merged);
        *h.handles = installed;
        written.map_err(|e| apply_issue(Some(node.id.as_str()), None, e.message()))?;
        for (_, key, value) in tunable_values(node, spec) {
            instance
                .set_tunable(key, value)
                .map_err(|e| apply_issue(Some(node.id.as_str()), Some(key), e.message()))?;
        }
    }
    Ok(())
}

/// Writes the Tunable values of the first node of each instance type to the installed handle, which
/// the tunables extension has just written with the last node's values (S57, E5).
fn write_installed_tunables(
    plan: &GraphPlan,
    catalog: &Catalog,
    h: &mut FeatureHandles<'_>,
) -> Result<(), Vec<Issue>> {
    for (index, node) in plan.nodes.iter().enumerate() {
        let first = !plan.nodes[..index]
            .iter()
            .any(|earlier| earlier.type_id == node.type_id);
        if !first || !INSTANCE_TYPES.contains(&node.type_id.as_str()) {
            continue;
        }
        let Some(spec) = catalog.node(&node.type_id) else {
            continue;
        };
        for (feature, key, value) in tunable_values(node, spec) {
            set_installed_tunable(h.handles, &feature, key, value)
                .map_err(|e| apply_issue(Some(node.id.as_str()), Some(key), e.message()))?;
        }
    }
    Ok(())
}

/// Sets a tunable on the installed handle of `feature`; a feature without a tunable handle is
/// skipped.
fn set_installed_tunable(
    handles: &HandleSet,
    feature: &FeatureId,
    key: &str,
    value: f64,
) -> Result<(), RaaError> {
    let written = match feature {
        FeatureId::Taa => handles.taa.as_ref().map(|h| h.set_tunable(key, value)),
        FeatureId::Ssr => handles.ssr.as_ref().map(|h| h.set_tunable(key, value)),
        FeatureId::Dof => handles.dof.as_ref().map(|h| h.set_tunable(key, value)),
        FeatureId::Shafts => handles.shafts.as_ref().map(|h| h.set_tunable(key, value)),
        _ => None,
    };
    written.unwrap_or(Ok(()))
}

/// An input of a view node that reads an instance node's image: it is rebound to that image in the
/// view's `before_record`, because the `views` frame binds the first node's image (S57 step 3).
struct ViewInput {
    entry: String,
    port: &'static str,
    source: (String, String),
}

/// The inputs of view nodes that a plan links from an instance node.
fn view_inputs(
    plan: &GraphPlan,
    catalog: &Catalog,
    entries: &[(String, String)],
    instances: &[(String, String)],
) -> Vec<ViewInput> {
    let mut inputs = Vec::new();
    for node in plan.nodes.iter().filter(|n| n.type_id.starts_with("view.")) {
        let (Some(spec), Some(view)) = (catalog.node(&node.type_id), entry_of(entries, &node.id))
        else {
            continue;
        };
        for input in &node.inputs {
            let InputSource::Link {
                node: source, port, ..
            } = &input.source
            else {
                continue;
            };
            if !instances.iter().any(|(id, _)| id == source) {
                continue;
            }
            let key = spec
                .inputs
                .iter()
                .find(|p| p.key == input.port)
                .map(|p| p.key);
            let (Some(port_key), Some(source_entry)) = (key, entry_of(entries, source)) else {
                continue;
            };
            inputs.push(ViewInput {
                entry: view.clone(),
                port: port_key,
                source: (source_entry, port.clone()),
            });
        }
    }
    inputs
}

/// The statistics of one published image.
fn image_stat(port: &str, desc: &ImageDesc) -> ResourceStat {
    let texel = u64::from(desc.format.block_copy_size(None).unwrap_or(0));
    let bytes = (0..desc.mips)
        .map(|mip| {
            let width = u64::from(desc.width.checked_shr(mip).unwrap_or(0).max(1));
            let height = u64::from(desc.height.checked_shr(mip).unwrap_or(0).max(1));
            width * height * u64::from(desc.layers) * texel
        })
        .sum();
    ResourceStat {
        port: port.to_owned(),
        width: desc.width,
        height: desc.height,
        layers: desc.layers,
        mips: desc.mips,
        format: format!("{:?}", desc.format).to_lowercase(),
        bytes,
    }
}

/// The statistics of one published buffer: no extent, one layer and one mip.
fn buffer_stat(port: &str, bytes: u64) -> ResourceStat {
    ResourceStat {
        port: port.to_owned(),
        width: 0,
        height: 0,
        layers: 1,
        mips: 1,
        format: "buffer".to_owned(),
        bytes,
    }
}

/// A field-by-field copy of `stat` (`ResourceStat` is not `Clone`).
fn copy_stat(stat: &ResourceStat) -> ResourceStat {
    ResourceStat {
        port: stat.port.clone(),
        width: stat.width,
        height: stat.height,
        layers: stat.layers,
        mips: stat.mips,
        format: stat.format.clone(),
        bytes: stat.bytes,
    }
}

struct Binding {
    entry: String,
    source: Option<(String, String)>,
    dst: Option<usize>,
    forward: bool,
    tonemap: bool,
    resolve_display: bool,
    display_post: bool,
}

/// Fullscreen post passes that can use a display-sized HDR input inherit its extent.
/// Scene-space passes require the entire incoming branch to stay on the scene grid.
fn resolve_display_branches(bindings: &mut [Binding]) {
    let parent = |binding: &Binding| {
        let (entry, _) = binding.source.as_ref()?;
        bindings.iter().position(|b| b.entry == *entry)
    };
    let root = |mut index: usize| {
        for _ in 0..bindings.len() {
            let binding = &bindings[index];
            if !binding.display_post {
                return binding.resolve_display.then_some(index);
            }
            index = parent(binding)?;
        }
        None
    };
    let roots: Vec<_> = (0..bindings.len()).map(root).collect();
    let mut blocked = vec![false; bindings.len()];
    for binding in bindings.iter().filter(|b| !b.tonemap && !b.display_post) {
        if let Some(index) = parent(binding).and_then(|i| roots[i]) {
            blocked[index] = true;
        }
    }
    for (binding, root) in bindings.iter_mut().zip(roots) {
        binding.resolve_display = root.is_some_and(|i| !blocked[i]);
    }
}

/// The colour bindings, view inputs and slot count that a successful `apply` installs.
struct Layout {
    bindings: Vec<Binding>,
    view_inputs: Vec<ViewInput>,
    slot_count: usize,
}

/// What a refusing `apply` must undo: the entries it installed and the passes it removed.
#[derive(Default)]
struct Undo {
    installed: Vec<String>,
    removed: Vec<(String, Box<dyn FeaturePass>)>,
}

/// One executor-owned history: two images at frame size, the index `current` written this frame
/// and the validity of the other one.
///
/// LIFECYCLE (history images, contract rg-wiring-frame): created in the frame's first
/// `before_prepare` of a frame when missing or when the frame size differs, together with the
/// pool; copied in the producer's `after_record`; bound in the consumer's `before_record`; parity
/// flipped in `submitted`; dropped when their history leaves the plan and in `clear`.
struct History {
    key: String,
    producer: (String, String),
    consumer: (String, &'static str),
    format: wgpu::TextureFormat,
    display: bool,
    taa: Option<raa_taa::TaaHandle>,
    images: Vec<(wgpu::Texture, wgpu::TextureView)>,
    current: usize,
    valid: bool,
    copied: bool,
}

#[derive(Default)]
struct State {
    histories: Vec<History>,
    bindings: Vec<Binding>,
    view_inputs: Vec<ViewInput>,
    slot_count: usize,
    size: (u32, u32),
    pool: Vec<(wgpu::Texture, wgpu::TextureView)>,
    display_pool: Vec<Option<wgpu::TextureView>>,
    display_size: (u32, u32),
}

struct WiringFrame {
    state: Rc<RefCell<State>>,
    prepared: bool,
}

impl GraphFrame for WiringFrame {
    fn begin(&mut self, _table: &ResourceTable) {
        self.prepared = false;
        for history in &mut self.state.borrow_mut().histories {
            history.copied = false;
        }
    }

    fn before_prepare(&mut self, _pass: &str, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        if self.prepared {
            return Ok(());
        }
        self.prepared = true;
        let mut guard = self.state.borrow_mut();
        let state = &mut *guard;
        let display_size = ctx.targets.display_rect().size;
        if state.size != ctx.size {
            state.pool.clear();
            state.size = ctx.size;
        }
        if state.display_size != display_size {
            state.display_pool.clear();
            state.display_size = display_size;
        }
        if display_size != ctx.size {
            state.display_pool.resize(state.slot_count, None);
            for binding in state.bindings.iter().filter(|b| b.resolve_display) {
                if let Some(i) = binding.dst {
                    state.display_pool[i].get_or_insert_with(|| {
                        make_color_texture(ctx.gpu, display_size)
                            .create_view(&wgpu::TextureViewDescriptor::default())
                    });
                }
            }
        }
        while state.pool.len() < state.slot_count {
            let texture = make_color_texture(ctx.gpu, ctx.size);
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            state.pool.push((texture, view));
        }
        for history in &mut state.histories {
            let size = if history.display {
                display_size
            } else {
                ctx.size
            };
            if history
                .images
                .first()
                .is_some_and(|(texture, _)| (texture.width(), texture.height()) == size)
            {
                continue;
            }
            history.images.clear();
            for _ in 0..2 {
                let texture = ctx.gpu.device().create_texture(&wgpu::TextureDescriptor {
                    label: Some("raa.wiring.history"),
                    size: wgpu::Extent3d {
                        width: size.0,
                        height: size.1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: history.format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });
                let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
                history.images.push((texture, view));
            }
            history.current = 0;
            history.valid = false;
            if let Some(taa) = &history.taa {
                taa.invalidate();
            }
        }
        Ok(())
    }

    fn before_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let state = self.state.borrow();
        for input in state.view_inputs.iter().filter(|input| input.entry == pass) {
            if let Some(image) = ctx.resources.get(&input.source.0, &input.source.1) {
                ctx.resources.bind_input(pass, input.port, image);
            }
        }
        for history in state
            .histories
            .iter()
            .filter(|h| h.valid && h.consumer.0 == pass)
        {
            if let Some((texture, view)) = history.images.get(1 - history.current) {
                let desc = ImageDesc::new(
                    ImageDim::D2,
                    history.format,
                    texture.width(),
                    texture.height(),
                    1,
                    1,
                );
                let image = PublishedImage {
                    view: view.clone(),
                    desc,
                };
                ctx.resources.bind_input(pass, history.consumer.1, image);
            }
        }
        let Some(binding) = state.bindings.iter().find(|b| b.entry == pass) else {
            return Ok(());
        };
        let dst = chain_destination(binding, &state, ctx.targets.display_rect().size != ctx.size);
        let source = binding
            .source
            .as_ref()
            .and_then(|(entry, port)| ctx.resources.get(entry, port));
        let src = if binding.forward {
            dst
        } else {
            source.as_ref().map(|image| &image.view)
        };
        let src = src.ok_or_else(|| {
            RaaError::Resource(format!(
                "{pass}: source {:?} was not published",
                binding.source
            ))
        })?;
        let dst = if binding.tonemap {
            src
        } else {
            dst.ok_or_else(|| {
                RaaError::Resource(format!(
                    "{pass}: no destination for source {:?}",
                    binding.source
                ))
            })?
        };
        let hdr = ctx.targets.hdr();
        hdr.bind(src, dst);
        if !hdr.is_bound() {
            return Err(RaaError::Resource(format!(
                "{pass}: unable to bind source {:?}",
                binding.source
            )));
        }
        Ok(())
    }

    fn after_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let mut guard = self.state.borrow_mut();
        let state = &mut *guard;
        if let Some(binding) = state.bindings.iter().find(|b| b.entry == pass) {
            ctx.targets.hdr().unbind();
            publish_color(pass, binding, state, ctx);
        }
        for history in state.histories.iter_mut().filter(|h| h.producer.0 == pass) {
            let Some(image) = ctx.resources.get(pass, &history.producer.1) else {
                continue;
            };
            let Some((texture, _)) = history.images.get(history.current) else {
                continue;
            };
            if (image.desc.width, image.desc.height) != (texture.width(), texture.height())
                || image.desc.format != history.format
            {
                continue;
            }
            ctx.encoder.copy_texture_to_texture(
                image.view.texture().as_image_copy(),
                texture.as_image_copy(),
                wgpu::Extent3d {
                    width: texture.width(),
                    height: texture.height(),
                    depth_or_array_layers: 1,
                },
            );
            history.copied = true;
        }
        Ok(())
    }

    fn invalidate_history(&mut self) {
        for history in &mut self.state.borrow_mut().histories {
            history.valid = false;
            history.copied = false;
            if let Some(taa) = &history.taa {
                taa.invalidate();
            }
        }
    }

    fn submitted(&mut self) {
        for history in &mut self.state.borrow_mut().histories {
            if history.copied {
                history.copied = false;
                history.valid = true;
                history.current = 1 - history.current;
            }
        }
    }
}

fn make_color_texture(gpu: &raa_core::GpuContext, size: (u32, u32)) -> wgpu::Texture {
    gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("raa.wiring.pool"),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn chain_destination<'a>(
    binding: &Binding,
    state: &'a State,
    scaled: bool,
) -> Option<&'a wgpu::TextureView> {
    let i = binding.dst?;
    if binding.resolve_display && scaled {
        state.display_pool.get(i)?.as_ref()
    } else {
        state.pool.get(i).map(|(_, view)| view)
    }
}

/// Republishes (entry, "color") with the pool view the chain pass `pass` wrote.
fn publish_color(pass: &str, binding: &Binding, state: &State, ctx: &mut RecordCtx<'_>) {
    if !binding.tonemap
        && let Some(view) =
            chain_destination(binding, state, ctx.targets.display_rect().size != ctx.size)
    {
        ctx.resources.publish(
            pass,
            "color",
            view,
            ImageDesc::new(
                ImageDim::D2,
                wgpu::TextureFormat::Rgba16Float,
                view.texture().width(),
                view.texture().height(),
                1,
                1,
            ),
        );
    }
}
