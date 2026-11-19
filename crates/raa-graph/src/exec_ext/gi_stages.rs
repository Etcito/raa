//! Device-free plan mapping for GI stage nodes and the placeholder executor extension.
#![allow(dead_code)] // The next extension share consumes these plan helpers.

use crate::catalog::{Catalog, NodeTypeSpec, ParamValue, PortType};
use crate::error::{Issue, Severity, Stage};
use crate::exec::{ExecExtension, FeatureHandles};
use crate::exec::{NodeStats, ResourceStat};
use crate::plan::{GraphPlan, InputSource, PlanNode};
use raa_core::RaaError;
use raa_gi::stages::{GiStageHandle, GiStageKind, install_stage, stage_entry};
use raa_pipeline::{GraphFrame, ImageDesc, PrepareCtx, RecordCtx, ResourceTable};
use std::{cell::RefCell, rc::Rc};

/// Shared installed stages and history registry.
/// Handoff: raa-graph exec gi_stages 3/3: graph frame binding stage inputs, executor-owned histories and forward slots. `GiStagesState` holds the GI handle, plan-ordered stage handles, image bindings, history pairs, forward-slot sources and saved slot views today; the graph frame clones `GiStagesExtension.state` after `state_from_plan` replaces it. The plan helpers are anchored by `stage_nodes` in this file.
#[derive(Default)]
pub(crate) struct GiStagesState {
    pub(crate) gi: Option<raa_gi::GiHandle>,
    pub(crate) stages: Vec<(String, String, GiStageHandle)>,
    pub(crate) image_bindings: Vec<(String, &'static str, String, String)>,
    pub(crate) histories: Vec<StageHistory>,
    pub(crate) slot_sources: Vec<(u32, String, String)>,
    pub(crate) saved_slots: Vec<(u32, wgpu::TextureView)>,
}

/// Returns the previous image index when the executor has a usable history pair.
fn history_binding_index(history: &StageHistory) -> Option<usize> {
    history_binding_index_for(history.valid, history.images.len(), history.current)
}

fn history_binding_index_for(valid: bool, image_count: usize, current: usize) -> Option<usize> {
    (valid && image_count == 2).then_some(1 - current)
}

/// Commits a copied frame and alternates the history pair.
fn flip(history: &mut StageHistory) {
    if history.copied {
        history.valid = true;
        history.current = 1 - history.current;
        history.copied = false;
    }
}
pub(crate) struct StageHistory {
    pub(crate) key: String,
    pub(crate) consumer: (String, &'static str),
    pub(crate) producer: (String, String),
    pub(crate) images: Vec<(wgpu::Texture, wgpu::TextureView)>,
    pub(crate) desc: Option<ImageDesc>,
    pub(crate) current: usize,
    pub(crate) valid: bool,
    pub(crate) copied: bool,
}

/// The extension currently has no device work; `apply` succeeds and `clear` does nothing today.
pub fn extension() -> Box<dyn ExecExtension> {
    Box::new(GiStagesExtension {
        state: Rc::new(RefCell::new(GiStagesState::default())),
        frame_added: false,
    })
}

struct GiStagesExtension {
    state: Rc<RefCell<GiStagesState>>,
    frame_added: bool,
}

impl ExecExtension for GiStagesExtension {
    fn name(&self) -> &'static str {
        "gi_stages"
    }
    fn apply(
        &mut self,
        plan: &GraphPlan,
        previous: Option<&GraphPlan>,
        h: &mut FeatureHandles<'_>,
    ) -> Result<(), Vec<Issue>> {
        let nodes = stage_nodes(plan);
        if !nodes.is_empty() && (h.handles.gi.is_none() || h.lit.is_none()) {
            return Err(vec![issue(
                "graph.unavailable",
                Some(nodes[0].0.clone()),
                None,
                "the GI feature or lit renderer is not installed",
                "Install the GI feature.",
            )]);
        }
        if nodes.is_empty() && self.state.borrow().stages.is_empty() {
            return Ok(());
        }
        let issues = accumulate_issues(plan);
        if !issues.is_empty() {
            return Err(issues);
        }
        let mut state = std::mem::take(&mut *self.state.borrow_mut());
        let old_stages = std::mem::take(&mut state.stages);
        let mut kept = Vec::new();
        for (_, entry, handle) in &old_stages {
            if let Ok(pass) = h.pipeline.borrow_mut().remove(entry) {
                kept.push((entry.clone(), pass, handle.clone()));
            }
        }
        let catalog = Catalog::build();
        let entries = stage_entries(plan, &catalog);
        let buffers = buffer_sources(plan, &catalog, &entries);
        let mut installed = Vec::new();
        let result: Result<(), Vec<Issue>> = (|| {
            for (id, kind) in nodes {
                let name = stage_entry(kind, &id);
                let handle = if let Some(i) = kept.iter().position(|(entry, _, _)| *entry == name) {
                    let (_, pass, handle) = kept.swap_remove(i);
                    h.pipeline
                        .borrow_mut()
                        .add(pass)
                        .map_err(|e| vec![apply_issue(&id, &e)])?;
                    handle
                } else {
                    install_stage(h.gpu, h.pipeline, kind, &id)
                        .map_err(|e| vec![apply_issue(&id, &e)])?
                };
                installed.push((id, name, handle));
            }
            let gi = h.handles.gi.as_ref().expect("availability checked");
            let lit = h.lit.as_ref().expect("availability checked").handle();
            let changed = restarted(previous, plan);
            for (id, entry, handle) in &installed {
                handle.attach(gi, lit.clone());
                let node = plan
                    .nodes
                    .iter()
                    .find(|n| n.id == *id)
                    .expect("stage node exists");
                let spec = catalog.node(&node.type_id).expect("stage type exists");
                handle
                    .set_values(&stage_values(node, spec))
                    .map_err(|e| vec![apply_issue(id, &e)])?;
                handle.clear_buffer_sources();
                for (consumer, port, source, source_port) in &buffers {
                    if consumer == entry {
                        handle.set_buffer_source(port, source, source_port);
                    }
                }
                if handle.kind().internal_history() && changed.contains(id) {
                    handle.set_history_valid(false);
                }
            }
            Ok(())
        })();
        if let Err(issues) = result {
            for (_, entry, _) in &installed {
                let _ = h.pipeline.borrow_mut().remove(entry);
            }
            self.state.borrow_mut().stages.clear();
            return Err(issues);
        }
        state.gi = h.handles.gi.clone();
        let next = state_from_plan(plan, previous, &catalog, installed, state);
        *self.state.borrow_mut() = next;
        let has_stages = !self.state.borrow().stages.is_empty();
        let mut pipeline = h.pipeline.borrow_mut();
        if has_stages && !self.frame_added {
            pipeline
                .add_graph_frame(
                    "gi_stages",
                    Box::new(GiStagesFrame {
                        state: self.state.clone(),
                        restored: false,
                    }),
                )
                .map_err(|error| vec![apply_issue("gi_stages", &error)])?;
            self.frame_added = true;
        } else if !has_stages && self.frame_added {
            pipeline.remove_graph_frame("gi_stages");
            self.frame_added = false;
        }
        Ok(())
    }
    fn clear(&mut self, h: &mut FeatureHandles<'_>) {
        let mut pipeline = h.pipeline.borrow_mut();
        let mut state = self.state.borrow_mut();
        for (slot, view) in state.saved_slots.drain(..) {
            let _ = pipeline.lighting_mut().set_slot_view(slot, view);
        }
        for (_, entry, _) in state.stages.drain(..) {
            let _ = pipeline.remove(&entry);
        }
        pipeline.remove_graph_frame("gi_stages");
        self.frame_added = false;
        *state = GiStagesState::default();
    }
    fn stats(&self, out: &mut Vec<NodeStats>) {
        for (id, _, handle) in &self.state.borrow().stages {
            if let Some(stat) = out.iter_mut().find(|stat| stat.node == *id) {
                stat.ran = handle.ran();
                stat.resources = handle
                    .published()
                    .iter()
                    .map(|(port, desc)| image_stat(port, desc))
                    .collect();
            }
        }
    }
}

struct GiStagesFrame {
    state: Rc<RefCell<GiStagesState>>,
    restored: bool,
}

impl GraphFrame for GiStagesFrame {
    fn begin(&mut self, _table: &ResourceTable) {
        self.restored = false;
        if let Some(gi) = self.state.borrow().gi.as_ref()
            && !gi.shared().camera_valid()
        {
            let mut state = self.state.borrow_mut();
            for history in &mut state.histories {
                history.valid = false;
            }
            for (_, _, stage) in &state.stages {
                if stage.kind().internal_history() {
                    stage.set_history_valid(false);
                }
            }
        }
    }

    fn before_prepare(&mut self, pass: &str, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let mut state = self.state.borrow_mut();
        if !self.restored {
            self.restored = true;
            for (slot, view) in state.saved_slots.drain(..) {
                let _ = ctx.lighting.set_slot_view(slot, view);
            }
        }
        if pass == "forward" {
            let GiStagesState {
                slot_sources,
                saved_slots,
                ..
            } = &mut *state;
            for (slot, source, port) in slot_sources {
                if let Some(image) = ctx.resources.get(source, port) {
                    if let Some(view) = ctx.lighting.slot_view(*slot) {
                        saved_slots.push((*slot, view));
                    }
                    ctx.lighting.set_slot_view(*slot, image.view)?;
                }
            }
        }
        for (consumer, port, source, source_port) in &state.image_bindings {
            if consumer == pass
                && let Some(image) = ctx.resources.get(source, source_port)
            {
                ctx.resources.bind_input(pass, port, image);
            }
        }
        for history in state.histories.iter().filter(|h| h.consumer.0 == pass) {
            if let Some(index) = history_binding_index(history)
                && let Some((_, view)) = history.images.get(index)
                && let Some(desc) = history.desc
            {
                ctx.resources.bind_input(
                    pass,
                    history.consumer.1,
                    raa_pipeline::PublishedImage {
                        view: view.clone(),
                        desc,
                    },
                );
            }
        }
        Ok(())
    }

    fn after_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let mut state = self.state.borrow_mut();
        for history in state.histories.iter_mut().filter(|h| h.producer.0 == pass) {
            let Some(image) = ctx.resources.get(pass, &history.producer.1) else {
                continue;
            };
            if history.images.is_empty() || history.desc != Some(image.desc) {
                history.images.clear();
                for _ in 0..2 {
                    let texture = ctx.gpu.device().create_texture(&wgpu::TextureDescriptor {
                        label: Some("raa.gi_stages.history"),
                        size: wgpu::Extent3d {
                            width: image.desc.width,
                            height: image.desc.height,
                            depth_or_array_layers: image.desc.layers,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: image.desc.format,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING
                            | wgpu::TextureUsages::COPY_SRC
                            | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    });
                    let view = texture.create_view(&wgpu::TextureViewDescriptor {
                        dimension: Some(if image.desc.layers > 1 {
                            wgpu::TextureViewDimension::D2Array
                        } else {
                            wgpu::TextureViewDimension::D2
                        }),
                        ..Default::default()
                    });
                    history.images.push((texture, view));
                }
                history.desc = Some(image.desc);
                history.current = 0;
                history.valid = false;
            }
            if let Some((texture, _)) = history.images.get(history.current) {
                ctx.encoder.copy_texture_to_texture(
                    image.view.texture().as_image_copy(),
                    texture.as_image_copy(),
                    wgpu::Extent3d {
                        width: image.desc.width,
                        height: image.desc.height,
                        depth_or_array_layers: image.desc.layers,
                    },
                );
                history.copied = true;
            }
        }
        Ok(())
    }

    fn end(&mut self, _table: &ResourceTable) {
        let mut state = self.state.borrow_mut();
        for history in &mut state.histories {
            flip(history);
        }
        for (_, _, stage) in &state.stages {
            if stage.kind().internal_history() && stage.ran() {
                stage.set_history_valid(true);
            }
        }
    }
}

/// Handoff: `stage_nodes` supplies stage identities to 'raa-graph exec gi_stages 2/3: install, attach, values, refusals and stats' and 'raa-graph exec gi_stages 3/3: graph frame binding stage inputs, executor-owned histories and forward slots'. It returns the plan-ordered `(node id, GiStageKind)` pairs; this file also provides their entries, bindings, restarts, slots, values and accumulate validation. Upstream `GiStageKind::from_type_id` and `stage_entry` are at `crates/raa-gi/src/stages.rs:41 GiStageKind` and `crates/raa-gi/src/stages.rs:133 stage_entry`.
pub(crate) fn stage_nodes(plan: &GraphPlan) -> Vec<(String, GiStageKind)> {
    plan.nodes
        .iter()
        .filter_map(|n| GiStageKind::from_type_id(&n.type_id).map(|k| (n.id.clone(), k)))
        .collect()
}

pub(crate) fn stage_entries(plan: &GraphPlan, catalog: &Catalog) -> Vec<(String, String)> {
    let mut seen: Vec<String> = Vec::new();
    plan.nodes
        .iter()
        .filter_map(|n| {
            let kind = GiStageKind::from_type_id(&n.type_id);
            let spec = catalog.node(&n.type_id)?;
            let entry = if let Some(kind) = kind {
                stage_entry(kind, &n.id)
            } else if n.type_id.starts_with("view.") {
                format!("view#{}", n.id)
            } else {
                if spec.contributor {
                    return None;
                }
                let pass = spec.pass?;
                let duplicate = spec.multi_instance && seen.contains(&n.type_id);
                seen.push(n.type_id.clone());
                if duplicate {
                    format!("{pass}#{}", n.id)
                } else {
                    pass.to_owned()
                }
            };
            Some((n.id.clone(), entry))
        })
        .collect()
}

fn entry(entries: &[(String, String)], id: &str) -> Option<String> {
    entries
        .iter()
        .find(|(node, _)| node == id)
        .map(|(_, e)| e.clone())
}
fn stage_id(id: &str) -> bool {
    GiStageKind::from_type_id(id).is_some()
}
fn bindings(
    plan: &GraphPlan,
    catalog: &Catalog,
    entries: &[(String, String)],
    buffers: bool,
) -> Vec<(String, &'static str, String, String)> {
    let mut out = Vec::new();
    for node in &plan.nodes {
        if !stage_id(&node.type_id) {
            continue;
        }
        let (Some(consumer), Some(spec)) = (entry(entries, &node.id), catalog.node(&node.type_id))
        else {
            continue;
        };
        for port in &spec.inputs {
            if matches!(port.ty, PortType::Buffer) != buffers {
                continue;
            }
            let source = node
                .inputs
                .iter()
                .find(|i| i.port == port.key)
                .and_then(|i| match &i.source {
                    InputSource::Link {
                        node,
                        port,
                        history: false,
                    }
                    | InputSource::Default { node, port } => Some((node, port)),
                    _ => None,
                });
            if let Some((source_node, source_port)) = source
                && let Some(source_entry) = entry(entries, source_node)
            {
                out.push((
                    consumer.clone(),
                    port.key,
                    source_entry,
                    source_port.clone(),
                ));
            }
        }
    }
    out
}
pub(crate) fn image_bindings(
    p: &GraphPlan,
    c: &Catalog,
    e: &[(String, String)],
) -> Vec<(String, &'static str, String, String)> {
    bindings(p, c, e, false)
}
pub(crate) fn buffer_sources(
    p: &GraphPlan,
    c: &Catalog,
    e: &[(String, String)],
) -> Vec<(String, &'static str, String, String)> {
    bindings(p, c, e, true)
}

pub(crate) fn stage_histories(
    plan: &GraphPlan,
    catalog: &Catalog,
    entries: &[(String, String)],
) -> Vec<(String, &'static str, String, String)> {
    plan.histories
        .iter()
        .filter_map(|h| {
            let consumer_node = plan.nodes.iter().find(|n| n.id == h.consumer.node)?;
            if !stage_id(&consumer_node.type_id) {
                return None;
            }
            let port = catalog
                .node(&consumer_node.type_id)?
                .inputs
                .iter()
                .find(|p| p.key == h.consumer.port)?
                .key;
            let producer = plan
                .resources
                .iter()
                .find(|r| r.id == h.resource)?
                .producer
                .clone();
            Some((
                entry(entries, &h.consumer.node)?,
                port,
                entry(entries, &producer.node)?,
                producer.port,
            ))
        })
        .collect()
}

pub(crate) fn restarted(previous: Option<&GraphPlan>, next: &GraphPlan) -> Vec<String> {
    next.nodes
        .iter()
        .filter(|n| {
            stage_id(&n.type_id)
                && previous
                    .and_then(|p| p.nodes.iter().find(|o| o.id == n.id))
                    .is_none_or(|o| o.type_id != n.type_id || o.params != n.params)
        })
        .map(|n| n.id.clone())
        .collect()
}
pub(crate) fn history_restarts(previous: Option<&GraphPlan>, next: &GraphPlan) -> Vec<String> {
    let changed = restarted(previous, next);
    next.histories
        .iter()
        .filter_map(|h| {
            let producer = next
                .resources
                .iter()
                .find(|r| r.id == h.resource)?
                .producer
                .clone();
            if !stage_id(&next.nodes.iter().find(|n| n.id == h.consumer.node)?.type_id) {
                return None;
            }
            let old_source = previous
                .and_then(|p| p.histories.iter().find(|x| x.consumer == h.consumer))
                .and_then(|x| previous?.resources.iter().find(|r| r.id == x.resource))
                .map(|r| &r.producer);
            (changed.contains(&h.consumer.node)
                || changed.contains(&producer.node)
                || old_source != Some(&producer))
            .then(|| format!("{}.{}", h.consumer.node, h.consumer.port))
        })
        .collect()
}
pub(crate) fn slot_sources(
    plan: &GraphPlan,
    catalog: &Catalog,
    entries: &[(String, String)],
) -> Vec<(u32, String, String)> {
    let mut out = Vec::new();
    for node in &plan.nodes {
        if node.type_id != "pass.forward" {
            continue;
        }
        let Some(spec) = catalog.node(&node.type_id) else {
            continue;
        };
        for input in &node.inputs {
            let Some(slot) = spec
                .inputs
                .iter()
                .find(|p| p.key == input.port)
                .and_then(|p| p.slot)
            else {
                continue;
            };
            if let InputSource::Link {
                node: src, port, ..
            } = &input.source
                && stage_id(
                    plan.nodes
                        .iter()
                        .find(|n| n.id == *src)
                        .map_or("", |n| n.type_id.as_str()),
                )
                && let Some(e) = entry(entries, src)
            {
                out.push((slot, e, port.clone()));
            }
        }
    }
    out
}
pub(crate) fn stage_values(node: &PlanNode, spec: &NodeTypeSpec) -> Vec<(String, f64)> {
    node.params
        .iter()
        .filter_map(|(key, value)| {
            let param = spec.params.iter().find(|p| p.key == key)?;
            let number = match value {
                ParamValue::Bool(v) => f64::from(*v),
                ParamValue::Int(v) => *v as f64,
                ParamValue::Number(v) => *v,
                ParamValue::Enum(v) => {
                    param.options.iter().position(|x| *x == v).unwrap_or(0) as f64
                }
                _ => return None,
            };
            Some((key.clone(), number))
        })
        .collect()
}
pub(crate) fn accumulate_issues(plan: &GraphPlan) -> Vec<Issue> {
    plan.nodes
        .iter()
        .filter(|n| n.type_id == "gi.accumulate")
        .filter_map(|n| {
            let linked = |key| {
                n.inputs
                    .iter()
                    .any(|i| i.port == key && matches!(i.source, InputSource::Link { .. }))
            };
            let (code, port, message, fix) = match (linked("probes"), linked("irradiance")) {
                (false, false) => (
                    "graph.input_missing",
                    "probes",
                    "GI accumulate needs exactly one of probes or irradiance.",
                    "Link either probes or irradiance.",
                ),
                (true, true) => (
                    "graph.input_taken",
                    "irradiance",
                    "GI accumulate has both probes and irradiance linked.",
                    "Remove the irradiance link or the probes link.",
                ),
                _ => return None,
            };
            Some(Issue {
                code,
                severity: Severity::Error,
                stage: Stage::Apply,
                node: Some(n.id.clone()),
                port: Some(port.into()),
                param: None,
                link: None,
                message: message.into(),
                fix: fix.into(),
            })
        })
        .collect()
}

fn issue(
    code: &'static str,
    node: Option<String>,
    param: Option<String>,
    message: &str,
    fix: &str,
) -> Issue {
    Issue {
        code,
        severity: Severity::Error,
        stage: Stage::Apply,
        node,
        port: None,
        param,
        link: None,
        message: message.into(),
        fix: fix.into(),
    }
}
fn apply_issue(node: &str, error: &RaaError) -> Issue {
    let message = error.to_string();
    let param = message.split(['`', '\'']).nth(1).map(str::to_owned);
    issue(
        "graph.apply",
        Some(node.to_owned()),
        param,
        &message,
        "Use a value accepted by this GI stage.",
    )
}
fn image_stat(port: &str, desc: &ImageDesc) -> ResourceStat {
    let aspect = desc
        .format
        .has_depth_aspect()
        .then_some(wgpu::TextureAspect::DepthOnly);
    let texel = desc.format.block_copy_size(aspect).unwrap_or(0);
    let bytes = (0..desc.mips)
        .map(|mip| {
            u64::from(desc.width.checked_shr(mip).unwrap_or(0).max(1))
                * u64::from(desc.height.checked_shr(mip).unwrap_or(0).max(1))
                * u64::from(desc.layers)
                * u64::from(texel)
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

pub(crate) fn carried_histories(
    mut old: Vec<StageHistory>,
    plan: &GraphPlan,
    previous: Option<&GraphPlan>,
    entries: &[(String, String)],
) -> Vec<StageHistory> {
    let restarts = history_restarts(previous, plan);
    stage_histories(plan, &Catalog::build(), entries)
        .into_iter()
        .map(|(entry, port, producer, source_port)| {
            let consumer = entries
                .iter()
                .find(|(_, e)| *e == entry)
                .map(|(id, _)| id.clone())
                .unwrap_or_default();
            let key = format!("{consumer}.{port}");
            let next_producer = (producer, source_port);
            if !restarts.contains(&key)
                && let Some(i) = old.iter().position(|h| h.key == key)
            {
                let mut history = old.remove(i);
                history.producer = next_producer;
                return history;
            }
            StageHistory {
                key,
                consumer: (consumer, port),
                producer: next_producer,
                images: Vec::new(),
                desc: None,
                current: 0,
                valid: false,
                copied: false,
            }
        })
        .collect()
}

pub(crate) fn state_from_plan(
    plan: &GraphPlan,
    _previous: Option<&GraphPlan>,
    catalog: &Catalog,
    stages: Vec<(String, String, GiStageHandle)>,
    mut old: GiStagesState,
) -> GiStagesState {
    let entries = stage_entries(plan, catalog);
    old.stages = stages;
    old.image_bindings = image_bindings(plan, catalog, &entries);
    old.slot_sources = slot_sources(plan, catalog, &entries);
    old.histories = carried_histories(
        std::mem::take(&mut old.histories),
        plan,
        _previous,
        &entries,
    );
    old.saved_slots.clear();
    old
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::plan_from_json;
    use raa_core::testing::{skip, software_gpu};

    fn plan(nodes: &str, links: &str) -> GraphPlan {
        let text = format!(
            r#"{{"format":"raa.render-graph","version":1,"name":"test","nodes":[{{"id":"pre","type":"pass.prepass"}},{{"id":"forward","type":"pass.forward"}},{{"id":"tone","type":"pass.tonemap"}},{{"id":"out","type":"output.display"}}{nodes}],"links":[{{"from":"forward.color","to":"tone.color"}},{{"from":"tone.color","to":"out.color"}}{links}]}}"#
        );
        plan_from_json(&text).unwrap()
    }
    fn entry_pairs(p: &GraphPlan, c: &Catalog) -> Vec<(String, String)> {
        stage_entries(p, c)
    }

    #[test]
    fn device_free_stage_plan_functions() {
        let c = Catalog::build();
        let p = plan(
            r#",{"id":"a1","type":"gi.atrous"},{"id":"a2","type":"gi.atrous"},{"id":"p","type":"gi.pyramid"},{"id":"acc","type":"gi.accumulate"},{"id":"pr","type":"gi.probes"},{"id":"r","type":"gi.resolve"}"#,
            r#",{"from":"p.pyramid","to":"pr.pyramid"},{"from":"pr.probesFiltered","to":"acc.probes"},{"from":"acc.output","to":"acc.history","history":true},{"from":"acc.output","to":"a1.src"},{"from":"a1.dst","to":"a2.src"},{"from":"a2.dst","to":"r.irradiance"},{"from":"r.gi","to":"forward.gi"}"#,
        );
        let es = entry_pairs(&p, &c);
        assert!(es.contains(&("a1".into(), "gi.atrous#a1".into())));
        assert!(es.contains(&("pre".into(), "prepass".into())));
        assert!(!es.iter().any(|(id, _)| id == "out"));
        assert!(image_bindings(&p, &c, &es).contains(&(
            "gi.atrous#a2".into(),
            "src",
            "gi.atrous#a1".into(),
            "dst".into()
        )));
        assert!(image_bindings(&p, &c, &es).contains(&(
            "gi.probes#pr".into(),
            "depth",
            "prepass".into(),
            "depth".into()
        )));
        assert!(buffer_sources(&p, &c, &es).contains(&(
            "gi.accumulate#acc".into(),
            "probes",
            "gi.probes#pr".into(),
            "probesFiltered".into()
        )));
        assert!(stage_histories(&p, &c, &es).contains(&(
            "gi.accumulate#acc".into(),
            "history",
            "gi.accumulate#acc".into(),
            "output".into()
        )));
        assert!(slot_sources(&p, &c, &es).contains(&(14, "gi.resolve#r".into(), "gi".into())));
        let mut changed = p.clone();
        let atrous = changed.nodes.iter_mut().find(|n| n.id == "a1").unwrap();
        atrous
            .params
            .iter_mut()
            .find(|(k, _)| k == "step")
            .unwrap()
            .1 = ParamValue::Int(2);
        assert_eq!(restarted(None, &p).len(), 6);
        assert_eq!(history_restarts(None, &p), vec!["acc.history"]);
        assert!(restarted(Some(&p), &p).is_empty());
        assert_eq!(restarted(Some(&p), &changed), vec!["a1"]);
        assert!(history_restarts(Some(&p), &changed).is_empty());
        let mut history_changed = p.clone();
        history_changed
            .nodes
            .iter_mut()
            .find(|n| n.id == "acc")
            .unwrap()
            .params
            .iter_mut()
            .find(|(k, _)| k == "historyMax")
            .unwrap()
            .1 = ParamValue::Int(10);
        assert_eq!(
            history_restarts(Some(&p), &history_changed),
            vec!["acc.history"]
        );
    }

    #[test]
    fn defaults_values_and_accumulate_refusals() {
        let c = Catalog::build();
        let p = plan(
            r#",{"id":"trace","type":"ssgi.trace","params":{"quality":"high"}},{"id":"resolve","type":"ssgi.resolve"}"#,
            // The output path keeps both nodes in the plan.
            r#",{"from":"trace.raw","to":"resolve.raw"},{"from":"resolve.ssgi","to":"forward.gi"}"#,
        );
        let node = p.nodes.iter().find(|n| n.id == "trace").unwrap();
        assert!(
            stage_values(node, c.node("ssgi.trace").unwrap()).contains(&("quality".into(), 2.0))
        );
        let bool_spec = c.node("gi.probes").unwrap();
        let value_node = PlanNode {
            params: vec![
                ("screenTraces".into(), ParamValue::Bool(true)),
                ("probeVolumeSteps".into(), ParamValue::Int(4)),
            ],
            ..node.clone()
        };
        let values = stage_values(&value_node, bool_spec);
        assert!(values.contains(&("screenTraces".into(), 1.0)));
        assert!(values.contains(&("probeVolumeSteps".into(), 4.0)));
        let base = plan(
            r#",{"id":"pyramid","type":"gi.pyramid"},{"id":"probes","type":"gi.probes"},{"id":"acc","type":"gi.accumulate"},{"id":"atrous","type":"gi.atrous"},{"id":"resolve2","type":"gi.resolve"}"#,
            r#",{"from":"pyramid.pyramid","to":"probes.pyramid"},{"from":"probes.probesFiltered","to":"acc.probes"},{"from":"acc.output","to":"atrous.src"},{"from":"atrous.dst","to":"resolve2.irradiance"},{"from":"resolve2.gi","to":"forward.gi"}"#,
        );
        assert!(accumulate_issues(&base).is_empty());
        let mut neither = base.clone();
        neither
            .nodes
            .iter_mut()
            .find(|n| n.id == "acc")
            .unwrap()
            .inputs
            .clear();
        assert_eq!(accumulate_issues(&neither)[0].code, "graph.input_missing");
        let mut both = base;
        both.nodes
            .iter_mut()
            .find(|n| n.id == "acc")
            .unwrap()
            .inputs
            .push(crate::plan::PlanInput {
                port: "irradiance".into(),
                source: InputSource::Link {
                    node: "resolve".into(),
                    port: "gi".into(),
                    history: false,
                },
            });
        assert_eq!(
            accumulate_issues(&both)[0].port.as_deref(),
            Some("irradiance")
        );
    }

    #[test]
    fn carried_histories_keep_or_restart_state() {
        let catalog = Catalog::build();
        let p = plan(
            r#",{"id":"acc","type":"gi.accumulate"},{"id":"a","type":"gi.atrous"},{"id":"r","type":"gi.resolve"}"#,
            r#",{"from":"acc.output","to":"acc.history","history":true},{"from":"acc.output","to":"a.src"},{"from":"a.dst","to":"r.irradiance"},{"from":"r.gi","to":"forward.gi"}"#,
        );
        let entries = stage_entries(&p, &catalog);
        let history = StageHistory {
            key: "acc.history".into(),
            consumer: ("acc".into(), "history"),
            producer: ("acc".into(), "output".into()),
            images: Vec::new(),
            desc: None,
            current: 1,
            valid: true,
            copied: true,
        };
        let kept = carried_histories(vec![history], &p, Some(&p), &entries);
        assert_eq!((kept[0].current, kept[0].valid), (1, true));
        let mut changed = p.clone();
        changed
            .nodes
            .iter_mut()
            .find(|n| n.id == "acc")
            .unwrap()
            .params
            .iter_mut()
            .find(|(key, _)| key == "historyMax")
            .unwrap()
            .1 = ParamValue::Int(12);
        let restarted =
            carried_histories(kept, &changed, Some(&p), &stage_entries(&changed, &catalog));
        assert!(restarted[0].images.is_empty());
        assert!(!restarted[0].valid);
    }

    #[test]
    fn history_helpers_flip_only_copied_pairs() {
        let mut history = StageHistory {
            key: "x.history".into(),
            consumer: ("x".into(), "history"),
            producer: ("x".into(), "output".into()),
            images: Vec::new(),
            desc: None,
            current: 0,
            valid: false,
            copied: false,
        };
        assert_eq!(history_binding_index(&history), None);
        flip(&mut history);
        assert_eq!(
            (history.valid, history.current, history.copied),
            (false, 0, false)
        );
        history.copied = true;
        flip(&mut history);
        assert_eq!(
            (history.valid, history.current, history.copied),
            (true, 1, false)
        );
        assert_eq!(history_binding_index_for(true, 2, 1), Some(0));
        assert_eq!(history_binding_index_for(false, 2, 1), None);
        assert_eq!(history_binding_index_for(true, 0, 1), None);
    }

    #[test]
    fn missing_gi_refuses_before_stage_install() {
        let name = "missing_gi_refuses_before_stage_install";
        let Some(gpu) = software_gpu() else {
            skip(name, "no software adapter");
            return;
        };
        let pipeline = Rc::new(RefCell::new(raa_pipeline::Pipeline::new(
            &gpu,
            wgpu::TextureFormat::Rgba8Unorm,
            false,
        )));
        let mut handles = crate::exec::HandleSet::default();
        let mut feature_handles = FeatureHandles {
            gpu: &gpu,
            pipeline: &pipeline,
            handles: &mut handles,
            viewports: 1,
            msaa: 1,
            lit: None,
        };
        let p = plan(
            r#",{"id":"first","type":"ssgi.trace"}"#,
            r#",{"from":"first.raw","to":"forward.gi"}"#,
        );
        let mut extension = GiStagesExtension {
            state: Rc::new(RefCell::new(GiStagesState::default())),
            frame_added: false,
        };
        let issues = extension.apply(&p, None, &mut feature_handles).unwrap_err();
        assert_eq!(
            (issues[0].code, issues[0].stage, issues[0].node.as_deref()),
            ("graph.unavailable", Stage::Apply, Some("first"))
        );
        assert_eq!(
            issues[0].message,
            "the GI feature or lit renderer is not installed"
        );
        assert!(
            pipeline
                .borrow()
                .passes()
                .iter()
                .all(|pass| !pass.name.starts_with("ssgi.trace#"))
        );
    }
}
