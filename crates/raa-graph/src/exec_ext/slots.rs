//! Executor extension `slots`: bind generic G-buffer inputs, forward slot views and instance
//! outputs shown by view nodes.

use super::exec_wiring::plan_fns;
#[cfg(test)]
use crate as raa_graph;
use crate::catalog::Catalog;
use crate::catalog::{FeatureId, ParamBinding, ParamValue};
use crate::error::{Issue, Severity, Stage};
use crate::exec::{ExecExtension, FeatureHandles, default_values, merge_values, write_values};
use crate::plan::InputSource;
use crate::plan::{GraphPlan, PlanNode};
use raa_core::RaaError;
use raa_pipeline::{GraphFrame, PrepareCtx, RecordCtx};
use std::{cell::RefCell, rc::Rc};

const FIXED_SLOTS: [u32; 9] = [0, 3, 7, 8, 9, 10, 14, 15, 20];

/// The executor extension, discovered by build.rs as `exec_slots`.
pub fn extension() -> Box<dyn ExecExtension> {
    Box::new(SlotsExtension::new())
}

fn entry_for<'a>(entries: &'a [(String, String)], node: &str) -> Option<&'a str> {
    entries
        .iter()
        .find(|(id, _)| id == node)
        .map(|(_, entry)| entry.as_str())
}

fn gbuffer_port(key: &str) -> Option<&'static str> {
    match key {
        "depth" => Some("depth"),
        "normal" => Some("normal"),
        "material" => Some("material"),
        "motion" => Some("motion"),
        _ => None,
    }
}

/// Generic linked G-buffer inputs, in plan node and input order.
pub(crate) fn gbuffer_bindings(
    plan: &GraphPlan,
    catalog: &Catalog,
    entries: &[(String, String)],
) -> Vec<(String, &'static str, String, String)> {
    let mut bindings = Vec::new();
    for node in &plan.nodes {
        let Some(entry) = entry_for(entries, &node.id) else {
            continue;
        };
        let Some(spec) = catalog.node(&node.type_id) else {
            continue;
        };
        for input in &node.inputs {
            let Some(port) = gbuffer_port(&input.port) else {
                continue;
            };
            if !spec.inputs.iter().any(|p| p.key == port && p.generic) {
                continue;
            }
            let InputSource::Link {
                node: source,
                port: source_port,
                ..
            } = &input.source
            else {
                continue;
            };
            if let Some(source_entry) = entry_for(entries, source) {
                bindings.push((
                    entry.to_owned(),
                    port,
                    source_entry.to_owned(),
                    source_port.clone(),
                ));
            }
        }
    }
    bindings
}

fn budget_issue(node: &str, port: &str, slot: u32) -> Issue {
    Issue {
        code: "graph.budget",
        severity: Severity::Error,
        stage: Stage::Apply,
        node: Some(node.to_owned()),
        port: Some(port.to_owned()),
        param: None,
        link: None,
        message: format!("input `{port}` names slot {slot}, outside the fixed slot set"),
        fix: "Link the image into one of the forward slot inputs.".to_owned(),
    }
}

fn default_source(
    plan: &GraphPlan,
    entries: &[(String, String)],
    port: &crate::catalog::PortSpec,
) -> Option<(String, String)> {
    for source in port.default_source {
        let Some((type_id, source_port)) = source.rsplit_once('.') else {
            continue;
        };
        let Some(node) = plan.nodes.iter().find(|node| node.type_id == type_id) else {
            continue;
        };
        if let Some(entry) = entry_for(entries, &node.id) {
            return Some((entry.to_owned(), source_port.to_owned()));
        }
    }
    None
}

/// Generic forward slot links that replace their input's default source.
pub(crate) fn slot_bindings(
    plan: &GraphPlan,
    catalog: &Catalog,
    entries: &[(String, String)],
) -> Result<Vec<(u32, String, String)>, Vec<Issue>> {
    let mut bindings = Vec::new();
    for node in plan
        .nodes
        .iter()
        .filter(|node| node.type_id == "pass.forward")
    {
        let Some(spec) = catalog.node(&node.type_id) else {
            continue;
        };
        for input in &node.inputs {
            let Some(port) = spec.inputs.iter().find(|port| port.key == input.port) else {
                continue;
            };
            let Some(slot) = port.slot.filter(|_| port.generic) else {
                continue;
            };
            let InputSource::Link {
                node: source,
                port: source_port,
                ..
            } = &input.source
            else {
                continue;
            };
            if !FIXED_SLOTS.contains(&slot) {
                return Err(vec![budget_issue(&node.id, port.key, slot)]);
            }
            let Some(source_entry) = entry_for(entries, source) else {
                continue;
            };
            let actual = (source_entry.to_owned(), source_port.clone());
            if default_source(plan, entries, port).as_ref() != Some(&actual) {
                bindings.push((slot, actual.0, actual.1));
            }
        }
    }
    Ok(bindings)
}

/// View inputs linked from a later GTAO or contact instance, in plan node and input order.
pub(crate) fn view_rebinds(
    plan: &GraphPlan,
    catalog: &Catalog,
    entries: &[(String, String)],
) -> Vec<(String, &'static str, String, String)> {
    let mut bindings = Vec::new();
    for node in plan
        .nodes
        .iter()
        .filter(|node| node.type_id.starts_with("view."))
    {
        let Some(view_entry) = entry_for(entries, &node.id) else {
            continue;
        };
        let Some(spec) = catalog.node(&node.type_id) else {
            continue;
        };
        for input in &node.inputs {
            let Some(port) = spec.inputs.iter().find(|port| port.key == input.port) else {
                continue;
            };
            let InputSource::Link {
                node: source,
                port: source_port,
                ..
            } = &input.source
            else {
                continue;
            };
            let Some(source_node) = plan.nodes.iter().find(|candidate| candidate.id == *source)
            else {
                continue;
            };
            if !matches!(
                source_node.type_id.as_str(),
                "pass.ao.gtao" | "pass.shadow.contact"
            ) {
                continue;
            }
            let Some(source_entry) = entry_for(entries, source) else {
                continue;
            };
            if source_entry.contains('#') {
                bindings.push((
                    view_entry.to_owned(),
                    port.key,
                    source_entry.to_owned(),
                    source_port.clone(),
                ));
            }
        }
    }
    bindings
}

/// Later `pass.ao.gtao` and `pass.shadow.contact` nodes as (node id, type id): the entries holding
/// `#`, in plan order.
pub(crate) fn slot_instances(
    plan: &GraphPlan,
    entries: &[(String, String)],
) -> Vec<(String, String)> {
    plan.nodes
        .iter()
        .filter(|node| {
            matches!(
                node.type_id.as_str(),
                "pass.ao.gtao" | "pass.shadow.contact"
            )
        })
        .filter(|node| entry_for(entries, &node.id).is_some_and(|entry| entry.contains('#')))
        .map(|node| (node.id.clone(), node.type_id.clone()))
        .collect()
}

/// The handle of an instance pass `ao.gtao#<id>` or `shadow.contact#<id>`.
#[derive(Clone)]
enum SlotInstance {
    Gtao(raa_gtao::GtaoHandle),
    Contact(raa_contact::ContactHandle),
}

impl SlotInstance {
    fn type_id(&self) -> &'static str {
        match self {
            SlotInstance::Gtao(_) => "pass.ao.gtao",
            SlotInstance::Contact(_) => "pass.shadow.contact",
        }
    }

    fn feature(&self) -> FeatureId {
        match self {
            SlotInstance::Gtao(_) => FeatureId::Gtao,
            SlotInstance::Contact(_) => FeatureId::Contact,
        }
    }

    fn set_tunable(&self, key: &str, value: f64) -> Result<(), RaaError> {
        match self {
            SlotInstance::Gtao(h) => h.set_tunable(key, value),
            SlotInstance::Contact(h) => h.set_tunable(key, value),
        }
    }
}

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
        fix: "Change the refused value or retry applying the render graph.".to_owned(),
    }]
}

fn node_value<'a>(node: &'a PlanNode, key: &str) -> Option<&'a ParamValue> {
    node.params
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value)
}

/// The Setting changes and Tunable values of `node` for `feature`.
#[allow(clippy::type_complexity)]
fn node_values(
    catalog: &Catalog,
    node: &PlanNode,
    feature: &FeatureId,
) -> (Vec<(String, ParamValue)>, Vec<(&'static str, f64)>) {
    let (mut settings, mut tunables) = (Vec::new(), Vec::new());
    let Some(spec) = catalog.node(&node.type_id) else {
        return (settings, tunables);
    };
    for param in &spec.params {
        let Some(value) = node_value(node, param.key) else {
            continue;
        };
        match (&param.binding, value) {
            (ParamBinding::Setting { feature: f, key }, _) if f == feature => {
                settings.push(((*key).to_owned(), value.clone()))
            }
            (ParamBinding::Tunable { feature: f, key }, ParamValue::Number(x)) if f == feature => {
                tunables.push((*key, *x))
            }
            (ParamBinding::Tunable { feature: f, key }, ParamValue::Int(i)) if f == feature => {
                tunables.push((*key, *i as f64))
            }
            _ => {}
        }
    }
    (settings, tunables)
}

/// Writes each instance node's Setting values merged over the feature defaults (the installed
/// handle set aside behind a stand-in), then its Tunable values.
fn write_instances(
    plan: &GraphPlan,
    catalog: &Catalog,
    instances: &[(String, String, SlotInstance)],
    h: &mut FeatureHandles<'_>,
) -> Result<(), Vec<Issue>> {
    for (id, _, instance) in instances {
        let Some(node) = plan.nodes.iter().find(|node| node.id == *id) else {
            continue;
        };
        let feature = instance.feature();
        let (settings, tunables) = node_values(catalog, node, &feature);
        let merged = merge_values(&feature, &default_values(&feature), &settings)
            .map_err(|(key, message)| apply_issue(Some(id), Some(&key), &message))?;
        let installed = std::mem::take(&mut *h.handles);
        match instance {
            SlotInstance::Gtao(handle) => h.handles.gtao = Some(handle.clone()),
            SlotInstance::Contact(handle) => h.handles.contact = Some(handle.clone()),
        }
        let written = write_values(h, &feature, &merged);
        *h.handles = installed;
        written.map_err(|e| apply_issue(Some(id), None, e.message()))?;
        for (key, value) in tunables {
            instance
                .set_tunable(key, value)
                .map_err(|e| apply_issue(Some(id), Some(key), e.message()))?;
        }
    }
    Ok(())
}

/// Sets the Tunable values of the first gtao and contact node on the installed handles.
fn write_installed_tunables(
    plan: &GraphPlan,
    catalog: &Catalog,
    h: &mut FeatureHandles<'_>,
) -> Result<(), RaaError> {
    for (type_id, feature) in [
        ("pass.ao.gtao", FeatureId::Gtao),
        ("pass.shadow.contact", FeatureId::Contact),
    ] {
        let Some(node) = plan.nodes.iter().find(|node| node.type_id == type_id) else {
            continue;
        };
        for (key, value) in node_values(catalog, node, &feature).1 {
            match (&feature, &h.handles.gtao, &h.handles.contact) {
                (FeatureId::Gtao, Some(handle), _) => handle.set_tunable(key, value)?,
                (FeatureId::Contact, _, Some(handle)) => handle.set_tunable(key, value)?,
                _ => {}
            }
        }
    }
    Ok(())
}

/// LIFECYCLE: `SlotsExtension::apply` builds `SlotsState` (G-buffer bindings, slot bindings, view
/// rebinds) and re-adds the `slots` graph frame, removing the previous one. The first
/// `SlotsExtension::frame` after an apply moves `slots` to the end of the graph frames, so it runs
/// after `views` and `wiring`. `SlotsExtension::clear` restores overridden slot views through
/// `Pipeline::lighting_mut` and removes the frame. Each frame begins with an unrestored flag; its
/// first `before_prepare` restores the previous frame's overrides.
#[derive(Default)]
struct SlotsState {
    gbuffer: Vec<(String, &'static str, String, String)>,
    slots: Vec<(u32, String, String)>,
    views: Vec<(String, &'static str, String, String)>,
    saved: Vec<(u32, wgpu::TextureView)>,
    restored: bool,
}

/// Handoff: `SlotsExtension` is extended by card 'Slots 3/3: gtao and contact instance passes in
/// the slots extension'. Today `apply` stores G-buffer bindings, forward slot overrides and view
/// rebinds, then adds the `slots` frame; `frame` moves it behind `views` and `wiring` once, and
/// `clear` restores saved slot views. The extension holds its catalog, shared `SlotsState`, and
/// `moved` flag. The upstream slot-view handoff is `crates/raa-pipeline/src/lighting.rs:1285
/// LightingResources::slot_view`.
pub(crate) struct SlotsExtension {
    catalog: Catalog,
    state: Rc<RefCell<SlotsState>>,
    moved: bool,
    instances: Vec<(String, String, SlotInstance)>,
    plan: Option<GraphPlan>,
}

impl SlotsExtension {
    fn new() -> Self {
        Self {
            catalog: Catalog::build(),
            state: Rc::new(RefCell::new(SlotsState::default())),
            moved: false,
            instances: Vec::new(),
            plan: None,
        }
    }

    /// Removes, installs and enables the instance passes of `plan` and writes their values,
    /// recording in `installed` and `removed` what it changed.
    #[allow(clippy::type_complexity)]
    fn commit_instances(
        &mut self,
        plan: &GraphPlan,
        entries: &[(String, String)],
        h: &mut FeatureHandles<'_>,
        installed: &mut Vec<String>,
        removed: &mut Vec<(String, Box<dyn raa_pipeline::FeaturePass>)>,
    ) -> Result<(), Vec<Issue>> {
        let kept: Vec<(String, String)> = self
            .instances
            .iter()
            .map(|(id, _, handle)| (id.clone(), handle.type_id().to_owned()))
            .collect();
        let (install, keep, remove) =
            plan_fns::instance_plan(&kept, &slot_instances(plan, entries));
        for id in &remove {
            let Some(index) = self.instances.iter().position(|(node, _, _)| node == id) else {
                continue;
            };
            let entry = self.instances[index].1.clone();
            let pass = h
                .pipeline
                .borrow_mut()
                .remove(&entry)
                .map_err(|e| apply_issue(Some(id), None, e.message()))?;
            removed.push((entry, pass));
            self.instances.remove(index);
        }
        for (id, type_id) in &install {
            let Some(entry) = entry_for(entries, id) else {
                continue;
            };
            let handle = if type_id == "pass.ao.gtao" {
                raa_gtao::GtaoFeature::install_instance(h.gpu, h.pipeline, id)
                    .map(SlotInstance::Gtao)
            } else {
                raa_contact::ContactFeature::install_instance(h.gpu, h.pipeline, id)
                    .map(SlotInstance::Contact)
            }
            .map_err(|e| apply_issue(Some(id), None, e.message()))?;
            installed.push(entry.to_owned());
            self.instances.push((id.clone(), entry.to_owned(), handle));
        }
        for (id, _) in &keep {
            if let Some((_, entry, _)) = self.instances.iter().find(|(node, _, _)| node == id) {
                h.pipeline
                    .borrow_mut()
                    .set_enabled(entry, true)
                    .map_err(|e| apply_issue(Some(id), None, e.message()))?;
            }
        }
        write_instances(plan, &self.catalog, &self.instances, h)
    }
}

impl ExecExtension for SlotsExtension {
    fn name(&self) -> &'static str {
        "slots"
    }

    fn apply(
        &mut self,
        plan: &GraphPlan,
        previous: Option<&GraphPlan>,
        h: &mut FeatureHandles<'_>,
    ) -> Result<(), Vec<Issue>> {
        let entries = plan_fns::node_entries(plan, &self.catalog);
        let gbuffer = gbuffer_bindings(plan, &self.catalog, &entries);
        let slots = slot_bindings(plan, &self.catalog, &entries)?;
        let views = view_rebinds(plan, &self.catalog, &entries);
        let snapshot = self.instances.clone();
        let (mut installed, mut removed) = (Vec::new(), Vec::new());
        let mut result = self.commit_instances(plan, &entries, h, &mut installed, &mut removed);
        if result.is_ok() {
            let frame = SlotsFrame {
                state: self.state.clone(),
            };
            let mut pipeline = h.pipeline.borrow_mut();
            let old = pipeline.remove_graph_frame("slots");
            if let Err(error) = pipeline.add_graph_frame("slots", Box::new(frame)) {
                if let Some(old) = old {
                    let _ = pipeline.add_graph_frame("slots", old);
                }
                result = Err(apply_issue(None, None, error.message()));
            }
        }
        if let Err(issues) = result {
            {
                let mut pipeline = h.pipeline.borrow_mut();
                for entry in installed.iter().rev() {
                    let _ = pipeline.remove(entry);
                }
                for (entry, pass) in removed {
                    if pipeline.add(pass).is_ok() {
                        let _ = pipeline.set_enabled(&entry, true);
                    }
                }
            }
            self.instances = snapshot;
            if let Some(previous) = previous {
                let _ = write_instances(previous, &self.catalog, &self.instances, h);
            }
            return Err(issues);
        }
        self.plan = Some(plan.clone());
        let mut state = self.state.borrow_mut();
        state.gbuffer = gbuffer;
        state.slots = slots;
        state.views = views;
        self.moved = false;
        Ok(())
    }

    fn clear(&mut self, h: &mut FeatureHandles<'_>) {
        let mut pipeline = h.pipeline.borrow_mut();
        let mut state = self.state.borrow_mut();
        for (slot, view) in state.saved.drain(..) {
            let _ = pipeline.lighting_mut().set_slot_view(slot, view);
        }
        *state = SlotsState::default();
        pipeline.remove_graph_frame("slots");
        for (_, entry, _) in self.instances.drain(..) {
            let _ = pipeline.remove(&entry);
        }
        self.plan = None;
        self.moved = false;
    }

    fn reorder(&mut self, _plan: &GraphPlan, h: &mut FeatureHandles<'_>) -> Result<(), RaaError> {
        let mut pipeline = h.pipeline.borrow_mut();
        for (_, entry, _) in &self.instances {
            pipeline.set_enabled(entry, true)?;
        }
        Ok(())
    }

    fn frame(&mut self, h: &mut FeatureHandles<'_>) -> Result<(), RaaError> {
        for (_, _, instance) in &self.instances {
            match (instance, &h.handles.gtao, &h.handles.contact) {
                (SlotInstance::Gtao(handle), Some(source), _) => handle.copy_camera_from(source),
                (SlotInstance::Contact(handle), _, Some(source)) => handle.copy_camera_from(source),
                _ => {}
            }
        }
        if self.moved {
            return Ok(());
        }
        if let Some(plan) = &self.plan {
            write_installed_tunables(plan, &self.catalog, h)?;
        }
        let mut pipeline = h.pipeline.borrow_mut();
        let old = pipeline.remove_graph_frame("slots");
        let frame = SlotsFrame {
            state: self.state.clone(),
        };
        if let Err(error) = pipeline.add_graph_frame("slots", Box::new(frame)) {
            if let Some(old) = old {
                let _ = pipeline.add_graph_frame("slots", old);
            }
            return Err(error);
        }
        self.moved = true;
        Ok(())
    }
}

struct SlotsFrame {
    state: Rc<RefCell<SlotsState>>,
}

impl GraphFrame for SlotsFrame {
    fn begin(&mut self, _table: &raa_pipeline::ResourceTable) {
        self.state.borrow_mut().restored = false;
    }

    fn before_prepare(&mut self, pass: &str, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let mut state = self.state.borrow_mut();
        if !state.restored {
            state.restored = true;
            for (slot, view) in state.saved.drain(..) {
                let _ = ctx.lighting.set_slot_view(slot, view);
            }
        }
        for (entry, port, source, source_port) in &state.gbuffer {
            if entry == pass
                && let Some(image) = ctx.resources.get(source, source_port)
            {
                ctx.resources.bind_input(pass, port, image);
            }
        }
        let SlotsState {
            slots,
            saved: saved_overrides,
            ..
        } = &mut *state;
        for (slot, source, source_port) in slots.iter() {
            if pass == "forward"
                && let Some(image) = ctx.resources.get(source, source_port)
                && let Some(previous_view) = ctx.lighting.slot_view(*slot)
            {
                ctx.lighting.set_slot_view(*slot, image.view.clone())?;
                saved_overrides.push((*slot, previous_view));
            }
        }
        Ok(())
    }

    fn before_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let state = self.state.borrow();
        for (entry, port, source, source_port) in &state.views {
            if entry == pass
                && let Some(image) = ctx.resources.get(source, source_port)
            {
                ctx.resources.bind_input(pass, port, image);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TWO_AO: &str = r#"{"format":"raa.render-graph","version":1,"name":"Simple forward two AO branches","nodes":[{"id":"skyAtmosphere","type":"pass.sky.atmosphere"},{"id":"skyCloudLighting","type":"pass.sky.cloud-lighting"},{"id":"skyEnv","type":"pass.sky.env"},{"id":"shadowCsm","type":"pass.shadow.csm"},{"id":"prepass","type":"pass.prepass"},{"id":"lightTiles","type":"pass.light.tiles"},{"id":"aoNarrow","type":"pass.ao.gtao","params":{"radius":0.3,"intensity":1,"power":2}},{"id":"aoWide","type":"pass.ao.gtao","params":{"radius":2,"intensity":1,"power":2}},{"id":"forward","type":"pass.forward"},{"id":"skyBackground","type":"pass.sky.background"},{"id":"tonemap","type":"pass.tonemap","params":{"mapper":"agx","exposureEv":0}},{"id":"vNarrow","type":"view.image","params":{"channels":"r","label":"ao 0.3 m"}},{"id":"vWide","type":"view.image","params":{"channels":"r","label":"ao 2 m"}},{"id":"aoNarrowVsWide","type":"view.compare","params":{"mode":"split","position":0.5}},{"id":"vFinal","type":"view.image","params":{"label":"final, forward ao from 2 m"}},{"id":"grid","type":"view.grid","params":{"columns":2}},{"id":"out","type":"output.display"}],"links":[{"from":"prepass.depth","to":"lightTiles.depth"},{"from":"shadowCsm.shadowMap","to":"forward.shadowMap"},{"from":"lightTiles.lightTiles","to":"forward.lightTiles"},{"from":"aoWide.ao","to":"forward.ao"},{"from":"forward.color","to":"tonemap.color"},{"from":"aoNarrow.ao","to":"vNarrow.src"},{"from":"aoWide.ao","to":"vWide.src"},{"from":"vNarrow.color","to":"aoNarrowVsWide.a"},{"from":"vWide.color","to":"aoNarrowVsWide.b"},{"from":"tonemap.color","to":"vFinal.src"},{"from":"aoNarrowVsWide.color","to":"grid.t0"},{"from":"vFinal.color","to":"grid.t1"},{"from":"grid.color","to":"out.color"}]}"#;

    #[test]
    fn exec_slots_simple_forward_uses_default_slot_source() {
        let catalog = Catalog::build();
        let plan = raa_graph::plan_from_json(include_str!("../../graphs/simple-forward.json"))
            .expect("Simple forward plans");
        let entries = plan_fns::node_entries(&plan, &catalog);
        assert_eq!(
            gbuffer_bindings(&plan, &catalog, &entries),
            vec![(
                "light.tiles".to_owned(),
                "depth",
                "prepass".to_owned(),
                "depth".to_owned()
            )]
        );
        assert_eq!(slot_bindings(&plan, &catalog, &entries), Ok(Vec::new()));
        assert!(view_rebinds(&plan, &catalog, &entries).is_empty());
        assert!(slot_instances(&plan, &entries).is_empty());
    }

    #[test]
    fn exec_slots_two_ao_rebinds_wide_instance() {
        let catalog = Catalog::build();
        let plan = raa_graph::plan_from_json(TWO_AO).expect("two-AO document plans");
        let entries = plan_fns::node_entries(&plan, &catalog);
        assert_eq!(
            gbuffer_bindings(&plan, &catalog, &entries),
            vec![(
                "light.tiles".to_owned(),
                "depth",
                "prepass".to_owned(),
                "depth".to_owned()
            )]
        );
        assert_eq!(
            slot_bindings(&plan, &catalog, &entries),
            Ok(vec![(7, "ao.gtao#aoWide".to_owned(), "ao".to_owned())])
        );
        assert_eq!(
            slot_instances(&plan, &entries),
            vec![("aoWide".to_owned(), "pass.ao.gtao".to_owned())]
        );
        assert_eq!(
            view_rebinds(&plan, &catalog, &entries),
            vec![(
                "view#vWide".to_owned(),
                "src",
                "ao.gtao#aoWide".to_owned(),
                "ao".to_owned()
            )]
        );
    }

    #[test]
    fn exec_slots_refuses_catalog_slot_outside_fixed_set() {
        let mut catalog = Catalog::build();
        let forward = catalog
            .nodes
            .iter_mut()
            .find(|node| node.id == "pass.forward")
            .expect("forward type");
        forward
            .inputs
            .iter_mut()
            .find(|port| port.key == "ao")
            .expect("forward ao input")
            .slot = Some(5);
        let plan = raa_graph::plan_from_json(TWO_AO).expect("two-AO document plans");
        let entries = plan_fns::node_entries(&plan, &catalog);
        let issue = &slot_bindings(&plan, &catalog, &entries).unwrap_err()[0];
        assert_eq!(issue.code, "graph.budget");
        assert_eq!(issue.node.as_deref(), Some("forward"));
        assert_eq!(issue.port.as_deref(), Some("ao"));
    }
}
