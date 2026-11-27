//! The graph executor (S53, S54). It plans a document, applies the plan to the live pipeline and
//! features in rank order, and restores the configuration that was live before the first graph when
//! a change is refused or the graph is cleared. `settings_targets` and `external_changes` are the
//! device-free helpers that the executor and its tests share.

#[path = "area_model.rs"]
mod area_model;

pub use area_model::AreaGraphModel;
use area_model::{AreaModelFrame, SharedAreaFrame};
#[path = "real_model.rs"]
mod real_model;
#[path = "denoise.rs"]
mod denoise;
use denoise::{DenoiseFrame, SharedDenoiseFrame};
use real_model::{NeuralRoutes, RealModelFrame, SharedNeuralRoutes, SharedRealFrame};
pub use real_model::{RealModelHeads, RealModelPackage, RealModelSelection, RealModelStats};

use raa_core::RaaError;
use raa_pipeline::{PassInfo, PassRole, Pipeline};
use std::{cell::RefCell, rc::Rc};

use super::{
    AppliedDescs, ExecExtension, FeatureHandles, NodeStats, SetResult, SettingsSnapshot,
    export_snapshot, graph_owns_virtual_geometry, merge_values, pipeline_with_virtual_geometry,
    read_values, snapshot_keys, write_values,
};
use crate::catalog::{Catalog, FeatureId, ParamBinding, ParamSpec, ParamValue, Rebuild};
use crate::error::{Issue, Severity, Stage};
use crate::format::GraphDoc;
use crate::plan::{GraphPlan, InputSource, plan_from_doc};
use crate::runtime_lighting::{ISOLATION, SPHERICAL, SPHERICAL_TYPE};
use crate::validate::{ValidateCtx, canonical_doc};

/// The pipeline entry that the contributors follow, and the entry that `output.display` follows.
const HOST_ENTRY: &str = "forward";
const LAST_CHAIN_ENTRY: &str = "tonemap";

/// The features in S53 rank order. The pipeline is applied apart from them.
const RANK: [FeatureId; 20] = [
    FeatureId::Tonemap,
    FeatureId::Shadow,
    FeatureId::Vg,
    FeatureId::GiQuality,
    FeatureId::Gi,
    SPHERICAL,
    ISOLATION,
    FeatureId::AreaShadows,
    FeatureId::AreaShading,
    FeatureId::Gtao,
    FeatureId::Contact,
    FeatureId::Ssr,
    FeatureId::Fog,
    FeatureId::Shafts,
    FeatureId::Oit,
    FeatureId::Taa,
    FeatureId::Dof,
    FeatureId::Bloom,
    FeatureId::Grading,
    FeatureId::Cel,
];

/// The executor of one Engine's render graph. It owns its catalog, the active document and plan,
/// the snapshot of the configuration before the graph and the descriptors the graph applied.
pub struct GraphExecutor {
    catalog: Catalog,
    extensions: Vec<Box<dyn ExecExtension>>,
    doc: Option<GraphDoc>,
    plan: Option<GraphPlan>,
    /// The configuration before the first graph: taken on first activation, dropped by `clear_graph`.
    no_graph: Option<SettingsSnapshot>,
    /// The configuration the graph applied: captured after each activation and each restore of an
    /// external change. `frame` compares the live state with it.
    applied: Option<SettingsSnapshot>,
    /// The descriptors the graph applied, compared by `frame` through `AppliedDescs::changed`.
    applied_descs: AppliedDescs,
    /// `Pipeline::order_generation` when `applied` was captured: the order is compared only when it moved.
    applied_order_generation: u64,
    /// The features `frame` found changed; kept between frames so a steady frame allocates nothing.
    scratch: Vec<FeatureId>,
    /// The index of the latest frame of the active graph; `None` before one.
    frame: Option<u64>,
    last_issues: Vec<Issue>,
    area_model: Option<Rc<RefCell<AreaModelFrame>>>,
    real_model: Option<Rc<RefCell<RealModelFrame>>>,
    neural_routes: Option<Rc<RefCell<NeuralRoutes>>>,
    denoise: Option<Rc<RefCell<DenoiseFrame>>>,
}

impl Default for GraphExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphExecutor {
    /// An executor with no graph. Its catalog is built here and its extensions come from the
    /// registry, in registry order.
    pub fn new() -> GraphExecutor {
        GraphExecutor {
            catalog: Catalog::build(),
            extensions: crate::registry::extensions_all(),
            doc: None,
            plan: None,
            no_graph: None,
            applied: None,
            applied_descs: AppliedDescs::default(),
            applied_order_generation: 0,
            scratch: Vec::new(),
            frame: None,
            last_issues: Vec::new(),
            area_model: None,
            real_model: None,
            neural_routes: None,
            denoise: None,
        }
    }

    /// Validates and plans `doc`, then activates it with its canonical text (`canonical_doc`). A
    /// refused document changes nothing: `last_issues` and the result carry its issues. The planning
    /// warnings are appended to the result's issues.
    pub fn set_graph(
        &mut self,
        doc: GraphDoc,
        ctx: &ValidateCtx<'_>,
        h: &mut FeatureHandles<'_>,
    ) -> SetResult {
        self.set_graph_with_area_model(doc, ctx, h, None)
    }

    pub fn set_graph_with_area_model(
        &mut self,
        doc: GraphDoc,
        ctx: &ValidateCtx<'_>,
        h: &mut FeatureHandles<'_>,
        model: Option<AreaGraphModel>,
    ) -> SetResult {
        self.set_graph_with_models(doc, ctx, h, model, None)
    }

    pub fn set_graph_with_models(
        &mut self,
        doc: GraphDoc,
        ctx: &ValidateCtx<'_>,
        h: &mut FeatureHandles<'_>,
        area: Option<AreaGraphModel>,
        real: Option<RealModelSelection>,
    ) -> SetResult {
        match plan_from_doc(&doc, ctx) {
            Err(issues) => {
                self.last_issues.clone_from(&issues);
                SetResult {
                    ok: false,
                    issues,
                    rebuilt: 0,
                }
            }
            Ok((plan, warnings)) => {
                let canonical = canonical_doc(&doc, &self.catalog);
                let mut result = self.set_plan_with_models(plan, Some(canonical), h, area, real);
                result.issues.extend(warnings);
                self.last_issues.clone_from(&result.issues);
                result
            }
        }
    }

    /// Activates `plan`, with its document (`None` for a frozen plan). Nothing changes unless every
    /// step applies: the pipeline, the features in rank order, then the extensions. A refusal undoes
    /// the steps already taken and returns `ok` false.
    pub fn set_plan(
        &mut self,
        plan: GraphPlan,
        doc: Option<GraphDoc>,
        h: &mut FeatureHandles<'_>,
    ) -> SetResult {
        self.set_plan_with_area_model(plan, doc, h, None)
    }

    pub fn set_plan_with_area_model(
        &mut self,
        plan: GraphPlan,
        doc: Option<GraphDoc>,
        h: &mut FeatureHandles<'_>,
        model: Option<AreaGraphModel>,
    ) -> SetResult {
        self.set_plan_with_models(plan, doc, h, model, None)
    }

    pub fn set_plan_with_models(
        &mut self,
        plan: GraphPlan,
        doc: Option<GraphDoc>,
        h: &mut FeatureHandles<'_>,
        area: Option<AreaGraphModel>,
        real: Option<RealModelSelection>,
    ) -> SetResult {
        let result = self.activate_with_models(plan, doc, h, area, real);
        self.last_issues.clone_from(&result.issues);
        result
    }

    fn activate_with_models(
        &mut self,
        plan: GraphPlan,
        doc: Option<GraphDoc>,
        h: &mut FeatureHandles<'_>,
        model: Option<AreaGraphModel>,
        real: Option<RealModelSelection>,
    ) -> SetResult {
        const FRAME: &str = "area_model";
        let routes = match NeuralRoutes::from_plan(&plan) {
            Ok(routes) => routes,
            Err(error) => {
                return area_model::refused(area_model::issue(
                    "graph.neural_route",
                    error.to_string(),
                ));
            }
        };
        let denoise_routes = match denoise::routes_from_plan(&plan, &self.catalog) {
            Ok(routes) => routes,
            Err(error) => {
                return area_model::refused(area_model::issue(error.code(), error.to_string()));
            }
        };
        if model.is_some() && real.as_ref().is_some_and(|r| r.heads.area) {
            return area_model::refused(area_model::issue(
                "graph.neural_route",
                "choose one area reconstruction operator",
            ));
        }
        let real = match real
            .map(|model| RealModelFrame::prepare(model, &plan, h))
            .transpose()
        {
            Ok(real) => real,
            Err(error) => {
                return area_model::refused(area_model::issue(error.code(), error.to_string()));
            }
        };
        let candidate = match model
            .map(|model| AreaModelFrame::prepare(model, &plan, h))
            .transpose()
        {
            Ok(candidate) => candidate.map(|frame| Rc::new(RefCell::new(frame))),
            Err(issue) => return area_model::refused(issue),
        };
        if let Some(frame) = &candidate
            && self.area_model.is_none()
            && let Err(error) = h
                .pipeline
                .borrow_mut()
                .add_graph_frame(FRAME, Box::new(SharedAreaFrame(frame.clone())))
        {
            return area_model::refused(area_model::issue("graph.model_device", error.to_string()));
        }
        let result = self.activate(plan, doc, h);
        if result.ok {
            self.clear_real_model(h);
            self.install_neural_routes(h, routes);
            self.install_denoise(h, denoise_routes);
            if let Some(real) = real {
                self.install_real_model(h, real)
                    .expect("executor owns neural frame name");
            }
        }
        match (result.ok, candidate, &self.area_model) {
            (true, Some(candidate), Some(active)) => {
                std::mem::swap(&mut *active.borrow_mut(), &mut *candidate.borrow_mut());
            }
            (true, Some(candidate), None) => self.area_model = Some(candidate),
            (true, None, Some(_)) => self.clear_area_model(h),
            (false, Some(_), None) => {
                h.pipeline.borrow_mut().remove_graph_frame(FRAME);
            }
            _ => {}
        }
        result
    }

    /// Select a validated model on the active editable or frozen graph.
    pub fn select_real_model(
        &mut self,
        h: &mut FeatureHandles<'_>,
        selection: Option<RealModelSelection>,
    ) -> Result<(), RaaError> {
        let Some(selection) = selection else {
            self.clear_real_model(h);
            return Ok(());
        };
        let Some(plan) = self.plan.as_ref() else {
            return Err(RaaError::FrameGraph {
                code: "graph.neural_route",
                message:
                    "select an explicit render graph before enabling real-frame reconstruction"
                        .into(),
            });
        };
        if selection.heads.area && self.area_model.is_some() {
            return Err(RaaError::FrameGraph {
                code: "graph.neural_route",
                message: "choose one area reconstruction operator".into(),
            });
        }
        let candidate = RealModelFrame::prepare(selection, plan, h)?;
        self.install_real_model(h, candidate)
    }

    fn install_real_model(
        &mut self,
        h: &mut FeatureHandles<'_>,
        candidate: RealModelFrame,
    ) -> Result<(), RaaError> {
        if let Some(active) = &self.real_model {
            *active.borrow_mut() = candidate;
        } else {
            let frame = Rc::new(RefCell::new(candidate));
            h.pipeline.borrow_mut().add_graph_frame(
                "neural.real-frame",
                Box::new(SharedRealFrame(frame.clone())),
            )?;
            self.real_model = Some(frame);
        }
        if let Some(taa) = &h.handles.taa {
            taa.invalidate();
        }
        Ok(())
    }

    pub fn real_model_stats(&self) -> Option<(RealModelHeads, RealModelStats)> {
        self.real_model.as_ref().map(|frame| {
            let frame = frame.borrow();
            (frame.heads(), frame.stats())
        })
    }

    fn install_neural_routes(&mut self, h: &mut FeatureHandles<'_>, routes: NeuralRoutes) {
        const FRAME: &str = "neural.native-inputs";
        if let Some(active) = self.neural_routes.take() {
            h.pipeline.borrow_mut().remove_graph_frame(FRAME);
            drop(active);
        }
        if routes.active() {
            let routes = Rc::new(RefCell::new(routes));
            h.pipeline
                .borrow_mut()
                .add_graph_frame(FRAME, Box::new(SharedNeuralRoutes(routes.clone())))
                .expect("executor owns neural bypass frame name");
            self.neural_routes = Some(routes);
        }
    }

    fn install_denoise(&mut self, h: &mut FeatureHandles<'_>, routes: Vec<denoise::Route>) {
        const FRAME: &str = "denoise";
        if self.denoise.take().is_some() {
            h.pipeline.borrow_mut().remove_graph_frame(FRAME);
        }
        if !routes.is_empty() {
            let frame = Rc::new(RefCell::new(DenoiseFrame::new(routes, h)));
            h.pipeline
                .borrow_mut()
                .add_graph_frame(FRAME, Box::new(SharedDenoiseFrame(frame.clone())))
                .expect("executor owns the denoise frame name");
            self.denoise = Some(frame);
        }
    }

    fn clear_real_model(&mut self, h: &mut FeatureHandles<'_>) {
        if self.real_model.take().is_some() {
            h.pipeline
                .borrow_mut()
                .remove_graph_frame("neural.real-frame");
            if let Some(taa) = &h.handles.taa {
                taa.invalidate();
            }
        }
    }

    fn clear_area_model(&mut self, h: &mut FeatureHandles<'_>) {
        if self.area_model.take().is_some() {
            h.pipeline.borrow_mut().remove_graph_frame("area_model");
        }
    }

    /// Clears every extension, then restores the no-graph configuration (pipeline enable and order,
    /// then each feature whose live value differs, in rank order) and drops the graph. If the restore
    /// is refused the error is returned and the graph stays recorded.
    pub fn clear_graph(&mut self, h: &mut FeatureHandles<'_>) -> Result<(), RaaError> {
        for extension in &mut self.extensions {
            extension.clear(h);
        }
        if let Some(no_graph) = &self.no_graph {
            restore(no_graph, h)?;
        }
        self.clear_area_model(h);
        self.clear_real_model(h);
        if self.neural_routes.take().is_some() {
            h.pipeline
                .borrow_mut()
                .remove_graph_frame("neural.native-inputs");
        }
        if self.denoise.take().is_some() {
            h.pipeline.borrow_mut().remove_graph_frame("denoise");
        }
        self.no_graph = None;
        self.doc = None;
        self.plan = None;
        self.applied = None;
        self.applied_descs = AppliedDescs::default();
        h.pipeline.borrow_mut().set_display_readable(true);
        self.frame = None;
        self.last_issues.clear();
        Ok(())
    }

    /// The canonical document of the active graph; `None` without a graph or for a frozen plan.
    pub fn active_doc(&self) -> Option<&GraphDoc> {
        self.doc.as_ref()
    }

    /// The plan of the active graph.
    pub fn active_plan(&self) -> Option<&GraphPlan> {
        self.plan.as_ref()
    }

    /// The issues of the latest `set_graph` or `set_plan`; `clear_graph` empties them.
    pub fn last_issues(&self) -> &[Issue] {
        &self.last_issues
    }

    /// The index of the latest frame of the active graph; `None` before its first frame.
    pub fn frame_index(&self) -> Option<u64> {
        self.frame
    }

    /// The per-frame hook (S53, S54). With no graph it returns at once. With a graph it finds the
    /// external setter calls since the last frame and restores the graph's values (`restore_external`),
    /// then runs each extension's `frame`, and counts the frame. Every extension runs; the first
    /// refusal is returned.
    pub fn frame(&mut self, h: &mut FeatureHandles<'_>) -> Result<(), RaaError> {
        if self.plan.is_none() {
            return Ok(());
        }
        let restored = self.restore_external(h);
        let mut ran: Result<(), RaaError> = Ok(());
        for extension in &mut self.extensions {
            ran = ran.and(extension.frame(h));
        }
        self.frame = Some(self.frame.map_or(0, |index| index + 1));
        restored.and(ran)
    }

    /// One statistics entry per plan node, in plan order, then each extension's entries. `ran` is
    /// whether the node's pipeline entry is enabled in the applied configuration: `output.display`
    /// follows `tonemap`, a contributor follows `forward`. `resources` is empty until the resource
    /// table lands (M3). Empty without a graph.
    pub fn stats(&self) -> Vec<NodeStats> {
        let Some(plan) = self.plan.as_ref() else {
            return Vec::new();
        };
        let mut out: Vec<NodeStats> = plan
            .nodes
            .iter()
            .map(|node| NodeStats {
                node: node.id.clone(),
                type_id: node.type_id.clone(),
                ran: self.node_ran(&node.type_id),
                resources: Vec::new(),
                history_valid: None,
            })
            .collect();
        for extension in &self.extensions {
            extension.stats(&mut out);
        }
        out
    }

    /// The no-graph export of the configuration the handles hold: `export_snapshot` of the snapshot
    /// read from `h`, with `h.msaa` (S54).
    pub fn export_current(catalog: &Catalog, h: &FeatureHandles<'_>) -> GraphDoc {
        export_snapshot(catalog, &snapshot(h), h.msaa)
    }

    /// The setters the graph owns while it is active, in S54 order.
    pub fn owned_setters() -> &'static [&'static str] {
        &[
            "setPassEnabled",
            "setPassOrder",
            "setToneMapping",
            "setShadows",
            "setAo",
            "setContactShadows",
            "setBloom",
            "setTransparency",
            "setSsr",
            "setDof",
            "setFog",
            "setSunShafts",
            "setGrading",
            "applyEnvironment",
            "setCel",
            "setVirtualGeometry",
            "setAreaLightShading",
            "setAreaLightShadows",
            "setGlobalIllumination",
            "setGiQuality",
            "setSphericalGi",
        ]
    }

    /// Whether the pipeline entry of a node of the plan ran in the applied configuration.
    fn node_ran(&self, type_id: &str) -> bool {
        let Some(spec) = self.catalog.node(type_id) else {
            return false;
        };
        let entry = match type_id {
            "output.display" => LAST_CHAIN_ENTRY,
            _ if spec.contributor => HOST_ENTRY,
            _ => match spec.pass {
                Some(entry) => entry,
                None => return false,
            },
        };
        self.applied
            .as_ref()
            .is_some_and(|applied| applied.passes.iter().any(|(name, on)| name == entry && *on))
    }

    /// Finds the external setter calls since the last frame and undoes them for the graph (S54): each
    /// difference is recorded in the no-graph snapshot, the graph's values are applied again, and the
    /// applied state is captured anew. Nothing is allocated unless a difference is found; the order
    /// read through `Pipeline::passes` is the one allocation of the check.
    fn restore_external(&mut self, h: &mut FeatureHandles<'_>) -> Result<(), RaaError> {
        let (Some(plan), Some(applied), Some(no_graph)) = (
            self.plan.as_ref(),
            self.applied.as_ref(),
            self.no_graph.as_mut(),
        ) else {
            return Ok(());
        };
        let pipeline_moved = pipeline_moved(applied, self.applied_order_generation, h);
        self.scratch.clear();
        self.applied_descs.changed(h, &mut self.scratch);
        if !pipeline_moved && self.scratch.is_empty() {
            return Ok(());
        }
        if pipeline_moved {
            let observed = h.pipeline.borrow().passes();
            adopt_pipeline(no_graph, applied, &observed);
            reapply_pipeline(&self.catalog, plan, h)?;
            for extension in &mut self.extensions {
                extension.reorder(plan, h)?;
            }
        }
        let inherit_vg = !graph_owns_virtual_geometry(plan);
        for feature in &self.scratch {
            let applied_values = values_for(applied, feature);
            if let Some(observed) = read_values(h, feature) {
                adopt_values(no_graph, feature, applied_values, &observed);
            }
            if let Some(target) = applied_values
                && !(inherit_vg && *feature == FeatureId::Vg)
            {
                write_values(h, feature, target)?;
            }
        }
        self.applied = Some(snapshot(h));
        self.applied_descs = AppliedDescs::capture(h);
        self.applied_order_generation = h.pipeline.borrow().order_generation();
        Ok(())
    }

    fn activate(
        &mut self,
        plan: GraphPlan,
        doc: Option<GraphDoc>,
        h: &mut FeatureHandles<'_>,
    ) -> SetResult {
        let before = snapshot(h);
        let no_graph = self.no_graph.clone().unwrap_or_else(|| before.clone());
        let targets = match settings_targets(&plan, &self.catalog, &no_graph) {
            Ok(targets) => targets,
            Err(issues) => {
                return SetResult {
                    ok: false,
                    issues,
                    rebuilt: 0,
                };
            }
        };
        let mut progress = Progress::default();
        let outcome = apply(
            &self.catalog,
            &mut self.extensions,
            self.plan.as_ref(),
            &plan,
            &targets,
            h,
            &mut progress,
        );
        match outcome {
            Ok(rebuilt) => {
                self.plan = Some(plan);
                self.doc = doc;
                self.no_graph = Some(no_graph);
                self.applied = Some(snapshot(h));
                self.applied_descs = AppliedDescs::capture(h);
                self.applied_order_generation = h.pipeline.borrow().order_generation();
                h.pipeline
                    .borrow_mut()
                    .set_display_readable(plan_reads_display(self.plan.as_ref()));
                self.frame = None;
                SetResult {
                    ok: true,
                    issues: Vec::new(),
                    rebuilt,
                }
            }
            Err(issues) => {
                rollback(
                    &mut self.extensions,
                    self.plan.as_ref(),
                    &before,
                    &progress,
                    h,
                );
                SetResult {
                    ok: false,
                    issues,
                    rebuilt: 0,
                }
            }
        }
    }
}

/// The settings `plan` writes, computed against the configuration `current`: the pipeline (the
/// enabled entries and the order, by `pipeline_instructions`), then every feature that `current`
/// holds, merged with the values its plan nodes write and its presence keys. A feature without
/// values in `current` gets no entry. A value the feature cannot hold is a `graph.param_range`
/// issue on the node that writes it.
pub fn settings_targets(
    plan: &GraphPlan,
    catalog: &Catalog,
    current: &SettingsSnapshot,
) -> Result<SettingsSnapshot, Vec<Issue>> {
    let installed: Vec<PassInfo> = current
        .passes
        .iter()
        .map(|(name, enabled)| PassInfo {
            name: name.clone(),
            role: if current.order.contains(name) {
                PassRole::Pass
            } else {
                PassRole::Contributor
            },
            enabled: *enabled,
            order: default_order(catalog, name),
            // `pipeline_instructions` compares orders only; `required` is not read.
            required: false,
        })
        .collect();
    let pipeline = pipeline_with_virtual_geometry(
        plan,
        catalog,
        &installed,
        virtual_geometry_enabled(current),
    );

    if let Some(node) = plan
        .nodes
        .iter()
        .find(|node| node.type_id == SPHERICAL_TYPE)
        && values_for(current, &SPHERICAL).is_none()
    {
        return Err(vec![Issue {
            code: "graph.unavailable",
            severity: Severity::Error,
            stage: Stage::Apply,
            node: Some(node.id.clone()),
            port: None,
            param: None,
            link: None,
            message: "The spherical GI feature is unavailable in this engine.".into(),
            fix: "Install the spherical GI feature before applying this graph.".into(),
        }]);
    }
    let mut values = Vec::with_capacity(current.values.len());
    for (feature, base) in &current.values {
        let mut changes = plan_settings(catalog, plan, feature);
        changes.extend(presence(plan, feature));
        match merge_values(feature, base, &changes) {
            Ok(merged) => values.push((feature.clone(), merged)),
            Err((key, message)) => {
                return Err(vec![range_issue(catalog, plan, feature, &key, &message)]);
            }
        }
    }
    Ok(SettingsSnapshot {
        passes: pipeline.enable,
        order: pipeline.order,
        values,
    })
}

fn virtual_geometry_enabled(settings: &SettingsSnapshot) -> bool {
    values_for(settings, &FeatureId::Vg).is_some_and(|values| {
        values
            .iter()
            .any(|(key, value)| key == "enabled" && *value == ParamValue::Bool(true))
    })
}

/// The features whose applied values differ from the observed ones, in S53 rank order. The pipeline
/// comes first when its passes or order differ.
pub fn external_changes(applied: &SettingsSnapshot, observed: &SettingsSnapshot) -> Vec<FeatureId> {
    let mut changed = Vec::new();
    if applied.passes != observed.passes || applied.order != observed.order {
        changed.push(FeatureId::Pipeline);
    }
    for feature in &RANK {
        if values_for(applied, feature) != values_for(observed, feature) {
            changed.push(feature.clone());
        }
    }
    changed
}

/// The live pipeline and feature values, in the form `settings_targets` and `external_changes`
/// read. A feature with no snapshot keys, or whose handle is absent, has no entry.
fn snapshot(h: &FeatureHandles<'_>) -> SettingsSnapshot {
    let installed = h.pipeline.borrow().passes();
    let passes = installed
        .iter()
        .filter(|info| !owned(&info.name))
        .map(|info| (info.name.clone(), info.enabled))
        .collect();
    let order = installed
        .iter()
        .filter(|info| info.role == PassRole::Pass && !owned(&info.name))
        .map(|info| info.name.clone())
        .collect();
    let values = RANK
        .iter()
        .filter_map(|feature| {
            let values = read_values(h, feature)?;
            (!values.is_empty()).then(|| (feature.clone(), values))
        })
        .collect();
    SettingsSnapshot {
        passes,
        order,
        values,
    }
}

/// What one activation has changed so far. A refusal undoes exactly these.
#[derive(Default)]
struct Progress {
    pipeline: bool,
    features: Vec<FeatureId>,
    extensions: usize,
}

/// Applies `plan` over `targets` in S53 rank order: the pipeline, the features whose live value
/// differs from their target, then the extensions. Returns the number of changed parameters whose
/// rebuild is not `none`. On `Err` the caller undoes what `progress` records.
fn apply(
    catalog: &Catalog,
    extensions: &mut [Box<dyn ExecExtension>],
    previous: Option<&GraphPlan>,
    plan: &GraphPlan,
    targets: &SettingsSnapshot,
    h: &mut FeatureHandles<'_>,
    progress: &mut Progress,
) -> Result<u32, Vec<Issue>> {
    apply_pipeline(
        catalog,
        plan,
        h,
        progress,
        virtual_geometry_enabled(targets),
    )?;
    let rebuilt = apply_features(catalog, plan, targets, h, progress)?;
    for extension in extensions.iter_mut() {
        extension.apply(plan, previous, h)?;
        progress.extensions += 1;
    }
    Ok(rebuilt)
}

/// `pipeline_instructions` of the plan against the installed passes: each entry's enable flag,
/// then the order.
fn apply_pipeline(
    catalog: &Catalog,
    plan: &GraphPlan,
    h: &FeatureHandles<'_>,
    progress: &mut Progress,
    virtual_geometry: bool,
) -> Result<(), Vec<Issue>> {
    let installed = h.pipeline.borrow().passes();
    let instructions = pipeline_with_virtual_geometry(plan, catalog, &installed, virtual_geometry);
    progress.pipeline = true;
    let mut pipeline = h.pipeline.borrow_mut();
    for (name, enabled) in &instructions.enable {
        pipeline
            .set_enabled(name, *enabled)
            .map_err(|error| vec![pipeline_issue(&error)])?;
    }
    set_order(&mut pipeline, &instructions.order).map_err(|error| vec![pipeline_issue(&error)])
}

/// Writes every feature whose live value differs from its target, in rank order, and counts the
/// changed parameters whose catalog rebuild is not `none`.
fn apply_features(
    catalog: &Catalog,
    plan: &GraphPlan,
    targets: &SettingsSnapshot,
    h: &mut FeatureHandles<'_>,
    progress: &mut Progress,
) -> Result<u32, Vec<Issue>> {
    let mut rebuilt = 0;
    for feature in &RANK {
        let (Some(target), Some(current)) = (values_for(targets, feature), read_values(h, feature))
        else {
            continue;
        };
        if current == target {
            continue;
        }
        rebuilt += rebuilds(catalog, feature, &current, target);
        write_values(h, feature, target)
            .map_err(|error| vec![feature_issue(catalog, plan, feature, &error)])?;
        progress.features.push(feature.clone());
    }
    Ok(rebuilt)
}

/// Undoes an activation that stopped at `progress`: the extensions it applied (the previous plan
/// is applied again, or the extension is cleared when there was none), the features it wrote (back
/// to `before`, in reverse rank order) and the pipeline. `before` was live when the call began, so
/// these restorations are not refused and their results are ignored.
fn rollback(
    extensions: &mut [Box<dyn ExecExtension>],
    previous: Option<&GraphPlan>,
    before: &SettingsSnapshot,
    progress: &Progress,
    h: &mut FeatureHandles<'_>,
) {
    for extension in extensions.iter_mut().take(progress.extensions).rev() {
        match previous {
            Some(plan) => {
                let _ = extension.apply(plan, None, h);
            }
            None => extension.clear(h),
        }
    }
    for feature in progress.features.iter().rev() {
        if let Some(values) = values_for(before, feature) {
            let _ = write_values(h, feature, values);
        }
    }
    if progress.pipeline {
        let _ = restore_pipeline(before, h);
    }
}

/// Restores `snapshot`: the pipeline, then each feature whose live value differs, in rank order.
fn restore(snapshot: &SettingsSnapshot, h: &mut FeatureHandles<'_>) -> Result<(), RaaError> {
    restore_pipeline(snapshot, h)?;
    for feature in &RANK {
        let Some(values) = values_for(snapshot, feature) else {
            continue;
        };
        if read_values(h, feature).as_deref() != Some(values) {
            write_values(h, feature, values)?;
        }
    }
    Ok(())
}

fn restore_pipeline(snapshot: &SettingsSnapshot, h: &FeatureHandles<'_>) -> Result<(), RaaError> {
    let mut pipeline = h.pipeline.borrow_mut();
    for (name, enabled) in &snapshot.passes {
        pipeline.set_enabled(name, *enabled)?;
    }
    set_order(&mut pipeline, &snapshot.order)
}

/// Sets the pass order `order` with every owned pass-role entry of the live pipeline put back.
fn set_order(pipeline: &mut Pipeline, order: &[String]) -> Result<(), RaaError> {
    let live: Vec<(String, bool)> = pipeline
        .passes()
        .into_iter()
        .map(|info| (info.name, info.role == PassRole::Pass))
        .collect();
    let names = merge_owned(order, &live);
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    pipeline.set_order(&names)
}

/// Whether the pipeline entry `name` is owned by the extension that installed it: a name with `#`
/// (`view#<nodeId>`). The executor leaves owned entries out of its pipeline state.
fn owned(name: &str) -> bool {
    name.contains('#')
}

/// The pass order to set: `order`, with each owned pass-role entry of `live` (name, pass role, in
/// live order) inserted right after the nearest pass-role entry before it, or at the front when none
/// precedes it. Other entries are never merged, and without an owned entry the result is `order`.
fn merge_owned(order: &[String], live: &[(String, bool)]) -> Vec<String> {
    let mut merged = order.to_vec();
    let mut previous: Option<&str> = None;
    for (name, _) in live.iter().filter(|(_, is_pass)| *is_pass) {
        if owned(name) {
            let at = previous
                .and_then(|before| merged.iter().position(|known| known == before))
                .map_or(0, |index| index + 1);
            merged.insert(at, name.clone());
        }
        previous = Some(name.as_str());
    }
    merged
}

/// Whether the live pipeline differs from `applied`: an applied entry's enabled flag, read through
/// `Pipeline::entry_enabled` without allocation, or the order of the unowned pass-role entries, read
/// through `Pipeline::passes`.
fn pipeline_moved(applied: &SettingsSnapshot, order_generation: u64, h: &FeatureHandles<'_>) -> bool {
    let pipeline = h.pipeline.borrow();
    if applied
        .passes
        .iter()
        .any(|(name, on)| pipeline.entry_enabled(name) != Some(*on))
    {
        return true;
    }
    // Entries are added, removed or reordered only through calls that move the generation: until
    // it moves, the pass list is neither copied nor compared.
    pipeline.order_generation() != order_generation && pipeline_differs(applied, &pipeline.passes())
}

/// Whether a node other than `output.display` reads the colour of a `pass.tonemap` or `pass.overlay`
/// node (a `view.*` node): only then does the pipeline keep the display target those passes write.
fn plan_reads_display(plan: Option<&GraphPlan>) -> bool {
    let Some(plan) = plan else {
        return true;
    };
    let is_display = |id: &str| {
        plan.nodes
            .iter()
            .any(|n| n.id == id && matches!(n.type_id.as_str(), "pass.tonemap" | "pass.overlay"))
    };
    plan.nodes
        .iter()
        .filter(|n| n.type_id != "output.display")
        .flat_map(|n| n.inputs.iter())
        .any(|input| match &input.source {
            InputSource::Link { node, port, .. } | InputSource::Default { node, port } => {
                port == "color" && is_display(node)
            }
            InputSource::Absent => false,
        })
}

/// Compares the graph's base entries with the observed pipeline, ignoring extension entries.
pub fn pipeline_differs(applied: &SettingsSnapshot, observed: &[PassInfo]) -> bool {
    applied
        .passes
        .iter()
        .filter(|(name, _)| !owned(name))
        .any(|(name, enabled)| {
            observed
                .iter()
                .find(|info| info.name == *name)
                .is_none_or(|info| info.enabled != *enabled)
        })
        || observed
            .iter()
            .filter(|info| !owned(&info.name))
            .any(|info| !applied.passes.iter().any(|(name, _)| *name == info.name))
        || !observed
            .iter()
            .filter(|info| info.role == PassRole::Pass && !owned(&info.name))
            .map(|info| info.name.as_str())
            .eq(applied
                .order
                .iter()
                .filter(|name| !owned(name))
                .map(String::as_str))
}

/// Applies the pipeline instructions of `plan` again: each entry's enabled flag, then the order.
fn reapply_pipeline(
    catalog: &Catalog,
    plan: &GraphPlan,
    h: &FeatureHandles<'_>,
) -> Result<(), RaaError> {
    let installed = h.pipeline.borrow().passes();
    let enabled = h.handles.vg.as_ref().is_some_and(|vg| vg.get().enabled);
    let instructions = pipeline_with_virtual_geometry(plan, catalog, &installed, enabled);
    let mut pipeline = h.pipeline.borrow_mut();
    for (name, enabled) in &instructions.enable {
        pipeline.set_enabled(name, *enabled)?;
    }
    set_order(&mut pipeline, &instructions.order)
}

/// Records the observed pipeline in the no-graph snapshot: each entry whose enabled flag differs from
/// the applied one takes the observed flag, and an order that differs from the applied one is taken
/// whole.
pub fn adopt_pipeline(
    no_graph: &mut SettingsSnapshot,
    applied: &SettingsSnapshot,
    observed: &[PassInfo],
) {
    no_graph.passes.retain(|(name, _)| !owned(name));
    no_graph.order.retain(|name| !owned(name));
    for info in observed.iter().filter(|info| !owned(&info.name)) {
        let applied_on = applied
            .passes
            .iter()
            .find(|(name, _)| *name == info.name)
            .map(|(_, on)| *on);
        if applied_on != Some(info.enabled) {
            set_entry(&mut no_graph.passes, &info.name, info.enabled);
        }
    }
    let order: Vec<String> = observed
        .iter()
        .filter(|info| info.role == PassRole::Pass && !owned(&info.name))
        .map(|info| info.name.clone())
        .collect();
    if order != applied.order {
        no_graph.order = order;
    }
}

/// Records the observed values of `feature` that differ from the applied ones in the no-graph
/// snapshot. A feature the no-graph snapshot lacks takes all of its observed values.
fn adopt_values(
    no_graph: &mut SettingsSnapshot,
    feature: &FeatureId,
    applied: Option<&[(String, ParamValue)]>,
    observed: &[(String, ParamValue)],
) {
    let Some(index) = no_graph
        .values
        .iter()
        .position(|(candidate, _)| candidate == feature)
    else {
        no_graph.values.push((feature.clone(), observed.to_vec()));
        return;
    };
    for (key, value) in observed {
        let applied_value = applied
            .and_then(|values| values.iter().find(|(name, _)| name == key))
            .map(|(_, value)| value);
        if applied_value != Some(value) {
            set_entry(&mut no_graph.values[index].1, key, value.clone());
        }
    }
}

/// Sets `key` to `value` in `entries`, appending the entry when the list does not hold it.
fn set_entry<T>(entries: &mut Vec<(String, T)>, key: &str, value: T) {
    match entries.iter_mut().find(|(name, _)| name == key) {
        Some(entry) => entry.1 = value,
        None => entries.push((key.to_owned(), value)),
    }
}

/// The changed values of `feature` whose catalog setting needs a rebuild.
fn rebuilds(
    catalog: &Catalog,
    feature: &FeatureId,
    current: &[(String, ParamValue)],
    target: &[(String, ParamValue)],
) -> u32 {
    let changed = target.iter().filter(|(key, value)| {
        current
            .iter()
            .any(|(name, live)| name == key && live != value)
            && needs_rebuild(catalog, feature, key)
    });
    u32::try_from(changed.count()).unwrap_or(u32::MAX)
}

/// Whether the catalog marks the setting `key` of `feature` as needing a rebuild.
fn needs_rebuild(catalog: &Catalog, feature: &FeatureId, key: &str) -> bool {
    catalog
        .nodes
        .iter()
        .flat_map(|spec| spec.params.iter())
        .find(|param| setting_key(param, feature) == Some(key))
        .is_some_and(|param| !matches!(param.rebuild, Rebuild::None))
}

/// The setting key that `param` writes to `feature`, `None` when it writes elsewhere.
fn setting_key(param: &ParamSpec, feature: &FeatureId) -> Option<&'static str> {
    match &param.binding {
        ParamBinding::Setting {
            feature: bound,
            key,
        } if bound == feature => Some(*key),
        _ => None,
    }
}

/// The values the plan's nodes write to `feature`, under the setting key each parameter binds. A
/// multi-instance node after the first node of its type writes nothing: the installed pass takes
/// the first node's values, and the other nodes write their own instance handles (rule E5).
fn plan_settings(
    catalog: &Catalog,
    plan: &GraphPlan,
    feature: &FeatureId,
) -> Vec<(String, ParamValue)> {
    let mut changes = Vec::new();
    for (index, node) in plan.nodes.iter().enumerate() {
        let Some(spec) = catalog.node(&node.type_id) else {
            continue;
        };
        if spec.multi_instance
            && plan.nodes[..index]
                .iter()
                .any(|earlier| earlier.type_id == node.type_id)
        {
            continue;
        }
        for (key, value) in &node.params {
            let bound = spec
                .params
                .iter()
                .find(|param| param.key == key.as_str())
                .and_then(|param| setting_key(param, feature));
            if let Some(setting) = bound {
                changes.push((setting.to_owned(), value.clone()));
            }
        }
    }
    changes
}

/// The presence keys of `feature`. Authored VG nodes opt in; other graphs inherit VG. Gi
/// `enabled` when it has `pass.gi.volume`, Gi `screenSpace` when it has `pass.gi.ssgi`.
fn presence(plan: &GraphPlan, feature: &FeatureId) -> Vec<(String, ParamValue)> {
    let has = |type_id: &str| plan.nodes.iter().any(|node| node.type_id == type_id);
    let flag = |name: &str, on: bool| (name.to_owned(), ParamValue::Bool(on));
    match feature {
        FeatureId::Vg if graph_owns_virtual_geometry(plan) => vec![flag("enabled", true)],
        FeatureId::Gi => vec![
            flag("enabled", has("pass.gi.volume")),
            flag("screenSpace", has("pass.gi.ssgi")),
        ],
        &SPHERICAL => vec![flag("enabled", has(SPHERICAL_TYPE))],
        _ => Vec::new(),
    }
}

/// The catalog default order of the pass `entry`; an entry no node type drives sorts last.
fn default_order(catalog: &Catalog, entry: &str) -> u32 {
    catalog
        .nodes
        .iter()
        .find(|spec| spec.pass == Some(entry))
        .map_or(u32::MAX, |spec| spec.default_order)
}

/// The values of `feature` in `snapshot`, `None` when it has none.
fn values_for<'a>(
    snapshot: &'a SettingsSnapshot,
    feature: &FeatureId,
) -> Option<&'a [(String, ParamValue)]> {
    snapshot
        .values
        .iter()
        .find(|(candidate, _)| candidate == feature)
        .map(|(_, values)| values.as_slice())
}

/// The first plan node whose type binds `feature` (to `key`, when one is given).
fn bound_node(
    catalog: &Catalog,
    plan: &GraphPlan,
    feature: &FeatureId,
    key: Option<&str>,
) -> Option<String> {
    plan.nodes
        .iter()
        .find(|node| {
            catalog.node(&node.type_id).is_some_and(|spec| {
                spec.params
                    .iter()
                    .filter_map(|param| setting_key(param, feature))
                    .any(|bound| key.is_none_or(|key| key == bound))
            })
        })
        .map(|node| node.id.clone())
}

fn range_issue(
    catalog: &Catalog,
    plan: &GraphPlan,
    feature: &FeatureId,
    key: &str,
    message: &str,
) -> Issue {
    Issue {
        code: "graph.param_range",
        severity: Severity::Error,
        stage: Stage::Apply,
        node: bound_node(catalog, plan, feature, Some(key))
            .or_else(|| bound_node(catalog, plan, feature, None)),
        port: None,
        param: Some(key.to_owned()),
        link: None,
        message: format!("{key}: {message}"),
        fix: format!("set {key} to a value the feature accepts"),
    }
}

/// A feature refused a value during apply. The refused key is named when the feature's message
/// starts with a snapshot key of `feature`.
fn feature_issue(
    catalog: &Catalog,
    plan: &GraphPlan,
    feature: &FeatureId,
    error: &RaaError,
) -> Issue {
    let message = error.message();
    let param = message
        .split_once(": ")
        .map(|(key, _)| key)
        .filter(|key| snapshot_keys(feature).iter().any(|known| known == key));
    Issue {
        code: "graph.apply",
        severity: Severity::Error,
        stage: Stage::Apply,
        node: bound_node(catalog, plan, feature, None),
        port: None,
        param: param.map(str::to_owned),
        link: None,
        message: format!("{}: {message}", error.code()),
        fix: "set the refused value to one the feature accepts".to_owned(),
    }
}

fn pipeline_issue(error: &RaaError) -> Issue {
    Issue {
        code: "graph.order",
        severity: Severity::Error,
        stage: Stage::Apply,
        node: None,
        port: None,
        param: None,
        link: None,
        message: error.message().to_owned(),
        fix: "reorder the graph's passes, or remove the pass the pipeline refused".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::{NeuralRoutes, merge_owned};
    use crate::format::{parse_graph, write_graph};
    use crate::plan::plan_from_json;

    #[test]
    fn neural_starter_passes_runtime_route_admission() {
        let source = include_str!("../../graphs/real-frame-model.json");
        let plan = plan_from_json(source).expect("neural starter plans");
        let routes = NeuralRoutes::from_plan(&plan).expect("runtime accepts the starter's routes");
        assert!(routes.active());

        // The alternate feed has the right image type, but is upstream of the HDR endpoint.
        let mut doc = parse_graph(source).unwrap();
        let generation = doc
            .links
            .iter_mut()
            .find(|link| link.to.node == "frameGeneration" && link.to.port == "color")
            .unwrap();
        generation.from.node = "forward".into();
        let plan = plan_from_json(&write_graph(&doc)).expect("alternate feed still plans");
        let error = NeuralRoutes::from_plan(&plan)
            .err()
            .expect("runtime rejects a mismatched endpoint");
        assert_eq!(error.code(), "graph.neural_route");
        assert!(
            error
                .message()
                .contains("same real HDR endpoint as tonemap")
        );
    }

    /// Owned names of string slices.
    fn names(entries: &[&str]) -> Vec<String> {
        entries.iter().map(|name| (*name).to_owned()).collect()
    }

    /// Owned `(name, is_pass)` pairs of string slices.
    fn pairs(entries: &[(&str, bool)]) -> Vec<(String, bool)> {
        entries
            .iter()
            .map(|(name, is_pass)| ((*name).to_owned(), *is_pass))
            .collect()
    }

    #[test]
    fn an_owned_pass_keeps_its_place_after_its_neighbour() {
        let order = names(&["prepass", "forward", "tonemap"]);
        let live = pairs(&[
            ("prepass", true),
            ("view#a", true),
            ("forward", true),
            ("tonemap", true),
            ("view#b", true),
        ]);
        assert_eq!(
            merge_owned(&order, &live),
            names(&["prepass", "view#a", "forward", "tonemap", "view#b"])
        );
    }

    #[test]
    fn an_owned_pass_first_in_live_order_comes_first() {
        let order = names(&["prepass", "forward"]);
        let live = pairs(&[("view#a", true), ("prepass", true), ("forward", true)]);
        assert_eq!(
            merge_owned(&order, &live),
            names(&["view#a", "prepass", "forward"])
        );
    }

    #[test]
    fn an_owned_contributor_is_not_merged() {
        let order = names(&["prepass", "forward", "tonemap"]);
        let live = pairs(&[
            ("prepass", true),
            ("forward", true),
            ("view#c", false),
            ("tonemap", true),
        ]);
        assert_eq!(merge_owned(&order, &live), order);
    }
}
