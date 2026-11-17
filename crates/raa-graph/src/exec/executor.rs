//! The graph executor (S53, S54). It plans a document, applies the plan to the live pipeline and
//! features in rank order, and restores the configuration that was live before the first graph when
//! a change is refused or the graph is cleared. `settings_targets` and `external_changes` are the
//! device-free helpers that the executor and its tests share.

use raa_core::RaaError;
use raa_pipeline::{PassInfo, PassRole, Pipeline};

use super::{
    AppliedDescs, ExecExtension, FeatureHandles, SetResult, SettingsSnapshot, merge_values,
    pipeline_instructions, read_values, snapshot_keys, write_values,
};
use crate::catalog::{Catalog, FeatureId, ParamBinding, ParamSpec, ParamValue, Rebuild};
use crate::error::{Issue, Severity, Stage};
use crate::format::GraphDoc;
use crate::plan::{GraphPlan, plan_from_doc};
use crate::validate::{ValidateCtx, canonical_doc};

/// The features in S53 rank order. The pipeline (rank 0) is applied apart from them.
const RANK: [FeatureId; 18] = [
    FeatureId::Tonemap,
    FeatureId::Shadow,
    FeatureId::Vg,
    FeatureId::GiQuality,
    FeatureId::Gi,
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
    // The per-frame hook of 'raa-graph exec 5/5: frame ownership, stats and export' reads these
    // three fields; until it lands they are written and never read.
    #[allow(dead_code)]
    applied: Option<SettingsSnapshot>,
    #[allow(dead_code)]
    applied_descs: AppliedDescs,
    #[allow(dead_code)]
    frame: u64,
    last_issues: Vec<Issue>,
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
            frame: 0,
            last_issues: Vec::new(),
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
                let mut result = self.set_plan(plan, Some(canonical), h);
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
        let result = self.activate(plan, doc, h);
        self.last_issues.clone_from(&result.issues);
        result
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
        self.no_graph = None;
        self.doc = None;
        self.plan = None;
        self.applied = None;
        self.applied_descs = AppliedDescs::default();
        self.frame = 0;
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
                self.frame = 0;
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
    let pipeline = pipeline_instructions(plan, catalog, &installed);

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
        .map(|info| (info.name.clone(), info.enabled))
        .collect();
    let order = installed
        .iter()
        .filter(|info| info.role == PassRole::Pass)
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
    apply_pipeline(catalog, plan, h, progress)?;
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
) -> Result<(), Vec<Issue>> {
    let installed = h.pipeline.borrow().passes();
    let instructions = pipeline_instructions(plan, catalog, &installed);
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

fn set_order(pipeline: &mut Pipeline, order: &[String]) -> Result<(), RaaError> {
    let names: Vec<&str> = order.iter().map(String::as_str).collect();
    pipeline.set_order(&names)
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

/// The values the plan's nodes write to `feature`, under the setting key each parameter binds.
fn plan_settings(
    catalog: &Catalog,
    plan: &GraphPlan,
    feature: &FeatureId,
) -> Vec<(String, ParamValue)> {
    let mut changes = Vec::new();
    for node in &plan.nodes {
        let Some(spec) = catalog.node(&node.type_id) else {
            continue;
        };
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

/// The presence keys of `feature`. Vg `enabled` holds when the plan has a `pass.vg.*` node; Gi
/// `enabled` when it has `pass.gi.volume`, Gi `screenSpace` when it has `pass.gi.ssgi`.
fn presence(plan: &GraphPlan, feature: &FeatureId) -> Vec<(String, ParamValue)> {
    let has = |type_id: &str| plan.nodes.iter().any(|node| node.type_id == type_id);
    let flag = |name: &str, on: bool| (name.to_owned(), ParamValue::Bool(on));
    match feature {
        FeatureId::Vg => vec![flag(
            "enabled",
            plan.nodes
                .iter()
                .any(|node| node.type_id.starts_with("pass.vg.")),
        )],
        FeatureId::Gi => vec![
            flag("enabled", has("pass.gi.volume")),
            flag("screenSpace", has("pass.gi.ssgi")),
        ],
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
