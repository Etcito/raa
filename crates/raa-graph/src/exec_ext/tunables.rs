//! Executor extension `tunables` (S56): writes the `ParamBinding::Tunable` parameters of a plan
//! through each feature handle's `set_tunable`. The extension keeps `applied`, the non-default
//! values it has set, so a plan that stops writing a key returns it to its catalog default. The
//! executor's `rollback` re-applies the previous plan (or calls `clear`) when a later extension
//! refuses, so `applied` and the handles return to the previous state.

use raa_core::RaaError;

use crate::catalog::{Catalog, FeatureId, ParamBinding, ParamValue};
use crate::error::{Issue, Severity, Stage};
use crate::exec::{ExecExtension, FeatureHandles, HandleSet};
use crate::plan::GraphPlan;

/// The tunables extension, discovered by build.rs (stem `tunables`). The trait passes no catalog,
/// so the extension builds its own: the defaults come from the `Tunable` bindings of its nodes.
pub fn extension() -> Box<dyn ExecExtension> {
    Box::new(TunablesExtension {
        catalog: Catalog::build(),
        applied: Vec::new(),
    })
}

/// State: `applied`, the non-default values this extension has set, by feature and catalog key.
struct TunablesExtension {
    catalog: Catalog,
    applied: Vec<(FeatureId, &'static str, f64)>,
}

/// Every `Tunable` parameter the plan writes: for each plan node in plan order, each catalog
/// parameter of its type in catalog key order, as `(feature, key, value, node id, param key)`.
/// `ParamValue::Number` and `ParamValue::Int` give their value; any other kind is skipped.
pub(crate) fn tunable_targets(
    plan: &GraphPlan,
    catalog: &Catalog,
) -> Vec<(FeatureId, &'static str, f64, String, &'static str)> {
    let mut targets = Vec::new();
    for node in &plan.nodes {
        let Some(spec) = catalog.node(&node.type_id) else {
            continue;
        };
        for param in &spec.params {
            let ParamBinding::Tunable { feature, key } = &param.binding else {
                continue;
            };
            let value = node
                .params
                .iter()
                .find(|(name, _)| name.as_str() == param.key)
                .and_then(|(_, value)| as_f64(value));
            if let Some(value) = value {
                targets.push((feature.clone(), *key, value, node.id.clone(), param.key));
            }
        }
    }
    targets
}

/// The applied keys no target writes any more, each with the catalog default of its parameter.
/// The caller sets each one back to that default.
pub(crate) fn tunable_resets(
    applied: &[(FeatureId, &'static str, f64)],
    targets: &[(FeatureId, &'static str, f64, String, &'static str)],
    catalog: &Catalog,
) -> Vec<(FeatureId, &'static str, f64)> {
    applied
        .iter()
        .filter(|(feature, key, _)| !targets.iter().any(|t| t.0 == *feature && t.1 == *key))
        .filter_map(|(feature, key, _)| {
            Some((feature.clone(), *key, default_of(catalog, feature, key)?))
        })
        .collect()
}

/// The catalog default of the first parameter bound to `feature` and `key`, when it is a number.
fn default_of(catalog: &Catalog, feature: &FeatureId, key: &str) -> Option<f64> {
    catalog
        .nodes
        .iter()
        .flat_map(|node| &node.params)
        .find_map(|param| match &param.binding {
            ParamBinding::Tunable {
                feature: bound,
                key: name,
            } if bound == feature && *name == key => as_f64(&param.default),
            _ => None,
        })
}

/// The value of a number or integer parameter as f64; the other kinds carry no tunable value.
fn as_f64(value: &ParamValue) -> Option<f64> {
    match value {
        ParamValue::Number(x) => Some(*x),
        ParamValue::Int(i) => Some(*i as f64),
        _ => None,
    }
}

/// Calls `set_tunable` on the handle of `feature`. A feature whose handle is not installed (its
/// node was refused by `graph.unavailable`) and a feature without a tunable handle are skipped.
fn write(handles: &HandleSet, feature: &FeatureId, key: &str, value: f64) -> Result<(), RaaError> {
    let written = match feature {
        FeatureId::Taa => handles.taa.as_ref().map(|h| h.set_tunable(key, value)),
        FeatureId::Gtao => handles.gtao.as_ref().map(|h| h.set_tunable(key, value)),
        FeatureId::Contact => handles.contact.as_ref().map(|h| h.set_tunable(key, value)),
        FeatureId::Ssr => handles.ssr.as_ref().map(|h| h.set_tunable(key, value)),
        FeatureId::Dof => handles.dof.as_ref().map(|h| h.set_tunable(key, value)),
        FeatureId::Shafts => handles.shafts.as_ref().map(|h| h.set_tunable(key, value)),
        FeatureId::Gi => handles.gi.as_ref().map(|h| h.set_tunable(key, value)),
        FeatureId::Sky => handles.sky.as_ref().map(|h| h.set_tunable(key, value)),
        _ => None,
    };
    written.unwrap_or(Ok(()))
}

/// Sets each recorded `(feature, key, value)` back, newest first. Errors are ignored: each value
/// was live before the call that wrote over it.
fn restore(handles: &HandleSet, undo: &[(FeatureId, &'static str, f64)]) {
    for (feature, key, value) in undo.iter().rev() {
        let _ = write(handles, feature, key, *value);
    }
}

/// The `graph.apply` issue of a refused write: the node and parameter of a plan target, or none
/// for a reset of a key the plan no longer writes.
fn apply_issue(at: Option<(String, &'static str)>, error: &RaaError) -> Issue {
    let (node, param) = match at {
        Some((node, param)) => (Some(node), Some(param.to_string())),
        None => (None, None),
    };
    Issue {
        code: "graph.apply",
        severity: Severity::Error,
        stage: Stage::Apply,
        node,
        port: None,
        param,
        link: None,
        message: error.to_string(),
        fix: "use a value the feature accepts".to_string(),
    }
}

impl ExecExtension for TunablesExtension {
    fn name(&self) -> &'static str {
        "tunables"
    }

    /// Resets the keys the plan no longer writes, then writes every target whose value differs
    /// from the value the handle holds (the applied value, or the default). The first refusal
    /// restores the values this call wrote and leaves `applied` unchanged. On success `applied`
    /// holds the keys whose value differs from the default.
    fn apply(
        &mut self,
        plan: &GraphPlan,
        _previous: Option<&GraphPlan>,
        h: &mut FeatureHandles<'_>,
    ) -> Result<(), Vec<Issue>> {
        let targets = tunable_targets(plan, &self.catalog);
        let resets = tunable_resets(&self.applied, &targets, &self.catalog);
        let steps = resets
            .into_iter()
            .map(|(feature, key, value)| (feature, key, value, None))
            .chain(
                targets
                    .into_iter()
                    .map(|(feature, key, value, node, param)| {
                        (feature, key, value, Some((node, param)))
                    }),
            );
        let mut live = self.applied.clone();
        let mut undo: Vec<(FeatureId, &'static str, f64)> = Vec::new();
        for (feature, key, value, at) in steps {
            let Some(default) = default_of(&self.catalog, &feature, key) else {
                continue;
            };
            let before = live
                .iter()
                .find(|(f, k, _)| *f == feature && *k == key)
                .map_or(default, |(_, _, v)| *v);
            if before == value {
                continue;
            }
            if let Err(error) = write(&*h.handles, &feature, key, value) {
                restore(&*h.handles, &undo);
                return Err(vec![apply_issue(at, &error)]);
            }
            undo.push((feature.clone(), key, before));
            live.retain(|(f, k, _)| !(*f == feature && *k == key));
            if value != default {
                live.push((feature, key, value));
            }
        }
        self.applied = live;
        Ok(())
    }

    /// Sets every applied key back to its catalog default (errors ignored), then clears `applied`.
    fn clear(&mut self, h: &mut FeatureHandles<'_>) {
        for (feature, key, _) in std::mem::take(&mut self.applied) {
            if let Some(default) = default_of(&self.catalog, &feature, key) {
                let _ = write(&*h.handles, &feature, key, default);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{Category, NodeTypeSpec, ParamKind, ParamSpec, Rebuild};
    use crate::plan::{PlanNode, PlanPortRef};

    fn tunable(key: &'static str) -> ParamBinding {
        ParamBinding::Tunable {
            feature: FeatureId::Taa,
            key,
        }
    }

    fn param(
        key: &'static str,
        kind: ParamKind,
        default: ParamValue,
        binding: ParamBinding,
    ) -> ParamSpec {
        ParamSpec {
            key,
            label: key,
            group: "test",
            kind,
            default,
            min: None,
            max: None,
            step: None,
            options: &[],
            unit: "",
            rebuild: Rebuild::None,
            primary: false,
            unstated_range: false,
            binding,
            source: "test",
        }
    }

    /// One node type `tuned`: `a` (Tunable number, default 1), `n` (Tunable int, default 4) and a
    /// Setting `mode`, in catalog key order.
    fn catalog() -> Catalog {
        let params = vec![
            param(
                "a",
                ParamKind::Number,
                ParamValue::Number(1.0),
                tunable("a"),
            ),
            param("n", ParamKind::Int, ParamValue::Int(4), tunable("n")),
            param(
                "mode",
                ParamKind::Enum,
                ParamValue::Enum("x".into()),
                ParamBinding::Setting {
                    feature: FeatureId::Taa,
                    key: "mode",
                },
            ),
        ];
        let tuned = NodeTypeSpec {
            id: "tuned",
            version: 1,
            category: Category::Post,
            label: "Tuned",
            description: "test node",
            pass: None,
            contributor: false,
            default_order: 0,
            multi_instance: false,
            side_effect: false,
            required_in_graph: false,
            inputs: Vec::new(),
            outputs: Vec::new(),
            params,
            budgets: Vec::new(),
            upgrade: |_, _| Ok(()),
            family: "test",
        };
        Catalog {
            nodes: vec![tuned],
            starters: Vec::new(),
            presets: Vec::new(),
            frozen: Vec::new(),
        }
    }

    /// A plan with one `tuned` node `t1` carrying `params`.
    fn plan(params: Vec<(String, ParamValue)>) -> GraphPlan {
        GraphPlan {
            name: "test".into(),
            nodes: vec![PlanNode {
                id: "t1".into(),
                type_id: "tuned".into(),
                type_version: 1,
                order_index: 0,
                params,
                inputs: Vec::new(),
                compute: None,
            }],
            links: Vec::new(),
            resources: Vec::new(),
            histories: Vec::new(),
            output: PlanPortRef {
                node: "out".into(),
                port: "color".into(),
            },
        }
    }

    #[test]
    fn exec_ext_tunable_targets_in_catalog_order_skipping_settings() {
        let plan = plan(vec![
            ("a".into(), ParamValue::Number(2.0)),
            ("mode".into(), ParamValue::Enum("y".into())),
            ("n".into(), ParamValue::Int(4)),
        ]);
        let targets = tunable_targets(&plan, &catalog());
        assert_eq!(
            targets,
            vec![
                (FeatureId::Taa, "a", 2.0, "t1".to_string(), "a"),
                (FeatureId::Taa, "n", 4.0, "t1".to_string(), "n"),
            ]
        );
    }

    #[test]
    fn exec_ext_tunable_resets_return_defaults_of_untargeted_keys() {
        let applied = vec![(FeatureId::Taa, "a", 2.0)];
        assert_eq!(
            tunable_resets(&applied, &[], &catalog()),
            vec![(FeatureId::Taa, "a", 1.0)]
        );
        let kept = plan(vec![("a".into(), ParamValue::Number(2.0))]);
        let targets = tunable_targets(&kept, &catalog());
        assert!(tunable_resets(&applied, &targets, &catalog()).is_empty());
    }

    #[test]
    fn exec_ext_int_value_converts_to_f64() {
        assert_eq!(as_f64(&ParamValue::Int(3)), Some(3.0));
        assert_eq!(as_f64(&ParamValue::Number(0.5)), Some(0.5));
        assert_eq!(as_f64(&ParamValue::Bool(true)), None);
    }
}
