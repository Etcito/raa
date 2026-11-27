//! Family `tunables_lighting`: the promoted tunables of the lighting passes (S56).
//!
//! `extend_types` appends one `ParamBinding::Tunable` parameter per triage row to the node type
//! the row names. The rows are copied from the crate rows of docs/render-graph/tunables.md; the
//! default, range and rebuild class of each come from the crate's `tunables()` spec of that key.

use raa_pipeline::{TunableSpec, find_tunable};

use crate::catalog::{
    FeatureId, NodeFamily, NodeTypeSpec, ParamBinding, ParamKind, ParamSpec, ParamValue, Rebuild,
};

/// The `tunables_lighting` family, discovered by build.rs.
pub fn family() -> Box<dyn NodeFamily> {
    Box::new(TunablesLighting)
}

/// One triage row: `(key, node type, group, label)`.
type MetaRow = (&'static str, &'static str, &'static str, &'static str);

/// The raa-gtao rows of docs/render-graph/tunables.md, in table order.
const GTAO_ROWS: &[MetaRow] = &[
    ("slices", "pass.ao.gtao", "Quality", "Slices"),
    ("steps", "pass.ao.gtao", "Quality", "Steps"),
    ("scale", "pass.ao.gtao", "Quality", "Pixels per AO texel"),
    ("upsample", "pass.ao.gtao", "Quality", "Bilateral upsample"),
];

/// The raa-contact row of docs/render-graph/tunables.md.
const CONTACT_ROWS: &[MetaRow] = &[("steps", "pass.shadow.contact", "Trace", "Steps")];

/// The feature crates of this family: feature, the crate's `tunables()`, its triage rows.
fn crates() -> [(FeatureId, &'static [TunableSpec], &'static [MetaRow]); 2] {
    [
        (FeatureId::Gtao, raa_gtao::tunables(), GTAO_ROWS),
        (FeatureId::Contact, raa_contact::tunables(), CONTACT_ROWS),
    ]
}

struct TunablesLighting;

impl NodeFamily for TunablesLighting {
    fn name(&self) -> &'static str {
        "tunables_lighting"
    }

    fn node_types(&self) -> Vec<NodeTypeSpec> {
        Vec::new()
    }

    fn extend_types(&self, types: &mut Vec<NodeTypeSpec>) {
        for (feature, specs, rows) in crates() {
            for &(key, node_type, group, label) in rows {
                let spec = find_tunable(specs, key)
                    .unwrap_or_else(|| panic!("tunables_lighting: no tunable spec for `{key}`"));
                let node = types
                    .iter_mut()
                    .find(|node| node.id == node_type)
                    .unwrap_or_else(|| {
                        panic!("tunables_lighting: no node type `{node_type}` for `{key}`")
                    });
                node.params.push(param(&feature, spec, group, label));
            }
            for spec in specs {
                assert!(
                    rows.iter().any(|row| row.0 == spec.key),
                    "tunables_lighting: tunable `{}` has no triage row",
                    spec.key
                );
            }
        }
    }
}

/// The parameter of one tunable: kind, default, range and rebuild taken from its spec.
fn param(
    feature: &FeatureId,
    spec: &TunableSpec,
    group: &'static str,
    label: &'static str,
) -> ParamSpec {
    let (kind, default, step) = if spec.integer {
        (
            ParamKind::Int,
            ParamValue::Int(spec.default as i64),
            Some(1.0),
        )
    } else {
        (ParamKind::Number, ParamValue::Number(spec.default), None)
    };
    ParamSpec {
        key: spec.key,
        label,
        group,
        kind,
        default,
        min: spec.min,
        max: spec.max,
        step,
        options: &[],
        unit: "",
        rebuild: rebuild(spec),
        primary: false,
        unstated_range: spec.min.is_none() && spec.max.is_none(),
        binding: ParamBinding::Tunable {
            feature: feature.clone(),
            key: spec.key,
        },
        source: spec.source,
    }
}

/// The rebuild class of a spec: `none`, `pipeline` or `resources` (S56).
fn rebuild(spec: &TunableSpec) -> Rebuild {
    match spec.rebuild {
        "none" => Rebuild::None,
        "pipeline" => Rebuild::Pipeline,
        "resources" => Rebuild::Resources,
        other => panic!("tunables_lighting: `{}` has rebuild `{other}`", spec.key),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Catalog;

    /// The pass node types before any extension: the input `extend_types` gets inside the catalog,
    /// before this family has added its parameters (`Catalog::build()` already holds them).
    fn passes_types() -> Vec<NodeTypeSpec> {
        crate::registry::passes::family().node_types()
    }

    fn find_node<'a>(types: &'a [NodeTypeSpec], id: &str) -> &'a NodeTypeSpec {
        types.iter().find(|node| node.id == id).expect("node type")
    }

    fn find_param<'a>(node: &'a NodeTypeSpec, key: &str) -> &'a ParamSpec {
        node.params
            .iter()
            .find(|p| p.key == key)
            .expect("parameter")
    }

    #[test]
    fn family_names_itself_and_adds_no_node_type() {
        assert_eq!(family().name(), "tunables_lighting");
        assert!(family().node_types().is_empty());
    }

    #[test]
    fn extend_adds_gtao_slices_and_steps_after_the_pass_params() {
        let mut types = passes_types();
        let before = find_node(&types, "pass.ao.gtao").params.len();
        family().extend_types(&mut types);
        let gtao = find_node(&types, "pass.ao.gtao");
        assert_eq!(gtao.params.len(), before + 4);
        assert_eq!(gtao.params[before].key, "slices");
        assert_eq!(gtao.params[before + 1].key, "steps");

        let slices = find_param(gtao, "slices");
        assert_eq!(slices.kind, ParamKind::Int);
        assert_eq!(slices.default, ParamValue::Int(4));
        assert_eq!((slices.min, slices.max), (Some(1.0), Some(8.0)));
        assert_eq!(slices.step, Some(1.0));
        assert_eq!(slices.group, "Quality");
        assert_eq!(slices.label, "Slices");
        assert_eq!(slices.rebuild, Rebuild::None);
        assert!(!slices.unstated_range);
        assert_eq!(
            slices.binding,
            ParamBinding::Tunable {
                feature: FeatureId::Gtao,
                key: "slices"
            }
        );

        let steps = find_param(gtao, "steps");
        assert_eq!(steps.kind, ParamKind::Int);
        assert_eq!(steps.default, ParamValue::Int(8));
        assert_eq!((steps.min, steps.max), (Some(1.0), Some(16.0)));
        assert_eq!(steps.rebuild, Rebuild::None);
        assert_eq!(
            steps.binding,
            ParamBinding::Tunable {
                feature: FeatureId::Gtao,
                key: "steps"
            }
        );
    }

    #[test]
    fn extend_adds_contact_steps_with_an_unstated_range() {
        let mut types = passes_types();
        family().extend_types(&mut types);
        let steps = find_param(find_node(&types, "pass.shadow.contact"), "steps");
        assert_eq!(steps.kind, ParamKind::Int);
        assert_eq!(steps.default, ParamValue::Int(24));
        assert_eq!((steps.min, steps.max), (None, None));
        assert!(steps.unstated_range);
        assert_eq!(steps.group, "Trace");
        assert_eq!(steps.rebuild, Rebuild::Pipeline);
        assert_eq!(
            steps.binding,
            ParamBinding::Tunable {
                feature: FeatureId::Contact,
                key: "steps"
            }
        );
    }

    #[test]
    fn every_tunable_key_has_one_row_and_one_catalog_parameter() {
        let catalog = Catalog::build();
        for (feature, specs, rows) in crates() {
            for spec in specs {
                let in_rows = rows.iter().filter(|row| row.0 == spec.key).count();
                assert_eq!(in_rows, 1, "triage rows of `{}`", spec.key);
                let bound = catalog
                    .nodes
                    .iter()
                    .flat_map(|node| node.params.iter())
                    .filter(|p| {
                        p.binding
                            == ParamBinding::Tunable {
                                feature: feature.clone(),
                                key: spec.key,
                            }
                    })
                    .count();
                assert_eq!(bound, 1, "{feature:?} parameters `{}`", spec.key);
            }
        }
    }

    #[test]
    fn catalog_builds_with_the_lighting_tunables() {
        let catalog = Catalog::build();
        let gtao = catalog.node("pass.ao.gtao").expect("pass.ao.gtao");
        assert!(gtao.params.iter().any(|p| p.key == "slices"));
        assert!(gtao.params.iter().any(|p| p.key == "steps"));
        let contact = catalog
            .node("pass.shadow.contact")
            .expect("pass.shadow.contact");
        assert!(contact.params.iter().any(|p| p.key == "steps"));
    }
}
