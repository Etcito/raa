//! Catalog family `tunables_post`: the promoted tunables of taa, ssr, dof and sunshafts.

use crate::catalog::{
    FeatureId, NodeFamily, NodeTypeSpec, ParamBinding, ParamKind, ParamSpec, ParamValue, Rebuild,
};
use raa_pipeline::{TunableSpec, find_tunable};

/// (key, node type, group, label), copied from docs/render-graph/tunables.md.
type Row = (&'static str, &'static str, &'static str, &'static str);

const TAA: &[Row] = &[
    ("filterSharpness", "pass.taa", "Resolve", "Filter sharpness"),
    ("clipTolerance", "pass.taa", "Resolve", "Clip tolerance"),
    ("blendBase", "pass.taa", "Resolve", "Blend base"),
    (
        "blendMotionGain",
        "pass.taa",
        "Resolve",
        "Blend motion gain",
    ),
    ("blendMin", "pass.taa", "Resolve", "Blend min"),
    ("blendMax", "pass.taa", "Resolve", "Blend max"),
    (
        "historySharpness",
        "pass.taa",
        "Resolve",
        "History sharpness",
    ),
    ("blendStatic", "pass.taa", "Resolve", "Blend at rest"),
    ("clipStatic", "pass.taa", "Resolve", "Clip scale at rest"),
    (
        "upsampleConfidence",
        "pass.taa",
        "Resolve",
        "Upsample confidence",
    ),
    (
        "upsampleClipScene",
        "pass.taa",
        "Resolve",
        "Upsample clip from scene samples",
    ),
    ("upsampleKernel", "pass.taa", "Resolve", "Upsample kernel"),
];
const SSR: &[Row] = &[
    ("maxSteps", "pass.ssr", "Trace", "Max steps"),
    ("refineSteps", "pass.ssr", "Trace", "Refine steps"),
    ("edgeFade", "pass.ssr", "Trace", "Edge fade"),
];
const DOF: &[Row] = &[
    ("taps", "pass.dof", "Gather", "Taps"),
    ("radiusPerCoc", "pass.dof", "Gather", "Radius per coc"),
];
const SHAFTS: &[Row] = &[
    ("steps", "pass.sunshafts", "Shafts", "Steps"),
    ("hgG", "pass.sunshafts", "Shafts", "Hg g"),
];

struct TunablesPost;

pub fn family() -> Box<dyn NodeFamily> {
    Box::new(TunablesPost)
}

fn attach(
    types: &mut [NodeTypeSpec],
    feature: FeatureId,
    rows: &[Row],
    specs: &'static [TunableSpec],
) {
    for &(key, node_type, group, label) in rows {
        let spec = find_tunable(specs, key).unwrap_or_else(|| panic!("no tunable spec {key}"));
        let node = types
            .iter_mut()
            .find(|t| t.id == node_type)
            .unwrap_or_else(|| panic!("no node type {node_type} for {key}"));
        node.params.push(ParamSpec {
            key,
            label,
            group,
            kind: if spec.integer {
                ParamKind::Int
            } else {
                ParamKind::Number
            },
            default: if spec.integer {
                ParamValue::Int(spec.default as i64)
            } else {
                ParamValue::Number(spec.default)
            },
            min: spec.min,
            max: spec.max,
            step: spec.integer.then_some(1.0),
            options: &[],
            unit: "",
            rebuild: match spec.rebuild {
                "none" => Rebuild::None,
                "pipeline" => Rebuild::Pipeline,
                "resources" => Rebuild::Resources,
                other => panic!("tunable {key}: unknown rebuild {other}"),
            },
            primary: false,
            unstated_range: spec.min.is_none() && spec.max.is_none(),
            binding: ParamBinding::Tunable {
                feature: feature.clone(),
                key,
            },
            source: spec.source,
        });
    }
    for spec in specs {
        if !rows.iter().any(|r| r.0 == spec.key) {
            panic!("tunable {} has no META row", spec.key);
        }
    }
}

impl NodeFamily for TunablesPost {
    fn name(&self) -> &'static str {
        "tunables_post"
    }

    fn node_types(&self) -> Vec<NodeTypeSpec> {
        Vec::new()
    }

    fn extend_types(&self, types: &mut Vec<NodeTypeSpec>) {
        attach(types, FeatureId::Taa, TAA, raa_taa::tunables());
        attach(types, FeatureId::Ssr, SSR, raa_ssr::tunables());
        attach(types, FeatureId::Dof, DOF, raa_dof::tunables());
        attach(types, FeatureId::Shafts, SHAFTS, raa_shafts::tunables());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Catalog;

    fn params(id: &str) -> Vec<ParamSpec> {
        Catalog::build().node(id).unwrap().params.clone()
    }

    #[test]
    fn taa_rows_in_pinned_order() {
        let p: Vec<_> = params("pass.taa")
            .into_iter()
            .filter(|p| matches!(p.binding, ParamBinding::Tunable { .. }))
            .collect();
        let keys: Vec<_> = p.iter().map(|p| p.key).collect();
        assert_eq!(
            keys,
            [
                "filterSharpness",
                "clipTolerance",
                "blendBase",
                "blendMotionGain",
                "blendMin",
                "blendMax",
                "historySharpness",
                "blendStatic",
                "clipStatic",
                "upsampleConfidence",
                "upsampleClipScene",
                "upsampleKernel"
            ]
        );
        let defaults = [8.0, 1.25, 0.04, 0.03, 0.04, 0.5, 0.6, 0.015, 3.0, 3.0, 1.0, 1.0];
        for (p, d) in p.iter().zip(defaults) {
            assert_eq!(p.kind, ParamKind::Number);
            assert_eq!(p.default, ParamValue::Number(d));
            assert_eq!(p.rebuild, Rebuild::Pipeline);
            assert!(p.unstated_range && p.min.is_none() && p.max.is_none());
        }
    }

    #[test]
    fn ssr_max_steps_is_int() {
        let p = params("pass.ssr");
        let m = p.iter().find(|p| p.key == "maxSteps").unwrap();
        assert_eq!(m.kind, ParamKind::Int);
        assert_eq!(m.default, ParamValue::Int(64));
    }

    #[test]
    fn every_key_appears_once() {
        let cat = Catalog::build();
        let all = [
            ("pass.taa", raa_taa::tunables()),
            ("pass.ssr", raa_ssr::tunables()),
            ("pass.dof", raa_dof::tunables()),
            ("pass.sunshafts", raa_shafts::tunables()),
        ];
        for (id, specs) in all {
            let node = cat.node(id).unwrap();
            for s in specs {
                let n = node.params.iter().filter(|p| p.key == s.key).count();
                assert_eq!(n, 1, "{id} {}", s.key);
            }
        }
    }

    #[test]
    fn catalog_builds() {
        Catalog::build();
    }
}
