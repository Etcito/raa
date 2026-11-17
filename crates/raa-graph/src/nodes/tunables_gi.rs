//! Family `tunables_gi`: the promoted tunables of the raa-gi node types (S56).
//!
//! `extend_types` appends one `ParamBinding::Tunable` parameter per triage row to the node type
//! the row names. The rows are copied from the raa-gi rows of docs/render-graph/tunables.md; the
//! default, range and rebuild class of each come from `raa_gi::tunables()`. The six
//! pass.gi.area-shadow uniform rows keep their table key as parameter key and bind the
//! `area`-prefixed store key of raa-gi (`area_store_key`).

use raa_pipeline::{TunableSpec, find_tunable};

use crate::catalog::{
    FeatureId, NodeFamily, NodeTypeSpec, ParamBinding, ParamKind, ParamSpec, ParamValue, Rebuild,
};

/// The `tunables_gi` family, discovered by build.rs.
pub fn family() -> Box<dyn NodeFamily> {
    Box::new(TunablesGi)
}

/// One triage row: `(key, node type, group, label)`.
type MetaRow = (&'static str, &'static str, &'static str, &'static str);

/// The raa-gi rows of docs/render-graph/tunables.md, in table order.
const ROWS: &[MetaRow] = &[
    (
        "restirDepthTolerance",
        "pass.gi.area-shadow",
        "ReSTIR",
        "Restir depth tolerance",
    ),
    (
        "spatialStride",
        "pass.gi.area-shadow",
        "Spatial",
        "Spatial stride",
    ),
    (
        "normalPower",
        "pass.gi.area-shadow",
        "Spatial",
        "Normal power",
    ),
    (
        "temporalAlpha",
        "pass.gi.area-shadow",
        "Temporal",
        "Temporal alpha",
    ),
    (
        "depthTolerance",
        "pass.gi.area-shadow",
        "Temporal",
        "Depth tolerance",
    ),
    (
        "accDepthTolerance",
        "pass.gi.screen",
        "Accumulation",
        "Acc depth tolerance",
    ),
    (
        "accNormalCos",
        "pass.gi.screen",
        "Accumulation",
        "Acc normal cos",
    ),
    (
        "accMinWeight",
        "pass.gi.screen",
        "Accumulation",
        "Acc min weight",
    ),
    (
        "accChangeSlack",
        "pass.gi.screen",
        "Accumulation",
        "Acc change slack",
    ),
    (
        "accChangeMin",
        "pass.gi.screen",
        "Accumulation",
        "Acc change min",
    ),
    (
        "occlusionDirections",
        "pass.gi.screen",
        "Accumulation",
        "Occlusion directions",
    ),
    (
        "occlusionSteps",
        "pass.gi.screen",
        "Accumulation",
        "Occlusion steps",
    ),
    (
        "occlusionRadius",
        "pass.gi.screen",
        "Accumulation",
        "Occlusion radius",
    ),
    (
        "atrousSigmaLuminance",
        "pass.gi.screen",
        "A-trous",
        "Atrous sigma luminance",
    ),
    (
        "atrousNormalPower",
        "pass.gi.screen",
        "A-trous",
        "Atrous normal power",
    ),
    ("atrousPlane", "pass.gi.screen", "A-trous", "Atrous plane"),
    (
        "probeScreenDistance",
        "pass.gi.screen",
        "Screen probes",
        "Probe screen distance",
    ),
    (
        "probeScreenSteps",
        "pass.gi.screen",
        "Screen probes",
        "Probe screen steps",
    ),
    (
        "probeScreenThickness",
        "pass.gi.screen",
        "Screen probes",
        "Probe screen thickness",
    ),
    (
        "probeVolumeDistance",
        "pass.gi.screen",
        "Screen probes",
        "Probe volume distance",
    ),
    (
        "probeVolumeSteps",
        "pass.gi.screen",
        "Screen probes",
        "Probe volume steps",
    ),
    (
        "probeVolumeHitVoxels",
        "pass.gi.screen",
        "Screen probes",
        "Probe volume hit voxels",
    ),
    (
        "restirJacobianMax",
        "pass.gi.screen",
        "ReSTIR GI",
        "Restir jacobian max",
    ),
    (
        "restirVisibilitySteps",
        "pass.gi.screen",
        "ReSTIR GI",
        "Restir visibility steps",
    ),
    (
        "restirVisibilityDistance",
        "pass.gi.screen",
        "ReSTIR GI",
        "Restir visibility distance",
    ),
    (
        "restirScreenDistance",
        "pass.gi.screen",
        "ReSTIR GI",
        "Restir screen distance",
    ),
    (
        "restirScreenSteps",
        "pass.gi.screen",
        "ReSTIR GI",
        "Restir screen steps",
    ),
    (
        "restirScreenThickness",
        "pass.gi.screen",
        "ReSTIR GI",
        "Restir screen thickness",
    ),
    (
        "restirVolumeDistance",
        "pass.gi.screen",
        "ReSTIR GI",
        "Restir volume distance",
    ),
    (
        "restirVolumeSteps",
        "pass.gi.screen",
        "ReSTIR GI",
        "Restir volume steps",
    ),
    (
        "restirVolumeHitVoxels",
        "pass.gi.screen",
        "ReSTIR GI",
        "Restir volume hit voxels",
    ),
    (
        "screenEdgeFade",
        "pass.gi.screen",
        "Screen trace",
        "Screen edge fade",
    ),
    ("changeSigma", "pass.gi.ssgi", "Resolve", "Change sigma"),
    ("behind", "pass.gi.volume", "Volume write", "Behind"),
    ("resetRays", "pass.gi.volume", "Voxel light", "Reset rays"),
    (
        "directChange",
        "pass.gi.volume",
        "Voxel light",
        "Direct change",
    ),
    ("slicesLow", "pass.gi.ssgi", "Quality table", "Slices (low)"),
    (
        "slicesMedium",
        "pass.gi.ssgi",
        "Quality table",
        "Slices (medium)",
    ),
    (
        "slicesHigh",
        "pass.gi.ssgi",
        "Quality table",
        "Slices (high)",
    ),
    ("stepsLow", "pass.gi.ssgi", "Quality table", "Steps (low)"),
    (
        "stepsMedium",
        "pass.gi.ssgi",
        "Quality table",
        "Steps (medium)",
    ),
    ("stepsHigh", "pass.gi.ssgi", "Quality table", "Steps (high)"),
    (
        "restirSpatialSamples",
        "pass.gi.screen",
        "ReSTIR GI",
        "Spatial samples",
    ),
    (
        "restirSpatialRadius",
        "pass.gi.screen",
        "ReSTIR GI",
        "Spatial radius",
    ),
    (
        "restirBoilingFactor",
        "pass.gi.screen",
        "ReSTIR GI",
        "Boiling factor",
    ),
    (
        "restirNormalCos",
        "pass.gi.screen",
        "ReSTIR GI",
        "Normal cosine",
    ),
    (
        "restirCandidates",
        "pass.gi.area-shadow",
        "ReSTIR",
        "Candidates",
    ),
    (
        "restirSpatialSamples",
        "pass.gi.area-shadow",
        "ReSTIR",
        "Spatial samples",
    ),
    (
        "restirSpatialRadius",
        "pass.gi.area-shadow",
        "ReSTIR",
        "Spatial radius",
    ),
    ("restirMCap", "pass.gi.area-shadow", "ReSTIR", "M cap"),
    (
        "restirBoilingFactor",
        "pass.gi.area-shadow",
        "ReSTIR",
        "Boiling factor",
    ),
    (
        "restirNormalCos",
        "pass.gi.area-shadow",
        "ReSTIR",
        "Normal cosine",
    ),
];

/// The raa-gi store key of a row: the six area-shadow uniform rows carry an `area` prefix there,
/// because the store is one flat registry and four of their keys are ReSTIR GI keys.
fn store_key(node_type: &str, key: &'static str) -> &'static str {
    if node_type != "pass.gi.area-shadow" {
        return key;
    }
    match key {
        "restirCandidates" => "areaRestirCandidates",
        "restirSpatialSamples" => "areaRestirSpatialSamples",
        "restirSpatialRadius" => "areaRestirSpatialRadius",
        "restirMCap" => "areaRestirMCap",
        "restirBoilingFactor" => "areaRestirBoilingFactor",
        "restirNormalCos" => "areaRestirNormalCos",
        _ => key,
    }
}

struct TunablesGi;

impl NodeFamily for TunablesGi {
    fn name(&self) -> &'static str {
        "tunables_gi"
    }

    fn node_types(&self) -> Vec<NodeTypeSpec> {
        Vec::new()
    }

    fn extend_types(&self, types: &mut Vec<NodeTypeSpec>) {
        let specs = raa_gi::tunables();
        for &(key, node_type, group, label) in ROWS {
            let bound = store_key(node_type, key);
            let spec = find_tunable(specs, bound)
                .unwrap_or_else(|| panic!("tunables_gi: no tunable spec for `{bound}`"));
            let node = types
                .iter_mut()
                .find(|node| node.id == node_type)
                .unwrap_or_else(|| panic!("tunables_gi: no node type `{node_type}` for `{key}`"));
            node.params.push(param(key, spec, group, label));
        }
        for spec in specs {
            assert!(
                ROWS.iter().any(|row| store_key(row.1, row.0) == spec.key),
                "tunables_gi: tunable `{}` has no triage row",
                spec.key
            );
        }
    }
}

/// The parameter of one tunable: kind, default, range and rebuild taken from its spec.
fn param(
    key: &'static str,
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
        key,
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
            feature: FeatureId::Gi,
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
        other => panic!("tunables_gi: `{}` has rebuild `{other}`", spec.key),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Catalog;

    fn param_of(catalog: &Catalog, node: &str, key: &str) -> ParamSpec {
        let node = catalog.node(node).expect("node type");
        node.params
            .iter()
            .find(|p| p.key == key)
            .expect("parameter")
            .clone()
    }

    #[test]
    fn screen_probe_rows_are_pinned() {
        let catalog = Catalog::build();
        let rows = [
            (
                "probeScreenDistance",
                ParamValue::Number(8.0),
                ParamKind::Number,
            ),
            ("probeScreenSteps", ParamValue::Int(32), ParamKind::Int),
            (
                "probeScreenThickness",
                ParamValue::Number(0.15),
                ParamKind::Number,
            ),
        ];
        for (key, default, kind) in rows {
            let p = param_of(&catalog, "pass.gi.screen", key);
            assert_eq!((p.default, p.kind), (default, kind), "{key}");
            assert_eq!(p.group, "Screen probes");
            assert_eq!(p.rebuild, Rebuild::Pipeline);
            assert!(p.unstated_range);
            assert_eq!(
                p.binding,
                ParamBinding::Tunable {
                    feature: FeatureId::Gi,
                    key
                }
            );
        }
    }

    #[test]
    fn ssgi_quality_rows_are_pinned() {
        let catalog = Catalog::build();
        let rows = [
            ("slicesLow", 2),
            ("slicesMedium", 4),
            ("slicesHigh", 6),
            ("stepsLow", 6),
            ("stepsMedium", 8),
            ("stepsHigh", 12),
        ];
        for (key, default) in rows {
            let p = param_of(&catalog, "pass.gi.ssgi", key);
            assert_eq!(p.kind, ParamKind::Int, "{key}");
            assert_eq!(p.default, ParamValue::Int(default), "{key}");
            assert_eq!(p.group, "Quality table");
            assert_eq!(p.rebuild, Rebuild::None);
        }
    }

    #[test]
    fn area_shadow_temporal_alpha_is_pinned() {
        let p = param_of(&Catalog::build(), "pass.gi.area-shadow", "temporalAlpha");
        assert_eq!(p.default, ParamValue::Number(0.1));
        assert_eq!(p.group, "Temporal");
        assert_eq!(p.rebuild, Rebuild::Pipeline);
    }

    #[test]
    fn area_rows_bind_the_prefixed_store_key() {
        let p = param_of(&Catalog::build(), "pass.gi.area-shadow", "restirMCap");
        assert_eq!(p.default, ParamValue::Number(20.0));
        assert_eq!(
            p.binding,
            ParamBinding::Tunable {
                feature: FeatureId::Gi,
                key: "areaRestirMCap"
            }
        );
    }

    #[test]
    fn every_store_key_appears_exactly_once() {
        for spec in raa_gi::tunables() {
            let n = ROWS
                .iter()
                .filter(|row| store_key(row.1, row.0) == spec.key)
                .count();
            assert_eq!(n, 1, "rows of `{}`", spec.key);
        }
        assert_eq!(ROWS.len(), raa_gi::tunables().len());
    }

    #[test]
    fn catalog_builds() {
        let _ = Catalog::build();
        assert_eq!(family().name(), "tunables_gi");
        assert!(family().node_types().is_empty());
    }
}
