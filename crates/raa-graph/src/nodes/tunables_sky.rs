//! Catalog family `tunables_sky`: the promoted tunables of raa-sky, bound to the three sky pass
//! nodes (atmosphere rows, cloud volume and shadow rows, dome and screen rows). They are
//! render-quality constants of the sky passes; sky, time and weather values are scene content and
//! never graph parameters (S54).

use crate::catalog::{
    FeatureId, NodeFamily, NodeTypeSpec, ParamBinding, ParamKind, ParamSpec, ParamValue, Rebuild,
};
use raa_pipeline::{TunableSpec, find_tunable};

/// (key, node type, group, label), copied from the raa-sky rows of docs/render-graph/tunables.md
/// in the table's order: cloud volume, dome, screen, shadow, then atmosphere.
type Row = (&'static str, &'static str, &'static str, &'static str);

const ATMOSPHERE: &str = "pass.sky.atmosphere";
const CLOUD_LIGHTING: &str = "pass.sky.cloud-lighting";
const CLOUD_SCREEN: &str = "pass.sky.cloud-screen";

const META: &[Row] = &[
    (
        "cloudWeatherTile",
        CLOUD_LIGHTING,
        "Cloud volume",
        "Cloud weather tile",
    ),
    (
        "cloudShapeTile",
        CLOUD_LIGHTING,
        "Cloud volume",
        "Cloud shape tile",
    ),
    (
        "cloudDetailTile",
        CLOUD_LIGHTING,
        "Cloud volume",
        "Cloud detail tile",
    ),
    (
        "cloudExtinction",
        CLOUD_LIGHTING,
        "Cloud volume",
        "Cloud extinction",
    ),
    (
        "cloudAlbedo",
        CLOUD_LIGHTING,
        "Cloud volume",
        "Cloud albedo",
    ),
    (
        "cloudCirrusHeight",
        CLOUD_LIGHTING,
        "Cloud volume",
        "Cloud cirrus height",
    ),
    (
        "cloudMaxDistance",
        CLOUD_LIGHTING,
        "Cloud volume",
        "Cloud max distance",
    ),
    (
        "cloudMaxPath",
        CLOUD_LIGHTING,
        "Cloud volume",
        "Cloud max path",
    ),
    (
        "cloudDomeBelow",
        CLOUD_SCREEN,
        "Cloud dome",
        "Cloud dome below",
    ),
    (
        "cloudDomeLightSamples",
        CLOUD_SCREEN,
        "Cloud dome",
        "Cloud dome light samples",
    ),
    (
        "cloudDomeAnimatedFloor",
        CLOUD_SCREEN,
        "Cloud dome",
        "Cloud dome animated floor",
    ),
    (
        "cloudDomeBurst",
        CLOUD_SCREEN,
        "Cloud dome",
        "Cloud dome burst",
    ),
    (
        "cloudScreenLightSamples",
        CLOUD_SCREEN,
        "Cloud screen",
        "Cloud screen light samples",
    ),
    (
        "cloudScreenMovingBlend",
        CLOUD_SCREEN,
        "Cloud screen",
        "Cloud screen moving blend",
    ),
    (
        "cloudScreenPrior",
        CLOUD_SCREEN,
        "Cloud screen",
        "Cloud screen prior",
    ),
    (
        "cloudScreenAnimatedFloor",
        CLOUD_SCREEN,
        "Cloud screen",
        "Cloud screen animated floor",
    ),
    (
        "cloudScreenBurst",
        CLOUD_SCREEN,
        "Cloud screen",
        "Cloud screen burst",
    ),
    (
        "cloudShadowSamples",
        CLOUD_LIGHTING,
        "Cloud shadow",
        "Cloud shadow samples",
    ),
    ("skyViewSteps", ATMOSPHERE, "Atmosphere", "Sky-view steps"),
    ("horizonFade", ATMOSPHERE, "Atmosphere", "Horizon fade"),
];

struct TunablesSky;

/// Registry entry of the family. `extend_types` appends one ParamSpec per row of META to its sky
/// pass node, bound by key to `raa_sky::tunables` (the 20 raa-sky rows of the triage table).
pub fn family() -> Box<dyn NodeFamily> {
    Box::new(TunablesSky)
}

/// Appends every raa-sky tunable to its sky pass node, in META order.
fn attach(types: &mut [NodeTypeSpec]) {
    let specs = raa_sky::tunables();
    for &(key, node_type, group, label) in META {
        let spec = find_tunable(specs, key).unwrap_or_else(|| panic!("no tunable spec {key}"));
        let node = types
            .iter_mut()
            .find(|t| t.id == node_type)
            .unwrap_or_else(|| panic!("no node type {node_type} for {key}"));
        node.params.push(param(key, label, group, spec));
    }
    for spec in specs {
        if !META.iter().any(|row| row.0 == spec.key) {
            panic!("tunable {} has no META row", spec.key);
        }
    }
}

/// The catalog parameter of one tunable: default, range and rebuild class read from its spec.
fn param(
    key: &'static str,
    label: &'static str,
    group: &'static str,
    spec: &TunableSpec,
) -> ParamSpec {
    let integer = spec.integer;
    ParamSpec {
        key,
        label,
        group,
        kind: if integer {
            ParamKind::Int
        } else {
            ParamKind::Number
        },
        default: if integer {
            ParamValue::Int(spec.default as i64)
        } else {
            ParamValue::Number(spec.default)
        },
        min: spec.min,
        max: spec.max,
        step: integer.then_some(1.0),
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
            feature: FeatureId::Sky,
            key,
        },
        source: spec.source,
    }
}

impl NodeFamily for TunablesSky {
    fn name(&self) -> &'static str {
        "tunables_sky"
    }

    fn node_types(&self) -> Vec<NodeTypeSpec> {
        Vec::new()
    }

    fn extend_types(&self, types: &mut Vec<NodeTypeSpec>) {
        attach(types);
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
    fn atmosphere_holds_sky_view_steps_as_pipeline_int() {
        let p = params("pass.sky.atmosphere");
        let steps = p.iter().find(|p| p.key == "skyViewSteps").unwrap();
        assert_eq!(steps.kind, ParamKind::Int);
        assert_eq!(steps.default, ParamValue::Int(32));
        assert_eq!(steps.rebuild, Rebuild::Pipeline);
        assert_eq!(steps.group, "Atmosphere");
        assert_eq!(steps.label, "Sky-view steps");
    }

    #[test]
    fn cloud_lighting_holds_cloud_albedo_as_number() {
        let p = params("pass.sky.cloud-lighting");
        let albedo = p.iter().find(|p| p.key == "cloudAlbedo").unwrap();
        let spec = find_tunable(raa_sky::tunables(), "cloudAlbedo").unwrap();
        assert_eq!(albedo.kind, ParamKind::Number);
        assert_eq!(albedo.default, ParamValue::Number(spec.default));
        assert_eq!(albedo.default, ParamValue::Number(0.98));
        assert_eq!(albedo.group, "Cloud volume");
        assert!(albedo.unstated_range && albedo.min.is_none() && albedo.max.is_none());
    }

    #[test]
    fn every_sky_tunable_appears_exactly_once() {
        let cat = Catalog::build();
        let bound: Vec<&str> = [ATMOSPHERE, CLOUD_LIGHTING, CLOUD_SCREEN]
            .iter()
            .filter_map(|id| cat.node(id))
            .flat_map(|node| node.params.iter())
            .filter_map(|p| match p.binding {
                ParamBinding::Tunable {
                    feature: FeatureId::Sky,
                    key,
                } => Some(key),
                _ => None,
            })
            .collect();
        let specs = raa_sky::tunables();
        assert_eq!(bound.len(), specs.len());
        for spec in specs {
            let count = bound.iter().filter(|&&key| key == spec.key).count();
            assert_eq!(count, 1, "tunable {}", spec.key);
        }
    }

    #[test]
    fn family_is_named_and_adds_no_node_type() {
        let family = family();
        assert_eq!(family.name(), "tunables_sky");
        assert!(family.node_types().is_empty());
    }

    #[test]
    fn catalog_builds() {
        Catalog::build();
    }
}
