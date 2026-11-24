//! Pins of the pass and output node families (S51) against the settings accessors of
//! `raa_graph::exec` (`rg-settings-accessors`) and the typings in `packages/raa/types`. Reads the
//! catalog and the contract functions only.

use std::collections::BTreeSet;

use raa_graph::Catalog;
use raa_graph::catalog::{Category, FeatureId, NodeTypeSpec, ParamBinding, ParamSpec};
use raa_graph::exec::{default_values, setting_keys};
use raa_loader::json::JsonValue;

const M07: &str = include_str!("../../../packages/raa/types/m07-meshes-hdr.d.ts");
const M09: &str = include_str!("../../../packages/raa/types/m09-shadows-ao.d.ts");
const M10: &str = include_str!("../../../packages/raa/types/m10-materials.d.ts");
const M11: &str = include_str!("../../../packages/raa/types/m11-transmission.d.ts");
const M14: &str = include_str!("../../../packages/raa/types/m14-camera-effects.d.ts");
const M23: &str = include_str!("../../../packages/raa/types/m23-sdf-trace.d.ts");
const M27: &str = include_str!("../../../packages/raa/types/m27-cel.d.ts");
const M28: &str = include_str!("../../../packages/raa/types/m28-lights.d.ts");
const M29: &str = include_str!("../../../packages/raa/types/m29-virtual-geometry.d.ts");
const M30: &str = include_str!("../../../packages/raa/types/m30-global-illumination.d.ts");

const CATEGORIES: &str = "geometry lighting gi shadows post compute view output";
const PORT_TYPES: &str = "image2d image2dArray image3d imageCube depth2d depth2dArray buffer";
const NODE_KEYS: &str = "id version category label description pass contributor \
    defaultOrder multiInstance sideEffect requiredInGraph available inputs outputs params";
const SCENE_SETTERS: &str = "setSky setTimeOfDay setWeather setEnvironmentLighting";

/// Typings members that are not catalog keys, as `(interface, member, reason)`. An entry must
/// still be a member of its interface and must not be a catalog key, so a stale entry fails.
#[rustfmt::skip]
const EXCLUDED: &[(&str, &str, &str)] = &[
    ("GlobalIlluminationSettings", "enabled", "node presence: a pass.gi.* node sets it"),
    ("GlobalIlluminationSettings", "screenSpace", "node presence: pass.gi.ssgi sets it"),
    ("SphericalGiSettings", "enabled", "node presence: pass.gi.spherical sets it"),
    ("VirtualGeometrySettings", "enabled", "node presence: a pass.vg.* node sets it"),
    ("FogSettings", "preset", "preset name applied first, not a Setting key"),
    ("GradingSettings", "preset", "preset name applied first, not a Setting key"),
    ("VirtualGeometrySettings", "runtimeBake", "bake policy of the Engine, not a pipeline setting"),
    ("VirtualGeometrySettings", "instanceDefault", "default for instances, not a pipeline setting"),
    ("DofSettings", "autofocus", "[number, number] | null: no ParamValue holds null (m14:29)"),
    ("AoSettings", "slices", "AoQuality effort (raa-gtao desc.rs:46), not in S51"),
    ("AoSettings", "steps", "AoQuality effort (raa-gtao desc.rs:46), not in S51"),
];

/// A feature with Setting keys: its setter, and the typings file and interface of its input.
struct Feature {
    feature: FeatureId,
    setter: &'static str,
    typings: &'static str,
    interface: &'static str,
}

/// One catalog parameter bound to a Setting key: `(node type, feature, key, spec)`.
type SettingParam<'a> = (&'static str, &'a FeatureId, &'static str, &'a ParamSpec);

#[rustfmt::skip]
fn features() -> Vec<Feature> {
    let row = |feature, setter, typings, interface| Feature { feature, setter, typings, interface };
    vec![
        row(FeatureId::Tonemap, "setToneMapping", M07, "ToneMappingDesc"),
        row(FeatureId::Shadow, "setShadows", M09, "ShadowSettings"),
        row(FeatureId::Gtao, "setAo", M09, "AoSettings"),
        row(FeatureId::Contact, "setContactShadows", M09, "ContactShadowSettings"),
        row(FeatureId::Bloom, "setBloom", M10, "BloomSettings"),
        row(FeatureId::Oit, "setTransparency", M11, "TransparencySettings"),
        row(FeatureId::Ssr, "setSsr", M14, "SsrSettings"),
        row(FeatureId::Dof, "setDof", M14, "DofSettings"),
        row(FeatureId::Fog, "setFog", M14, "FogSettings"),
        row(FeatureId::Shafts, "setSunShafts", M14, "SunShaftsSettings"),
        row(FeatureId::Grading, "setGrading", M14, "GradingSettings"),
        row(FeatureId::Cel, "setCel", M27, "CelSettings"),
        row(FeatureId::AreaShading, "setAreaLightShading", M28, "AreaLightShadingDesc"),
        row(FeatureId::AreaShadows, "setAreaLightShadows", M28, "AreaLightShadowsDesc"),
        row(FeatureId::Vg, "setVirtualGeometry", M29, "VirtualGeometrySettings"),
        row(FeatureId::Gi, "setGlobalIllumination", M30, "GlobalIlluminationSettings"),
        row(FeatureId::GiQuality, "setGiQuality", M30, "GiQuality"),
        row(FeatureId::Other("sphericalGi"), "setSphericalGi", M30, "SphericalGiSettings"),
        row(FeatureId::Other("lightingIsolation"), "setMeshTracing", M23, "MeshTracingSettings"),
    ]
}

/// The members of `export interface <interface> {` in `typings`, one level deep.
fn members<'a>(typings: &'a str, interface: &str) -> Vec<&'a str> {
    let header = format!("export interface {interface} {{\n");
    let start = typings
        .find(header.as_str())
        .unwrap_or_else(|| panic!("the typings have no `{interface}`"))
        + header.len();
    let body = &typings[start..];
    let end = body
        .find("\n}")
        .unwrap_or_else(|| panic!("`{interface}` is not closed"));
    body[..end].lines().filter_map(member_name).collect()
}

/// The name of a top-level property line (`  name?: type;`); `None` for comments and nested lines.
fn member_name(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("  ")?;
    let end = rest.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))?;
    let (name, tail) = rest.split_at(end);
    let tail = tail.strip_prefix('?').unwrap_or(tail);
    (!name.is_empty() && tail.starts_with(':')).then_some(name)
}

/// Every catalog parameter bound to a Setting key, in catalog order.
fn setting_params(catalog: &Catalog) -> Vec<SettingParam<'_>> {
    catalog
        .nodes
        .iter()
        .flat_map(|node| {
            node.params
                .iter()
                .filter_map(move |param| match &param.binding {
                    ParamBinding::Setting { feature, key } => Some((node.id, feature, *key, param)),
                    _ => None,
                })
        })
        .collect()
}

/// The catalog Setting keys of `feature`.
fn catalog_keys(params: &[SettingParam<'_>], feature: &FeatureId) -> BTreeSet<&'static str> {
    params
        .iter()
        .filter(|(_, owner, _, _)| *owner == feature)
        .map(|(_, _, key, _)| *key)
        .collect()
}

/// The entry `name` of a JSON object.
fn field<'a>(entries: &'a [(String, JsonValue)], name: &str) -> &'a JsonValue {
    entries
        .iter()
        .find(|(key, _)| key.as_str() == name)
        .map(|(_, value)| value)
        .unwrap_or_else(|| panic!("the catalog JSON has no `{name}`"))
}

/// The string items of a JSON array.
fn strings(value: &JsonValue) -> Vec<&str> {
    let JsonValue::Array(items) = value else {
        panic!("{value:?} is not an array")
    };
    items
        .iter()
        .filter_map(|item| match item {
            JsonValue::String(text) => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

#[test]
fn catalog_builds_with_unique_ids_ports_and_params() {
    let catalog = Catalog::build();
    assert!(
        catalog.node("pass.forward").is_some() && catalog.node("output.display").is_some(),
        "the pass and output families are not registered"
    );
    let mut ids = BTreeSet::new();
    let mut ports = BTreeSet::new();
    let mut params = BTreeSet::new();
    for node in &catalog.nodes {
        assert!(ids.insert(node.id), "duplicate node id `{}`", node.id);
        // Inputs and outputs are separate namespaces: `color` is both on `pass.ssr`.
        for (direction, specs) in [("input", &node.inputs), ("output", &node.outputs)] {
            for port in specs {
                let fresh = ports.insert((node.id, direction, port.key));
                assert!(fresh, "duplicate {direction} port `{}`", port.key);
            }
        }
        for param in &node.params {
            let fresh = params.insert((node.id, param.key));
            assert!(
                fresh,
                "duplicate parameter `{}` on `{}`",
                param.key, node.id
            );
        }
    }
}

#[test]
fn setting_keys_match_the_typings_both_ways() {
    let catalog = Catalog::build();
    let features = features();
    let params = setting_params(&catalog);

    // Direction A: every catalog Setting key is a member of its feature's interface.
    for (node, feature, key, _) in &params {
        let row = features
            .iter()
            .find(|row| row.feature == **feature)
            .unwrap_or_else(|| panic!("`{node}` binds `{key}` to no setter"));
        let listed = members(row.typings, row.interface);
        assert!(
            listed.contains(key),
            "`{node}`: `{key}` is not a member of {}",
            row.interface
        );
    }

    // Direction B: every member is a catalog key, or excluded with a reason.
    for row in &features {
        if row.feature == FeatureId::Other("lightingIsolation") {
            // Forward lighting uses a single field from the legacy mesh trace setter.
            assert_eq!(
                catalog_keys(&params, &row.feature),
                BTreeSet::from(["lightingOutput"])
            );
            continue;
        }
        let keys = catalog_keys(&params, &row.feature);
        for member in members(row.typings, row.interface) {
            let excluded = EXCLUDED
                .iter()
                .any(|(interface, name, _)| *interface == row.interface && *name == member);
            assert!(
                keys.contains(member) || excluded,
                "`{}`: member `{member}` of {} is neither a key nor excluded",
                row.setter,
                row.interface
            );
        }
    }
    for (interface, name, reason) in EXCLUDED {
        assert!(!reason.is_empty(), "`{interface}.{name}` has no reason");
        let row = features
            .iter()
            .find(|row| row.interface == *interface)
            .unwrap_or_else(|| panic!("exclusion `{interface}.{name}` names no feature"));
        let member = members(row.typings, interface).contains(name);
        assert!(member, "stale exclusion `{interface}.{name}`");
        let key = catalog_keys(&params, &row.feature).contains(name);
        assert!(
            !key,
            "`{interface}.{name}` is a catalog key, drop its exclusion"
        );
    }

    // The S54 scene setters are members of no input interface.
    let all: Vec<&str> = features
        .iter()
        .flat_map(|row| members(row.typings, row.interface))
        .collect();
    for scene in SCENE_SETTERS.split(' ') {
        assert!(
            !all.contains(&scene),
            "scene setter `{scene}` is an input member"
        );
    }
}

#[test]
fn setting_defaults_equal_default_values() {
    let catalog = Catalog::build();
    let params = setting_params(&catalog);
    for (node, feature, key, spec) in &params {
        let defaults = default_values(feature);
        let value = defaults
            .iter()
            .find(|(name, _)| name.as_str() == *key)
            .map(|(_, value)| value);
        assert_eq!(
            value,
            Some(&spec.default),
            "`{node}.{key}`: default differs"
        );
    }

    // setting_keys lists exactly the catalog Setting keys of each feature, once each.
    for row in features() {
        let listed = setting_keys(&row.feature);
        let unique: BTreeSet<&str> = listed.iter().copied().collect();
        assert_eq!(
            unique.len(),
            listed.len(),
            "`{}` lists a key twice",
            row.setter
        );
        assert_eq!(
            unique,
            catalog_keys(&params, &row.feature),
            "`{}`",
            row.setter
        );
    }
    for feature in [FeatureId::Pipeline, FeatureId::Taa, FeatureId::Other("x")] {
        assert!(setting_keys(&feature).is_empty());
        assert!(default_values(&feature).is_empty());
    }
}

#[test]
fn nodes_are_listed_by_rank_default_order_and_id() {
    let catalog = Catalog::build();
    // The S51 order: category rank, default order, then the id bytewise.
    let key = |node: &NodeTypeSpec| (node.category.rank(), node.default_order, node.id);
    for pair in catalog.nodes.windows(2) {
        let (before, after) = (&pair[0], &pair[1]);
        assert!(
            key(before) <= key(after),
            "`{}` is listed before `{}`",
            before.id,
            after.id
        );
    }
    let required: BTreeSet<&str> = catalog
        .nodes
        .iter()
        .filter(|node| node.required_in_graph)
        .map(|node| node.id)
        .collect();
    assert_eq!(
        required,
        BTreeSet::from(["pass.forward", "pass.prepass", "pass.tonemap"])
    );
    let unpassed: Vec<&str> = catalog
        .nodes
        .iter()
        .filter(|node| node.category == Category::Output && node.pass.is_none())
        .map(|node| node.id)
        .collect();
    assert_eq!(unpassed, ["output.display"]);
}

#[test]
fn to_json_has_the_s51_shape() {
    let catalog = Catalog::build();
    let JsonValue::Object(top) = catalog.to_json(&|_: &NodeTypeSpec| true) else {
        panic!("the catalog JSON is not an object")
    };
    let format = JsonValue::String("raa.render-graph-catalog".to_string());
    assert_eq!(field(&top, "format"), &format);
    assert_eq!(field(&top, "version"), &JsonValue::Number(1.0));
    assert_eq!(field(&top, "graphVersion"), &JsonValue::Number(1.0));
    assert_eq!(
        strings(field(&top, "categories")),
        CATEGORIES.split(' ').collect::<Vec<_>>()
    );
    assert_eq!(
        strings(field(&top, "portTypes")),
        PORT_TYPES.split(' ').collect::<Vec<_>>()
    );

    let JsonValue::Array(nodes) = field(&top, "nodes") else {
        panic!("the catalog JSON nodes are not an array")
    };
    assert_eq!(nodes.len(), catalog.nodes.len());
    let keys: Vec<&str> = NODE_KEYS.split(' ').collect();
    for node in nodes {
        let JsonValue::Object(entries) = node else {
            panic!("a catalog node is not an object")
        };
        assert_eq!(
            entries.len(),
            keys.len(),
            "a node has extra or missing keys"
        );
        for key in &keys {
            assert!(
                entries.iter().any(|(name, _)| name == key),
                "a node has no `{key}`"
            );
        }
    }
}

#[test]
fn pass_names_and_default_sources_resolve() {
    let catalog = Catalog::build();
    for node in &catalog.nodes {
        assert_eq!(
            node.pass,
            node.id.strip_prefix("pass."),
            "`{}`: pass != id",
            node.id
        );
        for port in &node.inputs {
            for entry in port.default_source {
                let (type_id, key) = entry
                    .rsplit_once('.')
                    .unwrap_or_else(|| panic!("default source `{entry}` has no `.`"));
                let source = catalog
                    .node(type_id)
                    .unwrap_or_else(|| panic!("`{entry}` names no node type"));
                let has_port = source.outputs.iter().any(|out| out.key == key);
                assert!(has_port, "`{entry}` names no output port of `{type_id}`");
            }
        }
    }
}
