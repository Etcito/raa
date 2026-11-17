//! The no-graph export (S54, contract `rg-export-snapshot`): the graph document that describes a
//! `SettingsSnapshot`. `GraphExecutor::export_current` reads the snapshot from the handles and calls
//! [`export_snapshot`]; the export itself needs no device.

use crate::catalog::{Catalog, FeatureId, NodeTypeSpec, ParamBinding, ParamSpec, ParamValue};
use crate::format::{GraphDoc, LinkDoc, NodeDoc, PortRef};

use super::SettingsSnapshot;

/// The name every export carries.
const NAME: &str = "Engine default";
/// The id of the output node every export ends with.
const OUTPUT_ID: &str = "out";
/// The entry after which the contributors are placed, and the last colour-chain entry.
const HOST: &str = "forward";
const LAST_CHAIN: &str = "tonemap";

/// Handoff: `export_snapshot` (crates/raa-graph/src/exec/export.rs:28) is the entry point of this
/// file. Today it is a pure function of its arguments, with no device and no handles: it writes the
/// document named "Engine default", one node per enabled catalog pass entry (`pass.<entry>`, with the
/// contributors after `forward`), each parameter at its snapshot value or catalog default, and the
/// colour chain from `forward` to `tonemap`, then `out`. The downstream card 'raa-graph starters:
/// Simple forward and Full GI' takes it over: its test writes the Full GI starter as
/// `write_graph(&export_snapshot(catalog, snapshot, msaa))`.
///
/// The render-graph document of the snapshot (S54): one node per enabled pass entry that has a
/// catalog node type, in execution order, then the `output.display` node `out`.
pub fn export_snapshot(catalog: &Catalog, snapshot: &SettingsSnapshot, msaa: u32) -> GraphDoc {
    let entries = emitted(catalog, snapshot);
    let mut nodes: Vec<NodeDoc> = entries
        .iter()
        .map(|(entry, spec)| node_doc(entry, spec, snapshot, msaa))
        .collect();
    nodes.push(NodeDoc {
        id: OUTPUT_ID.to_owned(),
        type_id: "output.display".to_owned(),
        version: None,
        params: Vec::new(),
        compute: None,
    });
    let links = colour_links(&colour_chain(&entries));
    GraphDoc {
        name: NAME.to_owned(),
        nodes,
        links,
        editor: None,
    }
}

/// The entries the export writes, in execution order: the enabled pass-role entries in
/// `snapshot.order` order, then the enabled contributors right after `forward`, by default order.
/// An entry without a catalog node type is skipped.
fn emitted<'a>(
    catalog: &'a Catalog,
    snapshot: &'a SettingsSnapshot,
) -> Vec<(&'a str, &'a NodeTypeSpec)> {
    let mut passes = Vec::new();
    let mut contributors = Vec::new();
    for (name, on) in &snapshot.passes {
        if !*on {
            continue;
        }
        let Some(spec) = entry_spec(catalog, name) else {
            continue;
        };
        if spec.contributor {
            contributors.push((name.as_str(), spec));
        } else {
            passes.push((name.as_str(), spec));
        }
    }
    passes.sort_by_key(|(name, _)| {
        snapshot
            .order
            .iter()
            .position(|listed| listed.as_str() == *name)
            .unwrap_or(usize::MAX)
    });
    contributors.sort_by_key(|(_, spec)| spec.default_order);
    let at = passes
        .iter()
        .position(|(name, _)| *name == HOST)
        .map_or(passes.len(), |index| index + 1);
    passes.splice(at..at, contributors);
    passes
}

/// The catalog node type of the pipeline entry `entry`.
fn entry_spec<'a>(catalog: &'a Catalog, entry: &str) -> Option<&'a NodeTypeSpec> {
    catalog.node(&format!("pass.{entry}"))
}

/// The node ids of the colour chain: `forward`, then every entry up to `tonemap` whose type has a
/// `color` input and a `color` output (contributors and `forward`'s missing input excluded).
fn colour_chain(entries: &[(&str, &NodeTypeSpec)]) -> Vec<String> {
    let mut chain = Vec::new();
    for (entry, spec) in entries {
        if chain.is_empty() {
            if *entry == HOST {
                chain.push(node_id(entry));
            }
            continue;
        }
        if has_colour_io(spec) {
            chain.push(node_id(entry));
        }
        if *entry == LAST_CHAIN {
            break;
        }
    }
    chain
}

fn has_colour_io(spec: &NodeTypeSpec) -> bool {
    spec.inputs.iter().any(|port| port.key == "color")
        && spec.outputs.iter().any(|port| port.key == "color")
}

/// The links between consecutive chain nodes, then `tonemap.color -> out.color` when the chain ends
/// at `tonemap`.
fn colour_links(chain: &[String]) -> Vec<LinkDoc> {
    let mut links: Vec<LinkDoc> = chain
        .windows(2)
        .map(|pair| link(&pair[0], &pair[1]))
        .collect();
    if let Some(last) = chain.last().filter(|last| last.as_str() == LAST_CHAIN) {
        links.push(link(last, OUTPUT_ID));
    }
    links
}

fn link(from: &str, to: &str) -> LinkDoc {
    LinkDoc {
        from: colour(from),
        to: colour(to),
        history: false,
    }
}

fn colour(node: &str) -> PortRef {
    PortRef {
        node: node.to_owned(),
        port: "color".to_owned(),
    }
}

fn node_doc(entry: &str, spec: &NodeTypeSpec, snapshot: &SettingsSnapshot, msaa: u32) -> NodeDoc {
    NodeDoc {
        id: node_id(entry),
        type_id: spec.id.to_owned(),
        version: None,
        params: spec
            .params
            .iter()
            .map(|param| {
                (
                    param.key.to_owned(),
                    param_value(param, snapshot, msaa).to_json(),
                )
            })
            .collect(),
        compute: None,
    }
}

/// The value an export writes for `param`: the snapshot value of a Setting key the snapshot holds,
/// `msaa` for the Engine `msaa` option, and the catalog default otherwise.
fn param_value(param: &ParamSpec, snapshot: &SettingsSnapshot, msaa: u32) -> ParamValue {
    match &param.binding {
        ParamBinding::Setting { feature, key } => {
            snapshot_value(snapshot, feature, key).unwrap_or_else(|| param.default.clone())
        }
        ParamBinding::Engine { option: "msaa" } => ParamValue::Int(i64::from(msaa)),
        ParamBinding::Engine { .. } | ParamBinding::Tunable { .. } | ParamBinding::Node => {
            param.default.clone()
        }
    }
}

/// The snapshot value of `key` in `feature`, `None` when the snapshot has no entry for either.
fn snapshot_value(
    snapshot: &SettingsSnapshot,
    feature: &FeatureId,
    key: &str,
) -> Option<ParamValue> {
    let (_, values) = snapshot
        .values
        .iter()
        .find(|(candidate, _)| candidate == feature)?;
    values
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value.clone())
}

/// The lower camel case node id of a pass entry: the first part keeps its text and every later part
/// (split at `.` and `-`) is capitalised, so `ao.gtao` is `aoGtao` and `sky.cloud-lighting` is
/// `skyCloudLighting`.
fn node_id(entry: &str) -> String {
    let mut parts = entry.split(['.', '-']).filter(|part| !part.is_empty());
    let mut id = parts.next().unwrap_or_default().to_owned();
    for part in parts {
        let mut chars = part.chars();
        if let Some(first) = chars.next() {
            id.extend(first.to_uppercase());
            id.push_str(chars.as_str());
        }
    }
    id
}
