//! The tunables triage table `docs/render-graph/tunables.md` (contract `rg-m4-triage-format`) pinned
//! against `Catalog::build()` (S56), the feature crates' `tunables()` specs and the pinned rows of
//! contract `rg-m4-pinned-tunables`. Reads the table, the catalog and those specs only.

use raa_graph::Catalog;
use raa_graph::catalog::{FeatureId, ParamBinding, ParamKind, ParamSpec, ParamValue};
use raa_pipeline::{TunableSpec, find_tunable};

const TABLE: &str = include_str!("../../../docs/render-graph/tunables.md");

const HEADER: &str = "| crate | file:line | name | value | row | class | node type | key | group | label | integer | mechanism | rebuild | range | reason |";

/// The pinned rows of contract `rg-m4-pinned-tunables`: node type, key, value, row, group, label,
/// integer, mechanism, rebuild and range, separated by `|`.
const PINNED: &[&str] = &[
    "pass.taa|filterSharpness|2.3|setting|Resolve|Filter sharpness|no|override|pipeline|unstated",
    "pass.taa|clipTolerance|1.25|setting|Resolve|Clip tolerance|no|override|pipeline|unstated",
    "pass.taa|blendBase|0.06|setting|Resolve|Blend base|no|override|pipeline|unstated",
    "pass.taa|blendMotionGain|0.012|setting|Resolve|Blend motion gain|no|override|pipeline|unstated",
    "pass.taa|blendMin|0.04|setting|Resolve|Blend min|no|override|pipeline|unstated",
    "pass.taa|blendMax|0.3|setting|Resolve|Blend max|no|override|pipeline|unstated",
    "pass.ssr|maxSteps|64u|wgsl|Trace|Max steps|yes|override|pipeline|unstated",
    "pass.ao.gtao|slices|4|setting|Quality|Slices|yes|uniform|none|1..8",
    "pass.ao.gtao|steps|8|setting|Quality|Steps|yes|uniform|none|1..16",
    "pass.gi.screen|probeScreenDistance|8.0|wgsl|Screen probes|Probe screen distance|no|override|pipeline|unstated",
    "pass.gi.screen|probeScreenSteps|32u|wgsl|Screen probes|Probe screen steps|yes|override|pipeline|unstated",
    "pass.gi.screen|probeScreenThickness|0.15|wgsl|Screen probes|Probe screen thickness|no|override|pipeline|unstated",
    "pass.gi.area-shadow|temporalAlpha|0.1|wgsl|Temporal|Temporal alpha|no|override|pipeline|unstated",
    "pass.gi.ssgi|slicesLow|2|setting|Quality table|Slices (low)|yes|uniform|none|unstated",
    "pass.gi.ssgi|slicesMedium|4|setting|Quality table|Slices (medium)|yes|uniform|none|unstated",
    "pass.gi.ssgi|slicesHigh|6|setting|Quality table|Slices (high)|yes|uniform|none|unstated",
    "pass.gi.ssgi|stepsLow|6|setting|Quality table|Steps (low)|yes|uniform|none|unstated",
    "pass.gi.ssgi|stepsMedium|8|setting|Quality table|Steps (medium)|yes|uniform|none|unstated",
    "pass.gi.ssgi|stepsHigh|12|setting|Quality table|Steps (high)|yes|uniform|none|unstated",
    "pass.sky.atmosphere|skyViewSteps|32|wgsl|Atmosphere|Sky-view steps|yes|override|pipeline|unstated",
    "pass.sky.cloud-lighting|cloudAlbedo|0.98|wgsl|Cloud volume|Cloud albedo|no|override|pipeline|unstated",
];

/// One data row of the triage table, cell by cell.
struct Row {
    krate: &'static str,
    location: &'static str,
    name: &'static str,
    value: &'static str,
    kind: &'static str,
    class: &'static str,
    node: &'static str,
    key: &'static str,
    group: &'static str,
    label: &'static str,
    integer: &'static str,
    mechanism: &'static str,
    rebuild: &'static str,
    range: &'static str,
    reason: &'static str,
}

/// The data rows under the one header row. A cell never holds `|`, so every row has 15 cells.
fn rows() -> Vec<Row> {
    assert_eq!(TABLE.matches("| crate |").count(), 1, "one table");
    let lines: Vec<&str> = TABLE.lines().collect();
    let head = lines
        .iter()
        .position(|line| *line == HEADER)
        .expect("the contract header row");
    assert!(
        lines[head + 1].starts_with("|---"),
        "the alignment row follows the header"
    );
    lines[head + 2..]
        .iter()
        .take_while(|line| line.starts_with('|'))
        .map(|line| {
            let inner = line.trim_matches('|');
            let cells: Vec<&str> = inner.split('|').map(str::trim).collect();
            let cells: [&str; 15] = cells
                .try_into()
                .unwrap_or_else(|cells: Vec<&str>| panic!("{} cells in: {line}", cells.len()));
            let [
                krate,
                location,
                name,
                value,
                kind,
                class,
                node,
                key,
                group,
                label,
                integer,
                mechanism,
                rebuild,
                range,
                reason,
            ] = cells;
            Row {
                krate,
                location,
                name,
                value,
                kind,
                class,
                node,
                key,
                group,
                label,
                integer,
                mechanism,
                rebuild,
                range,
                reason,
            }
        })
        .collect()
}

/// The count the header paragraph prints (`Count on this tree: **265** lines`).
fn header_count() -> usize {
    let line = TABLE
        .lines()
        .find(|line| line.contains("Count on this tree:"))
        .expect("the count line");
    line.split("**")
        .nth(1)
        .and_then(|count| count.parse().ok())
        .expect("the count in bold")
}

/// Every catalog parameter with the id of its node type.
fn params(catalog: &Catalog) -> Vec<(&'static str, &ParamSpec)> {
    catalog
        .nodes
        .iter()
        .flat_map(|node| node.params.iter().map(move |param| (node.id, param)))
        .collect()
}

/// The default a row's value cell names: the value without its `u`, `i` or `f` suffix, read as an
/// integer for an `integer` row and as a decimal number otherwise.
fn row_default(row: &Row) -> Option<ParamValue> {
    let text = row.value.strip_suffix(['u', 'i', 'f']).unwrap_or(row.value);
    if row.integer == "yes" {
        text.parse().ok().map(ParamValue::Int)
    } else {
        text.parse().ok().map(ParamValue::Number)
    }
}

/// The inclusive bounds of a range cell; `unstated` and an empty side give `None`.
fn row_range(row: &Row) -> (Option<f64>, Option<f64>) {
    match row.range.split_once("..") {
        Some((low, high)) => (low.parse().ok(), high.parse().ok()),
        None => (None, None),
    }
}

fn number(value: &ParamValue) -> f64 {
    match value {
        ParamValue::Int(int) => *int as f64,
        ParamValue::Number(n) => *n,
        _ => f64::NAN,
    }
}

/// The feature a tunable crate binds, and the `tunables()` specs of that crate.
fn feature_of(krate: &str) -> (FeatureId, &'static [TunableSpec]) {
    match krate {
        "raa-taa" => (FeatureId::Taa, raa_taa::tunables()),
        "raa-gtao" => (FeatureId::Gtao, raa_gtao::tunables()),
        "raa-contact" => (FeatureId::Contact, raa_contact::tunables()),
        "raa-ssr" => (FeatureId::Ssr, raa_ssr::tunables()),
        "raa-dof" => (FeatureId::Dof, raa_dof::tunables()),
        "raa-shafts" => (FeatureId::Shafts, raa_shafts::tunables()),
        "raa-gi" => (FeatureId::Gi, raa_gi::tunables()),
        "raa-sky" => (FeatureId::Sky, raa_sky::tunables()),
        other => panic!("no tunables for crate {other}"),
    }
}

/// A row of the pinned contract, projected to the same ten cells.
fn pinned_view(row: &Row) -> String {
    [
        row.node,
        row.key,
        row.value,
        row.kind,
        row.group,
        row.label,
        row.integer,
        row.mechanism,
        row.rebuild,
        row.range,
    ]
    .join("|")
}

#[test]
fn the_table_has_the_contract_format() {
    for row in rows() {
        assert!(
            matches!(row.kind, "wgsl" | "setting"),
            "row cell of {}: {}",
            row.name,
            row.kind
        );
        assert!(
            matches!(row.class, "tunable" | "maths" | "layout"),
            "class of {}: {}",
            row.name,
            row.class
        );
        assert!(!row.reason.is_empty(), "a reason for {}", row.name);
    }
    let families = TABLE
        .split("## Families without tunables")
        .nth(1)
        .expect("the families section");
    for family in ["shadow", "fog", "bloom", "grading", "tonemap", "oit"] {
        assert!(
            families.contains(&format!("- {family}:")),
            "{family} is listed without tunables"
        );
    }
}

#[test]
fn every_tunable_row_maps_to_exactly_one_catalog_param() {
    let catalog = Catalog::build();
    let all = params(&catalog);
    let mut bad = Vec::new();
    for row in rows().iter().filter(|row| row.class == "tunable") {
        let found: Vec<&ParamSpec> = all
            .iter()
            .filter(|(node, param)| *node == row.node && param.key == row.key)
            .map(|(_, param)| *param)
            .collect();
        let &[param] = found.as_slice() else {
            bad.push(format!(
                "{} {}: {} catalog params",
                row.node,
                row.key,
                found.len()
            ));
            continue;
        };
        let ParamBinding::Tunable {
            feature,
            key: binding,
        } = &param.binding
        else {
            bad.push(format!("{} {}: binding is not Tunable", row.node, row.key));
            continue;
        };
        let (crate_feature, specs) = feature_of(row.krate);
        let spec = find_tunable(specs, binding);
        let range = row_range(row);
        let checks = [
            (*feature == crate_feature, "binding feature"),
            (row_default(row) == Some(param.default.clone()), "default"),
            (
                (row.integer == "yes") == (param.kind == ParamKind::Int),
                "integer",
            ),
            (param.group == row.group, "group"),
            (param.label == row.label, "label"),
            (
                format!("{:?}", param.rebuild).to_lowercase() == row.rebuild,
                "rebuild",
            ),
            (
                matches!(
                    (row.rebuild, row.mechanism),
                    ("none", "uniform") | ("pipeline", "override" | "define")
                ),
                "mechanism",
            ),
            (param.source == row.location, "source"),
            (
                param.unstated_range == (row.range == "unstated")
                    && (param.min, param.max) == range,
                "range",
            ),
            (
                spec.is_some_and(|spec| {
                    spec.default == number(&param.default)
                        && spec.integer == (row.integer == "yes")
                        && spec.rebuild == row.rebuild
                        && (spec.min, spec.max) == range
                        && spec.source == row.location
                }),
                "TunableSpec of the binding",
            ),
        ];
        let failed: Vec<&str> = checks
            .into_iter()
            .filter(|(ok, _)| !ok)
            .map(|(_, what)| what)
            .collect();
        if !failed.is_empty() {
            bad.push(format!(
                "{} {} [{}]: {}",
                row.node,
                row.key,
                row.location,
                failed.join(", ")
            ));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

#[test]
fn maths_and_layout_rows_map_to_no_catalog_param() {
    let catalog = Catalog::build();
    let sources: Vec<&str> = params(&catalog)
        .iter()
        .map(|(_, param)| param.source)
        .collect();
    let mut bad = Vec::new();
    for row in rows().iter().filter(|row| row.class != "tunable") {
        let cells = [
            row.node,
            row.key,
            row.group,
            row.label,
            row.integer,
            row.mechanism,
            row.rebuild,
            row.range,
        ];
        if !matches!(row.class, "maths" | "layout") || cells.iter().any(|cell| *cell != "-") {
            bad.push(format!(
                "{}: the non-tunable row has a setting cell",
                row.name
            ));
        }
        if sources.contains(&row.location) {
            bad.push(format!(
                "{}: a catalog param is declared at {}",
                row.name, row.location
            ));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

#[test]
fn every_tunable_catalog_param_has_one_tunable_row() {
    let catalog = Catalog::build();
    let table = rows();
    let mut bad = Vec::new();
    for (node, param) in params(&catalog) {
        if !matches!(param.binding, ParamBinding::Tunable { .. }) {
            continue;
        }
        let count = table
            .iter()
            .filter(|row| row.class == "tunable" && row.node == node && row.key == param.key)
            .count();
        if count != 1 {
            bad.push(format!("{node} {}: {count} tunable rows", param.key));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

#[test]
fn the_pinned_rows_of_the_contract_are_present() {
    let table = rows();
    let missing: Vec<&str> = PINNED
        .iter()
        .copied()
        .filter(|pin| !table.iter().any(|row| pinned_view(row) == *pin))
        .collect();
    assert!(missing.is_empty(), "pinned rows missing: {missing:?}");
}

#[test]
fn the_wgsl_rows_equal_the_count_in_the_table_header() {
    let wgsl = rows().iter().filter(|row| row.kind == "wgsl").count();
    assert_eq!(wgsl, header_count());
}
