//! prims-check: runs the prims v0 round-trip, refusal and build cases. Each case prints one
//! line `<label> PASS|FAIL <detail>`, then a summary line. Exit 0 when every selected case
//! passes, 1 otherwise, 2 for usage errors (see USAGE).
//!
//! Cases run sequentially in this one process: each takes milliseconds, so spreading them
//! over NMC_CPUS processes would cost more than it saves.

use std::process::ExitCode;

use raa_geom::analysis::{analyze, passes};
use raa_geom::{GeometryRegistry, Quality};
use raa_loader::json::{self, JsonValue};
use raa_loader::prims::samples::{invalid_cases, sample_assets};
use raa_loader::prims::{NodeOp, PrimNode, build, parse, to_canonical_json};
use raa_loader::{LoadContext, NoBooleans};

const USAGE: &str = "usage: prims-check [--only <pattern on label>] [--json <path>]";

/// The `--only` subset: optional `^`, `|` alternatives, at most one `( ... )` group between
/// literal text, optional `$`; matched as a search over the label.
struct Pattern {
    start: bool,
    end: bool,
    alts: Vec<String>,
}

impl Pattern {
    fn parse(s: &str) -> Result<Pattern, String> {
        let (start, rest) = s.strip_prefix('^').map_or((false, s), |r| (true, r));
        let (end, rest) = rest.strip_suffix('$').map_or((false, rest), |r| (true, r));
        let literal = |c: &char| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | ':' | '|');
        if let Some(c) = rest
            .chars()
            .find(|c| !(literal(c) || matches!(c, '(' | ')')))
        {
            return Err(format!(
                "--only {s:?}: unsupported character {c:?} (only ^, ( ), | and $ over literal label text)"
            ));
        }
        let unbalanced = || format!("--only {s:?}: unbalanced or misplaced '(' ')'");
        // One optional group, with literal text (no `|`) before and after it.
        let alts = match rest.split_once('(') {
            None if rest.contains(')') => return Err(unbalanced()),
            None => rest.split('|').map(str::to_string).collect(),
            Some((before, after)) => {
                let (inner, tail) = after.split_once(')').ok_or_else(unbalanced)?;
                let nested = inner.contains('(') || tail.contains(['(', ')']);
                if nested || before.contains('|') || tail.contains('|') {
                    return Err(unbalanced());
                }
                inner
                    .split('|')
                    .map(|a| format!("{before}{a}{tail}"))
                    .collect()
            }
        };
        Ok(Pattern { start, end, alts })
    }

    fn matches(&self, label: &str) -> bool {
        self.alts.iter().any(|a| match (self.start, self.end) {
            (true, true) => label == a,
            (true, false) => label.starts_with(a.as_str()),
            (false, true) => label.ends_with(a.as_str()),
            (false, false) => label.contains(a.as_str()),
        })
    }
}

enum Case {
    Roundtrip(String),
    /// (sample name, what is non-canonical, text).
    NonCanonical(&'static str, &'static str, String),
    Refusal(String, &'static str),
    Build(String),
}

struct Outcome {
    label: String,
    group: &'static str,
    ok: bool,
    detail: String,
    /// The optional JSON fields of this group.
    extra: Vec<(&'static str, JsonValue)>,
}

/// Hand-written non-canonical spellings of the `crate` sample.
fn noncanonical() -> Vec<(&'static str, &'static str, String)> {
    const SLAT_R: &str = "[0,0,0.7071067811865476,0.7071067811865476]";
    let compact = |num1: &str, zero: &str, half: &str| {
        format!(
            r#"{{"format":"raa.prims","version":{zero},"id":"crate","name":"Wooden crate","nodes":[{{"id":"body","kind":"roundBox","params":{{"halfExtents":[{half},{half},{half}],"radius":0.05}},"material":"wood","children":[{{"id":"plank-top","kind":"box","params":{{"halfExtents":[0.52,0.05,0.08]}},"transform":{{"t":[{zero},{half},{zero}],"r":[{zero},{zero},{zero},{num1}],"s":[{num1},{num1},{num1}]}},"material":"plank"}},{{"id":"plank-bottom","kind":"box","params":{{"halfExtents":[0.52,0.05,0.08]}},"transform":{{"t":[{zero},-{half},{zero}],"r":[{zero},{zero},{zero},{num1}],"s":[{num1},{num1},{num1}]}},"material":"plank"}},{{"id":"slat","kind":"box","params":{{"halfExtents":[0.52,0.05,0.05]}},"transform":{{"t":[{zero},{zero},0.52],"r":{SLAT_R},"s":[{num1},{num1},{num1}]}},"material":"plank"}}]}}],"materials":{{"plank":{{"color":[0.45,0.3,0.15],"roughness":0.85,"metallic":{zero},"emissive":[{zero},{zero},{zero}]}},"wood":{{"color":[0.65,0.45,0.25],"roughness":0.8,"metallic":{zero},"emissive":[{zero},{zero},{zero}]}}}},"anchors":{{}}}}"#
        )
    };
    let reordered = format!(
        r#"{{
 "anchors": {{}},
 "materials": {{
  "wood": {{"emissive": [0, 0, 0], "metallic": 0, "roughness": 0.8, "color": [0.65, 0.45, 0.25]}},
  "plank": {{"emissive": [0, 0, 0], "metallic": 0, "roughness": 0.85, "color": [0.45, 0.3, 0.15]}}
 }},
 "nodes": [{{
  "children": [
   {{"material": "plank", "transform": {{"s": [1, 1, 1], "r": [0, 0, 0, 1], "t": [0, 0.5, 0]}},
    "params": {{"halfExtents": [0.52, 0.05, 0.08]}}, "kind": "box", "id": "plank-top"}},
   {{"material": "plank", "transform": {{"s": [1, 1, 1], "r": [0, 0, 0, 1], "t": [0, -0.5, 0]}},
    "params": {{"halfExtents": [0.52, 0.05, 0.08]}}, "kind": "box", "id": "plank-bottom"}},
   {{"material": "plank", "transform": {{"s": [1, 1, 1], "r": {SLAT_R}, "t": [0, 0, 0.52]}},
    "params": {{"halfExtents": [0.52, 0.05, 0.05]}}, "kind": "box", "id": "slat"}}
  ],
  "material": "wood",
  "params": {{"radius": 0.05, "halfExtents": [0.5, 0.5, 0.5]}},
  "kind": "roundBox",
  "id": "body"
 }}],
 "name": "Wooden crate",
 "id": "crate",
 "version": 0,
 "format": "raa.prims"
}}"#
    );
    let base = compact("1", "0", "0.5");
    // Every node except body is a leaf, so each gains the key just before its closing brace.
    let explicit_union = base.replace(
        r#""material":"plank"}"#,
        r#""material":"plank","op":"union"}"#,
    );
    let empty_children = base.replace(
        r#""material":"plank"}"#,
        r#""material":"plank","children":[]}"#,
    );
    let unsorted_materials = base.replace(
        r#""materials":{"plank":{"color":[0.45,0.3,0.15],"roughness":0.85,"metallic":0,"emissive":[0,0,0]},"wood":{"color":[0.65,0.45,0.25],"roughness":0.8,"metallic":0,"emissive":[0,0,0]}}"#,
        r#""materials":{"wood":{"color":[0.65,0.45,0.25],"roughness":0.8,"metallic":0,"emissive":[0,0,0]},"plank":{"color":[0.45,0.3,0.15],"roughness":0.85,"metallic":0,"emissive":[0,0,0]}}"#,
    );
    vec![
        ("crate", "reordered keys", reordered),
        ("crate", "compact whitespace", base.clone()),
        ("crate", "1.0 spelling", compact("1.0", "0.0", "0.50")),
        ("crate", "1E0 spelling", compact("1E0", "0E0", "5E-1")),
        ("crate", "-0 spelling", compact("1", "-0", "0.5")),
        ("crate", "explicit union op", explicit_union),
        ("crate", "empty children", empty_children),
        ("crate", "unsorted materials", unsorted_materials),
    ]
}

fn outcome(label: &str, group: &'static str, ok: bool, detail: String) -> Outcome {
    let label = label.to_string();
    let extra = Vec::new();
    Outcome {
        label,
        group,
        ok,
        detail,
        extra,
    }
}

/// Byte offset of the first difference between two texts.
fn first_diff(a: &str, b: &str) -> usize {
    let same = a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count();
    same.min(a.len().max(b.len()))
}

fn canon(text: &str) -> Result<String, String> {
    parse(text)
        .map(|a| to_canonical_json(&a))
        .map_err(|e| e.to_string())
}

fn run_roundtrip(label: &str, text: &str) -> Outcome {
    let (ok, detail) = match canon(text) {
        Err(e) => (false, format!("parse refused: {e}")),
        Ok(out) if out == text => (true, format!("{} bytes identical", text.len())),
        Ok(out) => (false, format!("differs at byte {}", first_diff(&out, text))),
    };
    outcome(label, "roundtrip", ok, detail)
}

fn run_noncanonical(label: &str, sample: &str, what: &str, text: &str, canonical: &str) -> Outcome {
    let (ok, detail) = match canon(text).and_then(|once| canon(&once).map(|twice| (once, twice))) {
        Err(e) => (false, format!("{what}: parse refused: {e}")),
        Ok(_) if text == canonical => (false, format!("{what}: input is already canonical")),
        Ok((once, _)) if once != canonical => {
            let at = first_diff(&once, canonical);
            (
                false,
                format!("{what}: canonical text of {sample} differs at byte {at}"),
            )
        }
        Ok((once, twice)) if twice != once => (false, format!("{what}: second pass differs")),
        Ok(_) => (
            true,
            format!("{what} -> {sample} canonical, second pass identical"),
        ),
    };
    outcome(label, "roundtrip", ok, detail)
}

fn run_refusal(label: &str, text: &str, expected: &str, ctx: &LoadContext<'_>) -> Outcome {
    let result = parse(text).and_then(|a| build(&a, ctx));
    let (ok, got, detail) = match result {
        Ok(_) => (false, None, format!("accepted, expected {expected}")),
        Err(e) => {
            let ok = e.code == expected && !e.message.is_empty();
            let empty = if e.message.is_empty() {
                " (empty message)"
            } else {
                ""
            };
            let detail = format!("got {} expected {expected}{empty}: {e}", e.code);
            (ok, Some(e.code), detail)
        }
    };
    let mut o = outcome(label, "refusal", ok, detail);
    let code = got.map_or(JsonValue::Null, |c| JsonValue::String(c.into()));
    o.extra = vec![
        ("code", code),
        ("expected", JsonValue::String(expected.into())),
    ];
    o
}

/// Ids of the nodes that become meshes (roots and union children), in pre-order.
fn union_ids(nodes: &[PrimNode], roots: bool, out: &mut Vec<String>) {
    for n in nodes {
        if roots || n.op == NodeOp::Union {
            out.push(n.id.clone());
        }
        union_ids(&n.children, false, out);
    }
}

fn run_build(label: &str, text: &str, ctx: &LoadContext<'_>) -> Outcome {
    let asset = match parse(text).and_then(|a| build(&a, ctx).map(|l| (a, l))) {
        Ok(x) => x,
        Err(e) => return outcome(label, "build", false, format!("refused: {e}")),
    };
    let (doc, loaded) = asset;
    let mut unions = Vec::new();
    union_ids(&doc.nodes, true, &mut unions);
    let mut fails = Vec::new();
    for m in &loaded.meshes {
        if let Err(why) = passes(&analyze(&m.mesh)) {
            fails.push(format!("mesh {} not closed/valid: {why}", m.node_id));
        }
    }
    let mesh_ids: Vec<&str> = loaded.meshes.iter().map(|m| m.node_id.as_str()).collect();
    if mesh_ids != unions.iter().map(String::as_str).collect::<Vec<_>>() {
        fails.push(format!("meshes {mesh_ids:?} != union nodes {unions:?}"));
    }
    let sum: usize = loaded.meshes.iter().map(|m| m.mesh.triangle_count()).sum();
    if sum != loaded.triangles as usize {
        fails.push(format!(
            "mesh triangles {sum} != reported {}",
            loaded.triangles
        ));
    }
    if loaded.triangles > loaded.budget {
        fails.push(format!(
            "{} triangles > budget {}",
            loaded.triangles, loaded.budget
        ));
    }
    let collider_ids: Vec<Option<&str>> = loaded
        .colliders
        .iter()
        .map(|c| c.node_id.as_deref())
        .collect();
    let source = match &doc.collision {
        Some(docs) => {
            let want: Vec<Option<&str>> = docs.iter().map(|d| d.node.as_deref()).collect();
            if docs.is_empty() || collider_ids != want {
                fails.push(format!("colliders {collider_ids:?} != override {want:?}"));
            }
            "override"
        }
        None => {
            let want: Vec<Option<&str>> = unions.iter().map(|s| Some(s.as_str())).collect();
            if collider_ids != want {
                fails.push(format!(
                    "colliders {collider_ids:?} != derived per union node"
                ));
            }
            "derived"
        }
    };
    let (tris, budget, meshes, n) = (
        loaded.triangles,
        loaded.budget,
        loaded.meshes.len(),
        unions.len(),
    );
    let colliders = loaded.colliders.len();
    let mut detail =
        format!("triangles={tris}/{budget} meshes={meshes}/{n} colliders={colliders} {source}");
    if !fails.is_empty() {
        detail = format!("{detail}: {}", fails.join("; "));
    }
    let mut o = outcome(label, "build", fails.is_empty(), detail);
    let num = |x: usize| JsonValue::Number(x as f64);
    o.extra = vec![
        ("triangles", num(tris as usize)),
        ("budget", num(budget as usize)),
        ("meshes", num(meshes)),
        ("unionNodes", num(n)),
        ("colliders", num(colliders)),
    ];
    o
}

fn json_table(outcomes: &[Outcome]) -> String {
    let rows = outcomes
        .iter()
        .map(|o| {
            let mut m = vec![
                ("label".to_string(), JsonValue::String(o.label.clone())),
                ("group".to_string(), JsonValue::String(o.group.into())),
                ("ok".to_string(), JsonValue::Bool(o.ok)),
                ("detail".to_string(), JsonValue::String(o.detail.clone())),
            ];
            m.extend(o.extra.iter().map(|(k, v)| (k.to_string(), v.clone())));
            JsonValue::Object(m)
        })
        .collect();
    let mut s = String::new();
    json::write_pretty(&JsonValue::Array(rows), &mut s);
    s.push('\n');
    s
}

fn main() -> ExitCode {
    run().unwrap_or_else(|(code, m)| {
        eprintln!("prims-check: {m}");
        ExitCode::from(code)
    })
}

/// Runs the cases; an `Err` carries the exit code (2 usage, 1 failure) and the message.
fn run() -> Result<ExitCode, (u8, String)> {
    let usage = |m: String| (2, format!("{m}\n{USAGE}"));
    let (mut only, mut json) = (None::<Pattern>, None::<String>);
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let value = it
            .next()
            .ok_or_else(|| usage(format!("{flag} needs a value")))?;
        match flag.as_str() {
            "--only" => only = Some(Pattern::parse(&value).map_err(usage)?),
            "--json" => json = Some(value),
            _ => return Err(usage(format!("unknown argument {flag:?}"))),
        }
    }

    let samples = sample_assets();
    let mut cases: Vec<(String, Case)> = Vec::new();
    for (name, text) in &samples {
        cases.push((format!("roundtrip:{name}"), Case::Roundtrip(text.clone())));
    }
    for (i, (sample, what, text)) in noncanonical().into_iter().enumerate() {
        let label = format!("roundtrip:noncanonical-{}", i + 1);
        cases.push((label, Case::NonCanonical(sample, what, text)));
    }
    for (label, text, code) in invalid_cases() {
        cases.push((format!("refusal:{label}"), Case::Refusal(text, code)));
    }
    for (name, text) in &samples {
        cases.push((format!("build:{name}"), Case::Build(text.clone())));
    }
    cases.retain(|(label, _)| only.as_ref().is_none_or(|p| p.matches(label)));
    if cases.is_empty() {
        return Err(usage("--only matches no case".to_string()));
    }

    let mut geometry = GeometryRegistry::new();
    raa_prims::register_all(&mut geometry).map_err(|e| (1, format!("register_all: {e}")))?;
    let ctx = LoadContext {
        geometry: &geometry,
        booleans: &NoBooleans,
        quality: Quality::LOD0,
        strict_kinds: true,
    };
    let canonical_of = |name: &str| {
        samples
            .iter()
            .find(|(n, _)| *n == name)
            .map_or("", |(_, t)| t.as_str())
    };

    let mut outcomes = Vec::new();
    for (label, case) in &cases {
        let o = match case {
            Case::Roundtrip(text) => run_roundtrip(label, text),
            Case::NonCanonical(sample, what, text) => {
                run_noncanonical(label, sample, what, text, canonical_of(sample))
            }
            Case::Refusal(text, code) => run_refusal(label, text, code, &ctx),
            Case::Build(text) => run_build(label, text, &ctx),
        };
        let verdict = if o.ok { "PASS" } else { "FAIL" };
        println!("{} {verdict} {}", o.label, o.detail);
        outcomes.push(o);
    }

    let tally = |group: &str| {
        let of_group = outcomes.iter().filter(|o| o.group == group);
        let total = of_group.clone().count();
        format!("{group} {}/{total}", of_group.filter(|o| o.ok).count())
    };
    println!(
        "{} {} {}",
        tally("roundtrip"),
        tally("refusal"),
        tally("build")
    );
    if let Some(path) = &json {
        if let Some(dir) = std::path::Path::new(path).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        std::fs::write(path, json_table(&outcomes))
            .map_err(|e| (1, format!("cannot write {path}: {e}")))?;
    }
    let failed = outcomes.iter().any(|o| !o.ok);
    Ok(ExitCode::from(u8::from(failed)))
}
