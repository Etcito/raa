//! csg-cases: loads every CSG case through `PrimsLoader` + `CsgResolver` at LOD0 and checks
//! its outcome. Each case prints one line with the S9 fields, in table order; `--json` writes
//! the S9 array. For ok cases the volume identity is computed with the crate functions on
//! meshes rebuilt from the parsed asset. Exit 0 when every selected case matches its
//! expectation, 1 otherwise, 2 for usage errors (see USAGE).
//!
//! Cases run in `std::thread::scope` threads, at most NMC_CPUS (1 when unset or invalid).

use std::process::ExitCode;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use raa_csg::subtract_convex;
use raa_csg::{CsgCase, CsgError, CsgResolver, cases, convex_planes, intersect_convex};
use raa_geom::analysis::analyze;
use raa_geom::{GeometryRegistry, MeshData, ParamValue, Params, Quality};
use raa_loader::json::{self, JsonValue};
use raa_loader::prims::{NodeOp, PrimNode, PrimsLoader, TransformDoc, parse};
use raa_loader::{AssetLoader, LoadContext};
use raa_math::{Mat4, Quat, Vec3};

const USAGE: &str = "usage: csg-cases [--only <pattern on case id>] [--json <path>]";

/// CSG_EPS rule: volume identity relative difference.
const ID_TOL: f64 = 1e-4;

/// The `--only` subset: optional `^`, `|` alternatives, at most one `( ... )` group between
/// literal text, optional `$`; matched as a search over the case id.
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
                "--only {s:?}: unsupported character {c:?} (only ^, ( ), | and $ over literal id text)"
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

/// One S9 row plus the verdict.
struct Outcome {
    id: &'static str,
    expected: &'static str,
    op: &'static str,
    host: JsonValue,
    tools: JsonValue,
    ok: bool,
    triangles: u32,
    watertight: bool,
    volume: f64,
    id_err: Option<f64>,
    code: Option<&'static str>,
    message: Option<String>,
    /// Why the case fails (empty when it passes).
    fails: Vec<String>,
}

fn transform_json(t: Option<&TransformDoc>) -> JsonValue {
    let nums = |v: &[f64]| JsonValue::Array(v.iter().map(|x| JsonValue::Number(*x)).collect());
    t.map_or(JsonValue::Null, |t| {
        JsonValue::Object(vec![
            ("t".to_string(), nums(&t.t)),
            ("r".to_string(), nums(&t.r)),
            ("s".to_string(), nums(&t.s)),
        ])
    })
}

fn params_json(n: &PrimNode) -> JsonValue {
    JsonValue::Object(n.params.clone())
}

fn op_name(op: NodeOp) -> &'static str {
    match op {
        NodeOp::Union => "union",
        NodeOp::Subtract => "subtract",
        NodeOp::Intersect => "intersect",
    }
}

/// The boolean tools of a host (subtract/intersect children), in document order.
fn tools_of(host: &PrimNode) -> Vec<&PrimNode> {
    host.children
        .iter()
        .filter(|c| c.op != NodeOp::Union)
        .collect()
}

/// "subtract", "intersect" or "intersect+subtract".
fn case_op(tools: &[&PrimNode]) -> &'static str {
    let int = tools.iter().any(|t| t.op == NodeOp::Intersect);
    let sub = tools.iter().any(|t| t.op == NodeOp::Subtract);
    match (int, sub) {
        (true, true) => "intersect+subtract",
        (true, false) => "intersect",
        _ => "subtract",
    }
}

fn host_json(host: &PrimNode) -> JsonValue {
    JsonValue::Object(vec![
        ("kind".to_string(), JsonValue::String(host.kind.clone())),
        ("params".to_string(), params_json(host)),
        (
            "transform".to_string(),
            transform_json(host.transform.as_ref()),
        ),
    ])
}

fn tools_json(tools: &[&PrimNode]) -> JsonValue {
    let rows = tools
        .iter()
        .map(|t| {
            JsonValue::Object(vec![
                ("op".to_string(), JsonValue::String(op_name(t.op).into())),
                ("kind".to_string(), JsonValue::String(t.kind.clone())),
                ("params".to_string(), params_json(t)),
                (
                    "transform".to_string(),
                    transform_json(t.transform.as_ref()),
                ),
                (
                    "hasChildren".to_string(),
                    JsonValue::Bool(!t.children.is_empty()),
                ),
            ])
        })
        .collect();
    JsonValue::Array(rows)
}

/// Node params as provider `Params` (numbers and 3-arrays, as the loader converts them).
fn node_params(n: &PrimNode) -> Result<Params, String> {
    let mut out = Params::new();
    for (k, v) in &n.params {
        let value = match v {
            JsonValue::Number(x) => ParamValue::Scalar(*x as f32),
            JsonValue::Array(a) => match a.as_slice() {
                [
                    JsonValue::Number(x),
                    JsonValue::Number(y),
                    JsonValue::Number(z),
                ] => ParamValue::Vec3([*x as f32, *y as f32, *z as f32]),
                _ => return Err(format!("node '{}': param '{k}' is not a 3-array", n.id)),
            },
            _ => return Err(format!("node '{}': param '{k}' is not a number", n.id)),
        };
        out.set(k, value);
    }
    Ok(out)
}

/// `Mat4::from_trs` of a node transform (identity when absent), as the loader does.
fn node_matrix(n: &PrimNode) -> Mat4 {
    let f = |v: f64| v as f32;
    n.transform.as_ref().map_or(Mat4::IDENTITY, |t| {
        Mat4::from_trs(
            Vec3::new(f(t.t[0]), f(t.t[1]), f(t.t[2])),
            Quat::from_xyzw(f(t.r[0]), f(t.r[1]), f(t.r[2]), f(t.r[3])),
            Vec3::new(f(t.s[0]), f(t.s[1]), f(t.s[2])),
        )
    })
}

/// The node's LOD0 mesh in its own space (`m` identity) or transformed by `m`.
fn node_mesh(geometry: &GeometryRegistry, n: &PrimNode, m: &Mat4) -> Result<MeshData, String> {
    let mut local = MeshData::new();
    geometry
        .generate(&n.kind, &node_params(n)?, Quality::LOD0, &mut local)
        .map_err(|e| format!("node '{}': {e}", n.id))?;
    let mut out = MeshData::new();
    out.append_transformed(&local, m);
    Ok(out)
}

/// CSG_EPS: 1e-6 x max(1, diagonal of the union of the meshes' bboxes), in f64.
fn csg_eps(meshes: &[&MeshData]) -> f64 {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in meshes.iter().flat_map(|m| &m.positions) {
        for k in 0..3 {
            lo[k] = lo[k].min(f64::from(p[k]));
            hi[k] = hi[k].max(f64::from(p[k]));
        }
    }
    let d2: f64 = (0..3).map(|k| (hi[k] - lo[k]).max(0.0).powi(2)).sum();
    1e-6 * d2.sqrt().max(1.0)
}

fn volume(m: &MeshData) -> f64 {
    analyze(m).volume
}

/// V(a ∩ b); an EmptyResult counts as 0.
fn int_volume(a: &MeshData, b: &MeshData) -> Result<f64, String> {
    let mut out = MeshData::new();
    match intersect_convex(a, b, &mut out) {
        Ok(()) => Ok(volume(&out)),
        Err(CsgError::EmptyResult) => Ok(0.0),
        Err(e) => Err(format!("intersect_convex: {} {e}", e.code())),
    }
}

/// V(host − Σ tools) and V(host) − Σ V(host ∩ tool), host convex.
fn subtract_identity(host: &MeshData, tools: &[&MeshData], eps: f64) -> Result<(f64, f64), String> {
    let planes = convex_planes(host, eps).map_err(|e| format!("convex_planes: {e}"))?;
    let mut out = MeshData::new();
    subtract_convex(host, &planes, tools, &mut out)
        .map_err(|e| format!("subtract_convex: {} {e}", e.code()))?;
    let mut rhs = volume(host);
    for t in tools {
        rhs -= int_volume(host, t)?;
    }
    Ok((volume(&out), rhs))
}

/// The volume identity relative error |lhs − rhs| / V(host) for an ok case.
fn identity_err(geometry: &GeometryRegistry, host: &PrimNode) -> Result<f64, String> {
    let h = node_mesh(geometry, host, &Mat4::IDENTITY)?;
    let tools = tools_of(host);
    let mut ints = Vec::new();
    let mut subs = Vec::new();
    for t in &tools {
        let m = node_mesh(geometry, t, &node_matrix(t))?;
        match t.op {
            NodeOp::Intersect => ints.push(m),
            _ => subs.push(m),
        }
    }
    let mut all: Vec<&MeshData> = vec![&h];
    all.extend(ints.iter().chain(&subs));
    let eps = csg_eps(&all);
    let subs: Vec<&MeshData> = subs.iter().collect();
    let (lhs, rhs) = match ints.as_slice() {
        [] => subtract_identity(&h, &subs, eps)?,
        [i] if subs.is_empty() => {
            // V(H ∩ T) vs V(H) − V(H − T).
            let (rest, _) = subtract_identity(&h, &[i], eps)?;
            (int_volume(&h, i)?, volume(&h) - rest)
        }
        [i] => {
            // H' = H ∩ I, then V(H' − ΣT) vs V(H') − Σ V(H' ∩ T).
            let mut hp = MeshData::new();
            intersect_convex(&h, i, &mut hp)
                .map_err(|e| format!("intersect_convex: {} {e}", e.code()))?;
            subtract_identity(&hp, &subs, eps)?
        }
        _ => return Err("more than one intersect tool".to_string()),
    };
    let vh = volume(&h);
    let err = (lhs - rhs).abs() / vh;
    if err.is_finite() {
        Ok(err)
    } else {
        Err(format!(
            "identity not finite: lhs {lhs} rhs {rhs} V(host) {vh}"
        ))
    }
}

fn run_case(case: &CsgCase, geometry: &GeometryRegistry) -> Outcome {
    let mut o = Outcome {
        id: case.id,
        expected: case.expected,
        op: "subtract",
        host: JsonValue::Null,
        tools: JsonValue::Array(Vec::new()),
        ok: false,
        triangles: 0,
        watertight: false,
        volume: 0.0,
        id_err: None,
        code: None,
        message: None,
        fails: Vec::new(),
    };
    let doc = match parse(&case.asset) {
        Ok(d) => d,
        Err(e) => {
            o.fails.push(format!("case text does not parse: {e}"));
            return o;
        }
    };
    let Some(host) = doc.nodes.first() else {
        o.fails.push("case has no host node".to_string());
        return o;
    };
    let tools = tools_of(host);
    o.op = case_op(&tools);
    o.host = host_json(host);
    o.tools = tools_json(&tools);

    let ctx = LoadContext {
        geometry,
        booleans: &CsgResolver,
        quality: Quality::LOD0,
        strict_kinds: true,
    };
    match PrimsLoader.load(case.asset.as_bytes(), &ctx) {
        Ok(loaded) => {
            o.ok = true;
            o.triangles = loaded.triangles;
            match loaded.meshes.iter().find(|m| m.node_id == host.id) {
                Some(m) => {
                    let r = analyze(&m.mesh);
                    o.watertight = r.closed && r.volume > 0.0;
                    o.volume = r.volume;
                }
                None => o.fails.push(format!("no mesh for host node '{}'", host.id)),
            }
            match identity_err(geometry, host) {
                Ok(e) => o.id_err = Some(e),
                Err(why) => o.fails.push(format!("volume identity: {why}")),
            }
        }
        Err(e) => {
            o.code = Some(e.code);
            o.message = Some(e.message);
        }
    }

    let got = o.code.unwrap_or("ok");
    if got != case.expected {
        o.fails
            .push(format!("got {got}, expected {}", case.expected));
    }
    if o.ok && !o.watertight {
        o.fails.push("result not watertight".to_string());
    }
    if let Some(e) = o.id_err.filter(|e| *e > ID_TOL) {
        o.fails
            .push(format!("volume identity error {e:e} > {ID_TOL:e}"));
    }
    if o.message.as_deref().is_some_and(str::is_empty) {
        o.fails.push("empty refusal message".to_string());
    }
    o
}

fn line(o: &Outcome) -> String {
    let status = if o.ok { "ok" } else { "refused" };
    let id_err = o.id_err.map_or("-".to_string(), |e| format!("{e:.3e}"));
    format!(
        "{} op={} status={status} expected={} triangles={} watertight={} volume={:.6} idErr={id_err} code={} message={:?}",
        o.id,
        o.op,
        o.expected,
        o.triangles,
        o.watertight,
        o.volume,
        o.code.unwrap_or("-"),
        o.message.as_deref().unwrap_or(""),
    )
}

fn json_table(outcomes: &[Outcome]) -> String {
    let s = |x: &str| JsonValue::String(x.to_string());
    let opt_s = |x: Option<&str>| x.map_or(JsonValue::Null, |x| JsonValue::String(x.into()));
    let rows = outcomes
        .iter()
        .map(|o| {
            let fields = [
                ("id", s(o.id)),
                ("op", s(o.op)),
                ("host", o.host.clone()),
                ("tools", o.tools.clone()),
                ("status", s(if o.ok { "ok" } else { "refused" })),
                ("triangles", JsonValue::Number(f64::from(o.triangles))),
                ("watertight", JsonValue::Bool(o.watertight)),
                ("volume", JsonValue::Number(o.volume)),
                (
                    "volumeIdentityRelErr",
                    o.id_err.map_or(JsonValue::Null, JsonValue::Number),
                ),
                ("code", opt_s(o.code)),
                ("message", opt_s(o.message.as_deref())),
                ("expected", s(o.expected)),
            ];
            JsonValue::Object(
                fields
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect(),
            )
        })
        .collect();
    let mut out = String::new();
    json::write_pretty(&JsonValue::Array(rows), &mut out);
    out.push('\n');
    out
}

/// NMC_CPUS, 1 when unset or invalid.
fn cpus() -> usize {
    std::env::var("NMC_CPUS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(1)
}

fn new_registry() -> Result<GeometryRegistry, String> {
    let mut geometry = GeometryRegistry::new();
    raa_prims::register_all(&mut geometry).map_err(|e| format!("register_all: {e}"))?;
    Ok(geometry)
}

/// Runs the cases on at most `threads` threads; results come back in table order.
fn run_all(cases: &[CsgCase], threads: usize) -> Result<Vec<Outcome>, String> {
    let next = AtomicUsize::new(0);
    let slots: Mutex<Vec<Option<Outcome>>> = Mutex::new((0..cases.len()).map(|_| None).collect());
    let threads = threads.min(cases.len()).max(1);
    std::thread::scope(|s| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| -> Result<(), String> {
                    // GeometryRegistry holds plain trait objects: one per thread.
                    let geometry = new_registry()?;
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(case) = cases.get(i) else {
                            return Ok(());
                        };
                        let o = run_case(case, &geometry);
                        let mut slots = slots.lock().map_err(|_| "result lock poisoned")?;
                        if let Some(slot) = slots.get_mut(i) {
                            *slot = Some(o);
                        }
                    }
                })
            })
            .collect();
        workers
            .into_iter()
            .try_for_each(|w| w.join().map_err(|_| "worker panicked".to_string())?)
    })?;
    let slots = slots.into_inner().map_err(|_| "result lock poisoned")?;
    slots
        .into_iter()
        .zip(cases)
        .map(|(o, c)| o.ok_or_else(|| format!("case {} produced no result", c.id)))
        .collect()
}

fn main() -> ExitCode {
    run().unwrap_or_else(|(code, m)| {
        eprintln!("csg-cases: {m}");
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

    let mut table = cases();
    table.retain(|c| only.as_ref().is_none_or(|p| p.matches(c.id)));
    if table.is_empty() {
        return Err(usage("--only matches no case".to_string()));
    }

    let outcomes = run_all(&table, cpus()).map_err(|m| (1, m))?;
    for o in &outcomes {
        println!("{}", line(o));
        for why in &o.fails {
            eprintln!("FAIL {}: {why}", o.id);
        }
    }
    let failed = outcomes.iter().filter(|o| !o.fails.is_empty()).count();
    eprintln!("cases {}/{} pass", outcomes.len() - failed, outcomes.len());
    if let Some(path) = &json {
        if let Some(dir) = std::path::Path::new(path).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        std::fs::write(path, json_table(&outcomes))
            .map_err(|e| (1, format!("cannot write {path}: {e}")))?;
    }
    Ok(ExitCode::from(u8::from(failed > 0)))
}
