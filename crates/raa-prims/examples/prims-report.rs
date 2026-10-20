//! prims-report: generates every selected kind x set (A, B) x quality (LOW, LOD0, HIGH) twice,
//! checks `analysis::passes` and determinism and, with `--compare`, the SDF reference fixture
//! under SDF_REF_TOL. Exit 0 when every case is ok, 1 otherwise, 2 for usage errors (see USAGE).

use std::fmt::Write as _;
use std::process::ExitCode;

use raa_geom::analysis::{self, MeshReport, analyze, mesh_hash, unique_positions_1e6};
use raa_geom::{MeshData, Quality};
use raa_prims::reference::{self, BBOX_TOL, REL_TOL, RefDelta, RefFixture};
use raa_prims::{PrimKind, PrimParams, generate};

const USAGE: &str =
    "usage: prims-report [--only <pattern on kind>] [--json <path>] [--compare <fixture>]";
const QUALITIES: [Quality; 3] = [Quality::LOW, Quality::LOD0, Quality::HIGH];

/// The `--only` subset: optional `^`, optional `( ... )`, `|` alternatives, optional `$`.
struct Pattern {
    start: bool,
    end: bool,
    alts: Vec<String>,
}

impl Pattern {
    fn parse(s: &str) -> Result<Pattern, String> {
        let (start, rest) = s.strip_prefix('^').map_or((false, s), |r| (true, r));
        let (end, rest) = rest.strip_suffix('$').map_or((false, rest), |r| (true, r));
        let rest = rest
            .strip_prefix('(')
            .map_or(Some(rest), |r| r.strip_suffix(')'));
        let rest = rest.ok_or_else(|| format!("--only {s:?}: unbalanced '('"))?;
        let bad = |c: &char| !(c.is_ascii_alphanumeric() || *c == '_' || *c == '|');
        if let Some(c) = rest.chars().find(bad) {
            return Err(format!(
                "--only {s:?}: unsupported character {c:?} (only ^, ( ), | and $ over literal kind names)"
            ));
        }
        let alts = rest.split('|').map(str::to_string).collect();
        Ok(Pattern { start, end, alts })
    }

    fn matches(&self, name: &str) -> bool {
        self.alts.iter().any(|a| match (self.start, self.end) {
            (true, true) => name == a,
            (true, false) => name.starts_with(a.as_str()),
            (false, true) => name.ends_with(a.as_str()),
            (false, false) => name.contains(a.as_str()),
        })
    }
}

struct Case {
    kind: PrimKind,
    set: char,
    quality: u32,
    /// Analysis of the first generation (of an empty mesh when generate failed).
    r: MeshReport,
    unique_positions: u32,
    hash: Option<u64>,
    deterministic: bool,
    passes: Result<(), String>,
    /// With --compare: the delta, its bbox bound and the case's sdfInput.
    delta: Option<(RefDelta, f64, String)>,
}

impl Case {
    fn ok(&self) -> bool {
        self.passes.is_ok() && self.deterministic && self.delta.as_ref().is_none_or(|d| d.0.ok)
    }
}

fn run_case(kind: PrimKind, set: char, q: Quality, fx: Option<&RefFixture>) -> Case {
    let p = PrimParams::reference_set(kind, set);
    let (mut a, mut b) = (MeshData::new(), MeshData::new());
    let generated = generate(&p, q, &mut a).and(generate(&p, q, &mut b));
    let mut r = analyze(&a);
    r.closed &= generated.is_ok(); // an empty mesh is only vacuously closed
    let hash = generated.is_ok().then(|| mesh_hash(&a));
    let deterministic = hash.is_some() && hash == Some(mesh_hash(&b));
    let passes = match generated {
        Ok(()) => analysis::passes(&r),
        Err(e) => Err(format!("generate failed: {e}")), // Display starts with the code
    };
    let delta = fx.map(|fx| match fx.case(kind, set, q.segments) {
        Some(rc) => {
            let diag = (0..3)
                .map(|i| (rc.bbox_max[i] - rc.bbox_min[i]).powi(2))
                .sum::<f64>();
            let tol = BBOX_TOL * diag.sqrt().max(1.0);
            (reference::compare(fx, rc, &a), tol, rc.sdf_input.clone())
        }
        None => {
            let message = format!("fixture has no case {} {set} {}", kind.as_str(), q.segments);
            let (n, f) = (f64::NAN, false);
            let d = RefDelta {
                unique_positions_eq: f,
                triangles_eq: f,
                bbox_max_abs_err: n,
                area_rel_err: n,
                volume_rel_err: n,
                ok: f,
                message,
            };
            (d, n, String::new())
        }
    });
    let unique_positions = unique_positions_1e6(&a);
    let quality = q.segments;
    Case {
        kind,
        set,
        quality,
        r,
        unique_positions,
        hash,
        deterministic,
        passes,
        delta,
    }
}

/// A JSON number via Rust `{}` formatting; non-finite values are written as null.
fn num(v: f64) -> String {
    Some(v)
        .filter(|v| v.is_finite())
        .map_or("null".into(), |v| format!("{v}"))
}

fn print_case(c: &Case) {
    let (kind, set, q, r) = (c.kind.as_str(), c.set, c.quality, &c.r);
    let (tris, verts, uniq, closed) = (r.triangles, r.vertices, c.unique_positions, r.closed);
    let (volume, dev) = (num(r.volume), r.max_normal_deviation_deg);
    let hash = c.hash.map_or("-".to_string(), |h| format!("{h:016x}"));
    let det = c.deterministic;
    let pass = if c.passes.is_ok() { "yes" } else { "no" };
    let mut line = format!(
        "{kind:<9} {set} q={q:<2} triangles={tris} vertices={verts} uniquePositions={uniq} \
         closed={closed} volume={volume} maxNormalDevDeg={dev} hash={hash} deterministic={det} \
         passes={pass}"
    );
    if let Some((d, tol, _)) = &c.delta {
        let (ue, te, ok) = (d.unique_positions_eq, d.triangles_eq, d.ok);
        let (bb, tol) = (num(d.bbox_max_abs_err), num(*tol));
        let (ar, vr) = (num(d.area_rel_err), num(d.volume_rel_err));
        let _ = write!(
            line,
            " | ref uniquePositionsEq={ue} trianglesEq={te} bboxMaxAbsErr={bb} <= {tol} \
             areaRelErr={ar} <= {REL_TOL} volumeRelErr={vr} <= {REL_TOL} refOk={ok}"
        );
    }
    println!("{line} {}", if c.ok() { "OK" } else { "FAIL" });
    if let Err(m) = &c.passes {
        println!("  fail: {m}");
    }
    if c.hash.is_some() && !det {
        println!("  fail: the two generations have different hashes");
    }
    if let Some((d, _, sdf_input)) = c.delta.as_ref().filter(|d| !d.0.ok) {
        println!("  ref fail: {}", d.message);
        if !sdf_input.is_empty() {
            println!("  sdfInput: {sdf_input}");
        }
    }
}

fn json_table(cases: &[Case]) -> String {
    let mut s = String::from("[\n");
    for (i, c) in cases.iter().enumerate() {
        let r = match &c.delta {
            None => "null".to_string(),
            Some((d, _, _)) => {
                let (ue, te, ok) = (d.unique_positions_eq, d.triangles_eq, d.ok);
                let bb = num(d.bbox_max_abs_err);
                let (ar, vr) = (num(d.area_rel_err), num(d.volume_rel_err));
                format!(
                    "{{\"uniquePositionsEq\": {ue}, \"trianglesEq\": {te}, \"bboxMaxAbsErr\": {bb}, \
                     \"areaRelErr\": {ar}, \"volumeRelErr\": {vr}, \"ok\": {ok}}}"
                )
            }
        };
        let (kind, set, q) = (c.kind.as_str(), c.set, c.quality);
        let (tris, verts, uniq, closed) =
            (c.r.triangles, c.r.vertices, c.unique_positions, c.r.closed);
        let (pass, det) = (c.passes.is_ok(), c.deterministic);
        let hash = c
            .hash
            .map_or("null".to_string(), |h| format!("\"{h:016x}\""));
        let sep = if i + 1 < cases.len() { "," } else { "" };
        let _ = writeln!(
            s,
            "  {{\"kind\": \"{kind}\", \"set\": \"{set}\", \"quality\": {q}, \"triangles\": {tris}, \
             \"vertices\": {verts}, \"uniquePositions\": {uniq}, \"closed\": {closed}, \
             \"passes\": {pass}, \"hash\": {hash}, \"deterministic\": {det}, \"ref\": {r}}}{sep}"
        );
    }
    s.push_str("]\n");
    s
}

fn main() -> ExitCode {
    run().unwrap_or_else(|(code, m)| {
        eprintln!("prims-report: {m}");
        ExitCode::from(code)
    })
}

/// Runs the report; an `Err` carries the exit code (2 usage, 1 failure) and the message.
fn run() -> Result<ExitCode, (u8, String)> {
    let usage = |m: String| (2, format!("{m}\n{USAGE}"));
    let (mut only, mut json, mut compare) = (None::<Pattern>, None::<String>, None::<String>);
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let value = it
            .next()
            .ok_or_else(|| usage(format!("{flag} needs a value")))?;
        match flag.as_str() {
            "--only" => only = Some(Pattern::parse(&value).map_err(usage)?),
            "--json" => json = Some(value),
            "--compare" => compare = Some(value),
            _ => return Err(usage(format!("unknown argument {flag:?}"))),
        }
    }
    let kinds: Vec<PrimKind> = PrimKind::ALL
        .into_iter()
        .filter(|k| only.as_ref().is_none_or(|p| p.matches(k.as_str())))
        .collect();
    if kinds.is_empty() {
        return Err(usage("--only matches no kind".to_string()));
    }
    let fixture = match &compare {
        None => None,
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .map_err(|e| (2, format!("cannot read {path}: {e}")))?;
            let fx = reference::parse_fixture(&text);
            Some(fx.map_err(|e| (1, format!("invalid fixture {path}: {e}")))?)
        }
    };
    let mut cases = Vec::new();
    for &kind in &kinds {
        for set in ['A', 'B'] {
            for q in QUALITIES {
                let c = run_case(kind, set, q, fixture.as_ref());
                print_case(&c);
                cases.push(c);
            }
        }
    }
    let passed = cases.iter().filter(|c| c.ok()).count();
    println!("prims-report: {passed}/{} cases ok", cases.len());
    if let Some(path) = &json {
        if let Some(dir) = std::path::Path::new(path).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        std::fs::write(path, json_table(&cases))
            .map_err(|e| (1, format!("cannot write {path}: {e}")))?;
    }
    Ok(ExitCode::from(u8::from(passed != cases.len())))
}
