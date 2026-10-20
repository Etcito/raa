//! Q6: canonical prims text round-trips byte-identical, non-canonical variants are
//! canonicalised idempotently, and every parse-time invalid case is refused with its code.

use raa_loader::json::{self, JsonValue};
use raa_loader::prims::samples::{invalid_cases, sample_assets};
use raa_loader::prims::{parse, to_canonical_json};

/// Codes only `build` can emit; their cases parse successfully.
const BUILD_TIME: [&str; 3] = [
    "cap.budget",
    "collision.torus_needs_override",
    "boolean.unsupported",
];

fn canon(text: &str) -> String {
    let a = parse(text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    to_canonical_json(&a)
}

/// Single-line JSON with a custom number spelling.
fn compact(v: &JsonValue, num: &dyn Fn(f64) -> String, out: &mut String) {
    match v {
        JsonValue::Number(x) => out.push_str(&num(*x)),
        JsonValue::Array(items) => {
            out.push('[');
            for (i, x) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                compact(x, num, out);
            }
            out.push(']');
        }
        JsonValue::Object(members) => {
            out.push('{');
            for (i, (k, x)) in members.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                json::write_pretty(&JsonValue::String(k.clone()), out);
                out.push(':');
                compact(x, num, out);
            }
            out.push('}');
        }
        scalar => json::write_pretty(scalar, out),
    }
}

fn js(x: f64) -> String {
    let mut s = String::new();
    json::format_number(x, &mut s);
    s
}

fn compact_with(v: &JsonValue, num: &dyn Fn(f64) -> String) -> String {
    let mut s = String::new();
    compact(v, num, &mut s);
    s
}

fn pretty(v: &JsonValue) -> String {
    let mut s = String::new();
    json::write_pretty(v, &mut s);
    s
}

fn reverse_keys(v: &mut JsonValue) {
    match v {
        JsonValue::Array(items) => items.iter_mut().for_each(reverse_keys),
        JsonValue::Object(members) => {
            members.reverse();
            members.iter_mut().for_each(|(_, x)| reverse_keys(x));
        }
        _ => {}
    }
}

fn member<'a>(v: &'a mut JsonValue, key: &str) -> Option<&'a mut JsonValue> {
    match v {
        JsonValue::Object(m) => m.iter_mut().find(|(k, _)| k == key).map(|(_, x)| x),
        _ => None,
    }
}

type Members = Vec<(String, JsonValue)>;

/// Applies `f` to every node object, in pre-order.
fn each_node(v: &mut JsonValue, f: &dyn Fn(&mut Members)) {
    fn walk(nodes: &mut JsonValue, f: &dyn Fn(&mut Members)) {
        if let JsonValue::Array(items) = nodes {
            for n in items {
                if let JsonValue::Object(m) = n {
                    f(m);
                    if let Some((_, c)) = m.iter_mut().find(|(k, _)| k == "children") {
                        walk(c, f);
                    }
                }
            }
        }
    }
    if let Some(nodes) = member(v, "nodes") {
        walk(nodes, f);
    }
}

/// The non-canonical variants of one canonical sample text: (label, text).
fn variants(text: &str) -> Vec<(&'static str, String)> {
    let v = json::parse(text).expect("sample parses as JSON");
    let mut out = Vec::new();

    let mut r = v.clone();
    reverse_keys(&mut r);
    out.push(("reordered-keys", pretty(&r)));

    out.push(("compact-single-line", compact_with(&v, &js)));

    let one_point_zero = |x: f64| {
        if x.fract() == 0.0 {
            format!("{}.0", js(x))
        } else {
            js(x)
        }
    };
    out.push(("numbers-1.0", compact_with(&v, &one_point_zero)));
    out.push(("numbers-1E0", compact_with(&v, &|x| format!("{}E0", js(x)))));
    out.push((
        "numbers-minus-zero",
        compact_with(&v, &|x| if x == 0.0 { "-0".into() } else { js(x) }),
    ));

    let mut u = v.clone();
    each_node(&mut u, &|m| {
        if !m.iter().any(|(k, _)| k == "op") {
            m.push(("op".into(), JsonValue::String("union".into())));
        }
    });
    out.push(("explicit-op-union", pretty(&u)));

    let mut c = v.clone();
    each_node(&mut c, &|m| {
        if !m.iter().any(|(k, _)| k == "children") {
            m.push(("children".into(), JsonValue::Array(Vec::new())));
        }
    });
    out.push(("empty-children", pretty(&c)));

    let mut rev = v.clone();
    for key in ["materials", "anchors"] {
        if let Some(JsonValue::Object(m)) = member(&mut rev, key) {
            m.reverse();
        }
    }
    out.push(("materials-anchors-reversed", pretty(&rev)));
    out
}

#[test]
fn canonical_samples_round_trip_byte_identical() {
    let samples = sample_assets();
    assert_eq!(samples.len(), 3);
    for (name, text) in samples {
        assert_eq!(canon(&text), text, "sample {name}");
    }
}

#[test]
fn non_canonical_variants_are_canonicalised_idempotently() {
    let mut count = 0;
    for (name, text) in sample_assets() {
        for (label, variant) in variants(&text) {
            assert_ne!(
                variant, text,
                "{name}/{label} must differ from the canonical text"
            );
            let once = canon(&variant);
            assert_eq!(once, text, "{name}/{label}");
            assert_eq!(canon(&once), once, "{name}/{label} idempotent");
            count += 1;
        }
    }
    assert!(count >= 6 * 3, "{count} variants");
}

#[test]
fn parse_time_invalid_cases_are_refused_with_their_code() {
    let mut checked = 0;
    for (label, text, code) in invalid_cases() {
        if BUILD_TIME.contains(&code) {
            assert!(parse(&text).is_ok(), "{label}: build-time case must parse");
            continue;
        }
        match parse(&text) {
            Ok(_) => panic!("{label}: accepted, expected {code}"),
            Err(e) => {
                assert_eq!(e.code, code, "{label}: {e}");
                assert!(!e.message.is_empty(), "{label}: empty message");
            }
        }
        checked += 1;
    }
    assert!(checked > 0);
}
