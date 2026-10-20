//! JSON writer (JS `JSON.stringify(v, null, 2)` equivalence) and parser refusals.

use raa_loader::json::{JsonValue, format_number, parse, write_pretty};

fn num(x: f64) -> String {
    let mut s = String::new();
    format_number(x, &mut s);
    s
}

fn pretty(v: &JsonValue) -> String {
    let mut s = String::new();
    write_pretty(v, &mut s);
    s
}

fn obj(members: &[(&str, JsonValue)]) -> JsonValue {
    JsonValue::Object(
        members
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    )
}

#[test]
fn numbers_match_js_to_string() {
    let table: &[(f64, &str)] = &[
        (0.0, "0"),
        (-0.0, "0"),
        (1.0, "1"),
        (-2.5, "-2.5"),
        (0.1, "0.1"),
        (0.1 + 0.2, "0.30000000000000004"),
        (100.0, "100"),
        (1e21, "1e+21"),
        (123456789012345680000.0, "123456789012345680000"),
        (0.000001, "0.000001"),
        (1e-7, "1e-7"),
        (5e-324, "5e-324"),
        (1.7976931348623157e308, "1.7976931348623157e+308"),
        (1.5e-7, "1.5e-7"),
        (-1.25e22, "-1.25e+22"),
        (123.456, "123.456"),
        (0.00012, "0.00012"),
    ];
    for &(x, want) in table {
        assert_eq!(num(x), want, "{x:e}");
    }
}

#[test]
fn non_finite_numbers_are_null() {
    assert_eq!(num(f64::NAN), "null");
    assert_eq!(num(f64::INFINITY), "null");
    assert_eq!(num(f64::NEG_INFINITY), "null");
}

#[test]
fn string_escapes_match_json_stringify() {
    let v = JsonValue::String("q\"b\\\u{8}\u{c}\n\r\t\u{0}\u{1f}\u{7f}é😀/".into());
    assert_eq!(
        pretty(&v),
        "\"q\\\"b\\\\\\b\\f\\n\\r\\t\\u0000\\u001f\u{7f}é😀/\""
    );
    assert_eq!(pretty(&JsonValue::String("\u{1b}".into())), "\"\\u001b\"");
}

#[test]
fn nested_value_pretty_text() {
    let v = obj(&[
        ("format", JsonValue::String("raa.prims".into())),
        ("version", JsonValue::Number(0.0)),
        ("empty", JsonValue::Array(vec![])),
        ("none", obj(&[])),
        (
            "nodes",
            JsonValue::Array(vec![obj(&[
                ("id", JsonValue::String("a".into())),
                (
                    "r",
                    JsonValue::Array(vec![JsonValue::Number(0.0), JsonValue::Number(-0.5)]),
                ),
                ("ok", JsonValue::Bool(true)),
                ("n", JsonValue::Null),
            ])]),
        ),
        ("z", JsonValue::Number(1e-7)),
    ]);
    let want = r#"{
  "format": "raa.prims",
  "version": 0,
  "empty": [],
  "none": {},
  "nodes": [
    {
      "id": "a",
      "r": [
        0,
        -0.5
      ],
      "ok": true,
      "n": null
    }
  ],
  "z": 1e-7
}"#;
    assert_eq!(pretty(&v), want);
    assert_eq!(pretty(&JsonValue::Array(vec![])), "[]");
    assert_eq!(pretty(&obj(&[])), "{}");
}

#[test]
fn write_parse_write_is_byte_identical() {
    let v = obj(&[
        ("b", JsonValue::Number(0.30000000000000004)),
        (
            "a",
            JsonValue::Array(vec![JsonValue::Number(1e21), JsonValue::Number(5e-324)]),
        ),
        ("s", JsonValue::String("tab\there \u{1} \"x\" é".into())),
        (
            "o",
            obj(&[("deep", obj(&[("x", JsonValue::Array(vec![obj(&[])]))]))]),
        ),
        ("max", JsonValue::Number(1.7976931348623157e308)),
    ]);
    let first = pretty(&v);
    let parsed = parse(&first).expect("writer output parses");
    assert_eq!(parsed, v);
    assert_eq!(pretty(&parsed), first);
}

#[test]
fn syntax_error_has_offset_path() {
    let e = parse("[1,\n 2,]").unwrap_err();
    assert_eq!(e.code, "json.syntax");
    assert!(e.path.starts_with('@'), "{}", e.path);
    assert_eq!(e.path, "@7 2:4");
    assert!(!e.message.is_empty());
}

#[test]
fn refusals_have_their_codes() {
    let cases: &[(&str, &str)] = &[
        ("{\"a\":1,}", "json.syntax"),
        ("01", "json.syntax"),
        ("\u{feff}{}", "json.syntax"),
        ("{\"k\":1,\"k\":2}", "json.duplicate_key"),
    ];
    for &(text, code) in cases {
        let e = parse(text).unwrap_err();
        assert_eq!(e.code, code, "{text:?}");
        assert!(e.path.starts_with('@'), "{text:?}: {}", e.path);
        assert!(!e.message.is_empty());
    }
    let ok = format!("{}{}", "[".repeat(32), "]".repeat(32));
    assert!(parse(&ok).is_ok());
    let deep = format!("{}{}", "[".repeat(33), "]".repeat(33));
    let e = parse(&deep).unwrap_err();
    assert_eq!(e.code, "json.depth");
    assert!(e.path.starts_with('@'));
}
