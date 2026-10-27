//! S4 canonical JSON. Schema writers supply fields in canonical order,
//! including bytewise sorting of maps; this module preserves that order.

use crate::JsonValue;
use std::fmt::Write;

/// Two-space JSON with exactly one final LF. Schema validation must reject
/// non-finite document numbers before writing (JSON.stringify maps them to null).
pub fn to_canonical_json(value: &JsonValue) -> String {
    let mut out = String::new();
    write_pretty(value, &mut out);
    out.push('\n');
    out
}

/// Append pretty JSON without a final LF, retaining the supplied field order.
pub fn write_pretty(value: &JsonValue, out: &mut String) {
    write_value(value, 0, out);
}

fn newline(depth: usize, out: &mut String) {
    out.push('\n');
    for _ in 0..depth {
        out.push_str("  ");
    }
}

fn write_value(value: &JsonValue, depth: usize, out: &mut String) {
    match value {
        JsonValue::Null => out.push_str("null"),
        JsonValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        JsonValue::Number(n) => format_number(*n, out),
        JsonValue::String(s) => write_string(s, out),
        JsonValue::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                newline(depth + 1, out);
                write_value(item, depth + 1, out);
            }
            if !items.is_empty() {
                newline(depth, out);
            }
            out.push(']');
        }
        JsonValue::Object(fields) => {
            out.push('{');
            for (i, (key, item)) in fields.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                newline(depth + 1, out);
                write_string(key, out);
                out.push_str(": ");
                write_value(item, depth + 1, out);
            }
            if !fields.is_empty() {
                newline(depth, out);
            }
            out.push('}');
        }
    }
}

fn write_string(value: &str, out: &mut String) {
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\0'..='\u{1f}' => write!(out, "\\u{:04x}", c as u32).unwrap(),
            _ => out.push(c),
        }
    }
    out.push('"');
}

/// Append JavaScript's shortest round-trip number spelling. Negative zero is
/// zero; non-finite values become null, as in JSON.stringify.
pub fn format_number(value: f64, out: &mut String) {
    if !value.is_finite() {
        out.push_str("null");
        return;
    }
    if value == 0.0 {
        out.push('0');
        return;
    }
    if value < 0.0 {
        out.push('-');
    }
    let x = value.abs();
    let shortest = format!("{x:e}");
    let (mantissa, _) = shortest.split_once('e').unwrap();
    let precision = mantissa.bytes().filter(u8::is_ascii_digit).count() - 1;
    // Rust's shortest formatter can round a decimal tie upward. Explicit
    // precision rounds ties to even, as ECMAScript requires. Keep the original
    // if rounding would leave the float's (possibly asymmetric) rounding cell.
    let rounded = format!("{x:.precision$e}");
    let sci = if rounded.parse::<f64>() == Ok(x) {
        &rounded
    } else {
        &shortest
    };
    let (mantissa, exponent) = sci.split_once('e').unwrap();
    let exponent: i32 = exponent.parse().unwrap();
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let digits = digits.trim_end_matches('0');
    let n = exponent + 1;
    if (1..=21).contains(&n) {
        let split = (n as usize).min(digits.len());
        out.push_str(&digits[..split]);
        for _ in digits.len()..n as usize {
            out.push('0');
        }
        if split < digits.len() {
            out.push('.');
            out.push_str(&digits[split..]);
        }
    } else if (-5..=0).contains(&n) {
        out.push_str("0.");
        for _ in 0..-n {
            out.push('0');
        }
        out.push_str(digits);
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        write!(out, "e{exponent:+}").unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    fn number(x: f64) -> String {
        let mut out = String::new();
        format_number(x, &mut out);
        out
    }

    #[test]
    fn canon_exponent_boundaries() {
        for (x, expected) in [
            (0.0, "0"),
            (-0.0, "0"),
            (1.0, "1"),
            (1e-7, "1e-7"),
            (1e-6, "0.000001"),
            (
                f64::from_bits(1e-6_f64.to_bits() - 1),
                "9.999999999999997e-7",
            ),
            (
                f64::from_bits(1e-6_f64.to_bits() + 1),
                "0.0000010000000000000002",
            ),
            (1e20, "100000000000000000000"),
            (1e21, "1e+21"),
            (
                f64::from_bits(1e21_f64.to_bits() - 1),
                "999999999999999900000",
            ),
            (
                f64::from_bits(1e21_f64.to_bits() + 1),
                "1.0000000000000001e+21",
            ),
        ] {
            assert_eq!(number(x), expected);
            if x != 0.0 {
                assert_eq!(number(-x), format!("-{expected}"));
            }
        }
    }

    #[test]
    fn canon_difficult_round_trip_decimals() {
        for (x, expected) in [
            (0.30000000000000004, "0.30000000000000004"),
            (1.2345678901234567, "1.2345678901234567"),
            (1.0000000000000002, "1.0000000000000002"),
            (1000000000000000128.0, "1000000000000000100"),
            (100000000000000.25, "100000000000000.25"),
            (1000000000000000.25, "1000000000000000.2"), // decimal tie: even
            (f64::from_bits(1), "5e-324"),
            (f64::MIN_POSITIVE, "2.2250738585072014e-308"),
            (f64::MAX, "1.7976931348623157e+308"),
        ] {
            let text = number(x);
            assert_eq!(text, expected);
            assert_eq!(text.parse::<f64>().unwrap().to_bits(), x.to_bits());
        }
    }

    #[test]
    fn canon_escaped_strings() {
        let s = "\"\\/\u{8}\u{c}\n\r\t\0\u{1f}é😀\u{2028}\u{2029}";
        let text = to_canonical_json(&JsonValue::String(s.into()));
        assert_eq!(
            text,
            "\"\\\"\\\\/\\b\\f\\n\\r\\t\\u0000\\u001fé😀\u{2028}\u{2029}\"\n"
        );
        assert_eq!(parse(&text).unwrap(), JsonValue::String(s.into()));
        for c in '\0'..='\u{1f}' {
            let v = JsonValue::String(c.to_string());
            assert_eq!(parse(&to_canonical_json(&v)).unwrap(), v);
        }
    }

    #[test]
    fn canon_containers_and_idempotence() {
        let input = r#"{"z":[true,false,null,{},[],{"\"":-0}],"a":1.2300e-7}"#;
        let expected = "{\n  \"z\": [\n    true,\n    false,\n    null,\n    {},\n    [],\n    {\n      \"\\\"\": 0\n    }\n  ],\n  \"a\": 1.23e-7\n}\n";
        let canonical = to_canonical_json(&parse(input).unwrap());
        assert_eq!(canonical, expected);
        assert_eq!(to_canonical_json(&parse(&canonical).unwrap()), canonical);
        for text in ["null\n", "true\n", "false\n", "[]\n", "{}\n", "1e+21\n"] {
            assert_eq!(to_canonical_json(&parse(text).unwrap()), text);
        }
        for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(number(x), "null");
        }
    }
}
