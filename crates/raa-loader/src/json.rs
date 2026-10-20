//! Strict RFC 8259 JSON: [`JsonValue`], [`parse`], and the canonical writer
//! [`write_pretty`] / [`format_number`] (JS `JSON.stringify(v, null, 2)` rules).

use crate::error::LoadError;

/// Maximum nesting of arrays and objects.
pub const MAX_DEPTH: usize = 32;

/// A parsed JSON value. Objects keep document order.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

/// Parses exactly one JSON value followed by whitespace only.
pub fn parse(text: &str) -> Result<JsonValue, LoadError> {
    let mut p = Parser {
        b: text.as_bytes(),
        pos: 0,
    };
    p.ws();
    let v = p.value(0)?;
    p.ws();
    if p.pos < p.b.len() {
        return Err(p.err_at(p.pos, "json.syntax", "expected end of input"));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn err_at(&self, off: usize, code: &'static str, msg: &str) -> LoadError {
        let off = off.min(self.b.len());
        let head = self.b.get(..off).unwrap_or(&[]);
        let line = 1 + head.iter().filter(|&&c| c == b'\n').count();
        let line_start = head.iter().rposition(|&c| c == b'\n').map_or(0, |i| i + 1);
        let col = 1 + head
            .get(line_start..)
            .unwrap_or(&[])
            .iter()
            .filter(|&&c| (c & 0xC0) != 0x80)
            .count();
        LoadError::new(code, format!("@{off} {line}:{col}"), msg)
    }

    fn syntax(&self, msg: &str) -> LoadError {
        self.err_at(self.pos, "json.syntax", msg)
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.pos).copied()
    }

    fn ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, c: u8, what: &str) -> Result<(), LoadError> {
        if self.peek() == Some(c) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.syntax(&format!("expected {what}")))
        }
    }

    fn literal(&mut self, word: &str, v: JsonValue) -> Result<JsonValue, LoadError> {
        let end = self.pos + word.len();
        if self.b.get(self.pos..end) == Some(word.as_bytes()) {
            self.pos = end;
            Ok(v)
        } else {
            Err(self.syntax(&format!("expected `{word}`")))
        }
    }

    fn value(&mut self, depth: usize) -> Result<JsonValue, LoadError> {
        match self.peek() {
            Some(b'{') => self.object(depth + 1),
            Some(b'[') => self.array(depth + 1),
            Some(b'"') => Ok(JsonValue::String(self.string()?)),
            Some(b't') => self.literal("true", JsonValue::Bool(true)),
            Some(b'f') => self.literal("false", JsonValue::Bool(false)),
            Some(b'n') => self.literal("null", JsonValue::Null),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => Err(self.syntax("expected a JSON value")),
            None => Err(self.syntax("expected a JSON value, found end of input")),
        }
    }

    fn check_depth(&self, depth: usize) -> Result<(), LoadError> {
        if depth > MAX_DEPTH {
            Err(self.err_at(
                self.pos,
                "json.depth",
                &format!("nesting deeper than {MAX_DEPTH} arrays/objects"),
            ))
        } else {
            Ok(())
        }
    }

    fn array(&mut self, depth: usize) -> Result<JsonValue, LoadError> {
        self.check_depth(depth)?;
        self.pos += 1;
        let mut items = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            self.ws();
            items.push(self.value(depth)?);
            self.ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(JsonValue::Array(items));
                }
                _ => return Err(self.syntax("expected `,` or `]`")),
            }
        }
    }

    fn object(&mut self, depth: usize) -> Result<JsonValue, LoadError> {
        self.check_depth(depth)?;
        self.pos += 1;
        let mut members: Vec<(String, JsonValue)> = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Object(members));
        }
        loop {
            self.ws();
            if self.peek() != Some(b'"') {
                return Err(self.syntax("expected a string key"));
            }
            let key_pos = self.pos;
            let key = self.string()?;
            if members.iter().any(|(k, _)| *k == key) {
                return Err(self.err_at(
                    key_pos,
                    "json.duplicate_key",
                    &format!("duplicate key `{key}`"),
                ));
            }
            self.ws();
            self.expect(b':', "`:`")?;
            self.ws();
            let v = self.value(depth)?;
            members.push((key, v));
            self.ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(JsonValue::Object(members));
                }
                _ => return Err(self.syntax("expected `,` or `}`")),
            }
        }
    }

    fn digits(&mut self) -> usize {
        let start = self.pos;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        self.pos - start
    }

    fn number(&mut self) -> Result<JsonValue, LoadError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => {
                self.pos += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(self.syntax("leading zeros are not allowed in numbers"));
                }
            }
            Some(b'1'..=b'9') => {
                self.digits();
            }
            _ => return Err(self.syntax("expected a digit")),
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            if self.digits() == 0 {
                return Err(self.syntax("expected a digit after `.`"));
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if self.digits() == 0 {
                return Err(self.syntax("expected a digit in the exponent"));
            }
        }
        let slice = self.b.get(start..self.pos).unwrap_or(&[]);
        std::str::from_utf8(slice)
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .map(JsonValue::Number)
            .ok_or_else(|| self.err_at(start, "json.syntax", "invalid number"))
    }

    fn hex4(&mut self) -> Result<u32, LoadError> {
        let mut v = 0u32;
        for _ in 0..4 {
            let d = match self.peek() {
                Some(c @ b'0'..=b'9') => c - b'0',
                Some(c @ b'a'..=b'f') => c - b'a' + 10,
                Some(c @ b'A'..=b'F') => c - b'A' + 10,
                _ => return Err(self.syntax("expected 4 hex digits after `\\u`")),
            };
            v = v * 16 + u32::from(d);
            self.pos += 1;
        }
        Ok(v)
    }

    fn unicode_escape(&mut self) -> Result<char, LoadError> {
        let esc_start = self.pos - 2;
        let hi = self.hex4()?;
        let cp = match hi {
            0xD800..=0xDBFF => {
                if self.b.get(self.pos..self.pos + 2) != Some(b"\\u") {
                    return Err(self.err_at(
                        esc_start,
                        "json.syntax",
                        "lone high surrogate: expected `\\u` low surrogate",
                    ));
                }
                self.pos += 2;
                let lo = self.hex4()?;
                if !(0xDC00..=0xDFFF).contains(&lo) {
                    return Err(self.err_at(
                        esc_start,
                        "json.syntax",
                        "mismatched surrogate: expected a low surrogate (DC00-DFFF)",
                    ));
                }
                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
            }
            0xDC00..=0xDFFF => {
                return Err(self.err_at(esc_start, "json.syntax", "lone low surrogate"));
            }
            other => other,
        };
        char::from_u32(cp)
            .ok_or_else(|| self.err_at(esc_start, "json.syntax", "invalid code point"))
    }

    fn string(&mut self) -> Result<String, LoadError> {
        let open = self.pos;
        self.pos += 1;
        let mut out: Vec<u8> = Vec::new();
        loop {
            let Some(c) = self.peek() else {
                return Err(self.err_at(open, "json.syntax", "unterminated string: expected `\"`"));
            };
            match c {
                b'"' => {
                    self.pos += 1;
                    return String::from_utf8(out)
                        .map_err(|_| self.err_at(open, "json.syntax", "invalid UTF-8 in string"));
                }
                b'\\' => {
                    self.pos += 1;
                    let Some(e) = self.peek() else {
                        return Err(self.syntax("expected an escape character"));
                    };
                    self.pos += 1;
                    let simple = match e {
                        b'"' => Some(b'"'),
                        b'\\' => Some(b'\\'),
                        b'/' => Some(b'/'),
                        b'b' => Some(8),
                        b'f' => Some(12),
                        b'n' => Some(b'\n'),
                        b'r' => Some(b'\r'),
                        b't' => Some(b'\t'),
                        _ => None,
                    };
                    if let Some(s) = simple {
                        out.push(s);
                    } else if e == b'u' {
                        let ch = self.unicode_escape()?;
                        let mut buf = [0u8; 4];
                        out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    } else {
                        self.pos -= 1;
                        return Err(self.syntax("bad escape: expected one of \" \\ / b f n r t u"));
                    }
                }
                0..=0x1F => {
                    return Err(self.syntax("unescaped control character in string"));
                }
                _ => {
                    out.push(c);
                    self.pos += 1;
                }
            }
        }
    }
}

/// Writes `v` exactly like JS `JSON.stringify(v, null, 2)`: 2-space indentation,
/// `"key": value`, `[]` and `{}` for empty containers, keys in stored order.
/// No trailing newline is written.
pub fn write_pretty(v: &JsonValue, out: &mut String) {
    write_value(v, 0, out);
}

fn indent(level: usize, out: &mut String) {
    for _ in 0..level {
        out.push_str("  ");
    }
}

fn write_value(v: &JsonValue, level: usize, out: &mut String) {
    match v {
        JsonValue::Null => out.push_str("null"),
        JsonValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        JsonValue::Number(x) => format_number(*x, out),
        JsonValue::String(s) => write_string(s, out),
        JsonValue::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push('\n');
                indent(level + 1, out);
                write_value(item, level + 1, out);
            }
            out.push('\n');
            indent(level, out);
            out.push(']');
        }
        JsonValue::Object(members) => {
            if members.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            for (i, (k, item)) in members.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push('\n');
                indent(level + 1, out);
                write_string(k, out);
                out.push_str(": ");
                write_value(item, level + 1, out);
            }
            out.push('\n');
            indent(level, out);
            out.push('}');
        }
    }
}

fn write_string(s: &str, out: &mut String) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\0'..='\u{1f}' => {
                let b = c as usize;
                out.push_str("\\u00");
                out.push(char::from(HEX[b >> 4]));
                out.push(char::from(HEX[b & 0xF]));
            }
            _ => out.push(c),
        }
    }
    out.push('"');
}

/// Writes `x` following JS `Number.prototype.toString`: shortest round-trip
/// digits, positional notation for 1e-6 <= |x| < 1e21, otherwise
/// `d[.ddd]e+n` / `d[.ddd]e-n`; integers have no `.0`; -0 is written `0`.
/// Non-finite values are written `null`, like `JSON.stringify`.
pub fn format_number(x: f64, out: &mut String) {
    if !x.is_finite() {
        out.push_str("null");
        return;
    }
    if x == 0.0 {
        out.push('0');
        return;
    }
    if x < 0.0 {
        out.push('-');
    }
    // `{:e}` gives the shortest round-trip digits as `d[.ddd]e<exp>`.
    let sci = format!("{:e}", x.abs());
    let (mantissa, exp) = sci.split_once('e').unwrap_or((sci.as_str(), "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let k = digits.len() as i32;
    // JS: value = 0.digits * 10^n
    let n = exp + 1;
    if k <= n && n <= 21 {
        out.push_str(&digits);
        for _ in 0..(n - k) {
            out.push('0');
        }
    } else if 0 < n && n <= 21 {
        let (int, frac) = digits.split_at(n as usize);
        out.push_str(int);
        out.push('.');
        out.push_str(frac);
    } else if -6 < n && n <= 0 {
        out.push_str("0.");
        for _ in 0..(-n) {
            out.push('0');
        }
        out.push_str(&digits);
    } else {
        let (first, rest) = digits.split_at(1.min(digits.len()));
        out.push_str(first);
        if !rest.is_empty() {
            out.push('.');
            out.push_str(rest);
        }
        out.push('e');
        out.push(if n - 1 < 0 { '-' } else { '+' });
        out.push_str(&(n - 1).unsigned_abs().to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn code(text: &str) -> &'static str {
        parse(text).expect_err(text).code
    }

    #[test]
    fn accepts_basic_values() {
        assert_eq!(parse(" null ").unwrap(), JsonValue::Null);
        assert_eq!(
            parse("[true,false,-0.5e1]").unwrap(),
            JsonValue::Array(vec![
                JsonValue::Bool(true),
                JsonValue::Bool(false),
                JsonValue::Number(-5.0),
            ])
        );
        assert_eq!(
            parse("\"a\\n\\u00e9\\ud83d\\ude00\"").unwrap(),
            JsonValue::String("a\né😀".into())
        );
        assert_eq!(parse("0").unwrap(), JsonValue::Number(0.0));
    }

    #[test]
    fn refuses_syntax_variants() {
        for t in [
            "[1,]",
            "{\"a\":1,}",
            "01",
            "-01",
            "+1",
            ".5",
            "5.",
            "1e",
            "1e+",
            "// c\n1",
            "/* c */1",
            "\u{feff}1",
            "\"a\u{1}b\"",
            "\"a\nb\"",
            "\"\\x\"",
            "\"\\ud800\"",
            "\"\\udc00\"",
            "\"\\ud800\\u0041\"",
            "\"\\ud800x\"",
            "\"\\u12\"",
            "\u{a0}1",
            "\u{b}1",
            "1 2",
            "",
            "[1 2]",
            "{\"a\" 1}",
            "{1:2}",
            "tru",
            "\"abc",
            "nul",
            "-",
        ] {
            assert_eq!(code(t), "json.syntax", "{t:?}");
        }
    }

    #[test]
    fn overflow_is_infinity_not_error() {
        assert_eq!(parse("1e999").unwrap(), JsonValue::Number(f64::INFINITY));
        assert_eq!(
            parse("-1e999").unwrap(),
            JsonValue::Number(f64::NEG_INFINITY)
        );
    }

    #[test]
    fn keeps_key_order() {
        let JsonValue::Object(m) = parse("{\"b\":1,\"a\":2,\"c\":3}").unwrap() else {
            panic!("not an object");
        };
        let keys: Vec<&str> = m.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["b", "a", "c"]);
    }

    #[test]
    fn refuses_duplicate_key() {
        let e = parse("{\"a\":1,\"a\":2}").unwrap_err();
        assert_eq!(e.code, "json.duplicate_key");
        assert_eq!(e.path, "@7 1:8");
    }

    #[test]
    fn depth_32_ok_33_refused() {
        let ok = format!("{}{}", "[".repeat(32), "]".repeat(32));
        assert!(parse(&ok).is_ok());
        let bad = format!("{}{}", "[".repeat(33), "]".repeat(33));
        assert_eq!(code(&bad), "json.depth");
        let obj = format!("{}1{}", "{\"a\":".repeat(33), "}".repeat(33));
        assert_eq!(code(&obj), "json.depth");
        let deep = "[".repeat(100_000);
        assert_eq!(code(&deep), "json.depth");
    }

    #[test]
    fn error_position_on_line_three() {
        let e = parse("{\n  \"a\": 1,\n  \"é\": ?\n}").unwrap_err();
        assert_eq!(e.code, "json.syntax");
        // line 1 is 2 bytes, line 2 is 10 bytes, `  "é": ` is 8 bytes (7 chars)
        assert_eq!(e.path, "@20 3:8");
        assert!(e.message.contains("expected"));
    }

    #[test]
    fn whitespace_is_limited() {
        assert!(parse("\t\r\n [ ] \n").is_ok());
    }
}
