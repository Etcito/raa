//! Strict RFC 8259 JSON reader. Numbers stay `f64` document values so
//! canonical writing never serializes rounded `f32` runtime values.

use crate::AnimError;
use std::collections::BTreeSet;

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
pub fn parse(text: &str) -> Result<JsonValue, AnimError> {
    let mut p = Parser {
        b: text.as_bytes(),
        pos: 0,
        path: String::new(),
    };
    let v = p.value(0)?;
    p.ws();
    if p.pos < p.b.len() {
        return Err(p.err("json.syntax", "trailing input after the JSON value"));
    }
    Ok(v)
}

/// Append an object key or array index to an RFC 6901 JSON pointer.
/// Schema consumers can use the same escaping as parser errors.
pub fn pointer_child(parent: &str, token: &str) -> String {
    format!("{parent}/{}", token.replace('~', "~0").replace('/', "~1"))
}

/// Read a rig document, mapping reader failures to the S16 refusal codes.
pub fn parse_rig(text: &str) -> Result<JsonValue, AnimError> {
    parse_document(text, "rig.schema", "rig.not_finite")
}

/// Read a clip document, mapping reader failures to the S17 refusal codes.
pub fn parse_clip(text: &str) -> Result<JsonValue, AnimError> {
    parse_document(text, "clip.schema", "clip.not_finite")
}

fn parse_document(
    text: &str,
    schema: &'static str,
    not_finite: &'static str,
) -> Result<JsonValue, AnimError> {
    parse(text).map_err(|mut error| {
        error.code = if error.code == "json.not_finite" {
            not_finite
        } else {
            schema
        };
        error
    })
}

struct Parser<'a> {
    b: &'a [u8],
    pos: usize,
    path: String,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.b.get(self.pos).copied()
    }
    fn err(&self, code: &'static str, msg: &str) -> AnimError {
        AnimError::new(code, &self.path, format!("{msg} at byte {}", self.pos))
    }
    fn ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }
    fn value(&mut self, depth: usize) -> Result<JsonValue, AnimError> {
        self.ws();
        match self.peek() {
            Some(b'{') => self.object(depth + 1),
            Some(b'[') => self.array(depth + 1),
            Some(b'"') => self.string().map(JsonValue::String),
            Some(b't') => self.word("true", JsonValue::Bool(true)),
            Some(b'f') => self.word("false", JsonValue::Bool(false)),
            Some(b'n') => self.word("null", JsonValue::Null),
            Some(b'-' | b'0'..=b'9') => self.number().map(JsonValue::Number),
            Some(_) => Err(self.err("json.syntax", "expected a JSON value")),
            None => Err(self.err("json.syntax", "expected a JSON value, found end")),
        }
    }
    fn word(&mut self, word: &str, v: JsonValue) -> Result<JsonValue, AnimError> {
        let end = self.pos + word.len();
        if self.b.get(self.pos..end) == Some(word.as_bytes()) {
            self.pos = end;
            Ok(v)
        } else {
            Err(self.err("json.syntax", &format!("expected `{word}`")))
        }
    }
    fn array(&mut self, depth: usize) -> Result<JsonValue, AnimError> {
        if depth > MAX_DEPTH {
            return Err(self.err("json.depth", "too deep"));
        }
        self.pos += 1;
        let mut items = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            let parent_len = self.path.len();
            self.path = pointer_child(&self.path, &items.len().to_string());
            items.push(self.value(depth)?);
            self.path.truncate(parent_len);
            self.ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(JsonValue::Array(items));
                }
                _ => return Err(self.err("json.syntax", "expected `,` or `]`")),
            }
        }
    }
    fn object(&mut self, depth: usize) -> Result<JsonValue, AnimError> {
        if depth > MAX_DEPTH {
            return Err(self.err("json.depth", "too deep"));
        }
        self.pos += 1;
        let mut members = Vec::new();
        let mut keys = BTreeSet::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Object(members));
        }
        loop {
            self.ws();
            if self.peek() != Some(b'"') {
                return Err(self.err("json.syntax", "expected a string key"));
            }
            let key = self.string()?;
            let parent_len = self.path.len();
            self.path = pointer_child(&self.path, &key);
            if !keys.insert(key.clone()) {
                return Err(self.err("json.duplicate_key", "duplicate object key"));
            }
            self.ws();
            if self.peek() != Some(b':') {
                return Err(self.err("json.syntax", "expected `:`"));
            }
            self.pos += 1;
            members.push((key, self.value(depth)?));
            self.path.truncate(parent_len);
            self.ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(JsonValue::Object(members));
                }
                _ => return Err(self.err("json.syntax", "expected `,` or `}`")),
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
    fn number(&mut self) -> Result<f64, AnimError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => {
                self.pos += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(self.err("json.syntax", "leading zeros are not allowed"));
                }
            }
            Some(b'1'..=b'9') => {
                self.digits();
            }
            _ => return Err(self.err("json.syntax", "expected a digit")),
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            if self.digits() == 0 {
                return Err(self.err("json.syntax", "expected a digit after `.`"));
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if self.digits() == 0 {
                return Err(self.err("json.syntax", "expected a digit in the exponent"));
            }
        }
        let text = std::str::from_utf8(&self.b[start..self.pos]).unwrap_or("");
        let number = text
            .parse::<f64>()
            .map_err(|_| self.err("json.syntax", "invalid number"))?;
        if !number.is_finite() {
            return Err(self.err("json.not_finite", "number is not finite"));
        }
        Ok(number)
    }
    fn hex_quad(&mut self) -> Result<u32, AnimError> {
        let hex = self.b.get(self.pos..self.pos + 4).unwrap_or(&[]);
        if hex.len() != 4 || !hex.iter().all(u8::is_ascii_hexdigit) {
            return Err(self.err("json.syntax", "expected 4 hex digits"));
        }
        self.pos += 4;
        Ok(hex
            .iter()
            .fold(0, |n, &b| n * 16 + (b as char).to_digit(16).unwrap()))
    }
    fn escape(&mut self) -> Result<char, AnimError> {
        let e = self
            .peek()
            .ok_or_else(|| self.err("json.syntax", "expected an escape"))?;
        self.pos += 1;
        Ok(match e {
            b'"' => '"',
            b'\\' => '\\',
            b'/' => '/',
            b'b' => '\u{8}',
            b'f' => '\u{c}',
            b'n' => '\n',
            b'r' => '\r',
            b't' => '\t',
            b'u' => {
                let mut point = self.hex_quad()?;
                if (0xd800..=0xdbff).contains(&point) {
                    if self.b.get(self.pos..self.pos + 2) != Some(b"\\u") {
                        return Err(self.err("json.syntax", "expected low surrogate escape"));
                    }
                    self.pos += 2;
                    let low = self.hex_quad()?;
                    if !(0xdc00..=0xdfff).contains(&low) {
                        return Err(self.err("json.syntax", "expected low surrogate"));
                    }
                    point = 0x10000 + ((point - 0xd800) << 10) + low - 0xdc00;
                }
                char::from_u32(point)
                    .ok_or_else(|| self.err("json.syntax", "invalid code point"))?
            }
            _ => return Err(self.err("json.syntax", "bad escape")),
        })
    }
    fn string(&mut self) -> Result<String, AnimError> {
        self.pos += 1;
        let mut out: Vec<u8> = Vec::new();
        loop {
            match self.peek() {
                None => return Err(self.err("json.syntax", "unterminated string")),
                Some(b'"') => {
                    self.pos += 1;
                    return String::from_utf8(out)
                        .map_err(|_| self.err("json.syntax", "invalid UTF-8"));
                }
                Some(b'\\') => {
                    self.pos += 1;
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(self.escape()?.encode_utf8(&mut buf).as_bytes());
                }
                Some(c) if c < 0x20 => {
                    return Err(self.err("json.syntax", "unescaped control character"));
                }
                Some(c) => {
                    out.push(c);
                    self.pos += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn code(text: &str) -> &'static str {
        parse(text).expect_err(text).code
    }

    #[test]
    fn json_structure_nested_strings() {
        assert_eq!(
            parse(r#" {"a":[null,true,{"b":[]},{}],"z":1} "#).unwrap(),
            JsonValue::Object(vec![
                (
                    "a".into(),
                    JsonValue::Array(vec![
                        JsonValue::Null,
                        JsonValue::Bool(true),
                        JsonValue::Object(vec![("b".into(), JsonValue::Array(vec![]))]),
                        JsonValue::Object(vec![])
                    ])
                ),
                ("z".into(), JsonValue::Number(1.0)),
            ])
        );
        for (source, expected) in [
            (r#""\"\\\/\b\f\n\r\t""#, "\"\\/\u{8}\u{c}\n\r\t"),
            (r#""é水🦀""#, "é水🦀"),
            (r#""\u0000\u00E9\u6c34\ud83e\udd80""#, "\0é水🦀"),
            (r#""\ud800\udc00\uDBFF\uDFFF""#, "\u{10000}\u{10ffff}"),
        ] {
            assert_eq!(parse(source).unwrap(), JsonValue::String(expected.into()));
        }
        assert!(parse(r#"[{"a":1},{"a":2}]"#).is_ok());
    }

    #[test]
    fn json_structure_malformed() {
        for source in [
            "",
            "[",
            "{",
            "[1,]",
            "[,1]",
            "[1 2]",
            "{\"a\":1,}",
            "{a:1}",
            "{\"a\" 1}",
            "{\"a\":}",
            "{\"a\":1 \"b\":2}",
            "[]{}",
            "/* comment */null",
            "\u{feff}null",
            "[truex]",
            "[01]",
            r#""\x20""#,
            r#""\u123""#,
            r#""\u12xz""#,
            r#""\uDC00""#,
            r#""\uD800""#,
            r#""\uD800x""#,
            r#""\uD800\u0041""#,
            r#""\uD800\uD800""#,
            r#""\uD800\uZZZZ""#,
            "\"abc\\",
            "\"abc",
        ] {
            assert_eq!(code(source), "json.syntax", "{source:?}");
        }
        for byte in 0..32u8 {
            assert_eq!(code(&format!("\"{}\"", char::from(byte))), "json.syntax");
        }
    }

    #[test]
    fn json_structure_paths_and_duplicates() {
        for source in [
            r#"{"x":[{"a/b~":0,"a/b~":1}]}"#,
            r#"{"x":[{"a/b~":0,"a\u002fb\u007e":1}]}"#,
        ] {
            let error = parse(source).unwrap_err();
            assert_eq!(error.code, "json.duplicate_key");
            assert_eq!(error.path, "/x/0/a~1b~0");
        }
        assert_eq!(parse(r#"{"": [0, "\q"]}"#).unwrap_err().path, "//1");
        assert_eq!(
            parse(r#"{"a": [], "b": [false, ?]}"#).unwrap_err().path,
            "/b/1"
        );
        assert_eq!(parse("[] trailing").unwrap_err().path, "");
        assert_eq!(pointer_child("/bones/0", "~/"), "/bones/0/~0~1");
    }

    #[test]
    fn json_structure_depth_and_overflow() {
        for (open, close) in [("[", "]"), ("{\"k\":", "}")] {
            let nested = |n| format!("{}0{}", open.repeat(n), close.repeat(n));
            assert!(parse(&nested(MAX_DEPTH)).is_ok());
            assert_eq!(code(&nested(MAX_DEPTH + 1)), "json.depth");
            assert_eq!(
                parse_rig(&nested(MAX_DEPTH + 1)).unwrap_err().code,
                "rig.schema"
            );
        }
        for source in ["1e309", "-1e309", "9e99999"] {
            assert_eq!(code(source), "json.not_finite");
            let nested = format!("{{\"v\":[{source}]}}");
            for (parser, expected) in [
                (
                    parse_rig as fn(&str) -> Result<JsonValue, AnimError>,
                    "rig.not_finite",
                ),
                (parse_clip, "clip.not_finite"),
            ] {
                let error = parser(&nested).unwrap_err();
                assert_eq!(error.code, expected);
                assert_eq!(error.path, "/v/0");
            }
        }
        assert_eq!(
            parse_clip(r#"{"a":1,"a":2}"#).unwrap_err().code,
            "clip.schema"
        );
        assert!(parse("1.7976931348623157e308").is_ok());
        assert_eq!(parse("1e-9999").unwrap(), JsonValue::Number(0.0));
        assert_eq!(
            parse("1.0000000000000002").unwrap(),
            JsonValue::Number(1.0000000000000002)
        );
    }

    #[test]
    fn json_tokens() {
        assert_eq!(parse("null").unwrap(), JsonValue::Null);
        assert_eq!(parse(" true ").unwrap(), JsonValue::Bool(true));
        assert_eq!(parse("\"\\u00e9\"").unwrap(), JsonValue::String("é".into()));
        assert_eq!(parse("-0.5e1").unwrap(), JsonValue::Number(-5.0));
        assert_eq!(parse("1e-7").unwrap(), JsonValue::Number(1e-7));
        let bad = ["-", "+1", ".5", "5.", "01", "1e", "1e+", "1.e2"];
        for text in bad {
            assert_eq!(code(text), "json.syntax", "{text:?}");
        }
        for text in ["1 2", "truex", "nullx", "true false", "{}#", "[1,]"] {
            assert_eq!(code(text), "json.syntax", "{text:?}");
        }
    }
}
