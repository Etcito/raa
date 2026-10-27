//! Strict RFC 8259 JSON reader. Numbers stay `f64` document values so
//! canonical writing never serializes rounded `f32` runtime values.

use crate::AnimError;

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
    };
    let v = p.value(0)?;
    p.ws();
    if p.pos < p.b.len() {
        return Err(p.err("json.syntax", "trailing input after the JSON value"));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.b.get(self.pos).copied()
    }
    fn err(&self, code: &'static str, msg: &str) -> AnimError {
        AnimError::new(code, format!("@{}", self.pos), msg)
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
            items.push(self.value(depth)?);
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
            self.ws();
            if self.peek() != Some(b':') {
                return Err(self.err("json.syntax", "expected `:`"));
            }
            self.pos += 1;
            members.push((key, self.value(depth)?));
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
        text.parse::<f64>()
            .map_err(|_| self.err("json.syntax", "invalid number"))
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
                let hex = self.b.get(self.pos..self.pos + 4).unwrap_or(&[]);
                if hex.len() != 4 || !hex.iter().all(u8::is_ascii_hexdigit) {
                    return Err(self.err("json.syntax", "expected 4 hex digits"));
                }
                self.pos += 4;
                let hex = std::str::from_utf8(hex).unwrap_or("");
                char::from_u32(u32::from_str_radix(hex, 16).unwrap())
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
