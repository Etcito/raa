//! SDF Modeler reference (fixture S8): reader, mapping v1 validation, count rules and the
//! SDF_REF_TOL comparison shared by `prims-report` and `tests/sdf_reference.rs`.
//!
//! The fixture's per-case metrics are already in RAA's frame, so [`compare`] never re-maps a
//! generated mesh; [`map_to_raa`] is the axisMap transform for meshes in the reference frame.
//! No per-case code, and the tolerances are fixed.

use std::fmt;

use crate::{PrimKind, PrimParams};
use raa_geom::analysis::{analyze, unique_positions_1e6};
use raa_geom::{GeomError, MeshData, ParamValue, Params, Quality};

/// SDF_REF_TOL: each bbox coordinate abs diff <= BBOX_TOL x max(1, reference diagonal).
pub const BBOX_TOL: f64 = 1e-5;
/// SDF_REF_TOL: area and volume relative diff <= REL_TOL.
pub const REL_TOL: f64 = 1e-4;
/// Relative tolerance for a paramMap/mapping term reproducing a case's refParams.
const PARAM_REL_TOL: f64 = 1e-6;
/// QUALITY.segments at which every count equals its reference default.
const LOD0_SEGMENTS: u32 = 24;
const MAX_JSON_DEPTH: usize = 64;
const NOT_CLOSED: &str = "reference mesh not closed: report to the lead";

/// A refusal of the reference module. `path` is a JSON pointer into the fixture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RefError {
    /// The fixture is not valid JSON or does not have the S8 shape (`prims.fixture_invalid`).
    Fixture { path: String, reason: String },
    /// The embedded mapping v1 violates a rule (`prims.mapping_invalid`).
    MappingInvalid { path: String, reason: String },
    /// A case's `counts` differ from RAA's count rules (`prims.reference_counts_mismatch`).
    CountsMismatch { path: String, reason: String },
}

impl RefError {
    pub fn code(&self) -> &'static str {
        match self {
            RefError::Fixture { .. } => "prims.fixture_invalid",
            RefError::MappingInvalid { .. } => "prims.mapping_invalid",
            RefError::CountsMismatch { .. } => "prims.reference_counts_mismatch",
        }
    }

    pub fn path(&self) -> &str {
        match self {
            RefError::Fixture { path, .. }
            | RefError::MappingInvalid { path, .. }
            | RefError::CountsMismatch { path, .. } => path,
        }
    }

    pub fn reason(&self) -> &str {
        match self {
            RefError::Fixture { reason, .. }
            | RefError::MappingInvalid { reason, .. }
            | RefError::CountsMismatch { reason, .. } => reason,
        }
    }

    fn fixture(path: impl Into<String>, reason: impl Into<String>) -> RefError {
        RefError::Fixture {
            path: path.into(),
            reason: reason.into(),
        }
    }

    fn mapping(path: impl Into<String>, reason: impl Into<String>) -> RefError {
        RefError::MappingInvalid {
            path: path.into(),
            reason: reason.into(),
        }
    }
}

impl fmt::Display for RefError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} at {}", self.code(), self.reason(), self.path())
    }
}

impl std::error::Error for RefError {}

/// The geometry-crate form of a reference refusal: `InvalidParam { kind: "fixture", name:
/// <json path>, reason: "<code>: <reason>" }`.
impl From<RefError> for GeomError {
    fn from(e: RefError) -> GeomError {
        GeomError::InvalidParam {
            kind: "fixture".to_string(),
            name: e.path().to_string(),
            reason: format!("{}: {}", e.code(), e.reason()),
        }
    }
}

// ------------------------------------------------------------------ count rules

/// One tessellation count rule: `count(s) = max(min, multiple * round(ref * s / (24 * multiple)))`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CountSpec {
    pub name: &'static str,
    pub reference: u32,
    pub min: u32,
    pub multiple: u32,
}

impl CountSpec {
    /// The count at QUALITY segments `segments` (clamped to [3, 256]).
    pub fn count(&self, segments: u32) -> u32 {
        count(self.reference, self.min, self.multiple, segments)
    }
}

/// raa-prims' own count table (one entry per kind, in `PrimKind::ALL` order). It mirrors the
/// `segments` rules of mapping v1 (tools/sdf-ref/README.md §3.2/§4); [`parse_fixture`] refuses
/// a fixture whose rules differ. Every port and `triangle_count` derive their counts from it.
pub const COUNT_TABLE: [(PrimKind, &[CountSpec]); 8] = [
    (PrimKind::Sphere, &[spec("nu", 40, 3), spec("nv", 20, 2)]),
    (PrimKind::Box, &[]),
    (PrimKind::RoundBox, &[spec("cells", 10, 1)]),
    (PrimKind::Capsule, &[spec("nu", 32, 3), spec("nCap", 10, 1)]),
    (PrimKind::Cylinder, &[spec("nu", 40, 3)]),
    (PrimKind::Cone, &[spec("nu", 40, 3)]),
    (PrimKind::Torus, &[spec("nu", 48, 3), spec("nv", 20, 3)]),
    (PrimKind::Ellipsoid, &[spec("nu", 40, 3), spec("nv", 20, 2)]),
];

const fn spec(name: &'static str, reference: u32, min: u32) -> CountSpec {
    CountSpec {
        name,
        reference,
        min,
        multiple: 1,
    }
}

/// The count rules of `kind` from [`COUNT_TABLE`] (`[]` for box).
pub fn count_rules(kind: PrimKind) -> &'static [CountSpec] {
    COUNT_TABLE
        .iter()
        .find(|(k, _)| *k == kind)
        .map_or(&[], |(_, rules)| rules)
}

/// `max(min, multiple * round(reference * s / (24 * multiple)))` with `s` clamped to [3, 256],
/// rounding half away from zero, evaluated in f64.
pub fn count(reference: u32, min: u32, multiple: u32, segments: u32) -> u32 {
    let s = Quality::new(segments).segments;
    let m = multiple.max(1);
    let q = f64::from(reference) * f64::from(s) / (f64::from(LOD0_SEGMENTS) * f64::from(m));
    // q <= u32::MAX * 256 / 24, far inside f64's exact integers; the product saturates.
    let rounded = q.round() as u64 * u64::from(m);
    u32::try_from(rounded).unwrap_or(u32::MAX).max(min)
}

/// `sdf_value = scale * a + add.scale * b`, in f64 and in exactly that order.
pub fn sdf_value(scale: f64, a: f64, add: Option<(f64, f64)>) -> f64 {
    match add {
        Some((add_scale, b)) => scale * a + add_scale * b,
        None => scale * a,
    }
}

// ------------------------------------------------------------------ minimal JSON reader

/// A parsed JSON value; objects keep their keys in document order (duplicates included, so
/// each consumer refuses them with its own code).
#[derive(Clone, Debug, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn type_name(&self) -> &'static str {
        match self {
            Json::Null => "null",
            Json::Bool(_) => "a boolean",
            Json::Num(_) => "a number",
            Json::Str(_) => "a string",
            Json::Arr(_) => "an array",
            Json::Obj(_) => "an object",
        }
    }

    /// Compact text of the value (numbers in Rust's shortest round-trip form).
    fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Num(n) => out.push_str(&format!("{n}")),
            Json::Str(s) => write_str(s, out),
            Json::Arr(items) => {
                out.push('[');
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    v.write(out);
                }
                out.push(']');
            }
            Json::Obj(members) => {
                out.push('{');
                for (i, (k, v)) in members.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_str(k, out);
                    out.push(':');
                    v.write(out);
                }
                out.push('}');
            }
        }
    }
}

fn write_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

struct JsonParser<'a> {
    text: &'a str,
    pos: usize,
}

fn parse_json(text: &str) -> Result<Json, RefError> {
    let mut p = JsonParser { text, pos: 0 };
    p.skip_ws();
    let v = p.value(0)?;
    p.skip_ws();
    if p.pos != text.len() {
        return Err(p.err("trailing characters after the JSON value"));
    }
    Ok(v)
}

impl JsonParser<'_> {
    fn err(&self, reason: &str) -> RefError {
        RefError::fixture("", format!("JSON syntax at byte {}: {reason}", self.pos))
    }

    fn peek(&self) -> Option<u8> {
        self.text.as_bytes().get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn eat(&mut self, b: u8) -> Result<(), RefError> {
        if self.peek() == Some(b) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.err(&format!("expected `{}`", b as char)))
        }
    }

    fn literal(&mut self, word: &str, v: Json) -> Result<Json, RefError> {
        if self.text[self.pos..].starts_with(word) {
            self.pos += word.len();
            Ok(v)
        } else {
            Err(self.err("unexpected token"))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, RefError> {
        if depth > MAX_JSON_DEPTH {
            return Err(self.err("nesting deeper than 64"));
        }
        match self.peek() {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => self.string().map(Json::Str),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'n') => self.literal("null", Json::Null),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => Err(self.err("unexpected character")),
            None => Err(self.err("unexpected end of input")),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json, RefError> {
        self.eat(b'{')?;
        let mut members = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Obj(members));
        }
        loop {
            self.skip_ws();
            let key = self.string()?;
            self.skip_ws();
            self.eat(b':')?;
            self.skip_ws();
            members.push((key, self.value(depth + 1)?));
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Json::Obj(members));
                }
                _ => return Err(self.err("expected `,` or `}`")),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Json, RefError> {
        self.eat(b'[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            self.skip_ws();
            items.push(self.value(depth + 1)?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return Err(self.err("expected `,` or `]`")),
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, RefError> {
        let digits = self
            .text
            .get(self.pos..self.pos + 4)
            .ok_or_else(|| self.err("truncated \\u escape"))?;
        let v = u32::from_str_radix(digits, 16).map_err(|_| self.err("bad \\u escape"))?;
        self.pos += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<String, RefError> {
        self.eat(b'"')?;
        let mut out = String::new();
        loop {
            let rest = &self.text[self.pos..];
            let Some(c) = rest.chars().next() else {
                return Err(self.err("unterminated string"));
            };
            self.pos += c.len_utf8();
            match c {
                '"' => return Ok(out),
                '\\' => {
                    let e = self.peek().ok_or_else(|| self.err("unterminated escape"))?;
                    self.pos += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                self.eat(b'\\')?;
                                self.eat(b'u')?;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err(self.err("unpaired surrogate"));
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else {
                                hi
                            };
                            out.push(char::from_u32(cp).ok_or_else(|| self.err("bad code point"))?);
                        }
                        _ => return Err(self.err("unknown escape")),
                    }
                }
                c if (c as u32) < 0x20 => return Err(self.err("control character in string")),
                c => out.push(c),
            }
        }
    }

    fn number(&mut self) -> Result<Json, RefError> {
        let start = self.pos;
        let digits = |p: &mut JsonParser<'_>| {
            let s = p.pos;
            while matches!(p.peek(), Some(b'0'..=b'9')) {
                p.pos += 1;
            }
            p.pos - s
        };
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        if self.peek() == Some(b'0') {
            self.pos += 1;
        } else if digits(self) == 0 {
            return Err(self.err("bad number"));
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            if digits(self) == 0 {
                return Err(self.err("bad number fraction"));
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if digits(self) == 0 {
                return Err(self.err("bad number exponent"));
            }
        }
        let v: f64 = self.text[start..self.pos]
            .parse()
            .map_err(|_| self.err("bad number"))?;
        if !v.is_finite() {
            return Err(self.err("number out of range"));
        }
        Ok(Json::Num(v))
    }
}

// ------------------------------------------------------------------ typed access

type Mk = fn(String, String) -> RefError;

fn mk_fixture(path: String, reason: String) -> RefError {
    RefError::Fixture { path, reason }
}

fn mk_mapping(path: String, reason: String) -> RefError {
    RefError::MappingInvalid { path, reason }
}

/// The members of an object, refusing duplicate keys and keys outside `allowed`, and requiring
/// every key of `required`.
fn object<'a>(
    v: &'a Json,
    path: &str,
    required: &[&str],
    allowed: &[&str],
    mk: Mk,
) -> Result<&'a [(String, Json)], RefError> {
    let Json::Obj(members) = v else {
        return Err(mk(
            path.into(),
            format!("expected an object, found {}", v.type_name()),
        ));
    };
    for (i, (k, _)) in members.iter().enumerate() {
        if members[..i].iter().any(|(p, _)| p == k) {
            return Err(mk(format!("{path}/{k}"), "duplicate key".into()));
        }
        if !allowed.is_empty() && !allowed.contains(&k.as_str()) {
            return Err(mk(format!("{path}/{k}"), "unknown key".into()));
        }
    }
    if let Some(missing) = required
        .iter()
        .find(|r| !members.iter().any(|(k, _)| k == *r))
    {
        return Err(mk(format!("{path}/{missing}"), "missing key".into()));
    }
    Ok(members)
}

/// Any object, duplicate keys refused, every key allowed.
fn any_object<'a>(v: &'a Json, path: &str, mk: Mk) -> Result<&'a [(String, Json)], RefError> {
    object(v, path, &[], &[], mk)
}

fn get<'a>(members: &'a [(String, Json)], key: &str) -> Option<&'a Json> {
    members.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

/// A required member (callers have already checked presence through `object`).
fn member<'a>(
    members: &'a [(String, Json)],
    key: &str,
    path: &str,
    mk: Mk,
) -> Result<&'a Json, RefError> {
    get(members, key).ok_or_else(|| mk(format!("{path}/{key}"), "missing key".into()))
}

fn string<'a>(v: &'a Json, path: &str, mk: Mk) -> Result<&'a str, RefError> {
    match v {
        Json::Str(s) => Ok(s),
        _ => Err(mk(
            path.into(),
            format!("expected a string, found {}", v.type_name()),
        )),
    }
}

fn number(v: &Json, path: &str, mk: Mk) -> Result<f64, RefError> {
    match v {
        Json::Num(n) => Ok(*n),
        _ => Err(mk(
            path.into(),
            format!("expected a number, found {}", v.type_name()),
        )),
    }
}

fn uint(v: &Json, path: &str, mk: Mk) -> Result<u32, RefError> {
    let n = number(v, path, mk)?;
    if n.fract() != 0.0 || !(0.0..=f64::from(u32::MAX)).contains(&n) {
        return Err(mk(
            path.into(),
            format!("expected a non-negative integer, found {n}"),
        ));
    }
    Ok(n as u32)
}

fn array<'a>(v: &'a Json, path: &str, mk: Mk) -> Result<&'a [Json], RefError> {
    match v {
        Json::Arr(items) => Ok(items),
        _ => Err(mk(
            path.into(),
            format!("expected an array, found {}", v.type_name()),
        )),
    }
}

fn vec3_f64(v: &Json, path: &str, mk: Mk) -> Result<[f64; 3], RefError> {
    let items = array(v, path, mk)?;
    if items.len() != 3 {
        return Err(mk(
            path.into(),
            format!("expected 3 numbers, found {}", items.len()),
        ));
    }
    let mut out = [0.0; 3];
    for (i, item) in items.iter().enumerate() {
        out[i] = number(item, &format!("{path}/{i}"), mk)?;
    }
    Ok(out)
}

fn axis_index(c: &str) -> Option<usize> {
    match c {
        "x" => Some(0),
        "y" => Some(1),
        "z" => Some(2),
        _ => None,
    }
}

const AXES: [&str; 3] = ["x", "y", "z"];

// ------------------------------------------------------------------ mapping v1

/// Shape parameter of an SDF kind (non-advanced specs of sdfm `basic.ts`); `true` = vec3.
type SdfParamSpec = (&'static str, bool);

/// The non-advanced shape parameters of every SDF kind mapping v1 may name
/// (sdfm@50b45d5 `src/core/sdf/primitives/basic.ts`, tools/sdf-ref/README.md §3).
const SDF_SHAPE_PARAMS: [(&str, &[SdfParamSpec]); 8] = [
    ("sphere", &[("radius", false)]),
    ("box", &[("size", true)]),
    ("roundBox", &[("size", true), ("radius", false)]),
    ("capsule", &[("height", false), ("radius", false)]),
    ("cylinder", &[("height", false), ("radius", false)]),
    (
        "cone",
        &[
            ("height", false),
            ("radiusBottom", false),
            ("radiusTop", false),
        ],
    ),
    ("torus", &[("majorRadius", false), ("minorRadius", false)]),
    ("ellipsoid", &[("radii", true)]),
];

/// The optional additive term of a params entry (amendment Q5).
#[derive(Clone, Debug, PartialEq)]
pub struct AddTerm {
    pub from: String,
    /// RAA-frame component of a vector `from`; `None` for a scalar.
    pub component: Option<usize>,
    pub scale: f64,
}

/// One `paramMap.<kind>.params` entry: `sdf_value = scale * S3[from] (+ add.scale * S3[add.from])`.
#[derive(Clone, Debug, PartialEq)]
pub struct ParamTerm {
    /// The SDF parameter name, `<p>` for a scalar, `<p>.x|.y|.z` (SDF frame) for a vec3.
    pub sdf_param: String,
    pub sdf_name: String,
    /// SDF-frame component of a vec3 SDF parameter.
    pub sdf_index: Option<usize>,
    pub from: String,
    pub component: Option<usize>,
    pub scale: f64,
    pub add: Option<AddTerm>,
}

fn s3_value(params: &Params, name: &str, component: Option<usize>) -> Option<f64> {
    match (params.get(name)?, component) {
        (ParamValue::Scalar(v), None) => Some(f64::from(*v)),
        (ParamValue::Vec3(v), Some(c)) => v.get(c).map(|x| f64::from(*x)),
        _ => None,
    }
}

impl ParamTerm {
    /// The SDF value for RAA `params` (f32 values widened to f64); `None` when a key is absent
    /// or has the wrong type.
    pub fn evaluate(&self, params: &Params) -> Option<f64> {
        let a = s3_value(params, &self.from, self.component)?;
        let add = match &self.add {
            Some(t) => Some((t.scale, s3_value(params, &t.from, t.component)?)),
            None => None,
        };
        Some(sdf_value(self.scale, a, add))
    }
}

/// A `segments` rule of the fixture mapping.
#[derive(Clone, Debug, PartialEq)]
pub struct CountRule {
    pub count: String,
    pub reference: u32,
    pub min: u32,
    pub multiple: u32,
    pub source: String,
}

impl CountRule {
    pub fn count(&self, segments: u32) -> u32 {
        count(self.reference, self.min, self.multiple, segments)
    }
}

/// A `constants` value (number, string or boolean).
#[derive(Clone, Debug, PartialEq)]
pub enum ConstValue {
    Num(f64),
    Str(String),
    Bool(bool),
}

/// The validated `paramMap.<kind>` entry.
#[derive(Clone, Debug, PartialEq)]
pub struct KindMapping {
    pub kind: PrimKind,
    pub sdf_kind: String,
    pub params: Vec<ParamTerm>,
    pub constants: Vec<(String, ConstValue)>,
    pub segments: Vec<CountRule>,
}

impl KindMapping {
    /// `(count name, value)` at QUALITY segments `segments`, in rule order.
    pub fn counts(&self, segments: u32) -> Vec<(String, u32)> {
        self.segments
            .iter()
            .map(|r| (r.count.clone(), r.count(segments)))
            .collect()
    }
}

/// The fixture's validated top-level `mapping` (mapping v1).
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceMapping {
    pub commit: String,
    /// `axis_map[k] = (sdf axis 0..2, sign ±1)` for RAA axis k.
    pub axis_map: [(u8, i8); 3],
    pub det: i8,
    /// Each kind with its `sdfKind`, in `PrimKind::ALL` order.
    pub kinds: Vec<(PrimKind, String)>,
    /// The full per-kind entries, in `PrimKind::ALL` order.
    pub entries: Vec<KindMapping>,
}

impl ReferenceMapping {
    pub fn kind(&self, kind: PrimKind) -> Option<&KindMapping> {
        self.entries.iter().find(|e| e.kind == kind)
    }

    /// The RAA axis that SDF axis `sdf_axis` maps to.
    fn raa_axis_of(&self, sdf_axis: usize) -> Option<usize> {
        self.axis_map
            .iter()
            .position(|(a, _)| usize::from(*a) == sdf_axis)
    }
}

/// Parses and validates a standalone mapping v1 object (the README §4 block).
pub fn parse_mapping(text: &str) -> Result<ReferenceMapping, RefError> {
    validate_mapping(&parse_json(text)?, "/mapping")
}

fn s3_is_vector(kind: PrimKind, name: &str) -> Option<bool> {
    let params = PrimParams::reference_set(kind, 'A').to_params();
    params.get(name).map(|v| matches!(v, ParamValue::Vec3(_)))
}

fn validate_mapping(v: &Json, path: &str) -> Result<ReferenceMapping, RefError> {
    let mk: Mk = mk_mapping;
    let top = object(
        v,
        path,
        &["version", "reference", "axisMap", "paramMap"],
        &["version", "reference", "axisMap", "paramMap"],
        mk,
    )?;
    let version = number(
        member(top, "version", path, mk)?,
        &format!("{path}/version"),
        mk,
    )?;
    if version != 1.0 {
        return Err(RefError::mapping(
            format!("{path}/version"),
            format!("version must be 1, found {version}"),
        ));
    }

    let rpath = format!("{path}/reference");
    let reference = object(
        member(top, "reference", path, mk)?,
        &rpath,
        &["repo", "path", "commit"],
        &["repo", "path", "commit"],
        mk,
    )?;
    for key in ["repo", "path"] {
        if string(
            member(reference, key, &rpath, mk)?,
            &format!("{rpath}/{key}"),
            mk,
        )?
        .is_empty()
        {
            return Err(RefError::mapping(
                format!("{rpath}/{key}"),
                "must not be empty",
            ));
        }
    }
    let commit = string(
        member(reference, "commit", &rpath, mk)?,
        &format!("{rpath}/commit"),
        mk,
    )?;
    if commit.len() != 40
        || !commit
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Err(RefError::mapping(
            format!("{rpath}/commit"),
            format!("commit must be 40 lowercase hex, found `{commit}`"),
        ));
    }

    let (axis_map, det) = validate_axis_map(
        member(top, "axisMap", path, mk)?,
        &format!("{path}/axisMap"),
    )?;
    let mut mapping = ReferenceMapping {
        commit: commit.to_string(),
        axis_map,
        det,
        kinds: Vec::new(),
        entries: Vec::new(),
    };

    let ppath = format!("{path}/paramMap");
    let kinds = any_object(member(top, "paramMap", path, mk)?, &ppath, mk)?;
    for (name, _) in kinds {
        if PrimKind::parse(name).is_none() {
            return Err(RefError::mapping(
                format!("{ppath}/{name}"),
                "not an S3 kind",
            ));
        }
    }
    for kind in PrimKind::ALL {
        let kpath = format!("{ppath}/{}", kind.as_str());
        let entry =
            get(kinds, kind.as_str()).ok_or_else(|| RefError::mapping(&kpath, "kind missing"))?;
        let km = validate_kind(&mapping, kind, entry, &kpath)?;
        if mapping.entries.iter().any(|e| e.sdf_kind == km.sdf_kind) {
            return Err(RefError::mapping(
                format!("{kpath}/sdfKind"),
                format!("sdfKind `{}` already used", km.sdf_kind),
            ));
        }
        mapping.kinds.push((kind, km.sdf_kind.clone()));
        mapping.entries.push(km);
    }
    Ok(mapping)
}

fn validate_axis_map(v: &Json, path: &str) -> Result<([(u8, i8); 3], i8), RefError> {
    let mk: Mk = mk_mapping;
    let keys = ["x", "y", "z", "det"];
    let members = object(v, path, &keys, &keys, mk)?;
    let mut map = [(0u8, 1i8); 3];
    for (k, axis) in AXES.iter().enumerate() {
        let apath = format!("{path}/{axis}");
        let s = string(member(members, axis, path, mk)?, &apath, mk)?;
        let (sign, rest) = match s.as_bytes().first() {
            Some(b'+') => (1i8, &s[1..]),
            Some(b'-') => (-1i8, &s[1..]),
            _ => {
                return Err(RefError::mapping(
                    apath,
                    format!("expected a signed axis like \"+x\", found `{s}`"),
                ));
            }
        };
        let a = axis_index(rest).ok_or_else(|| {
            RefError::mapping(
                &apath,
                format!("expected a signed axis like \"+x\", found `{s}`"),
            )
        })?;
        if map[..k].iter().any(|(b, _)| usize::from(*b) == a) {
            return Err(RefError::mapping(
                apath,
                "axisMap is not a signed permutation (axis used twice)",
            ));
        }
        map[k] = (a as u8, sign);
    }
    // det of the signed permutation matrix M[k][axis_k] = sign_k: product of signs x parity.
    let perm = map.map(|(a, _)| a);
    let inversions = (0..3)
        .flat_map(|i| (i + 1..3).map(move |j| (i, j)))
        .filter(|&(i, j)| perm[i] > perm[j])
        .count();
    let parity: i8 = if inversions % 2 == 0 { 1 } else { -1 };
    let computed = parity * map[0].1 * map[1].1 * map[2].1;
    let dpath = format!("{path}/det");
    let det = number(member(members, "det", path, mk)?, &dpath, mk)?;
    if det != 1.0 && det != -1.0 {
        return Err(RefError::mapping(
            dpath,
            format!("det must be 1 or -1, found {det}"),
        ));
    }
    if det != f64::from(computed) {
        return Err(RefError::mapping(
            dpath,
            format!("det {det} does not match the axisMap determinant {computed}"),
        ));
    }
    Ok((map, computed))
}

fn validate_kind(
    m: &ReferenceMapping,
    kind: PrimKind,
    v: &Json,
    path: &str,
) -> Result<KindMapping, RefError> {
    let mk: Mk = mk_mapping;
    let keys = ["sdfKind", "params", "constants", "segments"];
    let members = object(v, path, &keys, &keys, mk)?;
    let spath = format!("{path}/sdfKind");
    let sdf_kind = string(member(members, "sdfKind", path, mk)?, &spath, mk)?;
    let shape = SDF_SHAPE_PARAMS
        .iter()
        .find(|(k, _)| *k == sdf_kind)
        .map(|(_, p)| *p)
        .ok_or_else(|| {
            RefError::mapping(
                &spath,
                format!("`{sdf_kind}` is not a basic leafGeometry kind"),
            )
        })?;
    // Every SDF key of the kind: `<p>` for scalars, `<p>.x|.y|.z` for vec3 (SDF frame).
    let sdf_keys: Vec<(String, &str, Option<usize>)> = shape
        .iter()
        .flat_map(|&(name, is_vec)| {
            let comps: Vec<Option<usize>> = if is_vec {
                vec![Some(0), Some(1), Some(2)]
            } else {
                vec![None]
            };
            comps.into_iter().map(move |c| match c {
                Some(i) => (format!("{name}.{}", AXES[i]), name, Some(i)),
                None => (name.to_string(), name, None),
            })
        })
        .collect();
    let lookup = |key: &str, at: &str| {
        sdf_keys.iter().find(|(k, _, _)| k == key).ok_or_else(|| {
            RefError::mapping(
                at,
                format!("`{key}` is not a shape parameter of SDF kind `{sdf_kind}`"),
            )
        })
    };

    let ppath = format!("{path}/params");
    let mut params = Vec::new();
    for (key, entry) in any_object(member(members, "params", path, mk)?, &ppath, mk)? {
        let epath = format!("{ppath}/{key}");
        let (_, sdf_name, sdf_index) = lookup(key, &epath)?;
        params.push(parse_term(
            m, kind, key, sdf_name, *sdf_index, entry, &epath,
        )?);
    }

    let cpath = format!("{path}/constants");
    let mut constants = Vec::new();
    for (key, value) in any_object(member(members, "constants", path, mk)?, &cpath, mk)? {
        let at = format!("{cpath}/{key}");
        lookup(key, &at)?;
        if params.iter().any(|t| t.sdf_param == *key) {
            return Err(RefError::mapping(
                at,
                "covered by both params and constants",
            ));
        }
        let c = match value {
            Json::Num(n) => ConstValue::Num(*n),
            Json::Str(s) => ConstValue::Str(s.clone()),
            Json::Bool(b) => ConstValue::Bool(*b),
            other => {
                return Err(RefError::mapping(
                    at,
                    format!(
                        "a constant must be a number, string or boolean, found {}",
                        other.type_name()
                    ),
                ));
            }
        };
        constants.push((key.clone(), c));
    }
    for (key, _, _) in &sdf_keys {
        if !params.iter().any(|t| t.sdf_param == *key) && !constants.iter().any(|(k, _)| k == key) {
            return Err(RefError::mapping(
                format!("{ppath}/{key}"),
                "SDF shape parameter not covered by params or constants",
            ));
        }
    }

    // Each S3 key (per component for vectors) is consumed exactly once by a main term.
    for &name in kind.required_param_names() {
        let comps: &[Option<usize>] = if s3_is_vector(kind, name) == Some(true) {
            &[Some(0), Some(1), Some(2)]
        } else {
            &[None]
        };
        for &c in comps {
            let uses = params
                .iter()
                .filter(|t| t.from == name && t.component == c)
                .count();
            if uses != 1 {
                let label = c.map_or(name.to_string(), |i| format!("{name}.{}", AXES[i]));
                return Err(RefError::mapping(
                    &ppath,
                    format!(
                        "S3 key `{label}` is consumed {uses} times by a main term (must be exactly 1)"
                    ),
                ));
            }
        }
    }

    let segments = validate_segments(
        kind,
        member(members, "segments", path, mk)?,
        &format!("{path}/segments"),
    )?;
    Ok(KindMapping {
        kind,
        sdf_kind: sdf_kind.to_string(),
        params,
        constants,
        segments,
    })
}

/// Reads `from`/`component` and checks them against the entry's SDF component.
fn parse_source(
    m: &ReferenceMapping,
    kind: PrimKind,
    members: &[(String, Json)],
    sdf_index: Option<usize>,
    path: &str,
) -> Result<(String, Option<usize>), RefError> {
    let mk: Mk = mk_mapping;
    let fpath = format!("{path}/from");
    let from = string(member(members, "from", path, mk)?, &fpath, mk)?;
    let is_vec = s3_is_vector(kind, from).ok_or_else(|| {
        RefError::mapping(
            &fpath,
            format!("`{from}` is not an S3 key of {}", kind.as_str()),
        )
    })?;
    let cpath = format!("{path}/component");
    let component = match member(members, "component", path, mk)? {
        Json::Null => None,
        Json::Str(s) => Some(axis_index(s).ok_or_else(|| {
            RefError::mapping(
                &cpath,
                format!("component must be \"x\", \"y\", \"z\" or null, found `{s}`"),
            )
        })?),
        other => {
            return Err(RefError::mapping(
                cpath,
                format!(
                    "component must be \"x\", \"y\", \"z\" or null, found {}",
                    other.type_name()
                ),
            ));
        }
    };
    match (is_vec, component, sdf_index) {
        (true, None, _) => Err(RefError::mapping(
            cpath,
            format!("component null on the vector S3 key `{from}`"),
        )),
        (false, Some(_), _) => Err(RefError::mapping(
            cpath,
            format!("component on the scalar S3 key `{from}` must be null"),
        )),
        (true, Some(_), None) => Err(RefError::mapping(
            cpath,
            "a vector S3 key cannot feed a scalar SDF parameter",
        )),
        (false, None, Some(_)) => Err(RefError::mapping(
            fpath,
            "a vec3 SDF component needs a vector S3 key",
        )),
        (true, Some(c), Some(a)) => {
            if m.raa_axis_of(a) != Some(c) {
                return Err(RefError::mapping(
                    cpath,
                    format!(
                        "component `{}` is not the RAA axis axisMap gives for SDF axis `{}`",
                        AXES[c], AXES[a]
                    ),
                ));
            }
            Ok((from.to_string(), Some(c)))
        }
        (false, None, None) => Ok((from.to_string(), None)),
    }
}

fn parse_term(
    m: &ReferenceMapping,
    kind: PrimKind,
    key: &str,
    sdf_name: &str,
    sdf_index: Option<usize>,
    v: &Json,
    path: &str,
) -> Result<ParamTerm, RefError> {
    let mk: Mk = mk_mapping;
    let members = object(
        v,
        path,
        &["from", "component", "scale"],
        &["from", "component", "scale", "add"],
        mk,
    )?;
    let (from, component) = parse_source(m, kind, members, sdf_index, path)?;
    let spath = format!("{path}/scale");
    let scale = number(member(members, "scale", path, mk)?, &spath, mk)?;
    if !scale.is_finite() || scale <= 0.0 {
        return Err(RefError::mapping(
            spath,
            format!("scale must be finite and > 0, found {scale}"),
        ));
    }
    let add = match get(members, "add") {
        None => None,
        Some(a) => {
            let apath = format!("{path}/add");
            if let Json::Obj(inner) = a
                && get(inner, "add").is_some()
            {
                return Err(RefError::mapping(format!("{apath}/add"), "nested add"));
            }
            let keys = ["from", "component", "scale"];
            let am = object(a, &apath, &keys, &keys, mk)?;
            let (afrom, acomp) = parse_source(m, kind, am, sdf_index, &apath)?;
            let aspath = format!("{apath}/scale");
            let ascale = number(member(am, "scale", &apath, mk)?, &aspath, mk)?;
            if !ascale.is_finite() || ascale == 0.0 {
                return Err(RefError::mapping(
                    aspath,
                    format!("add.scale must be finite and non-zero, found {ascale}"),
                ));
            }
            Some(AddTerm {
                from: afrom,
                component: acomp,
                scale: ascale,
            })
        }
    };
    Ok(ParamTerm {
        sdf_param: key.to_string(),
        sdf_name: sdf_name.to_string(),
        sdf_index,
        from,
        component,
        scale,
        add,
    })
}

fn validate_segments(kind: PrimKind, v: &Json, path: &str) -> Result<Vec<CountRule>, RefError> {
    let mk: Mk = mk_mapping;
    let keys = ["count", "ref", "min", "multiple", "source"];
    let mut rules: Vec<CountRule> = Vec::new();
    for (i, item) in array(v, path, mk)?.iter().enumerate() {
        let rpath = format!("{path}/{i}");
        let r = object(item, &rpath, &keys, &keys, mk)?;
        let field = |k: &str| member(r, k, &rpath, mk).map(|v| (v, format!("{rpath}/{k}")));
        let (c, cpath) = field("count")?;
        let name = string(c, &cpath, mk)?;
        if name.is_empty() || rules.iter().any(|x| x.count == name) {
            return Err(RefError::mapping(
                cpath,
                format!("count name `{name}` empty or repeated"),
            ));
        }
        let mut ints = [0u32; 3];
        for (slot, k) in ints.iter_mut().zip(["ref", "min", "multiple"]) {
            let (v, p) = field(k)?;
            *slot = uint(v, &p, mk)?;
            if *slot == 0 {
                return Err(RefError::mapping(p, format!("{k} must be >= 1")));
            }
        }
        let (s, spath) = field("source")?;
        let source = string(s, &spath, mk)?;
        if source.is_empty() {
            return Err(RefError::mapping(spath, "source must not be empty"));
        }
        let rule = CountRule {
            count: name.to_string(),
            reference: ints[0],
            min: ints[1],
            multiple: ints[2],
            source: source.to_string(),
        };
        if rule.count(LOD0_SEGMENTS) != rule.reference {
            return Err(RefError::mapping(
                rpath,
                format!(
                    "count at LOD0 is {}, not ref {}",
                    rule.count(LOD0_SEGMENTS),
                    rule.reference
                ),
            ));
        }
        rules.push(rule);
    }
    let own = count_rules(kind);
    let same = own.len() == rules.len()
        && own.iter().zip(&rules).all(|(o, r)| {
            o.name == r.count
                && o.reference == r.reference
                && o.min == r.min
                && o.multiple == r.multiple
        });
    if !same {
        return Err(RefError::mapping(
            path,
            format!(
                "segments differ from raa-prims' count table for {}: {own:?}",
                kind.as_str()
            ),
        ));
    }
    Ok(rules)
}

// ------------------------------------------------------------------ fixture S8

/// `paramMap` additive term in S8 names: `+ scale * params[raa_name][raa_index]`.
#[derive(Clone, Debug, PartialEq)]
pub struct ParamMapAdd {
    pub raa_name: String,
    pub raa_index: Option<usize>,
    pub scale: f64,
}

/// One S8 `paramMap` entry: `refParams[ref_name][ref_index] = raa_name.is_none() ? value :
/// scale * params[raa_name][raa_index] (+ add.scale * params[add.raa_name][add.raa_index])`.
#[derive(Clone, Debug, PartialEq)]
pub struct ParamMapEntry {
    pub ref_name: String,
    pub ref_index: Option<usize>,
    pub raa_name: Option<String>,
    pub raa_index: Option<usize>,
    pub scale: f64,
    pub value: Option<f64>,
    pub add: Option<ParamMapAdd>,
}

impl ParamMapEntry {
    /// The reference value this entry gives for RAA `params`.
    pub fn evaluate(&self, params: &Params) -> Option<f64> {
        let Some(name) = &self.raa_name else {
            return self.value;
        };
        let a = s3_value(params, name, self.raa_index)?;
        let add = match &self.add {
            Some(t) => Some((t.scale, s3_value(params, &t.raa_name, t.raa_index)?)),
            None => None,
        };
        Some(sdf_value(self.scale, a, add))
    }
}

/// One S8 case; metrics are in RAA's frame.
#[derive(Clone, Debug, PartialEq)]
pub struct RefCase {
    pub kind: PrimKind,
    pub set: char,
    pub quality: u32,
    pub params: Params,
    pub ref_params: Vec<(String, Vec<f64>)>,
    /// The exact `leafGeometry` arguments, as compact JSON text (printed on failure only).
    pub sdf_input: String,
    pub counts: Vec<(String, u32)>,
    pub unique_positions: u32,
    pub triangles: u32,
    pub bbox_min: [f64; 3],
    pub bbox_max: [f64; 3],
    pub area: f64,
    pub volume: f64,
    pub closed: bool,
}

impl RefCase {
    fn ref_param(&self, name: &str, index: Option<usize>) -> Option<f64> {
        let values = &self.ref_params.iter().find(|(n, _)| n == name)?.1;
        match index {
            None if values.len() == 1 => values.first().copied(),
            None => None,
            Some(i) => values.get(i).copied(),
        }
    }
}

/// The parsed and validated S8 fixture.
#[derive(Clone, Debug, PartialEq)]
pub struct RefFixture {
    pub command: String,
    pub sdf_modeler_revision: String,
    /// Entry k: (reference axis, sign) that becomes RAA axis k.
    pub axis_map: [(usize, f64); 3],
    pub param_map: Vec<(PrimKind, Vec<ParamMapEntry>)>,
    pub cases: Vec<RefCase>,
    mapping: ReferenceMapping,
}

impl RefFixture {
    /// The validated top-level `mapping` (mapping v1).
    pub fn mapping(&self) -> &ReferenceMapping {
        &self.mapping
    }

    pub fn case(&self, kind: PrimKind, set: char, quality: u32) -> Option<&RefCase> {
        self.cases
            .iter()
            .find(|c| c.kind == kind && c.set == set && c.quality == quality)
    }
}

const FIXTURE_KEYS: [&str; 10] = [
    "generatedBy",
    "command",
    "regenerate",
    "sdfModelerRevision",
    "leafMeshSha256",
    "axisMap",
    "paramMap",
    "mapping",
    "kinds",
    "cases",
];
const FIXTURE_REQUIRED: [&str; 6] = [
    "command",
    "sdfModelerRevision",
    "axisMap",
    "paramMap",
    "mapping",
    "cases",
];
const CASE_KEYS: [&str; 14] = [
    "kind",
    "set",
    "quality",
    "params",
    "refParams",
    "sdfInput",
    "counts",
    "uniquePositions",
    "triangles",
    "bboxMin",
    "bboxMax",
    "area",
    "volume",
    "closed",
];
/// Number of S8 cases: 8 kinds x sets A/B x LOW/LOD0/HIGH.
pub const CASE_COUNT: usize = 48;

/// Parses fixture S8 with its own JSON reader and validates it: the mapping v1 rules
/// (`prims.mapping_invalid`), each case's `counts` against the count rules
/// (`prims.reference_counts_mismatch`) and the S8 shape (`prims.fixture_invalid`).
pub fn parse_fixture(text: &str) -> Result<RefFixture, RefError> {
    let mk: Mk = mk_fixture;
    let root = parse_json(text)?;
    let top = object(&root, "", &FIXTURE_REQUIRED, &FIXTURE_KEYS, mk)?;
    let mapping = validate_mapping(member(top, "mapping", "", mk)?, "/mapping")?;

    let command = string(member(top, "command", "", mk)?, "/command", mk)?.to_string();
    if let Some(r) = get(top, "regenerate")
        && string(r, "/regenerate", mk)? != command
    {
        return Err(RefError::fixture("/regenerate", "differs from command"));
    }
    let revision = string(
        member(top, "sdfModelerRevision", "", mk)?,
        "/sdfModelerRevision",
        mk,
    )?
    .to_string();
    if revision != mapping.commit {
        return Err(RefError::fixture(
            "/sdfModelerRevision",
            "differs from mapping.reference.commit",
        ));
    }

    let axes = array(member(top, "axisMap", "", mk)?, "/axisMap", mk)?;
    if axes.len() != 3 {
        return Err(RefError::fixture("/axisMap", "expected 3 entries"));
    }
    let mut axis_map = [(0usize, 1.0f64); 3];
    for (k, a) in axes.iter().enumerate() {
        let path = format!("/axisMap/{k}");
        let o = object(a, &path, &["axis", "sign"], &["axis", "sign"], mk)?;
        let axis = string(member(o, "axis", &path, mk)?, &format!("{path}/axis"), mk)?;
        let sign = number(member(o, "sign", &path, mk)?, &format!("{path}/sign"), mk)?;
        let (ma, ms) = mapping.axis_map[k];
        if axis_index(axis) != Some(usize::from(ma)) || sign != f64::from(ms) {
            return Err(RefError::fixture(path, "disagrees with mapping.axisMap"));
        }
        axis_map[k] = (usize::from(ma), sign);
    }

    let pm = any_object(member(top, "paramMap", "", mk)?, "/paramMap", mk)?;
    let mut param_map = Vec::new();
    for kind in PrimKind::ALL {
        let path = format!("/paramMap/{}", kind.as_str());
        let entries =
            get(pm, kind.as_str()).ok_or_else(|| RefError::fixture(&path, "kind missing"))?;
        let parsed = array(entries, &path, mk)?
            .iter()
            .enumerate()
            .map(|(i, e)| parse_param_map_entry(e, &format!("{path}/{i}")))
            .collect::<Result<Vec<_>, _>>()?;
        param_map.push((kind, parsed));
    }
    if let Some((k, _)) = pm.iter().find(|(k, _)| PrimKind::parse(k).is_none()) {
        return Err(RefError::fixture(
            format!("/paramMap/{k}"),
            "not an S3 kind",
        ));
    }

    let mut cases: Vec<RefCase> = Vec::new();
    for (i, c) in array(member(top, "cases", "", mk)?, "/cases", mk)?
        .iter()
        .enumerate()
    {
        let path = format!("/cases/{i}");
        let case = parse_case(&mapping, c, &path)?;
        if cases
            .iter()
            .any(|o| o.kind == case.kind && o.set == case.set && o.quality == case.quality)
        {
            return Err(RefError::fixture(path, "duplicate kind/set/quality case"));
        }
        cases.push(case);
    }
    if cases.len() != CASE_COUNT {
        return Err(RefError::fixture(
            "/cases",
            format!("expected {CASE_COUNT} cases, found {}", cases.len()),
        ));
    }
    Ok(RefFixture {
        command,
        sdf_modeler_revision: revision,
        axis_map,
        param_map,
        cases,
        mapping,
    })
}

fn opt_index(v: &Json, path: &str) -> Result<Option<usize>, RefError> {
    match v {
        Json::Null => Ok(None),
        _ => uint(v, path, mk_fixture).map(|i| Some(i as usize)),
    }
}

fn opt_string(v: &Json, path: &str) -> Result<Option<String>, RefError> {
    match v {
        Json::Null => Ok(None),
        _ => string(v, path, mk_fixture).map(|s| Some(s.to_string())),
    }
}

fn parse_param_map_entry(v: &Json, path: &str) -> Result<ParamMapEntry, RefError> {
    let mk: Mk = mk_fixture;
    let keys = ["ref", "refIndex", "raa", "index", "scale", "value"];
    let o = object(
        v,
        path,
        &keys,
        &["ref", "refIndex", "raa", "index", "scale", "value", "add"],
        mk,
    )?;
    let at = |k: &str| format!("{path}/{k}");
    let value = match member(o, "value", path, mk)? {
        Json::Null => None,
        n => Some(number(n, &at("value"), mk)?),
    };
    let raa_name = opt_string(member(o, "raa", path, mk)?, &at("raa"))?;
    if raa_name.is_none() == value.is_none() {
        return Err(RefError::fixture(
            path,
            "exactly one of raa and value must be non-null",
        ));
    }
    let add = match get(o, "add") {
        None => None,
        Some(a) => {
            let ap = at("add");
            let ao = object(
                a,
                &ap,
                &["raa", "index", "scale"],
                &["raa", "index", "scale"],
                mk,
            )?;
            Some(ParamMapAdd {
                raa_name: string(member(ao, "raa", &ap, mk)?, &format!("{ap}/raa"), mk)?
                    .to_string(),
                raa_index: opt_index(member(ao, "index", &ap, mk)?, &format!("{ap}/index"))?,
                scale: number(member(ao, "scale", &ap, mk)?, &format!("{ap}/scale"), mk)?,
            })
        }
    };
    Ok(ParamMapEntry {
        ref_name: string(member(o, "ref", path, mk)?, &at("ref"), mk)?.to_string(),
        ref_index: opt_index(member(o, "refIndex", path, mk)?, &at("refIndex"))?,
        raa_name,
        raa_index: opt_index(member(o, "index", path, mk)?, &at("index"))?,
        scale: number(member(o, "scale", path, mk)?, &at("scale"), mk)?,
        value,
        add,
    })
}

fn parse_case(mapping: &ReferenceMapping, v: &Json, path: &str) -> Result<RefCase, RefError> {
    let mk: Mk = mk_fixture;
    let o = object(v, path, &CASE_KEYS, &CASE_KEYS, mk)?;
    let field = |k: &str| member(o, k, path, mk).map(|v| (v, format!("{path}/{k}")));

    let (k, kp) = field("kind")?;
    let kind_name = string(k, &kp, mk)?;
    let kind = PrimKind::parse(kind_name)
        .ok_or_else(|| RefError::fixture(&kp, format!("`{kind_name}` is not an S3 kind")))?;
    let (s, sp) = field("set")?;
    let set = match string(s, &sp, mk)? {
        "A" => 'A',
        "B" => 'B',
        other => {
            return Err(RefError::fixture(
                sp,
                format!("set must be \"A\" or \"B\", found `{other}`"),
            ));
        }
    };
    let (q, qp) = field("quality")?;
    let quality = uint(q, &qp, mk)?;
    if ![Quality::LOW, Quality::LOD0, Quality::HIGH]
        .iter()
        .any(|x| x.segments == quality)
    {
        return Err(RefError::fixture(
            qp,
            format!("quality must be 8, 24 or 64, found {quality}"),
        ));
    }

    let (p, pp) = field("params")?;
    let mut params = Params::new();
    for (name, value) in any_object(p, &pp, mk)? {
        let at = format!("{pp}/{name}");
        let pv = match value {
            Json::Arr(_) => ParamValue::Vec3(vec3_f64(value, &at, mk)?.map(|x| x as f32)),
            _ => ParamValue::Scalar(number(value, &at, mk)? as f32),
        };
        params.set(name, pv);
    }

    let (r, rp) = field("refParams")?;
    let mut ref_params = Vec::new();
    for (name, value) in any_object(r, &rp, mk)? {
        let at = format!("{rp}/{name}");
        let values = match value {
            Json::Arr(items) => items
                .iter()
                .enumerate()
                .map(|(i, x)| number(x, &format!("{at}/{i}"), mk))
                .collect::<Result<Vec<_>, _>>()?,
            _ => vec![number(value, &at, mk)?],
        };
        ref_params.push((name.clone(), values));
    }

    let (si, sip) = field("sdfInput")?;
    any_object(si, &sip, mk)?;
    let mut sdf_input = String::new();
    si.write(&mut sdf_input);

    let (c, cp) = field("counts")?;
    let mut counts = Vec::new();
    for (name, value) in any_object(c, &cp, mk)? {
        counts.push((name.clone(), uint(value, &format!("{cp}/{name}"), mk)?));
    }
    let expected = mapping
        .kind(kind)
        .map(|m| m.counts(quality))
        .unwrap_or_default();
    if counts != expected {
        return Err(RefError::CountsMismatch {
            path: cp,
            reason: format!(
                "{} quality {quality}: fixture counts {counts:?}, RAA count rules give {expected:?}",
                kind.as_str()
            ),
        });
    }

    let (u, up) = field("uniquePositions")?;
    let (t, tp) = field("triangles")?;
    let (bmin, bminp) = field("bboxMin")?;
    let (bmax, bmaxp) = field("bboxMax")?;
    let (a, ap) = field("area")?;
    let (vo, vp) = field("volume")?;
    let (cl, clp) = field("closed")?;
    let Json::Bool(closed) = cl else {
        return Err(RefError::fixture(
            clp,
            format!("expected a boolean, found {}", cl.type_name()),
        ));
    };
    Ok(RefCase {
        kind,
        set,
        quality,
        params,
        ref_params,
        sdf_input,
        counts,
        unique_positions: uint(u, &up, mk)?,
        triangles: uint(t, &tp, mk)?,
        bbox_min: vec3_f64(bmin, &bminp, mk)?,
        bbox_max: vec3_f64(bmax, &bmaxp, mk)?,
        area: number(a, &ap, mk)?,
        volume: number(vo, &vp, mk)?,
        closed: *closed,
    })
}

// ------------------------------------------------------------------ comparison

/// Result of [`compare`]; `message` lists every failed rule with its numbers ("" when ok).
#[derive(Clone, Debug, PartialEq)]
pub struct RefDelta {
    pub unique_positions_eq: bool,
    pub triangles_eq: bool,
    pub bbox_max_abs_err: f64,
    pub area_rel_err: f64,
    pub volume_rel_err: f64,
    pub ok: bool,
    pub message: String,
}

fn rel_err(got: f64, want: f64) -> f64 {
    if got == want {
        0.0
    } else {
        (got - want).abs() / want.abs()
    }
}

/// `err <= tol`, false for NaN.
fn within(err: f64, tol: f64) -> bool {
    err <= tol
}

fn params_close(got: f64, want: f64) -> bool {
    (got - want).abs() <= PARAM_REL_TOL * got.abs().max(want.abs())
}

/// Compares a generated RAA mesh with a fixture case under SDF_REF_TOL. The case metrics are
/// in RAA's frame already, so `mesh` is compared as is. Also checks that the case params are
/// the S3 reference set and that `paramMap` and the mapping reproduce `refParams`.
pub fn compare(fx: &RefFixture, case: &RefCase, mesh: &MeshData) -> RefDelta {
    let mut fails: Vec<String> = Vec::new();
    if !case.closed {
        fails.push(NOT_CLOSED.to_string());
    }
    if let Err(e) = mesh.validate() {
        fails.push(format!("generated mesh invalid: {e}"));
    }
    if case.params != PrimParams::reference_set(case.kind, case.set).to_params() {
        fails.push(format!("case params are not S3 reference set {}", case.set));
    }
    let entries = fx
        .param_map
        .iter()
        .find(|(k, _)| *k == case.kind)
        .map(|(_, e)| e.as_slice());
    for e in entries.unwrap_or_default() {
        let got = e.evaluate(&case.params);
        let want = case.ref_param(&e.ref_name, e.ref_index);
        if !matches!((got, want), (Some(g), Some(w)) if params_close(g, w)) {
            fails.push(format!(
                "paramMap {}[{:?}] gives {got:?}, refParams has {want:?}",
                e.ref_name, e.ref_index
            ));
        }
    }
    if let Some(km) = fx.mapping.kind(case.kind) {
        for t in &km.params {
            let got = t.evaluate(&case.params);
            let want = case.ref_param(&t.sdf_name, t.sdf_index);
            if !matches!((got, want), (Some(g), Some(w)) if params_close(g, w)) {
                fails.push(format!(
                    "mapping {} gives {got:?}, refParams has {want:?}",
                    t.sdf_param
                ));
            }
        }
        for (name, c) in &km.constants {
            if let ConstValue::Num(n) = c
                && case.ref_param(name, None) != Some(*n)
            {
                fails.push(format!(
                    "mapping constant {name} = {n} differs from refParams"
                ));
            }
        }
    }

    let report = analyze(mesh);
    let unique = unique_positions_1e6(mesh);
    let unique_positions_eq = unique == case.unique_positions;
    if !unique_positions_eq {
        fails.push(format!(
            "uniquePositions {unique} != {}",
            case.unique_positions
        ));
    }
    let triangles_eq = report.triangles == case.triangles;
    if !triangles_eq {
        fails.push(format!(
            "triangles {} != {}",
            report.triangles, case.triangles
        ));
    }
    let diag = (0..3)
        .map(|i| (case.bbox_max[i] - case.bbox_min[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    let mut bbox_max_abs_err = 0.0f64;
    for i in 0..3 {
        for (got, want) in [
            (report.bbox_min[i], case.bbox_min[i]),
            (report.bbox_max[i], case.bbox_max[i]),
        ] {
            let d = (f64::from(got) - want).abs();
            // NaN-propagating max: a non-finite bbox must fail.
            bbox_max_abs_err = if d.is_nan() || d > bbox_max_abs_err {
                d
            } else {
                bbox_max_abs_err
            };
        }
    }
    let bbox_tol = BBOX_TOL * diag.max(1.0);
    if !within(bbox_max_abs_err, bbox_tol) {
        fails.push(format!(
            "bbox max abs err {bbox_max_abs_err:e} > {bbox_tol:e}"
        ));
    }
    let area_rel_err = rel_err(report.area, case.area);
    if !within(area_rel_err, REL_TOL) {
        fails.push(format!(
            "area {} vs {} rel err {area_rel_err:e} > {REL_TOL:e}",
            report.area, case.area
        ));
    }
    let volume_rel_err = rel_err(report.volume, case.volume);
    if !within(volume_rel_err, REL_TOL) {
        fails.push(format!(
            "volume {} vs {} rel err {volume_rel_err:e} > {REL_TOL:e}",
            report.volume, case.volume
        ));
    }
    RefDelta {
        unique_positions_eq,
        triangles_eq,
        bbox_max_abs_err,
        area_rel_err,
        volume_rel_err,
        ok: fails.is_empty(),
        message: fails.join("; "),
    }
}

/// Maps a mesh from the reference frame to RAA's: `p_raa[k] = sign_k * p_sdf[axis_k]`, normals
/// likewise, and each triangle's index order reversed when `det = -1` (outward winding kept).
pub fn map_to_raa(m: &ReferenceMapping, sdf: &MeshData, out: &mut MeshData) {
    out.clear();
    let map3 = |v: &[f32; 3]| m.axis_map.map(|(a, s)| f32::from(s) * v[usize::from(a)]);
    out.positions.extend(sdf.positions.iter().map(map3));
    out.normals.extend(sdf.normals.iter().map(map3));
    for tri in sdf.indices.chunks(3) {
        if m.det < 0 {
            out.indices.extend(tri.iter().rev());
        } else {
            out.indices.extend_from_slice(tri);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../tests/fixtures/sdf-leaf-reference.json");

    type Mutation = (&'static str, Box<dyn Fn(&mut Json)>);

    fn fixture() -> RefFixture {
        parse_fixture(FIXTURE).expect("committed fixture parses")
    }

    fn root() -> Json {
        parse_json(FIXTURE).unwrap()
    }

    fn mapping_json() -> Json {
        let Json::Obj(top) = root() else {
            unreachable!()
        };
        top.into_iter().find(|(k, _)| k == "mapping").unwrap().1
    }

    /// The value at `path` (object keys, or array indices as decimal strings).
    fn at<'a>(v: &'a mut Json, path: &[&str]) -> &'a mut Json {
        path.iter().fold(v, |v, key| match v {
            Json::Obj(m) => &mut m.iter_mut().find(|(k, _)| k == key).unwrap().1,
            Json::Arr(a) => &mut a[key.parse::<usize>().unwrap()],
            _ => panic!("no {key}"),
        })
    }

    fn remove(v: &mut Json, path: &[&str], key: &str) {
        let Json::Obj(m) = at(v, path) else { panic!() };
        m.retain(|(k, _)| k != key);
    }

    fn insert(v: &mut Json, path: &[&str], key: &str, value: Json) {
        let Json::Obj(m) = at(v, path) else { panic!() };
        m.push((key.to_string(), value));
    }

    fn js(text: &str) -> Json {
        parse_json(text).unwrap()
    }

    fn refused(v: &Json, what: &str) -> RefError {
        let e = validate_mapping(v, "/mapping").expect_err(what);
        assert_eq!(e.code(), "prims.mapping_invalid", "{what}: {e}");
        e
    }

    #[test]
    fn committed_fixture_parses_with_a_valid_mapping() {
        let fx = fixture();
        assert_eq!(fx.cases.len(), CASE_COUNT);
        let m = fx.mapping();
        assert_eq!(m.commit, "50b45d5862be55ba85492d6bb0e457ed3b244dd6");
        assert_eq!(fx.sdf_modeler_revision, m.commit);
        assert_eq!(m.axis_map, [(0, 1), (1, 1), (2, 1)]);
        assert_eq!(m.det, 1);
        assert_eq!(fx.axis_map, [(0, 1.0), (1, 1.0), (2, 1.0)]);
        let kinds: Vec<PrimKind> = m.kinds.iter().map(|(k, _)| *k).collect();
        assert_eq!(kinds, PrimKind::ALL);
        for (k, sdf) in &m.kinds {
            assert_eq!(sdf, k.as_str());
        }
        for kind in PrimKind::ALL {
            for set in ['A', 'B'] {
                for q in [8, 24, 64] {
                    let c = fx.case(kind, set, q).expect("case present");
                    assert!(c.closed && c.volume > 0.0);
                    assert!(c.sdf_input.starts_with('{'));
                }
            }
        }
        let parsed = parse_mapping(&{
            let mut s = String::new();
            mapping_json().write(&mut s);
            s
        })
        .unwrap();
        assert_eq!(&parsed, m);
    }

    #[test]
    fn lod0_counts_equal_ref_and_the_count_table_matches_the_mapping() {
        let fx = fixture();
        for kind in PrimKind::ALL {
            let km = fx.mapping().kind(kind).unwrap();
            let own = count_rules(kind);
            assert_eq!(own.len(), km.segments.len(), "{kind:?}");
            for (o, r) in own.iter().zip(&km.segments) {
                assert_eq!(o.count(Quality::LOD0.segments), o.reference);
                assert_eq!(r.count(Quality::LOD0.segments), r.reference);
                assert_eq!((o.name, o.reference), (r.count.as_str(), r.reference));
            }
        }
        assert!(count_rules(PrimKind::Box).is_empty());
        let sphere = count_rules(PrimKind::Sphere);
        let at = |s| sphere.iter().map(|c| c.count(s)).collect::<Vec<_>>();
        assert_eq!(at(8), [13, 7]);
        assert_eq!(at(24), [40, 20]);
        assert_eq!(at(64), [107, 53]);
        let torus = count_rules(PrimKind::Torus);
        assert_eq!(
            torus.iter().map(|c| c.count(8)).collect::<Vec<_>>(),
            [16, 7]
        );
    }

    #[test]
    fn count_clamps_segments_rounds_half_away_and_respects_min_and_multiple() {
        assert_eq!(count(40, 3, 1, 0), count(40, 3, 1, 3));
        assert_eq!(count(40, 3, 1, 3), 5);
        assert_eq!(count(40, 3, 1, 1000), count(40, 3, 1, 256));
        assert_eq!(count(40, 3, 1, 256), 427);
        // 3 * 4 / 24 = 0.5 rounds away from zero to 1.
        assert_eq!(count(3, 1, 1, 4), 1);
        assert_eq!(count(10, 1, 1, 3), 1);
        assert_eq!(count(2, 3, 1, 3), 3);
        assert_eq!(count(40, 3, 4, 8), 12);
        assert_eq!(count(40, 3, 4, 24), 40);
    }

    #[test]
    fn capsule_height_is_two_half_heights_plus_two_radii() {
        assert_eq!(sdf_value(2.0, 0.5, Some((2.0, 0.4))), 2.0 * 0.5 + 2.0 * 0.4);
        assert_eq!(sdf_value(2.0, 0.5, None), 1.0);
        let fx = fixture();
        let km = fx.mapping().kind(PrimKind::Capsule).unwrap();
        let height = km.params.iter().find(|t| t.sdf_param == "height").unwrap();
        let add = height.add.as_ref().unwrap();
        assert_eq!((height.from.as_str(), height.scale), ("halfHeight", 2.0));
        assert_eq!(
            (add.from.as_str(), add.component, add.scale),
            ("radius", None, 2.0)
        );
        for (set, want) in [('A', 1.8), ('B', 2.9)] {
            let p = PrimParams::reference_set(PrimKind::Capsule, set).to_params();
            let PrimParams::Capsule {
                radius,
                half_height,
            } = PrimParams::reference_set(PrimKind::Capsule, set)
            else {
                unreachable!()
            };
            let h = height.evaluate(&p).unwrap();
            assert_eq!(h, 2.0 * f64::from(half_height) + 2.0 * f64::from(radius));
            assert!((h - want).abs() < 1e-6, "{set}: {h}");
            for q in [8, 24, 64] {
                let c = fx.case(PrimKind::Capsule, set, q).unwrap();
                assert!(params_close(h, c.ref_param("height", None).unwrap()));
                // The reference's Y extent is PrimParams::bounds' halfHeight + radius.
                let b = PrimParams::reference_set(PrimKind::Capsule, set).bounds();
                assert!((c.bbox_max[1] - f64::from(b.max.y)).abs() < 1e-6);
            }
        }
        let entry = &fx
            .param_map
            .iter()
            .find(|(k, _)| *k == PrimKind::Capsule)
            .unwrap()
            .1[0];
        assert_eq!(entry.add.as_ref().unwrap().raa_name, "radius");
    }

    #[test]
    fn mapping_header_violations_are_refused() {
        let ok = mapping_json();
        validate_mapping(&ok, "/mapping").unwrap();
        let cases: Vec<Mutation> = vec![
            (
                "version 2",
                Box::new(|m| *at(m, &["version"]) = Json::Num(2.0)),
            ),
            (
                "version string",
                Box::new(|m| *at(m, &["version"]) = js("\"1\"")),
            ),
            (
                "unknown top key",
                Box::new(|m| insert(m, &[], "extra", Json::Null)),
            ),
            (
                "missing reference",
                Box::new(|m| remove(m, &[], "reference")),
            ),
            (
                "short commit",
                Box::new(|m| *at(m, &["reference", "commit"]) = js("\"50b45d5\"")),
            ),
            (
                "upper-case commit",
                Box::new(|m| {
                    *at(m, &["reference", "commit"]) =
                        js("\"50B45D5862BE55BA85492D6BB0E457ED3B244DD6\"")
                }),
            ),
            (
                "non-hex commit",
                Box::new(|m| {
                    *at(m, &["reference", "commit"]) =
                        js("\"50b45d5862be55ba85492d6bb0e457ed3b244ddg\"")
                }),
            ),
            (
                "axis used twice",
                Box::new(|m| *at(m, &["axisMap", "y"]) = js("\"+x\"")),
            ),
            (
                "axis without sign",
                Box::new(|m| *at(m, &["axisMap", "x"]) = js("\"x\"")),
            ),
            (
                "axis unknown",
                Box::new(|m| *at(m, &["axisMap", "x"]) = js("\"+w\"")),
            ),
            (
                "det mismatch",
                Box::new(|m| *at(m, &["axisMap", "det"]) = Json::Num(-1.0)),
            ),
            (
                "det 0",
                Box::new(|m| *at(m, &["axisMap", "det"]) = Json::Num(0.0)),
            ),
            ("missing det", Box::new(|m| remove(m, &["axisMap"], "det"))),
            (
                "missing kind",
                Box::new(|m| remove(m, &["paramMap"], "torus")),
            ),
            (
                "unknown kind",
                Box::new(|m| insert(m, &["paramMap"], "plane", js("{}"))),
            ),
            (
                "duplicate kind",
                Box::new(|m| {
                    let sphere = at(m, &["paramMap", "sphere"]).clone();
                    insert(m, &["paramMap"], "sphere", sphere);
                }),
            ),
            (
                "unknown sdfKind",
                Box::new(|m| *at(m, &["paramMap", "sphere", "sdfKind"]) = js("\"ball\"")),
            ),
            (
                "duplicate sdfKind",
                Box::new(|m| *at(m, &["paramMap", "ellipsoid", "sdfKind"]) = js("\"sphere\"")),
            ),
        ];
        for (what, f) in cases {
            let mut m = ok.clone();
            f(&mut m);
            refused(&m, what);
        }
    }

    #[test]
    fn mapping_param_violations_are_refused() {
        let ok = mapping_json();
        let entry = |c: &str| {
            js(&format!(
                "{{\"from\":\"halfExtents\",\"component\":{c},\"scale\":2}}"
            ))
        };
        let cases: Vec<Mutation> = vec![
            (
                "vector from with null component",
                Box::new(move |m| *at(m, &["paramMap", "box", "params", "size.x"]) = entry("null")),
            ),
            (
                "component not the axisMap axis",
                Box::new(move |m| {
                    *at(m, &["paramMap", "box", "params", "size.x"]) = entry("\"y\"")
                }),
            ),
            (
                "unknown component",
                Box::new(move |m| {
                    *at(m, &["paramMap", "box", "params", "size.x"]) = entry("\"w\"")
                }),
            ),
            (
                "numeric component",
                Box::new(move |m| *at(m, &["paramMap", "box", "params", "size.x"]) = entry("0")),
            ),
            (
                "missing vector component",
                Box::new(|m| remove(m, &["paramMap", "box", "params"], "size.z")),
            ),
            (
                "bare vec3 name",
                Box::new(move |m| {
                    remove(m, &["paramMap", "box", "params"], "size.x");
                    insert(m, &["paramMap", "box", "params"], "size", entry("\"x\""));
                }),
            ),
            (
                "scalar with a component",
                Box::new(|m| {
                    *at(m, &["paramMap", "sphere", "params", "radius", "component"]) = js("\"x\"")
                }),
            ),
            (
                "unknown S3 from",
                Box::new(|m| {
                    *at(m, &["paramMap", "sphere", "params", "radius", "from"]) = js("\"diameter\"")
                }),
            ),
            (
                "unknown sdf param",
                Box::new(|m| {
                    insert(
                        m,
                        &["paramMap", "sphere", "params"],
                        "taper",
                        js("{\"from\":\"radius\",\"component\":null,\"scale\":1}"),
                    )
                }),
            ),
            (
                "advanced param as constant",
                Box::new(|m| {
                    insert(
                        m,
                        &["paramMap", "roundBox", "constants"],
                        "edgeStyle",
                        js("\"round\""),
                    )
                }),
            ),
            (
                "param also a constant",
                Box::new(|m| {
                    insert(
                        m,
                        &["paramMap", "sphere", "constants"],
                        "radius",
                        Json::Num(1.0),
                    )
                }),
            ),
            (
                "constant not covered",
                Box::new(|m| remove(m, &["paramMap", "cone", "constants"], "radiusTop")),
            ),
            (
                "constant of bad type",
                Box::new(|m| *at(m, &["paramMap", "cone", "constants", "radiusTop"]) = js("[0]")),
            ),
            (
                "S3 key consumed twice",
                Box::new(|m| {
                    *at(m, &["paramMap", "cylinder", "params", "height", "from"]) = js("\"radius\"")
                }),
            ),
            (
                "scale 0",
                Box::new(|m| {
                    *at(m, &["paramMap", "torus", "params", "majorRadius", "scale"]) =
                        Json::Num(0.0)
                }),
            ),
            (
                "negative scale",
                Box::new(|m| {
                    *at(m, &["paramMap", "torus", "params", "majorRadius", "scale"]) =
                        Json::Num(-1.0)
                }),
            ),
            (
                "missing scale",
                Box::new(|m| remove(m, &["paramMap", "torus", "params", "majorRadius"], "scale")),
            ),
            (
                "unknown entry key",
                Box::new(|m| {
                    insert(
                        m,
                        &["paramMap", "torus", "params", "majorRadius"],
                        "offset",
                        Json::Num(0.0),
                    )
                }),
            ),
            (
                "entry not an object",
                Box::new(|m| {
                    *at(m, &["paramMap", "torus", "params", "majorRadius"]) = Json::Num(1.0)
                }),
            ),
        ];
        for (what, f) in cases {
            let mut m = ok.clone();
            f(&mut m);
            refused(&m, what);
        }
    }

    #[test]
    fn mapping_add_violations_are_refused() {
        let ok = mapping_json();
        let add = ["paramMap", "capsule", "params", "height", "add"];
        let cases: Vec<Mutation> = vec![
            (
                "nested add",
                Box::new(move |m| {
                    insert(
                        m,
                        &add,
                        "add",
                        js("{\"from\":\"radius\",\"component\":null,\"scale\":1}"),
                    )
                }),
            ),
            (
                "add scale 0",
                Box::new(move |m| *at(m, &[&add[..], &["scale"]].concat()) = Json::Num(0.0)),
            ),
            (
                "add scale string",
                Box::new(move |m| *at(m, &[&add[..], &["scale"]].concat()) = js("\"2\"")),
            ),
            (
                "add unknown from",
                Box::new(move |m| *at(m, &[&add[..], &["from"]].concat()) = js("\"length\"")),
            ),
            (
                "add component on scalar",
                Box::new(move |m| *at(m, &[&add[..], &["component"]].concat()) = js("\"y\"")),
            ),
            (
                "add bad component",
                Box::new(move |m| *at(m, &[&add[..], &["component"]].concat()) = js("\"q\"")),
            ),
            (
                "add unknown key",
                Box::new(move |m| insert(m, &add, "offset", Json::Num(1.0))),
            ),
            (
                "add missing key",
                Box::new(move |m| remove(m, &add, "component")),
            ),
            (
                "add not an object",
                Box::new(move |m| *at(m, &add) = Json::Num(2.0)),
            ),
            (
                "add null on vector key",
                Box::new(|m| {
                    insert(
                        m,
                        &["paramMap", "box", "params", "size.x"],
                        "add",
                        js("{\"from\":\"halfExtents\",\"component\":null,\"scale\":1}"),
                    )
                }),
            ),
            (
                "add vector component off-axis",
                Box::new(|m| {
                    insert(
                        m,
                        &["paramMap", "box", "params", "size.x"],
                        "add",
                        js("{\"from\":\"halfExtents\",\"component\":\"z\",\"scale\":1}"),
                    )
                }),
            ),
        ];
        for (what, f) in cases {
            let mut m = ok.clone();
            f(&mut m);
            refused(&m, what);
        }
        // The add key is an extra use: removing the main radius term still fails coverage.
        let mut m = ok.clone();
        remove(&mut m, &["paramMap", "capsule", "params"], "radius");
        refused(&m, "radius only used by add");
        // A valid add on another entry is accepted and evaluates scale*a + add.scale*b.
        let mut m = ok.clone();
        insert(
            &mut m,
            &["paramMap", "torus", "params", "majorRadius"],
            "add",
            js("{\"from\":\"minorRadius\",\"component\":null,\"scale\":-1}"),
        );
        let v = validate_mapping(&m, "/mapping").unwrap();
        let t = &v.kind(PrimKind::Torus).unwrap().params[0];
        let p = PrimParams::reference_set(PrimKind::Torus, 'A').to_params();
        assert_eq!(t.evaluate(&p), Some(f64::from(0.6f32) - f64::from(0.22f32)));
    }

    #[test]
    fn mapping_segment_violations_are_refused() {
        let ok = mapping_json();
        let rule = ["paramMap", "sphere", "segments", "0"];
        let cases: Vec<Mutation> = vec![
            (
                "segments not an array",
                Box::new(|m| *at(m, &["paramMap", "sphere", "segments"]) = js("{}")),
            ),
            (
                "segments null for box",
                Box::new(|m| *at(m, &["paramMap", "box", "segments"]) = Json::Null),
            ),
            (
                "box gains a count",
                Box::new(|m| {
                    *at(m, &["paramMap", "box", "segments"]) = js(
                        "[{\"count\":\"n\",\"ref\":4,\"min\":1,\"multiple\":1,\"source\":\"x:1\"}]",
                    )
                }),
            ),
            (
                "rule dropped",
                Box::new(|m| *at(m, &["paramMap", "sphere", "segments"]) = js("[]")),
            ),
            (
                "missing source",
                Box::new(move |m| remove(m, &rule, "source")),
            ),
            (
                "empty source",
                Box::new(move |m| *at(m, &[&rule[..], &["source"]].concat()) = js("\"\"")),
            ),
            (
                "ref 0",
                Box::new(move |m| *at(m, &[&rule[..], &["ref"]].concat()) = Json::Num(0.0)),
            ),
            (
                "fractional ref",
                Box::new(move |m| *at(m, &[&rule[..], &["ref"]].concat()) = Json::Num(40.5)),
            ),
            (
                "min above ref",
                Box::new(move |m| *at(m, &[&rule[..], &["min"]].concat()) = Json::Num(41.0)),
            ),
            (
                "ref not a multiple",
                Box::new(move |m| *at(m, &[&rule[..], &["multiple"]].concat()) = Json::Num(3.0)),
            ),
            (
                "multiple 0",
                Box::new(move |m| *at(m, &[&rule[..], &["multiple"]].concat()) = Json::Num(0.0)),
            ),
            (
                "ref differs from table",
                Box::new(move |m| *at(m, &[&rule[..], &["ref"]].concat()) = Json::Num(48.0)),
            ),
            (
                "duplicate count name",
                Box::new(|m| {
                    *at(m, &["paramMap", "sphere", "segments", "1", "count"]) = js("\"nu\"")
                }),
            ),
            (
                "renamed count",
                Box::new(move |m| *at(m, &[&rule[..], &["count"]].concat()) = js("\"rings\"")),
            ),
            (
                "unknown rule key",
                Box::new(move |m| insert(m, &rule, "parity", Json::Num(2.0))),
            ),
        ];
        for (what, f) in cases {
            let mut m = ok.clone();
            f(&mut m);
            refused(&m, what);
        }
    }

    fn reparse(root: &Json) -> Result<RefFixture, RefError> {
        let mut s = String::new();
        root.write(&mut s);
        parse_fixture(&s)
    }

    #[test]
    fn fixture_violations_are_refused_with_their_codes() {
        let r = root();
        assert_eq!(reparse(&r).unwrap(), fixture());

        let mut m = r.clone();
        *at(&mut m, &["cases", "0", "counts", "nu"]) = Json::Num(14.0);
        assert_eq!(
            reparse(&m).unwrap_err().code(),
            "prims.reference_counts_mismatch"
        );
        let mut m = r.clone();
        remove(&mut m, &["cases", "0", "counts"], "nv");
        assert_eq!(
            reparse(&m).unwrap_err().code(),
            "prims.reference_counts_mismatch"
        );

        let mut m = r.clone();
        *at(&mut m, &["mapping", "version"]) = Json::Num(2.0);
        let e = reparse(&m).unwrap_err();
        assert_eq!(
            (e.code(), e.path()),
            ("prims.mapping_invalid", "/mapping/version")
        );
        let g: GeomError = e.into();
        assert_eq!(g.code(), "geom.invalid_param");
        assert!(
            matches!(&g, GeomError::InvalidParam { kind, name, .. } if kind == "fixture" && name == "/mapping/version")
        );

        let fixture_cases: Vec<Mutation> = vec![
            ("missing mapping", Box::new(|m| remove(m, &[], "mapping"))),
            (
                "unknown key",
                Box::new(|m| insert(m, &[], "extra", Json::Null)),
            ),
            (
                "revision differs",
                Box::new(|m| *at(m, &["sdfModelerRevision"]) = js("\"x\"")),
            ),
            (
                "regenerate differs",
                Box::new(|m| *at(m, &["regenerate"]) = js("\"x\"")),
            ),
            (
                "axisMap differs",
                Box::new(|m| *at(m, &["axisMap", "0", "sign"]) = Json::Num(-1.0)),
            ),
            (
                "paramMap kind missing",
                Box::new(|m| remove(m, &["paramMap"], "cone")),
            ),
            (
                "case dropped",
                Box::new(|m| {
                    let Json::Arr(a) = at(m, &["cases"]) else {
                        panic!()
                    };
                    a.pop();
                }),
            ),
            (
                "case duplicated",
                Box::new(|m| {
                    let Json::Arr(a) = at(m, &["cases"]) else {
                        panic!()
                    };
                    let c = a[0].clone();
                    a[1] = c;
                }),
            ),
            (
                "bad set",
                Box::new(|m| *at(m, &["cases", "0", "set"]) = js("\"C\"")),
            ),
            (
                "bad quality",
                Box::new(|m| *at(m, &["cases", "0", "quality"]) = Json::Num(12.0)),
            ),
            (
                "bad kind",
                Box::new(|m| *at(m, &["cases", "0", "kind"]) = js("\"plane\"")),
            ),
            (
                "closed not bool",
                Box::new(|m| *at(m, &["cases", "0", "closed"]) = Json::Num(1.0)),
            ),
            (
                "bbox of 2",
                Box::new(|m| *at(m, &["cases", "0", "bboxMin"]) = js("[0,0]")),
            ),
        ];
        for (what, f) in fixture_cases {
            let mut m = r.clone();
            f(&mut m);
            let e = reparse(&m).expect_err(what);
            assert_eq!(e.code(), "prims.fixture_invalid", "{what}: {e}");
        }
        for bad in [
            "",
            "{",
            "{\"a\":1,}",
            "[1 2]",
            "{\"a\":01}",
            "{\"a\":1e999}",
            "\"\\x\"",
            "{} x",
        ] {
            assert_eq!(
                parse_fixture(bad).unwrap_err().code(),
                "prims.fixture_invalid",
                "{bad:?}"
            );
        }
        assert_eq!(
            parse_json("[\"a\\u00e9\\ud83d\\ude00\", -0.5e1, true, null]").unwrap(),
            Json::Arr(vec![
                Json::Str("aé😀".into()),
                Json::Num(-5.0),
                Json::Bool(true),
                Json::Null
            ])
        );
    }

    /// A 24-vertex hard-edged box (per-face vertices and normals, CCW from outside).
    fn box_mesh(h: [f32; 3]) -> MeshData {
        let mut m = MeshData::new();
        for a in 0..3 {
            for s in [1.0f32, -1.0] {
                let (mut u, mut v) = ((a + 1) % 3, (a + 2) % 3);
                if s < 0.0 {
                    std::mem::swap(&mut u, &mut v);
                }
                let base = m.positions.len() as u32;
                for (du, dv) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                    let mut p = [0.0f32; 3];
                    p[a] = s * h[a];
                    p[u] = du * h[u];
                    p[v] = dv * h[v];
                    let mut n = [0.0f32; 3];
                    n[a] = s;
                    m.positions.push(p);
                    m.normals.push(n);
                }
                m.indices
                    .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
            }
        }
        m
    }

    #[test]
    fn compare_accepts_a_matching_box_and_rejects_mismatches() {
        let fx = fixture();
        for set in ['A', 'B'] {
            let PrimParams::Box { half_extents } = PrimParams::reference_set(PrimKind::Box, set)
            else {
                unreachable!()
            };
            let mesh = box_mesh(half_extents);
            assert!(raa_geom::analysis::passes(&analyze(&mesh)).is_ok());
            for q in [8, 24, 64] {
                let d = compare(&fx, fx.case(PrimKind::Box, set, q).unwrap(), &mesh);
                assert!(d.ok, "box {set} {q}: {}", d.message);
                assert!(d.message.is_empty());
                assert!(
                    d.bbox_max_abs_err <= BBOX_TOL
                        && d.area_rel_err <= REL_TOL
                        && d.volume_rel_err <= REL_TOL
                );
            }
        }
        let a = box_mesh([0.7, 0.6, 0.5]);
        // Wrong kind: counts and metrics differ.
        let d = compare(&fx, fx.case(PrimKind::Sphere, 'A', 24).unwrap(), &a);
        assert!(
            !d.ok && !d.triangles_eq && !d.unique_positions_eq,
            "{}",
            d.message
        );
        // Wrong set: same counts, metrics outside SDF_REF_TOL, params not reference set A.
        let case_b = fx.case(PrimKind::Box, 'B', 24).unwrap();
        let mut wrong_params = case_b.clone();
        wrong_params.params = fx.case(PrimKind::Box, 'A', 24).unwrap().params.clone();
        let d = compare(&fx, &wrong_params, &a);
        assert!(
            !d.ok && d.triangles_eq && d.message.contains("reference set B"),
            "{}",
            d.message
        );
        // A tiny bbox shift is caught.
        let mut shifted = a.clone();
        for p in &mut shifted.positions {
            p[0] += 1e-4;
        }
        let d = compare(&fx, fx.case(PrimKind::Box, 'A', 24).unwrap(), &shifted);
        assert!(!d.ok && d.bbox_max_abs_err > BBOX_TOL, "{}", d.message);
        // An empty (not yet ported) mesh never passes.
        let d = compare(
            &fx,
            fx.case(PrimKind::Box, 'A', 24).unwrap(),
            &MeshData::new(),
        );
        assert!(!d.ok && !d.triangles_eq);
        // refParams that the mapping does not reproduce.
        let mut bad_ref = fx.case(PrimKind::Box, 'A', 24).unwrap().clone();
        bad_ref.ref_params[0].1[0] += 0.01;
        let d = compare(&fx, &bad_ref, &a);
        assert!(
            !d.ok && d.message.contains("paramMap") && d.message.contains("mapping size.x"),
            "{}",
            d.message
        );
        // A reference mesh that is not closed is a mismatch for the lead.
        let mut open = fx.case(PrimKind::Box, 'A', 24).unwrap().clone();
        open.closed = false;
        let d = compare(&fx, &open, &a);
        assert!(!d.ok && d.message.contains(NOT_CLOSED));
    }

    #[test]
    fn map_to_raa_applies_the_axis_map_and_reverses_winding_when_det_is_negative() {
        let mut j = mapping_json();
        *at(&mut j, &["axisMap", "z"]) = js("\"-z\"");
        *at(&mut j, &["axisMap", "det"]) = Json::Num(-1.0);
        let m = validate_mapping(&j, "/mapping").unwrap();
        assert_eq!((m.axis_map[2], m.det), ((2, -1), -1));
        let src = box_mesh([0.7, 0.6, 0.5]);
        let before = analyze(&src);
        let mut out = MeshData::new();
        map_to_raa(&m, &src, &mut out);
        assert_eq!(
            out.positions[0],
            [
                src.positions[0][0],
                src.positions[0][1],
                -src.positions[0][2]
            ]
        );
        assert_eq!(
            out.normals[0],
            [src.normals[0][0], src.normals[0][1], -src.normals[0][2]]
        );
        assert_eq!(
            &out.indices[..3],
            &[src.indices[2], src.indices[1], src.indices[0]]
        );
        let after = analyze(&out);
        assert!(after.closed && after.volume > 0.0);
        assert!((after.volume - before.volume).abs() < 1e-9);
        assert_eq!(after.inward_normals, 0);

        // A rotation (det = +1): x <- -z, z <- +x keeps the index order.
        let mut j = mapping_json();
        *at(&mut j, &["axisMap", "x"]) = js("\"-z\"");
        *at(&mut j, &["axisMap", "z"]) = js("\"+x\"");
        refused(&j, "vec3 components not following the axisMap");
        // The vec3 components must follow the axisMap: SDF x is now RAA z and SDF z is RAA x.
        for (kind, p) in [
            ("box", "size"),
            ("roundBox", "size"),
            ("ellipsoid", "radii"),
        ] {
            *at(
                &mut j,
                &["paramMap", kind, "params", &format!("{p}.x"), "component"],
            ) = js("\"z\"");
            *at(
                &mut j,
                &["paramMap", kind, "params", &format!("{p}.z"), "component"],
            ) = js("\"x\"");
        }
        let m = validate_mapping(&j, "/mapping").unwrap();
        assert_eq!(m.det, 1);
        map_to_raa(&m, &src, &mut out);
        assert_eq!(out.indices, src.indices);
        assert!(analyze(&out).volume > 0.0);
    }
}
