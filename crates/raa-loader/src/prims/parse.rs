//! Parser of prims v0 text: the structural and schema pass (JSON shape, keys, types,
//! finiteness, format/version, transforms, materials, anchors, collision shape) and, in
//! the same pre-order node walk, the semantic checks (ids, caps, kinds and params, ops),
//! then the references (materials, collider shapes and nodes).
//! Errors carry a JSON pointer; the first error in document order wins.

use super::model::{
    ColliderDoc, NodeOp, ParamType, PrimMaterial, PrimNode, PrimsAsset, TransformDoc,
    collider_params, internal_shape, kind_params, optional_params, published_kind_params,
};
use super::{DEFAULT_SEGMENTS, FORMAT, MAX_LEVEL, MAX_NODES, VERSION};
use crate::DEFAULT_MATERIAL;
use crate::error::LoadError;
use crate::json::{self, JsonValue};

type Obj = [(String, JsonValue)];

/// Parses a prims document. `strict_kinds` limits kinds to the 8 S3 kinds; otherwise an
/// unknown kind is kept with its params in document order (its provider validates them).
pub(crate) fn parse_with(text: &str, strict_kinds: bool) -> Result<PrimsAsset, LoadError> {
    let root = json::parse(text)?;
    let mut w = Walk {
        strict: strict_kinds,
        count: 0,
        ids: Vec::new(),
        materials: Vec::new(),
    };
    let mut a = document(&root, &mut w)?;
    references(&mut a, &w)?;
    Ok(a)
}

/// State of the pre-order node walk.
struct Walk {
    strict: bool,
    /// Nodes seen so far.
    count: usize,
    /// Node ids in pre-order.
    ids: Vec<String>,
    /// (pointer of the `material` key, name) in pre-order.
    materials: Vec<(String, String)>,
}

/// Child pointer `<base>/<key>` with `~` and `/` escaped (RFC 6901).
fn ptr(base: &str, key: &str) -> String {
    let mut s = String::with_capacity(base.len() + key.len() + 1);
    s.push_str(base);
    s.push('/');
    for c in key.chars() {
        match c {
            '~' => s.push_str("~0"),
            '/' => s.push_str("~1"),
            _ => s.push(c),
        }
    }
    s
}

fn at(base: &str, i: usize) -> String {
    format!("{base}/{i}")
}

fn type_name(v: &JsonValue) -> &'static str {
    match v {
        JsonValue::Null => "null",
        JsonValue::Bool(_) => "a boolean",
        JsonValue::Number(_) => "a number",
        JsonValue::String(_) => "a string",
        JsonValue::Array(_) => "an array",
        JsonValue::Object(_) => "an object",
    }
}

fn type_err(v: &JsonValue, path: &str, want: &str) -> LoadError {
    LoadError::new(
        "schema.type",
        path,
        format!("expected {want}, found {}", type_name(v)),
    )
}

fn object<'a>(v: &'a JsonValue, path: &str) -> Result<&'a Obj, LoadError> {
    match v {
        JsonValue::Object(o) => Ok(o),
        _ => Err(type_err(v, path, "an object")),
    }
}

fn array<'a>(v: &'a JsonValue, path: &str) -> Result<&'a [JsonValue], LoadError> {
    match v {
        JsonValue::Array(a) => Ok(a),
        _ => Err(type_err(v, path, "an array")),
    }
}

fn string(v: &JsonValue, path: &str) -> Result<String, LoadError> {
    match v {
        JsonValue::String(s) => Ok(s.clone()),
        _ => Err(type_err(v, path, "a string")),
    }
}

/// A finite number (type checked first, then finiteness).
fn number(v: &JsonValue, path: &str) -> Result<f64, LoadError> {
    match v {
        JsonValue::Number(x) if x.is_finite() => Ok(*x),
        JsonValue::Number(_) => Err(LoadError::new(
            "value.not_finite",
            path,
            "number is not finite",
        )),
        _ => Err(type_err(v, path, "a number")),
    }
}

/// A fixed-length array of finite numbers.
fn numbers<const N: usize>(v: &JsonValue, path: &str) -> Result<[f64; N], LoadError> {
    let a = array(v, path)?;
    if a.len() != N {
        return Err(LoadError::new(
            "schema.type",
            path,
            format!("expected an array of {N} numbers, found {}", a.len()),
        ));
    }
    let mut out = [0.0; N];
    for (i, (o, x)) in out.iter_mut().zip(a).enumerate() {
        *o = number(x, &at(path, i))?;
    }
    Ok(out)
}

/// Refuses any non-finite number inside a raw value kept for a later pass.
fn finite_raw(v: &JsonValue, path: &str) -> Result<(), LoadError> {
    match v {
        JsonValue::Number(_) => number(v, path).map(|_| ()),
        JsonValue::Array(a) => a
            .iter()
            .enumerate()
            .try_for_each(|(i, x)| finite_raw(x, &at(path, i))),
        JsonValue::Object(o) => o.iter().try_for_each(|(k, x)| finite_raw(x, &ptr(path, k))),
        _ => Ok(()),
    }
}

fn range(path: &str, msg: impl Into<String>) -> LoadError {
    LoadError::new("value.range", path, msg)
}

/// Visits the keys of an object in document order: an unknown key is refused at once,
/// a known key goes to `f`; missing required keys are refused after the scan.
fn fields<'a>(
    obj: &'a Obj,
    path: &str,
    allowed: &[&str],
    required: &[&str],
    mut f: impl FnMut(&str, &'a JsonValue, &str) -> Result<(), LoadError>,
) -> Result<(), LoadError> {
    for (k, v) in obj {
        let p = ptr(path, k);
        if !allowed.contains(&k.as_str()) {
            return Err(LoadError::new(
                "schema.unknown_field",
                p,
                format!("unknown field `{k}`"),
            ));
        }
        f(k, v, &p)?;
    }
    for r in required {
        if !obj.iter().any(|(k, _)| k == r) {
            return Err(LoadError::new(
                "schema.missing_field",
                ptr(path, r),
                format!("missing required field `{r}`"),
            ));
        }
    }
    Ok(())
}

/// Raw `params` object: (name, value) pairs, finiteness checked.
fn raw_params(v: &JsonValue, path: &str) -> Result<Vec<(String, JsonValue)>, LoadError> {
    let o = object(v, path)?;
    let mut out = Vec::with_capacity(o.len());
    for (k, x) in o {
        finite_raw(x, &ptr(path, k))?;
        out.push((k.clone(), x.clone()));
    }
    Ok(out)
}

/// The published `budget` object: `triangles` (1..=100000) and `segments` (4..=128, a
/// multiple of 4). `triangles` is the triangle budget; `segments` is the tessellation of
/// the document (16 when absent, as in raa-assets-tools).
fn budget(v: &JsonValue, path: &str, a: &mut PrimsAsset) -> Result<(), LoadError> {
    let o = object(v, path)?;
    let mut segments = None;
    fields(o, path, &["triangles", "segments"], &[], |k, v, p| {
        let x = number(v, p)?;
        let integral = x.fract() == 0.0;
        if k == "triangles" {
            if !integral || !(1.0..=100_000.0).contains(&x) {
                return Err(range(
                    p,
                    format!("budget triangles {x} is not an integer in 1..100000"),
                ));
            }
            a.triangle_budget = Some(x as u32);
        } else if !integral || !(4.0..=128.0).contains(&x) || x % 4.0 != 0.0 {
            return Err(range(
                p,
                format!("budget segments {x} is not a multiple of 4 in 4..128"),
            ));
        } else {
            segments = Some(x as u32);
        }
        Ok(())
    })?;
    a.segments = Some(segments.unwrap_or(DEFAULT_SEGMENTS));
    Ok(())
}

fn document(root: &JsonValue, w: &mut Walk) -> Result<PrimsAsset, LoadError> {
    let o = object(root, "")?;
    let mut a = PrimsAsset {
        id: String::new(),
        name: String::new(),
        triangle_budget: None,
        segments: None,
        nodes: Vec::new(),
        materials: Vec::new(),
        anchors: Vec::new(),
        collision: None,
    };
    const ALLOWED: [&str; 10] = [
        "format",
        "version",
        "id",
        "name",
        "budget",
        "triangleBudget",
        "nodes",
        "materials",
        "anchors",
        "collision",
    ];
    const REQUIRED: [&str; 7] = [
        "format",
        "version",
        "id",
        "name",
        "nodes",
        "materials",
        "anchors",
    ];
    fields(o, "", &ALLOWED, &REQUIRED, |k, v, p| {
        match k {
            "format" => {
                let f = string(v, p)?;
                if f != FORMAT {
                    return Err(LoadError::new(
                        "format.unknown",
                        p,
                        format!("format `{f}` is not `{FORMAT}`"),
                    ));
                }
            }
            "version" => {
                let x = number(v, p)?;
                if x != f64::from(VERSION) {
                    return Err(LoadError::new(
                        "format.version",
                        p,
                        format!("version {x} is not supported (only {VERSION})"),
                    ));
                }
            }
            "id" => a.id = string(v, p)?,
            "name" => a.name = string(v, p)?,
            "budget" => {
                if o.iter().any(|(k, _)| k == "triangleBudget") {
                    return Err(LoadError::new(
                        "schema.unknown_field",
                        p,
                        "give either `budget` or `triangleBudget`, not both",
                    ));
                }
                budget(v, p, &mut a)?;
            }
            "triangleBudget" => {
                if o.iter().any(|(k, _)| k == "budget") {
                    return Err(LoadError::new(
                        "schema.unknown_field",
                        p,
                        "give either `budget` or `triangleBudget`, not both",
                    ));
                }
                let x = number(v, p)?;
                if x < 1.0 || x.fract() != 0.0 || x > f64::from(u32::MAX) {
                    return Err(range(
                        p,
                        format!("triangleBudget {x} is not a positive integer"),
                    ));
                }
                // In range and integral (checked above).
                a.triangle_budget = Some(x as u32);
            }
            "nodes" => {
                for (i, n) in array(v, p)?.iter().enumerate() {
                    a.nodes.push(node(n, &at(p, i), 0, w)?);
                }
            }
            "materials" => {
                for (name, m) in object(v, p)? {
                    let mp = ptr(p, name);
                    if name == DEFAULT_MATERIAL {
                        return Err(LoadError::new(
                            "id.invalid",
                            mp,
                            format!("material name `{DEFAULT_MATERIAL}` is reserved"),
                        ));
                    }
                    a.materials.push((name.clone(), material(m, &mp)?));
                }
            }
            "anchors" => {
                for (name, t) in object(v, p)? {
                    a.anchors.push((name.clone(), anchor(t, &ptr(p, name))?));
                }
            }
            _ => {
                // A published `collision` has a `mode`: the document is published v0.
                if let JsonValue::Object(m) = v
                    && m.iter().any(|(k, _)| k == "mode")
                {
                    a.segments.get_or_insert(DEFAULT_SEGMENTS);
                }
                a.collision = Some(collision(v, p)?);
            }
        }
        Ok(())
    })?;
    Ok(a)
}

fn node(v: &JsonValue, path: &str, level: usize, w: &mut Walk) -> Result<PrimNode, LoadError> {
    w.count += 1;
    if w.count > MAX_NODES {
        return Err(LoadError::new(
            "cap.nodes",
            path,
            format!("more than {MAX_NODES} nodes"),
        ));
    }
    if level > MAX_LEVEL {
        return Err(LoadError::new(
            "cap.depth",
            path,
            format!("node at level {level}, the maximum is {MAX_LEVEL} (roots are level 0)"),
        ));
    }
    let o = object(v, path)?;
    // Children are walked after this node's own checks, so the walk stays pre-order.
    let mut children: Option<(&[JsonValue], String)> = None;
    let mut n = PrimNode {
        id: String::new(),
        kind: String::new(),
        params: Vec::new(),
        transform: None,
        material: None,
        op: NodeOp::Union,
        children: Vec::new(),
    };
    const ALLOWED: [&str; 7] = [
        "id",
        "kind",
        "params",
        "transform",
        "material",
        "op",
        "children",
    ];
    fields(o, path, &ALLOWED, &["id", "kind", "params"], |k, v, p| {
        match k {
            "id" => n.id = string(v, p)?,
            "kind" => n.kind = string(v, p)?,
            "params" => n.params = raw_params(v, p)?,
            "transform" => n.transform = Some(transform(v, p)?),
            "material" => n.material = Some(string(v, p)?),
            "op" => {
                let s = string(v, p)?;
                n.op = NodeOp::from_name(&s).ok_or_else(|| {
                    range(p, format!("op `{s}` is not union, subtract or intersect"))
                })?;
            }
            _ => children = Some((array(v, p)?, p.to_string())),
        }
        Ok(())
    })?;
    node_semantics(&mut n, path, level, w)?;
    if let Some((cs, cp)) = children {
        for (i, c) in cs.iter().enumerate() {
            n.children.push(node(c, &at(&cp, i), level + 1, w)?);
        }
    }
    Ok(n)
}

/// The S3 params of a node. A published raa.prims v0 node of box, roundBox, capsule,
/// cylinder or cone (recognised by `size`, `height` or `radiusBottom`) is converted to S3:
/// box and roundBox `size` is the full size (halfExtents = size / 2); a capsule `height` is
/// the total height with the caps (halfHeight = height / 2 - radius, the body half-length);
/// a cylinder `height` is the full height (halfHeight = height / 2); a cone is the frustum
/// with `radiusBottom` at y = -height / 2 and `radiusTop` at y = +height / 2 (S3 `radius`
/// is the base, `radiusTop` is omitted when 0). Other documents are returned unchanged.
fn to_s3_params(
    kind: &str,
    params: &[(String, JsonValue)],
    pp: &str,
) -> Result<Vec<(String, JsonValue)>, LoadError> {
    let published = params
        .iter()
        .any(|(k, _)| matches!(k.as_str(), "size" | "height" | "radiusBottom"));
    let Some(list) = published_kind_params(kind).filter(|_| published) else {
        return Ok(params.to_vec());
    };
    let vals = typed_params(params, list, &[], pp, "kind.params", kind)?;
    let num = JsonValue::Number;
    let vec3_json = |a: [f64; 3]| JsonValue::Array(a.iter().map(|&x| num(x)).collect());
    let half = |a: [f64; 3]| [a[0] / 2.0, a[1] / 2.0, a[2] / 2.0];
    let out: Vec<(&str, JsonValue)> = match kind {
        "box" => vec![("halfExtents", vec3_json(half(vec3(&vals, "size"))))],
        "roundBox" => vec![
            ("halfExtents", vec3_json(half(vec3(&vals, "size")))),
            ("radius", num(scalar(&vals, "radius"))),
        ],
        "capsule" => {
            let (h, r) = (scalar(&vals, "height"), scalar(&vals, "radius"));
            if h < 2.0 * r {
                return Err(range(
                    &ptr(pp, "height"),
                    format!(
                        "capsule height must be at least twice its radius, found height {h} and radius {r}"
                    ),
                ));
            }
            vec![("radius", num(r)), ("halfHeight", num(h / 2.0 - r))]
        }
        "cylinder" => vec![
            ("radius", num(scalar(&vals, "radius"))),
            ("halfHeight", num(scalar(&vals, "height") / 2.0)),
        ],
        "cone" => {
            let rt = scalar(&vals, "radiusTop");
            let mut out = vec![
                ("radius", num(scalar(&vals, "radiusBottom"))),
                ("halfHeight", num(scalar(&vals, "height") / 2.0)),
            ];
            if rt > 0.0 {
                out.push(("radiusTop", num(rt)));
            }
            out
        }
        _ => return Ok(params.to_vec()),
    };
    Ok(out.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

/// Id, op, kind and params checks of one node (its keys are already scanned).
fn node_semantics(
    n: &mut PrimNode,
    path: &str,
    level: usize,
    w: &mut Walk,
) -> Result<(), LoadError> {
    let idp = ptr(path, "id");
    let valid = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '-';
    if n.id.is_empty() || !n.id.chars().all(valid) {
        return Err(LoadError::new(
            "id.invalid",
            idp,
            format!("node id `{}` does not match [A-Za-z0-9_-]+", n.id),
        ));
    }
    if w.ids.contains(&n.id) {
        return Err(LoadError::new(
            "id.duplicate",
            idp,
            format!("node id `{}` is already used", n.id),
        ));
    }
    w.ids.push(n.id.clone());
    if let Some(m) = &n.material {
        w.materials.push((ptr(path, "material"), m.clone()));
    }
    if level == 0 && n.op != NodeOp::Union {
        return Err(range(
            &ptr(path, "op"),
            format!("root node op must be union, found {}", n.op.name()),
        ));
    }
    let pp = ptr(path, "params");
    match kind_params(&n.kind) {
        Some(list) => {
            let raw = to_s3_params(&n.kind, &n.params, &pp)?;
            let optional = optional_params(&n.kind);
            let vals = typed_params(&raw, list, optional, &pp, "kind.params", &n.kind)?;
            kind_rules(&n.kind, &vals, &pp)?;
            // Re-ordered to S3 order (every required key is present, checked above).
            n.params = list
                .iter()
                .filter_map(|(k, _)| raw.iter().find(|(d, _)| d == k).cloned())
                .collect();
        }
        None if w.strict => {
            return Err(LoadError::new(
                "kind.unknown",
                ptr(path, "kind"),
                format!("unknown kind `{}`", n.kind),
            ));
        }
        None => {}
    }
    Ok(())
}

fn transform(v: &JsonValue, path: &str) -> Result<TransformDoc, LoadError> {
    let o = object(v, path)?;
    let mut t = TransformDoc::identity();
    fields(o, path, &["t", "r", "s"], &["t", "r", "s"], |k, v, p| {
        match k {
            "t" => t.t = numbers(v, p)?,
            "r" => {
                t.r = numbers(v, p)?;
                let norm = t.r.iter().map(|x| x * x).sum::<f64>().sqrt();
                if (norm - 1.0).abs() > 1e-3 {
                    return Err(range(
                        p,
                        format!("quaternion norm {norm} is not within 1e-3 of 1"),
                    ));
                }
            }
            _ => {
                t.s = numbers(v, p)?;
                if let Some(i) = t.s.iter().position(|&x| x <= 0.0) {
                    return Err(range(&at(p, i), "scale components must be > 0"));
                }
            }
        }
        Ok(())
    })?;
    Ok(t)
}

/// An anchor: `t` and `r` (published v0); an S3 `s` is accepted too.
fn anchor(v: &JsonValue, path: &str) -> Result<TransformDoc, LoadError> {
    let o = object(v, path)?;
    let mut t = TransformDoc::identity();
    fields(o, path, &["t", "r", "s"], &["t", "r"], |k, v, p| {
        match k {
            "t" => t.t = numbers(v, p)?,
            "r" => t.r = numbers(v, p)?,
            _ => t.s = numbers(v, p)?,
        }
        Ok(())
    })?;
    Ok(t)
}

fn material(v: &JsonValue, path: &str) -> Result<PrimMaterial, LoadError> {
    let o = object(v, path)?;
    let mut m = PrimMaterial {
        color: [0.0; 3],
        roughness: 0.0,
        metallic: 0.0,
        emissive: [0.0; 3],
    };
    const KEYS: [&str; 4] = ["color", "roughness", "metallic", "emissive"];
    let unit = |x: f64, p: &str, what: &str| {
        if (0.0..=1.0).contains(&x) {
            Ok(x)
        } else {
            Err(range(p, format!("{what} {x} is not in [0, 1]")))
        }
    };
    fields(o, path, &KEYS, &KEYS, |k, v, p| {
        match k {
            "color" => {
                m.color = numbers(v, p)?;
                for (i, &x) in m.color.iter().enumerate() {
                    unit(x, &at(p, i), "color component")?;
                }
            }
            "roughness" => m.roughness = unit(number(v, p)?, p, "roughness")?,
            "metallic" => m.metallic = unit(number(v, p)?, p, "metallic")?,
            _ => {
                m.emissive = numbers(v, p)?;
                if let Some(i) = m.emissive.iter().position(|&x| x < 0.0) {
                    return Err(range(&at(p, i), "emissive components must be >= 0"));
                }
            }
        }
        Ok(())
    })?;
    Ok(m)
}

fn collision(v: &JsonValue, path: &str) -> Result<Vec<ColliderDoc>, LoadError> {
    let o = object(v, path)?;
    let mut out = Vec::new();
    fields(
        o,
        path,
        &["mode", "colliders"],
        &["colliders"],
        |k, v, p| {
            if k == "mode" {
                let m = string(v, p)?;
                if m != "derived" && m != "override" {
                    return Err(range(
                        p,
                        format!("collision mode `{m}` is not derived or override"),
                    ));
                }
                return Ok(());
            }
            for (i, c) in array(v, p)?.iter().enumerate() {
                out.push(collider(c, &at(p, i))?);
            }
            Ok(())
        },
    )?;
    Ok(out)
}

/// A collider: `node` (a node id, or null in override mode), `shape`, `params`, then either
/// the published `t` and `r` (asset frame, no scale) or an S3 `transform` object.
fn collider(v: &JsonValue, path: &str) -> Result<ColliderDoc, LoadError> {
    let o = object(v, path)?;
    let mut c = ColliderDoc {
        node: None,
        shape: String::new(),
        params: Vec::new(),
        transform: TransformDoc::identity(),
    };
    let (mut t, mut r, mut whole) = (None, None, None);
    const ALLOWED: [&str; 6] = ["node", "shape", "params", "transform", "t", "r"];
    fields(o, path, &ALLOWED, &["shape", "params"], |k, v, p| {
        match k {
            "node" => {
                c.node = match v {
                    JsonValue::Null => None,
                    _ => Some(string(v, p)?),
                }
            }
            "shape" => c.shape = string(v, p)?,
            "params" => c.params = raw_params(v, p)?,
            "transform" => whole = Some(transform(v, p)?),
            "t" => t = Some(numbers::<3>(v, p)?),
            _ => r = Some(numbers::<4>(v, p)?),
        }
        Ok(())
    })?;
    c.transform = match (whole, t, r) {
        (Some(_), Some(_), _) | (Some(_), _, Some(_)) => {
            return Err(LoadError::new(
                "schema.unknown_field",
                path,
                "give either `transform` or `t` and `r`, not both",
            ));
        }
        (Some(tr), _, _) => tr,
        (None, t, r) => TransformDoc {
            t: t.unwrap_or([0.0; 3]),
            r: r.unwrap_or([0.0, 0.0, 0.0, 1.0]),
            s: [1.0; 3],
        },
    };
    Ok(c)
}

/// A checked param value.
#[derive(Clone, Copy)]
enum Param {
    Scalar(f64),
    Vec3([f64; 3]),
}

/// Checks a params object against a table entry: no extra key (document order), no
/// missing key (except the `optional` ones), each value of the table's type. Returns the
/// values in table order.
fn typed_params<'a>(
    params: &[(String, JsonValue)],
    list: &[(&'a str, ParamType)],
    optional: &[&str],
    pp: &str,
    code: &'static str,
    what: &str,
) -> Result<Vec<(&'a str, Param)>, LoadError> {
    if let Some((k, _)) = params
        .iter()
        .find(|(k, _)| !list.iter().any(|(n, _)| n == k))
    {
        return Err(LoadError::new(
            code,
            ptr(pp, k),
            format!("unexpected param `{k}` for `{what}`"),
        ));
    }
    let mut out = Vec::with_capacity(list.len());
    for &(name, ty) in list {
        let p = ptr(pp, name);
        let Some((_, v)) = params.iter().find(|(k, _)| k == name) else {
            if optional.contains(&name) {
                continue;
            }
            return Err(LoadError::new(
                code,
                p,
                format!("missing param `{name}` for `{what}`"),
            ));
        };
        let val = match (ty, v) {
            (ParamType::Scalar, JsonValue::Number(x)) => Some(Param::Scalar(*x)),
            (ParamType::Vec3, JsonValue::Array(a)) if a.len() == 3 => {
                let mut xs = [0.0; 3];
                let mut ok = true;
                for (o, x) in xs.iter_mut().zip(a) {
                    match x {
                        JsonValue::Number(x) => *o = *x,
                        _ => ok = false,
                    }
                }
                ok.then_some(Param::Vec3(xs))
            }
            _ => None,
        };
        let Some(val) = val else {
            let want = match ty {
                ParamType::Scalar => "a number",
                ParamType::Vec3 => "an array of 3 numbers",
            };
            return Err(LoadError::new(
                code,
                p,
                format!("param `{name}` of `{what}` must be {want}"),
            ));
        };
        out.push((name, val));
    }
    Ok(out)
}

fn scalar(vals: &[(&str, Param)], name: &str) -> f64 {
    match vals.iter().find(|(n, _)| *n == name) {
        Some((_, Param::Scalar(x))) => *x,
        // Unreachable after typed_params; NaN fails every rule below.
        _ => f64::NAN,
    }
}

fn vec3(vals: &[(&str, Param)], name: &str) -> [f64; 3] {
    match vals.iter().find(|(n, _)| *n == name) {
        Some((_, Param::Vec3(v))) => *v,
        // Unreachable after typed_params; NaN fails every rule below.
        _ => [f64::NAN; 3],
    }
}

/// `x > 0` (or `>= 0` when `allow_zero`), otherwise value.range naming the rule.
fn positive(x: f64, path: &str, what: &str, allow_zero: bool) -> Result<(), LoadError> {
    let ok = if allow_zero { x >= 0.0 } else { x > 0.0 };
    if ok {
        Ok(())
    } else {
        let rule = if allow_zero { ">= 0" } else { "> 0" };
        Err(range(path, format!("{what} must be {rule}, found {x}")))
    }
}

fn positive3(v: [f64; 3], path: &str, what: &str) -> Result<(), LoadError> {
    v.iter()
        .enumerate()
        .try_for_each(|(i, &x)| positive(x, &at(path, i), what, false))
}

/// The S3 rules of a kind (params already typed).
fn kind_rules(kind: &str, vals: &[(&str, Param)], pp: &str) -> Result<(), LoadError> {
    let p = |name: &str| ptr(pp, name);
    let what = |name: &str| format!("{kind} {name}");
    match kind {
        "box" => positive3(
            vec3(vals, "halfExtents"),
            &p("halfExtents"),
            &what("halfExtents"),
        ),
        "ellipsoid" => positive3(vec3(vals, "radii"), &p("radii"), &what("radii")),
        "roundBox" => {
            let he = vec3(vals, "halfExtents");
            positive3(he, &p("halfExtents"), &what("halfExtents"))?;
            // 0 is a plain box (published roundBox: radius in [0, min(size)/2]).
            let r = scalar(vals, "radius");
            positive(r, &p("radius"), &what("radius"), true)?;
            let min = he[0].min(he[1]).min(he[2]);
            if r <= min {
                Ok(())
            } else {
                Err(range(
                    &p("radius"),
                    format!("roundBox radius must be <= min(halfExtents) = {min}, found {r}"),
                ))
            }
        }
        "cone" => {
            // A frustum: each radius >= 0 and not both 0 (radiusTop is optional, 0 when absent).
            let radius = scalar(vals, "radius");
            let top = vals
                .iter()
                .find(|(n, _)| *n == "radiusTop")
                .map_or(0.0, |(_, v)| match v {
                    Param::Scalar(x) => *x,
                    Param::Vec3(_) => f64::NAN,
                });
            positive(radius, &p("radius"), &what("radius"), true)?;
            positive(top, &p("radiusTop"), &what("radiusTop"), true)?;
            if radius.max(top) <= 0.0 {
                return Err(range(&p("radius"), "a cone needs one radius > 0"));
            }
            positive(
                scalar(vals, "halfHeight"),
                &p("halfHeight"),
                &what("halfHeight"),
                false,
            )
        }
        "torus" => {
            let major = scalar(vals, "majorRadius");
            let minor = scalar(vals, "minorRadius");
            if minor > 0.0 && minor < major {
                Ok(())
            } else {
                Err(range(
                    &p("minorRadius"),
                    format!(
                        "torus needs 0 < minorRadius < majorRadius, found minorRadius {minor} and majorRadius {major}"
                    ),
                ))
            }
        }
        _ => {
            // sphere, capsule, cylinder, cone.
            for (name, _) in vals {
                let zero_ok = kind == "capsule" && *name == "halfHeight";
                positive(scalar(vals, name), &p(name), &what(name), zero_ok)?;
            }
            Ok(())
        }
    }
}

/// Checks run after the walk: material references, then collider docs (their params are
/// stored re-ordered to the collider table order).
fn references(a: &mut PrimsAsset, w: &Walk) -> Result<(), LoadError> {
    if let Some((p, m)) = w
        .materials
        .iter()
        .find(|(_, m)| !a.materials.iter().any(|(k, _)| k == m))
    {
        return Err(LoadError::new(
            "material.unknown",
            p.as_str(),
            format!("material `{m}` is not declared in materials"),
        ));
    }
    for (i, c) in a.collision.iter_mut().flatten().enumerate() {
        let cp = at("/collision/colliders", i);
        c.shape = internal_shape(&c.shape).to_string();
        let Some(list) = collider_params(&c.shape) else {
            return Err(LoadError::new(
                "collision.shape",
                ptr(&cp, "shape"),
                format!("unknown collider shape `{}`", c.shape),
            ));
        };
        let pp = ptr(&cp, "params");
        let vals = typed_params(&c.params, list, &[], &pp, "collision.shape", &c.shape)?;
        // A collider's own param values that are negative are collision.shape (like the
        // key and type problems above), not value.range. Zero is allowed: the published
        // derivation clamps halfHeight and halfExtents - borderRadius at 0.
        for (name, v) in &vals {
            let what = format!("{} {name}", c.shape);
            let bad = |x: f64| x.is_nan() || x < 0.0;
            match v {
                Param::Scalar(x) if bad(*x) => Err(range(
                    &ptr(&pp, name),
                    format!("{what} must be >= 0, found {x}"),
                )),
                Param::Vec3(xs) if xs.iter().any(|&x| bad(x)) => {
                    let i = xs.iter().position(|&x| bad(x)).unwrap_or(0);
                    Err(range(
                        &at(&ptr(&pp, name), i),
                        format!("{what} must be >= 0"),
                    ))
                }
                _ => Ok(()),
            }
            .map_err(|e| LoadError::new("collision.shape", e.path.as_str(), e.message))?;
        }
        c.params = list
            .iter()
            .filter_map(|(k, _)| c.params.iter().find(|(d, _)| d == k).cloned())
            .collect();
        if let Some(n) = &c.node
            && !w.ids.contains(n)
        {
            return Err(LoadError::new(
                "collision.shape",
                ptr(&cp, "node"),
                format!("collider node `{n}` does not exist"),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prims::samples::sample_assets_raw;
    use crate::prims::to_canonical_json;

    const MAT: &str = r#"{"color":[0.5,0.5,0.5],"roughness":0.5,"metallic":0,"emissive":[0,0,0]}"#;

    /// A valid document with `extra` inserted before `nodes` and the given materials.
    fn doc_with(extra: &str, nodes: &str, materials: &str) -> String {
        format!(
            r#"{{"format":"raa.prims","version":0,"id":"t","name":"t",{extra}"nodes":{nodes},"materials":{materials},"anchors":{{}}}}"#
        )
    }

    fn doc(extra: &str, nodes: &str) -> String {
        doc_with(extra, nodes, &format!(r#"{{"m":{MAT}}}"#))
    }

    fn sphere(rest: &str) -> String {
        doc(
            "",
            &format!(r#"[{{"id":"a","kind":"sphere","params":{{"radius":1}}{rest}}}]"#),
        )
    }

    #[track_caller]
    fn refused(text: &str, code: &str, path: &str) {
        let e = parse_with(text, true).expect_err("must be refused");
        assert_eq!((e.code, e.path.as_str()), (code, path), "{e}");
        assert!(!e.message.is_empty());
    }

    #[test]
    fn samples_parse_back_to_their_canonical_text() {
        for (name, asset) in sample_assets_raw() {
            let text = to_canonical_json(&asset);
            let parsed = parse_with(&text, true).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(parsed.nodes, asset.nodes, "{name}");
            assert_eq!(to_canonical_json(&parsed), text, "{name}");
        }
    }

    #[test]
    fn parse_reads_every_field() {
        let text = doc(
            r#""triangleBudget":900,"#,
            r#"[{"id":"a","kind":"box","params":{"halfExtents":[1,2,3]},"transform":{"t":[1,2,3],"r":[0,0,0,1],"s":[2,2,2]},"material":"m","op":"union","children":[{"id":"b","kind":"sphere","params":{"radius":0.5},"op":"subtract"}]}]"#,
        )
        .replace(
            r#""anchors":{}"#,
            r#""anchors":{"head":{"t":[0,1,0],"r":[0,0,0,1],"s":[1,1,1]}},"collision":{"colliders":[{"node":"a","shape":"ball","params":{"radius":1},"transform":{"t":[0,0,0],"r":[0,0,0,1],"s":[1,1,1]}}]}"#,
        );
        let a = crate::prims::parse(&text).unwrap();
        assert_eq!(a.triangle_budget, Some(900));
        let n = &a.nodes[0];
        assert_eq!(n.params[0].0, "halfExtents");
        assert_eq!(n.transform.as_ref().unwrap().s, [2.0; 3]);
        assert_eq!(n.material.as_deref(), Some("m"));
        assert_eq!(n.children[0].op, NodeOp::Subtract);
        assert_eq!(a.materials[0].1.roughness, 0.5);
        assert_eq!(a.anchors[0].0, "head");
        let c = &a.collision.as_ref().unwrap()[0];
        assert_eq!((c.node.as_deref(), c.shape.as_str()), (Some("a"), "ball"));
    }

    #[test]
    fn json_errors_pass_through() {
        let e = parse_with("{", true).unwrap_err();
        assert_eq!(e, json::parse("{").unwrap_err());
        assert_eq!(e.code, "json.syntax");
    }

    #[test]
    fn root_must_be_an_object() {
        refused("[]", "schema.type", "");
    }

    #[test]
    fn format_unknown() {
        refused(
            &doc("", "[]").replace("raa.prims", "x"),
            "format.unknown",
            "/format",
        );
    }

    #[test]
    fn format_version() {
        let t = doc("", "[]").replace(r#""version":0"#, r#""version":1"#);
        refused(&t, "format.version", "/version");
    }

    #[test]
    fn schema_unknown_field_top_and_node() {
        refused(
            &doc(r#""extra":1,"#, "[]"),
            "schema.unknown_field",
            "/extra",
        );
        refused(
            &sphere(r#","colour":"m""#),
            "schema.unknown_field",
            "/nodes/0/colour",
        );
    }

    #[test]
    fn schema_unknown_field_escapes_pointer() {
        refused(
            &doc(r#""a/b~c":1,"#, "[]"),
            "schema.unknown_field",
            "/a~1b~0c",
        );
    }

    #[test]
    fn schema_missing_field_after_scan() {
        let t = r#"{"format":"raa.prims","version":0,"id":"t","name":"t","materials":{},"anchors":{},"bad":1}"#;
        refused(t, "schema.unknown_field", "/bad");
        let t =
            r#"{"format":"raa.prims","version":0,"id":"t","name":"t","materials":{},"anchors":{}}"#;
        refused(t, "schema.missing_field", "/nodes");
        let t = doc("", r#"[{"id":"a","params":{"radius":1}}]"#);
        refused(&t, "schema.missing_field", "/nodes/0/kind");
        refused(
            &sphere(r#","transform":{"t":[0,0,0],"r":[0,0,0,1]}"#),
            "schema.missing_field",
            "/nodes/0/transform/s",
        );
    }

    #[test]
    fn schema_type() {
        let t = doc("", r#"[{"id":5,"kind":"sphere","params":{"radius":1}}]"#);
        refused(&t, "schema.type", "/nodes/0/id");
        refused(
            &sphere(r#","children":{}"#),
            "schema.type",
            "/nodes/0/children",
        );
        refused(
            &sphere(r#","transform":{"t":[0,0],"r":[0,0,0,1],"s":[1,1,1]}"#),
            "schema.type",
            "/nodes/0/transform/t",
        );
        refused(&doc("", "{}"), "schema.type", "/nodes");
    }

    #[test]
    fn value_not_finite_anywhere() {
        let t = doc(
            "",
            r#"[{"id":"a","kind":"sphere","params":{"radius":1e999}}]"#,
        );
        refused(&t, "value.not_finite", "/nodes/0/params/radius");
        refused(
            &sphere(r#","transform":{"t":[0,-1e999,0],"r":[0,0,0,1],"s":[1,1,1]}"#),
            "value.not_finite",
            "/nodes/0/transform/t/1",
        );
        // Finiteness is checked before the range rule.
        refused(
            &sphere(r#","transform":{"t":[0,0,0],"r":[0,0,0,1],"s":[1,-1e999,1]}"#),
            "value.not_finite",
            "/nodes/0/transform/s/1",
        );
    }

    #[test]
    fn value_range_transform() {
        refused(
            &sphere(r#","transform":{"t":[0,0,0],"r":[0,0,0,0.5],"s":[1,1,1]}"#),
            "value.range",
            "/nodes/0/transform/r",
        );
        refused(
            &sphere(r#","transform":{"t":[0,0,0],"r":[0,0,0,1],"s":[1,0,1]}"#),
            "value.range",
            "/nodes/0/transform/s/1",
        );
        // Within 1e-3 of 1 is accepted.
        let ok = sphere(r#","transform":{"t":[0,0,0],"r":[0,0,0,1.0009],"s":[1,1,1]}"#);
        assert!(parse_with(&ok, true).is_ok());
    }

    #[test]
    fn value_range_budget_and_op() {
        refused(
            &doc(r#""triangleBudget":0,"#, "[]"),
            "value.range",
            "/triangleBudget",
        );
        refused(
            &doc(r#""triangleBudget":1.5,"#, "[]"),
            "value.range",
            "/triangleBudget",
        );
        refused(&sphere(r#","op":"xor""#), "value.range", "/nodes/0/op");
    }

    #[test]
    fn value_range_material() {
        let bad = |m: &str| doc_with("", "[]", &format!(r#"{{"m":{m}}}"#));
        refused(
            &bad(&MAT.replace("[0.5,0.5,0.5]", "[0.5,1.5,0.5]")),
            "value.range",
            "/materials/m/color/1",
        );
        refused(
            &bad(&MAT.replace(r#""roughness":0.5"#, r#""roughness":2"#)),
            "value.range",
            "/materials/m/roughness",
        );
        refused(
            &bad(&MAT.replace(r#""metallic":0"#, r#""metallic":-0.1"#)),
            "value.range",
            "/materials/m/metallic",
        );
        refused(
            &bad(&MAT.replace("[0,0,0]", "[0,0,-1]")),
            "value.range",
            "/materials/m/emissive/2",
        );
    }

    #[test]
    fn id_invalid_reserved_material() {
        let t = doc_with("", "[]", &format!(r#"{{"_default":{MAT}}}"#));
        refused(&t, "id.invalid", "/materials/_default");
    }

    #[test]
    fn anchors_and_collision_shape_checked() {
        let t = doc("", "[]").replace(r#""anchors":{}"#, r#""anchors":{"h":{"t":[0,0,0]}}"#);
        refused(&t, "schema.missing_field", "/anchors/h/r");
        let t = doc("", "[]").replace(
            r#""anchors":{}"#,
            r#""anchors":{},"collision":{"colliders":[{"shape":"ball","params":{"radius":1},"transform":{"t":[0,0,0],"r":[0,0,0,1],"s":[1,1,1]},"x":0}]}"#,
        );
        refused(&t, "schema.unknown_field", "/collision/colliders/0/x");
        let t = doc("", "[]").replace(r#""anchors":{}"#, r#""anchors":{},"collision":{}"#);
        refused(&t, "schema.missing_field", "/collision/colliders");
    }

    #[test]
    fn first_error_in_document_order_wins() {
        let t = doc(r#""triangleBudget":0,"#, "[]").replace("raa.prims", "x");
        refused(&t, "format.unknown", "/format");
        refused(
            &doc(
                "",
                r#"[{"id":1,"kind":"sphere","params":{"radius":1},"zz":0}]"#,
            ),
            "schema.type",
            "/nodes/0/id",
        );
    }

    #[test]
    fn nested_children_paths() {
        let t = sphere(
            r#","children":[{"id":"b","kind":"sphere","params":{"radius":1},"children":[{"id":"c","kind":"sphere","params":{"radius":1},"op":"nope"}]}]"#,
        );
        refused(&t, "value.range", "/nodes/0/children/0/children/0/op");
    }

    fn leaf(id: &str, kind: &str, params: &str, rest: &str) -> String {
        format!(r#"{{"id":"{id}","kind":"{kind}","params":{params}{rest}}}"#)
    }

    fn one(kind: &str, params: &str) -> String {
        doc("", &format!("[{}]", leaf("a", kind, params, "")))
    }

    #[test]
    fn id_invalid_and_duplicate() {
        refused(
            &one("sphere", r#"{"radius":1}"#).replace(r#""id":"a""#, r#""id":"a b""#),
            "id.invalid",
            "/nodes/0/id",
        );
        refused(
            &one("sphere", r#"{"radius":1}"#).replace(r#""id":"a""#, r#""id":"""#),
            "id.invalid",
            "/nodes/0/id",
        );
        let s = leaf("a", "sphere", r#"{"radius":1}"#, "");
        let t = sphere(&format!(
            r#","children":[{}]"#,
            s.replace(r#""a""#, r#""b""#)
        ));
        assert!(parse_with(&t, true).is_ok());
        let t = doc(
            "",
            &format!(
                "[{},{}]",
                leaf(
                    "x_-9",
                    "sphere",
                    r#"{"radius":1}"#,
                    &format!(r#","children":[{s}]"#)
                ),
                s
            ),
        );
        refused(&t, "id.duplicate", "/nodes/1/id");
    }

    #[test]
    fn cap_nodes_at_the_65th_node() {
        let mk = |n: usize| {
            let nodes: Vec<String> = (0..n)
                .map(|i| leaf(&format!("n{i}"), "sphere", r#"{"radius":1}"#, ""))
                .collect();
            doc("", &format!("[{}]", nodes.join(",")))
        };
        assert!(parse_with(&mk(64), true).is_ok());
        refused(&mk(65), "cap.nodes", "/nodes/64");
    }

    #[test]
    fn cap_depth_above_level_8() {
        let chain = |levels: usize| {
            let mut n = leaf(&format!("l{levels}"), "sphere", r#"{"radius":1}"#, "");
            for l in (0..levels).rev() {
                n = leaf(
                    &format!("l{l}"),
                    "sphere",
                    r#"{"radius":1}"#,
                    &format!(r#","children":[{n}]"#),
                );
            }
            doc("", &format!("[{n}]"))
        };
        assert!(parse_with(&chain(8), true).is_ok());
        let path = format!("/nodes/0{}", "/children/0".repeat(9));
        refused(&chain(9), "cap.depth", &path);
    }

    #[test]
    fn kind_unknown_strict_and_kept_when_not_strict() {
        let t = one("pyramid", r#"{"b":1,"a":[1,2,3]}"#);
        refused(&t, "kind.unknown", "/nodes/0/kind");
        let a = parse_with(&t, false).unwrap();
        let keys: Vec<&str> = a.nodes[0].params.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["b", "a"]);
    }

    #[test]
    fn kind_params_keys_and_types() {
        refused(
            &one("capsule", r#"{"radius":1}"#),
            "kind.params",
            "/nodes/0/params/halfHeight",
        );
        refused(
            &one("sphere", r#"{"radius":1,"height":2}"#),
            "kind.params",
            "/nodes/0/params/height",
        );
        refused(
            &one("box", r#"{"halfExtents":1}"#),
            "kind.params",
            "/nodes/0/params/halfExtents",
        );
        refused(
            &one("box", r#"{"halfExtents":[1,1]}"#),
            "kind.params",
            "/nodes/0/params/halfExtents",
        );
        refused(
            &one("sphere", r#"{"radius":"1"}"#),
            "kind.params",
            "/nodes/0/params/radius",
        );
        refused(
            &one("ellipsoid", r#"{"radii":[1,true,1]}"#),
            "kind.params",
            "/nodes/0/params/radii",
        );
    }

    #[test]
    fn params_reordered_to_s3_order() {
        let a = parse_with(
            &one("roundBox", r#"{"radius":0.1,"halfExtents":[1,1,1]}"#),
            true,
        )
        .unwrap();
        let keys: Vec<&str> = a.nodes[0].params.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["halfExtents", "radius"]);
    }

    #[test]
    fn value_range_s3_rules() {
        let cases = [
            ("sphere", r#"{"radius":0}"#, "/nodes/0/params/radius"),
            (
                "box",
                r#"{"halfExtents":[1,-1,1]}"#,
                "/nodes/0/params/halfExtents/1",
            ),
            (
                "roundBox",
                r#"{"halfExtents":[1,0.5,1],"radius":0.6}"#,
                "/nodes/0/params/radius",
            ),
            (
                "capsule",
                r#"{"radius":1,"halfHeight":-0.1}"#,
                "/nodes/0/params/halfHeight",
            ),
            (
                "cylinder",
                r#"{"radius":1,"halfHeight":0}"#,
                "/nodes/0/params/halfHeight",
            ),
            (
                "cone",
                r#"{"radius":0,"halfHeight":1}"#,
                "/nodes/0/params/radius",
            ),
            (
                "torus",
                r#"{"majorRadius":1,"minorRadius":1}"#,
                "/nodes/0/params/minorRadius",
            ),
            (
                "torus",
                r#"{"majorRadius":1,"minorRadius":0}"#,
                "/nodes/0/params/minorRadius",
            ),
            (
                "ellipsoid",
                r#"{"radii":[1,1,0]}"#,
                "/nodes/0/params/radii/2",
            ),
        ];
        for (kind, params, path) in cases {
            refused(&one(kind, params), "value.range", path);
        }
        assert!(parse_with(&one("capsule", r#"{"radius":1,"halfHeight":0}"#), true).is_ok());
        assert!(
            parse_with(
                &one("roundBox", r#"{"halfExtents":[1,0.5,1],"radius":0.5}"#),
                true
            )
            .is_ok()
        );
    }

    #[test]
    fn value_range_boolean_ops() {
        refused(&sphere(r#","op":"subtract""#), "value.range", "/nodes/0/op");
        let tool = leaf(
            "b",
            "sphere",
            r#"{"radius":1}"#,
            r#","op":"intersect","children":[{"id":"c","kind":"sphere","params":{"radius":1}}]"#,
        );
        // A tool with children parses; refusing it is the BooleanResolver's job.
        let a = parse_with(&sphere(&format!(r#","children":[{tool}]"#)), true).unwrap();
        let t = &a.nodes[0].children[0];
        assert_eq!((t.op, t.children.len()), (NodeOp::Intersect, 1));
        assert_eq!(t.children[0].id, "c");
        let tool = leaf(
            "b",
            "sphere",
            r#"{"radius":1}"#,
            r#","op":"subtract","children":[]"#,
        );
        assert!(parse_with(&sphere(&format!(r#","children":[{tool}]"#)), true).is_ok());
    }

    #[test]
    fn material_unknown_first_in_pre_order() {
        let t = sphere(r#","material":"m""#);
        assert!(parse_with(&t, true).is_ok());
        let c1 = leaf("b", "sphere", r#"{"radius":1}"#, r#","material":"x""#);
        let c2 = leaf("c", "sphere", r#"{"radius":1}"#, r#","material":"y""#);
        let t = sphere(&format!(r#","material":"m","children":[{c1},{c2}]"#));
        refused(&t, "material.unknown", "/nodes/0/children/0/material");
    }

    fn with_collider(c: &str) -> String {
        sphere("").replace(
            r#""anchors":{}"#,
            &format!(r#""anchors":{{}},"collision":{{"colliders":[{c}]}}"#),
        )
    }

    const ID: &str = r#""transform":{"t":[0,0,0],"r":[0,0,0,1],"s":[1,1,1]}"#;

    #[test]
    fn collision_shape_and_params() {
        let ok = with_collider(&format!(
            r#"{{"node":"a","shape":"capsuleY","params":{{"radius":1,"halfHeight":2}},{ID}}}"#
        ));
        let a = parse_with(&ok, true).unwrap();
        let keys: Vec<&str> = a.collision.as_ref().unwrap()[0]
            .params
            .iter()
            .map(|(k, _)| k.as_str())
            .collect();
        assert_eq!(keys, ["halfHeight", "radius"]);
        let p = "/collision/colliders/0";
        refused(
            &with_collider(&format!(r#"{{"shape":"torusY","params":{{}},{ID}}}"#)),
            "collision.shape",
            &format!("{p}/shape"),
        );
        refused(
            &with_collider(&format!(r#"{{"shape":"ball","params":{{}},{ID}}}"#)),
            "collision.shape",
            &format!("{p}/params/radius"),
        );
        refused(
            &with_collider(&format!(
                r#"{{"shape":"ball","params":{{"radius":1,"r":1}},{ID}}}"#
            )),
            "collision.shape",
            &format!("{p}/params/r"),
        );
        refused(
            &with_collider(&format!(
                r#"{{"shape":"cuboid","params":{{"halfExtents":1}},{ID}}}"#
            )),
            "collision.shape",
            &format!("{p}/params/halfExtents"),
        );
        refused(
            &with_collider(&format!(
                r#"{{"node":"ghost","shape":"ball","params":{{"radius":1}},{ID}}}"#
            )),
            "collision.shape",
            &format!("{p}/node"),
        );
    }

    #[test]
    fn collision_values_must_be_positive() {
        let p = "/collision/colliders/0/params";
        refused(
            &with_collider(&format!(
                r#"{{"shape":"ball","params":{{"radius":-1}},{ID}}}"#
            )),
            "collision.shape",
            &format!("{p}/radius"),
        );
        refused(
            &with_collider(&format!(
                r#"{{"shape":"cuboid","params":{{"halfExtents":[1,-1,1]}},{ID}}}"#
            )),
            "collision.shape",
            &format!("{p}/halfExtents/1"),
        );
        refused(
            &with_collider(&format!(
                r#"{{"shape":"capsuleY","params":{{"halfHeight":-1,"radius":1}},{ID}}}"#
            )),
            "collision.shape",
            &format!("{p}/halfHeight"),
        );
    }

    /// Parsed node `n` of a one-node document: the S3 value of param `name`.
    fn s3_num(a: &PrimsAsset, name: &str) -> f64 {
        match a.nodes[0].params.iter().find(|(k, _)| k == name) {
            Some((_, JsonValue::Number(x))) => *x,
            other => panic!("{name}: {other:?}"),
        }
    }

    fn s3_vec(a: &PrimsAsset, name: &str) -> Vec<f64> {
        match a.nodes[0].params.iter().find(|(k, _)| k == name) {
            Some((_, JsonValue::Array(v))) => v
                .iter()
                .map(|x| match x {
                    JsonValue::Number(x) => *x,
                    other => panic!("{other:?}"),
                })
                .collect(),
            other => panic!("{name}: {other:?}"),
        }
    }

    fn has_param(a: &PrimsAsset, name: &str) -> bool {
        a.nodes[0].params.iter().any(|(k, _)| k == name)
    }

    #[test]
    fn published_box_and_round_box_sizes_become_half_extents() {
        let a = parse_with(&one("box", r#"{"size":[2,4,6]}"#), true).unwrap();
        assert_eq!(a.nodes[0].kind, "box");
        assert_eq!(s3_vec(&a, "halfExtents"), [1.0, 2.0, 3.0]);
        let rb = parse_with(&one("roundBox", r#"{"size":[2,2,2],"radius":0.5}"#), true).unwrap();
        assert_eq!(s3_vec(&rb, "halfExtents"), [1.0, 1.0, 1.0]);
        assert_eq!(s3_num(&rb, "radius"), 0.5);
        // Radius 0 is a plain box; radius above min(size)/2 is refused.
        let plain = parse_with(&one("roundBox", r#"{"size":[2,2,2],"radius":0}"#), true).unwrap();
        assert_eq!(s3_num(&plain, "radius"), 0.0);
        refused(
            &one("roundBox", r#"{"size":[2,2,2],"radius":1.5}"#),
            "value.range",
            "/nodes/0/params/radius",
        );
    }

    #[test]
    fn published_capsule_height_is_total_with_caps() {
        // height 1.2, radius 0.4: body half-length 0.6 - 0.4 = 0.2.
        let a = parse_with(&one("capsule", r#"{"height":1.2,"radius":0.4}"#), true).unwrap();
        assert_eq!(s3_num(&a, "radius"), 0.4);
        assert!((s3_num(&a, "halfHeight") - 0.2).abs() < 1e-12);
        assert!(!has_param(&a, "height"));
        // height = 2 * radius is a sphere: halfHeight 0, accepted.
        let sphere = parse_with(&one("capsule", r#"{"height":0.8,"radius":0.4}"#), true).unwrap();
        assert_eq!(s3_num(&sphere, "halfHeight"), 0.0);
        refused(
            &one("capsule", r#"{"height":0.7,"radius":0.4}"#),
            "value.range",
            "/nodes/0/params/height",
        );
    }

    #[test]
    fn published_cylinder_height_is_full_height() {
        let a = parse_with(&one("cylinder", r#"{"height":0.9,"radius":0.3}"#), true).unwrap();
        assert_eq!(
            (s3_num(&a, "radius"), s3_num(&a, "halfHeight")),
            (0.3, 0.45)
        );
    }

    #[test]
    fn published_cone_maps_base_and_top_radii() {
        let flat_top = parse_with(
            &one("cone", r#"{"height":1,"radiusBottom":0.4,"radiusTop":0}"#),
            true,
        )
        .unwrap();
        assert_eq!(
            (s3_num(&flat_top, "radius"), s3_num(&flat_top, "halfHeight")),
            (0.4, 0.5)
        );
        assert!(
            !has_param(&flat_top, "radiusTop"),
            "a 0 top is the S3 pointed cone"
        );
        let flat_bottom = parse_with(
            &one("cone", r#"{"height":0.8,"radiusBottom":0,"radiusTop":0.3}"#),
            true,
        )
        .unwrap();
        assert_eq!(s3_num(&flat_bottom, "radius"), 0.0);
        assert_eq!(s3_num(&flat_bottom, "radiusTop"), 0.3);
        let frustum = parse_with(
            &one(
                "cone",
                r#"{"height":0.6,"radiusBottom":0.35,"radiusTop":0.2}"#,
            ),
            true,
        )
        .unwrap();
        assert_eq!(s3_num(&frustum, "radiusTop"), 0.2);
        refused(
            &one("cone", r#"{"height":1,"radiusBottom":0,"radiusTop":0}"#),
            "value.range",
            "/nodes/0/params/radius",
        );
        // radiusTop is required by the published form.
        refused(
            &one("cone", r#"{"height":1,"radiusBottom":0.4}"#),
            "kind.params",
            "/nodes/0/params/radiusTop",
        );
    }

    #[test]
    fn s3_cone_accepts_an_optional_top_radius() {
        let a = parse_with(&one("cone", r#"{"radius":0.3,"halfHeight":0.5}"#), true).unwrap();
        assert!(!has_param(&a, "radiusTop"));
        let b = parse_with(
            &one("cone", r#"{"radius":0.3,"halfHeight":0.5,"radiusTop":0.1}"#),
            true,
        )
        .unwrap();
        assert_eq!(s3_num(&b, "radiusTop"), 0.1);
        refused(
            &one("cone", r#"{"radius":0,"halfHeight":0.5}"#),
            "value.range",
            "/nodes/0/params/radius",
        );
    }

    #[test]
    fn published_names_do_not_mix_with_s3_names_and_unknown_params_are_refused() {
        refused(
            &one("capsule", r#"{"height":1,"radius":0.4,"halfHeight":0.1}"#),
            "kind.params",
            "/nodes/0/params/halfHeight",
        );
        refused(
            &one("box", r#"{"size":[1,1,1],"depth":2}"#),
            "kind.params",
            "/nodes/0/params/depth",
        );
        refused(
            &one("cylinder", r#"{"height":1,"radius":0.4,"halfHeight":0.5}"#),
            "kind.params",
            "/nodes/0/params/halfHeight",
        );
        refused(
            &one("cone", r#"{"radius":1,"halfHeight":1,"radiusBottom":1}"#),
            "kind.params",
            "/nodes/0/params/radius",
        );
        refused(
            &one("sphere", r#"{"radius":1,"size":[1,1,1]}"#),
            "kind.params",
            "/nodes/0/params/size",
        );
    }

    #[test]
    fn published_budget_object_and_its_range() {
        let text = doc_with(
            r#""budget":{"triangles":4000,"segments":32},"#,
            &format!("[{}]", leaf("a", "sphere", r#"{"radius":1}"#, "")),
            &format!(r#"{{"m":{MAT}}}"#),
        );
        assert_eq!(parse_with(&text, true).unwrap().triangle_budget, Some(4000));
        let bad = doc_with(
            r#""budget":{"triangles":4000,"segments":30},"#,
            &format!("[{}]", leaf("a", "sphere", r#"{"radius":1}"#, "")),
            &format!(r#"{{"m":{MAT}}}"#),
        );
        refused(&bad, "value.range", "/budget/segments");
        let both = doc_with(
            r#""budget":{"triangles":4000},"triangleBudget":4000,"#,
            &format!("[{}]", leaf("a", "sphere", r#"{"radius":1}"#, "")),
            &format!(r#"{{"m":{MAT}}}"#),
        );
        refused(&both, "schema.unknown_field", "/budget");
    }

    #[test]
    fn published_collision_mode_anchors_and_colliders() {
        let text = format!(
            r#"{{"format":"raa.prims","version":0,"id":"t","name":"t","nodes":[{node}],"materials":{{"m":{MAT}}},"anchors":{{"spawn":{{"t":[0,1,0],"r":[0,0,0,1]}}}},"collision":{{"mode":"override","colliders":[{{"node":null,"shape":"capsule","params":{{"height":1,"radius":0.2}},"t":[0,0,0],"r":[0,0,0,1]}}]}}}}"#,
            node = leaf("a", "sphere", r#"{"radius":1}"#, ""),
        );
        // Published capsule collider params are halfHeight and radius (no height): refused
        // as a collision.shape with the unexpected key.
        refused(
            &text,
            "collision.shape",
            "/collision/colliders/0/params/height",
        );
        let ok = text.replace(
            r#""params":{"height":1,"radius":0.2}"#,
            r#""params":{"halfHeight":0.8,"radius":0.2}"#,
        );
        let a = parse_with(&ok, true).unwrap();
        let c = &a.collision.as_ref().unwrap()[0];
        assert_eq!((c.node.as_deref(), c.shape.as_str()), (None, "capsuleY"));
        assert_eq!(c.transform.t, [0.0, 0.0, 0.0]);
        assert_eq!(a.anchors[0].1.t, [0.0, 1.0, 0.0]);
        assert_eq!(a.anchors[0].1.s, [1.0, 1.0, 1.0]);
        let bad_mode = ok.replace(r#""mode":"override""#, r#""mode":"auto""#);
        refused(&bad_mode, "value.range", "/collision/mode");
        let both = ok.replace(
            r#""t":[0,0,0],"r":[0,0,0,1]}"#,
            r#""t":[0,0,0],"r":[0,0,0,1],"transform":{"t":[0,0,0],"r":[0,0,0,1],"s":[1,1,1]}}"#,
        );
        refused(&both, "schema.unknown_field", "/collision/colliders/0");
    }
}
