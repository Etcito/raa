//! S16 bone data and parent-before-child hierarchy validation.

use crate::{
    AnimError, JsonValue,
    json::{parse_rig, pointer_child},
};

/// Maximum number of bones accepted in an S16 rig; larger rigs are refused.
pub const MAX_BONES: usize = 256;

/// Local rest transform. Source numbers remain f64 for lossless canonical writing.
#[derive(Debug, Clone, PartialEq)]
pub struct RestTransform {
    pub t: [f64; 3],
    pub r: [f64; 4],
    pub s: [f64; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Bone {
    pub name: String,
    /// Index of an earlier bone, or None for the single root.
    pub parent: Option<usize>,
    pub rest: RestTransform,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Rig {
    pub name: String,
    pub bones: Vec<Bone>,
}

impl Rig {
    /// Parse bones and hierarchy. Roles, chains, limits and attachments are
    /// reserved S16 fields whose validation is implemented separately.
    pub fn load(text: &str) -> Result<Self, AnimError> {
        let value = parse_rig(text)?;
        let obj = object(
            &value,
            "",
            &[
                "format", "version", "name", "bones", "roles", "ikChains", "limits", "attach",
            ],
        )?;
        if required(obj, "format", "")? != &JsonValue::String("raa.rig".into()) {
            return Err(error("rig.format", "/format", "expected raa.rig"));
        }
        if required(obj, "version", "")? != &JsonValue::Number(1.0) {
            return Err(error("rig.version", "/version", "expected version 1"));
        }
        let name = string(required(obj, "name", "")?, "/name")?.to_owned();
        let JsonValue::Array(values) = required(obj, "bones", "")? else {
            return Err(error("rig.schema", "/bones", "expected array"));
        };
        if values.len() > MAX_BONES {
            return Err(error(
                "rig.cap_bones",
                "/bones",
                "at most 256 bones allowed",
            ));
        }
        let mut bones: Vec<Bone> = Vec::with_capacity(values.len());
        let mut roots = 0;
        for (i, value) in values.iter().enumerate() {
            let path = format!("/bones/{i}");
            let obj = object(value, &path, &["name", "parent", "rest"])?;
            let name_path = pointer_child(&path, "name");
            let name = string(required(obj, "name", &path)?, &name_path)?;
            if name.is_empty()
                || name.len() > 32
                || !name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b'-'))
            {
                return Err(error("rig.bone_name", &name_path, "invalid bone name"));
            }
            if bones.iter().any(|bone| bone.name == name) {
                return Err(error(
                    "rig.bone_duplicate",
                    &name_path,
                    "duplicate bone name",
                ));
            }
            let parent_path = pointer_child(&path, "parent");
            let parent = match required(obj, "parent", &path)? {
                JsonValue::Null => {
                    roots += 1;
                    None
                }
                JsonValue::String(parent) => Some(
                    bones
                        .iter()
                        .position(|b| &b.name == parent)
                        .ok_or_else(|| {
                            error(
                                "rig.parent_order",
                                &parent_path,
                                "parent must name an earlier bone",
                            )
                        })?,
                ),
                _ => return Err(error("rig.schema", &parent_path, "expected string or null")),
            };
            let rest_path = pointer_child(&path, "rest");
            let rest = object(required(obj, "rest", &path)?, &rest_path, &["t", "r", "s"])?;
            let t = vector(
                required(rest, "t", &rest_path)?,
                &pointer_child(&rest_path, "t"),
            )?;
            let r: [f64; 4] = vector(
                required(rest, "r", &rest_path)?,
                &pointer_child(&rest_path, "r"),
            )?;
            let s: [f64; 3] = vector(
                required(rest, "s", &rest_path)?,
                &pointer_child(&rest_path, "s"),
            )?;
            let norm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if !norm.is_finite() || (norm - 1.0).abs() > 1e-3 {
                return Err(error(
                    "rig.schema",
                    &pointer_child(&rest_path, "r"),
                    "quaternion norm must be within 1e-3 of 1",
                ));
            }
            for (j, scale) in s.iter().enumerate() {
                if *scale <= 0.0 {
                    return Err(error(
                        "rig.schema",
                        &format!("{rest_path}/s/{j}"),
                        "scale must be positive",
                    ));
                }
            }
            bones.push(Bone {
                name: name.to_owned(),
                parent,
                rest: RestTransform { t, r, s },
            });
        }
        if roots != 1 {
            return Err(error("rig.root", "/bones", "exactly one root required"));
        }
        Ok(Self { name, bones })
    }
}

fn error(code: &'static str, path: &str, message: &str) -> AnimError {
    AnimError::new(code, path, message)
}

fn object<'a>(
    value: &'a JsonValue,
    path: &str,
    allowed: &[&str],
) -> Result<&'a [(String, JsonValue)], AnimError> {
    let JsonValue::Object(fields) = value else {
        return Err(error("rig.schema", path, "expected object"));
    };
    for (key, _) in fields {
        if !allowed.contains(&key.as_str()) {
            return Err(error(
                "rig.schema",
                &pointer_child(path, key),
                "unknown field",
            ));
        }
    }
    Ok(fields)
}

fn required<'a>(
    fields: &'a [(String, JsonValue)],
    key: &str,
    path: &str,
) -> Result<&'a JsonValue, AnimError> {
    fields
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v)
        .ok_or_else(|| error("rig.schema", &pointer_child(path, key), "missing field"))
}

fn string<'a>(value: &'a JsonValue, path: &str) -> Result<&'a str, AnimError> {
    match value {
        JsonValue::String(s) => Ok(s),
        _ => Err(error("rig.schema", path, "expected string")),
    }
}

fn vector<const N: usize>(value: &JsonValue, path: &str) -> Result<[f64; N], AnimError> {
    let JsonValue::Array(values) = value else {
        return Err(error("rig.schema", path, "expected vector array"));
    };
    if values.len() != N {
        return Err(error("rig.schema", path, "wrong vector length"));
    }
    let mut result = [0.0; N];
    for (i, value) in values.iter().enumerate() {
        let JsonValue::Number(number) = value else {
            return Err(error(
                "rig.schema",
                &pointer_child(path, &i.to_string()),
                "expected number",
            ));
        };
        result[i] = *number;
    }
    Ok(result)
}
