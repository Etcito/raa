//! S16 bone data and parent-before-child hierarchy validation.

use crate::{
    AnimError, JsonValue,
    json::{parse_rig, pointer_child},
};
use std::collections::BTreeMap;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainSolver {
    TwoBone,
    Ccd,
    Fabrik,
}
#[derive(Debug, Clone, PartialEq)]
pub struct IkChain {
    pub name: String,
    pub solver: ChainSolver,
    pub bones: Vec<usize>,
    pub pole: Option<[f64; 3]>,
    pub iterations: usize,
}
#[derive(Debug, Clone, PartialEq)]
pub struct JointLimit {
    pub bone: usize,
    pub min: [f64; 3],
    pub max: [f64; 3],
}
#[derive(Debug, Clone, PartialEq)]
pub struct Attachment {
    pub bone: usize,
    pub t: [f64; 3],
    pub r: [f64; 4],
}
#[derive(Debug, Clone, PartialEq)]
pub struct Rig {
    pub name: String,
    pub bones: Vec<Bone>,
    pub roles: BTreeMap<String, usize>,
    pub ik_chains: Vec<IkChain>,
    pub limits: Vec<JointLimit>,
    pub attach: BTreeMap<String, Attachment>,
}
const ROLE_NAMES: &str = concat!(
    "root hips spine chest neck head jaw tail shoulder.l shoulder.r upperArm.l upperArm.r ",
    "lowerArm.l lowerArm.r hand.l hand.r upperLeg.l upperLeg.r lowerLeg.l lowerLeg.r ",
    "foot.l foot.r toe.l toe.r eye.l eye.r"
);

impl Rig {
    /// Parse S16 bones, hierarchy, roles, IK chains, limits and attachments.
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
        let (roles, ik_chains, limits, attach) = metadata(obj, &bones)?;
        Ok(Self {
            name,
            bones,
            roles,
            ik_chains,
            limits,
            attach,
        })
    }
}

type Metadata = (
    BTreeMap<String, usize>,
    Vec<IkChain>,
    Vec<JointLimit>,
    BTreeMap<String, Attachment>,
);

const CHAIN_KEYS: [&str; 5] = ["name", "solver", "bones", "pole", "iterations"];
const ATTACH_KEYS: [&str; 3] = ["bone", "t", "r"];

fn metadata(fields: &[(String, JsonValue)], bones: &[Bone]) -> Result<Metadata, AnimError> {
    let mut roles = BTreeMap::new();
    if let Some(value) = optional(fields, "roles") {
        let JsonValue::Object(entries) = value else {
            return Err(error("rig.schema", "/roles", "expected object"));
        };
        for (role, value) in entries {
            let path = pointer_child("/roles", role);
            if !ROLE_NAMES.split(' ').any(|name| name == role) {
                return Err(error("rig.role_unknown", &path, "unknown role"));
            }
            let bone = bone_at(bones, string(value, &path)?, &path, "rig.role_unknown")?;
            roles.insert(role.clone(), bone);
        }
    }
    let mut ik_chains = Vec::new();
    if let Some(value) = optional(fields, "ikChains") {
        let JsonValue::Array(entries) = value else {
            return Err(error("rig.schema", "/ikChains", "expected array"));
        };
        for (i, value) in entries.iter().enumerate() {
            let path = format!("/ikChains/{i}");
            let chain = object(value, &path, &CHAIN_KEYS)?;
            let name = string(
                required(chain, "name", &path)?,
                &pointer_child(&path, "name"),
            )?;
            let solver_path = pointer_child(&path, "solver");
            let solver = match string(required(chain, "solver", &path)?, &solver_path)? {
                "twoBone" => ChainSolver::TwoBone,
                "ccd" => ChainSolver::Ccd,
                "fabrik" => ChainSolver::Fabrik,
                _ => return Err(error("rig.chain", &solver_path, "unknown solver")),
            };
            let bones_path = pointer_child(&path, "bones");
            let JsonValue::Array(names) = required(chain, "bones", &path)? else {
                return Err(error("rig.chain", &bones_path, "expected array"));
            };
            let mut chain_bones = Vec::with_capacity(names.len());
            for (j, name) in names.iter().enumerate() {
                let item_path = pointer_child(&bones_path, &j.to_string());
                let bone = bone_at(bones, string(name, &item_path)?, &item_path, "rig.chain")?;
                if let Some(previous) = chain_bones.last()
                    && bones[bone].parent != Some(*previous)
                {
                    return Err(error("rig.chain", &item_path, "chain order"));
                }
                chain_bones.push(bone);
            }
            let count_ok = match solver {
                ChainSolver::TwoBone => chain_bones.len() == 3,
                _ => (2..=8).contains(&chain_bones.len()),
            };
            if !count_ok {
                return Err(error("rig.chain", &bones_path, "wrong bone count"));
            }
            let pole_path = pointer_child(&path, "pole");
            let pole = match optional(chain, "pole") {
                None | Some(JsonValue::Null) => None,
                Some(value) => {
                    let pole: [f64; 3] = vector(value, &pole_path)?;
                    let norm = pole.iter().map(|v| v * v).sum::<f64>().sqrt();
                    if (norm - 1.0).abs() > 1e-3 {
                        return Err(error("rig.chain", &pole_path, "pole must be unit"));
                    }
                    Some(pole)
                }
            };
            let iterations_path = pointer_child(&path, "iterations");
            let iterations = match optional(chain, "iterations") {
                None => 16,
                Some(JsonValue::Number(n)) if n.fract() == 0.0 && (1.0..=16.0).contains(n) => {
                    *n as usize
                }
                Some(_) => return Err(error("rig.chain", &iterations_path, "bad iterations")),
            };
            ik_chains.push(IkChain {
                name: name.to_owned(),
                solver,
                bones: chain_bones,
                pole,
                iterations,
            });
        }
    }
    let mut limits = Vec::new();
    if let Some(value) = optional(fields, "limits") {
        let JsonValue::Array(entries) = value else {
            return Err(error("rig.schema", "/limits", "expected array"));
        };
        for (i, value) in entries.iter().enumerate() {
            let path = format!("/limits/{i}");
            let limit = object(value, &path, &["bone", "min", "max"])?;
            let bone_path = pointer_child(&path, "bone");
            let name = string(required(limit, "bone", &path)?, &bone_path)?;
            let bone = bone_at(bones, name, &bone_path, "rig.limit")?;
            let min_path = pointer_child(&path, "min");
            let min: [f64; 3] = vector(required(limit, "min", &path)?, &min_path)?;
            let max: [f64; 3] =
                vector(required(limit, "max", &path)?, &pointer_child(&path, "max"))?;
            if min.iter().zip(max).any(|(low, high)| low > &high) {
                return Err(error("rig.limit", &min_path, "min > max"));
            }
            limits.push(JointLimit { bone, min, max });
        }
    }
    let mut attach = BTreeMap::new();
    if let Some(value) = optional(fields, "attach") {
        let JsonValue::Object(entries) = value else {
            return Err(error("rig.schema", "/attach", "expected object"));
        };
        for (name, value) in entries {
            let path = pointer_child("/attach", name);
            let point = object(value, &path, &ATTACH_KEYS)?;
            let bone_path = pointer_child(&path, "bone");
            let bone_name = string(required(point, "bone", &path)?, &bone_path)?;
            let bone = bone_at(bones, bone_name, &bone_path, "rig.attach")?;
            let t = vector(required(point, "t", &path)?, &pointer_child(&path, "t"))?;
            let r_path = pointer_child(&path, "r");
            let r: [f64; 4] = vector(required(point, "r", &path)?, &r_path)?;
            let norm = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            if (norm - 1.0).abs() > 1e-3 {
                return Err(error("rig.attach", &r_path, "bad quaternion"));
            }
            attach.insert(name.clone(), Attachment { bone, t, r });
        }
    }
    Ok((roles, ik_chains, limits, attach))
}

fn bone_at(bones: &[Bone], name: &str, path: &str, code: &'static str) -> Result<usize, AnimError> {
    bones
        .iter()
        .position(|bone| bone.name == name)
        .ok_or_else(|| error(code, path, "unknown bone"))
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

fn optional<'a>(fields: &'a [(String, JsonValue)], key: &str) -> Option<&'a JsonValue> {
    fields
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, value)| value)
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
