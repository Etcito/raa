//! Strict S19 field decoding for the document header, typed parameters,
//! conditions and blend-space declarations. Source numbers stay in [`Num::doc`];
//! layers, machines and root motion are decoded by their own passes.

use crate::error::{GraphError, codes};
use crate::model::{BlendPoint, BlendSpace, CondOp, Condition, GRAPH_CAPS, Num, Param, ParamType};
use raa_anim::json::{JsonValue, parse, pointer_child};

/// S19 document format tag.
pub const FORMAT: &str = "raa.animgraph";

const TOP: [&str; 9] = [
    "format",
    "version",
    "name",
    "rig",
    "params",
    "blendSpaces",
    "layers",
    "machines",
    "rootMotion",
];

/// Sections decoded by this pass.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphParts {
    pub name: String,
    pub rig: String,
    pub params: Vec<Param>,
    pub blend_spaces: Vec<BlendSpace>,
}

/// Parse a complete S19 document and decode the header, parameters and blend
/// spaces. Reader failures become `graph.schema`; unknown fields are refused.
pub fn decode_document(text: &str) -> Result<GraphParts, GraphError> {
    let value = parse(text).map_err(|e| err(codes::SCHEMA, &e.path, e.message))?;
    let fields = object(&value, "", &TOP)?;
    if required(fields, "format", "")? != &JsonValue::String(FORMAT.into()) {
        return Err(err(codes::FORMAT, "/format", "expected raa.animgraph"));
    }
    if required(fields, "version", "")? != &JsonValue::Number(1.0) {
        return Err(err(codes::VERSION, "/version", "expected version 1"));
    }
    let name = string(required(fields, "name", "")?, "/name")?.to_owned();
    let rig = string(required(fields, "rig", "")?, "/rig")?.to_owned();
    let params = params(array(required(fields, "params", "")?, "/params")?)?;
    let spaces = blend_spaces(
        array(required(fields, "blendSpaces", "")?, "/blendSpaces")?,
        &params,
    )?;
    Ok(GraphParts {
        name,
        rig,
        params,
        blend_spaces: spaces,
    })
}

fn params(values: &[JsonValue]) -> Result<Vec<Param>, GraphError> {
    GraphError::check_cap(values.len(), GRAPH_CAPS.max_params, "/params", "parameters")?;
    let mut params: Vec<Param> = Vec::with_capacity(values.len());
    for (i, value) in values.iter().enumerate() {
        let path = format!("/params/{i}");
        let fields = object(value, &path, &["name", "type", "default"])?;
        let name_path = pointer_child(&path, "name");
        let name = string(required(fields, "name", &path)?, &name_path)?;
        if name.is_empty() {
            return Err(err(codes::PARAM, &name_path, "empty parameter name"));
        }
        if params.iter().any(|param| param.name == name) {
            return Err(err(codes::PARAM, &name_path, "duplicate parameter name"));
        }
        let ty_path = pointer_child(&path, "type");
        let ty = match string(required(fields, "type", &path)?, &ty_path)? {
            "float" => ParamType::Float,
            "int" => ParamType::Int,
            "bool" => ParamType::Bool,
            "trigger" => ParamType::Trigger,
            _ => return Err(err(codes::PARAM, &ty_path, "unknown parameter type")),
        };
        let default_path = pointer_child(&path, "default");
        let default = match (ty, optional(fields, "default")) {
            (ParamType::Float, Some(value)) => number(value, &default_path, codes::PARAM)?,
            (ParamType::Int, Some(value)) => integer(value, &default_path, codes::PARAM)?,
            (ParamType::Bool, Some(JsonValue::Bool(b))) => Num::new(if *b { 1.0 } else { 0.0 }),
            (ParamType::Trigger, None) => Num::new(0.0),
            (ParamType::Trigger, Some(_)) => {
                return Err(err(codes::PARAM, &default_path, "triggers have no default"));
            }
            (_, Some(_)) => return Err(err(codes::PARAM, &default_path, "bad default type")),
            (_, None) => return Err(err(codes::PARAM, &default_path, "missing default")),
        };
        params.push(Param {
            name: name.to_owned(),
            ty,
            default,
        });
    }
    Ok(params)
}

/// Decode a transition `when` array against the declared parameters. The
/// operator must match the parameter type: `trigger` only for triggers,
/// equality for bools and comparisons for numeric parameters.
pub fn decode_conditions(
    value: &JsonValue,
    path: &str,
    params: &[Param],
) -> Result<Vec<Condition>, GraphError> {
    let values = array(value, path)?;
    GraphError::check_cap(
        values.len(),
        GRAPH_CAPS.max_conditions_per_transition,
        path,
        "conditions",
    )?;
    let mut conditions = Vec::with_capacity(values.len());
    for (i, value) in values.iter().enumerate() {
        let item = format!("{path}/{i}");
        let fields = object(value, &item, &["param", "op", "value"])?;
        let param_path = pointer_child(&item, "param");
        let name = string(required(fields, "param", &item)?, &param_path)?;
        let ty = params
            .iter()
            .find(|p| p.name == name)
            .map(|p| p.ty)
            .ok_or_else(|| err(codes::PARAM, &param_path, "unknown parameter"))?;
        let op_path = pointer_child(&item, "op");
        let op = condition_op(
            string(required(fields, "op", &item)?, &op_path)?,
            ty,
            &op_path,
        )?;
        let value_path = pointer_child(&item, "value");
        let value = match (op, optional(fields, "value")) {
            (CondOp::Trigger, None) => Num::new(0.0),
            (CondOp::Trigger, Some(_)) => {
                return Err(err(codes::PARAM, &value_path, "trigger takes no value"));
            }
            (_, Some(JsonValue::Bool(b))) if ty == ParamType::Bool => {
                Num::new(if *b { 1.0 } else { 0.0 })
            }
            (_, Some(_)) if ty == ParamType::Bool => {
                return Err(err(codes::PARAM, &value_path, "expected bool"));
            }
            (_, Some(value)) if ty == ParamType::Int => integer(value, &value_path, codes::PARAM)?,
            (_, Some(value)) => number(value, &value_path, codes::PARAM)?,
            (_, None) => return Err(err(codes::PARAM, &value_path, "missing value")),
        };
        conditions.push(Condition {
            param: name.to_owned(),
            op,
            value,
        });
    }
    Ok(conditions)
}

fn condition_op(name: &str, ty: ParamType, path: &str) -> Result<CondOp, GraphError> {
    let op = match name {
        "trigger" => CondOp::Trigger,
        "==" => CondOp::Eq,
        "!=" => CondOp::Ne,
        ">" => CondOp::Gt,
        "<" => CondOp::Lt,
        ">=" => CondOp::Ge,
        "<=" => CondOp::Le,
        _ => return Err(err(codes::PARAM, path, "unknown condition operator")),
    };
    let matches = match (ty, op) {
        (ParamType::Trigger, CondOp::Trigger) => true,
        (ParamType::Bool, CondOp::Eq | CondOp::Ne) => true,
        (ParamType::Float | ParamType::Int, _) => op != CondOp::Trigger,
        _ => false,
    };
    if !matches {
        return Err(err(
            codes::PARAM,
            path,
            "condition operator does not match the parameter type",
        ));
    }
    Ok(op)
}

fn blend_spaces(values: &[JsonValue], params: &[Param]) -> Result<Vec<BlendSpace>, GraphError> {
    GraphError::check_cap(
        values.len(),
        GRAPH_CAPS.max_blend_spaces,
        "/blendSpaces",
        "blend spaces",
    )?;
    let mut spaces: Vec<BlendSpace> = Vec::with_capacity(values.len());
    for (i, value) in values.iter().enumerate() {
        let path = format!("/blendSpaces/{i}");
        let fields = object(value, &path, &["name", "params", "points", "sync"])?;
        let name_path = pointer_child(&path, "name");
        let name = string(required(fields, "name", &path)?, &name_path)?;
        if name.is_empty() {
            return Err(err(
                codes::BLEND_SPACE,
                &name_path,
                "empty blend-space name",
            ));
        }
        if spaces.iter().any(|space| space.name == name) {
            return Err(err(
                codes::BLEND_SPACE,
                &name_path,
                "duplicate blend-space name",
            ));
        }
        let params_path = pointer_child(&path, "params");
        let names = array(required(fields, "params", &path)?, &params_path)?;
        if !(1..=2).contains(&names.len()) {
            return Err(err(
                codes::BLEND_SPACE,
                &params_path,
                "a blend space has one or two parameters",
            ));
        }
        let mut selected = Vec::with_capacity(names.len());
        for (j, value) in names.iter().enumerate() {
            let item_path = pointer_child(&params_path, &j.to_string());
            let param_name = string(value, &item_path)?;
            let param = params
                .iter()
                .find(|p| p.name == param_name)
                .ok_or_else(|| {
                    err(
                        codes::BLEND_SPACE,
                        &item_path,
                        "unknown blend-space parameter",
                    )
                })?;
            if !matches!(param.ty, ParamType::Float | ParamType::Int) {
                return Err(err(
                    codes::BLEND_SPACE,
                    &item_path,
                    "blend-space parameters must be numeric",
                ));
            }
            if selected.iter().any(|name: &String| name == param_name) {
                return Err(err(
                    codes::BLEND_SPACE,
                    &item_path,
                    "duplicate blend-space parameter",
                ));
            }
            selected.push(param_name.to_owned());
        }
        let points_path = pointer_child(&path, "points");
        let point_values = array(required(fields, "points", &path)?, &points_path)?;
        GraphError::check_cap(
            point_values.len(),
            GRAPH_CAPS.max_points_per_space,
            &points_path,
            "blend points",
        )?;
        if point_values.is_empty() {
            return Err(err(
                codes::BLEND_SPACE,
                &points_path,
                "a blend space needs at least one point",
            ));
        }
        let mut points = Vec::with_capacity(point_values.len());
        for (j, value) in point_values.iter().enumerate() {
            let item = format!("{points_path}/{j}");
            let fields = object(value, &item, &["clip", "at"])?;
            let clip_path = pointer_child(&item, "clip");
            let clip = string(required(fields, "clip", &item)?, &clip_path)?;
            if clip.is_empty() {
                return Err(err(codes::BLEND_SPACE, &clip_path, "empty clip name"));
            }
            let at_path = pointer_child(&item, "at");
            let coords = array(required(fields, "at", &item)?, &at_path)?;
            if coords.len() != selected.len() {
                return Err(err(
                    codes::BLEND_SPACE,
                    &at_path,
                    format!("expected {} coordinates", selected.len()),
                ));
            }
            let mut at = Vec::with_capacity(coords.len());
            for (k, value) in coords.iter().enumerate() {
                at.push(number(
                    value,
                    &pointer_child(&at_path, &k.to_string()),
                    codes::BLEND_SPACE,
                )?);
            }
            points.push(BlendPoint {
                clip: clip.to_owned(),
                at,
            });
        }
        let sync = match optional(fields, "sync") {
            None => false,
            Some(value) => boolean(value, &pointer_child(&path, "sync"))?,
        };
        spaces.push(BlendSpace {
            name: name.to_owned(),
            params: selected,
            points,
            sync,
        });
    }
    Ok(spaces)
}

fn err(code: &'static str, path: &str, message: impl Into<String>) -> GraphError {
    GraphError::new(code, path, message)
}

fn object<'a>(
    value: &'a JsonValue,
    path: &str,
    allowed: &[&str],
) -> Result<&'a [(String, JsonValue)], GraphError> {
    let JsonValue::Object(fields) = value else {
        return Err(err(codes::SCHEMA, path, "expected object"));
    };
    for (key, _) in fields {
        if !allowed.contains(&key.as_str()) {
            return Err(err(
                codes::SCHEMA,
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
        .find(|(field, _)| field == key)
        .map(|(_, value)| value)
}

fn required<'a>(
    fields: &'a [(String, JsonValue)],
    key: &str,
    path: &str,
) -> Result<&'a JsonValue, GraphError> {
    optional(fields, key)
        .ok_or_else(|| err(codes::SCHEMA, &pointer_child(path, key), "missing field"))
}

fn array<'a>(value: &'a JsonValue, path: &str) -> Result<&'a [JsonValue], GraphError> {
    match value {
        JsonValue::Array(values) => Ok(values),
        _ => Err(err(codes::SCHEMA, path, "expected array")),
    }
}

fn string<'a>(value: &'a JsonValue, path: &str) -> Result<&'a str, GraphError> {
    match value {
        JsonValue::String(value) => Ok(value),
        _ => Err(err(codes::SCHEMA, path, "expected string")),
    }
}

fn boolean(value: &JsonValue, path: &str) -> Result<bool, GraphError> {
    match value {
        JsonValue::Bool(value) => Ok(*value),
        _ => Err(err(codes::SCHEMA, path, "expected bool")),
    }
}

fn number(value: &JsonValue, path: &str, code: &'static str) -> Result<Num, GraphError> {
    let JsonValue::Number(value) = value else {
        return Err(err(code, path, "expected number"));
    };
    let num = Num::new(*value);
    if !num.rt.is_finite() {
        return Err(err(code, path, "number is not finite in f32"));
    }
    Ok(num)
}

fn integer(value: &JsonValue, path: &str, code: &'static str) -> Result<Num, GraphError> {
    let num = number(value, path, code)?;
    if num.doc.fract() != 0.0 {
        return Err(err(code, path, "expected an integer"));
    }
    Ok(num)
}
