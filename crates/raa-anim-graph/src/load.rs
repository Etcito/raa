//! Machine, layer and root-motion decoding, [`Graph`] assembly in file order and [`Graph::load`].
//! Only in-document references are checked here; rig and clip names are left to the reference pass.

use crate::decode::{decode_conditions, decode_document};
use crate::error::{GraphError, codes};
use crate::model::{GRAPH_CAPS, Graph, GraphRootMotion, Layer, Machine, Num, Param};
use crate::model::{ParamType::Float, ParamType::Int, State, Transition, WILDCARD};
use raa_anim::json::{JsonValue, parse, pointer_child};
use raa_anim::{Clip, Rig, RootMotion};
use raa_det::Tick;

type Fields = [(String, JsonValue)];
type Out<T> = Result<T, GraphError>;
type List<'a> = (&'a [JsonValue], String);

impl Graph {
    /// Decode an S19 document and bind it to `rig` and `clips`.
    pub fn load(text: &str, rig: &Rig, clips: &[Clip]) -> Out<Self> {
        resolve_references(decode_graph(text)?, rig, clips)
    }
}

/// Decode every S19 section in file order, leaving rig and clip references unresolved.
pub fn decode_graph(text: &str) -> Out<Graph> {
    let parts = decode_document(text)?;
    let JsonValue::Object(top) = parse(text)? else {
        return Err(GraphError::new(codes::SCHEMA, "", "expected object"));
    };
    let machines = list(&top, "machines", "", GRAPH_CAPS.max_machines)?;
    let machines = machines_of(machines, &parts.params)?;
    let layers = layers(list(&top, "layers", "", GRAPH_CAPS.max_layers)?, &machines)?;
    let root = field(&top, "rootMotion", "")?;
    let fields = object(root, "/rootMotion", &["bone", "mode"])?;
    let mode = match name(fields, "mode", "/rootMotion")?.as_str() {
        "none" => RootMotion::None,
        "xz" => RootMotion::Xz,
        "xyz" => RootMotion::Xyz,
        _ => return fail(codes::SCHEMA, "/rootMotion", "mode", "unknown mode"),
    };
    let bone = name(fields, "bone", "/rootMotion")?;
    Ok(Graph {
        name: parts.name,
        rig: parts.rig,
        params: parts.params,
        blend_spaces: parts.blend_spaces,
        layers,
        machines,
        root_motion: GraphRootMotion { bone, mode },
    })
}

/// Motion and bone names resolve in the reference pass; until then a graph is refused.
fn resolve_references(_graph: Graph, _rig: &Rig, _clips: &[Clip]) -> Out<Graph> {
    Err(GraphError::new(codes::SCHEMA, "", "references unresolved"))
}

fn machines_of(values: List, params: &[Param]) -> Out<Vec<Machine>> {
    let keys = ["name", "entry", "states", "transitions"];
    items(values, &keys, |fields, path, machines: &[Machine]| {
        let cap = GRAPH_CAPS.max_states_per_machine;
        let states = states(list(fields, "states", path, cap)?, params)?;
        let entry = name(fields, "entry", path)?;
        if !states.iter().any(|s| s.name == entry) {
            return fail(codes::STATE_UNKNOWN, path, "entry", "unknown entry state");
        }
        let cap = GRAPH_CAPS.max_transitions_per_machine;
        Ok(Machine {
            name: unique(fields, path, |n| machines.iter().any(|m| m.name == *n))?,
            entry,
            transitions: transitions(list(fields, "transitions", path, cap)?, &states, params)?,
            states,
        })
    })
}

fn states(values: List, params: &[Param]) -> Out<Vec<State>> {
    let keys = ["name", "motion", "loop", "speedParam"];
    let numeric = |p: &&Param| matches!(p.ty, Float | Int);
    items(values, &keys, |fields, path, states: &[State]| {
        let taken = |n: &String| n == WILDCARD || states.iter().any(|s| s.name == *n);
        let name = unique(fields, path, taken)?;
        let speed_param = match field(fields, "speedParam", path)? {
            JsonValue::Null => None,
            JsonValue::String(s) if params.iter().filter(numeric).any(|p| p.name == *s) => {
                Some(s.clone())
            }
            _ => return fail(codes::PARAM, path, "speedParam", "expected a numeric param"),
        };
        Ok(State {
            name,
            motion: self::name(fields, "motion", path)?,
            looping: flag(fields, "loop", path)?,
            speed_param,
        })
    })
}

fn transitions(values: List, states: &[State], params: &[Param]) -> Out<Vec<Transition>> {
    let keys = ["from", "to", "when", "blendTicks", "exitTick", "priority"];
    let known = |name: &str| states.iter().any(|s| s.name == name);
    items(values, &keys, |fields, path, _: &[Transition]| {
        let from = name(fields, "from", path)?;
        if from != WILDCARD && !known(&from) {
            return fail(codes::STATE_UNKNOWN, path, "from", "unknown source state");
        }
        let to = name(fields, "to", path)?;
        if !known(&to) {
            return fail(codes::STATE_UNKNOWN, path, "to", "unknown target state");
        }
        let when_path = pointer_child(path, "when");
        let priority = match field(fields, "priority", path)? {
            JsonValue::Number(v) if *v == (*v as i32).into() => *v as i32,
            _ => return fail(codes::SCHEMA, path, "priority", "expected an i32 integer"),
        };
        Ok(Transition {
            from,
            to,
            when: decode_conditions(field(fields, "when", path)?, &when_path, params)?,
            blend_ticks: tick(fields, "blendTicks", path)?,
            exit_tick: match field(fields, "exitTick", path)? {
                JsonValue::Null => None,
                _ => Some(tick(fields, "exitTick", path)?),
            },
            priority,
        })
    })
}

fn layers(values: List, machines: &[Machine]) -> Out<Vec<Layer>> {
    let keys = ["name", "machine", "mask", "weight", "additive"];
    items(values, &keys, |fields, path, _: &[Layer]| {
        let machine = name(fields, "machine", path)?;
        if !machines.iter().any(|m| m.name == machine) {
            return fail(codes::STATE_UNKNOWN, path, "machine", "unknown machine");
        }
        let weight = match field(fields, "weight", path)? {
            JsonValue::Number(v) if *v >= 0.0 && *v <= 1.0 => Num::new(*v),
            _ => return fail(codes::SCHEMA, path, "weight", "expected a number in [0, 1]"),
        };
        Ok(Layer {
            name: name(fields, "name", path)?,
            machine,
            mask: mask(fields, path)?,
            weight,
            additive: flag(fields, "additive", path)?,
        })
    })
}

/// `null` selects every bone; otherwise a list of distinct bones.
fn mask(fields: &Fields, path: &str) -> Out<Option<Vec<String>>> {
    if field(fields, "mask", path)? == &JsonValue::Null {
        return Ok(None);
    }
    let (values, path) = list(fields, "mask", path, GRAPH_CAPS.max_mask_bones)?;
    let mut bones: Vec<String> = Vec::with_capacity(values.len());
    for (i, value) in values.iter().enumerate() {
        match value {
            JsonValue::String(b) if !b.is_empty() && !bones.contains(b) => bones.push(b.clone()),
            _ => return fail(codes::MASK, &path, &i.to_string(), "bad bone name"),
        }
    }
    Ok(Some(bones))
}

/// Decode each object of a list in file order; `decode` sees the items before it.
fn items<T>(
    (values, path): List,
    keys: &[&str],
    mut decode: impl FnMut(&Fields, &str, &[T]) -> Out<T>,
) -> Out<Vec<T>> {
    let mut out = Vec::with_capacity(values.len());
    for (i, value) in values.iter().enumerate() {
        let item = format!("{path}/{i}");
        let item = decode(object(value, &item, keys)?, &item, &out)?;
        out.push(item);
    }
    Ok(out)
}

/// The required `name` field, refused when empty or `taken` (duplicate or reserved).
fn unique(fields: &Fields, path: &str, taken: impl Fn(&String) -> bool) -> Out<String> {
    let name = name(fields, "name", path)?;
    match taken(&name) {
        true => fail(codes::SCHEMA, path, "name", "duplicate name"),
        false => Ok(name),
    }
}

fn fail<T>(code: &'static str, path: &str, key: &str, message: &str) -> Out<T> {
    Err(GraphError::new(code, pointer_child(path, key), message))
}

fn object<'a>(value: &'a JsonValue, path: &str, keys: &[&str]) -> Out<&'a Fields> {
    let JsonValue::Object(fields) = value else {
        return Err(GraphError::new(codes::SCHEMA, path, "expected object"));
    };
    match fields.iter().find(|(key, _)| !keys.contains(&key.as_str())) {
        Some((key, _)) => fail(codes::SCHEMA, path, key, "unknown field"),
        None => Ok(fields),
    }
}

fn field<'a>(fields: &'a Fields, key: &str, path: &str) -> Out<&'a JsonValue> {
    match fields.iter().find(|(field, _)| field == key) {
        Some((_, value)) => Ok(value),
        None => fail(codes::SCHEMA, path, key, "missing field"),
    }
}

/// A required array of at most `cap` items.
fn list<'a>(fields: &'a Fields, key: &str, path: &str, cap: usize) -> Out<List<'a>> {
    let JsonValue::Array(values) = field(fields, key, path)? else {
        return fail(codes::SCHEMA, path, key, "expected array");
    };
    let item = pointer_child(path, key);
    GraphError::check_cap(values.len(), cap, &item, key)?;
    Ok((values, item))
}

/// A required non-empty string.
fn name(fields: &Fields, key: &str, path: &str) -> Out<String> {
    match field(fields, key, path)? {
        JsonValue::String(name) if !name.is_empty() => Ok(name.clone()),
        _ => fail(codes::SCHEMA, path, key, "expected a name"),
    }
}

fn flag(fields: &Fields, key: &str, path: &str) -> Out<bool> {
    match field(fields, key, path)? {
        JsonValue::Bool(value) => Ok(*value),
        _ => fail(codes::SCHEMA, path, key, "expected bool"),
    }
}

fn tick(fields: &Fields, key: &str, path: &str) -> Out<Tick> {
    match field(fields, key, path)? {
        JsonValue::Number(v) if *v == (*v as Tick).into() => Ok(*v as Tick),
        _ => fail(codes::SCHEMA, path, key, "expected a tick"),
    }
}
