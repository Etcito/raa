//! S19 canonical text: S4 numbers and whitespace with keys in S19 order.
//! Arrays keep file order, which is significant for transitions and layers.

use crate::model::{CondOp, Graph, Num, ParamType};
use raa_anim::{JsonValue, RootMotion, to_canonical_json};

fn s(text: &str) -> JsonValue {
    JsonValue::String(text.into())
}

fn num(n: Num) -> JsonValue {
    JsonValue::Number(n.doc)
}

fn strings(items: &[String]) -> JsonValue {
    JsonValue::Array(items.iter().map(|i| s(i)).collect())
}

fn obj(fields: Vec<(&str, JsonValue)>) -> JsonValue {
    JsonValue::Object(fields.into_iter().map(|(k, v)| (k.into(), v)).collect())
}

/// Bool params and their conditions are stored as 0/1 and written as JSON booleans.
fn typed(ty: ParamType, n: Num) -> JsonValue {
    match ty {
        ParamType::Bool => JsonValue::Bool(n.rt != 0.0),
        _ => num(n),
    }
}

fn op_name(op: CondOp) -> &'static str {
    match op {
        CondOp::Trigger => "trigger",
        CondOp::Eq => "==",
        CondOp::Ne => "!=",
        CondOp::Gt => ">",
        CondOp::Lt => "<",
        CondOp::Ge => ">=",
        CondOp::Le => "<=",
    }
}

fn mode_name(mode: RootMotion) -> &'static str {
    match mode {
        RootMotion::None => "none",
        RootMotion::Xz => "xz",
        RootMotion::Xyz => "xyz",
    }
}

/// The S19 document as a JSON value with fields in canonical order.
pub fn graph_value(graph: &Graph) -> JsonValue {
    let param_ty = |name: &str| {
        let found = graph.params.iter().find(|p| p.name == name);
        found.map_or(ParamType::Float, |p| p.ty)
    };
    let params = graph.params.iter().map(|p| {
        let ty = match p.ty {
            ParamType::Float => "float",
            ParamType::Int => "int",
            ParamType::Bool => "bool",
            ParamType::Trigger => "trigger",
        };
        let mut fields = vec![("name", s(&p.name)), ("type", s(ty))];
        if p.ty != ParamType::Trigger {
            fields.push(("default", typed(p.ty, p.default)));
        }
        obj(fields)
    });
    let spaces = graph.blend_spaces.iter().map(|b| {
        let points = b.points.iter().map(|p| {
            let at = JsonValue::Array(p.at.iter().map(|n| num(*n)).collect());
            obj(vec![("clip", s(&p.clip)), ("at", at)])
        });
        obj(vec![
            ("name", s(&b.name)),
            ("params", strings(&b.params)),
            ("points", JsonValue::Array(points.collect())),
            ("sync", JsonValue::Bool(b.sync)),
        ])
    });
    let layers = graph.layers.iter().map(|l| {
        let mask = l.mask.as_deref().map_or(JsonValue::Null, strings);
        obj(vec![
            ("name", s(&l.name)),
            ("machine", s(&l.machine)),
            ("mask", mask),
            ("weight", num(l.weight)),
            ("additive", JsonValue::Bool(l.additive)),
        ])
    });
    let machines = graph.machines.iter().map(|m| {
        let states = m.states.iter().map(|st| {
            let speed = st.speed_param.as_deref().map_or(JsonValue::Null, s);
            obj(vec![
                ("name", s(&st.name)),
                ("motion", s(&st.motion)),
                ("loop", JsonValue::Bool(st.looping)),
                ("speedParam", speed),
            ])
        });
        let transitions = m.transitions.iter().map(|t| {
            let when = t.when.iter().map(|c| {
                let mut fields = vec![("param", s(&c.param)), ("op", s(op_name(c.op)))];
                if c.op != CondOp::Trigger {
                    fields.push(("value", typed(param_ty(&c.param), c.value)));
                }
                obj(fields)
            });
            let exit = t.exit_tick.map_or(JsonValue::Null, |x| JsonValue::Number(x.into()));
            obj(vec![
                ("from", s(&t.from)),
                ("to", s(&t.to)),
                ("when", JsonValue::Array(when.collect())),
                ("blendTicks", JsonValue::Number(t.blend_ticks.into())),
                ("exitTick", exit),
                ("priority", JsonValue::Number(t.priority.into())),
            ])
        });
        obj(vec![
            ("name", s(&m.name)),
            ("entry", s(&m.entry)),
            ("states", JsonValue::Array(states.collect())),
            ("transitions", JsonValue::Array(transitions.collect())),
        ])
    });
    let root = &graph.root_motion;
    obj(vec![
        ("format", s("raa.animgraph")),
        ("version", JsonValue::Number(1.into())),
        ("name", s(&graph.name)),
        ("rig", s(&graph.rig)),
        ("params", JsonValue::Array(params.collect())),
        ("blendSpaces", JsonValue::Array(spaces.collect())),
        ("layers", JsonValue::Array(layers.collect())),
        ("machines", JsonValue::Array(machines.collect())),
        ("rootMotion", obj(vec![("bone", s(&root.bone)), ("mode", s(mode_name(root.mode)))])),
    ])
}

/// Canonical S19 text (S4 rules, final LF).
pub fn to_canonical(graph: &Graph) -> String {
    to_canonical_json(&graph_value(graph))
}

impl Graph {
    /// Canonical S19 text of this graph.
    pub fn canonical(&self) -> String {
        to_canonical(self)
    }
}
