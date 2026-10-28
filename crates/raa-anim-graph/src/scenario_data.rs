//! Penguin scenario data: a stub rig, synthetic S17 clips for the fifteen S19
//! penguin names, the penguin graph and a 600-tick parameter timeline. The clips
//! are test motion, not authored media: each moves the root, bobs the hips,
//! swings the flippers and carries events so runtime mistakes are observable.
use crate::error::{GraphError, codes};
use crate::params::{ParamValue, Params};
use raa_anim::{Clip, JsonValue, Rig};

pub const PENGUIN_GRAPH: &str = include_str!("fixtures/penguin.animgraph.json");
pub const PENGUIN_TIMELINE: &str = include_str!("fixtures/penguin.timeline.json");
pub const PENGUIN_RIG: &str = r#"{"format":"raa.rig","version":1,"name":"penguin","bones":[
{"name":"root","parent":null,"rest":{"t":[0,0,0],"r":[0,0,0,1],"s":[1,1,1]}},
{"name":"hips","parent":"root","rest":{"t":[0,0.4,0],"r":[0,0,0,1],"s":[1,1,1]}},
{"name":"spine","parent":"hips","rest":{"t":[0,0.3,0],"r":[0,0,0,1],"s":[1,1,1]}},
{"name":"head","parent":"spine","rest":{"t":[0,0.35,0],"r":[0,0,0,1],"s":[1,1,1]}},
{"name":"flipperL","parent":"spine","rest":{"t":[0.25,0.2,0],"r":[0,0,0,1],"s":[1,1,1]}},
{"name":"flipperR","parent":"spine","rest":{"t":[-0.25,0.2,0],"r":[0,0,0,1],"s":[1,1,1]}},
{"name":"footL","parent":"hips","rest":{"t":[0.12,-0.4,0],"r":[0,0,0,1],"s":[1,1,1]}},
{"name":"footR","parent":"hips","rest":{"t":[-0.12,-0.4,0],"r":[0,0,0,1],"s":[1,1,1]}}],
"roles":{"chest":"spine","head":"head","hips":"hips","root":"root","upperArm.l":"flipperL","upperArm.r":"flipperR"},
"ikChains":[],"limits":[],"attach":{}}"#;

/// `(name, lengthTicks, loop, root travel over the clip, hips bob, flipper swing, events)`.
type Spec = (&'static str, u32, bool, [f32; 3], f32, f32, &'static [(u32, &'static str)]);
pub const PENGUIN_CLIPS: [Spec; 15] = [
    ("idle", 40, true, [0.0, 0.0, 0.0], 0.02, 0.05, &[(20, "breathe")]),
    ("waddle", 24, true, [0.0, 0.0, 0.9], 0.05, 0.2, &[(6, "step.l"), (18, "step.r")]),
    ("run", 16, true, [0.0, 0.0, 1.6], 0.08, 0.4, &[(4, "step.l"), (12, "step.r")]),
    ("jump", 16, false, [0.0, 0.0, 0.8], 0.5, 0.6, &[(2, "takeoff")]),
    ("fall", 12, true, [0.0, 0.0, 0.3], -0.1, 0.8, &[(6, "flail")]),
    ("dive", 12, false, [0.0, 0.0, 1.2], -0.3, 0.5, &[(8, "splash")]),
    ("bellySlide", 20, true, [0.0, 0.0, 2.0], -0.35, 0.1, &[(10, "slide")]),
    ("handPush", 20, true, [0.0, 0.0, 0.4], 0.03, 0.5, &[(10, "push")]),
    ("grab", 10, false, [0.0, 0.2, 0.1], 0.1, 0.7, &[(5, "grab")]),
    ("hang", 30, true, [0.0, 0.0, 0.0], 0.04, 0.3, &[(15, "sway")]),
    ("climbUp", 24, false, [0.0, 0.6, 0.3], 0.2, 0.6, &[(20, "ledge")]),
    ("emote.wave", 20, true, [0.0; 3], 0.0, 0.6, &[(10, "wave")]),
    ("emote.dance", 32, true, [0.0; 3], 0.06, 0.5, &[(8, "beat"), (24, "beat")]),
    ("emote.cheer", 16, true, [0.0; 3], 0.1, 0.9, &[(8, "cheer")]),
    ("emote.laugh", 18, true, [0.0; 3], 0.03, 0.2, &[(9, "laugh")]),
];

fn key(tick: u32, v: [f32; 3]) -> String {
    format!(r#"{{"tick":{tick},"v":[{},{},{}],"interp":"LINEAR"}}"#, v[0], v[1], v[2])
}
fn track(bone: &str, keys: &[String]) -> String {
    format!(r#"{{"bone":"{bone}","channel":"translation","keys":[{}]}}"#, keys.join(","))
}
/// The S17 text of one penguin clip spec.
pub fn clip_text(spec: &Spec) -> String {
    let (name, len, looping, travel, bob, swing, events) = *spec;
    let (half, quarter) = (len / 2, len / 4);
    let root = track("root", &[key(0, [0.0; 3]), key(len, travel)]);
    let hips = track("hips", &[key(0, [0.0, 0.4, 0.0]), key(half, [0.0, 0.4 + bob, 0.0]), key(len, [0.0, 0.4, 0.0])]);
    let flip = |bone: &str, x: f32| track(bone, &[key(0, [x, 0.2, 0.0]), key(quarter, [x, 0.2 + swing, 0.0]),
        key(3 * quarter, [x, 0.2 - swing, 0.0]), key(len, [x, 0.2, 0.0])]);
    let events: Vec<_> = events.iter().map(|(t, n)| format!(r#"{{"tick":{t},"name":"{n}"}}"#)).collect();
    format!(r#"{{"format":"raa.clip","version":2,"name":"{name}","rig":"penguin","lengthTicks":{len},"loop":{looping},
"rootMotion":"xz","tracks":[{root},{hips},{},{}],"events":[{}]}}"#,
        flip("flipperL", 0.25), flip("flipperR", -0.25), events.join(","))
}
pub fn penguin_rig() -> Rig { Rig::load(PENGUIN_RIG).expect("penguin rig") }
pub fn penguin_clips(rig: &Rig) -> Vec<Clip> {
    PENGUIN_CLIPS.iter().map(|s| Clip::load(&clip_text(s), rig).expect("penguin clip")).collect()
}

/// One timeline key: from `tick` on, `param` holds `value` (a trigger fires once).
#[derive(Debug, Clone, PartialEq)]
pub struct TimelineKey { pub tick: u32, pub param: String, pub value: ParamValue }
#[derive(Debug, Clone, PartialEq)]
pub struct Timeline { pub ticks: u32, pub keys: Vec<TimelineKey> }

fn bad(path: &str, message: &str) -> GraphError { GraphError::new(codes::SCHEMA, path, message) }
fn field<'a>(fields: &'a [(String, JsonValue)], name: &str) -> Option<&'a JsonValue> {
    fields.iter().find(|f| f.0 == name).map(|f| &f.1)
}
fn uint(v: Option<&JsonValue>, path: &str) -> Result<u32, GraphError> {
    match v {
        Some(JsonValue::Number(n)) if *n >= 0.0 && *n <= u32::MAX as _ && n.fract() == 0.0 => Ok(*n as u32),
        _ => Err(bad(path, "expected a non-negative integer")),
    }
}
impl Timeline {
    /// Parse and check a timeline against declared params: keys are ordered,
    /// inside `[0, ticks)` and typed like their parameter.
    pub fn load(text: &str, decl: &Params) -> Result<Self, GraphError> {
        let doc = raa_anim::parse(text).map_err(|e| bad("", &format!("{e:?}")))?;
        let JsonValue::Object(top) = &doc else { return Err(bad("", "expected an object")) };
        let ticks = uint(field(top, "ticks"), "/ticks")?;
        let Some(JsonValue::Array(items)) = field(top, "keys") else { return Err(bad("/keys", "expected an array")) };
        let mut keys = Vec::with_capacity(items.len());
        for (i, item) in items.iter().enumerate() {
            let path = format!("/keys/{i}");
            let JsonValue::Object(f) = item else { return Err(bad(&path, "expected an object")) };
            let tick = uint(field(f, "tick"), &path)?;
            if tick >= ticks || keys.last().is_some_and(|k: &TimelineKey| k.tick > tick) {
                return Err(bad(&path, "tick out of order or outside the timeline"));
            }
            let Some(JsonValue::String(param)) = field(f, "param") else { return Err(bad(&path, "expected param")) };
            let index = decl.index_of(param).ok_or_else(|| GraphError::new(codes::PARAM, &path, "unknown parameter"))?;
            let value = match (decl.value(index), field(f, "value")) {
                (Some(ParamValue::Float(_)), Some(JsonValue::Number(n))) => ParamValue::Float(*n as f32),
                (Some(ParamValue::Int(_)), Some(JsonValue::Number(n))) if n.fract() == 0.0 => ParamValue::Int(*n as i32),
                (Some(ParamValue::Bool(_)), Some(JsonValue::Bool(b))) => ParamValue::Bool(*b),
                (Some(ParamValue::Trigger(_)), Some(JsonValue::Bool(true))) => ParamValue::Trigger(true),
                _ => return Err(GraphError::new(codes::PARAM, &path, "value does not match the parameter type")),
            };
            keys.push(TimelineKey { tick, param: param.clone(), value });
        }
        Ok(Self { ticks, keys })
    }
    /// Apply every key of `tick` to `params` (held values persist; triggers fire).
    pub fn apply(&self, tick: u32, params: &mut Params) -> Result<(), GraphError> {
        for k in self.keys.iter().filter(|k| k.tick == tick) {
            match k.value { ParamValue::Trigger(_) => params.fire(&k.param)?, v => params.set(&k.param, v)? }
        }
        Ok(())
    }
}
