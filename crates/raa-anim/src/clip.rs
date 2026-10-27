//! S17 clip documents: validation, track identity and S4 canonical writing.

use crate::{
    AnimError, JsonValue, Rig,
    json::{parse_clip, pointer_child},
};

/// S17 count limits. All are inclusive: a document exactly at a limit is
/// accepted and a document one above it fails with `clip.cap`, never truncated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClipCaps {
    pub max_tracks: usize,
    pub max_keys_per_track: usize,
    pub max_events: usize,
}

/// S17 `CLIP_CAPS` exported for clip and animation-graph consumers.
/// `lengthTicks` has no additional cap beyond its S17 integer/range rules.
pub const CLIP_CAPS: ClipCaps = ClipCaps {
    max_tracks: 256,
    max_keys_per_track: 4096,
    max_events: 1024,
};

/// S17 key channel; the value width is 3, or 4 for rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    Translation,
    Rotation,
    Scale,
}

impl Channel {
    pub fn width(self) -> usize {
        match self {
            Self::Rotation => 4,
            _ => 3,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Translation => "translation",
            Self::Rotation => "rotation",
            Self::Scale => "scale",
        }
    }
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "translation" => Self::Translation,
            "rotation" => Self::Rotation,
            "scale" => Self::Scale,
            _ => return None,
        })
    }
}

/// S17 key interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interp {
    Constant,
    Linear,
    Bezier,
}

impl Interp {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Constant => "CONSTANT",
            Self::Linear => "LINEAR",
            Self::Bezier => "BEZIER",
        }
    }
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "CONSTANT" => Self::Constant,
            "LINEAR" => Self::Linear,
            "BEZIER" => Self::Bezier,
            _ => return None,
        })
    }
}

/// S17 root-motion extraction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootMotion {
    None,
    Xz,
    Xyz,
}

impl RootMotion {
    fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Xz => "xz",
            Self::Xyz => "xyz",
        }
    }
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "none" => Self::None,
            "xz" => Self::Xz,
            "xyz" => Self::Xyz,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClipEvent {
    pub tick: u32,
    pub name: String,
}

/// One key. `v` and the handles keep their parsed source values so canonical
/// writing never reserializes them through runtime precision. Handle element 0
/// is the document dt offset (fractional and signed); sampling ignores it.
#[derive(Debug, Clone, PartialEq)]
pub struct ClipKey {
    pub tick: u32,
    pub interp: Interp,
    pub v: JsonValue,
    pub hl: Option<JsonValue>,
    pub hr: Option<JsonValue>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClipTrack {
    pub bone: String,
    pub channel: Channel,
    pub keys: Vec<ClipKey>,
}

/// An S17 clip bound to the rig it names.
#[derive(Debug, Clone, PartialEq)]
pub struct Clip {
    pub name: String,
    pub rig: String,
    pub length_ticks: u32,
    pub looping: bool,
    pub root_motion: RootMotion,
    pub tracks: Vec<ClipTrack>,
    pub events: Vec<ClipEvent>,
}

const TOP: [&str; 9] = [
    "format",
    "version",
    "name",
    "rig",
    "lengthTicks",
    "loop",
    "rootMotion",
    "tracks",
    "events",
];

impl Clip {
    /// Parse S17, requiring the document rig to be `rig`.
    pub fn load(text: &str, rig: &Rig) -> Result<Self, AnimError> {
        let value = parse_clip(text)?;
        let obj = object(&value, "", &TOP)?;
        if required(obj, "format", "")? != &JsonValue::String("raa.clip".into()) {
            return Err(error("clip.format", "/format", "expected raa.clip"));
        }
        if required(obj, "version", "")? != &JsonValue::Number(2.0) {
            return Err(error("clip.version", "/version", "expected version 2"));
        }
        let name = string(required(obj, "name", "")?, "/name")?.to_owned();
        let rig_name = string(required(obj, "rig", "")?, "/rig")?;
        let length_ticks = integer(required(obj, "lengthTicks", "")?, "/lengthTicks")?;
        if length_ticks == 0 {
            return Err(error(
                "clip.schema",
                "/lengthTicks",
                "lengthTicks must be positive",
            ));
        }
        let JsonValue::Array(track_values) = required(obj, "tracks", "")? else {
            return Err(error("clip.schema", "/tracks", "expected array"));
        };
        if track_values.len() > CLIP_CAPS.max_tracks {
            return Err(error("clip.cap", "/tracks", "too many tracks"));
        }
        let looping = match required(obj, "loop", "")? {
            JsonValue::Bool(looping) => *looping,
            _ => return Err(error("clip.schema", "/loop", "expected bool")),
        };
        let root_motion =
            RootMotion::parse(string(required(obj, "rootMotion", "")?, "/rootMotion")?)
                .ok_or_else(|| error("clip.schema", "/rootMotion", "unknown mode"))?;
        if rig_name != rig.name {
            return Err(error(
                "clip.rig_mismatch",
                "/rig",
                format!("clip rig `{rig_name}` does not match `{}`", rig.name),
            ));
        }
        let tracks = tracks(track_values, rig)?;
        let events = events(optional(obj, "events"))?;
        Ok(Self {
            name,
            rig: rig_name.to_owned(),
            length_ticks,
            looping,
            root_motion,
            tracks,
            events,
        })
    }

    /// Canonical S17 value in schema field order with the stored source values.
    pub fn to_json(&self) -> JsonValue {
        let tracks = self
            .tracks
            .iter()
            .map(|track| {
                let keys = track
                    .keys
                    .iter()
                    .map(crate::canon::clip_key)
                    .collect();
                JsonValue::Object(vec![
                    key("bone", JsonValue::String(track.bone.clone())),
                    key("channel", JsonValue::String(track.channel.name().into())),
                    key("keys", JsonValue::Array(keys)),
                ])
            })
            .collect();
        let events = self
            .events
            .iter()
            .map(|event| {
                JsonValue::Object(vec![
                    key("tick", JsonValue::Number(event.tick.into())),
                    key("name", JsonValue::String(event.name.clone())),
                ])
            })
            .collect();
        JsonValue::Object(vec![
            key("format", JsonValue::String("raa.clip".into())),
            key("version", JsonValue::Number(2.0)),
            key("name", JsonValue::String(self.name.clone())),
            key("rig", JsonValue::String(self.rig.clone())),
            key("lengthTicks", JsonValue::Number(self.length_ticks.into())),
            key("loop", JsonValue::Bool(self.looping)),
            key(
                "rootMotion",
                JsonValue::String(self.root_motion.name().into()),
            ),
            key("tracks", JsonValue::Array(tracks)),
            key("events", JsonValue::Array(events)),
        ])
    }

    /// S4 canonical text, including the final LF.
    pub fn to_canonical_json(&self) -> String {
        crate::canon::to_canonical_json(&self.to_json())
    }
}

fn tracks(values: &[JsonValue], rig: &Rig) -> Result<Vec<ClipTrack>, AnimError> {
    let mut tracks: Vec<ClipTrack> = Vec::with_capacity(values.len());
    for (i, value) in values.iter().enumerate() {
        let path = format!("/tracks/{i}");
        let obj = object(value, &path, &["bone", "channel", "keys"])?;
        let bone_path = pointer_child(&path, "bone");
        let bone = string(required(obj, "bone", &path)?, &bone_path)?;
        if !rig.bones.iter().any(|entry| entry.name == bone) {
            return Err(error(
                "clip.bone_unknown",
                &bone_path,
                "bone is not in the rig",
            ));
        }
        let channel_path = pointer_child(&path, "channel");
        let channel = Channel::parse(string(required(obj, "channel", &path)?, &channel_path)?)
            .ok_or_else(|| error("clip.channel", &channel_path, "unknown channel"))?;
        if tracks
            .iter()
            .any(|track| track.bone == bone && track.channel == channel)
        {
            return Err(error(
                "clip.duplicate_track",
                &path,
                "duplicate bone and channel",
            ));
        }
        let keys_path = pointer_child(&path, "keys");
        let JsonValue::Array(values) = required(obj, "keys", &path)? else {
            return Err(error("clip.schema", &keys_path, "expected array"));
        };
        if values.len() > CLIP_CAPS.max_keys_per_track {
            return Err(error("clip.cap", &keys_path, "too many keys"));
        }
        let mut keys: Vec<ClipKey> = Vec::with_capacity(values.len());
        for (j, value) in values.iter().enumerate() {
            let key_path = format!("{keys_path}/{j}");
            let parsed = parse_key(value, channel, &key_path)?;
            if let Some(previous) = keys.last()
                && parsed.tick <= previous.tick
            {
                return Err(error(
                    "clip.key_order",
                    &pointer_child(&key_path, "tick"),
                    "key ticks must strictly increase",
                ));
            }
            keys.push(parsed);
        }
        tracks.push(ClipTrack {
            bone: bone.to_owned(),
            channel,
            keys,
        });
    }
    Ok(tracks)
}

fn parse_key(value: &JsonValue, channel: Channel, path: &str) -> Result<ClipKey, AnimError> {
    let obj = object(value, path, &["tick", "v", "interp", "hl", "hr"])?;
    let tick = integer(required(obj, "tick", path)?, &pointer_child(path, "tick"))?;
    let v_path = pointer_child(path, "v");
    let v = required(obj, "v", path)?.clone();
    vector(&v, channel.width(), &v_path, "clip.channel")?;
    let interp_path = pointer_child(path, "interp");
    let interp = Interp::parse(string(required(obj, "interp", path)?, &interp_path)?)
        .ok_or_else(|| error("clip.interp", &interp_path, "unknown interpolation"))?;
    let hl = handle(obj, "hl", channel, &pointer_child(path, "hl"))?;
    let hr = handle(obj, "hr", channel, &pointer_child(path, "hr"))?;
    if interp != Interp::Bezier && (hl.is_some() || hr.is_some()) {
        return Err(error("clip.handles", path, "handles require BEZIER"));
    }
    Ok(ClipKey {
        tick,
        interp,
        v,
        hl,
        hr,
    })
}

fn handle(
    fields: &[(String, JsonValue)],
    name: &str,
    channel: Channel,
    path: &str,
) -> Result<Option<JsonValue>, AnimError> {
    let Some(value) = optional(fields, name) else {
        return Ok(None);
    };
    vector(value, channel.width() + 1, path, "clip.handles").map_err(|mut error| {
        error.code = "clip.handles";
        error
    })?;
    Ok(Some(value.clone()))
}

fn events(value: Option<&JsonValue>) -> Result<Vec<ClipEvent>, AnimError> {
    let values = match value {
        None => return Ok(Vec::new()),
        Some(JsonValue::Array(values)) => values,
        Some(_) => return Err(error("clip.schema", "/events", "expected array")),
    };
    if values.len() > CLIP_CAPS.max_events {
        return Err(error("clip.cap", "/events", "too many events"));
    }
    values
        .iter()
        .enumerate()
        .map(|(i, value)| {
            let path = format!("/events/{i}");
            let obj = object(value, &path, &["tick", "name"])?;
            Ok(ClipEvent {
                tick: integer(required(obj, "tick", &path)?, &pointer_child(&path, "tick"))?,
                name: string(required(obj, "name", &path)?, &pointer_child(&path, "name"))?
                    .to_owned(),
            })
        })
        .collect()
}

fn key(name: &str, value: JsonValue) -> (String, JsonValue) {
    (name.to_owned(), value)
}

fn error(code: &'static str, path: &str, message: impl Into<String>) -> AnimError {
    AnimError::new(code, path, message)
}

fn object<'a>(
    value: &'a JsonValue,
    path: &str,
    allowed: &[&str],
) -> Result<&'a [(String, JsonValue)], AnimError> {
    let JsonValue::Object(fields) = value else {
        return Err(error("clip.schema", path, "expected object"));
    };
    for (key, _) in fields {
        if !allowed.contains(&key.as_str()) {
            return Err(error(
                "clip.schema",
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
) -> Result<&'a JsonValue, AnimError> {
    optional(fields, key)
        .ok_or_else(|| error("clip.schema", &pointer_child(path, key), "missing field"))
}

fn string<'a>(value: &'a JsonValue, path: &str) -> Result<&'a str, AnimError> {
    match value {
        JsonValue::String(value) => Ok(value),
        _ => Err(error("clip.schema", path, "expected string")),
    }
}

fn integer(value: &JsonValue, path: &str) -> Result<u32, AnimError> {
    match value {
        JsonValue::Number(n) if n.fract() == 0.0 && *n >= 0.0 && *n < 4_294_967_296.0 => {
            Ok(*n as u32)
        }
        _ => Err(error(
            "clip.schema",
            path,
            "expected a non-negative integer",
        )),
    }
}

fn vector(
    value: &JsonValue,
    width: usize,
    path: &str,
    code: &'static str,
) -> Result<(), AnimError> {
    let JsonValue::Array(values) = value else {
        return Err(error("clip.schema", path, "expected array"));
    };
    if values.len() != width {
        return Err(error(code, path, "wrong channel width"));
    }
    for (i, value) in values.iter().enumerate() {
        if !matches!(value, JsonValue::Number(_)) {
            return Err(error(
                "clip.schema",
                &pointer_child(path, &i.to_string()),
                "expected number",
            ));
        }
    }
    Ok(())
}
