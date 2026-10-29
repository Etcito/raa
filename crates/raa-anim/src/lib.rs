//! Animation brick for RAA: strict JSON reading and the shared refusal type.

use std::fmt;
pub mod canon;
pub mod json;
pub use canon::to_canonical_json;
pub use json::{JsonValue, parse};

/// A refusal with a stable code (`json.*`, `rig.*`, `clip.*`) and path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnimError {
    pub code: &'static str,
    pub path: String,
    pub message: String,
}

impl AnimError {
    pub fn new(code: &'static str, path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code,
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for AnimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {}: {}", self.code, self.path, self.message)
    }
}

impl std::error::Error for AnimError {}

pub mod rig;
pub use rig::{Bone, MAX_BONES, RestTransform, Rig};

pub mod pose;
pub use pose::{LocalPose, Mat4, Pose};

pub mod clip;
pub use clip::{
    CLIP_CAPS, Channel, Clip, ClipCaps, ClipEvent, ClipKey, ClipTrack, Interp, RootMotion,
};

pub mod sample;
pub use sample::address_tick;
