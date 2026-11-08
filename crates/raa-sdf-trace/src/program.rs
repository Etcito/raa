//! Reader of the section 67 trace program v1.
//!
//! The file is little-endian 32-bit words: header (8), sources (12 each), code (one word
//! each), data (one `f32` each), materials (8 `f32` each) and instances (32 each). An
//! instruction word is `opcode | data_offset << 8`, the offset being relative to the source's
//! own data span. Integer record slots (source, material override, key, deform kind, axes)
//! keep their 32-bit patterns instead of being read as floats.

use std::fmt;

/// First word of a trace program: the bytes `SDTP`, little endian.
pub const MAGIC: u32 = u32::from_le_bytes(*b"SDTP");
/// Format version this reader accepts.
pub const VERSION: u32 = 2;
pub const HEADER_WORDS: usize = 8;
pub const SOURCE_WORDS: usize = 12;
pub const INSTANCE_WORDS: usize = 32;
pub const MATERIAL_WORDS: usize = 8;
/// Opcode positions in an instruction word.
pub const DATA_SHIFT: u32 = 8;
pub const OPCODE_MASK: u32 = 0xff;

/// Stable refusal codes.
pub const FORMAT: &str = "trace.format";
pub const VERSION_ERROR: &str = "trace.version";
pub const LIMITS: &str = "trace.limits";
pub const SOURCE: &str = "trace.source";
pub const OPCODE: &str = "trace.opcode";
pub const DATA: &str = "trace.data";
pub const STACK: &str = "trace.stack";
pub const REFERENCE: &str = "trace.reference";
pub const NOT_FINITE: &str = "trace.not_finite";
/// Used by the upload path when an instance update names the wrong count.
pub const INSTANCE_COUNT: &str = "trace.instance_count";

/// Trace instruction opcodes (section 67).
pub mod opcode {
    pub const BOX: u32 = 1;
    pub const SPHERE: u32 = 2;
    pub const ROUND_BOX: u32 = 3;
    pub const CAPSULE: u32 = 4;
    pub const CYLINDER: u32 = 5;
    pub const CONE: u32 = 6;
    pub const TORUS: u32 = 7;
    pub const ELLIPSOID: u32 = 8;
    pub const SAMPLED: u32 = 9;
    pub const UNION: u32 = 16;
    pub const SUBTRACT: u32 = 17;
    pub const INTERSECT: u32 = 18;

    pub fn is_primitive(opcode: u32) -> bool {
        (BOX..=SAMPLED).contains(&opcode)
    }

    /// Words after the primitive's 12-word local matrix, distance scale and material index.
    pub fn param_words(opcode: u32) -> Option<usize> {
        match opcode {
            SAMPLED => Some(10),
            BOX | ROUND_BOX => Some(4),              // half xyz + round
            SPHERE => Some(1),                       // radius
            CAPSULE | CYLINDER | TORUS => Some(2),   // half height + radius / radii
            CONE | ELLIPSOID => Some(3),             // half height + radii / radii xyz
            UNION | SUBTRACT | INTERSECT => Some(1), // smooth combine factor
            _ => None,
        }
    }

    /// Data words an instruction needs, including its matrix and material slots.
    pub fn data_words(opcode: u32) -> Option<usize> {
        if is_primitive(opcode) {
            Some(14 + param_words(opcode)?)
        } else {
            param_words(opcode)
        }
    }
}

/// A refused trace program: a stable code, where it is and a human message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceError {
    pub code: &'static str,
    pub path: String,
    pub message: String,
}

impl TraceError {
    pub(crate) fn at(
        code: &'static str,
        path: impl Into<String>,
        message: impl Into<String>,
    ) -> TraceError {
        TraceError {
            code,
            path: path.into(),
            message: message.into(),
        }
    }

    pub fn code(&self) -> &'static str {
        self.code
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for TraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {}: {}", self.code, self.path, self.message)
    }
}

impl std::error::Error for TraceError {}

macro_rules! refuse {
    ($code:ident, $path:expr, $($arg:tt)*) => {
        return Err(TraceError::at($code, $path, format!($($arg)*)))
    };
}

/// One source program: its instruction and data spans, conservative local bounds and
/// Lipschitz bound. `flags` bit 0 marks a placeholder (unsupported subtree).
#[derive(Clone, Debug, PartialEq)]
pub struct Source {
    pub code_start: u32,
    pub code_len: u32,
    pub data_start: u32,
    pub data_len: u32,
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub lipschitz: f32,
    pub flags: u32,
}

impl Source {
    pub fn placeholder(&self) -> bool {
        self.flags & 1 != 0
    }

    /// This source's instruction words (empty if the record was not produced by [`parse`]).
    pub fn code<'a>(&self, program: &'a TraceProgram) -> &'a [u32] {
        span(&program.code, self.code_start, self.code_len)
    }

    /// This source's float data (empty if the record was not produced by [`parse`]).
    pub fn data<'a>(&self, program: &'a TraceProgram) -> &'a [f32] {
        span(&program.data, self.data_start, self.data_len)
    }
}

fn span<T>(words: &[T], start: u32, len: u32) -> &[T] {
    let start = start as usize;
    match start.checked_add(len as usize) {
        Some(end) => words.get(start..end).unwrap_or(&[]),
        None => &[],
    }
}

/// One placed instance. The two 3x4 affine blocks keep file order (three rows of four);
/// `material` is -1 for no override, and `axes` is the raw packed bend-frame word.
#[derive(Clone, Debug, PartialEq)]
pub struct Instance {
    pub world_from_local: [[f32; 4]; 3],
    pub local_from_world: [[f32; 4]; 3],
    pub dist_scale: f32,
    pub source: u32,
    pub material: i32,
    pub key: u32,
    pub deform_kind: u32,
    pub bend_angle: f32,
    pub bend_length: f32,
    pub axes: u32,
    pub bend: Option<[f64; 8]>,
}

/// Linear material of the program: colour, roughness and metallic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Material {
    pub color: [f32; 3],
    pub roughness: f32,
    pub metallic: f32,
}

/// A parsed section 67 program.
#[derive(Clone, Debug, PartialEq)]
pub struct TraceProgram {
    pub sources: Vec<Source>,
    pub code: Vec<u32>,
    pub data: Vec<f32>,
    pub materials: Vec<Material>,
    pub instances: Vec<Instance>,
}

impl TraceProgram {
    pub fn parse(bytes: &[u8]) -> Result<TraceProgram, TraceError> {
        parse(bytes)
    }
}

#[path = "program/read.rs"]
mod read;
use read::*;
#[path = "program/validate.rs"]
mod validation;
use validation::validate;
#[path = "program/instances.rs"]
mod instances;
pub(crate) use instances::validate_instances;

#[path = "program/parse.rs"]
mod parser;
pub use parser::parse;
