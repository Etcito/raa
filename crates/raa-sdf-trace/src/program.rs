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
pub const VERSION: u32 = 1;
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
    pub const UNION: u32 = 16;
    pub const SUBTRACT: u32 = 17;
    pub const INTERSECT: u32 = 18;

    pub fn is_primitive(opcode: u32) -> bool {
        (BOX..=ELLIPSOID).contains(&opcode)
    }

    /// Words after the primitive's 12-word local matrix, distance scale and material index.
    pub fn param_words(opcode: u32) -> Option<usize> {
        match opcode {
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
    fn at(code: &'static str, path: impl Into<String>, message: impl Into<String>) -> TraceError {
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

/// Parse and validate a section 67 trace program.
pub fn parse(bytes: &[u8]) -> Result<TraceProgram, TraceError> {
    let word_count = bytes.len() / 4;
    if !bytes.len().is_multiple_of(4) {
        refuse!(
            FORMAT,
            "bytes",
            "{} bytes is not a whole number of 32-bit words",
            bytes.len()
        );
    }
    let header: Vec<u32> = bytes.chunks_exact(4).take(HEADER_WORDS).map(word).collect();
    if header.len() != HEADER_WORDS {
        refuse!(FORMAT, "header", "the program ends inside the header");
    }
    if header[0] != MAGIC {
        refuse!(
            FORMAT,
            "header.magic",
            "the program does not start with SDTP"
        );
    }
    if header[1] != VERSION {
        refuse!(
            VERSION_ERROR,
            "header.version",
            "version {} (this reader accepts {VERSION})",
            header[1]
        );
    }
    let expected = total_words(&header)
        .ok_or_else(|| TraceError::at(LIMITS, "header", "the section lengths overflow"))?;
    if expected != word_count as u64 {
        let detail = if expected > word_count as u64 {
            "the program is truncated"
        } else {
            "the program has extra words"
        };
        refuse!(
            FORMAT,
            "header",
            "the header names {expected} words, the program has {word_count}: {detail}"
        );
    }

    let body: Vec<u32> = bytes
        .get(HEADER_WORDS * 4..)
        .unwrap_or(&[])
        .chunks_exact(4)
        .map(word)
        .collect();
    let (source_words, rest) = cut(&body, header[2] as usize * SOURCE_WORDS, "sources")?;
    let (code, rest) = cut(rest, header[4] as usize, "code")?;
    let (data_words, rest) = cut(rest, header[5] as usize, "data")?;
    let (material_words, rest) = cut(rest, header[6] as usize * MATERIAL_WORDS, "materials")?;
    let (instance_words, bend_words) = cut(rest, header[3] as usize * INSTANCE_WORDS, "instances")?;
    let mut instances: Vec<Instance> = instance_words.chunks_exact(INSTANCE_WORDS).map(read_instance).collect();
    if header[7] != 0 && header[7] as usize != instances.iter().filter(|i| i.deform_kind == 1).count() {
        refuse!(FORMAT, "bends", "the bend records do not match the bent instances");
    }
    let mut records = bend_words.chunks_exact(16);
    instances.iter_mut().filter(|i| i.deform_kind == 1).for_each(|i| i.bend = records.next().map(read_bend));

    let program = TraceProgram {
        sources: source_words
            .chunks_exact(SOURCE_WORDS)
            .map(read_source)
            .collect(),
        code: code.to_vec(),
        data: data_words.iter().map(|w| f32::from_bits(*w)).collect(),
        materials: material_words
            .chunks_exact(MATERIAL_WORDS)
            .map(read_material)
            .collect(),
        instances,
    };
    validate(&program)?;
    Ok(program)
}

fn word(chunk: &[u8]) -> u32 {
    u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
}

fn total_words(header: &[u32]) -> Option<u64> {
    let sources = u64::from(header[2]).checked_mul(SOURCE_WORDS as u64)?;
    let instances = u64::from(header[3]).checked_mul(INSTANCE_WORDS as u64)?;
    let materials = u64::from(header[6]).checked_mul(MATERIAL_WORDS as u64)?;
    let bends = u64::from(header[7]).checked_mul(16)?;
    (HEADER_WORDS as u64)
        .checked_add(sources)?
        .checked_add(instances)?
        .checked_add(u64::from(header[4]))?
        .checked_add(u64::from(header[5]))?
        .checked_add(materials)?
        .checked_add(bends)
}

fn cut<'a>(
    words: &'a [u32],
    count: usize,
    section: &str,
) -> Result<(&'a [u32], &'a [u32]), TraceError> {
    words
        .split_at_checked(count)
        .ok_or_else(|| TraceError::at(FORMAT, section, "the section does not fit the program"))
}

fn affine(w: &[u32], base: usize) -> [[f32; 4]; 3] {
    let mut out = [[0.0; 4]; 3];
    for (index, slot) in out.iter_mut().flatten().enumerate() {
        *slot = f32::from_bits(w[base + index]);
    }
    out
}

fn read_source(w: &[u32]) -> Source {
    let mut min = [0.0; 3];
    let mut max = [0.0; 3];
    for (axis, slot) in min.iter_mut().enumerate() {
        *slot = f32::from_bits(w[4 + axis]);
    }
    for (axis, slot) in max.iter_mut().enumerate() {
        *slot = f32::from_bits(w[7 + axis]);
    }
    Source {
        code_start: w[0],
        code_len: w[1],
        data_start: w[2],
        data_len: w[3],
        min,
        max,
        lipschitz: f32::from_bits(w[10]),
        flags: w[11],
    }
}

fn read_material(w: &[u32]) -> Material {
    Material {
        color: [
            f32::from_bits(w[0]),
            f32::from_bits(w[1]),
            f32::from_bits(w[2]),
        ],
        roughness: f32::from_bits(w[3]),
        metallic: f32::from_bits(w[4]),
    }
}

fn read_instance(w: &[u32]) -> Instance {
    Instance {
        world_from_local: affine(w, 0),
        local_from_world: affine(w, 12),
        dist_scale: f32::from_bits(w[24]),
        source: w[25],
        material: w[26] as i32,
        key: w[27],
        deform_kind: w[28],
        bend_angle: f32::from_bits(w[29]),
        bend_length: f32::from_bits(w[30]),
        axes: w[31],
        bend: None,
    }
}

fn read_bend(w: &[u32]) -> [f64; 8] {
    core::array::from_fn(|i| f64::from_bits(u64::from(w[i * 2]) | (u64::from(w[i * 2 + 1]) << 32)))
}

fn validate(program: &TraceProgram) -> Result<(), TraceError> {
    for (index, material) in program.materials.iter().enumerate() {
        let values = [
            material.color[0],
            material.color[1],
            material.color[2],
            material.roughness,
            material.metallic,
        ];
        if values.iter().any(|v| !v.is_finite()) {
            refuse!(
                NOT_FINITE,
                format!("material[{index}]"),
                "a material value is not finite"
            );
        }
    }
    validate_sources(program)?;
    if let Some(index) = program.data.iter().position(|v| !v.is_finite()) {
        refuse!(
            NOT_FINITE,
            format!("data[{index}]"),
            "the data word is not finite"
        );
    }
    validate_code(program)?;
    validate_instances(program)
}

fn validate_sources(program: &TraceProgram) -> Result<(), TraceError> {
    for (index, source) in program.sources.iter().enumerate() {
        let path = format!("source[{index}]");
        let code_end = source
            .code_start
            .checked_add(source.code_len)
            .ok_or_else(|| {
                TraceError::at(
                    LIMITS,
                    format!("{path}.code"),
                    "the instruction span overflows",
                )
            })?;
        if code_end as usize > program.code.len() {
            refuse!(
                SOURCE,
                format!("{path}.code"),
                "words {}..{code_end} are outside the {} code words",
                source.code_start,
                program.code.len()
            );
        }
        let data_end = source
            .data_start
            .checked_add(source.data_len)
            .ok_or_else(|| {
                TraceError::at(LIMITS, format!("{path}.data"), "the data span overflows")
            })?;
        if data_end as usize > program.data.len() {
            refuse!(
                SOURCE,
                format!("{path}.data"),
                "words {}..{data_end} are outside the {} data words",
                source.data_start,
                program.data.len()
            );
        }
        if source
            .min
            .iter()
            .chain(source.max.iter())
            .any(|v| !v.is_finite())
        {
            refuse!(
                NOT_FINITE,
                format!("{path}.bounds"),
                "a bound coordinate is not finite"
            );
        }
        if (0..3).any(|axis| source.min[axis] > source.max[axis]) {
            refuse!(
                SOURCE,
                format!("{path}.bounds"),
                "a minimum bound exceeds its maximum"
            );
        }
        if !source.lipschitz.is_finite() {
            refuse!(
                NOT_FINITE,
                format!("{path}.lipschitz"),
                "the Lipschitz bound is not finite"
            );
        }
        if source.lipschitz <= 0.0 {
            refuse!(
                SOURCE,
                format!("{path}.lipschitz"),
                "the Lipschitz bound must be positive"
            );
        }
    }
    Ok(())
}

fn validate_code(program: &TraceProgram) -> Result<(), TraceError> {
    for (source_index, source) in program.sources.iter().enumerate() {
        let (code, data) = (source.code(program), source.data(program));
        let mut depth: usize = 0;
        for (index, instruction) in code.iter().enumerate() {
            let path = format!("source[{source_index}].code[{index}]");
            let operation = instruction & OPCODE_MASK;
            let offset = (instruction >> DATA_SHIFT) as usize;
            let needed = opcode::data_words(operation).ok_or_else(|| {
                TraceError::at(OPCODE, &path, format!("unknown opcode {operation}"))
            })?;
            let end = offset.checked_add(needed).ok_or_else(|| {
                TraceError::at(LIMITS, &path, "the instruction data offset overflows")
            })?;
            if end > data.len() {
                refuse!(
                    DATA,
                    &path,
                    "data words {offset}..{end} are outside the source's {} data words",
                    data.len()
                );
            }
            if opcode::is_primitive(operation) {
                let material = data[offset + 13];
                if material < 0.0
                    || material.fract() != 0.0
                    || material as u64 >= program.materials.len() as u64
                {
                    refuse!(
                        REFERENCE,
                        &path,
                        "primitive material {material} is not a material index"
                    );
                }
                depth += 1;
            } else {
                if depth < 2 {
                    refuse!(
                        STACK,
                        &path,
                        "opcode {operation} needs two operands, the stack has {depth}"
                    );
                }
                depth -= 1;
            }
        }
        if depth != 1 {
            refuse!(
                STACK,
                format!("source[{source_index}].code"),
                "the program ends with {depth} values on the stack, expected 1"
            );
        }
    }
    Ok(())
}

fn validate_instances(program: &TraceProgram) -> Result<(), TraceError> {
    for (index, instance) in program.instances.iter().enumerate() {
        let path = format!("instance[{index}]");
        if instance.source as usize >= program.sources.len() {
            refuse!(
                REFERENCE,
                format!("{path}.source"),
                "source {} is outside the {} sources",
                instance.source,
                program.sources.len()
            );
        }
        let material = instance.material;
        let material_ok =
            material == -1 || (material >= 0 && (material as usize) < program.materials.len());
        if !material_ok {
            refuse!(
                REFERENCE,
                format!("{path}.material"),
                "material override {material} is not -1 or a material index"
            );
        }
        let finite = instance
            .world_from_local
            .iter()
            .chain(instance.local_from_world.iter())
            .flatten()
            .chain(
                [
                    instance.dist_scale,
                    instance.bend_angle,
                    instance.bend_length,
                ]
                .iter(),
            )
            .all(|v| v.is_finite());
        if !finite {
            refuse!(
                NOT_FINITE,
                &path,
                "a matrix, distance scale or bend value is not finite"
            );
        }
        if instance.dist_scale <= 0.0 {
            refuse!(
                FORMAT,
                format!("{path}.distScale"),
                "the distance scale must be positive"
            );
        }
        let (along, toward) = (instance.axes & 7, (instance.axes >> 3) & 7);
        match instance.deform_kind {
            0 => {}
            1 if instance.bend_length > 0.0 => {
                if along > 5 || toward == 6 || instance.axes >> 6 != 0 {
                    refuse!(
                        FORMAT,
                        format!("{path}.bendAxes"),
                        "the bend frame is incomplete: along must be one of the six signed axes, toward one of them or 7 (no aim)"
                    );
                }
                if let Some(b) = instance.bend && (!(0.0..=2.0).contains(&b[0]) || b[0].fract() != 0.0 || b.iter().any(|v| !v.is_finite()) || b[1] == 0.0 || !(b[4] > b[3])) {
                    refuse!(FORMAT, format!("{path}.bend"), "the bend record is incomplete");
                }
            }
            1 => refuse!(
                FORMAT,
                format!("{path}.bendLength"),
                "the bend length must be positive"
            ),
            kind => refuse!(
                FORMAT,
                format!("{path}.deformKind"),
                "unknown deform kind {kind}"
            ),
        }
    }
    Ok(())
}
