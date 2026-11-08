//! Reader of the section 67 trace program v1 used by the RAA raytrace pass.
//!
//! The crate starts with the program reader (`program`), the instance BVH (`bvh`), the CPU
//! distance twin (`cpu`), the shared WGSL evaluator and frozen volume ABI (`wgsl`), and the
//! trace scene state with its volume, targets and frame passes (`pass`).

pub mod bvh;
pub mod cpu;
pub mod pass;
mod program;
mod sampled;
pub mod wgsl;

pub use cpu::{Hit, distance, material, trace_ray};
pub use pass::{
    COLOR_FORMAT, COMPOSITE_NAME, COMPOSITE_ORDER, DEPTH_FORMAT, FRAME_BYTES, TRACE_NAME,
    TRACE_ORDER, TraceFeature, TraceHandle, TraceScene, TraceStats, TraceTargets, TraceVolume,
    VOLUME_AXIS_ALIGN, VOLUME_MAX_AXIS, VOLUME_MIPS, VOLUME_ROW_ALIGN, VOLUME_UNIFORM_BYTES,
    VolumeLayout, VolumeMip, target_size,
};
pub use program::{
    DATA, DATA_SHIFT, FORMAT, HEADER_WORDS, INSTANCE_COUNT, INSTANCE_WORDS, Instance, LIMITS,
    MAGIC, MATERIAL_WORDS, Material, NOT_FINITE, OPCODE, OPCODE_MASK, REFERENCE, SOURCE,
    SOURCE_WORDS, STACK, Source, TraceError, TraceProgram, VERSION, VERSION_ERROR, opcode, parse,
};
pub use wgsl::{PROGRAM_EVAL_WGSL, eval_module};
