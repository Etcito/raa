//! Reader of the section 67 trace program v1 used by the RAA raytrace pass.
//!
//! The crate starts with the program reader (`program`), the instance BVH (`bvh`), the CPU
//! distance twin (`cpu`), the shared WGSL evaluator (`wgsl`) and the trace scene state with
//! its targets (`pass`); frame recording has its own card.

pub mod bvh;
pub mod cpu;
pub mod pass;
mod program;
pub mod wgsl;

pub use cpu::{Hit, distance, material, trace_ray};
pub use pass::{
    COLOR_FORMAT, COMPOSITE_NAME, COMPOSITE_ORDER, DEPTH_FORMAT, FRAME_BYTES, MAX_TARGET,
    TRACE_NAME, TRACE_ORDER, TraceFeature, TraceHandle, TraceScene, TraceStats, TraceTargets,
    target_size,
};
pub use program::{
    DATA, DATA_SHIFT, FORMAT, HEADER_WORDS, INSTANCE_COUNT, INSTANCE_WORDS, Instance, LIMITS,
    MAGIC, MATERIAL_WORDS, Material, NOT_FINITE, OPCODE, OPCODE_MASK, REFERENCE, SOURCE,
    SOURCE_WORDS, STACK, Source, TraceError, TraceProgram, VERSION, VERSION_ERROR, opcode, parse,
};
pub use wgsl::{PROGRAM_EVAL_WGSL, eval_module};
