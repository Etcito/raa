//! Reader of the section 67 trace program v1 used by the RAA raytrace pass.
//!
//! The crate starts with the program reader (`program`), the instance BVH (`bvh`) and the CPU
//! distance twin (`cpu`); the WGSL evaluator and the render pass land with their own cards.

pub mod bvh;
pub mod cpu;
mod program;

pub use cpu::{Hit, distance, material, trace_ray};
pub use program::{
    DATA, DATA_SHIFT, FORMAT, HEADER_WORDS, INSTANCE_COUNT, INSTANCE_WORDS, Instance,
    LIMITS, MAGIC, MATERIAL_WORDS, Material, NOT_FINITE, OPCODE, OPCODE_MASK, REFERENCE, SOURCE,
    SOURCE_WORDS, STACK, Source, TraceError, TraceProgram, VERSION, VERSION_ERROR, opcode, parse,
};
