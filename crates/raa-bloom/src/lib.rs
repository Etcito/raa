//! Physically based bloom: descriptor, validation, mip rule and the pass shaders.

mod desc;
#[allow(dead_code)] // consumed by BloomFeature (the `bloom` pass) and tests/variants.rs
mod shaders;

pub use desc::{BloomDesc, MAX_MIPS, MIN_MIP_SIZE, mip_sizes};
