//! Physically based bloom: descriptor, validation and mip rule.

mod desc;

pub use desc::{BloomDesc, MAX_MIPS, MIN_MIP_SIZE, mip_sizes};
