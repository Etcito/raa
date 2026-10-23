//! Physically based bloom: descriptor, validation, mip rule, the pass shaders and the
//! `bloom` feature pass (S12, order 120) with its handle.

mod desc;
mod feature;
mod shaders;

pub use desc::{BloomDesc, MAX_MIPS, MIN_MIP_SIZE, mip_sizes};
pub use feature::{BloomFeature, BloomHandle};
