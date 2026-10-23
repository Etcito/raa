//! Fur shells of the lit renderer (S12 `fur.shells`, FUR values).
//!
//! [`FurContributor::install`] registers the Fur draw class handler and the forward
//! contributor `fur.shells` (order 500). For every Fur draw it draws
//! N = clamp(round(fur.w), 1, 64) shells with the raa-shader `FurShell` variant of the draw's
//! material function, one draw per shell, the shell index given through the group-2
//! `RaaFurShellGpu` uniform of `raa_shader::PASS_FURSHELL_WGSL` (one 64-slot uniform buffer
//! read with a dynamic offset, written once at install). The pass shader supplies the shell
//! offset, gravity droop, strand mask and alpha-to-coverage; the Kajiya-Kay highlights come
//! from the `raa.plush` material function.

mod contributor;

pub use contributor::{
    FUR_NAME, FUR_ORDER, FUR_SHELLS_DEFAULT, FUR_SHELLS_MAX, FurContributor, SHELL_UNIFORM_SIZE,
    shell_count, shell_slot_stride, write_shell_slots,
};
