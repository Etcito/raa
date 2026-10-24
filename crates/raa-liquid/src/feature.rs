//! `install`: registers every [`LIQUID_BUILTINS`] row as a built-in material function of a
//! lit renderer.
//!
//! The install is atomic like `TransmissionFeature::install`: every row is first registered
//! on a clone of the registry, so a refusal (`material.fn_duplicate` on a second install)
//! leaves the lit renderer as it was.

use raa_core::RaaError;
use raa_lit::LitRenderer;
use raa_shader::{MaterialFnRegistry, ShaderError};

use crate::builtins::LIQUID_BUILTINS;

/// A shader refusal as a [`RaaError`] with the same code and message (raa-js `api_error`).
fn shader_error(e: ShaderError) -> RaaError {
    RaaError::FrameGraph {
        code: e.code,
        message: e.message,
    }
}

/// Registers every [`LIQUID_BUILTINS`] row on `fns`; probes a clone first so a refusal
/// registers nothing.
fn register_all(fns: &mut MaterialFnRegistry) -> Result<(), RaaError> {
    let mut probe = fns.clone();
    for b in LIQUID_BUILTINS {
        probe
            .register_builtin(b.name, b.wgsl)
            .map_err(shader_error)?;
    }
    for b in LIQUID_BUILTINS {
        fns.register_builtin(b.name, b.wgsl).map_err(shader_error)?;
    }
    Ok(())
}

/// Registers `raa.liquid.water`, `raa.liquid.slime` and `raa.liquid.mud` as built-in
/// material functions of `lit`.
///
/// The transmissive liquids refract through the `Transmission` slot and draw through the
/// Transparent class handler, which raa-transmission and raa-oit install; this function only
/// registers the functions.
///
/// Refusal (nothing registered): the registry refusal with its code and message
/// (`material.fn_duplicate` when a liquid name is already registered, so a second install).
pub fn install(lit: &mut LitRenderer) -> Result<(), RaaError> {
    register_all(&mut lit.fns_mut())
}

// `LitRenderer::new` needs a GPU and a pipeline; the install logic is the registry part,
// tested here without a GPU on a bare registry.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_all_adds_every_row() {
        let mut fns = MaterialFnRegistry::new();
        let before = fns.len();
        register_all(&mut fns).unwrap();
        assert_eq!(fns.len(), before + LIQUID_BUILTINS.len());
        for b in LIQUID_BUILTINS {
            let id = fns.id_of(b.name).expect("registered");
            let entry = fns.get(id).expect("entry");
            assert!(entry.builtin, "{}", b.name);
            assert!(entry.has_vertex, "{}: material_vertex", b.name);
        }
    }

    #[test]
    fn second_install_refuses_fn_duplicate_and_registers_nothing() {
        let mut fns = MaterialFnRegistry::new();
        register_all(&mut fns).unwrap();
        let count = fns.len();
        let err = register_all(&mut fns).err().unwrap();
        assert_eq!(err.code(), "material.fn_duplicate");
        assert!(!err.to_string().is_empty());
        assert_eq!(fns.len(), count);
    }

    #[test]
    fn one_registered_name_refuses_and_registers_no_other_row() {
        let mut fns = MaterialFnRegistry::new();
        let last = &LIQUID_BUILTINS[LIQUID_BUILTINS.len() - 1];
        fns.register_builtin(last.name, last.wgsl).unwrap();
        let count = fns.len();
        let err = register_all(&mut fns).err().unwrap();
        assert_eq!(err.code(), "material.fn_duplicate");
        assert_eq!(fns.len(), count);
        for b in &LIQUID_BUILTINS[..LIQUID_BUILTINS.len() - 1] {
            assert!(fns.id_of(b.name).is_none(), "{}", b.name);
        }
    }
}
