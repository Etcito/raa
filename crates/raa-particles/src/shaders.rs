//! WGSL sources and module builders of the particle passes.
//!
//! [`SIM_WGSL`] is the standalone `particles.sim` compute module (it does not include the
//! prelude): its private group 0 binds the 48 B particle states (read_write), the
//! `EmitterGpu` table, the slot -> emitter table, the `(first serial, count)` spawn
//! commands, the per-frame claim counters and the `StepGpu` uniform (dynamic offset).
//! Entry `cs_sim`, workgroup 64, one thread per slot.
//!
//! [`lit_module`] builds the head of the particle draw shaders: PRELUDE + LIB + the winner
//! of every slot in [`SlotId::ALL`] order, followed by the pass body, so the draws shade
//! with the installed lighting slots.

use raa_core::RaaError;
use raa_shader::{Composer, LIB_WGSL, PRELUDE_WGSL, SlotId};

/// The `particles.sim` compute module.
pub(crate) const SIM_WGSL: &str = include_str!("wgsl/sim.wgsl");

/// The `particles.sim` compute module (entry `cs_sim`); exposed for the variant tests.
pub fn sim_module_wgsl() -> &'static str {
    SIM_WGSL
}

/// PRELUDE + LIB + the winner WGSL of every slot in [`SlotId::ALL`] order + `body`.
/// Refuses `shader.slot_missing` when a slot has no provider. Not validated.
#[allow(
    dead_code,
    reason = "used by the opaque and blend module builders of the next task"
)]
pub(crate) fn lit_module(composer: &Composer, body: &str) -> Result<String, RaaError> {
    let mut out = String::new();
    section(&mut out, "prelude", PRELUDE_WGSL);
    section(&mut out, "lib", LIB_WGSL);
    for slot in SlotId::ALL {
        let p = composer.winner(slot).ok_or_else(|| RaaError::FrameGraph {
            code: "shader.slot_missing",
            message: format!(
                "slot {slot:?} (`{}`) has no provider for the particle draw shaders",
                slot.fn_name()
            ),
        })?;
        let title = format!(
            "slot {:?}, provider {:?}, priority {}",
            p.slot, p.provider, p.priority
        );
        section(&mut out, &title, &p.wgsl);
    }
    section(&mut out, "particles pass body", body);
    Ok(out)
}

fn section(out: &mut String, title: &str, wgsl: &str) {
    out.push_str("\n// ---- ");
    out.push_str(title);
    out.push_str(" ----\n");
    out.push_str(wgsl);
    if !wgsl.ends_with('\n') {
        out.push('\n');
    }
}

// raa-lit's default slot providers are crate-private (used by `LitRenderer::new`); the
// test compiles their source file (raa-lit is only read).
#[cfg(test)]
#[allow(dead_code, clippy::duplicate_mod)]
#[path = "../../raa-lit/src/slots.rs"]
mod lit_slots;

#[cfg(test)]
mod tests {
    use super::*;

    fn default_composer() -> Composer {
        let mut composer = Composer::new();
        for p in lit_slots::default_slot_providers() {
            if let Err(e) = composer.add_slot(p) {
                panic!("{e}");
            }
        }
        composer
    }

    #[test]
    fn sim_module_validates() {
        if let Err(e) = Composer::validate(sim_module_wgsl()) {
            panic!("{e}");
        }
        assert!(sim_module_wgsl().contains("fn cs_sim("));
        assert!(sim_module_wgsl().contains("@workgroup_size(64)"));
    }

    #[test]
    fn lit_module_with_default_slots_and_empty_body_validates() {
        let wgsl = lit_module(&default_composer(), "").unwrap_or_else(|e| panic!("{e}"));
        for slot in SlotId::ALL {
            assert!(
                wgsl.contains(&format!("fn {}(", slot.fn_name())),
                "{slot:?}"
            );
        }
        if let Err(e) = Composer::validate(&wgsl) {
            panic!("{e}");
        }
    }

    #[test]
    fn lit_module_refuses_a_missing_slot() {
        let mut composer = Composer::new();
        for p in lit_slots::default_slot_providers() {
            if p.slot == SlotId::Ambient {
                continue;
            }
            if let Err(e) = composer.add_slot(p) {
                panic!("{e}");
            }
        }
        let Err(e) = lit_module(&composer, "") else {
            panic!("a missing Ambient slot must be refused");
        };
        assert_eq!(e.code(), "shader.slot_missing");
        assert!(e.message().contains("Ambient"), "{}", e.message());
    }
}
