//! WGSL sources shared by the liquid material functions.
//!
//! [`VERTEX_WGSL`] is the Gerstner wave and ripple-ring `material_vertex` every liquid
//! function uses; [`SURFACE_WGSL`] holds the fragment helpers of their `material_main`.
//! The per-liquid functions concatenate both files with their own `material_main`.

/// Shared vertex stage: S11 `material_vertex`, the Gerstner waves and the impact ripples.
#[allow(
    dead_code,
    reason = "used by the liquid material functions of the next task"
)]
pub(crate) const VERTEX_WGSL: &str = include_str!("wgsl/liquid_vertex.wgsl");

/// Shared fragment helpers: crest, detail normals, shore foam and foam application.
#[allow(
    dead_code,
    reason = "used by the liquid material functions of the next task"
)]
pub(crate) const SURFACE_WGSL: &str = include_str!("wgsl/liquid_surface.wgsl");

// raa-lit's default slot providers are crate-private (used by `LitRenderer::new`); the
// test compiles their source file (raa-lit is only read).
#[cfg(test)]
#[allow(dead_code, clippy::duplicate_mod)]
#[path = "../../raa-lit/src/slots.rs"]
mod lit_slots;

#[cfg(test)]
mod tests {
    use super::*;
    use raa_lit::{FLAG_TRANSMISSIVE, FLAG_TRANSPARENT};
    use raa_shader::{Composer, MaterialFnRegistry, PassKind, VariantKey};

    /// A throwaway fragment function calling every shared helper.
    const THROWAWAY_MAIN: &str = "\
fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    let crest = liquid_waves_at(in.record.p, in.local_position.xz, in.time);
    let detail = liquid_detail_normal(o.normal, in.tangent, in.position, in.time, 0.35, 4.0);
    let shore = liquid_shore(in.frag_coord, 0.25);
    let foam = liquid_foam(shore, crest, in.position, in.time, 0.6);
    o.normal = detail;
    return liquid_apply_foam(o, foam);
}
";

    fn composer() -> Composer {
        let mut composer = Composer::new();
        for p in lit_slots::default_slot_providers() {
            if let Err(e) = composer.add_slot(p) {
                panic!("{e}");
            }
        }
        composer
    }

    #[test]
    fn liquid_shaders_compose_and_validate_in_every_pass() {
        let composer = composer();
        let mut reg = MaterialFnRegistry::new();
        let source = format!("{VERTEX_WGSL}\n{SURFACE_WGSL}\n{THROWAWAY_MAIN}");
        let material_fn = reg
            .register_builtin("raa.liquid.test", &source)
            .unwrap_or_else(|e| panic!("{e}"));
        let mut count = 0;
        for (pass, flags) in [
            (PassKind::Forward, 0),
            (PassKind::Prepass, 0),
            (PassKind::Shadow, 0),
            (PassKind::Transparent, FLAG_TRANSPARENT | FLAG_TRANSMISSIVE),
        ] {
            for msaa in [1, 4] {
                let key = VariantKey {
                    material_fn,
                    pass,
                    msaa,
                    flags,
                };
                let wgsl = composer
                    .compose(&reg, &key)
                    .unwrap_or_else(|e| panic!("{pass:?} msaa {msaa}: {e}"));
                if let Err(e) = Composer::validate(&wgsl) {
                    panic!("{pass:?} msaa {msaa}: {e}");
                }
                count += 1;
            }
        }
        assert_eq!(count, 8);
    }
}
