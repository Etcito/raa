//! The liquid built-in material functions: `raa.liquid.water` and `raa.liquid.slime`.
//!
//! Both rows carry `FLAG_VERTEX_FN | FLAG_TRANSMISSIVE` and class [`DrawClass::Transparent`]:
//! the Gerstner `material_vertex` displaces the flat XZ grid, the fragment stage refracts and
//! absorbs through the `Transmission` slot, and a desc with blend 'transmissive' draws them
//! through OIT (the scene contract's Transmissive class).
//!
//! Every row's WGSL is the shared vertex stage, the shared fragment helpers and its own
//! `material_main`, concatenated in that order.
//!
//! The wave packing is the one of `src/waves.rs`: `p[0..4]` is wave `i` as
//! `(amplitude m, wavelength m, direction angle rad, steepness)`, `p[4]` the four speeds in
//! m/s. The water speeds are the deep-water dispersion `sqrt(9.81 L / 2 pi)`; the slime
//! scales the amplitudes by 0.4 and the speeds by 0.25 of the water defaults.

use raa_lit::{FLAG_TRANSMISSIVE, FLAG_VERTEX_FN};
use raa_matfn::{BuiltinFn, DrawClass, ParamKind, ParamSpec};

/// `material_main` of `raa.liquid.water` over the shared vertex and surface WGSL.
const WATER_WGSL: &str = concat!(
    include_str!("wgsl/liquid_vertex.wgsl"),
    "\n",
    include_str!("wgsl/liquid_surface.wgsl"),
    "\n",
    include_str!("wgsl/water.wgsl"),
);

/// `material_main` of `raa.liquid.slime` over the shared vertex and surface WGSL.
const SLIME_WGSL: &str = concat!(
    include_str!("wgsl/liquid_vertex.wgsl"),
    "\n",
    include_str!("wgsl/liquid_surface.wgsl"),
    "\n",
    include_str!("wgsl/slime.wgsl"),
);

/// The built-in liquid material functions, installed by the liquid feature.
pub const LIQUID_BUILTINS: &[BuiltinFn] = &[
    BuiltinFn {
        name: "raa.liquid.water",
        wgsl: WATER_WGSL,
        params: &WATER_PARAMS,
        flags: FLAG_VERTEX_FN | FLAG_TRANSMISSIVE,
        class: DrawClass::Transparent,
    },
    BuiltinFn {
        name: "raa.liquid.slime",
        wgsl: SLIME_WGSL,
        params: &SLIME_PARAMS,
        flags: FLAG_VERTEX_FN | FLAG_TRANSMISSIVE,
        class: DrawClass::Transparent,
    },
];

/// Water parameters: four Gerstner waves, their speeds and the water look.
static WATER_PARAMS: [ParamSpec; 12] = [
    wave("wave0", [0.035, 1.6, 0.3, 0.6], 0),
    wave("wave1", [0.02, 0.9, 1.4, 0.5], 1),
    wave("wave2", [0.012, 0.55, -0.7, 0.4], 2),
    wave("wave3", [0.006, 0.31, 2.4, 0.35], 3),
    speed(4, [1.5805, 1.1854, 0.9267, 0.6957]),
    ParamSpec {
        name: "deepColor",
        kind: ParamKind::Color,
        default: [0.05, 0.45, 0.55, 0.0],
        slot: (5, 0),
    },
    ParamSpec {
        name: "clarity",
        kind: ParamKind::Float,
        default: [0.7, 0.0, 0.0, 0.0],
        slot: (5, 3),
    },
    ParamSpec {
        name: "shallowColor",
        kind: ParamKind::Color,
        default: [0.55, 0.92, 0.88, 0.0],
        slot: (6, 0),
    },
    ParamSpec {
        name: "foamWidth",
        kind: ParamKind::Float,
        default: [0.12, 0.0, 0.0, 0.0],
        slot: (6, 3),
    },
    ParamSpec {
        name: "foamAmount",
        kind: ParamKind::Float,
        default: [0.6, 0.0, 0.0, 0.0],
        slot: (7, 0),
    },
    ParamSpec {
        name: "detailStrength",
        kind: ParamKind::Float,
        default: [0.15, 0.0, 0.0, 0.0],
        slot: (7, 1),
    },
    ParamSpec {
        name: "detailScale",
        kind: ParamKind::Float,
        default: [3.0, 0.0, 0.0, 0.0],
        slot: (7, 2),
    },
];

/// Slime parameters: the same waves at 0.4 amplitude and 0.25 speed, and the slime look.
static SLIME_PARAMS: [ParamSpec; 13] = [
    wave("wave0", [0.014, 1.6, 0.3, 0.6], 0),
    wave("wave1", [0.008, 0.9, 1.4, 0.5], 1),
    wave("wave2", [0.0048, 0.55, -0.7, 0.4], 2),
    wave("wave3", [0.0024, 0.31, 2.4, 0.35], 3),
    speed(4, [0.39513, 0.29635, 0.23167, 0.17393]),
    ParamSpec {
        name: "color",
        kind: ParamKind::Color,
        default: [0.42, 1.0, 0.12, 0.0],
        slot: (5, 0),
    },
    ParamSpec {
        name: "absorption",
        kind: ParamKind::Float,
        default: [5.0, 0.0, 0.0, 0.0],
        slot: (5, 3),
    },
    ParamSpec {
        name: "glossiness",
        kind: ParamKind::Float,
        default: [0.9, 0.0, 0.0, 0.0],
        slot: (6, 0),
    },
    ParamSpec {
        name: "bubbleScale",
        kind: ParamKind::Float,
        default: [0.05, 0.0, 0.0, 0.0],
        slot: (6, 1),
    },
    ParamSpec {
        name: "foamWidth",
        kind: ParamKind::Float,
        default: [0.06, 0.0, 0.0, 0.0],
        slot: (6, 3),
    },
    ParamSpec {
        name: "foamAmount",
        kind: ParamKind::Float,
        default: [0.25, 0.0, 0.0, 0.0],
        slot: (7, 0),
    },
    ParamSpec {
        name: "detailStrength",
        kind: ParamKind::Float,
        default: [0.08, 0.0, 0.0, 0.0],
        slot: (7, 1),
    },
    ParamSpec {
        name: "detailScale",
        kind: ParamKind::Float,
        default: [2.0, 0.0, 0.0, 0.0],
        slot: (7, 2),
    },
];

/// One wave parameter row at `p[slot]`.
const fn wave(name: &'static str, default: [f32; 4], slot: usize) -> ParamSpec {
    ParamSpec {
        name,
        kind: ParamKind::Vec4,
        default,
        slot: (slot, 0),
    }
}

/// The four wave speeds at `p[slot]`.
const fn speed(slot: usize, default: [f32; 4]) -> ParamSpec {
    ParamSpec {
        name: "speeds",
        kind: ParamKind::Vec4,
        default,
        slot: (slot, 0),
    }
}

// raa-lit's default slot providers are crate-private; the test compiles their source file
// (raa-lit is only read). raa-transmission is a dev-dependency, built like its own
// tests/variants.rs builds the `screen` provider (the constructor is crate-private).
#[cfg(test)]
#[allow(dead_code, clippy::duplicate_mod)]
#[path = "../../raa-lit/src/slots.rs"]
mod lit_slots;

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    use raa_lit::{FLAG_TRANSMISSIVE, FLAG_TRANSPARENT, FLAG_VERTEX_FN};
    use raa_shader::{Composer, MaterialFnRegistry, PassKind, SlotId, SlotProvider, VariantKey};

    /// The `screen` provider as `TransmissionFeature::install` adds it.
    fn screen_provider() -> SlotProvider {
        SlotProvider {
            slot: SlotId::Transmission,
            priority: 10,
            provider: raa_transmission::SCREEN_PROVIDER,
            wgsl: Cow::Borrowed(include_str!(
                "../../raa-transmission/src/wgsl/transmission.wgsl"
            )),
        }
    }

    /// Default + layered + screen: the slot set of the M6 engine.
    fn composer() -> Composer {
        let mut composer = Composer::new();
        for p in lit_slots::default_slot_providers()
            .into_iter()
            .chain(raa_brdf::providers())
            .chain([screen_provider()])
        {
            if let Err(e) = composer.add_slot(p) {
                panic!("{e}");
            }
        }
        composer
    }

    fn registry() -> MaterialFnRegistry {
        let mut reg = MaterialFnRegistry::new();
        for b in LIQUID_BUILTINS {
            if let Err(e) = reg.register_builtin(b.name, b.wgsl) {
                panic!("{}: {e}", b.name);
            }
        }
        reg
    }

    /// Every `fn` defined in `src` (the row WGSL and its shared helpers).
    fn defined_fns(src: &str) -> Vec<&str> {
        src.lines()
            .filter_map(|l| l.trim_start().strip_prefix("fn "))
            .map(|rest| rest.split('(').next().unwrap_or(rest).trim())
            .collect()
    }

    /// The helper prefix of the row `name`: `raa.liquid.water` -> `water`.
    fn stem(name: &str) -> &str {
        name.rsplit('.').next().unwrap_or(name)
    }

    #[test]
    fn rows_are_well_formed() {
        assert_eq!(LIQUID_BUILTINS.len(), 2);
        for b in LIQUID_BUILTINS {
            assert!(b.name.starts_with("raa.liquid."), "{}: name", b.name);
            assert_eq!(
                b.flags & (FLAG_VERTEX_FN | FLAG_TRANSMISSIVE),
                FLAG_VERTEX_FN | FLAG_TRANSMISSIVE,
                "{}: flags",
                b.name
            );
            assert_eq!(b.flags & FLAG_TRANSPARENT, 0, "{}: flags", b.name);
            assert_eq!(b.class, DrawClass::Transparent, "{}: class", b.name);

            let helper_prefix = format!("{}_", stem(b.name));
            for f in defined_fns(b.wgsl) {
                if f == "material_main" || f == "material_vertex" {
                    continue;
                }
                assert!(
                    f.starts_with(&helper_prefix) || f.starts_with("liquid_"),
                    "{}: helper `{f}` is not prefixed {helper_prefix} or liquid_",
                    b.name
                );
            }

            for (i, p) in b.params.iter().enumerate() {
                let width = p.kind.components();
                assert!(p.slot.0 < 8, "{} {}: slot index", b.name, p.name);
                assert!(p.slot.1 + width <= 4, "{} {}: component", b.name, p.name);
                for q in &b.params[i + 1..] {
                    assert_ne!(p.name, q.name, "{}: duplicate name", b.name);
                    assert_ne!(
                        p.slot, q.slot,
                        "{} {} {}: same slot",
                        b.name, p.name, q.name
                    );
                    let overlap = p.slot.0 == q.slot.0
                        && p.slot.1 < q.slot.1 + q.kind.components()
                        && q.slot.1 < p.slot.1 + width;
                    assert!(!overlap, "{} {} {}: overlapping", b.name, p.name, q.name);
                }
            }
        }
    }

    #[test]
    fn water_and_slime_compose_and_validate_in_every_pass() {
        let composer = composer();
        let reg = registry();
        let mut count = 0;
        for b in LIQUID_BUILTINS {
            let material_fn = reg.id_of(b.name).expect("registered");
            for pass in PassKind::ALL {
                for msaa in [1, 4] {
                    let key = VariantKey {
                        material_fn,
                        pass,
                        msaa,
                        flags: b.flags,
                    };
                    let wgsl = composer
                        .compose(&reg, &key)
                        .unwrap_or_else(|e| panic!("{} {key:?}: {e}", b.name));
                    if pass == PassKind::Transparent {
                        assert!(wgsl.contains("fn raa_screen_refract("), "{key:?}");
                    }
                    if let Err(e) = Composer::validate(&wgsl) {
                        panic!("{} {key:?}: {e}", b.name);
                    }
                    count += 1;
                }
            }
        }
        assert_eq!(count, LIQUID_BUILTINS.len() * PassKind::ALL.len() * 2);
    }
}
