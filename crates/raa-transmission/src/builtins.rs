//! The transmissive built-in material functions (`raa.ice`, `raa.glass`, `raa.slime`,
//! `raa.water`).
//!
//! Every row has flags 0 and class Opaque: a desc with blend 'opaque' draws the function
//! opaque (the transmission-off variant), blend 'transmissive' makes it Transparent.

use raa_matfn::{BuiltinFn, DrawClass, ParamKind, ParamSpec};

/// The transmissive built-ins, pushed into `raa_matfn::ExtraBuiltins` by raa-js.
pub const TRANSMISSIVE_BUILTINS: &[BuiltinFn] = &[
    BuiltinFn {
        name: "raa.ice",
        wgsl: include_str!("wgsl/ice.wgsl"),
        params: &[
            ParamSpec {
                name: "tint",
                kind: ParamKind::Color,
                default: [0.7, 0.9, 0.98, 0.0],
                slot: (0, 0),
            },
            ParamSpec {
                name: "absorption",
                kind: ParamKind::Float,
                default: [1.5, 0.0, 0.0, 0.0],
                slot: (1, 0),
            },
            ParamSpec {
                name: "cloudiness",
                kind: ParamKind::Float,
                default: [0.3, 0.0, 0.0, 0.0],
                slot: (1, 1),
            },
            ParamSpec {
                name: "crackScale",
                kind: ParamKind::Float,
                default: [0.25, 0.0, 0.0, 0.0],
                slot: (1, 2),
            },
        ],
        flags: 0,
        class: DrawClass::Opaque,
    },
    BuiltinFn {
        name: "raa.glass",
        wgsl: include_str!("wgsl/glass.wgsl"),
        params: &[
            ParamSpec {
                name: "tint",
                kind: ParamKind::Color,
                default: [0.95, 0.98, 0.97, 0.0],
                slot: (0, 0),
            },
            ParamSpec {
                name: "roughness",
                kind: ParamKind::Float,
                default: [0.03, 0.0, 0.0, 0.0],
                slot: (1, 0),
            },
        ],
        flags: 0,
        class: DrawClass::Opaque,
    },
    BuiltinFn {
        name: "raa.slime",
        wgsl: include_str!("wgsl/slime.wgsl"),
        params: &[
            ParamSpec {
                name: "color",
                kind: ParamKind::Color,
                default: [0.45, 0.95, 0.15, 0.0],
                slot: (0, 0),
            },
            ParamSpec {
                name: "absorption",
                kind: ParamKind::Float,
                default: [3.0, 0.0, 0.0, 0.0],
                slot: (1, 0),
            },
            ParamSpec {
                name: "glossiness",
                kind: ParamKind::Float,
                default: [0.85, 0.0, 0.0, 0.0],
                slot: (1, 1),
            },
            ParamSpec {
                name: "bubbleScale",
                kind: ParamKind::Float,
                default: [0.06, 0.0, 0.0, 0.0],
                slot: (1, 2),
            },
        ],
        flags: 0,
        class: DrawClass::Opaque,
    },
    BuiltinFn {
        name: "raa.water",
        wgsl: include_str!("wgsl/water.wgsl"),
        params: &[
            ParamSpec {
                name: "deepColor",
                kind: ParamKind::Color,
                default: [0.05, 0.45, 0.55, 0.0],
                slot: (0, 0),
            },
            ParamSpec {
                name: "shallowColor",
                kind: ParamKind::Color,
                default: [0.55, 0.92, 0.88, 0.0],
                slot: (1, 0),
            },
            ParamSpec {
                name: "clarity",
                kind: ParamKind::Float,
                default: [0.7, 0.0, 0.0, 0.0],
                slot: (2, 0),
            },
        ],
        flags: 0,
        class: DrawClass::Opaque,
    },
];

// slot.rs compiles the same file for its own tests; its module is private to slot.
#[cfg(test)]
#[allow(dead_code, clippy::duplicate_mod)]
#[path = "../../raa-lit/src/slots.rs"]
mod lit_slots;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slot::screen_provider;
    use raa_lit::{FLAG_TRANSMISSIVE, FLAG_TRANSPARENT};
    use raa_shader::{Composer, MaterialFnRegistry, PassKind, VariantKey};

    fn composer() -> Composer {
        let mut composer = Composer::new();
        for p in lit_slots::default_slot_providers()
            .into_iter()
            .chain([screen_provider()])
        {
            if let Err(e) = composer.add_slot(p) {
                panic!("{e}");
            }
        }
        composer
    }

    #[test]
    fn transmissive_builtins_validate_forward_and_transparent() {
        let composer = composer();
        let mut reg = MaterialFnRegistry::new();
        let mut count = 0;
        for b in TRANSMISSIVE_BUILTINS {
            let material_fn = reg
                .register_builtin(b.name, b.wgsl)
                .unwrap_or_else(|e| panic!("{}: {e}", b.name));
            for (pass, flags) in [
                (PassKind::Forward, b.flags),
                (
                    PassKind::Transparent,
                    b.flags | FLAG_TRANSPARENT | FLAG_TRANSMISSIVE,
                ),
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
                        .unwrap_or_else(|e| panic!("{} {key:?}: {e}", b.name));
                    if let Err(e) = Composer::validate(&wgsl) {
                        panic!("{} {key:?}: {e}", b.name);
                    }
                    count += 1;
                }
            }
        }
        assert_eq!(TRANSMISSIVE_BUILTINS.len(), 4);
        assert_eq!(count, TRANSMISSIVE_BUILTINS.len() * 4);
    }

    #[test]
    fn rows_are_well_formed() {
        for b in TRANSMISSIVE_BUILTINS {
            assert_eq!((b.flags, b.class), (0, DrawClass::Opaque), "{}", b.name);
            let mut used = [[false; 4]; 8];
            for p in b.params {
                let (i, c) = p.slot;
                assert!(
                    i < 8 && c + p.kind.components() <= 4,
                    "{} {}",
                    b.name,
                    p.name
                );
                for u in &mut used[i][c..c + p.kind.components()] {
                    assert!(!*u, "{} {} overlaps", b.name, p.name);
                    *u = true;
                }
            }
            let helpers = b
                .wgsl
                .lines()
                .filter_map(|l| l.strip_prefix("fn "))
                .filter_map(|l| l.split('(').next());
            let prefix = format!("{}_", b.name.trim_start_matches("raa."));
            for f in helpers {
                assert!(f == "material_main" || f.starts_with(&prefix), "{f}");
            }
        }
    }
}
