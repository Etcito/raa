//! The built-in material function table.

use crate::{BuiltinFn, DrawClass, ParamKind, ParamSpec};
use raa_shader::{MaterialFnRegistry, ShaderError};

/// Every built-in material function, in registration order.
pub const BUILTINS: &[BuiltinFn] = &[
    BuiltinFn {
        name: "raa.inflatable",
        wgsl: include_str!("wgsl/inflatable.wgsl"),
        params: &[
            ParamSpec {
                name: "color",
                kind: ParamKind::Color,
                default: [0.95, 0.22, 0.18, 0.0],
                slot: (0, 0),
            },
            ParamSpec {
                name: "stripeColor",
                kind: ParamKind::Color,
                default: [0.98, 0.96, 0.92, 0.0],
                slot: (1, 0),
            },
            ParamSpec {
                name: "stripeWidth",
                kind: ParamKind::Float,
                default: [0.6, 0.0, 0.0, 0.0],
                slot: (2, 0),
            },
            ParamSpec {
                name: "seamDepth",
                kind: ParamKind::Float,
                default: [0.1, 0.0, 0.0, 0.0],
                slot: (2, 1),
            },
            ParamSpec {
                name: "seamCount",
                kind: ParamKind::Float,
                default: [8.0, 0.0, 0.0, 0.0],
                slot: (2, 2),
            },
        ],
        flags: 0,
        class: DrawClass::Opaque,
    },
    BuiltinFn {
        name: "raa.candy",
        wgsl: include_str!("wgsl/candy.wgsl"),
        params: &[
            ParamSpec {
                name: "color",
                kind: ParamKind::Color,
                default: [0.97, 0.95, 0.92, 0.0],
                slot: (0, 0),
            },
            ParamSpec {
                name: "stripeColor",
                kind: ParamKind::Color,
                default: [0.86, 0.08, 0.12, 0.0],
                slot: (1, 0),
            },
            ParamSpec {
                name: "stripeCount",
                kind: ParamKind::Float,
                default: [6.0, 0.0, 0.0, 0.0],
                slot: (2, 0),
            },
            ParamSpec {
                name: "swirl",
                kind: ParamKind::Float,
                default: [2.0, 0.0, 0.0, 0.0],
                slot: (2, 1),
            },
        ],
        flags: 0,
        class: DrawClass::Opaque,
    },
    BuiltinFn {
        name: "raa.plush",
        wgsl: include_str!("wgsl/plush.wgsl"),
        params: &[
            ParamSpec {
                name: "color",
                kind: ParamKind::Color,
                default: [0.92, 0.9, 0.86, 0.0],
                slot: (0, 0),
            },
            ParamSpec {
                name: "tipColor",
                kind: ParamKind::Color,
                default: [1.0, 0.99, 0.97, 0.0],
                slot: (1, 0),
            },
            ParamSpec {
                name: "sheenColor",
                kind: ParamKind::Color,
                default: [0.95, 0.93, 0.9, 0.0],
                slot: (2, 0),
            },
            ParamSpec {
                name: "furLength",
                kind: ParamKind::Float,
                default: [0.02, 0.0, 0.0, 0.0],
                slot: (3, 0),
            },
            ParamSpec {
                name: "furDensity",
                kind: ParamKind::Float,
                default: [600.0, 0.0, 0.0, 0.0],
                slot: (3, 1),
            },
        ],
        flags: raa_lit::FLAG_FUR,
        class: DrawClass::Fur,
    },
    BuiltinFn {
        name: "raa.neon",
        wgsl: include_str!("wgsl/neon.wgsl"),
        params: &[
            ParamSpec {
                name: "color",
                kind: ParamKind::Color,
                default: [1.0, 0.25, 0.6, 0.0],
                slot: (0, 0),
            },
            ParamSpec {
                name: "strength",
                kind: ParamKind::Float,
                default: [12.0, 0.0, 0.0, 0.0],
                slot: (1, 0),
            },
            ParamSpec {
                name: "coreWhiteness",
                kind: ParamKind::Float,
                default: [0.8, 0.0, 0.0, 0.0],
                slot: (1, 1),
            },
        ],
        flags: raa_lit::FLAG_UNLIT,
        class: DrawClass::Opaque,
    },
];

/// Registers every row of [`BUILTINS`] into `reg`, in table order. When any built-in name
/// is already registered nothing is registered and `material.fn_duplicate` is returned.
pub fn register_builtins(reg: &mut MaterialFnRegistry) -> Result<(), ShaderError> {
    if let Some(b) = BUILTINS.iter().find(|b| reg.id_of(b.name).is_some()) {
        return Err(ShaderError {
            code: ShaderError::FN_DUPLICATE,
            message: format!("material function `{}` is already registered", b.name),
        });
    }
    for b in BUILTINS {
        reg.register_builtin(b.name, b.wgsl)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_shader::MaterialFnId;

    #[test]
    fn registers_builtins_and_refuses_duplicates() {
        let mut reg = MaterialFnRegistry::new();
        assert!(register_builtins(&mut reg).is_ok());
        assert_eq!(reg.len(), 1 + BUILTINS.len());
        assert_eq!(reg.id_of("raa.inflatable"), Some(MaterialFnId(1)));
        assert_eq!(reg.id_of("raa.candy"), Some(MaterialFnId(2)));
        assert_eq!(reg.id_of("raa.plush"), Some(MaterialFnId(3)));
        assert_eq!(reg.id_of("raa.neon"), Some(MaterialFnId(4)));
        assert_eq!(BUILTINS.len(), 4);
        assert_eq!(
            (BUILTINS[2].flags, BUILTINS[2].class),
            (raa_lit::FLAG_FUR, DrawClass::Fur)
        );
        assert_eq!(
            (BUILTINS[3].flags, BUILTINS[3].class),
            (raa_lit::FLAG_UNLIT, DrawClass::Opaque)
        );
        for b in BUILTINS {
            let mut used = [0u32; 8];
            for p in b.params {
                let mask = ((1u32 << p.kind.components()) - 1) << p.slot.1;
                assert!(
                    mask < 16 && used[p.slot.0] & mask == 0,
                    "{} {}",
                    b.name,
                    p.name
                );
                used[p.slot.0] |= mask;
            }
        }
        let err = register_builtins(&mut reg).err().map(|e| e.code);
        assert_eq!(err, Some("material.fn_duplicate"));
        assert_eq!(reg.len(), 1 + BUILTINS.len());
    }
}
