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
