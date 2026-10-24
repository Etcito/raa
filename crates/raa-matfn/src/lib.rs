//! Built-in material functions of the lit renderer (S11) and their parameter tables.
//!
//! Each row of [`BUILTINS`] is a [`BuiltinFn`]: a registry name (`raa.<fn>`), the WGSL of its
//! `material_main`, the [`ParamSpec`] table that maps JS parameters into the material
//! record's `p` array, the material flags it implies and its [`DrawClass`].
//! [`register_builtins`] registers every row into a [`MaterialFnRegistry`].
//! [`SurfaceDesc`] (with [`Blend`] and [`StandardOverrides`]) is the JS surface material
//! description; [`to_surface_material`] maps it into a validated `SurfaceMaterial`.
//!
//! Rules every [`ParamSpec`] follows:
//! - `default` is in the unit JS sends: [`ParamKind::Color`] defaults are sRGB in [0, 1];
//!   raa-matfn converts colours to linear when it maps a description into a material, so the
//!   shader always reads linear colours.
//! - `slot` is `(p index, first component)`: the parameter occupies
//!   `p[slot.0][slot.1 .. slot.1 + kind.components()]`, inside one `vec4`. Two parameters of
//!   one function never overlap the same components.
//! - Every material function reads its parameters from `in.record.p[i]`.
//! - Helpers of a built-in function are prefixed `<fnname>_` (e.g. `inflatable_stripe`),
//!   never `raa_` (that prefix belongs to the engine library).

mod builtins;
mod desc;

pub use builtins::{BUILTINS, register_builtins};
pub use desc::{
    Blend, ExtraBuiltins, FUR_DENSITY_DEFAULT, FUR_SHELLS_DEFAULT, FUR_SHELLS_MAX,
    StandardOverrides, SurfaceDesc, map_desc, map_desc_with, to_surface_material,
    to_surface_material_with,
};
pub use raa_lit::DrawClass;
pub use raa_shader::{MaterialFnRegistry, ShaderError};

/// The type of one material function parameter, as JS sends it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ParamKind {
    /// One number.
    Float,
    /// Two numbers.
    Vec2,
    /// Three numbers.
    Vec3,
    /// Four numbers.
    Vec4,
    /// An sRGB colour in [0, 1] (three numbers), stored linear in the record.
    Color,
}

impl ParamKind {
    /// Number of `f32` components the parameter occupies in its `vec4`.
    pub const fn components(self) -> usize {
        match self {
            ParamKind::Float => 1,
            ParamKind::Vec2 => 2,
            ParamKind::Vec3 | ParamKind::Color => 3,
            ParamKind::Vec4 => 4,
        }
    }
}

/// One parameter of a built-in material function.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ParamSpec {
    /// JS name (camelCase).
    pub name: &'static str,
    pub kind: ParamKind,
    /// Default in the unit JS sends (Color: sRGB [0, 1]); unused components are 0.
    pub default: [f32; 4],
    /// `(p index, first component)` in the material record.
    pub slot: (usize, usize),
}

/// One built-in material function.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BuiltinFn {
    /// Registry name, `raa.<fn>`.
    pub name: &'static str,
    /// Source defining `material_main` and its `<fn>_` helpers.
    pub wgsl: &'static str,
    pub params: &'static [ParamSpec],
    /// Material flags (S10) every material using the function gets.
    pub flags: u32,
    /// Draw class of a material using the function with no other flag.
    pub class: DrawClass,
}
