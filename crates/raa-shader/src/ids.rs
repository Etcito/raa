//! Identifiers of pass kinds, slots, material functions and shader variants (S11).

/// Pass a composed module provides entry points (`vs_main`, `fs_main`) for.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PassKind {
    /// vs + fs writing the S12 normal and material targets.
    Prepass,
    /// vs + fs to the HDR target, MSAA.
    Forward,
    /// vs only, with material_vertex and the deform slot.
    Shadow,
    /// vs + fs appending to the OIT list.
    Transparent,
    /// vs extruding shells + fs with alpha-to-coverage.
    FurShell,
}

impl PassKind {
    /// Every pass kind, in declaration order.
    pub const ALL: [PassKind; 5] = [
        PassKind::Prepass,
        PassKind::Forward,
        PassKind::Shadow,
        PassKind::Transparent,
        PassKind::FurShell,
    ];
}

/// Function slot a provider fills; the composed module calls the slot function by name.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SlotId {
    Deform,
    Shadow,
    Ao,
    Ambient,
    Direct,
    Transmission,
    Fog,
}

impl SlotId {
    /// Every slot, in composition order.
    pub const ALL: [SlotId; 7] = [
        SlotId::Deform,
        SlotId::Shadow,
        SlotId::Ao,
        SlotId::Ambient,
        SlotId::Direct,
        SlotId::Transmission,
        SlotId::Fog,
    ];

    /// Name of the WGSL function a provider of this slot defines (S11).
    pub fn fn_name(self) -> &'static str {
        match self {
            SlotId::Deform => "raa_slot_deform",
            SlotId::Shadow => "raa_slot_shadow",
            SlotId::Ao => "raa_slot_ao",
            SlotId::Ambient => "raa_slot_ambient",
            SlotId::Direct => "raa_slot_direct",
            SlotId::Transmission => "raa_slot_transmission",
            SlotId::Fog => "raa_slot_fog",
        }
    }
}

/// Dense index of a material function in its [`MaterialFnRegistry`](crate::MaterialFnRegistry).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MaterialFnId(pub u32);

impl MaterialFnId {
    /// The built-in `raa.standard` function, registered first by every registry.
    pub const STANDARD: MaterialFnId = MaterialFnId(0);
}

/// Shader variant: material function, pass kind, MSAA sample count and the material
/// flags subset the variant depends on (SurfaceMaterialGpu.header.z bits, S10).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct VariantKey {
    pub material_fn: MaterialFnId,
    pub pass: PassKind,
    pub msaa: u32,
    pub flags: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_fn_names_follow_s11() {
        let names: Vec<&str> = SlotId::ALL.iter().map(|s| s.fn_name()).collect();
        assert_eq!(
            names,
            [
                "raa_slot_deform",
                "raa_slot_shadow",
                "raa_slot_ao",
                "raa_slot_ambient",
                "raa_slot_direct",
                "raa_slot_transmission",
                "raa_slot_fog",
            ]
        );
    }

    #[test]
    fn standard_id_is_zero() {
        assert_eq!(MaterialFnId::STANDARD, MaterialFnId(0));
    }
}
