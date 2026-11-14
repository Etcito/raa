//! Voxel membership of the visible scene instances (S37): which instances the GI volume voxelizes,
//! which driver voxelizes each one (forward A or virtual B), and the first refusal of the others.
//!
//! The rank table of [`refusal`] is evaluated in order and the first match wins. Ranks 1-3 are
//! raa-vg's own table (`raa_vg::eligibility::refusal`), mapped onto the `gi.*` codes; rank 4 is the
//! material flag NO_SHADOW. Every instance, member or not, still receives GI.

use std::collections::BTreeMap;

use raa_core::{MaterialId, MeshId};
use raa_lit::{FLAG_NO_SHADOW, InstanceShape, LitHandle, SurfaceMaterial};
use raa_vg::error::{DEFORMED_UNSUPPORTED, SKINNED_UNSUPPORTED, TRANSPARENT_MATERIAL};
use raa_vg::scene::SyncItem;

/// Rank 0: the group's request is `false`, so it is kept out of the voxels.
pub const OPTED_OUT: &str = "gi.opted_out";
/// Rank 1: a posed skin, mapped from raa-vg's `vg.skinned_unsupported`.
pub const SKINNED_INSTANCE: &str = "gi.skinned_instance";
/// Rank 2: deformed or with a material vertex function, mapped from `vg.deformed_unsupported`.
pub const DEFORMED_INSTANCE: &str = "gi.deformed_instance";
/// Rank 3: a material class other than Opaque, mapped from `vg.transparent_material`.
pub const NON_OPAQUE_MATERIAL: &str = "gi.non_opaque_material";
/// Rank 4: the material flag NO_SHADOW.
pub const NO_SHADOW_MATERIAL: &str = "gi.no_shadow_material";

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// The driver that voxelizes a member: forward (driver A) or virtual (driver B).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Driver {
    Forward,
    Virtual,
}

/// The facts membership is planned from: an instance's path shape and a material's record.
/// [`LitHandle`] answers both from its shared state.
pub trait MemberSource {
    fn instance_eligibility(&self, tag: u32, mesh: MeshId, material: MaterialId) -> InstanceShape;
    fn material_record(&self, material: MaterialId) -> Option<SurfaceMaterial>;
}

impl MemberSource for LitHandle {
    fn instance_eligibility(&self, tag: u32, mesh: MeshId, material: MaterialId) -> InstanceShape {
        LitHandle::instance_eligibility(self, tag, mesh, material)
    }

    fn material_record(&self, material: MaterialId) -> Option<SurfaceMaterial> {
        LitHandle::material_record(self, material)
    }
}

/// One voxel member: a visible scene item and the driver that voxelizes it.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Member {
    pub item: SyncItem,
    pub driver: Driver,
}

/// The membership of one scene tag, as `instanceGlobalIllumination` reports it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InstanceGi {
    pub requested: Option<bool>,
    pub voxelized: bool,
    pub driver: Option<Driver>,
    pub code: Option<&'static str>,
    pub message: Option<&'static str>,
}

/// The planned membership of one frame. [`plan_members`] clears and refills it, keeping capacity.
#[derive(Clone, Debug, Default)]
pub struct MembershipReport {
    pub members: Vec<Member>,
    pub per_tag: BTreeMap<u32, InstanceGi>,
    pub forward_count: u32,
    pub virtual_count: u32,
    pub signature: u64,
}

/// The first refusal code of one item, or `None` when it may be voxelized. `request` is the
/// group's request, `shape` the instance's path shape and `material_flags` its material's flags.
pub fn refusal(
    request: Option<bool>,
    shape: &InstanceShape,
    material_flags: u32,
) -> Option<&'static str> {
    let vg = match raa_vg::eligibility::refusal(shape) {
        Some(SKINNED_UNSUPPORTED) => Some(SKINNED_INSTANCE),
        Some(DEFORMED_UNSUPPORTED) => Some(DEFORMED_INSTANCE),
        Some(TRANSPARENT_MATERIAL) => Some(NON_OPAQUE_MATERIAL),
        _ => None,
    };
    [
        (request == Some(false)).then_some(OPTED_OUT),
        vg,
        ((material_flags & FLAG_NO_SHADOW) != 0).then_some(NO_SHADOW_MATERIAL),
    ]
    .into_iter()
    .flatten()
    .next()
}

/// The fixed text of a refusal code; an unknown code gives `instance is not voxelized`.
pub fn refusal_message(code: &str) -> &'static str {
    match code {
        OPTED_OUT => {
            "opted out: setInstanceGlobalIllumination(false) keeps this group out of the voxels"
        }
        SKINNED_INSTANCE => "skinned instance: posed skins are not voxelized",
        DEFORMED_INSTANCE => {
            "deformed instance or material vertex function: deformed geometry is not voxelized"
        }
        NON_OPAQUE_MATERIAL => {
            "material class other than opaque (transparent, transmissive or fur): only opaque surfaces are voxelized"
        }
        NO_SHADOW_MATERIAL => {
            "material flag NO_SHADOW: materials that cast no shadow are not voxelized"
        }
        _ => "instance is not voxelized",
    }
}

/// Handoff: raa-gi: membership tests calls `plan_members`, `refusal` and `refusal_message` of this
/// file, feeds `plan_members` a `MemberSource` table double and reads the `MembershipReport` it
/// fills: `members` in item order, `per_tag` keyed by tag, the two driver counts and `signature`.
/// Today `plan_members` clears `out`, asks the source for each item's shape and material record,
/// and pushes every item with no refusal as a member, driven by `claimed` (virtual) or forward.
///
/// Each item is refused by the rank table of `refusal` or becomes a member. `per_tag` records the
/// request, whether any item of the tag is a member, the first member's driver and, for a tag with
/// no member, the first refusal and its message. `signature` is an order-stable FNV-1a 64 over
/// each member's slot, mesh, material, world matrix, base colour, emissive and flags; it is a
/// change detector, not a check, and it is 0 when there is no member.
pub fn plan_members(
    items: &[SyncItem],
    source: &dyn MemberSource,
    requests: &BTreeMap<u32, bool>,
    claimed: &dyn Fn(&SyncItem) -> bool,
    out: &mut MembershipReport,
) {
    out.members.clear();
    out.per_tag.clear();
    out.forward_count = 0;
    out.virtual_count = 0;
    out.signature = 0;
    let mut hash = FNV_OFFSET;
    for item in items {
        let request = requests.get(&item.tag).copied();
        let shape = source.instance_eligibility(item.tag, item.mesh, item.material);
        let record = source.material_record(item.material);
        let flags = record.as_ref().map_or(0, |m| m.flags);
        let code = refusal(request, &shape, flags);
        let entry = out.per_tag.entry(item.tag).or_insert_with(|| InstanceGi {
            requested: request,
            ..InstanceGi::default()
        });
        match code {
            Some(code) => {
                if entry.code.is_none() {
                    entry.code = Some(code);
                }
            }
            None => {
                let driver = if claimed(item) {
                    Driver::Virtual
                } else {
                    Driver::Forward
                };
                match driver {
                    Driver::Forward => out.forward_count += 1,
                    Driver::Virtual => out.virtual_count += 1,
                }
                if entry.driver.is_none() {
                    entry.driver = Some(driver);
                }
                entry.voxelized = true;
                hash = fold_member(hash, item, record.as_ref());
                out.members.push(Member {
                    item: *item,
                    driver,
                });
            }
        }
    }
    for entry in out.per_tag.values_mut() {
        if entry.voxelized {
            entry.code = None;
        }
        entry.message = entry.code.map(refusal_message);
    }
    out.signature = if out.members.is_empty() { 0 } else { hash };
}

/// Folds one 32-bit word into `hash`, byte by byte, least significant first (FNV-1a).
fn mix(hash: u64, word: u32) -> u64 {
    word.to_le_bytes().into_iter().fold(hash, |h, byte| {
        (h ^ u64::from(byte)).wrapping_mul(FNV_PRIME)
    })
}

/// Folds one member's slot, mesh, material, world floats (as bits), base colour, emissive and
/// flags into `hash`. A material without a record contributes zeros.
fn fold_member(hash: u64, item: &SyncItem, record: Option<&SurfaceMaterial>) -> u64 {
    let (base, emissive, flags) = record.map_or(([0.0; 3], [0.0; 3], 0), |m| {
        (m.base_color, m.emissive, m.flags)
    });
    let mut h = mix(hash, item.slot);
    h = mix(h, item.mesh.0);
    h = mix(h, item.material.0);
    for value in item
        .world
        .to_cols_array()
        .into_iter()
        .chain(base)
        .chain(emissive)
    {
        h = mix(h, value.to_bits());
    }
    mix(h, flags)
}
