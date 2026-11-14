//! The S37 voxel membership (`raa_gi::membership`): the rank table of `refusal`, the fixed refusal
//! messages, and `plan_members` over a table double. No GPU: the double answers the
//! `MemberSource` queries from plain maps, and the items are built from a `raa_scene::Scene`.

use std::collections::BTreeMap;

use raa_core::{MaterialId, MeshId};
use raa_gi::membership::{
    DEFORMED_INSTANCE, Driver, MemberSource, MembershipReport, NO_SHADOW_MATERIAL,
    NON_OPAQUE_MATERIAL, OPTED_OUT, SKINNED_INSTANCE, plan_members, refusal, refusal_message,
};
use raa_lit::{DrawClass, FLAG_NO_SHADOW, InstanceShape, SurfaceMaterial};
use raa_math::{Mat4, Vec3};
use raa_scene::{Scene, Transform};
use raa_vg::scene::SyncItem;

const OPAQUE: InstanceShape = InstanceShape {
    skinned: false,
    deformed: false,
    class: DrawClass::Opaque,
    vertex_fn: false,
    has_colors: false,
};
const SKINNED: InstanceShape = InstanceShape {
    skinned: true,
    ..OPAQUE
};
const DEFORMED: InstanceShape = InstanceShape {
    deformed: true,
    ..OPAQUE
};
const VERTEX_FN: InstanceShape = InstanceShape {
    vertex_fn: true,
    ..OPAQUE
};
const TRANSPARENT: InstanceShape = InstanceShape {
    class: DrawClass::Transparent,
    ..OPAQUE
};
const FUR: InstanceShape = InstanceShape {
    class: DrawClass::Fur,
    ..OPAQUE
};

/// Shapes by tag and records by material id; anything unlisted is opaque with no record.
#[derive(Default)]
struct Table {
    shapes: BTreeMap<u32, InstanceShape>,
    materials: BTreeMap<u32, SurfaceMaterial>,
}

impl MemberSource for Table {
    fn instance_eligibility(
        &self,
        tag: u32,
        _mesh: MeshId,
        _material: MaterialId,
    ) -> InstanceShape {
        self.shapes.get(&tag).copied().unwrap_or(OPAQUE)
    }

    fn material_record(&self, material: MaterialId) -> Option<SurfaceMaterial> {
        self.materials.get(&material.0).cloned()
    }
}

/// Tag 2 is transparent; every other tag is opaque.
fn transparent_table() -> Table {
    let mut table = Table::default();
    table.shapes.insert(2, TRANSPARENT);
    table
}

/// Claims tag 3 for the virtual driver.
fn tag3(item: &SyncItem) -> bool {
    item.tag == 3
}

/// One scene instance per tag, on one node, all on material 0. Each item takes its index as its
/// slot and mesh and a translation along x as its world matrix.
fn items(tags: &[u32]) -> Vec<SyncItem> {
    let mut scene = Scene::new();
    let node = scene.add_node(None, Transform::IDENTITY).expect("a node");
    tags.iter()
        .enumerate()
        .map(|(index, &tag)| {
            let instance = scene
                .add_instance(node, MeshId(index as u32), MaterialId(0))
                .expect("an instance");
            scene.set_instance_tag(instance, tag).expect("a tag");
            SyncItem {
                instance,
                slot: index as u32,
                tag,
                mesh: MeshId(index as u32),
                material: MaterialId(0),
                world: Mat4::from_translation(Vec3 {
                    x: index as f32,
                    y: 0.0,
                    z: 0.0,
                }),
            }
        })
        .collect()
}

fn plan(items: &[SyncItem], table: &Table) -> MembershipReport {
    let mut out = MembershipReport::default();
    plan_members(items, table, &BTreeMap::new(), &tag3, &mut out);
    out
}

const TAGS: [u32; 4] = [1, 1, 2, 3];

#[test]
fn rank_table_gives_one_code_per_rank_and_the_first_match_wins() {
    type Case = (
        &'static str,
        Option<bool>,
        InstanceShape,
        u32,
        Option<&'static str>,
    );
    let cases: [Case; 9] = [
        (
            "opted out beats a skin",
            Some(false),
            SKINNED,
            0,
            Some(OPTED_OUT),
        ),
        ("skinned", None, SKINNED, 0, Some(SKINNED_INSTANCE)),
        ("deformed", None, DEFORMED, 0, Some(DEFORMED_INSTANCE)),
        (
            "vertex function",
            None,
            VERTEX_FN,
            0,
            Some(DEFORMED_INSTANCE),
        ),
        (
            "transparent",
            None,
            TRANSPARENT,
            0,
            Some(NON_OPAQUE_MATERIAL),
        ),
        ("fur", None, FUR, 0, Some(NON_OPAQUE_MATERIAL)),
        (
            "no shadow",
            None,
            OPAQUE,
            FLAG_NO_SHADOW,
            Some(NO_SHADOW_MATERIAL),
        ),
        ("opaque without a request", None, OPAQUE, 0, None),
        ("opaque requested", Some(true), OPAQUE, 0, None),
    ];
    for (name, request, shape, flags, code) in cases {
        assert_eq!(refusal(request, &shape, flags), code, "{name}");
    }
}

#[test]
fn refusal_messages_are_the_fixed_texts() {
    let texts = [
        (
            OPTED_OUT,
            "opted out: setInstanceGlobalIllumination(false) keeps this group out of the voxels",
        ),
        (
            SKINNED_INSTANCE,
            "skinned instance: posed skins are not voxelized",
        ),
        (
            DEFORMED_INSTANCE,
            "deformed instance or material vertex function: deformed geometry is not voxelized",
        ),
        (
            NON_OPAQUE_MATERIAL,
            "material class other than opaque (transparent, transmissive or fur): only opaque surfaces are voxelized",
        ),
        (
            NO_SHADOW_MATERIAL,
            "material flag NO_SHADOW: materials that cast no shadow are not voxelized",
        ),
        ("gi.unknown", "instance is not voxelized"),
    ];
    for (code, text) in texts {
        assert_eq!(refusal_message(code), text, "{code}");
    }
}

#[test]
fn plan_members_drives_forward_virtual_and_refused_tags() {
    let report = plan(&items(&TAGS), &transparent_table());

    let tags: Vec<u32> = report.members.iter().map(|m| m.item.tag).collect();
    assert_eq!(tags, [1, 1, 3]);
    let drivers: Vec<Driver> = report.members.iter().map(|m| m.driver).collect();
    assert_eq!(drivers, [Driver::Forward, Driver::Forward, Driver::Virtual]);
    assert_eq!((report.forward_count, report.virtual_count), (2, 1));

    let tag1 = &report.per_tag[&1];
    assert!(tag1.voxelized);
    assert_eq!(tag1.driver, Some(Driver::Forward));
    assert_eq!(tag1.code, None);

    let tag2 = &report.per_tag[&2];
    assert!(!tag2.voxelized);
    assert_eq!(tag2.code, Some(NON_OPAQUE_MATERIAL));
    assert_eq!(tag2.message, Some(refusal_message(NON_OPAQUE_MATERIAL)));

    assert_eq!(report.per_tag[&3].driver, Some(Driver::Virtual));
}

#[test]
fn signature_is_stable_and_changes_with_world_emissive_and_membership() {
    let all = items(&TAGS);
    let table = transparent_table();
    let first = plan(&all, &table).signature;
    assert_ne!(first, 0);
    assert_eq!(plan(&all, &table).signature, first, "same input");

    let mut moved = items(&TAGS);
    moved[0].world.cols[3][0] += 1.0;
    assert_ne!(plan(&moved, &table).signature, first, "one world entry");

    let mut lit = transparent_table();
    lit.materials.insert(
        0,
        SurfaceMaterial {
            emissive: [1.0, 0.0, 0.0],
            ..SurfaceMaterial::default()
        },
    );
    assert_ne!(plan(&all, &lit).signature, first, "one material emissive");

    let without_virtual = plan(&all[..3], &table);
    assert_eq!(without_virtual.virtual_count, 0);
    assert_ne!(without_virtual.signature, first, "one member removed");
}

#[test]
fn reused_report_keeps_no_member_of_an_earlier_call() {
    let all = items(&TAGS);
    let table = transparent_table();
    let mut out = MembershipReport::default();
    plan_members(&all, &table, &BTreeMap::new(), &tag3, &mut out);
    assert_eq!(out.members.len(), 3);

    plan_members(&all[..2], &table, &BTreeMap::new(), &tag3, &mut out);
    assert_eq!(out.members.len(), 2);
    assert_eq!((out.forward_count, out.virtual_count), (2, 0));
    let tags: Vec<u32> = out.per_tag.keys().copied().collect();
    assert_eq!(tags, [1]);

    let mut fresh = MembershipReport::default();
    plan_members(&all[..2], &table, &BTreeMap::new(), &tag3, &mut fresh);
    assert_eq!(out.members, fresh.members);
    assert_eq!(out.per_tag, fresh.per_tag);
    assert_eq!(out.signature, fresh.signature);
}
