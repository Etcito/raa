//! The S32 scene pool: the instance record offsets, the hierarchy word, the mesh-key and
//! request registries, and (with a software adapter) the pool upload, its section layout and
//! `release`. The GPU test attaches one baked displaced sphere to a `ClusterStore` and maps its
//! lit sub-mesh through `MeshKeys`; it records no pass, so it reads no depth or colour input.

#![cfg(not(target_arch = "wasm32"))]

use raa_cluster::{BuildOptions, ClusterStore, OwnedClusterInput, bake};
use raa_core::MeshId;
use raa_core::testing::{skip, software_gpu};
use raa_math::Mat4;
use raa_vg::{
    INSTANCE_CASTS_SHADOW, INSTANCE_DOUBLE_SIDED, INSTANCE_MOVABLE, INSTANCE_PREVIOUS_VALID,
    InstanceRequests, MeshKeys, SectionRange, VG_INSTANCE_SIZE, VgInstanceRecord, VgScene,
    hierarchy_word, write_instance,
};

fn word(bytes: &[u8; VG_INSTANCE_SIZE], at: usize) -> [u8; 4] {
    bytes[at..at + 4].try_into().expect("four bytes")
}

fn f32_at(bytes: &[u8; VG_INSTANCE_SIZE], at: usize) -> f32 {
    f32::from_le_bytes(word(bytes, at))
}

fn u32_at(bytes: &[u8; VG_INSTANCE_SIZE], at: usize) -> u32 {
    u32::from_le_bytes(word(bytes, at))
}

fn record() -> VgInstanceRecord {
    VgInstanceRecord {
        model: Mat4 {
            cols: [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ],
        },
        previous_rows: [
            [20.0, 21.0, 22.0, 23.0],
            [30.0, 31.0, 32.0, 33.0],
            [40.0, 41.0, 42.0, 43.0],
        ],
        quant: [50.0, 51.0, 52.0, 53.0],
        uv_quant: [60.0, 61.0, 62.0, 63.0],
        ranges: [70, 71, 72, 73],
        ids: [80, 81, INSTANCE_PREVIOUS_VALID | INSTANCE_CASTS_SHADOW, 0],
        bounds: [90.0, 91.0, 92.0, 93.0],
    }
}

#[test]
fn write_instance_puts_every_word_at_its_revised_s32_offset() {
    let record = record();
    let mut bytes = [0u8; VG_INSTANCE_SIZE];
    write_instance(&mut bytes, &record);
    assert_eq!(bytes.len(), 192);

    for (index, value) in record.model.cols.iter().flatten().enumerate() {
        assert_eq!(f32_at(&bytes, index * 4), *value, "model word {index}");
    }
    assert_eq!((f32_at(&bytes, 64), f32_at(&bytes, 76)), (20.0, 23.0));
    assert_eq!((f32_at(&bytes, 80), f32_at(&bytes, 92)), (30.0, 33.0));
    assert_eq!((f32_at(&bytes, 96), f32_at(&bytes, 108)), (40.0, 43.0));
    assert_eq!((f32_at(&bytes, 112), f32_at(&bytes, 124)), (50.0, 53.0));
    assert_eq!((f32_at(&bytes, 128), f32_at(&bytes, 140)), (60.0, 63.0));
    assert_eq!((u32_at(&bytes, 144), u32_at(&bytes, 156)), (70, 73));
    assert_eq!((u32_at(&bytes, 160), u32_at(&bytes, 168)), (80, 5));
    assert_eq!(u32_at(&bytes, 172), 0);
    assert_eq!((f32_at(&bytes, 176), f32_at(&bytes, 188)), (90.0, 93.0));

    assert_eq!(INSTANCE_PREVIOUS_VALID, 1);
    assert_eq!(INSTANCE_DOUBLE_SIDED, 2);
    assert_eq!(INSTANCE_CASTS_SHADOW, 4);
    assert_eq!(INSTANCE_MOVABLE, 8);
}

#[test]
fn hierarchy_word_packs_depth_below_the_relative_root() {
    assert_eq!(hierarchy_word(7, 300), 7 | 300 << 8);
    assert_eq!(hierarchy_word(7, 300) & 0xff, 7);
    assert_eq!(hierarchy_word(7, 300) >> 8, 300);
    assert_eq!(hierarchy_word(0, 0), 0);
}

#[test]
fn mesh_keys_map_sub_meshes_and_bump_only_on_change() {
    let mut keys = MeshKeys::default();
    assert_eq!(keys.generation(), 0);
    keys.set(5, &[MeshId(10), MeshId(11)]);
    assert_eq!(keys.get(MeshId(10)), Some((5, 0)));
    assert_eq!(keys.get(MeshId(11)), Some((5, 1)));
    assert_eq!(keys.keys(), vec![5]);
    assert_eq!(keys.generation(), 1);

    keys.set(5, &[MeshId(10), MeshId(11)]);
    assert_eq!(keys.generation(), 1, "the same list does not bump");

    keys.set(5, &[MeshId(12)]);
    assert_eq!(keys.generation(), 2);
    assert_eq!(keys.get(MeshId(10)), None);
    assert_eq!(keys.get(MeshId(12)), Some((5, 0)));

    keys.set(2, &[MeshId(20)]);
    assert_eq!(keys.keys(), vec![2, 5]);
    assert_eq!(keys.generation(), 3);

    keys.remove(5);
    assert_eq!(keys.generation(), 4);
    assert_eq!(keys.get(MeshId(12)), None);
    assert_eq!(keys.keys(), vec![2]);
    keys.remove(99);
    assert_eq!(keys.generation(), 4, "an absent key does not bump");
}

#[test]
fn instance_requests_set_get_and_remove() {
    let mut requests = InstanceRequests::default();
    assert_eq!(requests.get(7), None);
    requests.set(7, Some(true));
    assert_eq!(requests.get(7), Some(true));
    requests.set(7, Some(false));
    assert_eq!(requests.get(7), Some(false));
    requests.set(7, None);
    assert_eq!(requests.get(7), None);
    requests.set(8, Some(true));
    assert_eq!(requests.get(8), Some(true));
}

#[test]
fn upload_pool_builds_the_layout_and_release_drops_every_buffer() {
    let test = "upload_pool_builds_the_layout_and_release_drops_every_buffer";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let input = OwnedClusterInput::displaced_sphere(2);
    let mesh = bake(input.as_input(), &BuildOptions::default()).expect("the sphere bakes");
    let root_node = mesh.root_node;
    let hierarchy_depth = mesh.hierarchy_depth;
    let mut store = ClusterStore::new();
    store.attach(3, vec![mesh]);
    let mut keys = MeshKeys::default();
    keys.set(3, &[MeshId(30)]);

    let mut scene = VgScene::new();
    assert_eq!(scene.allocated_bytes(), 0);
    assert!(scene.upload_pool(&gpu, &store, &keys));
    assert!(
        !scene.upload_pool(&gpu, &store, &keys),
        "nothing changed, so the pool is not rebuilt"
    );

    let range = scene.section_range(3, 0).expect("the section is packed");
    assert_eq!(range.node_base, 0);
    assert_eq!(range.cluster_base, 0);
    assert_eq!(range.vertex_base, 0);
    assert_eq!(range.triangle_base, 0);
    assert_eq!(scene.section_range(3, 1), None);
    assert_eq!(scene.section_range(4, 0), None);
    assert_eq!(
        scene.section_header(3, 0),
        Some((
            SectionRange {
                node_base: 0,
                cluster_base: 0,
                vertex_base: 0,
                triangle_base: 0,
            },
            hierarchy_depth,
            root_node,
        ))
    );
    assert!(scene.allocated_bytes() > 0);
    assert!(scene.instances().is_some());
    assert!(scene.nodes().is_some());
    assert!(scene.clusters().is_some());
    assert!(scene.vertices().is_some());
    assert!(scene.triangles().is_some());

    let mut record = record();
    record.ids[3] = hierarchy_word(5, 2);
    scene.upload_instances(&gpu, &[record]);
    assert_eq!(scene.instance_count(), 1);
    assert_eq!(scene.max_hierarchy_depth(), 5);
    let generation = scene.generation();
    scene.upload_instances(&gpu, &[record]);
    assert_eq!(scene.generation(), generation, "equal bytes write nothing");

    scene.release();
    assert!(scene.instances().is_none());
    assert!(scene.nodes().is_none());
    assert!(scene.clusters().is_none());
    assert!(scene.vertices().is_none());
    assert!(scene.triangles().is_none());
    assert_eq!(scene.allocated_bytes(), 0);
    assert_eq!(scene.instance_count(), 0);
    assert_eq!(scene.max_hierarchy_depth(), 0);
    assert!(scene.section_range(3, 0).is_none());
    assert!(scene.upload_pool(&gpu, &store, &keys), "release rebuilds");
}

mod plan {
    use std::collections::BTreeMap;

    use raa_cluster::{BuildOptions, ClusterMesh, ClusterStore, OwnedClusterInput, bake};
    use raa_core::testing::{skip, software_gpu};
    use raa_core::{MaterialId, MeshId};
    use raa_lit::{DrawClass, InstanceShape, MaterialFnId};
    use raa_scene::{InstanceId, Scene, Transform};
    use raa_vg::scene::{ItemScratch, SyncSource, VgPlan, plan_sync};
    use raa_vg::{
        INSTANCE_CASTS_SHADOW, INSTANCE_DOUBLE_SIDED, INSTANCE_MOVABLE, INSTANCE_PREVIOUS_VALID,
        InstancePath, InstanceRequests, MeshKeys, VgInstanceRecord, VgScene, VgSettings,
    };

    const ELIGIBLE: InstanceShape = InstanceShape {
        skinned: false,
        deformed: false,
        class: DrawClass::Opaque,
        vertex_fn: false,
        has_colors: false,
    };

    #[derive(Default)]
    struct Double {
        shapes: BTreeMap<u32, InstanceShape>,
        functions: BTreeMap<u32, MaterialFnId>,
        flags: BTreeMap<u32, u32>,
        previous: BTreeMap<u32, [[f32; 4]; 3]>,
    }

    impl SyncSource for Double {
        fn instance_eligibility(
            &self,
            tag: u32,
            _mesh: MeshId,
            _material: MaterialId,
        ) -> InstanceShape {
            self.shapes.get(&tag).copied().unwrap_or(ELIGIBLE)
        }

        fn material_fn_of(&self, material: MaterialId) -> MaterialFnId {
            self.functions
                .get(&material.0)
                .copied()
                .unwrap_or(MaterialFnId::STANDARD)
        }

        fn material_flags(&self, material: MaterialId) -> u32 {
            self.flags.get(&material.0).copied().unwrap_or(0)
        }

        fn previous_instance_row(&self, slot: u32) -> Option<[[f32; 4]; 3]> {
            self.previous.get(&slot).copied()
        }
    }

    fn scene_of(instances: &[(u32, u32, u32)]) -> (Scene, Vec<InstanceId>) {
        let mut scene = Scene::new();
        let node = scene.add_node(None, Transform::IDENTITY).expect("a node");
        let mut ids = Vec::new();
        for &(tag, mesh, material) in instances {
            let id = scene
                .add_instance(node, MeshId(mesh), MaterialId(material))
                .expect("an instance");
            scene.set_instance_tag(id, tag).expect("a tag");
            ids.push(id);
        }
        scene.update_world();
        (scene, ids)
    }

    fn baked() -> ClusterMesh {
        let input = OwnedClusterInput::displaced_sphere(2);
        bake(input.as_input(), &BuildOptions::default()).expect("the sphere bakes")
    }

    fn record_of(plan: &VgPlan, instance: InstanceId) -> &VgInstanceRecord {
        plan.instances
            .iter()
            .find(|planned| planned.instance == instance)
            .map(|planned| &planned.record)
            .expect("the instance is planned")
    }

    #[test]
    fn virtual_groups_report_functions_ids_and_record_flags() {
        let section = baked();
        let depth = section.hierarchy_depth;
        let mut store = ClusterStore::new();
        store.attach(1, vec![section]);
        let mut keys = MeshKeys::default();
        keys.set(1, &[MeshId(101)]);
        let (scene, ids) = scene_of(&[(10, 101, 201), (17, 101, 202), (18, 101, 201)]);
        let mut source = Double::default();
        source.functions.insert(201, MaterialFnId(3));
        source.functions.insert(202, MaterialFnId(1));
        source.flags.insert(201, 8);
        source.flags.insert(202, 16);
        source.previous.insert(
            0,
            [
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0],
            ],
        );

        let mut scratch = ItemScratch::default();
        let items = scratch.fill(&scene);
        let plan = plan_sync(
            items,
            &source,
            &store,
            &Default::default(),
            &keys,
            &VgSettings::default(),
        );

        assert_eq!(plan.report.virtual_instances, 3);
        assert_eq!(plan.report.forward_instances, 0);
        assert_eq!(plan.report.ready_meshes, 1);
        assert_eq!(plan.report.baking_meshes, 0);
        assert_eq!(
            plan.report.material_fns,
            vec![MaterialFnId(1), MaterialFnId(3)]
        );
        assert_eq!(plan.report.virtual_ids, ids);
        for tag in [10, 17, 18] {
            assert_eq!(plan.report.paths.get(&tag), Some(&InstancePath::Virtual));
        }

        let moving = record_of(&plan, ids[0]);
        assert_eq!(moving.ranges, [0; 4]);
        assert_eq!(moving.ids[0], 0);
        assert_eq!(moving.ids[1], 201);
        assert_eq!(moving.ids[3] & 0xff, depth);
        assert_eq!(moving.ids[3] >> 8, 0, "the root is filled at upload");
        assert_ne!(moving.ids[2] & INSTANCE_DOUBLE_SIDED, 0);
        assert_ne!(moving.ids[2] & INSTANCE_CASTS_SHADOW, 0);
        assert_ne!(moving.ids[2] & INSTANCE_PREVIOUS_VALID, 0);
        assert_ne!(moving.ids[2] & INSTANCE_MOVABLE, 0);

        let still = record_of(&plan, ids[1]);
        assert_eq!(still.ids[1], 202);
        assert_eq!(still.ids[2] & INSTANCE_DOUBLE_SIDED, 0);
        assert_eq!(
            still.ids[2] & INSTANCE_CASTS_SHADOW,
            0,
            "NO_SHADOW clears it"
        );
        assert_eq!(still.ids[2] & INSTANCE_PREVIOUS_VALID, 0);
        assert_eq!(still.ids[2] & INSTANCE_MOVABLE, 0);

        let double_sided = record_of(&plan, ids[2]);
        assert_ne!(double_sided.ids[2] & INSTANCE_DOUBLE_SIDED, 0);
        assert_ne!(double_sided.ids[2] & INSTANCE_CASTS_SHADOW, 0);
    }

    #[test]
    fn ineligible_groups_are_refused_only_when_requested() {
        let mut store = ClusterStore::new();
        store.attach(1, vec![baked()]);
        let mut keys = MeshKeys::default();
        keys.set(1, &[MeshId(101)]);
        let (scene, _) = scene_of(&[(11, 101, 201), (12, 101, 201)]);
        let mut source = Double::default();
        source.shapes.insert(
            11,
            InstanceShape {
                skinned: true,
                ..ELIGIBLE
            },
        );
        source.shapes.insert(
            12,
            InstanceShape {
                class: DrawClass::Transparent,
                ..ELIGIBLE
            },
        );
        let mut scratch = ItemScratch::default();
        let items = scratch.fill(&scene);
        let settings = VgSettings::default();
        let mut requests = InstanceRequests::default();

        let plan = plan_sync(items, &source, &store, &requests, &keys, &settings);
        assert_eq!(plan.report.virtual_instances, 0);
        assert_eq!(plan.report.forward_requested, 2);
        assert_eq!(
            plan.report.ineligible.get(&11),
            Some(&"vg.skinned_unsupported")
        );
        assert_eq!(
            plan.report.ineligible.get(&12),
            Some(&"vg.transparent_material")
        );
        assert_eq!(plan.report.refused.get("vg.skinned_unsupported"), Some(&1));
        assert_eq!(plan.report.refused.get("vg.transparent_material"), Some(&1));

        requests.set(11, Some(false));
        let plan = plan_sync(items, &source, &store, &requests, &keys, &settings);
        assert_eq!(plan.report.forward_requested, 1);
        assert_eq!(
            plan.report.ineligible.get(&11),
            Some(&"vg.skinned_unsupported"),
            "reported whether requested or not"
        );
        assert_eq!(plan.report.refused.get("vg.skinned_unsupported"), None);
        assert_eq!(plan.report.refused.get("vg.transparent_material"), Some(&1));
        assert!(matches!(
            plan.report.paths.get(&11),
            Some(InstancePath::Forward {
                pending: false,
                code: None,
                ..
            })
        ));
    }

    #[test]
    fn pending_meshes_bake_only_when_selected() {
        let mut store = ClusterStore::new();
        store
            .start_bake(
                2,
                vec![OwnedClusterInput::displaced_sphere(1)],
                &BuildOptions::default(),
            )
            .expect("the bake starts");
        let mut keys = MeshKeys::default();
        keys.set(2, &[MeshId(102)]);
        keys.set(3, &[MeshId(103)]);
        let (scene, _) = scene_of(&[(13, 103, 201), (14, 102, 201)]);
        let source = Double::default();
        let mut scratch = ItemScratch::default();
        let items = scratch.fill(&scene);
        let mut settings = VgSettings::default();
        let requests = InstanceRequests::default();

        let plan = plan_sync(items, &source, &store, &requests, &keys, &settings);
        assert_eq!(plan.report.baking_meshes, 1);
        assert_eq!(plan.report.ready_meshes, 0);
        assert_eq!(plan.report.bake_requests, vec![3]);
        for tag in [13, 14] {
            assert!(matches!(
                plan.report.paths.get(&tag),
                Some(InstancePath::Forward {
                    pending: true,
                    code: None,
                    ..
                })
            ));
        }

        settings.runtime_bake = false;
        let plan = plan_sync(items, &source, &store, &requests, &keys, &settings);
        assert!(plan.report.bake_requests.is_empty());
        assert!(matches!(
            plan.report.paths.get(&13),
            Some(InstancePath::Forward { pending: true, .. })
        ));
        assert!(matches!(
            plan.report.paths.get(&14),
            Some(InstancePath::Forward { pending: true, .. })
        ));
    }

    #[test]
    fn refused_meshes_map_their_codes_and_missing_sections_stay_forward() {
        let mut store = ClusterStore::new();
        store.refuse(4, "cluster.empty", "the mesh has no triangles".to_string());
        store.attach(6, vec![baked()]);
        let mut keys = MeshKeys::default();
        keys.set(4, &[MeshId(105)]);
        keys.set(6, &[MeshId(106), MeshId(107)]);
        let (scene, _) = scene_of(&[(15, 105, 201), (16, 106, 201), (16, 107, 201)]);
        let source = Double::default();
        let mut scratch = ItemScratch::default();
        let items = scratch.fill(&scene);
        let plan = plan_sync(
            items,
            &source,
            &store,
            &Default::default(),
            &keys,
            &VgSettings::default(),
        );

        assert!(matches!(
            plan.report.paths.get(&15),
            Some(InstancePath::Forward {
                pending: false,
                code: Some("vg.mesh_empty"),
                message: Some(message),
            }) if message == "the mesh has no triangles"
        ));
        assert!(
            plan.report.refused.is_empty(),
            "mesh codes are not instance refusals"
        );
        assert!(matches!(
            plan.report.paths.get(&16),
            Some(InstancePath::Forward {
                pending: false,
                code: None,
                ..
            })
        ));
        assert!(plan.instances.is_empty());
        assert_eq!(plan.report.forward_requested, 2);
    }

    #[test]
    fn upload_plan_attaches_the_virtual_records() {
        let test = "upload_plan_attaches_the_virtual_records";
        let Some(gpu) = software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let mesh = baked();
        let depth = mesh.hierarchy_depth;
        let mut store = ClusterStore::new();
        store.attach(1, vec![mesh]);
        let mut keys = MeshKeys::default();
        keys.set(1, &[MeshId(101)]);
        let (scene, _) = scene_of(&[(10, 101, 201)]);
        let source = Double::default();
        let mut scratch = ItemScratch::default();
        let items = scratch.fill(&scene);
        let plan = plan_sync(
            items,
            &source,
            &store,
            &Default::default(),
            &keys,
            &VgSettings::default(),
        );
        assert_eq!(plan.report.virtual_instances, 1);

        let mut vg = VgScene::new();
        vg.upload_plan(&gpu, &store, &keys, &plan);
        assert_eq!(vg.instance_count(), 1);
        assert_eq!(vg.max_hierarchy_depth(), depth);
        let (range, section_depth, _) = vg.section_header(1, 0).expect("the section is packed");
        assert_eq!(range.node_base, 0);
        assert_eq!(section_depth, depth);
    }
}
