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
