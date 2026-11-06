//! Persistent per-slot instance and extras records: compare-before-dirty writes, merged
//! page ranges, growth and the S1 record layout.

use std::collections::HashMap;
use std::ops::Range;

use raa_core::{MaterialId, MeshId};
use raa_lit::{INSTANCE_GPU_SIZE, InstanceExtra, InstanceStore, StoreKind, write_instance_record};
use raa_math::{Mat4, Quat, Vec3};
use raa_scene::{RenderItem, RenderList};

const PAGE: u64 = 4096;

fn record(seed: u8) -> [u8; INSTANCE_GPU_SIZE] {
    let mut out = [0u8; INSTANCE_GPU_SIZE];
    for (i, b) in out.iter_mut().enumerate() {
        *b = seed.wrapping_add(i as u8);
    }
    out
}

fn ranges(store: &InstanceStore, kind: StoreKind) -> Vec<Range<u64>> {
    let mut out = Vec::new();
    store.dirty_ranges(kind, &mut out);
    out
}

fn f32_at(b: &[u8], off: usize) -> f32 {
    f32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn le_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A list of `n` items at slots 0..n, each with `tags[i] == slots[i]`.
fn synced_list(n: usize) -> (RenderList, Vec<u32>, Vec<u32>) {
    let mut list = RenderList::with_capacity(n);
    let slots: Vec<u32> = (0..n as u32).collect();
    let tags = slots.clone();
    for (i, &slot) in slots.iter().enumerate() {
        list.items.push(RenderItem {
            mesh: MeshId(0),
            material: MaterialId(1),
            world: Mat4::from_translation(Vec3::new(slot as f32, i as f32, 0.0)),
        });
    }
    (list, tags, slots)
}

#[test]
fn new_bytes_dirty_one_page_and_repeats_do_not() {
    let mut store = InstanceStore::new(64);
    assert_eq!(store.capacity(), 64);
    assert!(store.write(3, &record(7)));
    assert_eq!(ranges(&store, StoreKind::Instances), vec![0..PAGE]);
    assert!(!store.write(3, &record(7)));
    assert_eq!(ranges(&store, StoreKind::Instances), vec![0..PAGE]);
    assert_eq!(store.record(3).unwrap(), &record(7)[..]);
}

#[test]
fn slots_in_adjacent_pages_give_one_range_each() {
    let mut store = InstanceStore::new(128);
    for slot in [0u32, 1, 40, 41] {
        assert!(store.write(slot, &record(slot as u8)));
    }
    assert_eq!(
        ranges(&store, StoreKind::Instances),
        vec![0..PAGE, PAGE..2 * PAGE]
    );
    assert!(ranges(&store, StoreKind::Extras).is_empty());
}

#[test]
fn clear_dirty_empties_both_kinds() {
    let mut store = InstanceStore::new(64);
    assert!(store.write(0, &record(1)));
    assert!(store.write_extra(0, &InstanceExtra::default()));
    assert!(!ranges(&store, StoreKind::Instances).is_empty());
    assert!(!ranges(&store, StoreKind::Extras).is_empty());
    store.clear_dirty();
    assert!(ranges(&store, StoreKind::Instances).is_empty());
    assert!(ranges(&store, StoreKind::Extras).is_empty());
}

#[test]
fn ensure_doubles_dirties_every_page_and_keeps_bytes() {
    let mut store = InstanceStore::new(32);
    let before = record(9);
    assert!(store.write(5, &before));
    store.clear_dirty();
    assert!(!store.ensure(32));
    assert!(!store.ensure(1));
    assert_eq!(store.capacity(), 32);
    assert!(store.ensure(33));
    assert_eq!(store.capacity(), 64);
    let every_page = vec![0..PAGE, PAGE..2 * PAGE];
    assert_eq!(ranges(&store, StoreKind::Instances), every_page);
    assert_eq!(ranges(&store, StoreKind::Extras), every_page);
    assert_eq!(store.record(5).unwrap(), &before[..]);
}

#[test]
fn sync_marks_only_changed_records() {
    let (mut list, tags, slots) = synced_list(100);
    let mut store = InstanceStore::new(100);
    let extras = HashMap::new();
    let all_pages = vec![
        0..PAGE,
        PAGE..2 * PAGE,
        2 * PAGE..3 * PAGE,
        3 * PAGE..4 * PAGE,
    ];

    store.sync(&list, &tags, &slots, &extras, &|_| 0);
    assert_eq!(ranges(&store, StoreKind::Instances), all_pages);
    assert_eq!(ranges(&store, StoreKind::Extras), all_pages);

    store.clear_dirty();
    store.sync(&list, &tags, &slots, &extras, &|_| 0);
    assert!(ranges(&store, StoreKind::Instances).is_empty());
    assert!(ranges(&store, StoreKind::Extras).is_empty());

    list.items[50].world = Mat4::from_translation(Vec3::new(0.0, 0.0, 9.0));
    store.sync(&list, &tags, &slots, &extras, &|_| 0);
    assert_eq!(ranges(&store, StoreKind::Instances), vec![PAGE..2 * PAGE]);
    assert!(ranges(&store, StoreKind::Extras).is_empty());
}

#[test]
fn write_instance_record_matches_the_s1_layout() {
    let world = Mat4::from_trs(
        Vec3::new(1.0, 2.0, 3.0),
        Quat::from_axis_angle(Vec3::Y, 0.5),
        Vec3::new(2.0, 3.0, 4.0),
    );
    let mut out = [0xFFu8; INSTANCE_GPU_SIZE];
    write_instance_record(&mut out, &world, 7, 0xdead_beef, 12);

    assert_eq!(out[0..64], le_bytes(&world.to_cols_array())[..]);
    assert_eq!(
        out[64..112],
        le_bytes(world.normal_matrix().as_flattened())[..]
    );
    assert_eq!(u32_at(&out, 112), 7);
    assert_eq!(u32_at(&out, 116), 0xdead_beef);
    assert_eq!(u32_at(&out, 120), 12);
    assert_eq!(f32_at(&out, 48), 1.0);
    assert!(out[124..128].iter().all(|&b| b == 0));
}
