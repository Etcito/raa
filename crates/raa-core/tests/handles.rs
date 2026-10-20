use raa_core::{Handle, Pool};
use std::collections::HashSet;

// Not Clone, Eq, Hash or Debug: the handle traits must not depend on T.
struct Opaque(u32);

#[test]
fn insert_then_get_round_trips() {
    let mut pool = Pool::new();
    let a = pool.insert(Opaque(1));
    let b = pool.insert(Opaque(2));
    assert_eq!(pool.get(a).map(|v| v.0), Some(1));
    assert_eq!(pool.get(b).map(|v| v.0), Some(2));
    if let Some(v) = pool.get_mut(b) {
        v.0 = 20;
    }
    assert_eq!(pool.get(b).map(|v| v.0), Some(20));
}

#[test]
fn handle_is_copy_eq_hash_debug_without_bounds_on_t() {
    let mut pool = Pool::new();
    let a: Handle<Opaque> = pool.insert(Opaque(1));
    let copy = a;
    assert_eq!(a, copy);
    let set: HashSet<Handle<Opaque>> = [a, copy].into_iter().collect();
    assert_eq!(set.len(), 1);
    assert_eq!(format!("{a:?}"), "Handle { index: 0, generation: 0 }");
}

#[test]
fn remove_returns_value_and_stale_handle_returns_none() {
    let mut pool = Pool::new();
    let a = pool.insert(Opaque(7));
    assert_eq!(pool.remove(a).map(|v| v.0), Some(7));
    assert!(pool.get(a).is_none());
    assert!(pool.get_mut(a).is_none());
    assert!(pool.remove(a).is_none());
}

#[test]
fn stale_handle_does_not_reach_reused_slot() {
    let mut pool = Pool::new();
    let a = pool.insert(Opaque(1));
    pool.remove(a);
    let b = pool.insert(Opaque(2));
    assert_eq!(b.index(), a.index());
    assert!(b.generation() > a.generation());
    assert!(pool.get(a).is_none());
    assert!(pool.remove(a).is_none());
    assert_eq!(pool.get(b).map(|v| v.0), Some(2));
}

#[test]
fn freed_slots_are_reused_lifo() {
    let mut pool = Pool::new();
    let a = pool.insert(Opaque(0));
    let _b = pool.insert(Opaque(1));
    let c = pool.insert(Opaque(2));
    pool.remove(a);
    pool.remove(c);
    let first = pool.insert(Opaque(3));
    let second = pool.insert(Opaque(4));
    assert_eq!(first.index(), c.index());
    assert_eq!(second.index(), a.index());
    let third = pool.insert(Opaque(5));
    assert_eq!(third.index(), 3);
}

#[test]
fn iter_yields_live_entries_in_index_order() {
    let mut pool = Pool::new();
    let handles: Vec<_> = (0..5).map(|i| pool.insert(Opaque(i))).collect();
    pool.remove(handles[1]);
    pool.remove(handles[3]);
    let seen: Vec<(u32, u32)> = pool.iter().map(|(h, v)| (h.index(), v.0)).collect();
    assert_eq!(seen, vec![(0, 0), (2, 2), (4, 4)]);
    for (h, v) in pool.iter() {
        assert_eq!(pool.get(h).map(|x| x.0), Some(v.0));
    }
}

#[test]
fn len_tracks_inserts_and_removes() {
    let mut pool = Pool::new();
    assert_eq!(pool.len(), 0);
    assert!(pool.is_empty());
    let a = pool.insert(Opaque(1));
    let b = pool.insert(Opaque(2));
    assert_eq!(pool.len(), 2);
    pool.remove(a);
    assert_eq!(pool.len(), 1);
    pool.remove(a);
    assert_eq!(pool.len(), 1);
    pool.remove(b);
    assert_eq!(pool.len(), 0);
    pool.insert(Opaque(3));
    assert_eq!(pool.len(), 1);
}

#[test]
fn with_capacity_does_not_change_len() {
    let pool: Pool<Opaque> = Pool::with_capacity(64);
    assert_eq!(pool.len(), 0);
    assert!(pool.is_empty());
    assert_eq!(pool.iter().count(), 0);
}
