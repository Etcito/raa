//! m37 asynchronous billboard baking: conversion, pending/failed state, stale completion
//! disposal, removal ordering and the pending/failed preparation gate. The pure store and
//! the group preparation core run natively; no GPU is involved.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use raa_core::RaaError;
use raa_lit::BillboardBakeResult;
use raa_math::{Aabb, Mat4, Vec3};

#[allow(dead_code)]
#[path = "../src/api/m36_billboard_groups.rs"]
mod m36;

#[allow(dead_code)]
#[path = "../src/api/m37_billboard_bake.rs"]
mod m37;

use m36::{
    BillboardGroupConfig, BillboardGroupRecord, BillboardGroupRegistry, BillboardGroupStatus,
    BillboardLevel, BillboardPrepareRecipe, fail_billboard_record, prepare_billboard_record,
};
use m37::{
    BILLBOARD_BAKE_FAILED, BILLBOARD_INVALID, BillboardBake, BillboardBakeState,
    BillboardBakeStore, Completion, RemovalStep, StateGuard, expand_tags, remove_bake, root_frame,
    validate_resolution,
};

fn result() -> BillboardBakeResult {
    BillboardBakeResult {
        diffuse: 11,
        normal: 12,
        opacity: 13,
        center: [1.0, 2.0, 3.0],
        size: [4.0, 5.0],
        tiles: [
            [0, 0, 64, 64],
            [66, 0, 64, 64],
            [132, 0, 64, 64],
            [0, 66, 64, 64],
            [66, 66, 64, 64],
            [132, 66, 64, 64],
            [0, 132, 64, 64],
            [66, 132, 64, 64],
        ],
        extent: [198, 198],
    }
}

fn code_of(e: &RaaError) -> &'static str {
    e.code()
}

#[test]
fn registry_exports_the_bake_and_proxy_methods() {
    let path = Path::new(env!("OUT_DIR")).join("api_registry.rs");
    let reg = fs::read_to_string(&path).expect("api_registry.rs");
    assert!(reg.contains("mod m37_billboard_bake;"));
    assert!(reg.contains("m37_billboard_bake::install(ctx)?;"));
    assert!(reg.contains("m37_billboard_bake::frame(st, gpu, input)?;"));
    let src = include_str!("../src/api/m37_billboard_bake.rs");
    for name in [
        "bakeBillboard",
        "removeBillboardBake",
        "createBillboardProxy",
    ] {
        assert!(src.contains(&format!("js_name = {name}")), "no {name}");
    }
}

#[test]
fn conversion_copies_the_atlas_framing_tiles_and_extent() {
    let bake = BillboardBake::from_result(7, result());
    assert_eq!(bake.id, 7);
    assert_eq!(bake.atlas(), [11, 12, 13]);
    assert_eq!(bake.center, [1.0, 2.0, 3.0]);
    assert_eq!(bake.size, [4.0, 5.0]);
    assert_eq!(bake.directions, 8);
    assert_eq!(bake.tiles, result().tiles);
    assert_eq!(bake.extent, [198, 198]);
}

#[test]
fn a_pending_bake_is_not_selectable_and_a_failed_one_stays_failed() {
    let mut store = BillboardBakeStore::new();
    let id = store.begin(10);
    assert!(!store.is_ready(id));
    assert_eq!(store.ready(id), None);
    assert!(store.ready_ids().is_empty());
    assert_eq!(store.state(id), Some(&BillboardBakeState::Pending));

    assert!(store.fail(id, "GPU validation"));
    assert!(!store.is_ready(id));
    assert!(store.ready_ids().is_empty());
    assert_eq!(
        store.state(id),
        Some(&BillboardBakeState::Failed("GPU validation".into()))
    );
    // A late completion of a failed entry is discarded, never published.
    assert_eq!(
        store.complete(id, BillboardBake::from_result(id, result())),
        Completion::Discarded
    );
    assert!(!store.is_ready(id));
}

#[test]
fn a_complete_bake_becomes_ready_only_after_publication() {
    let mut store = BillboardBakeStore::new();
    let id = store.begin(10);
    let bake = BillboardBake::from_result(id, result());
    assert_eq!(store.complete(id, bake), Completion::Published);
    assert!(store.is_ready(id));
    assert_eq!(store.ready(id), Some(bake));
    assert_eq!(store.ready_ids(), BTreeSet::from([id]));
}

#[test]
fn a_stale_completion_hands_its_atlas_back_for_disposal() {
    let mut store = BillboardBakeStore::new();
    let id = store.begin(10);
    let removed = store.take(id).expect("pending entry");
    assert_eq!(removed.state, BillboardBakeState::Pending);
    let bake = BillboardBake::from_result(id, result());
    assert_eq!(
        store.complete(id, bake),
        Completion::Stale([11, 12, 13]),
        "a removed bake cannot be published and its images are disposed"
    );
    assert!(!store.is_ready(id));
    assert_eq!(store.ready_ids(), BTreeSet::new());
}

#[test]
fn removing_a_bake_detaches_groups_before_freeing_its_images() {
    let mut store = BillboardBakeStore::new();
    let id = store.begin(10);
    store.complete(id, BillboardBake::from_result(id, result()));
    let mut steps = Vec::new();
    let bake =
        remove_bake(&mut store, id, &mut |step| steps.push(step)).expect("complete bake removed");
    assert_eq!(
        steps,
        vec![
            RemovalStep::Detach(id),
            RemovalStep::Free(11),
            RemovalStep::Free(12),
            RemovalStep::Free(13),
        ]
    );
    assert_eq!(bake.id, id);
    assert!(!store.is_ready(id));
    assert_eq!(
        code_of(&remove_bake(&mut store, id, &mut |_| {}).err().unwrap()),
        BILLBOARD_INVALID
    );
}

#[test]
fn handles_expand_to_the_complete_tag_set() {
    let mut tags = BTreeMap::new();
    tags.insert(1u32, 101u32);
    tags.insert(2u32, 102u32);
    let live = |h: u32| tags.get(&h).copied();
    assert_eq!(expand_tags(&[1, 2], live), Ok(vec![101, 102]));
    assert_eq!(
        code_of(&expand_tags(&[], live).err().unwrap()),
        BILLBOARD_INVALID
    );
    assert_eq!(
        code_of(&expand_tags(&[1, 1], live).err().unwrap()),
        BILLBOARD_INVALID
    );
    assert_eq!(
        code_of(&expand_tags(&[1, 9], live).err().unwrap()),
        BILLBOARD_INVALID
    );
}

#[test]
fn the_root_frame_must_be_live_and_invertible() {
    let world = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
    assert_eq!(root_frame(4, |_| Some(world)), Ok(world));
    assert_eq!(
        code_of(&root_frame(4, |_| None).err().unwrap()),
        BILLBOARD_INVALID
    );
    let singular = Mat4::from_scale(Vec3::new(1.0, 0.0, 1.0));
    assert_eq!(
        code_of(&root_frame(4, |_| Some(singular)).err().unwrap()),
        BILLBOARD_INVALID
    );
    assert_eq!(validate_resolution(64), Ok(64));
    assert_eq!(
        code_of(&validate_resolution(65).err().unwrap()),
        BILLBOARD_INVALID
    );
    assert_eq!(BILLBOARD_BAKE_FAILED, "billboard.bake_failed");
}

fn config(bake: u32) -> BillboardGroupConfig {
    BillboardGroupConfig {
        root: 10,
        full: vec![1],
        proxy: 2,
        bake,
        switch_pixels: 100.0,
        bias: 0,
        level: BillboardLevel::Auto,
    }
}

fn registered(bake: u32) -> BillboardGroupRegistry {
    let mut registry = BillboardGroupRegistry::new();
    let roots: BTreeSet<u32> = [10].into_iter().collect();
    let live: BTreeMap<u32, bool> = [(1, true), (2, true)].into_iter().collect();
    registry
        .set("tree", Some(config(bake)), &roots, &live)
        .expect("valid group");
    registry
}

fn recipe() -> BillboardPrepareRecipe {
    BillboardPrepareRecipe {
        eye: Vec3::new(0.0, 0.0, 5.0),
        target: Vec3::ZERO,
        up: Vec3::Y,
        fov_y_rad: 60.0_f32.to_radians(),
        near: 0.1,
        ortho_height: None,
        viewport: (1000, 1000),
        root_world: Mat4::IDENTITY,
        proxy_world: Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)),
        bounds: Aabb {
            min: Vec3::splat(-0.05),
            max: Vec3::splat(0.05),
        },
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Gate {
    Waiting,
    Failed(String),
    Ready,
}

fn gate(status: &BillboardGroupStatus) -> Gate {
    if status.ready {
        Gate::Ready
    } else if let Some(error) = &status.error {
        Gate::Failed(error.clone())
    } else {
        Gate::Waiting
    }
}

fn run_action(gate: &Gate, ran: &mut u32) {
    if *gate == Gate::Ready {
        *ran += 1;
    }
}

fn prepared(registry: &mut BillboardGroupRegistry, bake_ready: bool) -> m36::BillboardPrepared {
    let record: &mut BillboardGroupRecord = registry.get_mut_for_test("tree");
    prepare_billboard_record(record, 0, &recipe(), bake_ready).expect("valid recipe")
}

#[test]
fn a_pending_preparation_keeps_loading_and_the_gated_action_waits() {
    let mut store = BillboardBakeStore::new();
    let bake = store.begin(10);
    let mut registry = registered(bake);
    let mut ran = 0;
    let prepared = prepared(&mut registry, store.is_ready(bake));
    assert_eq!(prepared.level, 0, "a pending bake keeps LOD0");
    let record = registry.get("tree").expect("registered");
    assert_eq!(record.status.level, 0);
    assert!(!record.status.ready, "pending never reports ready");
    assert_eq!(record.status.error, None, "pending is not a failure");
    let gate = gate(&record.status);
    assert_eq!(gate, Gate::Waiting);
    run_action(&gate, &mut ran);
    assert_eq!(ran, 0, "the gated action waits for the bake");
}

#[test]
fn a_failed_preparation_surfaces_the_failure_and_the_gated_action_never_proceeds() {
    let mut store = BillboardBakeStore::new();
    let bake = store.begin(10);
    store.fail(bake, "bake failed");
    let mut registry = registered(bake);
    let mut ran = 0;
    let prepared = prepared(&mut registry, store.is_ready(bake));
    assert_eq!(prepared.level, 0);
    {
        let record = registry.get_mut_for_test("tree");
        fail_billboard_record(record, 0, "the bake could not be prepared").expect("valid view");
    }
    let record = registry.get("tree").expect("registered");
    assert_eq!(record.status.level, 0);
    assert!(!record.status.ready);
    assert_eq!(
        record.status.error.as_deref(),
        Some("the bake could not be prepared")
    );
    let gate = gate(&record.status);
    assert_eq!(gate, Gate::Failed("the bake could not be prepared".into()));
    run_action(&gate, &mut ran);
    assert_eq!(ran, 0, "a failed preparation never proceeds");
}

#[test]
fn the_proxy_is_one_quad_from_the_bake_center_and_size() {
    let bake = BillboardBake::from_result(7, result());
    let mesh = raa_lit::billboard_proxy(bake.center, bake.size).expect("valid proxy");
    assert_eq!(mesh.positions.len(), 4, "four vertices");
    assert_eq!(mesh.normals.len(), 4);
    assert_eq!(
        mesh.indices,
        vec![0, 1, 2, 0, 2, 3],
        "six indices, two triangles"
    );
    assert!(mesh.normals.iter().all(|n| *n == [0.0, 0.0, 1.0]));
    let [cx, cy, cz] = bake.center;
    let (hw, hh) = (bake.size[0] * 0.5, bake.size[1] * 0.5);
    let mut xs: Vec<f32> = mesh.positions.iter().map(|p| p[0]).collect();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(xs.first().copied(), Some(cx - hw));
    assert_eq!(xs.last().copied(), Some(cx + hw));
    let mut ys: Vec<f32> = mesh.positions.iter().map(|p| p[1]).collect();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(ys.first().copied(), Some(cy - hh));
    assert_eq!(ys.last().copied(), Some(cy + hh));
    assert!(mesh.positions.iter().all(|p| p[2] == cz));
}

#[test]
fn an_invalid_proxy_framing_refuses_before_allocation() {
    for (center, size) in [
        ([f32::NAN, 0.0, 0.0], [1.0, 1.0]),
        ([0.0; 3], [0.0, 1.0]),
        ([0.0; 3], [1.0, -1.0]),
    ] {
        let err = raa_lit::billboard_proxy(center, size).err().unwrap();
        assert_eq!(code_of(&err), BILLBOARD_INVALID);
    }
}

#[test]
fn only_a_ready_bake_produces_proxy_metadata() {
    let mut store = BillboardBakeStore::new();
    let pending = store.begin(10);
    assert!(
        store.ready(pending).is_none(),
        "pending has no proxy framing"
    );
    assert_eq!(
        store.complete(pending, BillboardBake::from_result(pending, result())),
        Completion::Published
    );
    let ready = store.ready(pending).expect("ready bake");
    let mesh = raa_lit::billboard_proxy(ready.center, ready.size).expect("from the ready bake");
    assert_eq!(mesh.positions.len(), 4);
    assert_eq!(ready.center, [1.0, 2.0, 3.0]);
    assert_eq!(ready.size, [4.0, 5.0]);

    let failed = store.begin(10);
    store.fail(failed, "bake failed");
    assert!(
        store.ready(failed).is_none(),
        "a failed bake has no proxy framing"
    );
}

#[test]
fn the_state_guard_restores_on_success_error_and_cancellation() {
    use std::cell::RefCell;
    use std::rc::Rc;

    // Success: the guard drops at the end of its scope.
    let cell: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(Some("engine".into())));
    {
        let guard = StateGuard::take(Rc::clone(&cell)).expect("state taken");
        assert!(
            cell.borrow().is_none(),
            "the cell is empty during the await"
        );
        assert_eq!(guard.get().map(String::as_str), Some("engine"));
    }
    assert_eq!(cell.borrow().as_deref(), Some("engine"));

    // Error: an early return still drops the guard.
    fn failing(cell: Rc<RefCell<Option<String>>>) -> Result<(), &'static str> {
        let _guard = StateGuard::take(cell).expect("state taken");
        Err("bake failed")
    }
    let cell: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(Some("engine".into())));
    assert_eq!(failing(Rc::clone(&cell)), Err("bake failed"));
    assert_eq!(cell.borrow().as_deref(), Some("engine"));

    // Cancellation: dropping the future drops the guard without any completion path.
    let cell: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(Some("engine".into())));
    let guard = StateGuard::take(Rc::clone(&cell)).expect("state taken");
    drop(guard);
    assert_eq!(cell.borrow().as_deref(), Some("engine"));

    // An empty cell is refused, never fabricated.
    let empty: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(None));
    assert!(StateGuard::take(empty).is_none());
}

#[test]
fn the_state_guard_never_overwrites_newer_state() {
    use std::cell::RefCell;
    use std::rc::Rc;

    let cell: Rc<RefCell<Option<String>>> = Rc::new(RefCell::new(Some("old".into())));
    let guard = StateGuard::take(Rc::clone(&cell)).expect("state taken");
    *cell.borrow_mut() = Some("new".into());
    drop(guard);
    assert_eq!(cell.borrow().as_deref(), Some("new"));
}

#[test]
fn the_async_bake_uses_the_safe_engine_context_and_shared_upload() {
    let src = include_str!("../src/api/m37_billboard_bake.rs");
    assert!(
        src.contains("bake_context("),
        "the bake uses Engine::bake_context"
    );
    assert!(
        src.contains("st.upload_scratch(gpu)"),
        "the proxy uses the shared upload"
    );
    assert!(!src.contains("self.runtime"), "never reads Engine.runtime");
    assert!(
        !src.contains("Rc::clone(&self.state)") && !src.contains("self.state.borrow"),
        "never reads Engine.state directly"
    );
    assert!(
        !src.contains("AttributedMesh::from_mesh"),
        "the duplicated upload path is gone"
    );

    let engine = include_str!("../src/engine.rs");
    assert!(
        engine.contains("pub(crate) fn upload_scratch"),
        "upload_scratch is crate-visible for the proxy"
    );
    assert!(
        engine.contains("pub(crate) fn bake_context"),
        "bake_context is crate-visible for the async bake"
    );
    let upload = &engine[engine.find("pub(crate) fn upload_scratch").unwrap()..];
    let upload = &upload[..upload.find("\n    }").unwrap_or(upload.len())];
    assert!(
        upload.contains("self.scratch = mesh.base;"),
        "failure restores scratch"
    );
    assert!(
        upload.contains("retain_mesh_source"),
        "the mesh source is retained"
    );
}
