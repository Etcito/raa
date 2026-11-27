//! CPU tests of the M7 foliage bake coordinator (m7-bake-service): routing, validation of
//! settings, source and material envelope, stale generations, failure cleanup and the
//! release order. No GPU work is executed here.

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::{MaterialId, RaaError};
use raa_lit::LitMeshId;
use raa_lit::foliage_bake::{
    FOLIAGE_DATA_MAGIC, FoliageBakeSettings, FoliageBakeState, FoliageRoute, Publication,
    foliage_route, retire_foliage, validate_material_data, validate_source,
};
use raa_lit::foliage_resources::{FoliageParams, FoliageSource, SourceTriangle};

fn code(e: RaaError) -> &'static str {
    match e {
        RaaError::FrameGraph { code, .. } => code,
        other => panic!("unexpected error {other:?}"),
    }
}

fn settings(mode: u32, views: u32) -> FoliageBakeSettings {
    FoliageBakeSettings {
        params: FoliageParams {
            min: [-1.0, 0.0, -1.0],
            max: [1.0, 3.0, 1.0],
            resolution: 32,
            mode,
            translucency: 0.35,
            generation: 7,
            cutoff: 0.5,
            view_count: views,
            texture_resolution: 128,
        },
        material: MaterialId(12),
        count: 4096,
        seed: 0,
        iterations: 64,
        silhouette_weight: 1.0,
        normal_weight: 1.0,
        edge_weight: 0.01,
    }
}

fn source(triangles: usize, generation: u32) -> FoliageSource {
    let t = SourceTriangle {
        positions: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        normals: [[0.0, 0.0, 1.0]; 3],
        uv: [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        material: 3,
        source_tag: 1,
        foliage: true,
    };
    FoliageSource {
        triangles: vec![t; triangles],
        min: [0.0; 3],
        max: [1.0, 1.0, 0.0],
        root_world: raa_math::Mat4::IDENTITY,
        generation,
    }
}

#[test]
fn modes_route_to_their_producers() {
    for mode in 0..=3 {
        assert_eq!(foliage_route(mode).unwrap(), FoliageRoute::Impostor);
        assert_eq!(
            settings(mode, 8).validate().unwrap(),
            FoliageRoute::Impostor
        );
    }
    assert_eq!(foliage_route(4).unwrap(), FoliageRoute::Stochastic);
    assert_eq!(foliage_route(5).unwrap(), FoliageRoute::Differentiable);
    assert_eq!(
        settings(5, 96).validate().unwrap(),
        FoliageRoute::Differentiable
    );
    assert_eq!(code(foliage_route(6).unwrap_err()), "foliage.invalid");
}

#[test]
fn invalid_settings_are_refused_before_allocation() {
    assert_eq!(
        code(settings(0, 96).validate().unwrap_err()),
        "foliage.invalid"
    );
    assert_eq!(
        code(settings(5, 8).validate().unwrap_err()),
        "foliage.invalid"
    );
    let mut s = settings(4, 1);
    s.count = 0;
    assert_eq!(code(s.validate().unwrap_err()), "foliage.invalid");
    let mut s = settings(4, 1);
    (s.silhouette_weight, s.normal_weight, s.edge_weight) = (0.0, 0.0, 0.0);
    assert_eq!(code(s.validate().unwrap_err()), "foliage.invalid");
}

#[test]
fn material_handle_is_preserved_into_optimizer_settings() {
    let s = settings(5, 96);
    let copy = s;
    assert_eq!(copy.material, MaterialId(12));
    let o = s.optimize_settings();
    assert_eq!((o.count, o.resolution, o.iterations), (4096, 32, 64));
}

#[test]
fn empty_and_mismatched_sources_are_refused() {
    let p = settings(4, 1).params;
    assert_eq!(
        code(validate_source(&source(0, 7), &p).unwrap_err()),
        "foliage.invalid"
    );
    assert_eq!(
        code(validate_source(&source(2, 6), &p).unwrap_err()),
        "foliage.invalid"
    );
    validate_source(&source(2, 7), &p).unwrap();
}

#[test]
fn material_envelope_is_validated() {
    let good = [FOLIAGE_DATA_MAGIC, 1, 0b1011, 2, 10, 20];
    assert_eq!(validate_material_data(Some(&good)).unwrap(), 0b1011);
    assert_eq!(
        code(validate_material_data(None).unwrap_err()),
        "foliage.invalid"
    );
    for bad in [
        [0, 1, 0, 0],
        [FOLIAGE_DATA_MAGIC, 2, 0, 0],
        [FOLIAGE_DATA_MAGIC, 1, 0x10, 0],
        [FOLIAGE_DATA_MAGIC, 1, 0, 1],
    ] {
        assert_eq!(
            code(validate_material_data(Some(&bad)).unwrap_err()),
            "foliage.invalid"
        );
    }
}

#[test]
fn only_the_newest_generation_is_current() {
    let mut state = FoliageBakeState::default();
    let m = MaterialId(12);
    assert!(!state.is_current(m, 1));
    state.begin(m, 1);
    assert!(state.is_current(m, 1));
    state.begin(m, 2);
    assert!(!state.is_current(m, 1));
    assert!(state.is_current(m, 2));
    assert!(!state.is_current(MaterialId(13), 2));
}

#[test]
fn published_meshes_are_reported_once() {
    let mut state = FoliageBakeState::default();
    assert!(state.take_published().is_empty());
    state.publish(LitMeshId(4));
    assert_eq!(state.take_published(), vec![LitMeshId(4)]);
    assert!(state.take_published().is_empty());
}

#[test]
fn partial_publication_is_released_on_failure_only() {
    let released = Rc::new(RefCell::new(0));
    {
        let r = Rc::clone(&released);
        let _guard = Publication::new(move || *r.borrow_mut() += 1);
    }
    assert_eq!(
        *released.borrow(),
        1,
        "a failed or cancelled publication releases"
    );
    let r = Rc::clone(&released);
    Publication::new(move || *r.borrow_mut() += 1).complete();
    assert_eq!(
        *released.borrow(),
        1,
        "a completed publication keeps its allocation"
    );
}

#[test]
fn release_detaches_before_removing_and_dropping() {
    let log = Rc::new(RefCell::new(Vec::new()));
    let resources = Rc::new(());
    let weak = Rc::downgrade(&resources);
    let (a, b) = (Rc::clone(&log), Rc::clone(&log));
    let w = weak.clone();
    retire_foliage(
        resources,
        move || a.borrow_mut().push(("detach", w.upgrade().is_some())),
        move || b.borrow_mut().push(("mesh", true)),
    );
    assert_eq!(*log.borrow(), vec![("detach", true), ("mesh", true)]);
    assert!(weak.upgrade().is_none(), "the generation is dropped last");
}

#[test]
fn impostor_route_refuses_too_large_allocations_without_panicking() {
    use raa_lit::foliage_bake::check_impostor_allocation;
    let small = check_impostor_allocation(4).expect("a small source fits");
    assert!(small > 0);
    for triangles in [usize::MAX, usize::MAX / 128, isize::MAX as usize / 100] {
        let outcome = std::panic::catch_unwind(|| check_impostor_allocation(triangles));
        let err = outcome
            .expect("the impostor allocation check must not panic")
            .expect_err("a too-large impostor allocation must be refused");
        assert!(err.message().contains("bytes"), "{}", err.message());
        assert_eq!(code(err), "foliage.invalid");
    }
}
