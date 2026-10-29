//! m21 views (S25, S26): the registry exports the 5 methods; viewport rect, index, resize
//! and overlay validation run natively on the pure part of `src/api/m21_views.rs`.

use std::fs;
use std::path::Path;

#[path = "../src/api/m21_views.rs"]
mod m21_views;

use m21_views::{
    ViewCamera, ViewRect, ViewsState, backing_size, validate_overlay, validate_viewports,
};

fn rect(x: f64, y: f64, width: f64, height: f64) -> ViewRect {
    ViewRect {
        x,
        y,
        width,
        height,
    }
}

fn quarters() -> Vec<ViewRect> {
    vec![
        rect(0.0, 0.0, 0.5, 0.5),
        rect(0.5, 0.0, 0.5, 0.5),
        rect(0.0, 0.5, 0.5, 0.5),
        rect(0.5, 0.5, 0.5, 0.5),
    ]
}

fn code(r: Result<impl std::fmt::Debug, raa_core::RaaError>) -> &'static str {
    r.expect_err("refused").code()
}

fn camera() -> ViewCamera {
    ViewCamera {
        eye: [0.0, 5.0, -10.0],
        target: [0.0, 0.0, 0.0],
        fov_y_deg: 50.0,
        near: 0.1,
        far: 200.0,
    }
}

#[test]
fn registry_exports_the_five_methods() {
    let listed = fs::read_to_string(Path::new(env!("OUT_DIR")).join("api_files.txt"))
        .expect("api_files.txt");
    assert!(
        listed.lines().any(|l| l == "m21_views"),
        "m21_views not registered"
    );
    let reg = fs::read_to_string(Path::new(env!("OUT_DIR")).join("api_registry.rs"))
        .expect("api_registry.rs");
    assert!(reg.contains("m21_views::install(ctx)?;"));
    assert!(reg.contains("m21_views::frame(st, gpu, input)?;"));
    let src = include_str!("../src/api/m21_views.rs");
    for name in [
        "setViewports",
        "setViewportCamera",
        "setOverlayLines",
        "clearOverlay",
    ] {
        assert!(src.contains(&format!("js_name = {name})]")), "no {name}");
    }
    assert!(src.contains("pub fn resize("), "no resize");
    // engine.rs keeps no second `resize` export (one JS name per method).
    let engine = include_str!("../src/engine.rs");
    assert!(!engine.contains("pub fn resize("));
}

#[test]
fn rect_validation_refuses_with_codes() {
    assert_eq!(code(validate_viewports(&[])), "views.count");
    let mut five = quarters();
    five.push(rect(0.0, 0.0, 0.1, 0.1));
    assert_eq!(code(validate_viewports(&five)), "views.count");
    let overlap = [rect(0.0, 0.0, 0.6, 1.0), rect(0.5, 0.0, 0.5, 1.0)];
    assert_eq!(code(validate_viewports(&overlap)), "views.rect");
    for bad in [
        rect(0.6, 0.0, 0.5, 1.0),
        rect(-0.1, 0.0, 0.5, 1.0),
        rect(0.0, 0.0, 1.5, 1.0),
        rect(0.0, 0.0, f64::NAN, 1.0),
        rect(0.0, 0.0, 0.0, 1.0),
    ] {
        assert_eq!(code(validate_viewports(&[bad])), "views.rect", "{bad:?}");
    }
    validate_viewports(&quarters()).expect("4 quarters");
    validate_viewports(&[ViewRect::FULL]).expect("single view");
    let thirds = [
        rect(0.0, 0.0, 1.0 / 3.0, 1.0),
        rect(1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0),
    ];
    validate_viewports(&thirds).expect("touching rects");
}

#[test]
fn state_defaults_to_one_full_view_and_checks_indices() {
    let mut s = ViewsState::default();
    assert_eq!(s.viewports(), &[ViewRect::FULL]);
    assert_eq!(code(s.set_camera(1.0, camera())), "views.index");
    let before = s.clone();
    assert_eq!(code(s.set_viewports(&[])), "views.count");
    assert_eq!(s, before, "a refusal changes nothing");
    s.set_viewports(&quarters()).expect("quarters");
    assert_eq!(s.set_camera(0.0, camera()).expect("viewport 0"), 0);
    assert!(s.camera(0).is_none(), "viewport 0 is the scene camera");
    assert_eq!(s.set_camera(3.0, camera()).expect("viewport 3"), 3);
    assert_eq!(s.camera(3), Some(&camera()));
    for bad in [4.0, -1.0, 1.5, f64::NAN] {
        assert_eq!(code(s.set_camera(bad, camera())), "views.index", "{bad}");
    }
    let mut c = camera();
    c.target = c.eye;
    assert_eq!(code(s.set_camera(1.0, c)), "api.invalid_argument");
}

#[test]
fn resize_caps_dpr_and_keeps_the_two_argument_call() {
    assert_eq!(backing_size(800.0, 600.0, None).unwrap(), (800, 600));
    assert_eq!(backing_size(800.0, 600.0, Some(1.5)).unwrap(), (1200, 900));
    assert_eq!(backing_size(800.0, 600.0, Some(3.0)).unwrap(), (1600, 1200));
    assert_eq!(backing_size(333.3, 0.0, Some(1.0)).unwrap(), (333, 1));
    assert_eq!(
        code(backing_size(800.0, 600.0, Some(0.0))),
        "api.invalid_argument"
    );
    assert_eq!(
        code(backing_size(f64::NAN, 600.0, None)),
        "api.invalid_argument"
    );
}

#[test]
fn overlay_length_and_layer_validation() {
    assert_eq!(validate_overlay(0.0, 6, 8).unwrap(), 0);
    assert_eq!(validate_overlay(7.0, 12, 16).unwrap(), 7);
    validate_overlay(3.0, 0, 0).expect("empty layer");
    assert_eq!(code(validate_overlay(0.0, 3, 4)), "overlay.length");
    assert_eq!(code(validate_overlay(0.0, 7, 8)), "overlay.length");
    assert_eq!(code(validate_overlay(0.0, 6, 6)), "overlay.length");
    assert_eq!(code(validate_overlay(0.0, 6, 12)), "overlay.length");
    for bad in [8.0, -1.0, 0.5, f64::NAN] {
        assert_eq!(code(validate_overlay(bad, 6, 8)), "overlay.layer", "{bad}");
    }

    let mut s = ViewsState::default();
    let pos = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let col = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    s.set_overlay_lines(2.0, &pos, &col, true, 2.0)
        .expect("one segment");
    assert_eq!(s.overlays()[2].positions, pos);
    assert_eq!(s.overlays()[2].colors, col);
    assert!(s.overlays()[2].depth_test);
    assert_eq!(
        code(s.set_overlay_lines(2.0, &pos[..3], &col, true, 2.0)),
        "overlay.length"
    );
    assert_eq!(s.overlays()[2].positions, pos, "a refusal keeps the layer");
    assert_eq!(code(s.clear_overlay(8.0)), "overlay.layer");
    s.clear_overlay(2.0).expect("clear");
    assert!(s.overlays()[2].is_empty());
}
