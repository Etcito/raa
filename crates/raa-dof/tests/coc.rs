//! Q12: `DofDesc::validate` accepts the default and refuses every invalid field with
//! `post.invalid`, naming the field. CoC contract: zero at focus, signed, grows as the
//! aperture widens, and the clamp scales with the viewport height.

use raa_dof::{DofDesc, coc_px};

#[test]
fn default_is_dof_values_and_valid() {
    let d = DofDesc::default();
    assert_eq!(
        (
            d.focus_distance_m,
            d.f_stop,
            d.focal_length_mm,
            d.autofocus,
            d.max_coc_px
        ),
        (10.0, 2.8, 50.0, None, 24.0)
    );
    d.validate().expect("default");
}

#[test]
fn valid_boundary_values_are_accepted() {
    for desc in [
        DofDesc {
            focus_distance_m: 0.0081,
            f_stop: 0.7,
            focal_length_mm: 8.0,
            autofocus: Some([0.0, 0.0]),
            max_coc_px: f32::MIN_POSITIVE,
        },
        DofDesc {
            focus_distance_m: 10.0,
            f_stop: 64.0,
            focal_length_mm: 600.0,
            autofocus: Some([1.0, 1.0]),
            max_coc_px: 64.0,
        },
    ] {
        desc.validate().unwrap_or_else(|e| panic!("{desc:?}: {e}"));
    }
}

#[test]
fn each_invalid_field_is_refused_with_post_invalid() {
    type Edit = fn(&mut DofDesc);
    let cases: [(&str, Edit); 12] = [
        // focus_distance_m must be greater than the focal length in metres.
        ("focus_distance_m", |d| d.focus_distance_m = 0.05),
        ("focus_distance_m", |d| d.focus_distance_m = 0.0),
        // f_stop in [0.7, 64].
        ("f_stop", |d| d.f_stop = 0.69),
        ("f_stop", |d| d.f_stop = 64.1),
        // focal_length_mm in [8, 600].
        ("focal_length_mm", |d| d.focal_length_mm = 7.9),
        ("focal_length_mm", |d| d.focal_length_mm = 600.1),
        // max_coc_px in (0, 64].
        ("max_coc_px", |d| d.max_coc_px = 0.0),
        ("max_coc_px", |d| d.max_coc_px = 64.1),
        // autofocus uv in [0, 1].
        ("autofocus", |d| d.autofocus = Some([-0.01, 0.5])),
        ("autofocus", |d| d.autofocus = Some([0.5, 1.01])),
        // A focus inside the lens is also refused.
        ("focus_distance_m", |d| d.focus_distance_m = 0.001),
        ("focal_length_mm", |d| d.focal_length_mm = 700.0),
    ];
    for (field, edit) in cases {
        let mut desc = DofDesc::default();
        edit(&mut desc);
        let err = desc.validate().expect_err(field);
        assert_eq!(err.code(), "post.invalid", "{field}");
        assert!(err.message().contains(field), "{field}: {err}");
    }
}

#[test]
fn non_finite_values_are_refused() {
    for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for (field, desc) in [
            (
                "focus_distance_m",
                DofDesc {
                    focus_distance_m: v,
                    ..DofDesc::default()
                },
            ),
            (
                "f_stop",
                DofDesc {
                    f_stop: v,
                    ..DofDesc::default()
                },
            ),
            (
                "focal_length_mm",
                DofDesc {
                    focal_length_mm: v,
                    ..DofDesc::default()
                },
            ),
            (
                "max_coc_px",
                DofDesc {
                    max_coc_px: v,
                    ..DofDesc::default()
                },
            ),
            (
                "autofocus",
                DofDesc {
                    autofocus: Some([v, 0.5]),
                    ..DofDesc::default()
                },
            ),
        ] {
            let err = desc.validate().expect_err(field);
            assert_eq!(err.code(), "post.invalid", "{field} {v}");
            assert!(err.message().contains(field), "{field} {v}: {err}");
        }
    }
}

#[test]
fn zero_at_the_focus_distance() {
    for desc in [
        DofDesc::default(),
        DofDesc {
            focus_distance_m: 2.5,
            f_stop: 1.4,
            focal_length_mm: 85.0,
            ..DofDesc::default()
        },
    ] {
        assert_eq!(coc_px(desc.focus_distance_m, &desc, 720.0), 0.0);
    }
}

#[test]
fn negative_in_front_and_positive_behind() {
    let d = DofDesc::default();
    assert!(coc_px(5.0, &d, 720.0) < 0.0);
    assert!(coc_px(20.0, &d, 720.0) > 0.0);
}

#[test]
fn wider_aperture_grows_the_coc() {
    let d = DofDesc::default();
    for z in [5.0, 20.0] {
        let narrow = coc_px(z, &DofDesc { f_stop: 5.6, ..d }, 720.0).abs();
        let mid = coc_px(z, &DofDesc { f_stop: 2.8, ..d }, 720.0).abs();
        let wide = coc_px(z, &DofDesc { f_stop: 1.4, ..d }, 720.0).abs();
        assert!(narrow > 0.0, "z {z}");
        assert!(mid > narrow && wide > mid, "z {z}: {narrow} {mid} {wide}");
        assert!((wide / mid - 2.0).abs() < 1e-4, "z {z}: {}", wide / mid);
    }
}

#[test]
fn sensor_mapping_matches_the_thin_lens() {
    let d = DofDesc::default();
    // f 50 mm, N 2.8, focus 10 m, z 5 m: coc_mm = A*f*|z-s|/(z*(s-f)) = 0.089733 mm,
    // px = coc_mm / 36 * 720 = 1.79466.
    assert!((coc_px(5.0, &d, 720.0) + 1.79466).abs() < 1e-3);
    assert!((coc_px(5.0, &d, 1440.0) + 2.0 * 1.79466).abs() < 2e-3);
}

#[test]
fn clamp_scales_with_viewport_height() {
    let d = DofDesc::default();
    // In front of the focus plane the unclamped CoC is about 34 px at 720p.
    assert!((coc_px(0.5, &d, 720.0) + 24.0).abs() < 1e-4);
    assert!((coc_px(0.5, &d, 1440.0) + 48.0).abs() < 1e-4);

    // Behind it the same clamp applies; a near focus reaches it too.
    let near = DofDesc {
        focus_distance_m: 0.1,
        ..d
    };
    assert!((coc_px(1000.0, &near, 720.0) - 24.0).abs() < 1e-4);
    assert!((coc_px(1000.0, &near, 1440.0) - 48.0).abs() < 1e-4);
}

#[test]
fn non_positive_or_non_finite_depth_is_zero() {
    let d = DofDesc::default();
    for z in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(coc_px(z, &d, 720.0), 0.0, "z {z}");
    }
}
