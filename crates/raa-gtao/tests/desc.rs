//! Q12: `AoDesc::validate` accepts the default and refuses every field and bound with
//! `ao.invalid`, naming the field in the message; NaN and infinities are refused.

use raa_gtao::AoDesc;

#[test]
fn default_is_valid() {
    AoDesc::default().validate().expect("default");
}

#[test]
fn valid_boundary_values_are_accepted() {
    // radius_m (0, 10] closed at the top, intensity [0, 1] closed, power (0, 8] closed.
    for desc in [
        AoDesc {
            radius_m: 10.0,
            intensity: 0.0,
            power: 8.0,
        },
        AoDesc {
            radius_m: f32::MIN_POSITIVE,
            intensity: 1.0,
            power: f32::MIN_POSITIVE,
        },
    ] {
        desc.validate().unwrap_or_else(|e| panic!("{desc:?}: {e}"));
    }
}

#[test]
fn each_field_and_bound_is_refused() {
    type Edit = fn(&mut AoDesc);
    let cases: [(&str, Edit); 6] = [
        // radius_m in (0, 10]: 0 is excluded, > 10 refused.
        ("radius_m", |d| d.radius_m = 0.0),
        ("radius_m", |d| d.radius_m = 10.0001),
        // intensity in [0, 1]: below and above refused.
        ("intensity", |d| d.intensity = -0.001),
        ("intensity", |d| d.intensity = 1.0001),
        // power in (0, 8]: 0 is excluded, > 8 refused.
        ("power", |d| d.power = 0.0),
        ("power", |d| d.power = 8.0001),
    ];
    for (field, edit) in cases {
        let mut desc = AoDesc::default();
        edit(&mut desc);
        let err = desc.validate().expect_err(field);
        assert_eq!(err.code(), "ao.invalid", "{field}");
        let message = err.message();
        assert!(message.contains(field), "{field}: {message}");
    }
}

#[test]
fn nan_and_infinity_are_refused() {
    let non_finite = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
    for v in non_finite {
        for (field, desc) in [
            (
                "radius_m",
                AoDesc {
                    radius_m: v,
                    ..AoDesc::default()
                },
            ),
            (
                "intensity",
                AoDesc {
                    intensity: v,
                    ..AoDesc::default()
                },
            ),
            (
                "power",
                AoDesc {
                    power: v,
                    ..AoDesc::default()
                },
            ),
        ] {
            let err = desc.validate().expect_err(field);
            assert_eq!(err.code(), "ao.invalid", "{field} {v}");
            assert!(
                err.message().contains(field),
                "{field} {v}: {}",
                err.message()
            );
        }
    }
}
