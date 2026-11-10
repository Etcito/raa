use raa_core::RaaError;
use raa_math::Vec3;
use raa_sky::{CelestialSky, DominantLight, Place, TimeOfDay, TimeOfDaySettings, TimeOfDayState};
use std::f32::consts::FRAC_PI_2;

fn near(a: Vec3, b: Vec3, tolerance: f32) {
    assert!((a - b).length() < tolerance, "{a:?} != {b:?}");
}

fn refused(result: Result<(), RaaError>, field: &str) {
    let error = result.unwrap_err();
    assert_eq!(error.code(), "sky.invalid");
    assert!(error.message().contains(field), "{error}");
}

#[test]
fn equinox_sun_rises_east_peaks_overhead_and_sets_west() {
    let place = Place {
        latitude: 0.0,
        north: 0.0,
    };
    // The source's approximate March equinox is day 81.25.
    for (hours, expected) in [(6.0, Vec3::X), (12.0, Vec3::Y), (18.0, -Vec3::X)] {
        near(
            TimeOfDay { day: 81, hours }.sky(&place).sun,
            expected,
            0.002,
        );
    }
}

#[test]
fn latitude_north_and_celestial_basis_follow_scene_coordinates() {
    let time = TimeOfDay {
        day: 81,
        hours: 12.0,
    };
    for latitude in [-90.0_f32, -38.0, 0.0, 38.0, 90.0] {
        let place = Place {
            latitude,
            north: 0.0,
        };
        let sky = time.sky(&place);
        let angle = latitude.to_radians();
        near(sky.pole, Vec3::new(0.0, angle.sin(), -angle.cos()), 1e-6);
        assert!((sky.sun.y - angle.cos()).abs() < 0.002);
        let rotated = time.sky(&Place {
            north: FRAC_PI_2,
            ..place
        });
        for (a, b) in [
            (sky.sun, rotated.sun),
            (sky.moon, rotated.moon),
            (sky.pole, rotated.pole),
            (sky.equinox, rotated.equinox),
        ] {
            near(Vec3::new(a.z, a.y, -a.x), b, 1e-6);
            assert!(a.is_finite());
            assert!((a.length() - 1.0).abs() < 1e-5);
        }
        assert!(sky.pole.dot(sky.equinox).abs() < 1e-5);
        near(sky.celestial(0.0, 0.0), sky.equinox, 1e-6);
        near(sky.celestial(0.0, FRAC_PI_2), sky.pole, 1e-6);
        near(
            sky.celestial(FRAC_PI_2, 0.0),
            sky.pole.cross(sky.equinox),
            1e-6,
        );
    }
}

#[test]
fn moon_phase_tracks_new_quarter_and_full_positions() {
    let place = Place::default();
    for (age, lit, alignment) in [(0.0, 0.0, 1.0), (0.25, 0.5, 0.0), (0.5, 1.0, -1.0)] {
        let mut time = TimeOfDay { day: 6, hours: 0.0 };
        time.advance(age * 29.53059 * 24.0).unwrap();
        let sky = time.sky(&place);
        assert!((sky.moon_lit - lit).abs() < 1e-5);
        assert!((sky.sun.dot(sky.moon) - alignment).abs() < 0.01);
    }
}

#[test]
fn signed_advances_wrap_days_and_years_and_bound_extreme_deltas() {
    let mut time = TimeOfDay {
        day: 365,
        hours: 23.5,
    };
    for (delta, day, hours) in [
        (1.0, 1, 0.5),
        (-1.0, 365, 23.5),
        (8760.0 * 3.0 + 25.0, 2, 0.5),
        (-8760.0 * 3.0 - 25.0, 365, 23.5),
    ] {
        time.advance(delta).unwrap();
        assert_eq!(time, TimeOfDay { day, hours });
    }
    for delta in [f64::MAX, -f64::MAX, -f64::MIN_POSITIVE, 0.0] {
        time.advance(delta).unwrap();
        time.validate().unwrap();
    }
}

#[test]
fn invalid_inputs_refuse_with_fields_and_advance_is_atomic() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut time = TimeOfDay::default();
        let before = time;
        refused(time.advance(value), "advance.hours");
        assert_eq!(time, before);
        refused(
            TimeOfDay {
                hours: value,
                ..time
            }
            .validate(),
            "hours",
        );
        refused(
            TimeOfDaySettings {
                speed: value,
                ..Default::default()
            }
            .validate(),
            "speed",
        );
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        refused(
            Place {
                latitude: value,
                north: 0.0,
            }
            .validate(),
            "latitude",
        );
        refused(
            Place {
                north: value,
                ..Place::default()
            }
            .validate(),
            "north",
        );
    }
    for day in [0, 366, u16::MAX] {
        let mut time = TimeOfDay { day, hours: 9.0 };
        let before = time;
        refused(time.advance(1.0), "day");
        assert_eq!(time, before);
    }
    for hours in [-1.0, 24.0] {
        refused(TimeOfDay { day: 1, hours }.validate(), "hours");
    }
    for latitude in [-90.1, 90.1] {
        refused(
            Place {
                latitude,
                north: 0.0,
            }
            .validate(),
            "latitude",
        );
    }
    refused(
        TimeOfDaySettings {
            speed: -1.0,
            ..Default::default()
        }
        .validate(),
        "speed",
    );
}

#[test]
fn root_exports_defaults_and_extreme_finite_orientation_are_valid() {
    let state = TimeOfDayState::default();
    state.settings.validate().unwrap();
    assert_eq!(
        state.settings.time,
        TimeOfDay {
            day: 172,
            hours: 9.0
        }
    );
    assert_eq!(
        state.settings.place,
        Place {
            latitude: 38.0,
            north: 0.0
        }
    );
    assert_eq!(state.settings.speed, 1.0);
    assert!(!state.settings.enabled && !state.settings.playing);
    assert_eq!(state.current, state.settings.time);
    assert_eq!(state.dominant, DominantLight::Manual);
    for north in [-f32::MAX, f32::MAX] {
        let place = Place {
            north,
            ..Place::default()
        };
        place.validate().unwrap();
        let sky: CelestialSky = state.current.sky(&place);
        assert!(sky.sun.is_finite() && sky.moon.is_finite());
    }
}
