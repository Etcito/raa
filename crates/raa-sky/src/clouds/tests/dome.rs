//! The world-space cloud dome: direction mapping, lifecycle, uniform record, settings.

use super::super::dome::{DOME_BELOW, DomeState, dome_direction, dome_record, dome_uv};
use super::super::history::FREEZE_FRAMES;
use super::test_frame;
use crate::weather::WeatherSettings;

fn length(d: [f32; 3]) -> f32 {
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

#[test]
fn dome_mapping_round_trips_every_texel_and_reaches_below_the_horizon() {
    let size = 64_u32;
    let mut lowest = f32::MAX;
    let mut highest = f32::MIN;
    for j in 0..size {
        for i in 0..size {
            let uv = [(i as f32 + 0.5) / size as f32, (j as f32 + 0.5) / size as f32];
            let d = dome_direction(uv);
            assert!((length(d) - 1.0).abs() < 1.0e-5, "unit direction at {i},{j}");
            assert!(d[1] >= -DOME_BELOW - 1.0e-5);
            lowest = lowest.min(d[1]);
            highest = highest.max(d[1]);
            let back = dome_uv(d);
            assert!(
                (back[0] - uv[0]).abs() < 2.0e-4 && (back[1] - uv[1]).abs() < 2.0e-4,
                "uv {uv:?} -> {d:?} -> {back:?}"
            );
        }
    }
    assert!(lowest < -0.03, "the dome reaches below the horizon: {lowest}");
    assert!(highest > 0.999, "the zenith has a texel: {highest}");
    // The zenith is the centre of the map; a direction below the dome clamps inside [0, 1].
    let up = dome_uv([0.0, 1.0, 0.0]);
    assert!((up[0] - 0.5).abs() < 1.0e-6 && (up[1] - 0.5).abs() < 1.0e-6);
    for d in [[0.0, -1.0, 0.0], [1.0, -0.5, 0.0], [0.0, -0.2, 1.0]] {
        let uv = dome_uv(d);
        assert!((0.0..=1.0).contains(&uv[0]) && (0.0..=1.0).contains(&uv[1]), "{d:?} {uv:?}");
    }
}

#[test]
fn dome_accumulates_by_count_and_freezes_after_a_still_sky_without_a_camera() {
    let mut dome = DomeState::default();
    let mut last = 0;
    for invocation in 1..=u64::from(FREEZE_FRAMES) + 2 {
        let plan = dome.claim(invocation, 0, 7, false, false).expect("claimed");
        assert_eq!(plan.count, last + 1, "invocation {invocation}");
        assert_eq!(plan.frozen, plan.count >= FREEZE_FRAMES);
        // A second view of the same invocation never marches the dome again.
        assert_eq!(dome.claim(invocation, 1, 7, false, false), None);
        assert_eq!(dome.record(invocation, 1), None);
        assert_eq!(dome.record(invocation, 0), Some(plan));
        assert_eq!(dome.record(invocation, 0), None, "one march per invocation");
        // Nothing is committed before the submission is acknowledged.
        assert!(invocation == 1 && !dome.has_committed() || dome.has_committed());
        dome.submitted(0, invocation);
        last = plan.count;
    }
    assert_eq!(dome.observation(), (FREEZE_FRAMES + 2, true, 0));
    // An animated sky never freezes; a changed signature restarts the count.
    assert!(!dome.claim(900, 0, 7, true, false).expect("claimed").frozen);
    assert_eq!(dome.claim(901, 0, 8, false, false).expect("claimed").count, 1);
    dome.reset();
    assert!(!dome.has_committed() && dome.observation() == (0, false, 0));
}

#[test]
fn dome_record_ignores_the_camera_and_carries_count_flags_and_samples() {
    let frame = test_frame(0, 300, (1280, 720));
    let plan = dome_still_plan(&mut DomeState::default(), &frame, 5);
    let bytes = dome_record(&frame, &plan, 0);
    let f = |at: usize| f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
    assert_eq!(f(108), 5.0, "forward.w is the dome count");
    assert_eq!([f(112), f(116), f(124)], [1.0, 8.0, 56.0], "scale 1, noise size 1, samples");
    assert_eq!(f(120), 5.0, "a still dome walks the sequence by its count");
    let moved = crate::cloud_backend::CloudFrame {
        camera: raa_scene::Camera {
            eye: raa_math::Vec3::new(500.0, 40.0, -300.0),
            ..raa_scene::Camera::default()
        },
        ..frame
    };
    assert_eq!(dome_record(&moved, &plan, 0), bytes, "the record never sees the camera");
}

fn dome_still_plan(
    dome: &mut DomeState,
    frame: &crate::cloud_backend::CloudFrame,
    count: u32,
) -> super::super::dome::DomePlan {
    let mut plan = None;
    for invocation in 1..=u64::from(count) {
        plan = dome.claim(invocation, 0, frame.signature, false, false);
        dome.record(invocation, 0);
        dome.submitted(0, invocation);
    }
    plan.expect("claimed")
}

#[test]
fn cloud_dome_settings_default_off_and_are_ranged() {
    let defaults = WeatherSettings::default();
    assert!(!defaults.cloud_dome);
    assert_eq!(defaults.cloud_dome_size, 1024);
    for ok in [128_u32, 1024, 2048] {
        WeatherSettings { cloud_dome_size: ok, ..defaults }.validate().unwrap();
    }
    for bad in [0_u32, 127, 2049, u32::MAX] {
        let error = WeatherSettings { cloud_dome_size: bad, ..defaults }
            .validate()
            .expect_err("out of range");
        assert!(format!("{error:?}").contains("cloud_dome_size"), "{error:?}");
    }
}

#[test]
fn a_changed_sky_converges_inside_the_frame_with_a_bounded_burst() {
    use super::super::history::BURST_FRAMES;
    let frame = test_frame(0, 300, (1280, 720));
    let mut dome = DomeState::default();
    let plan = dome.claim(1, 0, 7, false, true).expect("claimed");
    assert_eq!((plan.burst, plan.count), (BURST_FRAMES, BURST_FRAMES));
    // Slot k of the burst is march k + 1 of a restarted mean, ending at the plan count.
    let count_at = |slot: u32| {
        f32::from_le_bytes(dome_record(&frame, &plan, slot)[108..112].try_into().unwrap())
    };
    assert_eq!([count_at(0), count_at(1), count_at(BURST_FRAMES - 1)], [1.0, 2.0, BURST_FRAMES as f32]);
    dome.record(1, 0);
    dome.submitted(0, 1);
    assert_eq!(dome.observation(), (BURST_FRAMES, false, 1));
    // Unchanged: one march per frame, no burst; dither off: one deterministic march is final.
    let next = dome.claim(2, 0, 7, false, true).expect("claimed");
    assert_eq!((next.burst, next.count), (1, BURST_FRAMES + 1));
    assert_eq!(dome.claim(3, 0, 9, false, false).expect("claimed").burst, 1);
}
