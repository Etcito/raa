use raa_core::RaaError;
use raa_deform::{JiggleConfig, JiggleSprings};
use raa_math::{Mat4, Quat, Vec3};

const DT: f32 = 1.0 / 60.0;
const TAG: u32 = 7;

fn at(x: f32) -> Mat4 {
    Mat4::from_trs(Vec3::new(x, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE)
}

fn length(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn bits(v: [f32; 3]) -> [u32; 3] {
    v.map(f32::to_bits)
}

#[test]
fn settles_to_zero_after_the_node_stops() {
    let mut springs = JiggleSprings::new();
    springs.set(TAG, Some(JiggleConfig::default()));

    let mut x = 0.0;
    springs.step(DT, TAG, &at(x));
    for _ in 0..30 {
        x += 2.0 * DT;
        let out = springs.step(DT, TAG, &at(x));
        assert!(length(out) <= 0.5 + 1e-6, "clamp: {out:?}");
    }

    let mut last = [0.0; 3];
    for _ in 0..180 {
        last = springs.step(DT, TAG, &at(x));
    }
    assert!(length(last) < 1e-4, "after 3 s: {last:?}");
}

#[test]
fn offset_stays_within_half_the_height_while_moving() {
    let config = JiggleConfig {
        height_m: 0.4,
        ..JiggleConfig::default()
    };
    let mut springs = JiggleSprings::new();
    springs.set(TAG, Some(config));

    let mut x = 0.0;
    springs.step(DT, TAG, &at(x));
    for _ in 0..60 {
        x += 2.0 * DT;
        let out = springs.step(DT, TAG, &at(x));
        assert!(length(out) <= 0.5 * 0.4 + 1e-6, "clamp: {out:?}");
    }
}

#[test]
fn offset_is_expressed_in_node_local_axes() {
    let rotation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
    let rotated = |x: f32| Mat4::from_trs(Vec3::new(x, 0.0, 0.0), rotation, Vec3::ONE);

    let mut world_springs = JiggleSprings::new();
    let mut local_springs = JiggleSprings::new();
    world_springs.set(TAG, Some(JiggleConfig::default()));
    local_springs.set(TAG, Some(JiggleConfig::default()));

    let mut x = 0.0;
    world_springs.step(DT, TAG, &at(x));
    local_springs.step(DT, TAG, &rotated(x));
    for _ in 0..20 {
        x += 2.0 * DT;
        let world = world_springs.step(DT, TAG, &at(x));
        let local = local_springs.step(DT, TAG, &rotated(x));
        assert!(local[0].abs() < 1e-5, "x axis leaks: {local:?}");
        assert!(local[1].abs() < 1e-5, "y axis leaks: {local:?}");
        assert!(
            (local[2].abs() - world[0].abs()).abs() < 1e-5,
            "world {world:?} local {local:?}"
        );
    }
}

#[test]
fn identical_inputs_give_bit_identical_outputs() {
    let dts = [DT, 0.0, 1.0 / 120.0, 1.0 / 30.0, 0.0, 1.0 / 144.0, DT, DT];
    let config = JiggleConfig {
        stiffness: 90.0,
        damping: 0.2,
        amount: 0.8,
        height_m: 0.9,
    };
    let mut a = JiggleSprings::new();
    let mut b = JiggleSprings::new();
    a.set(3, Some(config));
    b.set(3, Some(config));
    a.set(1, Some(JiggleConfig::default()));
    b.set(1, Some(JiggleConfig::default()));

    let mut x = 0.0_f32;
    for frame in 0..200 {
        x += 0.05 * (frame as f32 * 0.3).sin();
        let dt = dts[frame % dts.len()];
        for tag in [3, 1] {
            let out_a = a.step(dt, tag, &at(x));
            let out_b = b.step(dt, tag, &at(x));
            assert_eq!(bits(out_a), bits(out_b), "frame {frame} tag {tag}");
        }
    }
}

#[test]
fn zero_dt_leaves_the_offset_unchanged() {
    let mut springs = JiggleSprings::new();
    springs.set(TAG, Some(JiggleConfig::default()));

    springs.step(DT, TAG, &at(0.0));
    let mut x = 0.0;
    for _ in 0..10 {
        x += 0.05;
        springs.step(DT, TAG, &at(x));
    }
    let before = springs.step(0.0, TAG, &at(x));
    assert!(length(before) > 0.0);
    assert_eq!(bits(springs.step(0.0, TAG, &at(x + 1.0))), bits(before));
    assert_eq!(
        bits(springs.step(f32::NAN, TAG, &at(x + 1.0))),
        bits(before)
    );
    assert_eq!(bits(springs.step(-DT, TAG, &at(x + 1.0))), bits(before));
}

#[test]
fn validate_refuses_each_bad_field() {
    let base = JiggleConfig::default();
    let cases = [
        (
            "stiffness",
            JiggleConfig {
                stiffness: 0.0,
                ..base
            },
        ),
        (
            "stiffness",
            JiggleConfig {
                stiffness: -1.0,
                ..base
            },
        ),
        (
            "stiffness",
            JiggleConfig {
                stiffness: f32::NAN,
                ..base
            },
        ),
        (
            "damping",
            JiggleConfig {
                damping: -0.1,
                ..base
            },
        ),
        (
            "damping",
            JiggleConfig {
                damping: 2.5,
                ..base
            },
        ),
        (
            "damping",
            JiggleConfig {
                damping: f32::INFINITY,
                ..base
            },
        ),
        (
            "amount",
            JiggleConfig {
                amount: -0.1,
                ..base
            },
        ),
        (
            "amount",
            JiggleConfig {
                amount: f32::NAN,
                ..base
            },
        ),
        (
            "height_m",
            JiggleConfig {
                height_m: 0.0,
                ..base
            },
        ),
        (
            "height_m",
            JiggleConfig {
                height_m: -1.0,
                ..base
            },
        ),
        (
            "height_m",
            JiggleConfig {
                height_m: f32::INFINITY,
                ..base
            },
        ),
    ];
    for (field, config) in cases {
        match config.validate() {
            Err(err @ RaaError::FrameGraph { .. }) => {
                assert_eq!(err.code(), "deform.invalid");
                assert!(err.message().contains(field), "{field}: {err}");
            }
            other => panic!("{field} {config:?}: expected refusal, got {other:?}"),
        }
    }
    assert!(base.validate().is_ok());
    assert!(
        JiggleConfig {
            damping: 0.0,
            ..base
        }
        .validate()
        .is_ok()
    );
    assert!(
        JiggleConfig {
            damping: 2.0,
            amount: 0.0,
            ..base
        }
        .validate()
        .is_ok()
    );
}

#[test]
fn removing_a_tag_stops_its_offset() {
    let mut springs = JiggleSprings::new();
    springs.set(TAG, Some(JiggleConfig::default()));

    springs.step(DT, TAG, &at(0.0));
    let mut x = 0.0;
    for _ in 0..10 {
        x += 0.05;
        springs.step(DT, TAG, &at(x));
    }
    assert!(length(springs.step(DT, TAG, &at(x))) > 0.0);

    springs.set(TAG, None);
    assert_eq!(springs.step(DT, TAG, &at(x + 1.0)), [0.0; 3]);
    assert_eq!(springs.step(DT, 999, &at(x)), [0.0; 3]);
}
