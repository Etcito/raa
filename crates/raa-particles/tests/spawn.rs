//! Proof `unit` (deterministic spawn): the CPU twin is a pure function of
//! `(seed, serial)`, stays inside the emitter's box, cone and ranges, and
//! matches the reference PCG-RXS-M-XS values.

use raa_particles::{EmitterDesc, ParticleKind, pcg_hash, spawn_sample};

const KINDS: [ParticleKind; 4] = [
    ParticleKind::Splash,
    ParticleKind::Snow,
    ParticleKind::Confetti,
    ParticleKind::Sparkles,
];

fn dims(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn unit(v: [f32; 3]) -> [f32; 3] {
    let inv = 1.0 / dims(v);
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[test]
fn same_seed_and_serial_give_identical_samples() {
    for kind in KINDS {
        let desc = EmitterDesc::for_kind(kind);
        for serial in [0, 1, 7, 255, 65_535] {
            let a = spawn_sample(&desc, serial);
            let b = spawn_sample(&desc, serial);
            assert_eq!(a, b, "{kind:?} serial {serial}");
        }
    }
}

#[test]
fn different_seeds_give_different_sequences() {
    let base = EmitterDesc::for_kind(ParticleKind::Confetti);
    let a = EmitterDesc {
        seed: 7,
        ..base.clone()
    };
    let b = EmitterDesc {
        seed: 8,
        ..base.clone()
    };
    let seq_a: Vec<_> = (0..64).map(|s| spawn_sample(&a, s)).collect();
    let seq_b: Vec<_> = (0..64).map(|s| spawn_sample(&b, s)).collect();
    assert_ne!(seq_a, seq_b, "two seeds produced the same 64 samples");
    let differing = seq_a.iter().zip(&seq_b).filter(|(x, y)| x != y).count();
    assert!(
        differing >= 60,
        "seed change only affected {differing} of 64 serials"
    );
}

#[test]
fn serial_changes_the_sample() {
    let desc = EmitterDesc::for_kind(ParticleKind::Snow);
    let a = spawn_sample(&desc, 3);
    let b = spawn_sample(&desc, 4);
    assert_ne!(a, b);
}

#[test]
fn samples_lie_inside_the_volume_box() {
    for kind in KINDS {
        let mut desc = EmitterDesc::for_kind(kind);
        desc.position = [1.5, -0.25, 0.75];
        desc.volume_half = [2.0, 0.5, 1.25];
        for serial in 0..512 {
            let s = spawn_sample(&desc, serial);
            for axis in 0..3 {
                let delta = s.position[axis] - desc.position[axis];
                assert!(
                    delta.abs() <= desc.volume_half[axis] + 1e-5,
                    "{kind:?} serial {serial}: axis {axis} at {delta} outside {}",
                    desc.volume_half[axis]
                );
            }
        }
    }
}

#[test]
fn samples_lie_inside_the_cone() {
    let base = EmitterDesc::for_kind(ParticleKind::Snow);
    for (direction, spread_rad) in [
        ([0.0, -1.0, 0.0], 0.2),
        ([0.3, -1.0, 0.2], 0.7),
        ([-1.0, 0.0, 0.0], 0.0),
        ([0.2, 0.4, -0.9], std::f32::consts::PI),
    ] {
        let desc = EmitterDesc {
            direction,
            spread_rad,
            ..base.clone()
        };
        let axis = unit(direction);
        let min_cos = spread_rad.cos() - 1e-4;
        for serial in 0..512 {
            let s = spawn_sample(&desc, serial);
            let speed = dims(s.velocity);
            assert!(speed > 0.0);
            let cos_angle = dot(unit(s.velocity), axis);
            assert!(
                cos_angle >= min_cos,
                "serial {serial}: cos {cos_angle} below {min_cos} for direction {direction:?}"
            );
            assert!(cos_angle <= 1.0 + 1e-4, "cos {cos_angle} > 1");
        }
    }
}

#[test]
fn speed_life_and_size_stay_in_their_ranges() {
    for kind in KINDS {
        let desc = EmitterDesc::for_kind(kind);
        for serial in 0..1024 {
            let s = spawn_sample(&desc, serial);
            let speed = dims(s.velocity);
            assert!(
                speed >= desc.speed[0] - 1e-4 && speed <= desc.speed[1] + 1e-4,
                "{kind:?} serial {serial}: speed {speed} outside {:?}",
                desc.speed
            );
            assert!(
                s.life >= desc.lifetime_s[0] && s.life <= desc.lifetime_s[1],
                "{kind:?} serial {serial}: life {} outside {:?}",
                s.life,
                desc.lifetime_s
            );
            assert!(
                s.size >= desc.size[0] && s.size <= desc.size[1],
                "{kind:?} serial {serial}: size {} outside {:?}",
                s.size,
                desc.size
            );
            assert!(
                (0.0..2.0 * std::f32::consts::PI).contains(&s.rotation),
                "{kind:?} serial {serial}: rotation {}",
                s.rotation
            );
        }
    }
}

#[test]
fn color_index_is_inside_the_palette() {
    for kind in KINDS {
        let desc = EmitterDesc::for_kind(kind);
        for serial in 0..1024 {
            let s = spawn_sample(&desc, serial);
            assert!(
                (s.color_index as usize) < desc.colors.len(),
                "{kind:?} serial {serial}: index {} of {}",
                s.color_index,
                desc.colors.len()
            );
        }
    }
    let mut desc = EmitterDesc::for_kind(ParticleKind::Confetti);
    desc.colors.truncate(3);
    for serial in 0..1024 {
        assert!(spawn_sample(&desc, serial).color_index < 3);
    }
}

#[test]
fn pcg_hash_matches_the_reference_values() {
    // Reference PCG-RXS-M-XS, computed with the algorithm in the module docs:
    // state = v * 747796405 + 2891336453 (wrapping u32);
    // word = ((state >> ((state >> 28) + 4)) ^ state) * 277803737 (wrapping);
    // hash = (word >> 22) ^ word.
    assert_eq!(pcg_hash(0), 129_708_002, "pcg_hash(0)");
    assert_eq!(pcg_hash(1), 2_831_084_092, "pcg_hash(1)");
    // The hash is a pure function: repeated calls agree.
    for v in [0, 1, 2, 42, u32::MAX] {
        assert_eq!(pcg_hash(v), pcg_hash(v));
    }
}
