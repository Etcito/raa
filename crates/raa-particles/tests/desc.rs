//! Q12: every `for_kind` default validates and every rule is refused with
//! `particles.invalid` and a non-empty message naming the field.

use raa_particles::{EmitterDesc, ParticleKind};

const KINDS: [ParticleKind; 4] = [
    ParticleKind::Splash,
    ParticleKind::Snow,
    ParticleKind::Confetti,
    ParticleKind::Sparkles,
];

fn assert_invalid(desc: &EmitterDesc, field: &str) {
    let err = desc.validate().expect_err("expected a refusal");
    assert_eq!(err.code(), "particles.invalid", "{err}");
    let message = err.message();
    assert!(!message.is_empty(), "empty message for {field}");
    assert!(message.contains(field), "message `{message}` names {field}");
    assert!(err.to_string().starts_with("particles.invalid: "), "{err}");
}

#[test]
fn every_for_kind_default_validates() {
    for kind in KINDS {
        let d = EmitterDesc::for_kind(kind);
        assert_eq!(d.kind, kind);
        d.validate()
            .unwrap_or_else(|e| panic!("{kind:?} default refused: {e}"));
    }
}

#[test]
fn splash_defaults_match_the_contract() {
    let d = EmitterDesc::for_kind(ParticleKind::Splash);
    assert_eq!(d.direction, [0.0, 1.0, 0.0]);
    assert_eq!(d.spread_rad, 0.5);
    assert_eq!(d.rate_per_s, 0.0);
    assert_eq!(d.lifetime_s, [0.6, 1.2]);
    assert_eq!(d.speed, [1.5, 3.5]);
    assert_eq!(d.size, [0.015, 0.04]);
    assert_eq!(d.gravity, 9.81);
    assert_eq!(d.drag, 0.2);
    assert_eq!(d.colors, vec![[0.8, 0.9, 1.0]]);
    assert_eq!(d.emissive, 0.0);
    assert_eq!(d.volume_half, [0.0; 3]);
    assert_eq!(d.ground_y, None);
    assert_eq!(d.max_particles, 2048);
    assert_eq!(d.seed, 1);
}

#[test]
fn snow_defaults_match_the_contract() {
    let d = EmitterDesc::for_kind(ParticleKind::Snow);
    assert_eq!(d.direction, [0.0, -1.0, 0.0]);
    assert_eq!(d.spread_rad, 0.2);
    assert_eq!(d.rate_per_s, 400.0);
    assert_eq!(d.lifetime_s, [6.0, 10.0]);
    assert_eq!(d.speed, [0.2, 0.4]);
    assert_eq!(d.size, [0.01, 0.025]);
    assert_eq!(d.gravity, 1.0);
    assert_eq!(d.drag, 2.0);
    assert_eq!(d.colors, vec![[0.95, 0.97, 1.0]]);
    assert_eq!(d.volume_half, [4.0, 2.0, 4.0]);
    assert_eq!(d.max_particles, 8192);
}

#[test]
fn confetti_defaults_match_the_contract() {
    let d = EmitterDesc::for_kind(ParticleKind::Confetti);
    assert_eq!(d.direction, [0.0, 1.0, 0.0]);
    assert_eq!(d.spread_rad, 0.6);
    assert_eq!(d.rate_per_s, 120.0);
    assert_eq!(d.lifetime_s, [3.0, 5.0]);
    assert_eq!(d.speed, [3.0, 6.0]);
    assert_eq!(d.size, [0.02, 0.035]);
    assert_eq!(d.gravity, 9.81);
    assert_eq!(d.drag, 1.5);
    assert_eq!(d.colors.len(), 6);
    for c in &d.colors {
        assert!(c.iter().any(|v| *v >= 0.8), "not saturated: {c:?}");
    }
    assert_eq!(d.max_particles, 4096);
}

#[test]
fn sparkles_defaults_match_the_contract() {
    let d = EmitterDesc::for_kind(ParticleKind::Sparkles);
    assert_eq!(d.spread_rad, std::f32::consts::PI);
    assert_eq!(d.rate_per_s, 60.0);
    assert_eq!(d.lifetime_s, [0.6, 1.6]);
    assert_eq!(d.speed, [0.1, 0.4]);
    assert_eq!(d.size, [0.02, 0.06]);
    assert_eq!(d.gravity, 0.0);
    assert_eq!(d.drag, 0.5);
    assert_eq!(d.colors, vec![[1.0, 0.85, 0.3], [1.0, 1.0, 1.0]]);
    assert_eq!(d.emissive, 8.0);
    assert_eq!(d.volume_half, [0.5, 0.5, 0.5]);
    assert_eq!(d.max_particles, 1024);
}

#[test]
fn direction_must_be_non_zero() {
    for direction in [[0.0, 0.0, 0.0], [-0.0, 0.0, -0.0]] {
        let d = EmitterDesc {
            direction,
            ..EmitterDesc::for_kind(ParticleKind::Splash)
        };
        assert_invalid(&d, "direction");
    }
}

#[test]
fn spread_must_be_in_zero_to_pi() {
    let base = EmitterDesc::for_kind(ParticleKind::Splash);
    for spread_rad in [-0.01, std::f32::consts::PI + 0.01] {
        let d = EmitterDesc {
            spread_rad,
            ..base.clone()
        };
        assert_invalid(&d, "spread_rad");
    }
    for spread_rad in [0.0, std::f32::consts::PI] {
        EmitterDesc {
            spread_rad,
            ..base.clone()
        }
        .validate()
        .expect("boundary accepted");
    }
}

#[test]
fn rate_must_be_non_negative() {
    let d = EmitterDesc {
        rate_per_s: -1.0,
        ..EmitterDesc::for_kind(ParticleKind::Snow)
    };
    assert_invalid(&d, "rate_per_s");
}

#[test]
fn lifetime_must_be_positive_and_ordered() {
    let base = EmitterDesc::for_kind(ParticleKind::Splash);
    for lifetime_s in [[0.0, 1.0], [-0.5, 1.0], [2.0, 1.0]] {
        let d = EmitterDesc {
            lifetime_s,
            ..base.clone()
        };
        assert_invalid(&d, "lifetime_s");
    }
    EmitterDesc {
        lifetime_s: [1.0, 1.0],
        ..base
    }
    .validate()
    .expect("equal bounds accepted");
}

#[test]
fn speed_must_be_non_negative_and_ordered() {
    let base = EmitterDesc::for_kind(ParticleKind::Splash);
    for speed in [[-0.1, 1.0], [2.0, 1.0]] {
        let d = EmitterDesc {
            speed,
            ..base.clone()
        };
        assert_invalid(&d, "speed");
    }
    EmitterDesc {
        speed: [0.0, 0.0],
        ..base
    }
    .validate()
    .expect("zero speed accepted");
}

#[test]
fn size_must_be_positive_and_ordered() {
    let base = EmitterDesc::for_kind(ParticleKind::Splash);
    for size in [[0.0, 1.0], [-0.1, 1.0], [2.0, 1.0]] {
        let d = EmitterDesc {
            size,
            ..base.clone()
        };
        assert_invalid(&d, "size");
    }
    EmitterDesc {
        size: [0.5, 0.5],
        ..base
    }
    .validate()
    .expect("equal bounds accepted");
}

#[test]
fn drag_must_be_non_negative() {
    let d = EmitterDesc {
        drag: -0.01,
        ..EmitterDesc::for_kind(ParticleKind::Snow)
    };
    assert_invalid(&d, "drag");
}

#[test]
fn colors_must_have_one_to_eight_non_negative_entries() {
    let base = EmitterDesc::for_kind(ParticleKind::Confetti);
    for colors in [vec![], vec![[1.0; 3]; 9]] {
        let d = EmitterDesc {
            colors,
            ..base.clone()
        };
        assert_invalid(&d, "colors");
    }
    let mut colors = base.colors.clone();
    colors[2][1] = -0.1;
    assert_invalid(
        &EmitterDesc {
            colors,
            ..base.clone()
        },
        "colors[2][1]",
    );
    EmitterDesc {
        colors: vec![[0.0; 3]; 8],
        ..base
    }
    .validate()
    .expect("eight zero colours accepted");
}

#[test]
fn emissive_must_be_non_negative() {
    let d = EmitterDesc {
        emissive: -1.0,
        ..EmitterDesc::for_kind(ParticleKind::Sparkles)
    };
    assert_invalid(&d, "emissive");
}

#[test]
fn volume_half_must_be_non_negative() {
    let base = EmitterDesc::for_kind(ParticleKind::Snow);
    for i in 0..3 {
        let mut volume_half = base.volume_half;
        volume_half[i] = -0.5;
        let d = EmitterDesc {
            volume_half,
            ..base.clone()
        };
        assert_invalid(&d, &format!("volume_half[{i}]"));
    }
}

#[test]
fn max_particles_must_be_in_range() {
    let base = EmitterDesc::for_kind(ParticleKind::Splash);
    for max_particles in [0, 65537] {
        let d = EmitterDesc {
            max_particles,
            ..base.clone()
        };
        assert_invalid(&d, "max_particles");
    }
    for max_particles in [1, 65536] {
        EmitterDesc {
            max_particles,
            ..base.clone()
        }
        .validate()
        .expect("boundary accepted");
    }
}

#[test]
fn ground_y_must_be_finite() {
    let base = EmitterDesc::for_kind(ParticleKind::Splash);
    for ground_y in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let d = EmitterDesc {
            ground_y: Some(ground_y),
            ..base.clone()
        };
        assert_invalid(&d, "ground_y");
    }
}

#[test]
fn every_number_must_be_finite() {
    let bad = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
    let base = EmitterDesc::for_kind(ParticleKind::Splash);
    for v in bad {
        let mut position = base.position;
        position[1] = v;
        assert_invalid(
            &EmitterDesc {
                position,
                ..base.clone()
            },
            "position",
        );
        let mut direction = base.direction;
        direction[2] = v;
        assert_invalid(
            &EmitterDesc {
                direction,
                ..base.clone()
            },
            "direction",
        );
        assert_invalid(
            &EmitterDesc {
                spread_rad: v,
                ..base.clone()
            },
            "spread_rad",
        );
        assert_invalid(
            &EmitterDesc {
                rate_per_s: v,
                ..base.clone()
            },
            "rate_per_s",
        );
        assert_invalid(
            &EmitterDesc {
                lifetime_s: [v, 1.0],
                ..base.clone()
            },
            "lifetime_s[0]",
        );
        assert_invalid(
            &EmitterDesc {
                lifetime_s: [0.5, v],
                ..base.clone()
            },
            "lifetime_s[1]",
        );
        assert_invalid(
            &EmitterDesc {
                speed: [v, 1.0],
                ..base.clone()
            },
            "speed[0]",
        );
        assert_invalid(
            &EmitterDesc {
                speed: [0.5, v],
                ..base.clone()
            },
            "speed[1]",
        );
        assert_invalid(
            &EmitterDesc {
                size: [v, 1.0],
                ..base.clone()
            },
            "size[0]",
        );
        assert_invalid(
            &EmitterDesc {
                size: [0.5, v],
                ..base.clone()
            },
            "size[1]",
        );
        assert_invalid(
            &EmitterDesc {
                gravity: v,
                ..base.clone()
            },
            "gravity",
        );
        assert_invalid(
            &EmitterDesc {
                drag: v,
                ..base.clone()
            },
            "drag",
        );
        let colors = vec![[0.5, v, 0.5]];
        assert_invalid(
            &EmitterDesc {
                colors,
                ..base.clone()
            },
            "colors[0][1]",
        );
        assert_invalid(
            &EmitterDesc {
                emissive: v,
                ..base.clone()
            },
            "emissive",
        );
        let mut volume_half = base.volume_half;
        volume_half[2] = v;
        assert_invalid(
            &EmitterDesc {
                volume_half,
                ..base.clone()
            },
            "volume_half",
        );
    }
}

#[test]
fn opacity_must_be_in_zero_to_one() {
    let base = EmitterDesc::for_kind(ParticleKind::Splash);
    assert_eq!(base.opacity, 0.0);
    for opacity in [-0.01, 1.01, f32::NAN] {
        assert_invalid(
            &EmitterDesc {
                opacity,
                ..base.clone()
            },
            "opacity",
        );
    }
    for opacity in [0.0, 0.5, 1.0] {
        EmitterDesc {
            opacity,
            ..base.clone()
        }
        .validate()
        .expect("opacity accepted");
    }
}
