//! CPU checks of the AgX, ACES and linear curves the `tonemap` shader ports.

use raa_tonemap::curves::{luminance, map, srgb_oetf, tonemap};
use raa_tonemap::{Look, ToneMapper, ToneMapping};

const MAPPERS: [ToneMapper; 3] = [ToneMapper::Agx, ToneMapper::Aces, ToneMapper::Linear];
const LOOKS: [Look; 3] = [Look::None, Look::Punchy, Look::Golden];

fn grey(v: f32) -> [f32; 3] {
    [v; 3]
}

fn display(rgb: [f32; 3]) -> [f32; 3] {
    [srgb_oetf(rgb[0]), srgb_oetf(rgb[1]), srgb_oetf(rgb[2])]
}

/// 0..1000 sampled geometrically (plus 0), 2001 points.
fn ramp() -> Vec<f32> {
    let mut xs = vec![0.0];
    for i in 0..=2000 {
        xs.push(1e-5 * 1e8f32.powf(i as f32 / 2000.0));
    }
    xs
}

/// Checks luminance (display-encoded when `encoded`) never drops by more than
/// `tolerance` along the ramp.
fn assert_monotonic(mapper: ToneMapper, look: Look, tint: [f32; 3], encoded: bool, tolerance: f32) {
    let mut prev = f32::NEG_INFINITY;
    for x in ramp() {
        let y = luminance(map(mapper, look, [tint[0] * x, tint[1] * x, tint[2] * x]));
        let y = if encoded { srgb_oetf(y) } else { y };
        assert!(
            y.is_finite(),
            "{mapper:?} {look:?} {tint:?}: non-finite at {x}"
        );
        assert!(
            y >= prev - tolerance,
            "{mapper:?} {look:?} {tint:?}: luminance drops at {x}: {prev} -> {y}"
        );
        prev = prev.max(y);
    }
}

#[test]
fn each_mapper_is_monotonic_on_grey_luminance_over_0_to_1000() {
    for mapper in MAPPERS {
        for look in LOOKS {
            assert_monotonic(mapper, look, [1.0; 3], false, 0.0);
        }
    }
}

/// Tinted ramps: where channels saturate, AgX's outset desaturates towards white and
/// linear luminance dips by about 1e-4; the check allows half an 8-bit display step.
#[test]
fn each_mapper_is_monotonic_on_tinted_display_luminance() {
    let tints = [
        [1.0, 0.5, 0.1],
        [0.1, 0.4, 1.0],
        [0.2, 1.0, 0.3],
        [1.0, 0.0, 0.0],
    ];
    for mapper in MAPPERS {
        for tint in tints {
            assert_monotonic(mapper, Look::None, tint, true, 0.5 / 255.0);
        }
    }
}

#[test]
fn each_mapper_maps_black_to_black() {
    for mapper in MAPPERS {
        let out = map(mapper, Look::None, grey(0.0));
        let limit = if mapper == ToneMapper::Agx { 1e-3 } else { 0.0 };
        for c in out {
            assert!((0.0..=limit).contains(&c), "{mapper:?}: 0 -> {out:?}");
        }
    }
}

#[test]
fn mid_grey_lands_between_0_4_and_0_55_display() {
    for mapper in MAPPERS {
        let out = display(map(mapper, Look::None, grey(0.18)));
        for c in out {
            assert!(
                (0.4..=0.55).contains(&c),
                "{mapper:?}: 0.18 -> {out:?} display"
            );
        }
    }
}

#[test]
fn white_saturates_below_1_0001() {
    for mapper in MAPPERS {
        for look in LOOKS {
            for x in [1.0, 16.0, 100.0, 1000.0, 1e6] {
                for c in map(mapper, look, [x, x * 0.7, x * 0.2]) {
                    assert!(c <= 1.0001, "{mapper:?} {look:?}: {x} -> {c}");
                }
                for c in map(mapper, look, grey(x)) {
                    assert!(c <= 1.0001, "{mapper:?} {look:?}: grey {x} -> {c}");
                }
            }
        }
    }
}

#[test]
fn agx_and_aces_differ_at_0_18_and_4() {
    let max_diff = |rgb: [f32; 3]| {
        let a = display(map(ToneMapper::Agx, Look::None, rgb));
        let b = display(map(ToneMapper::Aces, Look::None, rgb));
        (0..3).map(|i| (a[i] - b[i]).abs()).fold(0.0f32, f32::max)
    };
    // Grey: the two curves are tuned to the same mid grey, so they only need to differ.
    assert!(max_diff(grey(0.18)) > 0.0);
    assert!(
        max_diff(grey(4.0)) >= 2.0 / 255.0,
        "grey 4.0: {}",
        max_diff(grey(4.0))
    );
    // A saturated colour shows the visible difference (hue and saturation handling).
    for x in [0.18, 4.0] {
        let d = max_diff([x, x * 0.35, x * 0.05]);
        assert!(d >= 2.0 / 255.0, "orange x {x}: {d}");
    }
}

#[test]
fn exposure_scales_input_by_two_to_the_ev() {
    let settings = ToneMapping {
        exposure_ev: 2.0,
        ..ToneMapping::default()
    };
    let exposed = tonemap(&settings, grey(0.045));
    let direct = map(ToneMapper::Agx, Look::None, grey(0.18));
    for i in 0..3 {
        assert!((exposed[i] - direct[i]).abs() <= 1e-6);
    }
}

#[test]
fn looks_change_agx_only() {
    let rgb = [0.4, 0.2, 0.05];
    let none = map(ToneMapper::Agx, Look::None, rgb);
    assert_ne!(none, map(ToneMapper::Agx, Look::Punchy, rgb));
    assert_ne!(none, map(ToneMapper::Agx, Look::Golden, rgb));
    for mapper in [ToneMapper::Aces, ToneMapper::Linear] {
        assert_eq!(map(mapper, Look::None, rgb), map(mapper, Look::Golden, rgb));
    }
}
