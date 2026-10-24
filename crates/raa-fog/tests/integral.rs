//! CHECK unit `fog integral`: the analytic height-fog integral agrees with 4096-step
//! composite Simpson quadrature across the contract grid (heights, directions, distances,
//! presets and start offsets), and the near-horizontal series limit never blows up.

use raa_fog::FogDesc;
use raa_fog::integral::{optical_depth, optical_depth_numeric, transmittance};

const ORIGINS: [f32; 4] = [-5.0, 0.0, 3.0, 20.0];
const DIRECTIONS: [f32; 7] = [-0.9, -0.2, -1e-6, 0.0, 1e-6, 0.3, 1.0];
const DISTANCES: [f32; 3] = [1.0, 50.0, 500.0];
const STARTS: [f32; 2] = [0.0, 10.0];
const PRESETS: [&str; 2] = ["jungle", "haze"];
const STEPS: usize = 4096;

fn agree(analytic: f32, numeric: f32) -> bool {
    if !analytic.is_finite() || !numeric.is_finite() {
        return analytic == numeric;
    }
    let scale = analytic.abs().max(numeric.abs());
    let tolerance = if scale > 1.0 { 1e-3 * scale } else { 1e-3 };
    (analytic - numeric).abs() <= tolerance
}

#[test]
fn analytic_matches_simpson_on_the_contract_grid() {
    let mut checked = 0;
    for preset in PRESETS {
        let desc = FogDesc::preset(preset).expect(preset);
        for origin in ORIGINS {
            for dir in DIRECTIONS {
                for distance in DISTANCES {
                    for start in STARTS {
                        let end = start + distance;
                        let analytic = optical_depth(origin, dir, start, end, &desc);
                        let numeric = optical_depth_numeric(origin, dir, start, end, &desc, STEPS);
                        assert!(
                            analytic >= 0.0 && !analytic.is_nan(),
                            "preset {preset} origin {origin} dir {dir} start {start}: {analytic}"
                        );
                        assert!(
                            agree(analytic, numeric),
                            "preset {preset} origin {origin} dir {dir} start {start} end {end}: \
                             analytic {analytic} vs simpson {numeric}"
                        );
                        if dir.abs() <= 1e-3 {
                            assert!(
                                analytic.is_finite(),
                                "horizontal ray blew up: preset {preset} origin {origin} \
                                 dir {dir} start {start} end {end}: {analytic}"
                            );
                        }
                        checked += 1;
                    }
                }
            }
        }
    }
    assert_eq!(
        checked,
        PRESETS.len() * ORIGINS.len() * DIRECTIONS.len() * DISTANCES.len() * STARTS.len()
    );
}

#[test]
fn empty_or_reversed_interval_is_zero() {
    let desc = FogDesc::preset("haze").expect("haze");
    assert_eq!(optical_depth(3.0, -0.9, 50.0, 50.0, &desc), 0.0);
    assert_eq!(optical_depth(3.0, -0.9, 50.0, 10.0, &desc), 0.0);
    assert_eq!(optical_depth_numeric(3.0, -0.9, 50.0, 10.0, &desc, 64), 0.0);
    assert_eq!(optical_depth(3.0, 1.0, -20.0, 0.0, &desc), 0.0);
}

#[test]
fn clear_fog_is_zero() {
    let clear = FogDesc::default();
    for origin in ORIGINS {
        for dir in DIRECTIONS {
            assert_eq!(optical_depth(origin, dir, 0.0, 500.0, &clear), 0.0);
            assert_eq!(
                optical_depth_numeric(origin, dir, 0.0, 500.0, &clear, 256),
                0.0
            );
            assert_eq!(transmittance(0.0), 1.0);
        }
    }
}

#[test]
fn start_never_precedes_the_description_start() {
    let mut desc = FogDesc::preset("jungle").expect("jungle");
    desc.start_m = 40.0;
    let clamped = optical_depth(2.0, 0.3, 0.0, 120.0, &desc);
    let explicit = optical_depth(2.0, 0.3, 40.0, 120.0, &desc);
    assert_eq!(clamped, explicit);
    assert_eq!(optical_depth(2.0, 0.3, 0.0, 30.0, &desc), 0.0);
}

#[test]
fn horizontal_ray_is_density_times_length() {
    let desc = FogDesc::preset("jungle").expect("jungle");
    let origin = 1.5;
    let expected =
        desc.density * (-desc.height_falloff * (origin - desc.base_height_m)).exp() * 250.0;
    let got = optical_depth(origin, 0.0, 0.0, 250.0, &desc);
    assert!(
        (got - expected).abs() <= 1e-3 + 1e-3 * expected.abs(),
        "{got} vs {expected}"
    );
}

#[test]
fn transmittance_is_exp_of_negative_depth() {
    assert!((transmittance(std::f32::consts::LN_2) - 0.5).abs() < 1e-6);
    assert!(transmittance(0.5) < transmittance(0.1));
    assert_eq!(transmittance(f32::INFINITY), 0.0);
}
