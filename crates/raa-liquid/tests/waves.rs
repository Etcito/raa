//! CPU twin of the liquid Gerstner waves: finiteness, periodicity and the amplitude bound.

use raa_liquid::{WAVE_COUNT, wave_height};
use std::f32::consts::TAU;

/// Pack four waves and their speeds, with the function-specific `p[5..8]` at zero.
fn pack(waves: [[f32; 4]; WAVE_COUNT], speeds: [f32; WAVE_COUNT]) -> [[f32; 4]; 8] {
    let mut params = [[0.0; 4]; 8];
    params[..WAVE_COUNT].copy_from_slice(&waves);
    params[4] = speeds;
    params
}

/// A parameter array with one wave and `p[5..8]` at zero.
fn one_wave(amplitude: f32, wavelength: f32, theta: f32, q: f32, speed: f32) -> [[f32; 4]; 8] {
    let mut params = [[0.0; 4]; 8];
    params[0] = [amplitude, wavelength, theta, q];
    params[4][0] = speed;
    params
}

fn close(a: f32, b: f32, tolerance: f32) -> bool {
    (a - b).abs() <= tolerance
}

#[test]
fn wave_height_is_finite_on_a_grid_with_degenerate_waves() {
    let params = pack(
        [
            [0.35, 6.0, 0.7, 0.6],
            [0.18, 0.0, -1.2, 0.3],
            [f32::NAN, 3.0, 0.4, 0.5],
            [-0.22, 4.5, 2.1, 0.8],
        ],
        [1.7, 0.0, 1.1, 0.4],
    );
    let coords = [-17.0, -2.5, 0.0, 0.75, 9.0, 31.0];
    let times = [0.0, 0.13, 1.0, 7.5, 120.0];
    for x in coords {
        for z in coords {
            for t in times {
                let h = wave_height(&params, x, z, t);
                assert!(h.is_finite(), "height at ({x}, {z}, {t}) is {h}");
            }
        }
    }
}

#[test]
fn degenerate_waves_contribute_zero() {
    let degenerate = [
        ([0.5, 0.0, 0.3, 0.5], 1.0),
        ([0.5, -2.0, 0.3, 0.5], 1.0),
        ([0.0, 3.0, 0.3, 0.5], 1.0),
        ([f32::NAN, 3.0, 0.3, 0.5], 1.0),
        ([f32::INFINITY, 3.0, 0.3, 0.5], 1.0),
        ([0.5, f32::NAN, 0.3, 0.5], 1.0),
        ([0.5, 3.0, f32::NAN, 0.5], 1.0),
        ([0.5, 3.0, 0.3, f32::NAN], 1.0),
        ([0.5, 3.0, 0.3, 0.5], f32::NAN),
        ([0.5, 3.0, 0.3, 0.5], f32::INFINITY),
        ([0.5, f32::MIN_POSITIVE, 0.3, 0.5], 1.0),
    ];
    for (wave, speed) in degenerate {
        let params = one_wave(wave[0], wave[1], wave[2], wave[3], speed);
        for x in [-4.0, 0.0, 7.5] {
            for t in [0.0, 0.4, 3.0] {
                let h = wave_height(&params, x, 1.25, t);
                assert_eq!(h, 0.0, "wave {wave:?} speed {speed} gave {h}");
            }
        }
    }
    let valid = one_wave(0.3, 5.0, 0.4, 0.5, 1.0);
    assert_eq!(wave_height(&valid, f32::NAN, 0.0, 0.0), 0.0);
    assert_eq!(wave_height(&valid, 0.0, f32::INFINITY, 0.0), 0.0);
    assert_eq!(wave_height(&valid, 0.0, 0.0, f32::NAN), 0.0);
}

#[test]
fn single_wave_matches_the_rest_position_height() {
    let (amplitude, wavelength, theta, q, speed) = (0.4, 5.0, 0.9, 0.5, 2.3);
    let params = one_wave(amplitude, wavelength, theta, q, speed);
    let k = TAU / wavelength;
    for x in [-6.5, 0.0, 2.25, 11.0] {
        for z in [-3.0, 0.0, 8.5] {
            for t in [0.0, 0.37, 4.2] {
                let phase = k * ((theta.cos() * x + theta.sin() * z) - speed * t);
                let expected = amplitude * phase.sin();
                let h = wave_height(&params, x, z, t);
                assert!(close(h, expected, 1e-6), "{h} against {expected}");
            }
        }
    }
}

#[test]
fn single_wave_is_periodic_in_time_and_along_its_direction() {
    let (amplitude, wavelength, theta, q, speed) = (0.35, 6.0, 0.7, 0.6, 1.7);
    let params = one_wave(amplitude, wavelength, theta, q, speed);
    let period = wavelength / speed;
    let (dx, dz) = (wavelength * theta.cos(), wavelength * theta.sin());
    for (x, z) in [(-8.0, -3.0), (0.0, 0.0), (2.5, 7.25), (13.0, -4.5)] {
        for t in [0.0, 0.37, 1.9, 47.0] {
            let base = wave_height(&params, x, z, t);
            let later = wave_height(&params, x, z, t + period);
            assert!(close(later, base, 1e-4), "time: {later} against {base}");
            let shifted = wave_height(&params, x + dx, z + dz, t);
            assert!(
                close(shifted, base, 1e-4),
                "direction: {shifted} against {base}"
            );
        }
    }
    let along_x = one_wave(amplitude, wavelength, 0.0, q, speed);
    for x in [-8.0, 0.0, 13.0] {
        let base = wave_height(&along_x, x, 3.0, 0.37);
        let shifted = wave_height(&along_x, x + wavelength, 3.0, 0.37);
        assert!(close(shifted, base, 1e-4), "x: {shifted} against {base}");
    }
}

#[test]
fn sum_of_commensurate_waves_is_periodic() {
    let params = pack(
        [
            [0.30, 4.0, 0.0, 0.7],
            [0.20, 3.0, 0.6, 0.4],
            [0.16, 12.0, 1.9, 0.5],
            [0.24, 10.0, -1.2, 0.6],
        ],
        [1.0, 1.5, 2.0, 2.5],
    );
    let common_period = 12.0;
    for (x, z) in [(-9.0, 2.0), (0.0, 0.0), (4.5, -6.25)] {
        for t in [0.37, 5.1] {
            let base = wave_height(&params, x, z, t);
            for shift in [common_period, 2.0 * common_period] {
                let later = wave_height(&params, x, z, t + shift);
                assert!(
                    close(later, base, 1e-4),
                    "t+{shift}: {later} against {base}"
                );
            }
        }
    }
}

#[test]
fn wave_height_is_bounded_by_the_amplitudes() {
    let params = pack(
        [
            [0.4, 5.0, 0.2, 0.6],
            [-0.15, 3.2, 1.1, 0.4],
            [0.0, 4.0, 0.4, 0.3],
            [0.25, 7.0, -0.8, 0.5],
        ],
        [1.2, 2.3, 0.7, 1.6],
    );
    let bound: f32 = params[..WAVE_COUNT].iter().map(|wave| wave[0].abs()).sum();
    for x in [-12.0, -1.5, 0.0, 3.75, 20.0] {
        for z in [-7.0, 0.0, 5.5] {
            for t in [0.0, 0.8, 6.25] {
                let h = wave_height(&params, x, z, t);
                assert!(
                    h.abs() <= bound + 1e-6,
                    "|{h}| exceeds the amplitude bound {bound}"
                );
            }
        }
    }
}
