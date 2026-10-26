//! White balance matrix: identity at 6500 K, warm/cool and tint directions, finiteness.

use raa_grading::white_balance_matrix;

fn apply(m: &[[f32; 3]; 3], c: [f32; 3]) -> [f32; 3] {
    [0, 1, 2].map(|i| m[i][0] * c[0] + m[i][1] * c[1] + m[i][2] * c[2])
}

#[test]
fn reference_is_identity() {
    let m = white_balance_matrix(6500.0, 0.0);
    for (i, row) in m.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!((v - want).abs() < 1e-3, "m[{i}][{j}] = {v}");
        }
    }
}

#[test]
fn high_temperature_warms() {
    let [r, _, b] = apply(&white_balance_matrix(9000.0, 0.0), [1.0; 3]);
    assert!(r > b, "r {r} b {b}");
}

#[test]
fn low_temperature_cools() {
    let [r, _, b] = apply(&white_balance_matrix(4000.0, 0.0), [1.0; 3]);
    assert!(b > r, "r {r} b {b}");
}

#[test]
fn positive_tint_is_magenta() {
    let [r, g, b] = apply(&white_balance_matrix(6500.0, 0.5), [1.0; 3]);
    assert!(g < r && g < b, "r {r} g {g} b {b}");
}

#[test]
fn finite_over_range() {
    for t in (1500..=15000).step_by(500) {
        for tint in [-1.0, 0.0, 1.0] {
            let m = white_balance_matrix(t as f32, tint);
            assert!(
                m.iter().flatten().all(|v| v.is_finite()),
                "{t} K tint {tint}: {m:?}"
            );
        }
    }
}
