//! CPU SH9 projection: constant, gradient, solid angles and basis orthonormality.
#[path = "../src/sh.rs"]
mod sh;

use std::f64::consts::PI;

fn build(size: u32, f: impl Fn([f64; 3]) -> [f32; 3]) -> Vec<Vec<[f32; 3]>> {
    (0..6)
        .map(|face| {
            let mut v = Vec::new();
            for y in 0..size {
                for x in 0..size {
                    v.push(f(sh::face_texel_direction(face, x, y, size)));
                }
            }
            v
        })
        .collect()
}

fn project(cube: &[Vec<[f32; 3]>], size: u32) -> [[f32; 4]; 9] {
    let refs: [&[[f32; 3]]; 6] = std::array::from_fn(|i| cube[i].as_slice());
    sh::project_cube(&refs, size).expect("valid cube")
}

fn directions() -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    for x in -1i32..=1 {
        for y in -1i32..=1 {
            for z in -1i32..=1 {
                if x == 0 && y == 0 && z == 0 {
                    continue;
                }
                let (x, y, z) = (f64::from(x), f64::from(y), f64::from(z));
                let l = (x * x + y * y + z * z).sqrt();
                out.push([x / l, y / l, z / l]);
            }
        }
    }
    out
}

#[test]
fn constant_cube_projects_to_its_colour() {
    let c = [0.3f32, 0.5, 0.9];
    let cube = build(16, |_| c);
    let sh = project(&cube, 16);
    let dirs = directions();
    assert_eq!(dirs.len(), 26);
    for d in dirs {
        let v = sh::evaluate(&sh, d);
        for k in 0..3 {
            assert!((v[k] - f64::from(c[k])).abs() < 1e-3, "{d:?} {v:?}");
        }
    }
}

#[test]
fn solid_angles_sum_to_four_pi() {
    for size in [1u32, 4, 16, 64] {
        let mut sum = 0.0;
        for _ in 0..6 {
            for y in 0..size {
                for x in 0..size {
                    sum += sh::texel_solid_angle(x, y, size);
                }
            }
        }
        assert!(
            (sum - 4.0 * PI).abs() / (4.0 * PI) < 1e-4,
            "size {size}: {sum}"
        );
    }
}

#[test]
fn sky_gradient_is_brighter_up() {
    let ground = [0.05f64, 0.04, 0.03];
    let zenith = [2.0f64, 3.0, 5.0];
    let cube = build(16, |d| {
        let t = 0.5 + 0.5 * d[1];
        std::array::from_fn(|k| (ground[k] + (zenith[k] - ground[k]) * t) as f32)
    });
    let sh = project(&cube, 16);
    let up = sh::evaluate(&sh, [0.0, 1.0, 0.0]);
    let down = sh::evaluate(&sh, [0.0, -1.0, 0.0]);
    let horizon = sh::evaluate(&sh, [1.0, 0.0, 0.0]);
    for k in 0..3 {
        assert!(up[k] > down[k] && up[k] > horizon[k], "channel {k}");
    }
}

#[test]
fn basis_is_orthonormal_over_the_cube() {
    let size = 32;
    let mut gram = [[0.0f64; 9]; 9];
    for face in 0..6 {
        for y in 0..size {
            for x in 0..size {
                let w = sh::texel_solid_angle(x, y, size);
                let b = sh::sh_basis(sh::face_texel_direction(face, x, y, size));
                for (row, bi) in gram.iter_mut().zip(b) {
                    for (g, bj) in row.iter_mut().zip(b) {
                        *g += bi * bj * w;
                    }
                }
            }
        }
    }
    for (i, row) in gram.iter().enumerate() {
        for (j, g) in row.iter().enumerate() {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!((g - want).abs() < 1e-2, "gram[{i}][{j}] = {g}");
        }
    }
}
