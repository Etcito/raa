//! CPU reference of the SH9 irradiance projection (cosine-convolved radiance / pi).
#![allow(dead_code)]

use std::f64::consts::PI;

/// Direction (unit) through the centre of texel (x, y) of a `size` px face.
/// Face order +X,-X,+Y,-Y,+Z,-Z; v grows with the texel row (downwards).
pub(crate) fn face_texel_direction(face: usize, x: u32, y: u32, size: u32) -> [f64; 3] {
    let s = f64::from(size.max(1));
    let u = 2.0 * (f64::from(x) + 0.5) / s - 1.0;
    let v = 2.0 * (f64::from(y) + 0.5) / s - 1.0;
    let d = match face {
        0 => [1.0, -v, -u],
        1 => [-1.0, -v, u],
        2 => [u, 1.0, v],
        3 => [u, -1.0, -v],
        4 => [u, -v, 1.0],
        _ => [-u, -v, -1.0],
    };
    let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    [d[0] / l, d[1] / l, d[2] / l]
}

fn area_element(x: f64, y: f64) -> f64 {
    (x * y).atan2((x * x + y * y + 1.0).sqrt())
}

/// Exact solid angle of texel (x, y) of a `size` px face (same for all faces).
pub(crate) fn texel_solid_angle(x: u32, y: u32, size: u32) -> f64 {
    let s = f64::from(size.max(1));
    let x0 = 2.0 * f64::from(x) / s - 1.0;
    let y0 = 2.0 * f64::from(y) / s - 1.0;
    let x1 = 2.0 * (f64::from(x) + 1.0) / s - 1.0;
    let y1 = 2.0 * (f64::from(y) + 1.0) / s - 1.0;
    area_element(x0, y0) - area_element(x0, y1) - area_element(x1, y0) + area_element(x1, y1)
}

/// The 9 real SH basis functions (order 2) at a unit direction.
pub(crate) fn sh_basis(d: [f64; 3]) -> [f64; 9] {
    let [x, y, z] = d;
    [
        0.282095,
        0.488603 * y,
        0.488603 * z,
        0.488603 * x,
        1.092548 * x * y,
        1.092548 * y * z,
        0.315392 * (3.0 * z * z - 1.0),
        1.092548 * x * z,
        0.546274 * (x * x - y * y),
    ]
}

/// Cosine-lobe factor (A_l / pi) per coefficient.
const COSINE: [f64; 9] = [
    1.0,
    2.0 / 3.0,
    2.0 / 3.0,
    2.0 / 3.0,
    0.25,
    0.25,
    0.25,
    0.25,
    0.25,
];

/// Projects a cube of 6 faces of linear rgb texels (row-major, square) into cosine-convolved
/// SH9 coefficients divided by pi. `None` when a face is not `size * size` texels or size is 0.
pub(crate) fn project_cube(faces: &[&[[f32; 3]]; 6], size: u32) -> Option<[[f32; 4]; 9]> {
    let n = (size as usize).checked_mul(size as usize)?;
    if size == 0 || faces.iter().any(|f| f.len() != n) {
        return None;
    }
    let mut acc = [[0.0f64; 3]; 9];
    for (face, texels) in faces.iter().enumerate() {
        for y in 0..size {
            for x in 0..size {
                let rgb = texels.get((y * size + x) as usize)?;
                let w = texel_solid_angle(x, y, size);
                let basis = sh_basis(face_texel_direction(face, x, y, size));
                for (a, b) in acc.iter_mut().zip(basis) {
                    for c in 0..3 {
                        a[c] += f64::from(rgb[c]) * b * w;
                    }
                }
            }
        }
    }
    let mut out = [[0.0f32; 4]; 9];
    for i in 0..9 {
        for c in 0..3 {
            out[i][c] = (acc[i][c] * COSINE[i]) as f32;
        }
    }
    Some(out)
}

/// Irradiance / pi (rgb) for the unit normal `n`.
pub(crate) fn evaluate(sh: &[[f32; 4]; 9], n: [f64; 3]) -> [f64; 3] {
    let basis = sh_basis(n);
    let mut out = [0.0f64; 3];
    for (coef, b) in sh.iter().zip(basis) {
        for c in 0..3 {
            out[c] += f64::from(coef[c]) * b;
        }
    }
    out
}

#[allow(dead_code)]
const _FOUR_PI: f64 = 4.0 * PI;
