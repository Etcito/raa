//! White balance: CAT02 von Kries adaptation in LMS, in linear Rec.709 through XYZ (D65).

/// Linear Rec.709 (sRGB primaries, D65) to CIE XYZ.
const RGB_TO_XYZ: [[f64; 3]; 3] = [
    [0.412_456_4, 0.357_576_1, 0.180_437_5],
    [0.212_672_9, 0.715_152_2, 0.072_175_0],
    [0.019_333_9, 0.119_192_0, 0.950_304_1],
];

/// CIE XYZ to CAT02 LMS.
const XYZ_TO_LMS: [[f64; 3]; 3] = [
    [0.7328, 0.4296, -0.1624],
    [-0.7036, 1.6975, 0.0061],
    [0.0030, 0.0136, 0.9834],
];

/// Tint scale: one unit of `tint` moves the white this far in CIE 1960 uv.
const TINT_UV: f64 = 0.02;

/// Reference white (no change): 6500 K, tint 0.
const REF_K: f64 = 6500.0;

/// White balance as a row-major 3x3 matrix applied to linear Rec.709 rgb column vectors
/// (`out[i] = sum_j m[i][j] * rgb[j]`).
///
/// Photo-editor convention: `temperature_k` is the colour temperature the scene is assumed
/// to be lit by, and the matrix brings that white to the reference white (6500 K, tint 0)
/// with a CAT02 von Kries scaling in LMS. So a temperature above 6500 K warms the image and
/// one below cools it; a positive `tint` turns the image magenta, a negative one green
/// (0.02 in CIE 1960 uv per unit, perpendicular to the locus). (6500, 0) is the identity.
/// Whites lie on the CIE daylight locus at and above 4000 K and on the Planckian locus
/// (Kim et al. approximation) below.
pub fn white_balance_matrix(temperature_k: f32, tint: f32) -> [[f32; 3]; 3] {
    let src = white_xyz(temperature_k as f64, tint as f64);
    let dst = white_xyz(REF_K, 0.0);
    let lms_src = mul_vec(&XYZ_TO_LMS, src);
    let lms_dst = mul_vec(&XYZ_TO_LMS, dst);
    let mut scale = [[0.0; 3]; 3];
    for i in 0..3 {
        scale[i][i] = lms_dst[i] / lms_src[i];
    }
    let rgb_to_lms = mul(&XYZ_TO_LMS, &RGB_TO_XYZ);
    let lms_to_rgb = inverse(&rgb_to_lms);
    let m = mul(&lms_to_rgb, &mul(&scale, &rgb_to_lms));
    m.map(|row| row.map(|v| v as f32))
}

/// XYZ (Y = 1) of the white at `t` kelvin shifted by `tint`. A positive tint moves the
/// assumed scene white toward green, so its compensation turns the image magenta.
fn white_xyz(t: f64, tint: f64) -> [f64; 3] {
    let daylight = t >= 4000.0;
    let (u, v) = xy_to_uv(locus_xy(t, daylight));
    // Locus tangent from a central difference on the same branch; the normal points to +v
    // (green side of the locus).
    let dt = 1.0;
    let (u0, v0) = xy_to_uv(locus_xy(t - dt, daylight));
    let (u1, v1) = xy_to_uv(locus_xy(t + dt, daylight));
    let (du, dv) = (u1 - u0, v1 - v0);
    let len = (du * du + dv * dv).sqrt();
    let (mut nu, mut nv) = (-dv / len, du / len);
    if nv < 0.0 {
        nu = -nu;
        nv = -nv;
    }
    let u = u + nu * tint * TINT_UV;
    let v = v + nv * tint * TINT_UV;
    // CIE 1960 uv to xy, then XYZ with Y = 1.
    let d = 2.0 * u - 8.0 * v + 4.0;
    let (x, y) = (3.0 * u / d, 2.0 * v / d);
    [x / y, 1.0, (1.0 - x - y) / y]
}

/// CIE xy chromaticity of `t` kelvin: CIE daylight formula when `daylight`, else the
/// Planckian locus approximation of Kim et al. (2002).
fn locus_xy(t: f64, daylight: bool) -> [f64; 2] {
    let (t1, t2, t3) = (1.0 / t, 1.0 / (t * t), 1.0 / (t * t * t));
    if daylight {
        let x = if t <= 7000.0 {
            -4.6070e9 * t3 + 2.9678e6 * t2 + 0.09911e3 * t1 + 0.244063
        } else {
            -2.0064e9 * t3 + 1.9018e6 * t2 + 0.24748e3 * t1 + 0.237040
        };
        [x, -3.0 * x * x + 2.87 * x - 0.275]
    } else {
        let x = -0.266_123_9e9 * t3 - 0.234_358_9e6 * t2 + 0.877_695_6e3 * t1 + 0.179_910;
        let y = if t <= 2222.0 {
            -1.106_381_4 * x * x * x - 1.348_110_20 * x * x + 2.185_558_32 * x - 0.202_196_83
        } else {
            -0.954_947_6 * x * x * x - 1.374_185_93 * x * x + 2.091_370_15 * x - 0.167_488_67
        };
        [x, y]
    }
}

fn xy_to_uv([x, y]: [f64; 2]) -> (f64, f64) {
    let d = -2.0 * x + 12.0 * y + 3.0;
    (4.0 * x / d, 6.0 * y / d)
}

fn mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    m
}

fn mul_vec(a: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| a[i][0] * v[0] + a[i][1] * v[1] + a[i][2] * v[2])
}

fn inverse(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let c = |i: usize, j: usize| {
        let (r0, r1) = ((i + 1) % 3, (i + 2) % 3);
        let (c0, c1) = ((j + 1) % 3, (j + 2) % 3);
        m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0]
    };
    let det = m[0][0] * c(0, 0) + m[0][1] * c(0, 1) + m[0][2] * c(0, 2);
    let mut inv = [[0.0; 3]; 3];
    for (i, row) in inv.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            *value = c(j, i) / det;
        }
    }
    inv
}
