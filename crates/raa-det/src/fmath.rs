//! Source-ordered f32 arithmetic. No reassociation, fused operations or trig.

/// Stable refusals from the deterministic math helpers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MathError {
    NotFinite,
    ZeroLength,
    NotUnit,
}
impl MathError {
    pub const fn code(self) -> &'static str {
        match self {
            Self::NotFinite => "det.not_finite",
            Self::ZeroLength => "det.zero_length",
            Self::NotUnit => "det.not_unit",
        }
    }
}
impl core::fmt::Display for MathError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.code())
    }
}
impl std::error::Error for MathError {}

fn finite<const N: usize>(v: [f32; N]) -> Result<[f32; N], MathError> {
    if v.iter().all(|x| x.is_finite()) {
        Ok(v)
    } else {
        Err(MathError::NotFinite)
    }
}

/// S17 linear sampling; finite extrapolation is allowed.
pub fn lerp(a: f32, b: f32, u: f32) -> Result<f32, MathError> {
    finite([a, b, u])?;
    Ok(finite([a + (b - a) * u])?[0])
}

/// S17 cubic sampling, with control points already including handle offsets.
pub fn bezier(p0: f32, p1: f32, p2: f32, p3: f32, u: f32) -> Result<f32, MathError> {
    finite([p0, p1, p2, p3, u])?;
    let v = 1.0 - u;
    let result =
        ((v * v * v) * p0 + (3.0 * v * v * u) * p1) + (3.0 * v * u * u) * p2 + (u * u * u) * p3;
    Ok(finite([result])?[0])
}

/// Sum squares in component order, sqrt, reciprocal, then multiply.
/// Zero (including squared-length underflow) and overflow are refused.
pub fn normalize<const N: usize>(v: [f32; N]) -> Result<[f32; N], MathError> {
    finite(v)?;
    let mut squared = 0.0;
    for x in v {
        squared += x * x;
    }
    finite([squared])?;
    if squared == 0.0 {
        return Err(MathError::ZeroLength);
    }
    let inverse = 1.0 / squared.sqrt();
    finite(v.map(|x| x * inverse))
}

fn dot(a: [f32; 4], b: [f32; 4]) -> f32 {
    ((a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]) + a[3] * b[3]
}

/// Negate b iff dot(a, b) < 0; a zero dot preserves b's sign bits.
pub fn sign_align(a: [f32; 4], b: [f32; 4]) -> Result<[f32; 4], MathError> {
    finite(a)?;
    finite(b)?;
    let d = finite([dot(a, b)])?[0];
    Ok(if d < 0.0 { b.map(|x| -x) } else { b })
}

/// Shortest-path quaternion interpolation: sign-align, lerp, normalize.
/// Endpoints must have nonzero representable squared lengths; u may extrapolate.
pub fn nlerp(a: [f32; 4], b: [f32; 4], u: f32) -> Result<[f32; 4], MathError> {
    normalize(a)?;
    normalize(b)?;
    let b = sign_align(a, b)?;
    let mut out = [0.0; 4];
    for i in 0..4 {
        out[i] = lerp(a[i], b[i], u)?;
    }
    normalize(out)
}

/// Hamilton product in xyzw order; applies b first, then a. No normalization.
pub fn quat_mul(a: [f32; 4], b: [f32; 4]) -> Result<[f32; 4], MathError> {
    finite(a)?;
    finite(b)?;
    let [x, y, z, w] = a;
    let [i, j, k, r] = b;
    finite([
        ((w * i + x * r) + y * k) - z * j,
        ((w * j - x * k) + y * r) + z * i,
        ((w * k + x * j) - y * i) + z * r,
        ((w * r - x * i) - y * j) - z * k,
    ])
}

fn unit(q: [f32; 4]) -> Result<(), MathError> {
    finite(q)?;
    let squared = dot(q, q);
    if !squared.is_finite() || (squared - 1.0).abs() > 1e-3 {
        return Err(MathError::NotUnit);
    }
    Ok(())
}

/// Rotate by a unit xyzw quaternion: v + w*t + cross(q.xyz, t).
pub fn quat_rotate(q: [f32; 4], v: [f32; 3]) -> Result<[f32; 3], MathError> {
    unit(q)?;
    finite(v)?;
    let [x, y, z, w] = q;
    let [a, b, c] = v;
    let t = [
        2.0 * (y * c - z * b),
        2.0 * (z * a - x * c),
        2.0 * (x * b - y * a),
    ];
    finite([
        (a + w * t[0]) + (y * t[2] - z * t[1]),
        (b + w * t[1]) + (z * t[0] - x * t[2]),
        (c + w * t[2]) + (x * t[1] - y * t[0]),
    ])
}

/// Column-major T*R*S for column vectors. Rotation must be unit length;
/// finite zero and negative scales are supported. Translation is column four.
pub fn trs_compose(t: [f32; 3], q: [f32; 4], s: [f32; 3]) -> Result<[f32; 16], MathError> {
    finite(t)?;
    unit(q)?;
    finite(s)?;
    let [x, y, z, w] = q;
    let (xx, yy, zz) = (x * x, y * y, z * z);
    let (xy, xz, yz) = (x * y, x * z, y * z);
    let (wx, wy, wz) = (w * x, w * y, w * z);
    finite([
        (1.0 - 2.0 * (yy + zz)) * s[0],
        2.0 * (xy + wz) * s[0],
        2.0 * (xz - wy) * s[0],
        0.0,
        2.0 * (xy - wz) * s[1],
        (1.0 - 2.0 * (xx + zz)) * s[1],
        2.0 * (yz + wx) * s[1],
        0.0,
        2.0 * (xz + wy) * s[2],
        2.0 * (yz - wx) * s[2],
        (1.0 - 2.0 * (xx + yy)) * s[2],
        0.0,
        t[0],
        t[1],
        t[2],
        1.0,
    ])
}
