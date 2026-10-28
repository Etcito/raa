//! S16 joint limits: XYZ Euler degrees relative to each bone's rest rotation.
//!
//! The rest-relative delta is `conj(rest) * local`, decomposed as intrinsic XYZ
//! (`delta = qx * qy * qz`). Only bones with a limit change, and only when an angle
//! leaves its range; the rebuilt rotation keeps the input's quaternion hemisphere.

use raa_anim::{LocalPose, Rig};

type Q = [f64; 4];

fn mul(a: Q, b: Q) -> Q {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

fn normalize(q: Q) -> Option<Q> {
    let n = q.iter().map(|c| c * c).sum::<f64>().sqrt();
    (n.is_finite() && n > 1e-12).then(|| q.map(|c| c / n))
}

/// Intrinsic XYZ Euler angles (degrees) of a unit quaternion.
pub fn to_euler_xyz_deg(q: [f32; 4]) -> [f32; 3] {
    let q = q.map(f64::from);
    euler(normalize(q).unwrap_or([0.0, 0.0, 0.0, 1.0])).map(|a| a as f32)
}

fn euler(q: Q) -> [f64; 3] {
    let [x, y, z, w] = q;
    let m13 = (2.0 * (x * z + w * y)).clamp(-1.0, 1.0);
    let ay = m13.asin();
    let (ax, az) = if m13.abs() < 0.999_999_9 {
        let (m23, m33) = (2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y));
        let (m12, m11) = (2.0 * (x * y - w * z), 1.0 - 2.0 * (y * y + z * z));
        ((-m23).atan2(m33), (-m12).atan2(m11))
    } else {
        let (m32, m22) = (2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + z * z));
        (m32.atan2(m22), 0.0)
    };
    [ax, ay, az].map(f64::to_degrees)
}

/// Unit quaternion `qx * qy * qz` from intrinsic XYZ Euler degrees.
pub fn from_euler_xyz_deg(deg: [f32; 3]) -> [f32; 4] {
    from_euler(deg.map(f64::from)).map(|c| c as f32)
}

fn from_euler(deg: [f64; 3]) -> Q {
    let h = deg.map(|d| d.to_radians() * 0.5);
    let qx = [h[0].sin(), 0.0, 0.0, h[0].cos()];
    let qy = [0.0, h[1].sin(), 0.0, h[1].cos()];
    let qz = [0.0, 0.0, h[2].sin(), h[2].cos()];
    mul(mul(qx, qy), qz)
}

/// Clamp `local` so its rest-relative XYZ Euler angles lie in `[min, max]` degrees.
/// Non-finite or degenerate input is returned unchanged.
pub fn clamp_rotation(rest: [f32; 4], local: [f32; 4], min: [f32; 3], max: [f32; 3]) -> [f32; 4] {
    let (Some(rest), Some(q)) = (
        normalize(rest.map(f64::from)),
        normalize(local.map(f64::from)),
    ) else {
        return local;
    };
    let conj = [-rest[0], -rest[1], -rest[2], rest[3]];
    let mut delta = mul(conj, q);
    if delta[3] < 0.0 {
        delta = delta.map(|c| -c);
    }
    let angles = euler(delta);
    let mut clamped = angles;
    for i in 0..3 {
        let (lo, hi) = (f64::from(min[i]), f64::from(max[i]));
        clamped[i] = clamped[i].max(lo).min(hi);
    }
    if clamped == angles {
        return local;
    }
    let Some(mut out) = normalize(mul(rest, from_euler(clamped))) else {
        return local;
    };
    if (0..4).map(|i| out[i] * q[i]).sum::<f64>() < 0.0 {
        out = out.map(|c| -c);
    }
    let out = out.map(|c| c as f32);
    if out.iter().all(|c| c.is_finite()) {
        out
    } else {
        local
    }
}

/// Apply every S16 limit of `rig` to the local rotations of `local` in place.
/// Bones without a limit, and bones beyond `local`, are left untouched.
pub fn clamp_pose(rig: &Rig, local: &mut [LocalPose]) {
    for limit in &rig.limits {
        let (Some(bone), Some(pose)) = (rig.bones.get(limit.bone), local.get_mut(limit.bone))
        else {
            continue;
        };
        let rest = bone.rest.r.map(|c| c as f32);
        let (min, max) = (limit.min.map(|c| c as f32), limit.max.map(|c| c as f32));
        pose.r = clamp_rotation(rest, pose.r, min, max);
    }
}
