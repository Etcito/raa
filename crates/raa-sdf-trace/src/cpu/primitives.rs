use super::*;
pub(super) fn primitive(operation: u32, d: &[f32], p: Vec3) -> f32 {
    let (x, y, z) = (p.x, p.y, p.z);
    match operation {
        opcode::BOX | opcode::ROUND_BOX => {
            let r = d[17];
            let q = Vec3::new(
                x.abs() - d[14] + r,
                y.abs() - d[15] + r,
                z.abs() - d[16] + r,
            );
            q.max(Vec3::ZERO).length() + q.x.max(q.y).max(q.z).min(0.0) - r
        }
        opcode::SPHERE => p.length() - d[14],
        opcode::CAPSULE => {
            let qy = y - clamp(y, -d[14], d[14]);
            (x * x + qy * qy + z * z).sqrt() - d[15]
        }
        opcode::CYLINDER => {
            let (dx, dy) = ((x * x + z * z).sqrt() - d[14], y.abs() - d[15]);
            let (mx, my) = (dx.max(0.0), dy.max(0.0));
            dx.max(dy).min(0.0) + (mx * mx + my * my).sqrt()
        }
        opcode::CONE => {
            let (h, rb, rt) = (d[14], d[15], d[16]);
            let qx = (x * x + z * z).sqrt();
            let (kx, ky) = (rt - rb, 2.0 * h);
            let ax = qx - qx.min(if y < 0.0 { rb } else { rt });
            let ay = y.abs() - h;
            let t = clamp(
                ((rt - qx) * kx + (h - y) * ky) / (kx * kx + ky * ky),
                0.0,
                1.0,
            );
            let (bx, by) = (qx - rt + kx * t, y - h + ky * t);
            let s = if bx < 0.0 && ay < 0.0 { -1.0 } else { 1.0 };
            s * (ax * ax + ay * ay).min(bx * bx + by * by).sqrt()
        }
        opcode::TORUS => {
            let q = (x * x + z * z).sqrt() - d[14];
            (q * q + y * y).sqrt() - d[15]
        }
        opcode::ELLIPSOID => {
            let (rx, ry, rz) = (d[14], d[15], d[16]);
            let (ax, ay, az) = (x / rx, y / ry, z / rz);
            let k0 = (ax * ax + ay * ay + az * az).sqrt();
            if k0 < 1.0 {
                return (k0 - 1.0) * rx.min(ry).min(rz);
            }
            let k1 = ((ax / rx).powi(2) + (ay / ry).powi(2) + (az / rz).powi(2)).sqrt();
            k0 * (k0 - 1.0) / k1
        }
        _ => f32::INFINITY,
    }
}
pub(super) fn combine(operation: u32, a: f32, b: f32, k: f32) -> f32 {
    if operation == opcode::UNION {
        if k <= 0.0 || k.is_nan() {
            return a.min(b);
        }
        let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
        if h >= 1.0 {
            a
        } else if h <= 0.0 {
            b
        } else {
            b + (a - b) * h - k * h * (1.0 - h)
        }
    } else if operation == opcode::SUBTRACT {
        if k <= 0.0 || k.is_nan() {
            return a.max(-b);
        }
        let h = clamp(0.5 - 0.5 * (b + a) / k, 0.0, 1.0);
        if h >= 1.0 {
            -b
        } else if h <= 0.0 {
            a
        } else {
            a + (-b - a) * h + k * h * (1.0 - h)
        }
    } else {
        if k <= 0.0 || k.is_nan() {
            return a.max(b);
        }
        let h = clamp(0.5 - 0.5 * (b - a) / k, 0.0, 1.0);
        if h >= 1.0 {
            a
        } else if h <= 0.0 {
            b
        } else {
            b + (a - b) * h + k * h * (1.0 - h)
        }
    }
}
