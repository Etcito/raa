//! CPU twin of the section 67 distance interpreter. A query point is mapped into the source
//! frame, the packed instructions evaluate the program and `distScale` returns world units.
use raa_math::Vec3;
use crate::program::{DATA_SHIFT, Instance, OPCODE_MASK, TraceProgram, opcode};
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub t: f32,
    pub normal: [f32; 3],
    pub instance: u32,
    pub material: u32,
}
fn clamp(value: f32, min: f32, max: f32) -> f32 {
    value.max(min).min(max)
}
fn map(matrix: &[[f32; 4]; 3], p: Vec3) -> Vec3 {
    Vec3::new(
        matrix[0][0] * p.x + matrix[0][1] * p.y + matrix[0][2] * p.z + matrix[0][3],
        matrix[1][0] * p.x + matrix[1][1] * p.y + matrix[1][2] * p.z + matrix[1][3],
        matrix[2][0] * p.x + matrix[2][1] * p.y + matrix[2][2] * p.z + matrix[2][3],
    )
}
fn primitive(operation: u32, d: &[f32], p: Vec3) -> f32 {
    let (x, y, z) = (p.x, p.y, p.z);
    match operation {
        opcode::BOX | opcode::ROUND_BOX => {
            let r = d[17];
            let q = Vec3::new(x.abs() - d[14] + r, y.abs() - d[15] + r, z.abs() - d[16] + r);
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
            let t = clamp(((rt - qx) * kx + (h - y) * ky) / (kx * kx + ky * ky), 0.0, 1.0);
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
fn combine(operation: u32, a: f32, b: f32, k: f32) -> f32 {
    if operation == opcode::UNION {
        if !(k > 0.0) { return a.min(b); }
        let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
        if h >= 1.0 { a } else if h <= 0.0 { b } else { b + (a - b) * h - k * h * (1.0 - h) }
    } else if operation == opcode::SUBTRACT {
        if !(k > 0.0) { return a.max(-b); }
        let h = clamp(0.5 - 0.5 * (b + a) / k, 0.0, 1.0);
        if h >= 1.0 { -b } else if h <= 0.0 { a } else { a + (-b - a) * h + k * h * (1.0 - h) }
    } else {
        if !(k > 0.0) { return a.max(b); }
        let h = clamp(0.5 - 0.5 * (b - a) / k, 0.0, 1.0);
        if h >= 1.0 { a } else if h <= 0.0 { b } else { b + (a - b) * h + k * h * (1.0 - h) }
    }
}
fn evaluate(program: &TraceProgram, source: usize, p: Vec3) -> (f32, u32) {
    let Some(source) = program.sources.get(source) else {
        return (f32::INFINITY, 0);
    };
    let (code, data) = (source.code(program), source.data(program));
    let mut stack = [(0.0f32, 0u32); 64];
    let mut depth = 0usize;
    for &word in code {
        let operation = word & OPCODE_MASK;
        let Some(d) = data.get((word >> DATA_SHIFT) as usize..) else {
            return (f32::INFINITY, 0);
        };
        if opcode::is_primitive(operation) {
            if depth == stack.len() || d.len() < opcode::data_words(operation).unwrap_or(0) {
                return (f32::INFINITY, 0);
            }
            let local = Vec3::new(
                d[0] * p.x + d[1] * p.y + d[2] * p.z + d[3],
                d[4] * p.x + d[5] * p.y + d[6] * p.z + d[7],
                d[8] * p.x + d[9] * p.y + d[10] * p.z + d[11],
            );
            stack[depth] = (primitive(operation, d, local) * d[12], d[13] as u32);
            depth += 1;
        } else if depth >= 2 && !d.is_empty() {
            let ((a, am), (b, bm)) = (stack[depth - 2], stack[depth - 1]);
            let material = match operation {
                opcode::UNION if a <= b => am,
                opcode::UNION => bm,
                opcode::INTERSECT if a >= b => am,
                opcode::INTERSECT => bm,
                opcode::SUBTRACT => am,
                _ => return (f32::INFINITY, 0),
            };
            stack[depth - 2] = (combine(operation, a, b, d[0]), material);
            depth -= 1;
        } else {
            return (f32::INFINITY, 0);
        }
    }
    if depth == 0 { (f32::INFINITY, 0) } else { stack[depth - 1] }
}
fn distance_of(program: &TraceProgram, instance: &Instance, p: Vec3) -> f32 {
    let local = map(&instance.local_from_world, p);
    let (distance, _) = evaluate(program, instance.source as usize, local);
    let mut distance = distance * instance.dist_scale;
    if instance.deform_kind == 1 {
        let extent = program.sources.get(instance.source as usize)
            .map(|source| (Vec3::from_array(source.max) - Vec3::from_array(source.min)).length())
            .unwrap_or(0.0);
        let lipschitz = 1.0 + instance.bend_angle.abs() * extent / instance.bend_length.max(f32::MIN_POSITIVE);
        distance /= lipschitz.max(1.0);
    }
    distance
}
pub fn distance(program: &TraceProgram, instance: usize, point: [f32; 3]) -> f32 {
    match program.instances.get(instance) {
        Some(instance) => distance_of(program, instance, Vec3::from_array(point)),
        None => f32::INFINITY,
    }
}
pub fn material(program: &TraceProgram, instance: usize, point: [f32; 3]) -> u32 {
    let Some(instance) = program.instances.get(instance) else { return 0 };
    if instance.material >= 0 {
        return instance.material as u32;
    }
    let local = map(&instance.local_from_world, Vec3::from_array(point));
    evaluate(program, instance.source as usize, local).1
}
fn gradient(program: &TraceProgram, instance: usize, p: Vec3) -> Vec3 {
    let h = 1e-3;
    let normal = Vec3::new(
        distance(program, instance, [p.x + h, p.y, p.z]) - distance(program, instance, [p.x - h, p.y, p.z]),
        distance(program, instance, [p.x, p.y + h, p.z]) - distance(program, instance, [p.x, p.y - h, p.z]),
        distance(program, instance, [p.x, p.y, p.z + h]) - distance(program, instance, [p.x, p.y, p.z - h]),
    ).normalize_or_zero();
    if normal == Vec3::ZERO { Vec3::Y } else { normal }
}
pub fn trace_ray(program: &TraceProgram, origin: [f32; 3], direction: [f32; 3]) -> Option<Hit> {
    let origin = Vec3::from_array(origin);
    let direction = Vec3::from_array(direction).normalize_or_zero();
    if direction == Vec3::ZERO {
        return None;
    }
    let mut nearest: Option<Hit> = None;
    for (index, instance) in program.instances.iter().enumerate() {
        let mut t = 0.0f32;
        let mut hit = None;
        for _ in 0..128 {
            let p = origin + direction * t;
            let d = distance_of(program, instance, p);
            if !d.is_finite() {
                break;
            }
            if d < (1e-4 * t).max(1e-4) {
                hit = Some(t);
                break;
            }
            t += d.max(1e-4);
        }
        let Some(t) = hit else { continue };
        if nearest.is_some_and(|best| best.t <= t) {
            continue;
        }
        let p = origin + direction * t;
        nearest = Some(Hit {
            t,
            normal: gradient(program, index, p).to_array(),
            instance: index as u32,
            material: material(program, index, p.to_array()),
        });
    }
    nearest
}
