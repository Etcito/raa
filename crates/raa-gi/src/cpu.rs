//! CPU twins of the GI kernel maths (share 1: triangle coverage and the voxel codecs), ported from
//! solstice@7e465b5 `crates/solstice-render/shaders/gi/{voxelize,voxel_light,radiance_codec}.wgsl`
//! and `shaders/common/math.wgsl`. MIT licence, Copyright (c) 2026 SamG-Coder.
//! The WGSL pack/unpack builtins are reimplemented per the WGSL spec as private helpers.

/// `EMISSIVE_RANGE` of `voxelize.wgsl:41` and `voxel_light.wgsl:28`.
const EMISSIVE_RANGE: f32 = 128.0;
/// `RADIANCE_RANGE` of `radiance_codec.wgsl:4`.
const RADIANCE_RANGE: f32 = 512.0;

/// Narrows the span of x values satisfying `slope * x + offset >= 0`; a horizontal edge keeps the
/// whole span or empties it to (1, 0). Port of `voxelize.wgsl:147-156` (`clip_span`).
pub fn clip_span(span: [f32; 2], slope: f32, offset: f32) -> [f32; 2] {
    if slope > 0.0 {
        return [span[0].max(-offset / slope), span[1]];
    }
    if slope < 0.0 {
        return [span[0], span[1].min(-offset / slope)];
    }
    if offset >= 0.0 { span } else { [1.0, 0.0] }
}

/// Collects every cell of [0, R)^3 a grid-space triangle covers, as `voxelize.wgsl:66-144`
/// (`voxelize_triangle`) writes them: dominant-axis scan, inside-positive edges widened 0.5,
/// depth spread `0.5 * (|dw/du| + |dw/dv|)`, triangles more than one voxel outside skipped.
/// `out` is cleared, then holds the cells sorted ascending by (x, y, z) and deduplicated.
///
/// Handoff: raa-gi: cpu_volume tests part 1 (voxelize, codecs) calls `voxelize_triangle`,
/// `clip_span`, `encode_emissive`, `decode_emissive`, `encode_radiance`, `decode_radiance`,
/// `oct_encode` and `oct_decode` of this file against its fixtures; the only state crossing is
/// the `out` Vec the test owns.
pub fn voxelize_triangle(v: [[f32; 3]; 3], resolution: u32, out: &mut Vec<[i32; 3]>) {
    out.clear();
    let [v0, v1, v2] = v;
    let e1 = sub(v1, v0);
    let e2 = sub(v2, v0);
    let normal = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    let an = normal.map(f32::abs);
    let (u, va, w) = if an[0] >= an[1] && an[0] >= an[2] {
        (1, 2, 0)
    } else if an[1] >= an[2] {
        (0, 2, 1)
    } else {
        (0, 1, 2)
    };
    if an[w] < 1.0e-8 {
        return;
    }
    let a = [v0[u], v0[va]];
    let b = [v1[u], v1[va]];
    let c = [v2[u], v2[va]];
    let limit = resolution as f32;
    let box_min = [a[0].min(b[0]).min(c[0]), a[1].min(b[1]).min(c[1])];
    let box_max = [a[0].max(b[0]).max(c[0]), a[1].max(b[1]).max(c[1])];
    let w_min = v0[w].min(v1[w]).min(v2[w]);
    let w_max = v0[w].max(v1[w]).max(v2[w]);
    if box_max[0] < -1.0
        || box_max[1] < -1.0
        || box_min[0] > limit + 1.0
        || box_min[1] > limit + 1.0
        || w_max < -1.0
        || w_min > limit + 1.0
    {
        return;
    }
    let clamp_cell = |x: f32| x.floor().clamp(0.0, limit - 1.0);
    let lo = [clamp_cell(box_min[0]), clamp_cell(box_min[1])];
    let hi = [clamp_cell(box_max[0]), clamp_cell(box_max[1])];

    let cross2 = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    let orientation = if cross2 >= 0.0 { 1.0 } else { -1.0 };
    let widen = 0.5;
    let edge = |p: [f32; 2], q: [f32; 2]| {
        let e = [(p[1] - q[1]) * orientation, (q[0] - p[0]) * orientation];
        let bias = -(e[0] * p[0] + e[1] * p[1]) + (e[0].abs() + e[1].abs()) * widen;
        (e, bias)
    };
    let edges = [edge(a, b), edge(b, c), edge(c, a)];

    let dw_du = -normal[u] / normal[w];
    let dw_dv = -normal[va] / normal[w];
    let w0 = v0[w] - dw_du * a[0] - dw_dv * a[1];
    let spread = 0.5 * (dw_du.abs() + dw_dv.abs());
    let n = resolution as i32;

    let mut y = lo[1];
    while y <= hi[1] {
        let py = y + 0.5;
        let mut span = [lo[0] + 0.5, hi[0] + 0.5];
        for (e, bias) in edges {
            span = clip_span(span, e[0], e[1] * py + bias);
        }
        if span[0] <= span[1] {
            let mut x = (span[0] - 0.5).ceil();
            let last_column = (span[1] - 0.5).floor();
            while x <= last_column {
                let depth = w0 + dw_du * (x + 0.5) + dw_dv * py;
                let first = (depth - spread).floor() as i32;
                let last = (depth + spread).floor() as i32;
                for z in first..=last {
                    let mut coord = [0i32; 3];
                    coord[u] = x as i32;
                    coord[va] = y as i32;
                    coord[w] = z;
                    if coord.iter().all(|&k| (0..n).contains(&k)) {
                        out.push(coord);
                    }
                }
                x += 1.0;
            }
        }
        y += 1.0;
    }
    out.sort_unstable();
    out.dedup();
}

/// Emissive RGBM over EMISSIVE_RANGE 128. Port of `voxelize.wgsl:49-53` (`encode_emissive`).
pub fn encode_emissive(e: [f32; 3]) -> u32 {
    let peak = (e[0].max(e[1]).max(e[2]) / EMISSIVE_RANGE).clamp(1.0e-6, 1.0);
    let scale = (peak * 255.0).ceil() / 255.0;
    let d = EMISSIVE_RANGE * scale;
    pack4x8unorm([e[0] / d, e[1] / d, e[2] / d, scale])
}

/// Inverse of [`encode_emissive`]. Port of `voxel_light.wgsl:31-34` (`decode_emissive`).
pub fn decode_emissive(packed: u32) -> [f32; 3] {
    let e = unpack4x8unorm(packed);
    [e[0], e[1], e[2]].map(|c| c * e[3] * EMISSIVE_RANGE)
}

/// Square-root radiance codec over RADIANCE_RANGE 512. Port of `radiance_codec.wgsl:6-9`.
pub fn encode_radiance(c: [f32; 3]) -> [u32; 2] {
    let e = c.map(|x| (x / RADIANCE_RANGE).clamp(0.0, 1.0).sqrt());
    [pack2x16unorm([e[0], e[1]]), pack2x16unorm([e[2], 0.0])]
}

/// Inverse of [`encode_radiance`]. Port of `radiance_codec.wgsl:11-14`.
pub fn decode_radiance(packed: [u32; 2]) -> [f32; 3] {
    let rg = unpack2x16unorm(packed[0]);
    let b = unpack2x16unorm(packed[1])[0];
    [rg[0], rg[1], b].map(|e| e * e * RADIANCE_RANGE)
}

/// Signed octahedral encoding packed with `pack2x16snorm`. Port of `math.wgsl:18-24`.
pub fn oct_encode(n: [f32; 3]) -> u32 {
    let s = n[0].abs() + n[1].abs() + n[2].abs();
    let p = [n[0] / s, n[1] / s];
    let e = if n[2] < 0.0 {
        let sg = sign_not_zero(p);
        [(1.0 - p[1].abs()) * sg[0], (1.0 - p[0].abs()) * sg[1]]
    } else {
        p
    };
    pack2x16snorm(e)
}

/// Unit vector of an `oct_encode` word. Port of `math.wgsl:7-15` (with `sign_not_zero`, 26-28).
pub fn oct_decode(packed: u32) -> [f32; 3] {
    let e = unpack2x16snorm(packed);
    let mut v = [e[0], e[1], 1.0 - e[0].abs() - e[1].abs()];
    if v[2] < 0.0 {
        let sg = sign_not_zero([v[0], v[1]]);
        v = [(1.0 - v[1].abs()) * sg[0], (1.0 - v[0].abs()) * sg[1], v[2]];
    }
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    v.map(|x| x / len)
}

/// `sign_not_zero` of `math.wgsl:26-28`.
fn sign_not_zero(v: [f32; 2]) -> [f32; 2] {
    v.map(|x| if x >= 0.0 { 1.0 } else { -1.0 })
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// WGSL `pack4x8unorm`: each byte is `floor(0.5 + 255 * clamp(v, 0, 1))`, x in the low byte.
fn pack4x8unorm(v: [f32; 4]) -> u32 {
    v.iter().enumerate().fold(0, |acc, (i, &x)| {
        acc | (((0.5 + 255.0 * x.clamp(0.0, 1.0)).floor() as u32) << (8 * i))
    })
}

/// WGSL `unpack4x8unorm`.
fn unpack4x8unorm(p: u32) -> [f32; 4] {
    [0, 1, 2, 3].map(|i| ((p >> (8 * i)) & 0xff) as f32 / 255.0)
}

/// WGSL `pack2x16unorm`: each half is `floor(0.5 + 65535 * clamp(v, 0, 1))`.
fn pack2x16unorm(v: [f32; 2]) -> u32 {
    let h = v.map(|x| (0.5 + 65535.0 * x.clamp(0.0, 1.0)).floor() as u32);
    h[0] | (h[1] << 16)
}

/// WGSL `unpack2x16unorm`.
fn unpack2x16unorm(p: u32) -> [f32; 2] {
    [p & 0xffff, p >> 16].map(|h| h as f32 / 65535.0)
}

/// WGSL `pack2x16snorm`: each half is the i16 bits of `floor(0.5 + 32767 * clamp(v, -1, 1))`.
fn pack2x16snorm(v: [f32; 2]) -> u32 {
    let h = v.map(|x| ((0.5 + 32767.0 * x.clamp(-1.0, 1.0)).floor() as i16 as u16) as u32);
    h[0] | (h[1] << 16)
}

/// WGSL `unpack2x16snorm`: `max(v / 32767, -1)` per half.
fn unpack2x16snorm(p: u32) -> [f32; 2] {
    [p as u16, (p >> 16) as u16].map(|h| (h as i16 as f32 / 32767.0).max(-1.0))
}
