//! CPU twins of the GI kernel maths (share 1: triangle coverage and the voxel codecs), ported from
//! solstice@7e465b5 `crates/solstice-render/shaders/gi/{voxelize,voxel_light,radiance_codec}.wgsl`
//! and `shaders/common/math.wgsl`. MIT licence, Copyright (c) 2026 SamG-Coder.
//! The WGSL pack/unpack builtins are reimplemented per the WGSL spec as private helpers.
//! `crates/raa-gi/src/lib.rs`: out of scope for this share: consumers use the module path.

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

/// `FAR_DISTANCE_VOXELS` of `volume_write.wgsl:16`.
const FAR_DISTANCE_VOXELS: f32 = 64.0;

/// One clipmap level occupied-cell mask. `occupied` uses the `x + R * (y + R * z)` order of
/// `jump_flood.wgsl`'s `index_of`, and `R` is `resolution`.
pub struct CpuGrid {
    pub resolution: u32,
    pub occupied: Vec<bool>,
}

/// Jump-floods one clipmap level's nearest occupied cell. Every occupied cell seeds itself; then
/// each step runs one pass per axis (x, then y, then z), each reading the previous buffer and
/// writing a new one. A cell keeps the candidate among itself and its neighbours at `±step` along
/// the axis (when inside the grid) with the smallest squared distance to the cell; ties keep the
/// current value, then the `-step` side. Port of `jump_flood.wgsl:49-80`.
///
/// Handoff: raa-gi: cpu_volume tests part 2 (flood, trace) calls `jump_flood`, `seed_distance`,
/// `CpuVolume::level_at`, `CpuVolume::sample`, `CpuVolume::trace` and `CpuVolume::visibility` of
/// this file against its fixtures; the state crossing is the `CpuGrid` and `CpuVolume` values
/// the test builds.
pub fn jump_flood(grid: &CpuGrid, steps: &[i32]) -> Vec<Option<[i32; 3]>> {
    let r = grid.resolution as i32;
    if r <= 0 {
        return Vec::new();
    }
    let count = (r * r * r) as usize;
    let mut seeds = vec![None; count];
    for z in 0..r {
        for y in 0..r {
            for x in 0..r {
                let index = cell_index([x, y, z], r);
                if grid.occupied.get(index).copied().unwrap_or(false) {
                    seeds[index] = Some([x, y, z]);
                }
            }
        }
    }
    let mut current = seeds;
    for &step in steps {
        for axis in 0..3 {
            let mut next = vec![None; count];
            for z in 0..r {
                for y in 0..r {
                    for x in 0..r {
                        let cell = [x, y, z];
                        let index = cell_index(cell, r);
                        let mut best = current[index];
                        let mut best_distance =
                            best.map_or(f32::INFINITY, |seed| squared(seed, cell));
                        for side in [-1, 1] {
                            let mut neighbor = cell;
                            neighbor[axis] += step * side;
                            if neighbor[axis] < 0 || neighbor[axis] >= r {
                                continue;
                            }
                            if let Some(candidate) = current[cell_index(neighbor, r)] {
                                let distance = squared(candidate, cell);
                                if distance < best_distance {
                                    best = Some(candidate);
                                    best_distance = distance;
                                }
                            }
                        }
                        next[index] = best;
                    }
                }
            }
            current = next;
        }
    }
    current
}

/// Nearest-seed distance in metres: `length(cell - seed) * voxel`, or `64 * voxel` without a seed
/// (`volume_write.wgsl:38-43`).
pub fn seed_distance(cell: [i32; 3], seed: Option<[i32; 3]>, voxel: f32) -> f32 {
    match seed {
        Some(seed) => squared(seed, cell).sqrt() * voxel,
        None => FAR_DISTANCE_VOXELS * voxel,
    }
}

/// Contour period of the distance debug view in metres (`RAA_GI_DEBUG_PERIOD` of
/// `debug_view.wgsl`): two 0.2 m finest voxels.
pub const DISTANCE_PERIOD_METRES: f32 = 0.4;

/// How far along the surface normal the distance debug view samples the field
/// (`RAA_GI_DEBUG_LIFT` of `debug_view.wgsl`): seven finest voxels.
pub const DISTANCE_LIFT_METRES: f32 = 1.4;

/// The distance debug view's grey level for a sampled distance `d` in metres:
/// `0.5 + 0.5 cos(2 pi d / 0.4)` (S38), light at whole periods and dark half-way between.
pub fn distance_band(d: f32) -> f32 {
    0.5 + 0.5 * (2.0 * std::f32::consts::PI * d / DISTANCE_PERIOD_METRES).cos()
}

/// Where the distance debug view samples the field for a depth surface point with the unit
/// `normal` facing the camera: `surface + normal * DISTANCE_LIFT_METRES`. There the field holds
/// the distance to the other surfaces nearby, so the bands run parallel to them.
pub fn distance_probe(surface: [f32; 3], normal: [f32; 3]) -> [f32; 3] {
    [0, 1, 2].map(|i| surface[i] + normal[i] * DISTANCE_LIFT_METRES)
}

fn cell_index(cell: [i32; 3], r: i32) -> usize {
    (cell[0] + r * (cell[1] + r * cell[2])) as usize
}

fn squared(a: [i32; 3], b: [i32; 3]) -> f32 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) as f32
}

/// CPU view of the sampled atlas. `levels` are (origin metres xyz, voxel size w); `texels` form
/// the R x R x (R * L) atlas at `x + R * (y + R * z_atlas)`, with level i in the z slab
/// `[i * R, (i + 1) * R)`.
pub struct CpuVolume {
    pub levels: Vec<[f32; 4]>,
    pub resolution: u32,
    pub texels: Vec<[f32; 4]>,
}

impl CpuVolume {
    /// Finest level at or after `first` whose local coordinates lie strictly inside
    /// `(1.5, R - 1.5)` on every axis, else `levels.len()`. Port of `environment.wgsl:141-152`.
    pub fn level_at(&self, p: [f32; 3], first: usize) -> usize {
        let upper = self.resolution as f32 - 1.5;
        let mut level = first;
        while level < self.levels.len() {
            let grid = self.levels[level];
            let local = [
                (p[0] - grid[0]) / grid[3],
                (p[1] - grid[1]) / grid[3],
                (p[2] - grid[2]) / grid[3],
            ];
            if local.iter().all(|&v| v > 1.5 && v < upper) {
                return level;
            }
            level += 1;
        }
        self.levels.len()
    }

    /// Trilinear `textureSampleLevel` at lod 0 with clamp-to-edge addressing. Texel centres are at
    /// `(i + 0.5) / size`, z is clamped half a texel inside `level`'s slab, and the atlas
    /// coordinate is `(z + level) / L`. Port of `environment.wgsl:155-164`.
    pub fn sample(&self, p: [f32; 3], level: usize) -> [f32; 4] {
        if level >= self.levels.len() || self.resolution == 0 {
            return [0.0; 4];
        }
        let r = self.resolution as f32;
        let l = self.levels.len() as f32;
        let grid = self.levels[level];
        let inv_voxel = 1.0 / grid[3];
        let local = [
            (p[0] - grid[0]) * inv_voxel / r,
            (p[1] - grid[1]) * inv_voxel / r,
            (p[2] - grid[2]) * inv_voxel / r,
        ];
        let half_texel = 0.5 / r;
        let z = local[2].clamp(half_texel, 1.0 - half_texel);
        let atlas_z = (z + level as f32) / l;
        let fx = local[0] * r - 0.5;
        let fy = local[1] * r - 0.5;
        let fz = atlas_z * (r * l) - 0.5;
        let x0 = fx.floor();
        let y0 = fy.floor();
        let z0 = fz.floor();
        let tx = fx - x0;
        let ty = fy - y0;
        let tz = fz - z0;
        let (x0, y0, z0) = (x0 as i32, y0 as i32, z0 as i32);
        let c00 = mix4(self.texel(x0, y0, z0), self.texel(x0 + 1, y0, z0), tx);
        let c10 = mix4(
            self.texel(x0, y0 + 1, z0),
            self.texel(x0 + 1, y0 + 1, z0),
            tx,
        );
        let c01 = mix4(
            self.texel(x0, y0, z0 + 1),
            self.texel(x0 + 1, y0, z0 + 1),
            tx,
        );
        let c11 = mix4(
            self.texel(x0, y0 + 1, z0 + 1),
            self.texel(x0 + 1, y0 + 1, z0 + 1),
            tx,
        );
        mix4(mix4(c00, c10, ty), mix4(c01, c11, ty), tz)
    }

    /// Sphere traces the atlas one level at a time. A hit needs the sampled distance below
    /// `voxel * hit_voxels` while the ray is approaching within one level; each step advances
    /// `max(distance - 0.5 voxel, 0.6 voxel)`. Port of `environment.wgsl:179-205`.
    pub fn trace(
        &self,
        origin: [f32; 3],
        dir: [f32; 3],
        start: f32,
        max_t: f32,
        max_steps: u32,
        hit_voxels: f32,
    ) -> CpuHit {
        let mut t = start;
        let mut level = 0;
        let mut measured_level = self.levels.len();
        let mut previous_distance = 0.0;
        let mut hit = false;
        let mut radiance = [0.0; 3];
        let mut steps = 0;
        while steps < max_steps && t < max_t {
            let p = [
                origin[0] + dir[0] * t,
                origin[1] + dir[1] * t,
                origin[2] + dir[2] * t,
            ];
            level = self.level_at(p, level);
            if level >= self.levels.len() {
                break;
            }
            let voxel = self.levels[level][3];
            let s = self.sample(p, level);
            let approaching = level == measured_level && s[3] <= previous_distance;
            measured_level = level;
            previous_distance = s[3];
            if s[3] < voxel * hit_voxels && approaching {
                hit = true;
                radiance = [s[0], s[1], s[2]];
                break;
            }
            t += (s[3] - voxel * 0.5).max(voxel * 0.6);
            steps += 1;
        }
        CpuHit { hit, t, radiance }
    }

    /// 0 on a [`CpuVolume::trace`] hit with `hit_voxels` 0.85, otherwise 1. An empty volume always
    /// answers 1. Port of `environment.wgsl:216-219`.
    pub fn visibility(
        &self,
        origin: [f32; 3],
        dir: [f32; 3],
        start: f32,
        max_t: f32,
        max_steps: u32,
    ) -> f32 {
        if self.trace(origin, dir, start, max_t, max_steps, 0.85).hit {
            0.0
        } else {
            1.0
        }
    }

    fn texel(&self, x: i32, y: i32, z: i32) -> [f32; 4] {
        let r = self.resolution as i32;
        if r <= 0 || self.levels.is_empty() {
            return [0.0; 4];
        }
        let z_max = r * self.levels.len() as i32;
        let xi = x.clamp(0, r - 1) as usize;
        let yi = y.clamp(0, r - 1) as usize;
        let zi = z.clamp(0, z_max - 1) as usize;
        let index = xi + r as usize * (yi + r as usize * zi);
        self.texels.get(index).copied().unwrap_or([0.0; 4])
    }
}

/// Result of [`CpuVolume::trace`]: whether a surface was hit at `t`, and its radiance.
pub struct CpuHit {
    pub hit: bool,
    pub t: f32,
    pub radiance: [f32; 3],
}

fn mix4(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}
