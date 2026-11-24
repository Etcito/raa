//! Camera-relative cubemap columns intersected with concentric radial shells.

use raa_core::RaaError;

use super::invalid_config;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum CubeFace {
    PosX,
    NegX,
    PosY,
    NegY,
    PosZ,
    NegZ,
}

impl CubeFace {
    pub const ALL: [Self; 6] = [
        Self::PosX,
        Self::NegX,
        Self::PosY,
        Self::NegY,
        Self::PosZ,
        Self::NegZ,
    ];

    fn frame(self) -> [[f64; 3]; 3] {
        match self {
            Self::PosX => [[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]],
            Self::NegX => [[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]],
            Self::PosY => [[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]],
            Self::NegY => [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]],
            Self::PosZ => [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]],
            Self::NegZ => [[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CellKey {
    pub shell: u16,
    pub face: CubeFace,
    pub x: u16,
    pub y: u16,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphericalGridConfig {
    pub angular_resolution: u16,
    pub radial_shells: u16,
    pub clip_start: f32,
    pub z_max: f32,
    /// Ratio between consecutive shell widths. A ratio of one gives uniform widths.
    pub radial_growth: f32,
}

impl Default for SphericalGridConfig {
    fn default() -> Self {
        Self {
            angular_resolution: 8,
            radial_shells: 12,
            clip_start: 0.05,
            z_max: 64.0,
            radial_growth: 1.5,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RadialShell {
    pub inner: f32,
    pub center: f32,
    pub outer: f32,
}

pub struct SphericalGrid {
    pub(super) origin: [f32; 3],
    config: SphericalGridConfig,
    edges: Vec<f32>,
}

impl SphericalGrid {
    pub fn new(config: SphericalGridConfig) -> Result<Self, RaaError> {
        if !(1..=64).contains(&config.angular_resolution)
            || !(1..=64).contains(&config.radial_shells)
            || !(1.0..=4.0).contains(&config.radial_growth)
            || !config.clip_start.is_finite()
            || !config.z_max.is_finite()
            || config.clip_start < 0.0
            || config.clip_start >= config.z_max
        {
            return Err(invalid_config(
                "expected 1 to 64 angular subdivisions and shells, growth in [1, 4], and finite 0 <= clipStart < zMax",
            ));
        }

        let count = usize::from(config.radial_shells);
        let ratio = f64::from(config.radial_growth);
        let range = f64::from(config.z_max) - f64::from(config.clip_start);
        let total: f64 = (0..count).map(|i| ratio.powi(i as i32)).sum();
        let mut edges = Vec::with_capacity(count + 1);
        edges.push(config.clip_start);
        let mut edge = f64::from(config.clip_start);
        for i in 0..count - 1 {
            edge += range * ratio.powi(i as i32) / total;
            edges.push(edge as f32);
        }
        edges.push(config.z_max);
        if edges.windows(2).any(|edge| edge[0] >= edge[1]) {
            return Err(invalid_config(
                "radial shell boundaries must be distinct in f32",
            ));
        }
        Ok(Self {
            origin: [0.0; 3],
            config,
            edges,
        })
    }

    pub fn origin(&self) -> [f32; 3] {
        self.origin
    }

    pub fn angular_resolution(&self) -> u16 {
        self.config.angular_resolution
    }

    pub fn shell_count(&self) -> u16 {
        self.config.radial_shells
    }

    pub fn radial_shell(&self, index: u16) -> Option<RadialShell> {
        let index = usize::from(index);
        let (&inner, &outer) = (self.edges.get(index)?, self.edges.get(index + 1)?);
        Some(RadialShell {
            inner,
            center: outer - (outer - inner) * 0.5,
            outer,
        })
    }

    /// Inverse grid lookup. An interior radial boundary belongs to its outer shell.
    /// Cubemap ties choose X, then Y, then Z. The grid includes its outer boundary.
    pub fn key_at(&self, position: [f32; 3]) -> Option<CellKey> {
        let p = sub(position.map(f64::from), self.origin.map(f64::from));
        let radius = dot(p, p).sqrt();
        if !radius.is_finite()
            || radius < f64::from(self.config.clip_start)
            || radius > f64::from(self.config.z_max)
        {
            return None;
        }
        let i = self
            .edges
            .partition_point(|&edge| f64::from(edge) <= radius)
            - 1;
        let shell = (i as u16).min(self.shell_count() - 1);
        let [x, y, z] = p.map(f64::abs);
        let face = if x >= y && x >= z {
            if p[0] >= 0.0 {
                CubeFace::PosX
            } else {
                CubeFace::NegX
            }
        } else if y >= z {
            if p[1] >= 0.0 {
                CubeFace::PosY
            } else {
                CubeFace::NegY
            }
        } else if p[2] >= 0.0 {
            CubeFace::PosZ
        } else {
            CubeFace::NegZ
        };
        let [axis, u, v] = face.frame();
        let depth = dot(p, axis);
        let count = self.angular_resolution();
        let index = |basis| {
            let coord = if depth > 0.0 {
                dot(p, basis) / depth
            } else {
                0.0
            };
            (((coord + 1.0) * f64::from(count) * 0.5) as u16).min(count - 1)
        };
        Some(CellKey {
            shell,
            face,
            x: index(u),
            y: index(v),
        })
    }

    pub fn center(&self, key: CellKey) -> Option<[f32; 3]> {
        if key.x >= self.angular_resolution() || key.y >= self.angular_resolution() {
            return None;
        }
        let shell = self.radial_shell(key.shell)?;
        let step = 2.0 / f64::from(self.angular_resolution());
        let [axis, axis_u, axis_v] = key.face.frame();
        let u = (f64::from(key.x) + 0.5) * step - 1.0;
        let v = (f64::from(key.y) + 0.5) * step - 1.0;
        let ray = std::array::from_fn(|i| axis[i] + u * axis_u[i] + v * axis_v[i]);
        let factor = f64::from(shell.center) / dot(ray, ray).sqrt();
        Some(std::array::from_fn(|i| {
            (f64::from(self.origin[i]) + ray[i] * factor) as f32
        }))
    }

    pub(super) fn cells_per_shell(&self) -> u32 {
        6 * u32::from(self.angular_resolution()).pow(2)
    }

    pub(super) fn key(&self, index: u32) -> CellKey {
        let n = u32::from(self.angular_resolution());
        CellKey {
            shell: (index / self.cells_per_shell()) as u16,
            face: CubeFace::ALL[(index / (n * n) % 6) as usize],
            x: (index / n % n) as u16,
            y: (index % n) as u16,
        }
    }

    pub(super) fn ordinal(&self, key: CellKey) -> u32 {
        let n = u32::from(self.angular_resolution());
        u32::from(key.shell) * self.cells_per_shell()
            + key.face as u32 * n * n
            + u32::from(key.x) * n
            + u32::from(key.y)
    }

    pub(super) fn cell(&self, key: CellKey) -> Cell {
        let step = 2.0 / f64::from(self.angular_resolution());
        let interval = |index| {
            [
                f64::from(index) * step - 1.0,
                (f64::from(index) + 1.0) * step - 1.0,
            ]
        };
        let u = interval(key.x);
        let v = interval(key.y);
        let [axis, axis_u, axis_v] = key.face.frame();
        let project = |i, u, v| axis[i] + axis_u[i] * u + axis_v[i] * v;
        let norm_min = |[lo, hi]: [f64; 2]| {
            if lo <= 0.0 && hi >= 0.0 {
                0.0
            } else {
                lo.abs().min(hi.abs())
            }
        };
        let norm_max = |[lo, hi]: [f64; 2]| lo.abs().max(hi.abs());
        let shell = self.radial_shell(key.shell).expect("valid grid cell");
        let inner = f64::from(shell.inner);
        let outer = f64::from(shell.outer);
        let scale_min = inner / (1.0 + norm_max(u).powi(2) + norm_max(v).powi(2)).sqrt();
        let scale_max = outer / (1.0 + norm_min(u).powi(2) + norm_min(v).powi(2)).sqrt();
        let origin = self.origin.map(f64::from);
        let mut min = origin;
        let mut max = origin;
        for i in 0..3 {
            let a = project(i, u[0], v[0]);
            let b = project(i, u[1], v[1]);
            let lo = a.min(b);
            let hi = a.max(b);
            min[i] += (lo * scale_min).min(lo * scale_max);
            max[i] += (hi * scale_min).max(hi * scale_max);
            // Keep inclusive boundary contacts through roundoff in the broad phase.
            let padding = 8.0 * f64::EPSILON * (origin[i].abs() + outer);
            min[i] -= padding;
            max[i] += padding;
        }
        let plane = |basis: [f64; 3], bound: f64, sign: f64| {
            std::array::from_fn(|i| sign * (basis[i] - bound * axis[i]))
        };
        Cell {
            origin,
            min,
            max,
            inner,
            outer,
            center_radius: f64::from(shell.center),
            planes: [
                plane(axis_u, u[0], 1.0),
                plane(axis_u, u[1], -1.0),
                plane(axis_v, v[0], 1.0),
                plane(axis_v, v[1], -1.0),
            ],
        }
    }
}

pub(super) struct Cell {
    origin: [f64; 3],
    pub min: [f64; 3],
    pub max: [f64; 3],
    inner: f64,
    outer: f64,
    center_radius: f64,
    planes: [[f64; 3]; 4],
}

#[derive(Clone, Copy)]
pub(super) struct Surface {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

impl Cell {
    /// First clip the triangle against the angular cone, then intersect the radius
    /// interval of the clipped convex polygon with the shell. AABB overlap alone
    /// is insufficient for either half of this test.
    pub(super) fn intersect(&self, triangle: [[f32; 3]; 3]) -> Option<Surface> {
        let points = triangle.map(|p| sub(p.map(f64::from), self.origin));
        let normal = cross(sub(points[1], points[0]), sub(points[2], points[0]));
        let normal_sq = dot(normal, normal);
        if normal_sq == 0.0 || !normal_sq.is_finite() {
            return None;
        }
        // Clipping a convex polygon by a half-space adds at most one vertex.
        let mut polygon = [[0.0; 3]; 7];
        polygon[..3].copy_from_slice(&points);
        let mut len = 3;
        for plane in self.planes {
            let mut clipped = [[0.0; 3]; 7];
            let mut count = 0;
            let mut a = polygon[len - 1];
            let mut da = dot(plane, a);
            for &b in &polygon[..len] {
                let db = dot(plane, b);
                if (da < 0.0 && db > 0.0) || (da > 0.0 && db < 0.0) {
                    clipped[count] = lerp(a, b, da / (da - db));
                    count += 1;
                }
                if db >= 0.0 {
                    clipped[count] = b;
                    count += 1;
                }
                a = b;
                da = db;
            }
            if count == 0 {
                return None;
            }
            polygon = clipped;
            len = count;
        }

        let mut closest = polygon[0];
        let mut farthest = polygon[0];
        let mut min_sq = dot(closest, closest);
        let mut max_sq = min_sq;
        let projected = normal.map(|n| n * (dot(normal, points[0]) / normal_sq));
        let projected_inside = (0..3).all(|i| {
            let a = points[i];
            let edge = sub(points[(i + 1) % 3], a);
            dot(cross(edge, sub(projected, a)), normal) >= 0.0
        }) && self
            .planes
            .iter()
            .all(|&plane| dot(plane, projected) >= 0.0);
        let mut a = polygon[len - 1];
        for &b in &polygon[..len] {
            let b_sq = dot(b, b);
            if b_sq > max_sq {
                farthest = b;
                max_sq = b_sq;
            }
            let edge = sub(b, a);
            let edge_sq = dot(edge, edge);
            if edge_sq > 0.0 {
                let p = lerp(a, b, (-dot(a, edge) / edge_sq).clamp(0.0, 1.0));
                let d_sq = dot(p, p);
                if d_sq < min_sq {
                    closest = p;
                    min_sq = d_sq;
                }
            }
            a = b;
        }
        let projected_sq = dot(projected, projected);
        if projected_inside && projected_sq < min_sq {
            closest = projected;
            min_sq = projected_sq;
        }
        if min_sq > self.outer * self.outer || max_sq < self.inner * self.inner {
            return None;
        }

        // Radius varies continuously along this segment inside the clipped triangle.
        let target_sq = (self.center_radius * self.center_radius).clamp(min_sq, max_sq);
        let point = if target_sq == min_sq {
            closest
        } else if target_sq == max_sq {
            farthest
        } else {
            let edge = sub(farthest, closest);
            let a = dot(edge, edge);
            let b = dot(closest, edge).max(0.0);
            let c = target_sq - min_sq;
            lerp(
                closest,
                farthest,
                (c / (b + (b * b + a * c).sqrt())).min(1.0),
            )
        };
        Some(Surface {
            position: std::array::from_fn(|i| (self.origin[i] + point[i]) as f32),
            normal: normal.map(|n| (n / normal_sq.sqrt()) as f32),
        })
    }
}

pub(super) fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| a[i] - b[i])
}

pub(super) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn lerp(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t)
}
