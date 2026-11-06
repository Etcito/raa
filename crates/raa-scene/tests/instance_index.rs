//! Deterministic random scenes compare the BVH queries with brute force.

use raa_math::{Aabb, Mat4, Vec3};
use raa_scene::{InstanceIndex, frustum_planes};

/// Deterministic 64-bit LCG (no `rand` crate).
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9e37_79b9_7f4a_7c15,
        }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 33) as u32
    }

    fn unit(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }

    fn below(&mut self, n: u32) -> u32 {
        self.next_u32() % n
    }
}

fn random_box(rng: &mut Lcg) -> Aabb {
    let c = Vec3::new(
        rng.range(-100.0, 100.0),
        rng.range(-100.0, 100.0),
        rng.range(-100.0, 100.0),
    );
    let h = Vec3::new(
        rng.range(0.25, 3.0),
        rng.range(0.25, 3.0),
        rng.range(0.25, 3.0),
    );
    Aabb {
        min: c - h,
        max: c + h,
    }
}

fn random_view(rng: &mut Lcg) -> Mat4 {
    let eye = Vec3::new(
        rng.range(-250.0, 250.0),
        rng.range(30.0, 200.0),
        rng.range(-250.0, 250.0),
    );
    let target = Vec3::new(
        rng.range(-60.0, 60.0),
        rng.range(-40.0, 40.0),
        rng.range(-60.0, 60.0),
    );
    let view = Mat4::look_at_rh(eye, target, Vec3::Y);
    let proj = Mat4::perspective_rh_zo(60.0f32.to_radians(), 16.0 / 9.0, 0.5, 1000.0);
    proj.mul(&view)
}

fn random_ray(rng: &mut Lcg) -> (Vec3, Vec3, f32) {
    let origin = Vec3::new(
        rng.range(-150.0, 150.0),
        rng.range(-150.0, 150.0),
        rng.range(-150.0, 150.0),
    );
    let mut dir = Vec3::new(
        rng.range(-1.0, 1.0),
        rng.range(-1.0, 1.0),
        rng.range(-1.0, 1.0),
    );
    if dir.length_squared() < 1e-6 {
        dir = Vec3::X;
    }
    (origin, dir.normalize_or_zero(), 500.0)
}

fn valid(b: &Aabb) -> bool {
    [b.min.x, b.min.y, b.min.z, b.max.x, b.max.y, b.max.z]
        .iter()
        .all(|v| v.is_finite())
        && b.min.x <= b.max.x
        && b.min.y <= b.max.y
        && b.min.z <= b.max.z
}

/// Same rule as raa-lit `skin_bounds::visible`: out when every corner is behind
/// one plane (largest signed distance < 0).
fn outside(b: &Aabb, planes: &[[f32; 4]]) -> bool {
    if !valid(b) {
        return false;
    }
    planes.iter().any(|p| {
        let pick = |v: f32, lo: f32, hi: f32| v * if v >= 0.0 { hi } else { lo };
        let d = pick(p[0], b.min.x, b.max.x)
            + pick(p[1], b.min.y, b.max.y)
            + pick(p[2], b.min.z, b.max.z)
            + p[3];
        d.is_finite() && d < 0.0
    })
}

fn slab(b: &Aabb, origin: Vec3, dir: Vec3, max_t: f32) -> Option<f32> {
    if !valid(b) {
        return None;
    }
    let mut tmin = 0.0f32;
    let mut tmax = max_t;
    let axes = [
        (origin.x, dir.x, b.min.x, b.max.x),
        (origin.y, dir.y, b.min.y, b.max.y),
        (origin.z, dir.z, b.min.z, b.max.z),
    ];
    for (o, d, lo, hi) in axes {
        if d.abs() < 1e-20 {
            if o < lo || o > hi {
                return None;
            }
        } else {
            let inv = 1.0 / d;
            let mut t1 = (lo - o) * inv;
            let mut t2 = (hi - o) * inv;
            if t1 > t2 {
                core::mem::swap(&mut t1, &mut t2);
            }
            tmin = tmin.max(t1);
            tmax = tmax.min(t2);
            if tmin > tmax {
                return None;
            }
        }
    }
    if tmax < 0.0 { None } else { Some(tmin) }
}

type Entries = Vec<(u32, Option<Aabb>)>;

fn brute_frustum(entries: &Entries, planes: &[[f32; 4]]) -> Vec<u32> {
    let mut out: Vec<u32> = entries
        .iter()
        .filter(|(_, b)| match b {
            None => true,
            Some(b) => !outside(b, planes),
        })
        .map(|(slot, _)| *slot)
        .collect();
    out.sort_unstable();
    out
}

fn brute_ray(entries: &Entries, origin: Vec3, dir: Vec3, max_t: f32) -> Vec<(u32, f32)> {
    let dir = dir.normalize_or_zero();
    let mut out = Vec::new();
    if dir.length_squared() == 0.0 {
        return out;
    }
    for (slot, b) in entries {
        if let Some(b) = b {
            if let Some(t) = slab(b, origin, dir, max_t) {
                out.push((*slot, t));
            }
        }
    }
    out.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    out
}

/// 2000 random boxes in a 200 m cube plus 8 unbounded slots at 2000..2008.
fn build_scene() -> (InstanceIndex, Entries) {
    let mut rng = Lcg::new(0x1234_5678);
    let mut index = InstanceIndex::new();
    let mut entries = Vec::new();
    for slot in 0..2000u32 {
        let b = random_box(&mut rng);
        index.insert(slot, Some(b));
        entries.push((slot, Some(b)));
    }
    for k in 0..8u32 {
        let slot = 2000 + k;
        index.insert(slot, None);
        entries.push((slot, None));
    }
    (index, entries)
}

fn snapshot(index: &InstanceIndex, max_slot: u32) -> Entries {
    (0..max_slot)
        .filter_map(|slot| index.bounds(slot).map(|b| (slot, b)))
        .collect()
}

fn assert_frustum_matches(index: &InstanceIndex, entries: &Entries, planes: &[[f32; 4]]) {
    let mut out = Vec::new();
    index.query_frustum(planes, &mut out);
    let brute = brute_frustum(entries, planes);
    assert_eq!(out, brute);
}

fn assert_ray_matches(
    index: &InstanceIndex,
    entries: &Entries,
    origin: Vec3,
    dir: Vec3,
    max_t: f32,
) {
    let mut out = Vec::new();
    index.query_ray(origin, dir, max_t, &mut out);
    let brute = brute_ray(entries, origin, dir, max_t);
    assert_eq!(out.len(), brute.len());
    for (a, b) in out.iter().zip(&brute) {
        assert_eq!(a.0, b.0, "ray hit slots differ");
        assert!(
            (a.1 - b.1).abs() < 1e-4,
            "entry t differs: {} vs {}",
            a.1,
            b.1
        );
    }
}

#[test]
fn frustum_matches_brute_force() {
    let (index, entries) = build_scene();
    let mut rng = Lcg::new(0xabcd_0001);
    for _ in 0..20 {
        let planes = frustum_planes(random_view(&mut rng));
        assert_frustum_matches(&index, &entries, &planes);
    }
}

#[test]
fn ray_matches_brute_force() {
    let (index, entries) = build_scene();
    let mut rng = Lcg::new(0xabcd_0002);
    for _ in 0..50 {
        let (origin, dir, max_t) = random_ray(&mut rng);
        assert_ray_matches(&index, &entries, origin, dir, max_t);
    }
}

#[test]
fn queries_match_after_mixed_operations() {
    let (mut index, _) = build_scene();
    let mut rng = Lcg::new(0xabcd_0003);
    for _ in 0..500 {
        let slot = rng.below(2000);
        match rng.below(3) {
            0 => {
                let bounds = if rng.below(10) == 0 {
                    None
                } else {
                    Some(random_box(&mut rng))
                };
                index.insert(slot, bounds);
            }
            1 => {
                index.remove(slot);
            }
            _ => {
                if rng.below(2) == 0 {
                    if let Some(current) = index.bounds(slot) {
                        assert!(!index.update(slot, current));
                    }
                } else {
                    let bounds = if rng.below(10) == 0 {
                        None
                    } else {
                        Some(random_box(&mut rng))
                    };
                    index.update(slot, bounds);
                }
            }
        }
    }

    let entries = snapshot(&index, 2008);
    assert_eq!(index.len(), entries.len());
    let mut view_rng = Lcg::new(0xabcd_0004);
    for _ in 0..20 {
        let planes = frustum_planes(random_view(&mut view_rng));
        assert_frustum_matches(&index, &entries, &planes);
    }
    let mut ray_rng = Lcg::new(0xabcd_0005);
    for _ in 0..50 {
        let (origin, dir, max_t) = random_ray(&mut ray_rng);
        assert_ray_matches(&index, &entries, origin, dir, max_t);
    }
}

#[test]
fn unbounded_slots_pass_frustum_and_never_hit_rays() {
    let (mut index, _) = build_scene();
    for k in 0..8u32 {
        index.insert(2000 + k, None);
    }
    let mut rng = Lcg::new(0xabcd_0006);
    let mut frustum_out = Vec::new();
    let mut ray_out = Vec::new();
    for _ in 0..10 {
        let planes = frustum_planes(random_view(&mut rng));
        index.query_frustum(&planes, &mut frustum_out);
        for k in 0..8u32 {
            assert!(frustum_out.contains(&(2000 + k)));
        }
        let (origin, dir, max_t) = random_ray(&mut rng);
        index.query_ray(origin, dir, max_t, &mut ray_out);
        for k in 0..8u32 {
            assert!(!ray_out.iter().any(|&(slot, _)| slot == 2000 + k));
        }
    }
}

#[test]
fn update_with_equal_bounds_returns_false() {
    let mut index = InstanceIndex::new();
    let first = Aabb::from_points(&[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]);
    let second = Aabb::from_points(&[[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]);
    index.insert(7, Some(first));
    assert!(!index.update(7, Some(first)));
    index.insert(9, None);
    assert!(!index.update(9, None));
    assert!(!index.update(42, Some(first)));
    assert!(index.update(7, Some(second)));
    assert_eq!(index.bounds(7), Some(Some(second)));
    assert!(index.update(7, None));
    assert_eq!(index.bounds(7), Some(None));
    assert_eq!(index.len(), 2);
}

#[test]
fn frustum_planes_perspective_rejects_behind_keeps_front() {
    let proj = Mat4::perspective_rh_zo(60.0f32.to_radians(), 1.0, 0.1, 100.0);
    let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), Vec3::Y);
    let planes = frustum_planes(proj.mul(&view));
    let behind = Aabb::from_points(&[[-1.0, -1.0, 10.0], [1.0, 1.0, 12.0]]);
    let front = Aabb::from_points(&[[-1.0, -1.0, -12.0], [1.0, 1.0, -10.0]]);
    assert!(outside(&behind, &planes));
    assert!(!outside(&front, &planes));

    let mut index = InstanceIndex::new();
    index.insert(0, Some(behind));
    index.insert(1, Some(front));
    let mut out = Vec::new();
    index.query_frustum(&planes, &mut out);
    assert_eq!(out, vec![1]);
}
