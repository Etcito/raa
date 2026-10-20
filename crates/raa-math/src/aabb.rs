//! `Aabb`: axis-aligned bounding box.

use crate::{Mat4, Vec3};

/// Axis-aligned box from `min` to `max`. `EMPTY` (min +inf, max -inf) contains nothing
/// and is the identity of `include` and `union`.
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub const EMPTY: Aabb = Aabb {
        min: Vec3::splat(f32::INFINITY),
        max: Vec3::splat(f32::NEG_INFINITY),
    };

    /// Bounds of `points`; `EMPTY` for an empty slice.
    pub fn from_points(points: &[[f32; 3]]) -> Aabb {
        let mut b = Aabb::EMPTY;
        for p in points {
            b.include(Vec3::from_array(*p));
        }
        b
    }

    /// True when some axis has `min > max` (for example `EMPTY`).
    fn is_empty(&self) -> bool {
        self.min.x > self.max.x || self.min.y > self.max.y || self.min.z > self.max.z
    }

    /// Grows the box to contain `p`.
    pub fn include(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    pub fn union(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// True when the boxes overlap; touching faces count as intersecting.
    pub fn intersects(&self, other: &Aabb) -> bool {
        self.min.x <= other.max.x
            && other.min.x <= self.max.x
            && self.min.y <= other.max.y
            && other.min.y <= self.max.y
            && self.min.z <= other.max.z
            && other.min.z <= self.max.z
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// `max - min` (the full size, not the half size).
    pub fn extent(&self) -> Vec3 {
        self.max - self.min
    }

    /// Length of `extent`; 0 for an empty box.
    pub fn diagonal(&self) -> f32 {
        if self.is_empty() {
            0.0
        } else {
            self.extent().length()
        }
    }

    /// Bounds of the 8 transformed corners; `EMPTY` stays `EMPTY`.
    pub fn transform(&self, m: &Mat4) -> Aabb {
        if self.is_empty() {
            return Aabb::EMPTY;
        }
        let mut out = Aabb::EMPTY;
        for i in 0..8 {
            let pick = |bit: usize, lo: f32, hi: f32| if i & bit == 0 { lo } else { hi };
            let corner = Vec3::new(
                pick(1, self.min.x, self.max.x),
                pick(2, self.min.y, self.max.y),
                pick(4, self.min.z, self.max.z),
            );
            out.include(m.transform_point(corner));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::approx_eq;

    #[test]
    fn from_empty_points_is_empty() {
        assert_eq!(Aabb::from_points(&[]), Aabb::EMPTY);
        assert_eq!(Aabb::EMPTY.diagonal(), 0.0);
        assert_eq!(Aabb::EMPTY.transform(&Mat4::IDENTITY), Aabb::EMPTY);
    }

    #[test]
    fn from_points_bounds_all_points() {
        let b = Aabb::from_points(&[[1.0, -2.0, 3.0], [-1.0, 4.0, 0.0], [0.5, 0.0, -5.0]]);
        assert_eq!(b.min, Vec3::new(-1.0, -2.0, -5.0));
        assert_eq!(b.max, Vec3::new(1.0, 4.0, 3.0));
        assert_eq!(b.center(), Vec3::new(0.0, 1.0, -1.0));
        assert_eq!(b.extent(), Vec3::new(2.0, 6.0, 8.0));
        assert!(approx_eq(b.diagonal(), 104.0f32.sqrt(), 1e-6));
    }

    #[test]
    fn union_with_empty_is_identity() {
        let b = Aabb::from_points(&[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]);
        assert_eq!(b.union(&Aabb::EMPTY), b);
        assert_eq!(Aabb::EMPTY.union(&b), b);
        let c = Aabb::from_points(&[[2.0, -1.0, 0.5]]);
        let u = b.union(&c);
        assert_eq!(u.min, Vec3::new(0.0, -1.0, 0.0));
        assert_eq!(u.max, Vec3::new(2.0, 1.0, 1.0));
    }

    #[test]
    fn touching_boxes_intersect_and_separate_do_not() {
        let a = Aabb::from_points(&[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]);
        let touching = Aabb::from_points(&[[1.0, 0.0, 0.0], [2.0, 1.0, 1.0]]);
        let apart = Aabb::from_points(&[[1.5, 0.0, 0.0], [2.0, 1.0, 1.0]]);
        assert!(a.intersects(&touching));
        assert!(touching.intersects(&a));
        assert!(!a.intersects(&apart));
        assert!(!a.intersects(&Aabb::EMPTY));
    }

    #[test]
    fn transform_by_translation_moves_box() {
        let a = Aabb::from_points(&[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]);
        let b = a.transform(&Mat4::from_translation(Vec3::new(1.0, -1.0, 2.0)));
        assert_eq!(b.min, Vec3::new(1.0, -1.0, 2.0));
        assert_eq!(b.max, Vec3::new(2.0, 1.0, 5.0));
    }
}
