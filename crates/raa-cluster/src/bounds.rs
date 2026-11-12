//! Bounding spheres of the cluster build: fitting a sphere to points, merging spheres and
//! testing containment, in the f32 precision of the packed records.
//! Adapted from solstice (MIT), Copyright (c) 2026 SamG-Coder:
//! `crates/solstice-builder/src/bounds.rs`, revision 7e465b5.

use raa_math::Vec3;

/// A bounding sphere.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

impl Sphere {
    /// Bounds `points` with Ritter's approximation seeded by the box centre: near-optimal in
    /// practice and linear in the point count. No point gives the default sphere.
    pub fn from_points(points: impl Iterator<Item = Vec3> + Clone) -> Sphere {
        let (mut min, mut max) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for p in points.clone() {
            min = min.min(p);
            max = max.max(p);
        }
        if min.x > max.x {
            return Sphere::default();
        }
        let mut sphere = Sphere {
            center: (min + max) * 0.5,
            radius: 0.0,
        };
        for p in points {
            sphere = sphere.grown_to(p);
        }
        sphere
    }

    fn grown_to(self, point: Vec3) -> Sphere {
        let offset = point - self.center;
        let distance = offset.length();
        if distance <= self.radius {
            return self;
        }
        let radius = (self.radius + distance) * 0.5;
        Sphere {
            center: self.center + offset * ((radius - self.radius) / distance),
            radius,
        }
    }

    /// The smallest sphere containing both, padded by `1 + 1e-5` so the containment still
    /// holds after rounding; a sphere that already contains the other is returned unchanged.
    pub fn merged(self, other: Sphere) -> Sphere {
        let offset = other.center - self.center;
        let distance = offset.length();
        if distance + other.radius <= self.radius {
            return self;
        }
        if distance + self.radius <= other.radius {
            return other;
        }
        let radius = (distance + self.radius + other.radius) * 0.5;
        Sphere {
            center: self.center + offset * ((radius - self.radius) / distance),
            radius: radius * (1.0 + 1.0e-5),
        }
    }

    /// Merges any number of spheres; none gives the default sphere.
    pub fn merged_all(spheres: impl IntoIterator<Item = Sphere>) -> Sphere {
        let mut iter = spheres.into_iter();
        let first = iter.next().unwrap_or_default();
        iter.fold(first, Sphere::merged)
    }

    /// True when `other` lies inside this sphere, allowing for rounding.
    pub fn contains(self, other: Sphere) -> bool {
        (other.center - self.center).length() + other.radius
            <= self.radius * (1.0 + 1.0e-3) + 1.0e-5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_points_contains_every_point() {
        let points = [
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(-4.0, 0.5, 2.0),
            Vec3::new(0.0, -7.0, 1.0),
            Vec3::new(3.0, 3.0, -3.0),
        ];
        let sphere = Sphere::from_points(points.iter().copied());
        for p in points {
            assert!((p - sphere.center).length() <= sphere.radius + 1.0e-4);
        }
        assert_eq!(Sphere::from_points(std::iter::empty()), Sphere::default());
    }

    #[test]
    fn merged_contains_both_inputs() {
        let a = Sphere {
            center: Vec3::ZERO,
            radius: 1.0,
        };
        let b = Sphere {
            center: Vec3::X * 5.0,
            radius: 2.0,
        };
        let merged = a.merged(b);
        assert!(merged.contains(a));
        assert!(merged.contains(b));
    }

    #[test]
    fn merged_with_contained_sphere_returns_the_outer_one() {
        let outer = Sphere {
            center: Vec3::ZERO,
            radius: 10.0,
        };
        let inner = Sphere {
            center: Vec3::X,
            radius: 1.0,
        };
        assert_eq!(outer.merged(inner), outer);
        assert_eq!(inner.merged(outer), outer);
        assert_eq!(Sphere::merged_all([inner, outer]), outer);
    }
}
