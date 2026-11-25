//! Invalidation for the direct lighting evaluated at spherical capture hits.

use std::hash::{DefaultHasher, Hash, Hasher};

use crate::mesh::geometry::Bounds;

use super::{
    ALL_FACES,
    grid::{dot, sub},
};

pub const SKY_LIGHT_ID: u64 = u64::MAX - 1;
pub const SUN_LIGHT_ID: u64 = u64::MAX;

#[derive(Clone, Copy, Debug, PartialEq)]
struct LocalRange {
    position: [f32; 3],
    range: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Influence {
    Point(LocalRange),
    Spot {
        extent: LocalRange,
        direction: [f32; 3],
        cone: [f32; 2],
    },
    Directional([f32; 3]),
    Area {
        extent: LocalRange,
        direction: [f32; 3],
        size: [f32; 2],
    },
}

/// A local light's stable renderer ID and the values used by mesh capture shading.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphericalLight {
    id: u64,
    rgb: [f32; 3],
    influence: Influence,
}

impl SphericalLight {
    /// Decode one dense world-space light record. Omit dark lights so a light
    /// switched on is evaluated like a newly added light.
    pub fn from_packed(id: u64, record: &[u8]) -> Option<Self> {
        if id >= SKY_LIGHT_ID || record.len() != raa_shader::LIGHT_GPU_SIZE {
            return None;
        }
        let word = |i: usize| u32::from_le_bytes(record[4 * i..4 * i + 4].try_into().unwrap());
        let lane = |i| f32::from_bits(word(i));
        let rgb = [lane(4), lane(5), lane(6)];
        if !rgb.iter().any(|&value| value > 0.0) {
            return None;
        }
        let extent = LocalRange {
            position: [lane(0), lane(1), lane(2)],
            range: lane(3),
        };
        let direction = [lane(8), lane(9), lane(10)];
        let influence = match word(13) {
            0 => Influence::Point(extent),
            1 => Influence::Spot {
                extent,
                direction,
                cone: [lane(11), lane(12)],
            },
            2 => Influence::Directional(direction),
            3 => Influence::Area {
                extent,
                direction,
                size: [lane(14), lane(15)],
            },
            _ => return None,
        };
        Some(Self { id, rgb, influence })
    }

    pub fn id(self) -> u64 {
        self.id
    }

    fn may_light(self, bounds: Bounds) -> bool {
        match self.influence {
            Influence::Point(extent) => intersects_sphere(bounds, extent, 0.0),
            Influence::Spot {
                extent,
                direction,
                cone,
            } => {
                if !intersects_sphere(bounds, extent, 0.0) {
                    return false;
                }
                // The AABB's circumsphere bounds its closest approach to the outer cone.
                let (center, half) = centered(bounds, extent.position);
                let axial = dot(center, direction.map(f64::from));
                let lateral = (dot(center, center) - axial * axial).max(0.0).sqrt();
                let radius = dot(half, half).sqrt();
                let cos = f64::from(cone[0]);
                let sin = (1.0 - cos * cos).max(0.0).sqrt();
                let padding = 1e-6 * (axial.abs() + lateral + radius);
                axial + radius + padding >= 0.0
                    && axial * sin - lateral * cos + radius + padding >= 0.0
            }
            Influence::Directional(_) => true,
            Influence::Area {
                extent,
                direction,
                size,
            } => {
                let radius = 0.5 * f64::from(size[0]).hypot(f64::from(size[1]));
                if !intersects_sphere(bounds, extent, radius) {
                    return false;
                }
                // Rect samples are in the emitter plane. The shader only evaluates its front.
                let (center, half) = centered(bounds, extent.position);
                let direction = direction.map(f64::from);
                dot(center, direction) + dot(half, direction.map(f64::abs)) >= 0.0
            }
        }
    }

    fn hash_values(self, hash: &mut impl Hasher) {
        hash_floats(self.rgb, hash);
        std::mem::discriminant(&self.influence).hash(hash);
        match self.influence {
            Influence::Point(extent) => hash_extent(extent, hash),
            Influence::Spot {
                extent,
                direction,
                cone,
            } => {
                hash_extent(extent, hash);
                hash_floats(direction, hash);
                hash_floats(cone, hash);
            }
            Influence::Directional(direction) => hash_floats(direction, hash),
            Influence::Area {
                extent,
                direction,
                size,
            } => {
                hash_extent(extent, hash);
                hash_floats(direction, hash);
                hash_floats(size, hash);
            }
        }
    }
}

fn hash_floats<const N: usize>(values: [f32; N], hash: &mut impl Hasher) {
    for value in values {
        // Match float equality for signed zero.
        (if value == 0.0 { 0 } else { value.to_bits() }).hash(hash);
    }
}

fn hash_extent(extent: LocalRange, hash: &mut impl Hasher) {
    hash_floats(extent.position, hash);
    hash_floats([extent.range], hash);
}

fn intersects_sphere(bounds: Bounds, extent: LocalRange, extra_radius: f64) -> bool {
    let radius = f64::from(extent.range) + extra_radius;
    bounds.radial_range(extent.position).0 <= radius * (1.0 + 1e-6)
}

fn centered(bounds: Bounds, origin: [f32; 3]) -> ([f64; 3], [f64; 3]) {
    let lo = bounds.min.map(f64::from);
    let hi = bounds.max.map(f64::from);
    let center = std::array::from_fn(|i| (lo[i] + hi[i]) * 0.5);
    (
        sub(center, origin.map(f64::from)),
        sub(hi, lo).map(|value| value * 0.5),
    )
}

pub(super) fn light_hash(global: u64, ids: &[u64], lights: &[SphericalLight]) -> u64 {
    let mut hash = DefaultHasher::new();
    for &id in ids {
        id.hash(&mut hash);
        if id >= SKY_LIGHT_ID {
            global.hash(&mut hash);
        } else if let Some(light) = find_light(lights, id) {
            light.hash_values(&mut hash);
        }
    }
    hash.finish()
}

fn find_light(lights: &[SphericalLight], id: u64) -> Option<SphericalLight> {
    lights
        .binary_search_by_key(&id, |light| light.id)
        .ok()
        .map(|i| lights[i])
}

#[derive(Clone, Copy)]
pub(super) struct LightChange {
    before: Option<SphericalLight>,
    after: Option<SphericalLight>,
}

impl LightChange {
    /// The light after the change when it may now reach surfaces it did not: added, switched
    /// on, moved or reshaped (range, cone, size). These need the reach test.
    pub(super) fn reach_target(&self) -> Option<SphericalLight> {
        let after = self.after?;
        match self.before {
            Some(before) if before.influence == after.influence => None,
            _ => Some(after),
        }
    }
}

pub(super) fn find_light_by_id(lights: &[SphericalLight], id: u64) -> Option<SphericalLight> {
    find_light(lights, id)
}

pub(super) fn light_changes(
    before: &[SphericalLight],
    after: &[SphericalLight],
) -> Vec<LightChange> {
    before
        .iter()
        .filter_map(|&old| {
            let new = find_light(after, old.id);
            (new != Some(old)).then_some(LightChange {
                before: Some(old),
                after: new,
            })
        })
        .chain(after.iter().filter_map(|&new| {
            find_light(before, new.id).is_none().then_some(LightChange {
                before: None,
                after: Some(new),
            })
        }))
        .collect()
}

#[derive(Default)]
pub(super) struct FaceLighting {
    faces: u8,
    hit_bounds: [Option<Bounds>; 6],
    /// Contributors reported by each face, so a light edit can reset single faces.
    face_contributors: [Vec<u64>; 6],
    pub(super) contributors: Vec<u64>,
}

impl FaceLighting {
    pub(super) fn is_known(&self) -> bool {
        self.faces == ALL_FACES
    }

    pub(super) fn record(
        &mut self,
        face: u8,
        hit_bounds: Option<Bounds>,
        contributors: &[u64],
    ) -> bool {
        if self.faces & face != 0 {
            return false;
        }
        self.faces |= face;
        let index = face.trailing_zeros() as usize;
        self.hit_bounds[index] = hit_bounds;
        let own = &mut self.face_contributors[index];
        own.clear();
        own.extend_from_slice(contributors);
        own.sort_unstable();
        own.dedup();
        self.contributors.extend_from_slice(contributors);
        self.contributors.sort_unstable();
        self.contributors.dedup();
        true
    }

    /// Drop the metadata of the faces in `bits` (they are being recaptured).
    pub(super) fn forget(&mut self, bits: u8) {
        for index in 0..6 {
            if bits & (1 << index) != 0 {
                self.hit_bounds[index] = None;
                self.face_contributors[index].clear();
            }
        }
        self.faces &= !bits;
        self.contributors.clear();
        for index in 0..6 {
            if self.faces & (1 << index) != 0 {
                self.contributors
                    .extend_from_slice(&self.face_contributors[index]);
            }
        }
        self.contributors.sort_unstable();
        self.contributors.dedup();
    }

    /// Faces (of `ready`) whose light a change can alter, per face. A face that reported a
    /// change's old light holds it; a face without metadata yet is conservative over the
    /// scene bounds. A light that is added or whose influence changed (moved, range, cone)
    /// is NOT tested here: the GPU reach test (`spherical_reach_test`) finds the faces whose
    /// stored hits it reaches. Colour or intensity alone cannot make an occluded or
    /// out-of-range light visible (the masks hold any non-zero contribution).
    pub(super) fn affected_faces(
        &self,
        ready: u8,
        changes: &[LightChange],
        scene_bounds: Option<Bounds>,
    ) -> u8 {
        let mut bits = 0;
        for index in 0..6u8 {
            let bit = 1 << index;
            if ready & bit == 0 {
                continue;
            }
            let known = self.faces & bit != 0;
            let hit = changes.iter().any(|change| {
                change.before.is_some_and(|before| {
                    if known {
                        self.face_contributors[usize::from(index)]
                            .binary_search(&before.id)
                            .is_ok()
                    } else {
                        scene_bounds.is_some_and(|bounds| before.may_light(bounds))
                    }
                })
            });
            if hit {
                bits |= bit;
            }
        }
        bits
    }

    /// Fallback of the reach test: faces whose own hit bounds the light may reach.
    pub(super) fn faces_in_bounds(
        &self,
        ready: u8,
        light: SphericalLight,
        scene_bounds: Option<Bounds>,
    ) -> u8 {
        let mut bits = 0;
        for index in 0..6 {
            let bit = 1 << index;
            if ready & bit == 0 {
                continue;
            }
            let bounds = if self.faces & bit != 0 {
                self.hit_bounds[index]
            } else {
                scene_bounds
            };
            if bounds.is_some_and(|bounds| light.may_light(bounds)) {
                bits |= bit;
            }
        }
        bits
    }

    pub(super) fn affected_by(
        &self,
        changes: &[LightChange],
        scene_bounds: Option<Bounds>,
    ) -> bool {
        let known = self.is_known();
        // Missing readback is conservative over the current BVH root.
        let may_light = |light: SphericalLight| {
            if known {
                self.hit_bounds
                    .iter()
                    .flatten()
                    .any(|&bounds| light.may_light(bounds))
            } else {
                scene_bounds.is_some_and(|bounds| light.may_light(bounds))
            }
        };
        changes.iter().any(|change| {
            if let Some(before) = change.before {
                if self.contributors.binary_search(&before.id).is_ok()
                    || (!known && may_light(before))
                {
                    return true;
                }
                if change
                    .after
                    .is_none_or(|after| before.influence == after.influence)
                {
                    // Color/intensity alone cannot make an occluded or out-of-range light visible.
                    return false;
                }
            }
            change.after.is_some_and(may_light)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn light_edits_reset_only_the_faces_that_reported_the_light() {
        let point = |id, position: [f32; 3], rgb: [f32; 3]| SphericalLight {
            id,
            rgb,
            influence: Influence::Point(LocalRange {
                position,
                range: 4.0,
            }),
        };
        let bounds = Bounds {
            min: [0.0; 3],
            max: [1.0; 3],
        };
        let mut lighting = FaceLighting::default();
        for face in 0..6u8 {
            let ids: &[u64] = if face == 2 { &[7] } else { &[] };
            assert!(lighting.record(1 << face, Some(bounds), ids));
        }
        let red = point(7, [0.5; 3], [1.0, 0.0, 0.0]);
        // A colour change resets only face 2 (the one whose capture saw light 7).
        let recolour = light_changes(&[red], &[point(7, [0.5; 3], [0.0, 0.0, 1.0])]);
        assert_eq!(lighting.affected_faces(ALL_FACES, &recolour, None), 1 << 2);
        assert!(recolour[0].reach_target().is_none());
        // A move resets face 2 now and asks the reach test about the new position.
        let moved = light_changes(&[red], &[point(7, [2.5; 3], [1.0, 0.0, 0.0])]);
        assert_eq!(lighting.affected_faces(ALL_FACES, &moved, None), 1 << 2);
        assert_eq!(moved[0].reach_target().map(SphericalLight::id), Some(7));
        // An added light resets nothing here: only the reach test can find its faces.
        let added = light_changes(&[red], &[red, point(9, [0.5; 3], [1.0; 3])]);
        assert_eq!(lighting.affected_faces(ALL_FACES, &added, None), 0);
        assert_eq!(added[0].reach_target().map(SphericalLight::id), Some(9));
        // Forgetting face 2 drops its contributors from the union.
        lighting.forget(1 << 2);
        assert!(lighting.contributors.is_empty());
        assert!(!lighting.is_known());
        // Without metadata a face is conservative over the scene bounds.
        assert_eq!(
            lighting.affected_faces(ALL_FACES, &recolour, Some(bounds)),
            1 << 2
        );
    }

    #[test]
    fn spot_and_area_candidate_bounds_cover_their_capture_support() {
        let center = [0.0; 3];
        let extent = LocalRange {
            position: center,
            range: 2.0,
        };
        let light = |influence| SphericalLight {
            id: 1,
            rgb: [1.0; 3],
            influence,
        };
        let point = |point| Bounds {
            min: point,
            max: point,
        };

        let spot = light(Influence::Spot {
            extent,
            direction: [0.0, 0.0, 1.0],
            cone: [0.8, 0.9],
        });
        assert!(spot.may_light(point([0.5, 0.0, 1.0])));
        assert!(!spot.may_light(point([1.0, 0.0, 0.5])));
        assert!(!spot.may_light(point([0.0, 0.0, -1.0])));

        let area = light(Influence::Area {
            extent,
            direction: [0.0, 0.0, 1.0],
            size: [3.0, 1.0],
        });
        // Beyond the center range but reachable from an emitting point on the rectangle.
        assert!(area.may_light(point([2.4, 0.0, 0.4])));
        assert!(!area.may_light(point([0.0, 0.0, -0.1])));
    }
}
