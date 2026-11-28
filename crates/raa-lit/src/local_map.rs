//! Own rasterized shadow maps of point and spot lights with `shadowMode: 'map'`.
//!
//! These maps reuse the area-light hero layers of the shadow array (`shadow.csm` renders every
//! extra layer from its caster list, dynamic nodes included): a spot light takes one perspective
//! layer along its direction whose field of view covers its cone, a point light six 90-degree
//! cube faces (+X, -X, +Y, -Y, +Z, -Z). The local maps follow the area heroes, at layer
//! `HERO_LAYER_BASE + area_capacity + i`.
//!
//! Each frame [`LocalMapSelector::update`] measures every map light's screen coverage (the
//! hero metric, [`screen_coverage`]: the fraction of the screen its range sphere covers) and
//! gives layers, most covering first, to the lights at or above their `shadow_min_screen`
//! while the capacity lasts. A selected light's height lane holds `-(first layer)`; the others
//! keep `shadow slot + 1` (their traced cube, the fallback) or 0. The shadow slots of the
//! selected lights leave the traced pass's list, so a moving light with a map is not traced too.
//!
//! The capacity (the layers the array holds) is the sum of every map light's faces, capped at
//! [`MAX_LOCAL_MAP_LAYERS`], so crossing the threshold never reallocates the array.

use raa_math::{Mat4, Vec3};
use raa_pipeline::LocalShadowSlot;
use raa_shader::{LIGHT_GPU_SIZE, MAX_LOCAL_LIGHTS};

use crate::area_shadow::{AreaShadowCamera, HERO_LAYER_BASE, HERO_NEAR, screen_coverage};
use crate::light_math;
use crate::local_light::{LocalLightKind, LocalShadowMode};
use crate::local_lights::LocalLights;

/// Largest number of shadow-array layers the local maps use (two point-light cubes).
pub const MAX_LOCAL_MAP_LAYERS: usize = 12;

/// Tangent of the half field of view of a point light's cube face: a little over 45 degrees, so
/// the PCF kernel near a face edge still reads inside the face (`RAA_LOCAL_MAP_CUBE_TAN` in WGSL).
pub const LOCAL_MAP_CUBE_TAN: f32 = 1.05;

/// Largest tangent of a spot map's half field of view (70 degrees, as the area hero maps).
pub const LOCAL_MAP_MAX_TAN: f32 = 2.747_477_4;

/// The cube face directions, in layer order.
const FACES: [Vec3; 6] = [
    Vec3::new(1.0, 0.0, 0.0),
    Vec3::new(-1.0, 0.0, 0.0),
    Vec3::new(0.0, 1.0, 0.0),
    Vec3::new(0.0, -1.0, 0.0),
    Vec3::new(0.0, 0.0, 1.0),
    Vec3::new(0.0, 0.0, -1.0),
];

/// Tangent of a spot map's half field of view from the record's outer half-angle cosine: the cone
/// widened by 10 percent plus 0.02 for the PCF kernel, at most [`LOCAL_MAP_MAX_TAN`]. The WGSL
/// side evaluates the same expression from `direction_cos_outer.w`.
pub fn spot_map_tan(cos_outer: f32) -> f32 {
    let c = cos_outer.clamp(1e-4, 1.0);
    ((1.0 - c * c).max(0.0).sqrt() / c * 1.1 + 0.02).min(LOCAL_MAP_MAX_TAN)
}

/// View-projection of one local map layer: a square perspective of half-angle tangent `tan_half`
/// looking along `direction` from `position`, depth 0 at [`HERO_NEAR`] and 1 at `range`, with the
/// basis the area hero maps use (`raa_area_basis` in WGSL).
pub fn local_map_view_proj(position: Vec3, direction: Vec3, tan_half: f32, range: f32) -> Mat4 {
    let far = range.max(2.0 * HERO_NEAR);
    let (_, bitangent) = light_math::rectangle_basis(direction);
    let projection = Mat4::perspective_rh_zo(2.0 * tan_half.atan(), 1.0, HERO_NEAR, far);
    let view = Mat4::look_at_rh(position, position + direction, bitangent);
    projection.mul(&view)
}

/// The record's own `cos_outer`, as `LocalLight::write_gpu_tiered` writes it.
fn cos_outer(angle_deg: f32) -> f32 {
    (angle_deg.to_radians() * 0.5).cos()
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    index: usize,
    coverage: f32,
    faces: usize,
}

/// Per-view selector of the local maps; allocates its scratch once.
#[derive(Debug, Default)]
pub struct LocalMapSelector {
    candidates: Vec<Candidate>,
    selected: Vec<(usize, u32)>,
    active: u32,
}

impl LocalMapSelector {
    pub fn new() -> Self {
        LocalMapSelector {
            candidates: Vec::with_capacity(MAX_LOCAL_LIGHTS),
            selected: Vec::with_capacity(MAX_LOCAL_MAP_LAYERS),
            active: 0,
        }
    }

    /// Lights that had their own map in the last update.
    pub fn active(&self) -> u32 {
        self.active
    }

    /// Selects this frame's local maps. `records` is the `LocalLights::write_gpu` output of
    /// `lights`; `slots` its `shadow_slots()`. Writes the height lane (offset 60) of every
    /// shadow-casting point or spot record, fills `targets[i]` with the view-projection of local
    /// layer `i` (None when unused) and returns `(capacity, changed, traced slots)`: the
    /// layers the array must hold, whether a record byte changed, and the slots the traced pass
    /// keeps (those of lights without a map this frame).
    pub fn update(
        &mut self,
        camera: &AreaShadowCamera,
        lights: &LocalLights,
        slots: &[LocalShadowSlot],
        area_capacity: u32,
        records: &mut [u8; MAX_LOCAL_LIGHTS * LIGHT_GPU_SIZE],
        targets: &mut [Option<Mat4>; MAX_LOCAL_MAP_LAYERS],
    ) -> (u32, bool, Vec<LocalShadowSlot>) {
        targets.fill(None);
        self.candidates.clear();
        self.selected.clear();
        let mut capacity = 0usize;
        for (index, _, light) in lights.world_lights() {
            let faces = match light.kind {
                LocalLightKind::Point => 6,
                LocalLightKind::Spot => 1,
                _ => continue,
            };
            if !light.shadows || light.shadow_mode != LocalShadowMode::Map || light.is_off() {
                continue;
            }
            if capacity + faces > MAX_LOCAL_MAP_LAYERS {
                continue;
            }
            capacity += faces;
            let coverage = screen_coverage(camera, Vec3::from_array(light.position), light.range);
            if coverage > 0.0 && coverage >= light.shadow_min_screen {
                self.candidates.push(Candidate {
                    index,
                    coverage,
                    faces,
                });
            }
        }
        self.candidates.sort_unstable_by(|a, b| {
            b.coverage
                .total_cmp(&a.coverage)
                .then(a.index.cmp(&b.index))
        });
        let mut used = 0usize;
        for candidate in &self.candidates {
            if used + candidate.faces <= capacity {
                self.selected.push((candidate.index, used as u32));
                used += candidate.faces;
            }
        }
        self.active = self.selected.len() as u32;
        let base = HERO_LAYER_BASE + area_capacity;
        let mut changed = false;
        for (index, _, light) in lights.world_lights() {
            if !matches!(light.kind, LocalLightKind::Point | LocalLightKind::Spot) {
                continue;
            }
            let lane = index * LIGHT_GPU_SIZE + 60;
            let value = if let Some(&(_, layer)) = self.selected.iter().find(|(i, _)| *i == index) {
                let position = Vec3::from_array(light.position);
                if matches!(light.kind, LocalLightKind::Spot) {
                    let tan = spot_map_tan(cos_outer(light.angle_deg));
                    let direction = Vec3::from_array(light.direction);
                    targets[layer as usize] =
                        Some(local_map_view_proj(position, direction, tan, light.range));
                } else {
                    for (face, direction) in FACES.iter().enumerate() {
                        targets[layer as usize + face] = Some(local_map_view_proj(
                            position,
                            *direction,
                            LOCAL_MAP_CUBE_TAN,
                            light.range,
                        ));
                    }
                }
                -((base + layer) as f32)
            } else {
                slots
                    .iter()
                    .find(|s| s.index as usize == index)
                    .map_or(0.0, |s| (s.slot + 1) as f32)
            };
            if let Some(bytes) = records.get_mut(lane..lane + 4) {
                let new = value.to_le_bytes();
                if bytes != new {
                    bytes.copy_from_slice(&new);
                    changed = true;
                }
            }
        }
        let traced = slots
            .iter()
            .filter(|s| !self.selected.iter().any(|(i, _)| *i == s.index as usize))
            .copied()
            .collect();
        (capacity as u32, changed, traced)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spot_tan_covers_the_cone_and_is_capped() {
        let c = cos_outer(60.0);
        let t = spot_map_tan(c);
        assert!(t > (30f32).to_radians().tan());
        assert!(spot_map_tan(cos_outer(179.0)) <= LOCAL_MAP_MAX_TAN);
    }

    #[test]
    fn cube_face_projects_its_axis_to_the_centre() {
        let m = local_map_view_proj(
            Vec3::ZERO,
            Vec3::new(0.0, -1.0, 0.0),
            LOCAL_MAP_CUBE_TAN,
            8.0,
        );
        let p = m.mul_vec4(raa_math::Vec4::new(0.0, -2.0, 0.0, 1.0));
        assert!((p.x / p.w).abs() < 1e-5 && (p.y / p.w).abs() < 1e-5);
        assert!(p.z / p.w > 0.0 && p.z / p.w < 1.0);
    }
}
