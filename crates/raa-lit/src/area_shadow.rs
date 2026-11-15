//! CPU base of the M4 area-light soft-shadow tiers: the settings record, its validation,
//! the per-tier visibility weights and the hero shadow-map projection.
//!
//! Every area light is shaded as `U x V x fade`; the tiers differ only in `V`. This module
//! supplies the settings ([`AreaShadowSettings`]), the tier selector ([`AreaShadowTier`]) and
//! the pure maths ([`tier_weights`], [`rect_distance`], [`hero_view_proj`]) that the renderer,
//! the hero selector and the shadow pass share. It allocates nothing and touches no GPU state.

use raa_core::RaaError;
use raa_math::{Mat4, Vec3};
use raa_shader::{LIGHT_GPU_SIZE, MAX_LOCAL_LIGHTS};

use crate::light_math;
use crate::local_light::{LightHandle, LocalLightKind};
use crate::local_lights::LocalLights;

/// Largest number of hero shadow-map lights in one frame.
pub const MAX_HERO_LIGHTS: usize = 4;

/// First shadow-array layer used by a hero light: hero slot `h` renders layer `4 + h`.
pub const HERO_LAYER_BASE: u32 = 4;

/// Vertical field of view of a hero shadow map, 140 degrees in radians (square map, aspect 1).
///
/// The tangent of the half angle is 2.7474774.
pub const HERO_FOV_Y_RAD: f32 = 2.443_461;

/// Near plane of a hero shadow map, in metres.
pub const HERO_NEAR: f32 = 0.05;

/// Which area-light shadow tier the settings select.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum AreaShadowTier {
    /// Hero shadow maps for the nearer lights, the cone further out and the far fade.
    #[default]
    Auto,
    /// Only the nearest lights get a hero shadow map; the rest keep the cone.
    Hero,
    /// The cone visibility only, with no hero map and no fade.
    Cone,
    /// The direct term fades to zero over the fade band; the cone stays.
    Fade,
}

impl AreaShadowTier {
    /// The stable lower-case name of the tier.
    pub fn as_str(self) -> &'static str {
        match self {
            AreaShadowTier::Auto => "auto",
            AreaShadowTier::Hero => "hero",
            AreaShadowTier::Cone => "cone",
            AreaShadowTier::Fade => "fade",
        }
    }

    /// Parses one of `auto`, `hero`, `cone` or `fade`; any other string is `None`.
    pub fn from_name(name: &str) -> Option<AreaShadowTier> {
        match name {
            "auto" => Some(AreaShadowTier::Auto),
            "hero" => Some(AreaShadowTier::Hero),
            "cone" => Some(AreaShadowTier::Cone),
            "fade" => Some(AreaShadowTier::Fade),
            _ => None,
        }
    }
}

/// Area-light soft-shadow settings shared by the renderer and the hero selector.
///
/// Distances are metres and `hero_count` is the number of hero shadow layers requested.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AreaShadowSettings {
    /// The tier selector.
    pub tier: AreaShadowTier,
    /// Number of hero shadow-map lights, at most [`MAX_HERO_LIGHTS`].
    pub hero_count: u32,
    /// Distance from the light at which the hero weight starts to fade, metres.
    pub hero_distance: f32,
    /// Distance at which the direct term has fully faded, metres.
    pub fade_distance: f32,
    /// Width of the smoothstep band at each distance, metres.
    pub band: f32,
}

impl Default for AreaShadowSettings {
    fn default() -> Self {
        AreaShadowSettings {
            tier: AreaShadowTier::Auto,
            hero_count: 0,
            hero_distance: 25.0,
            fade_distance: 80.0,
            band: 5.0,
        }
    }
}

impl AreaShadowSettings {
    /// Validates the settings in contract order, returning the first refusal.
    ///
    /// The refusals use `RaaError::FrameGraph` with code `light.invalid_area_shadows` and a
    /// `<key>: <reason>` message.
    pub fn validate(self) -> Result<AreaShadowSettings, RaaError> {
        let refuse = |message: String| RaaError::FrameGraph {
            code: "light.invalid_area_shadows",
            message,
        };
        if self.hero_count as usize > MAX_HERO_LIGHTS {
            return Err(refuse(format!(
                "heroCount: must be an integer in [0, 4] (got {})",
                self.hero_count
            )));
        }
        if !self.hero_distance.is_finite() || self.hero_distance <= 0.0 {
            return Err(refuse(format!(
                "heroDistance: must be finite and > 0 (got {})",
                self.hero_distance
            )));
        }
        if !self.fade_distance.is_finite() || self.fade_distance <= self.hero_distance {
            return Err(refuse(format!(
                "fadeDistance: must be finite and > heroDistance (got {})",
                self.fade_distance
            )));
        }
        if !self.band.is_finite() || self.band <= 0.0 || self.band > self.hero_distance {
            return Err(refuse(format!(
                "band: must be finite, > 0 and <= heroDistance (got {})",
                self.band
            )));
        }
        Ok(self)
    }
}

/// Smoothstep `s(e0, e1, x) = t * t * (3 - 2 t)` with `t = clamp((x - e0) / (e1 - e0), 0, 1)`.
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The visibility weights of one area light: `(w_cone, w_direct)`, each in `[0, 1]`.
///
/// `w_cone` blends the hero shadow map with the cone visibility and `w_direct` scales the
/// direct term. `hero` says whether the light holds a hero slot; `distance` is metres.
pub fn tier_weights(settings: &AreaShadowSettings, hero: bool, distance: f32) -> (f32, f32) {
    match settings.tier {
        AreaShadowTier::Auto => {
            let w_cone = if hero {
                smoothstep(
                    settings.hero_distance - settings.band,
                    settings.hero_distance,
                    distance,
                )
            } else {
                1.0
            };
            let w_direct = 1.0
                - smoothstep(
                    settings.fade_distance - settings.band,
                    settings.fade_distance,
                    distance,
                );
            (w_cone, w_direct)
        }
        AreaShadowTier::Hero => (if hero { 0.0 } else { 1.0 }, 1.0),
        AreaShadowTier::Cone => (1.0, 1.0),
        AreaShadowTier::Fade => (1.0, 0.0),
    }
}

/// Distance from `eye` to the light's representative point, in metres.
///
/// The representative point is [`light_math::representative_point`]: the eye clamped onto the
/// light rectangle along its basis.
pub fn rect_distance(eye: Vec3, position: Vec3, direction: Vec3, size: [f32; 2]) -> f32 {
    (light_math::representative_point(position, direction, size, eye) - eye).length()
}

/// The hero shadow-map view-projection of one area light.
///
/// It is `perspective_rh_zo(HERO_FOV_Y_RAD, 1, HERO_NEAR, far) * look_at_rh(position,
/// position + direction, bitangent)` with `far = range.max(2 * HERO_NEAR)` and
/// `(_, bitangent) = rectangle_basis(direction)`. A receiver at `dz = HERO_NEAR` maps to depth
/// 0 and one at `dz = far` maps to depth 1.
///
/// Handoff: `pub fn hero_view_proj(position: Vec3, direction: Vec3, range: f32) -> Mat4` lives in
/// `crates/raa-lit/src/area_shadow.rs`, re-exported by `crates/raa-lit/src/lib.rs`. The card
/// 'raa-lit: hero selection by screen coverage with hysteresis and the area-light lanes' calls
/// [`tier_weights`], [`rect_distance`] and [`hero_view_proj`] from `HeroSelector` and the
/// area-light lanes; this module passes no registry state or keys, only the settings values and
/// the per-light distance and range.
pub fn hero_view_proj(position: Vec3, direction: Vec3, range: f32) -> Mat4 {
    let far = range.max(2.0 * HERO_NEAR);
    let (_, bitangent) = light_math::rectangle_basis(direction);
    let projection = Mat4::perspective_rh_zo(HERO_FOV_Y_RAD, 1.0, HERO_NEAR, far);
    let view = Mat4::look_at_rh(position, position + direction, bitangent);
    projection.mul(&view)
}

/// The camera that hero selection measures screen coverage from.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AreaShadowCamera {
    /// Eye position, metres.
    pub eye: Vec3,
    /// Unit view direction.
    pub forward: Vec3,
    /// Vertical field of view, radians.
    pub fov_y_rad: f32,
    /// Viewport width and height, pixels.
    pub viewport: (u32, u32),
}

/// Fraction of the screen covered by a sphere of radius `range` at `position`, in `[0, 1]`.
///
/// A sphere that contains the eye covers the whole screen (1). One lying wholly behind the
/// camera, beyond its range, covers nothing (0). Otherwise its projected radius is
/// `r_px = focal * range / sqrt(dist^2 - range^2)` with `focal = h / (2 tan(fov_y / 2))`, and
/// the coverage is `min(1, pi * r_px^2 / (w * h))`, where `w` and `h` are at least 1.
pub fn screen_coverage(camera: &AreaShadowCamera, position: Vec3, range: f32) -> f32 {
    let offset = position - camera.eye;
    let dist = offset.length();
    if dist <= range {
        return 1.0;
    }
    if offset.dot(camera.forward) < -range {
        return 0.0;
    }
    let width = camera.viewport.0.max(1) as f32;
    let height = camera.viewport.1.max(1) as f32;
    let focal = height / (2.0 * (camera.fov_y_rad * 0.5).tan());
    let radius_px = focal * range / (dist * dist - range * range).sqrt();
    (std::f32::consts::PI * radius_px * radius_px / (width * height)).min(1.0)
}

/// Coverage ratio a waiting light must exceed to replace the weakest hero.
const HERO_SWAP_RATIO: f32 = 1.25;

/// One area light that may hold a slot in the current call.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Candidate {
    pub(crate) handle: LightHandle,
    pub(crate) coverage: f32,
}

/// Slot assignment shared by the hero selector and the mid-tier stochastic selector.
///
/// Handoff: `SlotAssigner` in `crates/raa-lit/src/area_shadow.rs` is attached by card
/// 'raa-lit: StochasticSelector picks the mid-tier area lights and writes the slot table', which
/// calls `SlotAssigner::new`, `SlotAssigner::assign`, `SlotAssigner::slots` and
/// `SlotAssigner::active` with `Candidate` values. Today `assign` takes candidates already
/// sorted by coverage descending then handle ascending, releases every slot past the clamped
/// `limit` or whose handle is no longer a candidate, fills the lowest free index below the limit
/// with the best unheld candidate, and performs at most one swap when the best unheld
/// candidate's coverage exceeds the weakest held slot's by more than `HERO_SWAP_RATIO`; it
/// allocates nothing.
#[derive(Debug)]
pub(crate) struct SlotAssigner<const N: usize> {
    slots: [Option<LightHandle>; N],
    coverages: [f32; N],
}

impl<const N: usize> SlotAssigner<N> {
    /// Creates an empty assigner: every slot is `None` and every coverage is 0.0.
    pub(crate) fn new() -> SlotAssigner<N> {
        SlotAssigner {
            slots: [None; N],
            coverages: [0.0; N],
        }
    }

    /// Number of occupied slots after the last [`SlotAssigner::assign`].
    pub(crate) fn active(&self) -> u32 {
        self.slots.iter().filter(|slot| slot.is_some()).count() as u32
    }

    /// The slot handles of the last [`SlotAssigner::assign`].
    pub(crate) fn slots(&self) -> &[Option<LightHandle>; N] {
        &self.slots
    }

    /// The index of the slot holding `handle`, or `None` when it holds no slot.
    pub(crate) fn slot_of(&self, handle: LightHandle) -> Option<usize> {
        self.slots().iter().position(|held| *held == Some(handle))
    }

    /// Assigns the slots of one call.
    ///
    /// `candidates` are already sorted by the caller, coverage descending then handle
    /// ascending; `limit` is clamped to `N`. The call releases the slots whose light is no
    /// longer a candidate or whose index is past the limit, fills the free slots below the limit
    /// with the best unheld candidates, and performs at most one swap when the best unheld
    /// candidate's coverage exceeds the weakest held slot's by more than [`HERO_SWAP_RATIO`].
    /// Allocates nothing.
    pub(crate) fn assign(&mut self, candidates: &[Candidate], limit: usize) {
        let limit = limit.min(N);

        // Release the slots whose light is no longer a candidate or whose index is past the limit.
        for (index, slot) in self.slots.iter_mut().enumerate() {
            let kept = index < limit
                && slot.is_some_and(|handle| {
                    candidates
                        .iter()
                        .any(|candidate| candidate.handle == handle)
                });
            if !kept {
                *slot = None;
            }
        }

        // Fill the free slots below the limit, lowest index first, best candidate first.
        for index in 0..limit {
            if matches!(self.slots.get(index), Some(None))
                && let Some(pick) = self.unheld(candidates)
                && let Some(slot) = self.slots.get_mut(index)
            {
                *slot = Some(pick.handle);
            }
        }

        // Refresh the coverages from the candidates of this call.
        for (slot, coverage) in self.slots.iter().zip(self.coverages.iter_mut()) {
            *coverage = slot
                .and_then(|handle| {
                    candidates
                        .iter()
                        .find(|candidate| candidate.handle == handle)
                })
                .map_or(0.0, |candidate| candidate.coverage);
        }

        // With every slot below the limit taken, the best waiting light may replace the weakest
        // hero, at most once per call and only when its coverage exceeds the hero's by the ratio.
        if let Some(challenger) = self.unheld(candidates)
            && let Some((index, weakest)) = self.weakest(limit)
            && challenger.coverage > HERO_SWAP_RATIO * weakest
        {
            if let Some(slot) = self.slots.get_mut(index) {
                *slot = Some(challenger.handle);
            }
            if let Some(coverage) = self.coverages.get_mut(index) {
                *coverage = challenger.coverage;
            }
        }
    }

    /// The best candidate that holds no slot, in coverage-descending then handle-ascending order.
    fn unheld(&self, candidates: &[Candidate]) -> Option<Candidate> {
        candidates
            .iter()
            .copied()
            .find(|candidate| !self.slots.contains(&Some(candidate.handle)))
    }

    /// The occupied slot below `limit` with the lowest coverage; ties go to the higher handle.
    fn weakest(&self, limit: usize) -> Option<(usize, f32)> {
        self.slots
            .iter()
            .zip(self.coverages.iter())
            .take(limit)
            .enumerate()
            .filter_map(|(index, (slot, coverage))| slot.map(|handle| (index, handle, *coverage)))
            .min_by(|a, b| a.2.total_cmp(&b.2).then(b.1.cmp(&a.1)))
            .map(|(index, _, coverage)| (index, coverage))
    }
}

/// Chooses which area lights hold the hero shadow slots and writes the area-light lanes.
///
/// Handoff: `HeroSelector` (`new`, `update`, `active_heroes`) in `crates/raa-lit/src/area_shadow.rs`
/// is attached by card 'raa-lit: area-light shadow tiers wired into LitRenderer::prepare', which
/// calls `HeroSelector::new`, `HeroSelector::update` and `HeroSelector::active_heroes`, and
/// `LocalLights::world_lights` (`crates/raa-lit/src/local_lights.rs`) through them. Today `update`
/// reads the settings, the camera and the registry, writes the three lanes of every area record
/// and fills `heroes`; its only state is the slot handles and their coverages of the last call.
#[derive(Debug)]
pub struct HeroSelector {
    assigner: SlotAssigner<MAX_HERO_LIGHTS>,
    candidates: Vec<Candidate>,
}

impl Default for HeroSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl HeroSelector {
    /// Creates an empty selector. The candidate scratch is allocated here, never in `update`.
    pub fn new() -> HeroSelector {
        HeroSelector {
            assigner: SlotAssigner::new(),
            candidates: Vec::with_capacity(MAX_LOCAL_LIGHTS),
        }
    }

    /// Number of hero slots occupied after the last [`HeroSelector::update`].
    pub fn active_heroes(&self) -> u32 {
        self.assigner.active()
    }

    /// Selects the hero lights of this call and writes the area-light lanes of `records`.
    ///
    /// `records` holds the `LocalLights::write_gpu` output of `lights`. Each area record gets
    /// `w_direct` at offset 28, `hero_layer` at 44 and `w_cone` at 48, and `heroes[slot]` is the
    /// hero view-projection of each occupied slot, `None` elsewhere. Returns whether any byte of
    /// `records` changed. Allocates nothing.
    pub fn update(
        &mut self,
        settings: &AreaShadowSettings,
        camera: &AreaShadowCamera,
        lights: &LocalLights,
        records: &mut [u8; MAX_LOCAL_LIGHTS * LIGHT_GPU_SIZE],
        heroes: &mut [Option<Mat4>; MAX_HERO_LIGHTS],
    ) -> bool {
        let limit = (settings.hero_count as usize).min(MAX_HERO_LIGHTS);
        self.candidates.clear();
        for (_, handle, light) in lights.world_lights() {
            if !matches!(light.kind, LocalLightKind::Area) {
                continue;
            }
            let position = Vec3::from_array(light.position);
            let coverage = screen_coverage(camera, position, light.range);
            let distance = rect_distance(
                camera.eye,
                position,
                Vec3::from_array(light.direction),
                light.size,
            );
            let tier_ok = match settings.tier {
                AreaShadowTier::Hero => true,
                AreaShadowTier::Auto => distance < settings.hero_distance,
                AreaShadowTier::Cone | AreaShadowTier::Fade => false,
            };
            if limit > 0 && coverage > 0.0 && tier_ok {
                self.candidates.push(Candidate { handle, coverage });
            }
        }
        self.candidates.sort_unstable_by(|a, b| {
            b.coverage
                .total_cmp(&a.coverage)
                .then(a.handle.cmp(&b.handle))
        });

        self.assigner.assign(&self.candidates, limit);

        // Lanes of every area record, and the hero projection of every occupied slot.
        heroes.fill(None);
        let mut changed = false;
        for (index, handle, light) in lights.world_lights() {
            if !matches!(light.kind, LocalLightKind::Area) {
                continue;
            }
            let position = Vec3::from_array(light.position);
            let direction = Vec3::from_array(light.direction);
            let distance = rect_distance(camera.eye, position, direction, light.size);
            let slot = self.assigner.slot_of(handle);
            let (hero_layer, (w_cone, w_direct)) = match slot {
                Some(slot) => (
                    HERO_LAYER_BASE as f32 + slot as f32,
                    tier_weights(settings, true, distance),
                ),
                None => (-1.0, tier_weights(settings, false, distance)),
            };
            let base = index * LIGHT_GPU_SIZE;
            changed |= write_lane(records, base + 28, w_direct);
            changed |= write_lane(records, base + 44, hero_layer);
            changed |= write_lane(records, base + 48, w_cone);
            if let Some(slot) = slot
                && let Some(entry) = heroes.get_mut(slot)
            {
                *entry = Some(hero_view_proj(position, direction, light.range));
            }
        }
        changed
    }
}

/// Writes one little-endian f32 lane at `offset`, returning whether its four bytes changed.
fn write_lane(records: &mut [u8], offset: usize, value: f32) -> bool {
    let Some(lane) = records.get_mut(offset..offset + 4) else {
        return false;
    };
    let bytes = value.to_le_bytes();
    if lane.iter().zip(bytes).all(|(dst, src)| *dst == src) {
        return false;
    }
    for (dst, src) in lane.iter_mut().zip(bytes) {
        *dst = src;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_the_contract_literal() {
        let expected = AreaShadowSettings {
            tier: AreaShadowTier::Auto,
            hero_count: 0,
            hero_distance: 25.0,
            fade_distance: 80.0,
            band: 5.0,
        };
        assert_eq!(AreaShadowSettings::default(), expected);
        assert_eq!(AreaShadowSettings::default().validate(), Ok(expected));
    }

    fn candidate(handle: LightHandle, coverage: f32) -> Candidate {
        Candidate { handle, coverage }
    }

    #[test]
    fn slot_assigner_fills_the_lowest_free_index_with_the_best_candidate() {
        let mut assigner = SlotAssigner::<2>::new();
        assigner.assign(&[candidate(1, 0.9), candidate(2, 0.5)], 2);
        assert_eq!(assigner.slots(), &[Some(1), Some(2)]);
        // Handle 1 leaves: slot 0 is released while slot 1 keeps 2, and the best unheld
        // candidate fills the lowest free index.
        assigner.assign(&[candidate(3, 0.7), candidate(2, 0.5)], 2);
        assert_eq!(assigner.slots(), &[Some(3), Some(2)]);
        assert_eq!(assigner.slot_of(3), Some(0));
        assert_eq!(assigner.active(), 2);
    }

    #[test]
    fn slot_assigner_keeps_a_held_light_below_the_swap_ratio() {
        let mut assigner = SlotAssigner::<2>::new();
        assigner.assign(&[candidate(1, 0.5)], 1);
        assert_eq!(assigner.slots(), &[Some(1), None]);
        // 0.5 / 0.45 = 1.11x, below 1.25x: the held light keeps its slot.
        assigner.assign(&[candidate(2, 0.5), candidate(1, 0.45)], 1);
        assert_eq!(assigner.slots(), &[Some(1), None]);
        assert_eq!(assigner.slot_of(2), None);
        assert_eq!(assigner.active(), 1);
    }

    #[test]
    fn slot_assigner_releases_a_slot_at_or_past_the_limit() {
        let mut assigner = SlotAssigner::<2>::new();
        assigner.assign(&[candidate(1, 0.9), candidate(2, 0.8)], 2);
        assert_eq!(assigner.slots(), &[Some(1), Some(2)]);
        // The limit drops to 1: index 1 is released even though handle 2 is still a candidate.
        assigner.assign(&[candidate(1, 0.9), candidate(2, 0.8)], 1);
        assert_eq!(assigner.slots(), &[Some(1), None]);
        assert_eq!(assigner.slot_of(2), None);
        assert_eq!(assigner.active(), 1);
    }

    #[test]
    fn slot_assigner_swaps_at_most_once_past_the_ratio() {
        let mut assigner = SlotAssigner::<2>::new();
        assigner.assign(&[candidate(1, 0.2), candidate(2, 0.2)], 2);
        assert_eq!(assigner.slots(), &[Some(1), Some(2)]);
        // Both challengers clear 1.25x over the 0.2 held coverage, but only one swap happens:
        // the weakest held slot (the tie goes to the higher handle, 2) takes the best challenger.
        assigner.assign(
            &[
                candidate(3, 0.5),
                candidate(4, 0.45),
                candidate(1, 0.2),
                candidate(2, 0.2),
            ],
            2,
        );
        assert_eq!(assigner.slots(), &[Some(1), Some(3)]);
        assert_eq!(assigner.slot_of(4), None);
        assert_eq!(assigner.active(), 2);
    }
}
