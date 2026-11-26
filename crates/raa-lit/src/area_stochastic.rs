//! CPU base of the M6 area-light stochastic shadows: the mid-tier settings and their validation,
//! the blend weight of the mid tier over distance, and the codec of the 80-byte slot table that
//! binding 19 of the lighting group reads.
//!
//! The mid tier traces stratified shadow rays for up to [`StochasticShadowSettings::count`]
//! area lights nearer than [`StochasticShadowSettings::distance`]. This module supplies the
//! settings ([`StochasticShadowSettings`]), the weight that blends the mid tier into the cone
//! tier ([`stochastic_weight`]) and the table bytes ([`write_table`] and its readers). It
//! allocates nothing and touches no GPU state.

use raa_core::RaaError;
use raa_math::Vec3;
use raa_pipeline::{AREA_STOCHASTIC_GPU_SIZE, MAX_STOCHASTIC_LIGHTS};
use raa_shader::{LIGHT_GPU_SIZE, MAX_LOCAL_LIGHTS};

use crate::area_shadow::{
    AreaShadowCamera, AreaShadowSettings, AreaShadowTier, Candidate, SlotAssigner, rect_distance,
    screen_coverage,
};
use crate::local_light::{LightHandle, LocalLightKind};
use crate::local_lights::LocalLights;

/// Byte offset of the packed light index of slot 0 (`lights: array<vec4<u32>, 2>`).
const LIGHTS_AT: usize = 0;

/// Byte offset of the blend weight of slot 0 (`weights: array<vec4<f32>, 2>`).
const WEIGHTS_AT: usize = 32;

/// Byte offset of `params: vec4<u32>`: x = limit, y = cells, z = 0, w = 0.
const PARAMS_AT: usize = 64;

/// Packed light index of an empty slot.
const EMPTY_SLOT: u32 = 0xFFFF_FFFF;

/// The stochastic shadow mode: the M6 mid tier (`Stratified`) or the M7 ultra tier (`Restir`).
///
/// Handoff: [`StochasticMode`] is consumed by the downstream card 'raa-js: stochasticMode key of
/// setAreaLightShadows and areaLightShadows (m28)', which parses the `stochasticMode` key with
/// [`StochasticMode::from_name`] and reports it with [`StochasticMode::as_str`]. Today the type
/// only names the two tiers: `Stratified` is the mid tier (one stratified ray per pixel and slot,
/// the existing gi.area-shadow trace) and `Restir` the ultra tier (ReSTIR DI in the same pass).
/// It holds no selection or GPU state, and the mode is never written into the 80-byte table.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum StochasticMode {
    /// The M6 mid tier: one stratified ray per pixel and slot.
    #[default]
    Stratified,
    /// The M7 ultra tier: ReSTIR DI in the same `gi.area-shadow` pass.
    Restir,
}

impl StochasticMode {
    /// The wire name of the mode: `"stratified"` or `"restir"`.
    pub fn as_str(self) -> &'static str {
        match self {
            StochasticMode::Stratified => "stratified",
            StochasticMode::Restir => "restir",
        }
    }

    /// Parses `"stratified"` or `"restir"`; any other string is `None`.
    pub fn from_name(name: &str) -> Option<StochasticMode> {
        match name {
            "stratified" => Some(StochasticMode::Stratified),
            "restir" => Some(StochasticMode::Restir),
            _ => None,
        }
    }
}

/// Mid-tier settings: how many area lights get stratified shadow rays, how far they reach and
/// how many strata each light rectangle is split into.
///
/// `count` is the number of mid-tier area lights per view (0 disables the tier), `distance` is
/// the metres from the camera to the light's representative point at which the tier has fully
/// blended to the cone tier, and `cells` is the strata of the light rectangle (4 = 2x2,
/// 9 = 3x3).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct StochasticShadowSettings {
    /// Mid-tier area lights per view, an integer in `[0, 8]`.
    pub count: u32,
    /// Distance at which the mid tier has fully blended out, metres; finite and `> 0`.
    pub distance: f32,
    /// Strata of the light rectangle: 4 (2x2) or 9 (3x3).
    pub cells: u32,
    /// The stochastic shadow mode: `Stratified` (mid tier) or `Restir` (ultra tier).
    pub mode: StochasticMode,
}

impl Default for StochasticShadowSettings {
    fn default() -> Self {
        StochasticShadowSettings::standard()
    }
}

impl StochasticShadowSettings {
    /// The standard settings: 4 mid-tier lights, 40 m, 2x2 strata, `Stratified`.
    pub fn standard() -> StochasticShadowSettings {
        StochasticShadowSettings {
            count: 4,
            distance: 40.0,
            cells: 4,
            mode: StochasticMode::Stratified,
        }
    }

    /// Validates the settings in contract order, returning the first refusal.
    ///
    /// The refusals use `RaaError::FrameGraph` with code `light.invalid_area_shadows` and a
    /// `<key>: <reason>` message.
    pub fn validate(self) -> Result<StochasticShadowSettings, RaaError> {
        let refuse = |message: String| RaaError::FrameGraph {
            code: "light.invalid_area_shadows",
            message,
        };
        if self.count > MAX_STOCHASTIC_LIGHTS as u32 {
            return Err(refuse(format!(
                "stochasticCount: must be an integer in [0, {MAX_STOCHASTIC_LIGHTS}] (got {})",
                self.count
            )));
        }
        if !self.distance.is_finite() || self.distance <= 0.0 {
            return Err(refuse(format!(
                "stochasticDistance: must be finite and > 0 (got {})",
                self.distance
            )));
        }
        if self.cells != 4 && self.cells != 9 {
            return Err(refuse(format!(
                "stochasticCells: must be 4 or 9 (got {})",
                self.cells
            )));
        }
        Ok(self)
    }
}

/// Smoothstep `s(e0, e1, x) = t * t * (3 - 2 t)` with `t = clamp((x - e0) / (e1 - e0), 0, 1)`.
///
/// This is the helper of `crates/raa-lit/src/area_shadow.rs`, kept private here.
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The blend weight of the mid tier at `distance` metres: `1 - smoothstep(settings.distance -
/// band, settings.distance, distance)`.
///
/// It is 1 near the camera and 0 at or beyond `settings.distance`, so a light that enters or
/// leaves the mid tier never pops: `V = mix(V_cone, V_mask, weight)` blends over the band.
pub fn stochastic_weight(settings: &StochasticShadowSettings, band: f32, distance: f32) -> f32 {
    1.0 - smoothstep(settings.distance - band, settings.distance, distance)
}

/// Frame state of the mid tier: the slot table bytes, the per-slot occupant generations and the
/// number of occupied slots.
///
/// `table` is the [`AREA_STOCHASTIC_GPU_SIZE`]-byte binding-19 layout, `generations[s]` counts
/// the occupant changes of slot `s`, and `active` is the number of occupied slots.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AreaStochasticFrame {
    /// The `RaaAreaStochasticGpu` bytes (little-endian).
    pub table: [u8; AREA_STOCHASTIC_GPU_SIZE],
    /// Occupant change counter of each mid-tier slot.
    pub generations: [u32; MAX_STOCHASTIC_LIGHTS],
    /// Number of occupied mid-tier slots.
    pub active: u32,
    /// The mode the selector wrote this frame.
    pub mode: StochasticMode,
}

impl Default for AreaStochasticFrame {
    fn default() -> Self {
        AreaStochasticFrame::empty()
    }
}

impl AreaStochasticFrame {
    /// The empty frame: an all-zero table (no mid tier), zero generations and no active slot.
    pub fn empty() -> AreaStochasticFrame {
        AreaStochasticFrame {
            table: [0; AREA_STOCHASTIC_GPU_SIZE],
            generations: [0; MAX_STOCHASTIC_LIGHTS],
            active: 0,
            mode: StochasticMode::Stratified,
        }
    }
}

/// Writes the slot table of the `area-stochastic-gpu` layout into `table`.
///
/// Slot `s` with `Some((index, w))` writes `index` as the packed light index and `w` clamped to
/// `[0, 1]` (a NaN weight writes 0); `None` writes the empty index `0xFFFFFFFF` and weight 0.
/// `params` becomes `(limit.min(8), cells, 0, 0)`. Every byte of `table` is overwritten.
///
/// Handoff: `write_table` is in `crates/raa-lit/src/area_stochastic.rs`. Today it only encodes
/// the slots it is given, the limit and the cells into `table`; it selects no lights and computes
/// no weights. The downstream card 'raa-lit: StochasticSelector picks the mid-tier area lights and
/// writes the slot table' calls [`stochastic_weight`] and `write_table`, and fills
/// [`AreaStochasticFrame`] (its `table`, `generations` and `active`) from them.
pub fn write_table(
    table: &mut [u8; AREA_STOCHASTIC_GPU_SIZE],
    slots: &[Option<(u32, f32)>; MAX_STOCHASTIC_LIGHTS],
    limit: u32,
    cells: u32,
) {
    for (slot, entry) in slots.iter().enumerate() {
        let (index, weight) = match entry {
            Some((index, weight)) => (*index, clamp_weight(*weight)),
            None => (EMPTY_SLOT, 0.0),
        };
        put(table, LIGHTS_AT + 4 * slot, &index.to_le_bytes());
        put(table, WEIGHTS_AT + 4 * slot, &weight.to_le_bytes());
    }
    let limit = limit.min(MAX_STOCHASTIC_LIGHTS as u32);
    put(table, PARAMS_AT, &limit.to_le_bytes());
    put(table, PARAMS_AT + 4, &cells.to_le_bytes());
    put(table, PARAMS_AT + 8, &0u32.to_le_bytes());
    put(table, PARAMS_AT + 12, &0u32.to_le_bytes());
}

/// Reads the mid-tier slot `slot` of a table written by [`write_table`].
///
/// It is `None` when `slot >= min(limit, 8)` or when the slot holds the empty index.
pub fn table_slot(table: &[u8; AREA_STOCHASTIC_GPU_SIZE], slot: usize) -> Option<(u32, f32)> {
    let limit = table_limit(table).min(MAX_STOCHASTIC_LIGHTS as u32) as usize;
    if slot >= limit {
        return None;
    }
    let index = read_u32(table, LIGHTS_AT + 4 * slot)?;
    if index == EMPTY_SLOT {
        return None;
    }
    let weight = f32::from_bits(read_u32(table, WEIGHTS_AT + 4 * slot)?);
    Some((index, weight))
}

/// The number of mid-tier slots in use (`params.x`), 0 meaning no mid tier.
pub fn table_limit(table: &[u8; AREA_STOCHASTIC_GPU_SIZE]) -> u32 {
    read_u32(table, PARAMS_AT).unwrap_or(0)
}

/// The rectangle strata of the table (`params.y`): 4 or 9.
pub fn table_cells(table: &[u8; AREA_STOCHASTIC_GPU_SIZE]) -> u32 {
    read_u32(table, PARAMS_AT + 4).unwrap_or(0)
}

/// Clamps a blend weight to `[0, 1]`, mapping NaN to 0 so the table never holds a NaN.
fn clamp_weight(weight: f32) -> f32 {
    if weight.is_nan() {
        0.0
    } else {
        weight.clamp(0.0, 1.0)
    }
}

/// Copies `bytes` into `table` at `at`; an out-of-range write is ignored.
fn put(table: &mut [u8; AREA_STOCHASTIC_GPU_SIZE], at: usize, bytes: &[u8; 4]) {
    if let Some(dst) = table.get_mut(at..at + 4) {
        dst.copy_from_slice(bytes);
    }
}

/// Reads the little-endian `u32` at `at`, or `None` when it is out of range.
fn read_u32(table: &[u8; AREA_STOCHASTIC_GPU_SIZE], at: usize) -> Option<u32> {
    let bytes: [u8; 4] = table.get(at..at + 4)?.try_into().ok()?;
    Some(u32::from_le_bytes(bytes))
}

/// Chooses which area lights hold the mid-tier slots and writes the slot table.
///
/// Handoff: `StochasticSelector` (`new`, `update`, `active`) in `crates/raa-lit/src/area_stochastic.rs`
/// is attached by card 'raa-lit: mid-tier selection wired into LitRenderer::prepare and published through LitHandle', which calls `StochasticSelector::new`, `StochasticSelector::update` and `StochasticSelector::active`. Today `update` reads the settings, the camera, the lights and the record lanes that `HeroSelector::update` wrote, assigns the eight slots through the shared `SlotAssigner` (`crates/raa-lit/src/area_shadow.rs:257 SlotAssigner`) and encodes them with `write_table` (`crates/raa-lit/src/area_stochastic.rs:159 write_table`). It writes no record, allocates nothing and keeps only the slot handles, the candidate scratch and the generations between calls.
#[derive(Debug)]
pub struct StochasticSelector {
    assigner: SlotAssigner<MAX_STOCHASTIC_LIGHTS>,
    candidates: Vec<Candidate>,
    weights: Vec<(LightHandle, u32, f32)>,
    previous: [Option<LightHandle>; MAX_STOCHASTIC_LIGHTS],
    generations: [u32; MAX_STOCHASTIC_LIGHTS],
}

impl Default for StochasticSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl StochasticSelector {
    /// Creates an empty selector: no slot is held and every generation is 0. The candidate and
    /// weight scratch are allocated here, never in `update`.
    pub fn new() -> StochasticSelector {
        StochasticSelector {
            assigner: SlotAssigner::new(),
            candidates: Vec::with_capacity(MAX_LOCAL_LIGHTS),
            weights: Vec::with_capacity(MAX_LOCAL_LIGHTS),
            previous: [None; MAX_STOCHASTIC_LIGHTS],
            generations: [0; MAX_STOCHASTIC_LIGHTS],
        }
    }

    /// Number of mid-tier slots occupied after the last [`StochasticSelector::update`].
    pub fn active(&self) -> u32 {
        self.assigner.active()
    }

    /// Selects the mid-tier lights of this call and writes `out` (table, generations, active).
    ///
    /// `records` is the `LocalLights::write_gpu` output carrying the lanes of the last
    /// `HeroSelector::update`; it is only read. The limit is `settings.count` (at most 8) for the
    /// `Auto` tier and 0 for every other tier. Allocates nothing.
    pub fn update(
        &mut self,
        area: &AreaShadowSettings,
        settings: &StochasticShadowSettings,
        camera: &AreaShadowCamera,
        lights: &LocalLights,
        records: &[u8; MAX_LOCAL_LIGHTS * LIGHT_GPU_SIZE],
        out: &mut AreaStochasticFrame,
    ) {
        let limit = match area.tier {
            AreaShadowTier::Auto => (settings.count as usize).min(MAX_STOCHASTIC_LIGHTS),
            _ => 0,
        };
        self.candidates.clear();
        self.weights.clear();
        for (index, handle, light) in lights.world_lights() {
            if !matches!(light.kind, LocalLightKind::Area) || light.is_off() {
                continue;
            }
            let position = Vec3::from_array(light.position);
            let direction = Vec3::from_array(light.direction);
            let distance = rect_distance(camera.eye, position, direction, light.size);
            let weight = stochastic_weight(settings, area.band, distance);
            let coverage = screen_coverage(camera, position, light.range);
            if limit == 0 || weight <= 0.0 || coverage <= 0.0 || hero_shadowed(records, index) {
                continue;
            }
            self.candidates.push(Candidate { handle, coverage });
            self.weights.push((handle, index as u32, weight));
        }
        self.candidates.sort_unstable_by(|a, b| {
            b.coverage
                .total_cmp(&a.coverage)
                .then(a.handle.cmp(&b.handle))
        });
        self.assigner.assign(&self.candidates, limit);

        for (slot, held) in self.assigner.slots().iter().enumerate() {
            if *held != self.previous[slot] {
                self.generations[slot] = self.generations[slot].wrapping_add(1);
            }
        }
        self.previous = *self.assigner.slots();

        let held = *self.assigner.slots();
        let slots: [Option<(u32, f32)>; MAX_STOCHASTIC_LIGHTS] = held.map(|slot| {
            slot.and_then(|handle| self.weights.iter().find(|entry| entry.0 == handle))
                .map(|entry| (entry.1, entry.2))
        });
        write_table(&mut out.table, &slots, limit as u32, settings.cells);
        out.generations = self.generations;
        out.active = self.assigner.active();
        out.mode = settings.mode;
    }
}

/// Whether light record `index` is fully hero-shadowed: its hero layer lane (byte 44, f32) is at
/// least 4 and its `w_cone` lane (byte 48) is at most 0. Such a light gets no mid-tier slot.
fn hero_shadowed(records: &[u8], index: usize) -> bool {
    let base = index * LIGHT_GPU_SIZE;
    let lane = |offset: usize| -> Option<f32> {
        let bytes: [u8; 4] = records
            .get(base + offset..base + offset + 4)?
            .try_into()
            .ok()?;
        Some(f32::from_le_bytes(bytes))
    };
    matches!((lane(44), lane(48)), (Some(layer), Some(w_cone)) if layer >= 4.0 && w_cone <= 0.0)
}
