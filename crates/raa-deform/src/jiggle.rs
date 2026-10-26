//! Per-tag jiggle springs on the lag of a node's world translation.

use raa_core::RaaError;
use raa_math::{Mat4, Vec3};

const MAX_SUBSTEP_RATE: f32 = 120.0;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct JiggleConfig {
    pub stiffness: f32,
    pub damping: f32,
    pub amount: f32,
    pub height_m: f32,
}

impl Default for JiggleConfig {
    fn default() -> Self {
        JiggleConfig {
            stiffness: 120.0,
            damping: 0.35,
            amount: 1.0,
            height_m: 1.0,
        }
    }
}

impl JiggleConfig {
    pub fn validate(&self) -> Result<(), RaaError> {
        if !(self.stiffness.is_finite() && self.stiffness > 0.0) {
            return Err(invalid("stiffness", "finite and > 0", self.stiffness));
        }
        if !(self.damping.is_finite() && (0.0..=2.0).contains(&self.damping)) {
            return Err(invalid("damping", "finite and in [0, 2]", self.damping));
        }
        if !(self.amount.is_finite() && self.amount >= 0.0) {
            return Err(invalid("amount", "finite and >= 0", self.amount));
        }
        if !(self.height_m.is_finite() && self.height_m > 0.0) {
            return Err(invalid("height_m", "finite and > 0", self.height_m));
        }
        Ok(())
    }
}

fn invalid(field: &str, requirement: &str, value: f32) -> RaaError {
    RaaError::FrameGraph {
        code: "deform.invalid",
        message: format!("{field} must be {requirement} (got {value})"),
    }
}

#[derive(Clone, Debug)]
struct Entry {
    tag: u32,
    config: JiggleConfig,
    initialised: bool,
    previous_translation: Vec3,
    lag: Vec3,
    velocity: Vec3,
    offset: Vec3,
}

/// Jiggle state for every tagged instance, kept sorted by tag for deterministic stepping.
#[derive(Clone, Debug, Default)]
pub struct JiggleSprings {
    entries: Vec<Entry>,
}

impl JiggleSprings {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets or replaces the config of `tag`; `None` removes the tag. Callers validate the config first.
    pub fn set(&mut self, tag: u32, c: Option<JiggleConfig>) {
        match (self.find(tag), c) {
            (Ok(i), Some(config)) => self.entries[i].config = config,
            (Ok(i), None) => {
                self.entries.remove(i);
            }
            (Err(i), Some(config)) => self.entries.insert(
                i,
                Entry {
                    tag,
                    config,
                    initialised: false,
                    previous_translation: Vec3::ZERO,
                    lag: Vec3::ZERO,
                    velocity: Vec3::ZERO,
                    offset: Vec3::ZERO,
                },
            ),
            (Err(_), None) => {}
        }
    }

    /// Advances the spring of `tag` by `dt_s` and returns the offset in node-local axes.
    pub fn step(&mut self, dt_s: f32, tag: u32, world: &Mat4) -> [f32; 3] {
        let Ok(i) = self.find(tag) else {
            return [0.0; 3];
        };
        let entry = &mut self.entries[i];
        if !(dt_s.is_finite() && dt_s > 0.0) {
            return entry.offset.to_array();
        }

        let translation = Vec3::new(world.cols[3][0], world.cols[3][1], world.cols[3][2]);
        if !entry.initialised {
            entry.initialised = true;
            entry.previous_translation = translation;
            entry.lag = Vec3::ZERO;
            entry.velocity = Vec3::ZERO;
            entry.offset = Vec3::ZERO;
            return [0.0; 3];
        }

        entry.lag -= translation - entry.previous_translation;
        entry.previous_translation = translation;

        let k = entry.config.stiffness;
        let c = 2.0 * entry.config.damping * k.sqrt();
        let substeps = (dt_s * MAX_SUBSTEP_RATE).ceil().max(1.0) as u32;
        let h = dt_s / substeps as f32;
        for _ in 0..substeps {
            let acceleration = entry.lag * -k - entry.velocity * c;
            entry.velocity += acceleration * h;
            entry.lag += entry.velocity * h;
        }

        let local = [
            column_axis(world, 0),
            column_axis(world, 1),
            column_axis(world, 2),
        ];
        let lag = entry.lag;
        let mut offset = Vec3::new(lag.dot(local[0]), lag.dot(local[1]), lag.dot(local[2]))
            * entry.config.amount;

        let limit = 0.5 * entry.config.height_m;
        let length = offset.length();
        if length > limit {
            offset = offset * (limit / length);
        }
        entry.offset = offset;
        offset.to_array()
    }

    fn find(&self, tag: u32) -> Result<usize, usize> {
        self.entries.binary_search_by_key(&tag, |e| e.tag)
    }
}

fn column_axis(world: &Mat4, column: usize) -> Vec3 {
    let c = world.cols[column];
    let axis = Vec3::new(c[0], c[1], c[2]).normalize_or_zero();
    if axis == Vec3::ZERO {
        let mut unit = [0.0; 3];
        unit[column] = 1.0;
        Vec3::from_array(unit)
    } else {
        axis
    }
}
