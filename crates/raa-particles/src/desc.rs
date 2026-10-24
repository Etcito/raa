//! [`ParticleKind`], [`EmitterDesc`] and its validation.

use raa_core::RaaError;

/// The look and motion family of an emitter.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ParticleKind {
    /// Ballistic water droplets thrown up from a surface.
    Splash,
    /// Slow drifting flakes falling from a volume.
    Snow,
    /// Fluttering paper rectangles with saturated colours.
    Confetti,
    /// Tiny emissive points that twinkle inside a volume.
    Sparkles,
}

/// One particle emitter. Colours are linear RGB.
#[derive(Clone, Debug, PartialEq)]
pub struct EmitterDesc {
    /// Effect family.
    pub kind: ParticleKind,
    /// Emitter centre in world space.
    pub position: [f32; 3],
    /// Mean spawn direction (normalised, non-zero).
    pub direction: [f32; 3],
    /// Cone half-angle in radians, in `[0, pi]`.
    pub spread_rad: f32,
    /// Continuous spawn rate in particles per second, `>= 0`.
    pub rate_per_s: f32,
    /// Lifetime range `[min, max]` in seconds, `0 < min <= max`.
    pub lifetime_s: [f32; 2],
    /// Initial speed range `[min, max]` in m/s, `0 <= min <= max`.
    pub speed: [f32; 2],
    /// Particle size range `[min, max]` in metres, `0 < min <= max`.
    pub size: [f32; 2],
    /// Downward acceleration in m/s² (negative floats upwards).
    pub gravity: f32,
    /// Velocity damping per second, `>= 0`.
    pub drag: f32,
    /// Palette, 1 to 8 linear RGB entries with each component `>= 0`.
    pub colors: Vec<[f32; 3]>,
    /// Emissive strength added to the palette colour, `>= 0`.
    pub emissive: f32,
    /// Half extents of the spawn volume in metres, non-negative.
    pub volume_half: [f32; 3],
    /// Bounce plane height in metres; `None` disables the ground bounce.
    pub ground_y: Option<f32>,
    /// Hard cap on live particles for this emitter, in `1..=65536`.
    pub max_particles: u32,
    /// Seed of the deterministic spawn sequence.
    pub seed: u32,
}

impl EmitterDesc {
    /// Per-kind defaults. Gravity is 9.81 except snow (1.0) and sparkles (0).
    pub fn for_kind(kind: ParticleKind) -> EmitterDesc {
        match kind {
            ParticleKind::Splash => EmitterDesc {
                kind,
                position: [0.0; 3],
                direction: [0.0, 1.0, 0.0],
                spread_rad: 0.5,
                rate_per_s: 0.0,
                lifetime_s: [0.6, 1.2],
                speed: [1.5, 3.5],
                size: [0.015, 0.04],
                gravity: 9.81,
                drag: 0.2,
                colors: vec![[0.8, 0.9, 1.0]],
                emissive: 0.0,
                volume_half: [0.0; 3],
                ground_y: None,
                max_particles: 2048,
                seed: 1,
            },
            ParticleKind::Snow => EmitterDesc {
                kind,
                position: [0.0; 3],
                direction: [0.0, -1.0, 0.0],
                spread_rad: 0.2,
                rate_per_s: 400.0,
                lifetime_s: [6.0, 10.0],
                speed: [0.2, 0.4],
                size: [0.01, 0.025],
                gravity: 1.0,
                drag: 2.0,
                colors: vec![[0.95, 0.97, 1.0]],
                emissive: 0.0,
                volume_half: [4.0, 2.0, 4.0],
                ground_y: None,
                max_particles: 8192,
                seed: 1,
            },
            ParticleKind::Confetti => EmitterDesc {
                kind,
                position: [0.0; 3],
                direction: [0.0, 1.0, 0.0],
                spread_rad: 0.6,
                rate_per_s: 120.0,
                lifetime_s: [3.0, 5.0],
                speed: [3.0, 6.0],
                size: [0.02, 0.035],
                gravity: 9.81,
                drag: 1.5,
                colors: vec![
                    [1.0, 0.0, 0.0],
                    [1.0, 0.5, 0.0],
                    [1.0, 0.9, 0.0],
                    [0.0, 0.8, 0.2],
                    [0.0, 0.4, 1.0],
                    [0.8, 0.0, 1.0],
                ],
                emissive: 0.0,
                volume_half: [0.0; 3],
                ground_y: None,
                max_particles: 4096,
                seed: 1,
            },
            ParticleKind::Sparkles => EmitterDesc {
                kind,
                position: [0.0; 3],
                direction: [0.0, 1.0, 0.0],
                spread_rad: std::f32::consts::PI,
                rate_per_s: 60.0,
                lifetime_s: [0.6, 1.6],
                speed: [0.1, 0.4],
                size: [0.02, 0.06],
                gravity: 0.0,
                drag: 0.5,
                colors: vec![[1.0, 0.85, 0.3], [1.0, 1.0, 1.0]],
                emissive: 8.0,
                volume_half: [0.5, 0.5, 0.5],
                ground_y: None,
                max_particles: 1024,
                seed: 1,
            },
        }
    }

    /// Refuses with `particles.invalid`, naming the offending field. Every number must
    /// be finite; direction non-zero; spread in `[0, pi]`; rate `>= 0`; lifetime
    /// `0 < min <= max`; speed `0 <= min <= max`; size `0 < min <= max`; drag `>= 0`;
    /// 1 to 8 colours with non-negative components; emissive `>= 0`; volume half
    /// extents `>= 0`; `1 <= max_particles <= 65536`.
    pub fn validate(&self) -> Result<(), RaaError> {
        for (i, v) in self.position.iter().enumerate() {
            finite(&format!("position[{i}]"), *v)?;
        }
        for (i, v) in self.direction.iter().enumerate() {
            finite(&format!("direction[{i}]"), *v)?;
        }
        let len_sq = self.direction.iter().map(|v| v * v).sum::<f32>();
        if len_sq <= 0.0 {
            return Err(invalid("direction must be non-zero".into()));
        }
        finite("spread_rad", self.spread_rad)?;
        if !(0.0..=std::f32::consts::PI).contains(&self.spread_rad) {
            return Err(invalid(format!(
                "spread_rad must be in [0, pi] (got {})",
                self.spread_rad
            )));
        }
        finite("rate_per_s", self.rate_per_s)?;
        if self.rate_per_s < 0.0 {
            return Err(invalid(format!(
                "rate_per_s must be >= 0 (got {})",
                self.rate_per_s
            )));
        }
        finite("lifetime_s[0]", self.lifetime_s[0])?;
        finite("lifetime_s[1]", self.lifetime_s[1])?;
        if self.lifetime_s[0] <= 0.0 {
            return Err(invalid(format!(
                "lifetime_s min must be > 0 (got {})",
                self.lifetime_s[0]
            )));
        }
        if self.lifetime_s[0] > self.lifetime_s[1] {
            return Err(invalid(format!(
                "lifetime_s min must be <= max (got [{}, {}])",
                self.lifetime_s[0], self.lifetime_s[1]
            )));
        }
        range_pair("speed", self.speed, 0.0, false)?;
        range_pair("size", self.size, 0.0, true)?;
        finite("gravity", self.gravity)?;
        finite("drag", self.drag)?;
        if self.drag < 0.0 {
            return Err(invalid(format!("drag must be >= 0 (got {})", self.drag)));
        }
        if self.colors.is_empty() || self.colors.len() > 8 {
            return Err(invalid(format!(
                "colors must have 1 to 8 entries (got {})",
                self.colors.len()
            )));
        }
        for (i, c) in self.colors.iter().enumerate() {
            for (j, v) in c.iter().enumerate() {
                let field = format!("colors[{i}][{j}]");
                finite(&field, *v)?;
                if *v < 0.0 {
                    return Err(invalid(format!("{field} must be >= 0 (got {v})")));
                }
            }
        }
        finite("emissive", self.emissive)?;
        if self.emissive < 0.0 {
            return Err(invalid(format!(
                "emissive must be >= 0 (got {})",
                self.emissive
            )));
        }
        for (i, v) in self.volume_half.iter().enumerate() {
            let field = format!("volume_half[{i}]");
            finite(&field, *v)?;
            if *v < 0.0 {
                return Err(invalid(format!("{field} must be >= 0 (got {v})")));
            }
        }
        if let Some(y) = self.ground_y {
            finite("ground_y", y)?;
        }
        if !(1..=65536).contains(&self.max_particles) {
            return Err(invalid(format!(
                "max_particles must be in [1, 65536] (got {})",
                self.max_particles
            )));
        }
        Ok(())
    }
}

fn invalid(message: String) -> RaaError {
    RaaError::FrameGraph {
        code: "particles.invalid",
        message,
    }
}

fn finite(field: &str, v: f32) -> Result<(), RaaError> {
    if v.is_finite() {
        Ok(())
    } else {
        Err(invalid(format!("{field} must be finite (got {v})")))
    }
}

fn range_pair(
    field: &str,
    pair: [f32; 2],
    min_allowed: f32,
    min_exclusive: bool,
) -> Result<(), RaaError> {
    for (i, v) in pair.iter().enumerate() {
        finite(&format!("{field}[{i}]"), *v)?;
    }
    let min = pair[0];
    let below = if min_exclusive {
        min <= min_allowed
    } else {
        min < min_allowed
    };
    if below {
        let op = if min_exclusive { ">" } else { ">=" };
        return Err(invalid(format!(
            "{field} min must be {op} {min_allowed} (got {min})"
        )));
    }
    if min > pair[1] {
        return Err(invalid(format!(
            "{field} min must be <= max (got [{min}, {}])",
            pair[1]
        )));
    }
    Ok(())
}
