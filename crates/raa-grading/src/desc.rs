//! Colour grade descriptor, presets and validation (`post.invalid`).

use raa_core::RaaError;

/// Colour grade applied to HDR before tonemap; every control is identity at its default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradingDesc {
    /// White balance in kelvin, in [1500, 15000]; higher is warmer, 6500 is identity.
    pub temperature_k: f32,
    /// Green (negative) to magenta (positive) white balance shift, in [-1, 1]; 0 is identity.
    pub tint: f32,
    /// Exposure in stops, in [-10, 10]; 0 is identity.
    pub exposure_ev: f32,
    /// Log-space contrast about middle grey (0.18), in (0, 4]; 1 is identity.
    pub contrast: f32,
    /// Luma-preserving saturation, in [0, 4]; 1 is identity.
    pub saturation: f32,
    /// Luma-preserving vibrance, in [-1, 1]; 0 is identity.
    pub vibrance: f32,
    /// ASC CDL lift (shadow offset) per channel; 0 is identity.
    pub lift: [f32; 3],
    /// ASC CDL gamma (power) per channel, > 0; 1 is identity.
    pub gamma: [f32; 3],
    /// ASC CDL gain (slope) per channel, >= 0; 1 is identity.
    pub gain: [f32; 3],
    /// Shadow tint colour per channel, >= 0; 0.5 is no tint.
    pub shadows_tint: [f32; 3],
    /// Highlight tint colour per channel, >= 0; 0.5 is no tint.
    pub highlights_tint: [f32; 3],
    /// Split toning balance between shadows and highlights, in [-1, 1]; 0 is even.
    pub split_balance: f32,
    /// Vignette amount, in [0, 1]; 0 is none.
    pub vignette: f32,
}

impl Default for GradingDesc {
    /// The `neutral` preset.
    fn default() -> Self {
        GradingDesc {
            temperature_k: 6500.0,
            tint: 0.0,
            exposure_ev: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            vibrance: 0.0,
            lift: [0.0; 3],
            gamma: [1.0; 3],
            gain: [1.0; 3],
            shadows_tint: [0.5; 3],
            highlights_tint: [0.5; 3],
            split_balance: 0.0,
            vignette: 0.0,
        }
    }
}

impl GradingDesc {
    /// Built-in presets `neutral`, `candy`, `ice`, `jungle`, `neon`; `None` for any other name.
    pub fn preset(name: &str) -> Option<GradingDesc> {
        let neutral = GradingDesc::default();
        match name {
            "neutral" => Some(neutral),
            // Sunny candy: clean shadows and vivid colour, with room for glossy highlights.
            "candy" => Some(GradingDesc {
                temperature_k: 6700.0,
                contrast: 1.06,
                saturation: 1.3,
                vibrance: 0.3,
                vignette: 0.03,
                ..neutral
            }),
            // Cool ice: blue shadows over slightly lifted blacks, a little less colour.
            "ice" => Some(GradingDesc {
                temperature_k: 5600.0,
                saturation: 0.9,
                lift: [0.01; 3],
                shadows_tint: [0.45, 0.5, 0.6],
                ..neutral
            }),
            // Green-yellow jungle: green tint, punchier contrast, warm highlights.
            "jungle" => Some(GradingDesc {
                tint: -0.1,
                contrast: 1.1,
                saturation: 1.1,
                highlights_tint: [0.6, 0.53, 0.44],
                ..neutral
            }),
            // Neon: hard contrast, magenta shadows against cyan highlights.
            "neon" => Some(GradingDesc {
                contrast: 1.2,
                vibrance: 0.3,
                vignette: 0.35,
                shadows_tint: [0.6, 0.45, 0.6],
                highlights_tint: [0.45, 0.6, 0.6],
                ..neutral
            }),
            _ => None,
        }
    }

    /// Refuses with `post.invalid` and a message naming the first offending field. Every value
    /// must be finite; `lift` has no other range.
    pub fn validate(&self) -> Result<(), RaaError> {
        within("temperature_k", self.temperature_k, 1500.0, 15000.0)?;
        within("tint", self.tint, -1.0, 1.0)?;
        within("exposure_ev", self.exposure_ev, -10.0, 10.0)?;
        let contrast_ok = self.contrast > 0.0 && self.contrast <= 4.0;
        check("contrast", self.contrast, contrast_ok, "in (0, 4]")?;
        within("saturation", self.saturation, 0.0, 4.0)?;
        within("vibrance", self.vibrance, -1.0, 1.0)?;
        components("lift", self.lift, |_| true, "finite")?;
        components("gamma", self.gamma, |v| v > 0.0, "> 0")?;
        components("gain", self.gain, |v| v >= 0.0, ">= 0")?;
        components("shadows_tint", self.shadows_tint, |v| v >= 0.0, ">= 0")?;
        components(
            "highlights_tint",
            self.highlights_tint,
            |v| v >= 0.0,
            ">= 0",
        )?;
        within("split_balance", self.split_balance, -1.0, 1.0)?;
        within("vignette", self.vignette, 0.0, 1.0)
    }
}

fn check(field: &str, v: f32, ok: bool, rule: &str) -> Result<(), RaaError> {
    let message = if !v.is_finite() {
        format!("`{field}` must be finite, got {v}")
    } else if !ok {
        format!("`{field}` must be {rule}, got {v}")
    } else {
        return Ok(());
    };
    Err(RaaError::FrameGraph {
        code: "post.invalid",
        message,
    })
}

fn within(field: &str, v: f32, min: f32, max: f32) -> Result<(), RaaError> {
    check(
        field,
        v,
        (min..=max).contains(&v),
        &format!("in [{min}, {max}]"),
    )
}

fn components(field: &str, c: [f32; 3], ok: fn(f32) -> bool, rule: &str) -> Result<(), RaaError> {
    c.into_iter()
        .enumerate()
        .try_for_each(|(i, v)| check(&format!("{field}[{i}]"), v, ok(v), rule))
}
