//! The global-illumination settings of the S36 table (module m30): the [`GiSettings`] record with
//! its defaults and domains, the optional [`GiSettingsPatch`] merged over it, the [`GiQuality`]
//! volume size, the [`GiQualityPatch`] merged over it and the [`DebugView`] modes with their GPU
//! codes and JS names.
//!
//! A patch is applied atomically: the merged fields are validated in key order and the base value
//! is never changed. A refusal is `gi.invalid` and names the S36 field.

use raa_core::RaaError;

use crate::error::invalid;

/// The finest voxel size in metres (S36, S37): level `i` uses `FINEST_VOXEL_METRES * 2^i` metres.
pub const FINEST_VOXEL_METRES: f32 = 0.2;

/// `intensity` is finite in [0, 16].
const INTENSITY_MAX: f64 = 16.0;
/// `environmentDiffuse` is finite in [0, 1].
const ENVIRONMENT_DIFFUSE_MAX: f64 = 1.0;

/// The debug view of the volume (S38 `GiFrameGpu.counts.z`), with its GPU code.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum DebugView {
    /// The shaded frame (code 0).
    #[default]
    None,
    /// The first occupied voxel along the camera ray (code 1).
    Voxels,
    /// The atlas radiance at the first hit (code 2).
    Radiance,
    /// Contour bands of the distance field (code 3).
    Distance,
    /// The finest level containing the depth surface (code 4).
    Levels,
}

impl DebugView {
    /// The GPU code of the view (rank table: None 0, Voxels 1, Radiance 2, Distance 3, Levels 4).
    pub fn code(self) -> u32 {
        match self {
            DebugView::None => 0,
            DebugView::Voxels => 1,
            DebugView::Radiance => 2,
            DebugView::Distance => 3,
            DebugView::Levels => 4,
        }
    }

    /// The JS name of the view: `none`, `voxels`, `radiance`, `distance` or `levels`.
    pub fn as_str(self) -> &'static str {
        match self {
            DebugView::None => "none",
            DebugView::Voxels => "voxels",
            DebugView::Radiance => "radiance",
            DebugView::Distance => "distance",
            DebugView::Levels => "levels",
        }
    }

    /// The view named `text`, or `None` for any other string.
    pub fn parse(text: &str) -> Option<DebugView> {
        match text {
            "none" => Some(DebugView::None),
            "voxels" => Some(DebugView::Voxels),
            "radiance" => Some(DebugView::Radiance),
            "distance" => Some(DebugView::Distance),
            "levels" => Some(DebugView::Levels),
            _ => None,
        }
    }
}

/// The global-illumination settings (S36). [`Default`] is the S36 table.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GiSettings {
    /// `enabled`: GI is requested at all (default false).
    pub enabled: bool,
    /// `intensity`: multiplier of the Indirect term, finite in [0, 16] (default 1).
    pub intensity: f32,
    /// `environmentDiffuse`: scale over raa-ibl's diffuse while GI is active, finite in [0, 1]
    /// (default 0).
    pub environment_diffuse: f32,
    /// `localLightShadows`: the volume traces local-light visibility (default true).
    pub local_light_shadows: bool,
    /// `reflections`: the volume backs glossy SSR fallbacks (default true).
    pub reflections: bool,
    /// `screenTraces`: screen probes trace the history first (default true).
    pub screen_traces: bool,
    /// `debugView`: the debug view (default none).
    pub debug_view: DebugView,
}

impl Default for GiSettings {
    fn default() -> Self {
        GiSettings {
            enabled: false,
            intensity: 1.0,
            environment_diffuse: 0.0,
            local_light_shadows: true,
            reflections: true,
            screen_traces: true,
            debug_view: DebugView::None,
        }
    }
}

/// A partial update of [`GiSettings`]: every field is optional and `None` keeps the base value.
/// Numeric fields arrive as `f64` and are stored as `f32` once they pass their domain check.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GiSettingsPatch {
    pub enabled: Option<bool>,
    pub intensity: Option<f64>,
    pub environment_diffuse: Option<f64>,
    pub local_light_shadows: Option<bool>,
    pub reflections: Option<bool>,
    pub screen_traces: Option<bool>,
    pub debug_view: Option<DebugView>,
}

impl GiSettingsPatch {
    /// Handoff: `raa-gi: settings tests` calls `GiSettingsPatch::apply` (here), [`GiQualityPatch::apply`],
    /// [`DebugView::parse`], [`DebugView::as_str`], [`DebugView::code`] and [`crate::error::invalid`].
    /// This method merges the `Some` fields over `current` in S36 key order and returns the new
    /// [`GiSettings`]; a refused `intensity` or `environmentDiffuse` is `gi.invalid` with its key
    /// message and changes nothing.
    pub fn apply(&self, current: &GiSettings) -> Result<GiSettings, RaaError> {
        let intensity = match self.intensity {
            Some(value) => within(
                "intensity",
                value,
                INTENSITY_MAX,
                "must be finite and in [0, 16]",
            )?,
            None => current.intensity,
        };
        let environment_diffuse = match self.environment_diffuse {
            Some(value) => within(
                "environmentDiffuse",
                value,
                ENVIRONMENT_DIFFUSE_MAX,
                "must be finite and in [0, 1]",
            )?,
            None => current.environment_diffuse,
        };
        Ok(GiSettings {
            enabled: self.enabled.unwrap_or(current.enabled),
            intensity,
            environment_diffuse,
            local_light_shadows: self
                .local_light_shadows
                .unwrap_or(current.local_light_shadows),
            reflections: self.reflections.unwrap_or(current.reflections),
            screen_traces: self.screen_traces.unwrap_or(current.screen_traces),
            debug_view: self.debug_view.unwrap_or(current.debug_view),
        })
    }
}

/// The volume quality (S36, S37): `resolution` voxels per axis, `levels` clipmap levels and
/// `probeRays` rays per side of a screen probe. [`Default`] is 96 / 5 / 8.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GiQuality {
    /// `resolution`: 64 or 96 voxels per axis (default 96).
    pub resolution: u32,
    /// `levels`: 3, 4 or 5 clipmap levels (default 5).
    pub levels: u32,
    /// `probeRays`: 6 or 8 rays per probe side (default 8).
    pub probe_rays: u32,
}

impl Default for GiQuality {
    fn default() -> Self {
        GiQuality {
            resolution: 96,
            levels: 5,
            probe_rays: 8,
        }
    }
}

/// A partial update of [`GiQuality`]. Numbers are `f64` and must be exactly one of the listed
/// integers, so `96.5` is refused; an accepted number is stored as `u32`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GiQualityPatch {
    pub resolution: Option<f64>,
    pub levels: Option<f64>,
    pub probe_rays: Option<f64>,
}

impl GiQualityPatch {
    /// Merges the `Some` fields over `current` in S36 key order and returns the new [`GiQuality`];
    /// a number that is not exactly one of the listed integers is `gi.invalid` with its key
    /// message and changes nothing.
    pub fn apply(&self, current: &GiQuality) -> Result<GiQuality, RaaError> {
        let resolution = match self.resolution {
            Some(value) => choice("resolution", value, &[64, 96], "must be one of 64, 96")?,
            None => current.resolution,
        };
        let levels = match self.levels {
            Some(value) => choice("levels", value, &[3, 4, 5], "must be one of 3, 4, 5")?,
            None => current.levels,
        };
        let probe_rays = match self.probe_rays {
            Some(value) => choice("probeRays", value, &[6, 8], "must be one of 6, 8")?,
            None => current.probe_rays,
        };
        Ok(GiQuality {
            resolution,
            levels,
            probe_rays,
        })
    }
}

/// A finite `f64` in `[0, max]`, stored as `f32`; anything else is the `gi.invalid` `message`.
fn within(key: &str, value: f64, max: f64, message: &str) -> Result<f32, RaaError> {
    if value.is_finite() && (0.0..=max).contains(&value) {
        let stored = value as f32;
        if stored.is_finite() {
            return Ok(stored);
        }
    }
    Err(invalid(key, message))
}

/// An `f64` equal to one of `allowed`, stored as `u32`; anything else is the `gi.invalid`
/// `message`. `96.5` and negative or non-finite values are refused.
fn choice(key: &str, value: f64, allowed: &[u32], message: &str) -> Result<u32, RaaError> {
    if let Some(&picked) = allowed.iter().find(|&&option| f64::from(option) == value) {
        return Ok(picked);
    }
    Err(invalid(key, message))
}
