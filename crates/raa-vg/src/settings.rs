//! The virtualized-geometry settings of the S31 table (module m29): the [`VgSettings`] record
//! with its defaults and domains, the optional [`VgSettingsPatch`] merged over it, and the
//! [`DebugView`] modes with their GPU codes and JS names.
//!
//! A patch is applied atomically: the merged candidate is validated as a whole and the base
//! settings are never changed. A refusal is `vg.invalid` and names the S31 field.

use raa_core::RaaError;

use crate::error::invalid;

/// Lowest accepted `errorPixels` (pixels of screen-space error).
pub const ERROR_PIXELS_MIN: f32 = 0.25;
/// Highest accepted `errorPixels`.
pub const ERROR_PIXELS_MAX: f32 = 64.0;
/// Default `errorPixels`.
pub const DEFAULT_ERROR_PIXELS: f32 = 1.0;
/// Lowest accepted `maxVisibleClusters`.
pub const MAX_VISIBLE_CLUSTERS_MIN: u32 = 1024;
/// Highest accepted `maxVisibleClusters`.
pub const MAX_VISIBLE_CLUSTERS_MAX: u32 = 4_194_304;
/// Default `maxVisibleClusters`.
pub const DEFAULT_MAX_VISIBLE_CLUSTERS: u32 = 262_144;

/// The debug view of the resolve output (S32 `flags` debug view), with its GPU code.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum DebugView {
    /// Shaded output (code 0).
    #[default]
    None,
    /// Cluster colours (code 1).
    Clusters,
    /// Level of detail (code 2).
    Lod,
    /// Hardware versus software raster (code 3).
    Raster,
    /// Occlusion (code 4).
    Occlusion,
}

impl DebugView {
    /// The GPU code of the view (the rank table: None 0, Clusters 1, Lod 2, Raster 3, Occlusion 4).
    pub fn code(self) -> u32 {
        match self {
            DebugView::None => 0,
            DebugView::Clusters => 1,
            DebugView::Lod => 2,
            DebugView::Raster => 3,
            DebugView::Occlusion => 4,
        }
    }

    /// The JS name of the view: `none`, `clusters`, `lod`, `raster` or `occlusion`.
    pub fn name(self) -> &'static str {
        match self {
            DebugView::None => "none",
            DebugView::Clusters => "clusters",
            DebugView::Lod => "lod",
            DebugView::Raster => "raster",
            DebugView::Occlusion => "occlusion",
        }
    }

    /// The view named `name`, or `None` for any other string.
    pub fn from_name(name: &str) -> Option<DebugView> {
        match name {
            "none" => Some(DebugView::None),
            "clusters" => Some(DebugView::Clusters),
            "lod" => Some(DebugView::Lod),
            "raster" => Some(DebugView::Raster),
            "occlusion" => Some(DebugView::Occlusion),
            _ => None,
        }
    }
}

/// The virtualized-geometry settings (S31). [`Default`] is the S31 table.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct VgSettings {
    /// `enabled`: the path is requested at all (default false).
    pub enabled: bool,
    /// `errorPixels`: screen-space error threshold in pixels, finite in [0.25, 64] (default 1).
    pub error_pixels: f32,
    /// `softwareRaster`: small clusters go to the software rasteriser (default true).
    pub software_raster: bool,
    /// `occlusion`: the second phase and the occlusion pyramid run (default true).
    pub occlusion: bool,
    /// `shadows`: virtualized casters enter the cascade views (default true).
    pub shadows: bool,
    /// `runtimeBake`: a mesh without clusters is baked at runtime (default true).
    pub runtime_bake: bool,
    /// `instanceDefault`: instances are virtualized unless a request says otherwise (default true).
    pub instance_default: bool,
    /// `maxVisibleClusters`: integer in [1024, 4194304] (default 262144).
    pub max_visible_clusters: u32,
    /// `debugView`: the debug view (default none).
    pub debug_view: DebugView,
}

impl Default for VgSettings {
    fn default() -> Self {
        VgSettings {
            enabled: false,
            error_pixels: DEFAULT_ERROR_PIXELS,
            software_raster: true,
            occlusion: true,
            shadows: true,
            runtime_bake: true,
            instance_default: true,
            max_visible_clusters: DEFAULT_MAX_VISIBLE_CLUSTERS,
            debug_view: DebugView::None,
        }
    }
}

impl VgSettings {
    /// Checks the domains of the fields: `errorPixels` finite in [0.25, 64] and
    /// `maxVisibleClusters` an integer in [1024, 4194304]. Other fields have no domain.
    pub fn validate(&self) -> Result<(), RaaError> {
        if !self.error_pixels.is_finite()
            || !(ERROR_PIXELS_MIN..=ERROR_PIXELS_MAX).contains(&self.error_pixels)
        {
            return Err(invalid(
                "errorPixels",
                format!(
                    "must be finite and in [0.25, 64], got {}",
                    self.error_pixels
                ),
            ));
        }
        if !(MAX_VISIBLE_CLUSTERS_MIN..=MAX_VISIBLE_CLUSTERS_MAX)
            .contains(&self.max_visible_clusters)
        {
            return Err(invalid(
                "maxVisibleClusters",
                format!(
                    "must be an integer in [1024, 4194304], got {}",
                    self.max_visible_clusters
                ),
            ));
        }
        Ok(())
    }
}

/// A partial update of [`VgSettings`]: every field is optional and `None` keeps the base value.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct VgSettingsPatch {
    pub enabled: Option<bool>,
    pub error_pixels: Option<f32>,
    pub software_raster: Option<bool>,
    pub occlusion: Option<bool>,
    pub shadows: Option<bool>,
    pub runtime_bake: Option<bool>,
    pub instance_default: Option<bool>,
    pub max_visible_clusters: Option<u32>,
    pub debug_view: Option<DebugView>,
}

impl VgSettingsPatch {
    /// Merges the `Some` fields over `base` and validates the whole candidate. `base` is
    /// never changed; a refused candidate returns its `vg.invalid` error and no settings.
    pub fn apply(&self, base: &VgSettings) -> Result<VgSettings, RaaError> {
        let candidate = VgSettings {
            enabled: self.enabled.unwrap_or(base.enabled),
            error_pixels: self.error_pixels.unwrap_or(base.error_pixels),
            software_raster: self.software_raster.unwrap_or(base.software_raster),
            occlusion: self.occlusion.unwrap_or(base.occlusion),
            shadows: self.shadows.unwrap_or(base.shadows),
            runtime_bake: self.runtime_bake.unwrap_or(base.runtime_bake),
            instance_default: self.instance_default.unwrap_or(base.instance_default),
            max_visible_clusters: self
                .max_visible_clusters
                .unwrap_or(base.max_visible_clusters),
            debug_view: self.debug_view.unwrap_or(base.debug_view),
        };
        candidate.validate()?;
        Ok(candidate)
    }
}
