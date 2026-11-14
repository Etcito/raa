//! The global-illumination statistics of `giStats()` (S36): [`GiStats`].
//!
//! The feature handle fills it from CPU counters when a submission completes; there is no GPU
//! readback. Values that are not yet known are `None`, never inferred.

use crate::settings::GiQuality;

/// One `giStats()` snapshot (S36).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GiStats {
    /// GI runs a volume this frame.
    pub active: bool,
    /// The first inactive reason (`disabled`, `passDisabled`, `multiView`, `noInstance`), or
    /// `None` while active.
    pub inactive_reason: Option<&'static str>,
    /// The current quality.
    pub quality: GiQuality,
    /// `FINEST_VOXEL_METRES`: the level-0 voxel size in metres.
    pub finest_voxel_metres: f32,
    /// Scene instances voxelized by driver A in the latest prepared frame.
    pub voxelized_forward_instances: u32,
    /// Scene instances voxelized by driver B in the latest prepared frame.
    pub voxelized_virtual_instances: u32,
    /// A rebuild slice is pending, or a round is outdated or waiting.
    pub pending_rebuild: bool,
    /// Frames since the last rebuild round completed (0 while pending).
    pub frames_since_rebuild: u32,
    /// Every GPU byte the crate owns; 0 while inactive.
    pub allocated_bytes: u64,
    /// The invocation id of the newest frame whose submission completed.
    pub stats_frame_index: Option<u64>,
    /// A prepared frame whose submission is not yet observed.
    pub pending: bool,
}
