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
    /// At least one clipmap level needs further rebuild work.
    pub pending_rebuild: bool,
    /// Unique levels with an unfinished slice, newer scene inputs or a different snapped origin.
    pub rebuild_pending_levels: u32,
    /// The cause of the rebuild recorded this frame, or `None` on a relight-only or inactive frame.
    pub rebuild_reason: Option<&'static str>,
    /// The zero-based level recorded this frame, or `None` for a whole-volume build or no rebuild.
    pub rebuild_level: Option<u32>,
    /// Flood passes left on that slice after the frame, or `None` when no slice was recorded.
    pub rebuild_flood_passes_remaining: Option<u32>,
    /// Number of levels relit this frame, regardless of rebuild activity.
    pub relit_levels: u32,
    /// Frames with no rebuild pending, counting the completion frame as 1. Zero while inactive.
    pub frames_since_rebuild: u32,
    /// Every GPU byte the crate owns; 0 while inactive.
    pub allocated_bytes: u64,
    /// The invocation id of the newest frame whose submission completed.
    pub stats_frame_index: Option<u64>,
    /// A prepared frame whose submission is not yet observed.
    pub pending: bool,
    /// Real-frame age of the lit HDR that screen hits could sample in this frame.
    pub lit_history_age: Option<u64>,
    /// The mean history length of the screen accumulation (frames), read back from the GPU a few
    /// frames late over one pixel in 16; `None` until a readback completed.
    pub history_length_mean: Option<f32>,
    /// The share of those pixels whose history was rejected (disocclusion), in percent.
    pub rejected_percent: Option<f32>,
    /// The share of those pixels whose history the neighbourhood clamp moved, in percent.
    pub clamped_percent: Option<f32>,
}
