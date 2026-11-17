//! Readback pool and exposure state of the `exposure.auto` pass (S46), following light_stats.rs.
#![allow(dead_code)] // Staged until feature.rs calls it; see Handoff.

use crate::maths::settled;
use crate::shaders::{ExposureResult, read_result};
use crate::{READBACK_SLOTS, RESULT_SIZE};
use raa_core::{GpuContext, wgpu};
use std::sync::{Arc, Mutex, MutexGuard};

/// What the pass reports (S46 state): the latest exposure and the settled flag.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AutoExposureState {
    /// Log2 of the latest linear exposure, None before a result.
    pub exposure_ev: Option<f32>,
    /// Adapted average luminance of the latest result, None before a result.
    pub average_luminance: Option<f32>,
    /// True when the last two results are within `SETTLED_EV`; true when disabled.
    pub settled: bool,
    /// True while an enabled pass waits for a result since its last snap.
    pub pending: bool,
}

/// Lifecycle of one readback slot: free, copied by a recorded frame, or being mapped.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum SlotStatus {
    Idle,
    Copied,
    Mapping,
}

/// The lowest index whose slot is `Idle`, None when every slot is `Copied` or `Mapping`.
pub(crate) fn first_idle(statuses: &[SlotStatus; READBACK_SLOTS]) -> Option<usize> {
    statuses
        .iter()
        .position(|&status| status == SlotStatus::Idle)
}

/// Frame bookkeeping and the two latest decoded results. The map callbacks push into it
/// through `Arc<Mutex<ExposureTracker>>`.
pub(crate) struct ExposureTracker {
    last_frame: Option<(u64, (u32, u32))>,
    invalidated: bool,
    snap_frame: u32,
    latest: Option<ExposureResult>,
    before: Option<ExposureResult>,
}

impl ExposureTracker {
    pub(crate) fn new() -> ExposureTracker {
        ExposureTracker {
            last_frame: None,
            invalidated: false,
            snap_frame: 0,
            latest: None,
            before: None,
        }
    }

    /// The next `begin_frame` snaps.
    pub(crate) fn invalidate(&mut self) {
        self.invalidated = true;
    }

    /// Records a prepared frame. Returns true when the pass snaps: no frame yet, a size change,
    /// a frame gap or an invalidate. A snap clears the results and sets `snap_frame`.
    pub(crate) fn begin_frame(&mut self, frame: u64, size: (u32, u32)) -> bool {
        let continues = self.last_frame.is_some_and(|(previous, previous_size)| {
            !self.invalidated && previous_size == size && frame.checked_sub(1) == Some(previous)
        });
        self.last_frame = Some((frame, size));
        self.invalidated = false;
        if !continues {
            self.latest = None;
            self.before = None;
            self.snap_frame = frame as u32;
        }
        !continues
    }

    /// Keeps a valid result that is not older than the snap and newer than the latest kept one.
    pub(crate) fn push_result(&mut self, result: ExposureResult) {
        if !result.valid || result.frame < self.snap_frame {
            return;
        }
        if self
            .latest
            .is_some_and(|latest| result.frame <= latest.frame)
        {
            return;
        }
        self.before = self.latest;
        self.latest = Some(result);
    }

    /// The reported state: disabled is settled, enabled without a result is pending.
    pub(crate) fn state(&self, enabled: bool) -> AutoExposureState {
        if !enabled {
            return AutoExposureState {
                exposure_ev: None,
                average_luminance: None,
                settled: true,
                pending: false,
            };
        }
        let Some(latest) = self.latest else {
            return AutoExposureState {
                exposure_ev: None,
                average_luminance: None,
                settled: false,
                pending: true,
            };
        };
        let ev = latest.exposure.log2();
        AutoExposureState {
            exposure_ev: Some(ev),
            average_luminance: Some(latest.average_luminance),
            settled: self
                .before
                .is_some_and(|before| settled(before.exposure.log2(), ev)),
            pending: false,
        }
    }
}

/// Locks shared state and recovers a poisoned lock, as light_stats.rs does.
fn locked<T>(shared: &Mutex<T>) -> MutexGuard<'_, T> {
    shared.lock().unwrap_or_else(|e| e.into_inner())
}

/// Decodes the mapped 16 B record; None when the mapped range is not readable.
fn decode(mapped: &wgpu::Buffer) -> Option<ExposureResult> {
    let view = mapped.get_mapped_range(..).ok()?;
    let bytes: &[u8; RESULT_SIZE] = view.get(..RESULT_SIZE)?.try_into().ok()?;
    Some(read_result(bytes))
}

/// Handoff: `ReadbackSlots` (three 16 B `MAP_READ | COPY_DST` slots) is taken over by card
/// 'raa-exposure 6/6: exposure.auto pass, install and handle'. Its entry points, as they stand
/// today: `ReadbackSlots::new(device)` creates the three buffers with every slot Idle;
/// `ReadbackSlots::copy(encoder, state)` records a copy of the state into the first Idle slot
/// and marks it Copied, and records nothing when none is Idle; `ReadbackSlots::map_copied(gpu, sink)` maps each Copied slot once,
/// polls without waiting, and its callback pushes the decoded record into the tracker when the
/// map succeeded, then unmaps and frees the slot; `ExposureTracker::new()` starts with no frame
/// and no result; `ExposureTracker::begin_frame(frame, size)` returns true on a snap (first call,
/// size change, frame gap, invalidate) and clears the results on a snap;
/// `ExposureTracker::invalidate()` makes the next `begin_frame` snap;
/// `ExposureTracker::state(enabled)` reports the latest result and whether it settles against
/// the one before (disabled: settled; no result: pending).
pub(crate) struct ReadbackSlots {
    buffers: [wgpu::Buffer; READBACK_SLOTS],
    statuses: Arc<Mutex<[SlotStatus; READBACK_SLOTS]>>,
}

impl ReadbackSlots {
    pub(crate) fn new(device: &wgpu::Device) -> ReadbackSlots {
        ReadbackSlots {
            buffers: std::array::from_fn(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("raa-exposure.readback"),
                    size: RESULT_SIZE as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            }),
            statuses: Arc::new(Mutex::new([SlotStatus::Idle; READBACK_SLOTS])),
        }
    }

    /// Records the copy of `state` into the first Idle slot and marks it Copied.
    pub(crate) fn copy(&mut self, encoder: &mut wgpu::CommandEncoder, state: &wgpu::Buffer) {
        let mut statuses = locked(&self.statuses);
        let Some(slot) = first_idle(&statuses) else {
            return;
        };
        encoder.copy_buffer_to_buffer(state, 0, &self.buffers[slot], 0, RESULT_SIZE as u64);
        statuses[slot] = SlotStatus::Copied;
    }

    /// Maps every Copied slot once and polls without waiting. The caller must not hold the sink
    /// lock, because the poll can run the callbacks that lock it.
    pub(crate) fn map_copied(&mut self, gpu: &GpuContext, sink: &Arc<Mutex<ExposureTracker>>) {
        for slot in 0..READBACK_SLOTS {
            {
                let mut statuses = locked(&self.statuses);
                if statuses[slot] != SlotStatus::Copied {
                    continue;
                }
                statuses[slot] = SlotStatus::Mapping;
            }
            let mapped = self.buffers[slot].clone();
            let sink = Arc::clone(sink);
            let statuses = Arc::clone(&self.statuses);
            self.buffers[slot].map_async(wgpu::MapMode::Read, .., move |result| {
                if result.is_ok()
                    && let Some(decoded) = decode(&mapped)
                {
                    locked(&sink).push_result(decoded);
                }
                mapped.unmap();
                locked(&statuses)[slot] = SlotStatus::Idle;
            });
        }
        #[cfg(not(target_arch = "wasm32"))]
        let _ = gpu.device().poll(wgpu::PollType::Poll);
        #[cfg(target_arch = "wasm32")]
        let _ = gpu;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: (u32, u32) = (64, 32);

    fn result(frame: u32, exposure: f32) -> ExposureResult {
        ExposureResult {
            average_luminance: 0.18,
            exposure,
            valid: true,
            frame,
        }
    }

    #[test]
    fn begin_frame_snaps_on_first_call_frame_gap_size_change_and_invalidate() {
        let mut t = ExposureTracker::new();
        assert!(t.begin_frame(0, SIZE));
        assert!(!t.begin_frame(1, SIZE));
        assert!(!t.begin_frame(2, SIZE));
        assert!(!t.begin_frame(3, SIZE));
        assert!(t.begin_frame(5, SIZE));
        assert!(t.begin_frame(6, (32, 32)));
        assert!(!t.begin_frame(7, (32, 32)));
        t.invalidate();
        assert!(t.begin_frame(8, (32, 32)));
        assert!(!t.begin_frame(9, (32, 32)));
    }

    #[test]
    fn snap_clears_results_and_ignores_results_below_the_snap_frame() {
        let mut t = ExposureTracker::new();
        t.begin_frame(0, SIZE);
        t.begin_frame(1, SIZE);
        t.push_result(result(1, 2.0));
        assert!(!t.state(true).pending);
        assert!(t.begin_frame(5, SIZE));
        assert!(t.state(true).pending);
        t.push_result(result(3, 4.0));
        assert!(t.state(true).pending);
        t.push_result(result(5, 4.0));
        assert!(!t.state(true).pending);
    }

    #[test]
    fn disabled_is_settled_and_enabled_without_a_result_is_pending() {
        let t = ExposureTracker::new();
        assert_eq!(
            t.state(false),
            AutoExposureState {
                exposure_ev: None,
                average_luminance: None,
                settled: true,
                pending: false,
            }
        );
        assert_eq!(
            t.state(true),
            AutoExposureState {
                exposure_ev: None,
                average_luminance: None,
                settled: false,
                pending: true,
            }
        );
    }

    #[test]
    fn one_result_reports_its_ev_and_a_second_within_a_hundredth_of_an_ev_settles() {
        let mut t = ExposureTracker::new();
        t.begin_frame(0, SIZE);
        t.push_result(result(0, 2.0));
        let one = t.state(true);
        assert_eq!(one.exposure_ev, Some(2.0f32.log2()));
        assert_eq!(one.average_luminance, Some(0.18));
        assert!(!one.settled);
        assert!(!one.pending);
        for (offset, expect_settled) in [(0.009f32, true), (0.02f32, false)] {
            let mut second = ExposureTracker::new();
            second.begin_frame(0, SIZE);
            second.push_result(result(0, 2.0));
            second.begin_frame(1, SIZE);
            second.push_result(result(1, (1.0 + offset).exp2()));
            let two = second.state(true);
            assert_eq!(two.settled, expect_settled);
            assert!(!two.pending);
        }
    }

    #[test]
    fn invalid_and_stale_results_are_ignored() {
        let mut t = ExposureTracker::new();
        for frame in 0..3 {
            t.begin_frame(frame, SIZE);
        }
        t.push_result(ExposureResult {
            valid: false,
            ..result(1, 2.0)
        });
        assert!(t.state(true).pending);
        t.push_result(result(2, 2.0));
        let kept = t.state(true);
        t.push_result(result(2, 8.0));
        t.push_result(result(1, 8.0));
        assert_eq!(t.state(true), kept);
        assert!(!kept.settled);
    }

    #[test]
    fn first_idle_is_the_lowest_idle_slot_or_none() {
        use SlotStatus::{Copied, Idle, Mapping};
        assert_eq!(first_idle(&[Idle, Idle, Idle]), Some(0));
        assert_eq!(first_idle(&[Copied, Mapping, Idle]), Some(2));
        assert_eq!(first_idle(&[Mapping, Idle, Copied]), Some(1));
        assert_eq!(first_idle(&[Copied, Mapping, Copied]), None);
    }
}
