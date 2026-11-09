//! Bounded asynchronous overflow readback (M2-LIGHT-STATS) with a device-free state model.
#![allow(dead_code)] // Staged before the tiling/renderer shares land; see Handoff.
use raa_core::GpuContext;
use raa_pipeline::SubmissionState;
use raa_shader::TILE_MAX_LIGHTS;
use std::sync::{Arc, Mutex};
const POOL_SAMPLES: usize = 3;

/// Latest completed tile-overflow statistics of one renderer invocation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LocalLightStats {
    pub active_lights: u32,
    pub attached_lights: u32,
    pub tile_columns: u32,
    pub tile_rows: u32,
    pub tile_capacity: u32,
    pub dropped_light_references: Option<u64>,
    pub stats_frame_index: Option<u64>,
    pub pending: bool,
}

impl LocalLightStats {
    pub(crate) fn from_coordinator(
        active_lights: u32,
        attached_lights: u32,
        coordinator: &ReadbackCoordinator,
    ) -> LocalLightStats {
        let (frame, dropped, pending) = coordinator.latest();
        let (columns, rows) = coordinator
            .published
            .map_or(coordinator.latest_grid, |p| p.2);
        LocalLightStats {
            active_lights,
            attached_lights,
            tile_columns: columns,
            tile_rows: rows,
            tile_capacity: TILE_MAX_LIGHTS,
            dropped_light_references: dropped,
            stats_frame_index: frame,
            pending,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Default)]
enum ViewStatus {
    #[default]
    Idle,
    Submitted,
    Complete(u64),
}

#[derive(Copy, Clone, PartialEq, Eq, Default)]
enum SampleStatus {
    #[default]
    Free,
    Open,
    Awaiting,
}

#[derive(Copy, Clone, Default)]
struct Sample {
    status: SampleStatus,
    index: u64,
    viewport0: (u32, u32),
    views: [ViewStatus; 4],
    copied: u8,
    /// A stale sample still owns its staging buffer until the callback unmaps it.
    mapping: bool,
}

/// Fixed-capacity readback pool of submitted scene invocations.
///
/// Handoff: `pub(crate) struct ReadbackCoordinator` with `begin(&mut self, index: u64,
/// viewport0: (u32, u32))`, `submitted(&mut self, index: u64, view: usize)`,
/// `completed(&mut self, index: u64, view: usize, dropped: u64)`,
/// `finish(&mut self, index: u64)` and `latest(&self) -> (Option<u64>, Option<u64>, bool)`
/// in `crates/raa-lit/src/light_stats.rs` (`mod light_stats;` in `lib.rs`). The order-35
/// tiling share stores one `Arc<Mutex<ReadbackCoordinator>>` with three fixed staging
/// buffers, keyed by invocation index and view 0..3: `begin` with viewport 0's grid,
/// `submitted` only for actual queue submissions, `completed` from its map callback
/// with that view's drops, `finish` last. A full pool defers without growth;
/// `LocalLightStats::from_coordinator` maps the newest complete sum and pending status
/// for the renderer.
pub(crate) struct ReadbackCoordinator {
    samples: [Sample; POOL_SAMPLES],
    published: Option<(u64, u64, (u32, u32))>,
    latest_grid: (u32, u32),
    staging: Option<[wgpu::Buffer; POOL_SAMPLES]>,
    deferred: Option<u64>,
}

impl ReadbackCoordinator {
    #[allow(clippy::new_without_default)]
    pub(crate) fn new() -> Self {
        ReadbackCoordinator {
            samples: [Sample::default(); POOL_SAMPLES],
            published: None,
            latest_grid: (0, 0),
            staging: None,
            deferred: None,
        }
    }

    pub(crate) fn begin(&mut self, index: u64, viewport0: (u32, u32)) {
        if self.published.is_some_and(|(p, ..)| index <= p)
            || self
                .samples
                .iter()
                .any(|s| s.status != SampleStatus::Free && s.index == index)
        {
            return;
        }
        let Some(slot) = self
            .samples
            .iter()
            .position(|s| s.status == SampleStatus::Free)
        else {
            return;
        };
        self.samples[slot] = Sample {
            status: SampleStatus::Open,
            index,
            viewport0,
            views: [ViewStatus::Idle; 4],
            ..Sample::default()
        };
        self.latest_grid = viewport0;
    }

    pub(crate) fn submitted(&mut self, index: u64, view: usize) {
        let Some(status) = self.tracked_mut(index).and_then(|s| s.views.get_mut(view)) else {
            return;
        };
        if *status == ViewStatus::Idle {
            *status = ViewStatus::Submitted;
        }
    }

    pub(crate) fn completed(&mut self, index: u64, view: usize, dropped: u64) {
        {
            let Some(status) = self.tracked_mut(index).and_then(|s| s.views.get_mut(view)) else {
                return;
            };
            if *status != ViewStatus::Submitted {
                return;
            }
            *status = ViewStatus::Complete(dropped);
        }
        self.publish(index);
    }

    pub(crate) fn finish(&mut self, index: u64) {
        if let Some(sample) = self.tracked_mut(index)
            && sample.status == SampleStatus::Open
        {
            sample.status = SampleStatus::Awaiting;
        }
        self.publish(index);
    }

    pub(crate) fn latest(&self) -> (Option<u64>, Option<u64>, bool) {
        let pending = self
            .deferred
            .is_some_and(|i| self.published.is_none_or(|(p, ..)| i > p))
            || self.samples.iter().any(|s| {
                s.status != SampleStatus::Free && self.published.is_none_or(|(p, ..)| s.index > p)
            });
        self.published
            .map_or((None, None, pending), |(index, dropped, _)| {
                (Some(index), Some(dropped), pending)
            })
    }

    fn tracked_mut(&mut self, index: u64) -> Option<&mut Sample> {
        self.samples
            .iter_mut()
            .find(|s| s.status != SampleStatus::Free && s.index == index)
    }

    fn publish(&mut self, index: u64) {
        let Some(slot) = self
            .samples
            .iter()
            .position(|s| s.status == SampleStatus::Awaiting && s.index == index)
        else {
            return;
        };
        let sample = self.samples[slot];
        if sample.views.contains(&ViewStatus::Submitted) {
            return;
        }
        let dropped = sample
            .views
            .iter()
            .filter_map(|v| match v {
                ViewStatus::Complete(dropped) => Some(*dropped),
                _ => None,
            })
            .sum();
        if self.published.is_none_or(|(p, ..)| index > p) {
            self.published = Some((index, dropped, sample.viewport0));
        }
        let newest = self.published.map_or(0, |(p, ..)| p);
        for stale in &mut self.samples {
            if stale.status != SampleStatus::Free
                && stale.index <= newest
                && !stale.mapping
                && (stale.copied == 0 || stale.status == SampleStatus::Awaiting)
            {
                *stale = Sample::default();
            }
        }
    }

    pub(crate) fn attach(&mut self, gpu: &GpuContext) {
        self.staging.get_or_insert_with(|| {
            std::array::from_fn(|_| {
                gpu.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("raa.lit.overflow.readback"),
                    size: 16,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
        });
    }

    pub(crate) fn copy(
        &mut self,
        index: u64,
        view: usize,
        encoder: &mut wgpu::CommandEncoder,
        counter: &wgpu::Buffer,
    ) {
        let Some(slot) = self.samples.iter().position(|s| {
            s.status == SampleStatus::Open
                && s.index == index
                && view < 4
                && s.copied & (1 << view) == 0
        }) else {
            return;
        };
        if let Some(buffers) = &self.staging {
            encoder.copy_buffer_to_buffer(counter, 0, &buffers[slot], view as u64 * 4, 4);
            self.samples[slot].copied |= 1 << view;
        }
    }

    fn observe(
        &mut self,
        index: u64,
        observation: SubmissionState,
        final_submitted: bool,
    ) -> Option<usize> {
        let slot = self
            .samples
            .iter()
            .position(|s| s.status == SampleStatus::Open && s.index == index)?;
        let mask = observation.submitted_mask
            | if final_submitted && !observation.record_failed {
                observation.pending_mask
            } else {
                0
            };
        let copied = self.samples[slot].copied;
        self.samples[slot].mapping = copied & mask != 0;
        for view in 0..4 {
            if mask & (1 << view) != 0 {
                self.submitted(index, view);
                if copied & (1 << view) == 0 {
                    self.completed(index, view, 0);
                }
            }
        }
        self.finish(index);
        (copied & mask != 0).then_some(slot)
    }

    fn mapped(&mut self, index: u64, values: Option<[u64; 4]>) {
        let Some(slot) = self
            .samples
            .iter()
            .position(|s| s.index == index && s.mapping)
        else {
            return;
        };
        let sample = self.samples[slot];
        if let Some(values) = values {
            for (view, value) in values.into_iter().enumerate() {
                if sample.views[view] == ViewStatus::Submitted {
                    self.completed(index, view, value);
                }
            }
        } else {
            self.deferred = Some(self.deferred.map_or(index, |p| p.max(index)));
        }
        self.samples[slot] = Sample::default();
    }

    /// Handoff: `ReadbackCoordinator::resolve(shared: &Arc<Mutex<Self>>, gpu: &GpuContext,
    /// index: u64, observation: SubmissionState, final_submitted: bool) -> ()` in
    /// `crates/raa-lit/src/light_stats.rs` closes the invocation after runtime acknowledgment.
    /// The renderer share calls `begin` with viewport 0's grid and passes the same registry
    /// through `TileInputs::readback`, plus invocation `index` and viewport `view`.
    /// `LightTiles::new` attaches three 16-byte buffers; recording copies each counter
    /// into its viewport offset. `resolve` consumes only observed submissions (pending
    /// views require successful runtime submission), then maps once per invocation.
    pub(crate) fn resolve(
        shared: &Arc<Mutex<Self>>,
        gpu: &GpuContext,
        index: u64,
        observation: SubmissionState,
        final_submitted: bool,
    ) {
        let buffer = {
            let mut c = shared.lock().unwrap_or_else(|e| e.into_inner());
            c.observe(index, observation, final_submitted)
                .and_then(|slot| c.staging.as_ref().map(|b| b[slot].clone()))
        };
        if let Some(buffer) = buffer {
            let shared = Arc::clone(shared);
            let mapped = buffer.clone();
            buffer.map_async(wgpu::MapMode::Read, .., move |result| {
                let values = result.ok().and_then(|()| {
                    let bytes = mapped.get_mapped_range(..).ok()?;
                    let mut values = [0; 4];
                    for (value, word) in values.iter_mut().zip(bytes.chunks_exact(4)) {
                        *value = u64::from(u32::from_le_bytes(word.try_into().ok()?));
                    }
                    Some(values)
                });
                mapped.unmap();
                shared
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .mapped(index, values);
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

    const GRID: (u32, u32) = (80, 45);

    fn cycle(c: &mut ReadbackCoordinator, index: u64, views: &[(usize, u64)]) {
        c.begin(index, GRID);
        for &(view, _) in views {
            c.submitted(index, view);
        }
        c.finish(index);
        for &(view, dropped) in views {
            c.completed(index, view, dropped);
        }
    }

    #[test]
    fn pending_ready_and_out_of_order_callbacks() {
        let mut c = ReadbackCoordinator::new();
        c.begin(4, GRID);
        assert_eq!(c.latest(), (None, None, true));
        c.submitted(4, 0);
        c.submitted(4, 2);
        c.completed(4, 0, 7);
        c.finish(4);
        assert_eq!(c.latest(), (None, None, true));
        c.completed(4, 2, 5);
        assert_eq!(c.latest(), (Some(4), Some(12), false));
        let s = LocalLightStats::from_coordinator(3, 1, &c);
        assert_eq!(
            (s.stats_frame_index, s.dropped_light_references),
            (Some(4), Some(12))
        );
        assert_eq!(
            (s.tile_columns, s.tile_rows, s.tile_capacity),
            (80, 45, 127)
        );
        assert_eq!(
            (s.active_lights, s.attached_lights, s.pending),
            (3, 1, false)
        );
        let mut c = ReadbackCoordinator::new();
        c.begin(9, (40, 23));
        for view in 0..4 {
            c.submitted(9, view);
        }
        c.finish(9);
        c.completed(9, 3, 1);
        c.completed(9, 1, 2);
        c.completed(9, 0, 4);
        assert_eq!(c.latest(), (None, None, true));
        c.completed(9, 2, 8);
        assert_eq!(c.latest(), (Some(9), Some(15), false));
    }

    #[test]
    fn stale_or_cancelled_samples_never_replace_newer_data() {
        let mut c = ReadbackCoordinator::new();
        cycle(&mut c, 10, &[(0, 3)]);
        c.begin(11, GRID);
        c.submitted(11, 0);
        assert_eq!(c.latest(), (Some(10), Some(3), true));
        c.completed(10, 0, 99);
        assert_eq!(c.latest(), (Some(10), Some(3), true));
        c.finish(11);
        c.completed(11, 0, 5);
        c.begin(12, GRID);
        c.submitted(12, 0);
        c.finish(12);
        cycle(&mut c, 13, &[(0, 7)]);
        assert_eq!(c.latest(), (Some(13), Some(7), false));
    }

    #[test]
    fn full_pool_defers_and_empty_invocations_publish_zero() {
        let mut c = ReadbackCoordinator::new();
        cycle(&mut c, 1, &[(0, 11)]);
        for index in 2..=4 {
            c.begin(index, GRID);
            c.submitted(index, 0);
        }
        c.begin(5, GRID);
        c.submitted(5, 0);
        c.finish(5);
        c.completed(5, 0, 55);
        assert_eq!(c.latest(), (Some(1), Some(11), true));
        for (index, dropped) in [(2, 22), (4, 44), (3, 33)] {
            c.finish(index);
            c.completed(index, 0, dropped);
        }
        assert_eq!(c.latest(), (Some(4), Some(44), false));
        cycle(&mut c, 6, &[]);
        assert_eq!(c.latest(), (Some(6), Some(0), false));
    }

    fn recorded(c: &mut ReadbackCoordinator, index: u64, copied: u8) {
        c.begin(index, (index as u32, 2));
        c.tracked_mut(index).expect("reserved sample").copied = copied;
    }

    fn observation(submitted: u8, pending: u8, record_failed: bool) -> SubmissionState {
        SubmissionState {
            submitted_mask: submitted,
            pending_mask: pending,
            record_failed,
        }
    }

    #[test]
    fn submitted_view_sums_keep_identity_in_either_callback_order() {
        for newest_first in [false, true] {
            let mut c = ReadbackCoordinator::new();
            recorded(&mut c, 2, 0b1011);
            recorded(&mut c, 3, 0b1100);
            assert!(
                c.observe(2, observation(0b0011, 0b1000, false), true)
                    .is_some()
            );
            assert!(
                c.observe(3, observation(0b0100, 0b1000, false), true)
                    .is_some()
            );
            assert_eq!(c.latest(), (None, None, true));
            let large = u64::from(u32::MAX);
            if newest_first {
                c.mapped(3, Some([99, 99, large, 9]));
                assert_eq!(c.latest(), (Some(3), Some(large + 9), false));
                c.mapped(2, Some([1, 2, 99, 8]));
            } else {
                c.mapped(2, Some([1, 2, 99, 8]));
                assert_eq!(c.latest(), (Some(2), Some(11), true));
                c.mapped(3, Some([99, 99, large, 9]));
            }
            c.mapped(2, Some([100; 4])); // Duplicate/old callback is harmless.
            let stats = LocalLightStats::from_coordinator(4, 1, &c);
            assert_eq!((stats.tile_columns, stats.tile_rows), (3, 2));
            assert_eq!(c.latest(), (Some(3), Some(large + 9), false));
        }
    }

    #[test]
    fn incomplete_and_disabled_invocations_count_only_acknowledged_views() {
        for (submitted, pending, failed, acknowledged, copied, expected) in [
            (0, 0, false, true, 15, 0),
            (0, 8, false, false, 15, 0),
            (1, 8, false, false, 15, 1),
            (1, 8, true, true, 15, 1),
            (1, 8, false, true, 15, 9),
            (5, 8, false, true, 0, 0), // Disabled tiling: no counter copy.
        ] {
            let mut c = ReadbackCoordinator::new();
            recorded(&mut c, 7, copied);
            if c.observe(7, observation(submitted, pending, failed), acknowledged)
                .is_some()
            {
                assert_eq!(c.latest(), (None, None, true));
                assert!(c.observe(7, observation(15, 0, false), true).is_none());
                c.mapped(7, Some([1, 2, 4, 8]));
            }
            assert_eq!(c.latest(), (Some(7), Some(expected), false));
        }
    }

    #[test]
    fn stale_maps_reserve_buffers_and_failed_maps_release_without_partial_publication() {
        let mut c = ReadbackCoordinator::new();
        for index in 1..=3 {
            recorded(&mut c, index, 1);
            c.observe(index, observation(1, 0, false), false);
        }
        c.mapped(3, Some([30; 4]));
        recorded(&mut c, 4, 3);
        c.observe(4, observation(3, 0, false), false);
        c.begin(5, GRID);
        assert!(c.tracked_mut(5).is_none()); // Old maps still own two buffers.
        assert_eq!(c.latest(), (Some(3), Some(30), true));
        c.mapped(4, None);
        assert_eq!(c.latest(), (Some(3), Some(30), true));
        for index in 6..100 {
            recorded(&mut c, index, 1);
            c.observe(index, observation(1, 0, false), false);
            c.mapped(index, Some([index; 4]));
            assert_eq!(c.samples.len(), POOL_SAMPLES);
            assert_eq!(c.latest(), (Some(index), Some(index), false));
        }
        c.mapped(1, Some([1000; 4]));
        c.mapped(2, None);
        assert_eq!(c.latest(), (Some(99), Some(99), false));
        assert!(c.samples.iter().all(|s| s.status == SampleStatus::Free));
    }

    #[test]
    fn recorded_buffers_remain_reserved_until_submission_is_resolved() {
        for submitted in [0, 1, 2] {
            let mut c = ReadbackCoordinator::new();
            recorded(&mut c, 1, 1);
            cycle(&mut c, 2, &[(0, 20)]);
            assert!(c.tracked_mut(1).is_some());
            c.observe(1, observation(submitted, 0, false), false);
            c.mapped(1, Some([10; 4]));
            assert_eq!(c.latest(), (Some(2), Some(20), false));
            assert!(c.tracked_mut(1).is_none());
        }
    }
}
