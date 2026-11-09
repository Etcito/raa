//! Bounded asynchronous overflow readback (M2-LIGHT-STATS), device-free state model.
#![allow(dead_code)] // Staged before the tiling/renderer shares land; see Handoff.
use raa_shader::TILE_MAX_LIGHTS;
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
/// `submitted` per scheduled copy, `completed` from its map callback with that view's
/// drops, `finish` last. A full pool defers without growth; `LocalLightStats::from_coordinator`
/// maps the newest complete sum and pending status for the renderer.
pub(crate) struct ReadbackCoordinator {
    samples: [Sample; POOL_SAMPLES],
    published: Option<(u64, u64, (u32, u32))>,
    latest_grid: (u32, u32),
}

impl ReadbackCoordinator {
    #[allow(clippy::new_without_default)]
    pub(crate) fn new() -> Self {
        ReadbackCoordinator {
            samples: [Sample::default(); POOL_SAMPLES],
            published: None,
            latest_grid: (0, 0),
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
        if self.published.is_some_and(|(p, ..)| index <= p) {
            return;
        }
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
        let pending = self.samples.iter().any(|s| {
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
        self.samples[slot] = Sample::default();
        let newest = self.published.map_or(0, |(p, ..)| p);
        for stale in &mut self.samples {
            if stale.status != SampleStatus::Free && stale.index <= newest {
                *stale = Sample::default();
            }
        }
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
}
