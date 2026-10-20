//! Allocation-conscious CPU frame timing statistics.

use crate::FrameCounters;

/// A fixed-capacity ring containing recent CPU frame times in milliseconds.
#[derive(Clone, Debug)]
pub struct CpuRing {
    values: Vec<f64>,
    next: usize,
}

impl CpuRing {
    /// Creates a ring with storage reserved for exactly `capacity` samples.
    pub fn new(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity),
            next: 0,
        }
    }

    /// Adds a sample, replacing the oldest sample when the ring is full.
    pub fn push(&mut self, ms: f64) {
        if self.values.is_empty() && self.values.capacity() == 0 {
            return;
        }
        if self.values.len() < self.values.capacity() {
            self.values.push(ms);
        } else {
            self.values[self.next] = ms;
        }
        self.next = (self.next + 1) % self.values.capacity();
    }

    /// Returns the number of samples currently stored.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether this ring currently has no samples.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Computes median and p95 using caller-provided, preallocated scratch storage.
    ///
    /// The scratch vector should have capacity at least `self.len()` to avoid
    /// allocation while computing percentiles.
    pub fn percentiles(&self, scratch: &mut Vec<f64>) -> (f64, f64) {
        scratch.clear();
        scratch.extend_from_slice(&self.values);
        if scratch.is_empty() {
            return (0.0, 0.0);
        }
        scratch.sort_by(f64::total_cmp);

        let middle = scratch.len() / 2;
        let median = if scratch.len().is_multiple_of(2) {
            (scratch[middle - 1] + scratch[middle]) / 2.0
        } else {
            scratch[middle]
        };
        let p95_index = (scratch.len() * 95).div_ceil(100) - 1;
        (median, scratch[p95_index])
    }
}

/// Summary of frame timing and application work.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    pub frames: u64,
    pub first_frame_ms: Option<f64>,
    pub cpu_ms_median: f64,
    pub cpu_ms_p95: f64,
    pub counters: FrameCounters,
    pub allocs_last_frame: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::CpuRing;

    #[test]
    fn computes_median_and_p95_for_known_data() {
        let mut ring = CpuRing::new(100);
        for value in 1..=100 {
            ring.push(f64::from(value));
        }
        let (median, p95) = ring.percentiles(&mut Vec::with_capacity(100));
        assert_eq!(median, 50.5);
        assert_eq!(p95, 95.0);
    }

    #[test]
    fn wrap_around_keeps_only_last_capacity_values() {
        let mut ring = CpuRing::new(3);
        for value in 1..=5 {
            ring.push(f64::from(value));
        }
        assert_eq!(ring.len(), 3);
        let (median, p95) = ring.percentiles(&mut Vec::with_capacity(3));
        assert_eq!((median, p95), (4.0, 5.0));
    }

    #[test]
    fn empty_ring_has_zero_percentiles() {
        let ring = CpuRing::new(4);
        assert_eq!(ring.percentiles(&mut Vec::with_capacity(4)), (0.0, 0.0));
    }

    #[test]
    fn pushes_and_percentiles_preserve_preallocated_capacities() {
        let mut ring = CpuRing::new(8);
        let ring_capacity = ring.values.capacity();
        let mut scratch = Vec::with_capacity(8);
        let scratch_capacity = scratch.capacity();
        for value in 0..32 {
            ring.push(f64::from(value));
            let _ = ring.percentiles(&mut scratch);
            assert_eq!(scratch.capacity(), scratch_capacity);
            assert_eq!(ring.values.capacity(), ring_capacity);
        }
    }
}
