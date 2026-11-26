//! On-demand GPU timestamp intervals for single-view feature passes.

use std::sync::mpsc::{Receiver, TryRecvError, sync_channel};

use raa_core::GpuContext;

/// The result of requesting a GPU pass timing sample.
pub enum PassGpuTimings<'a> {
    /// Timing is unavailable for this device or pipeline configuration.
    Unavailable(&'static str),
    /// The first frame has yet to be measured and read back.
    Pending,
    /// The last completed sample.
    Sample(&'a PassGpuTimingSample),
}

/// GPU intervals from one submitted single-view frame.
pub struct PassGpuTimingSample {
    /// Zero-based frame graph index.
    pub frame: u64,
    /// Display width and height when the frame was recorded.
    pub size: (u32, u32),
    /// Feature pass names and milliseconds in pipeline order; `None` if a query was invalid.
    pub passes: Vec<(String, Option<f64>)>,
}

pub(crate) const MAX_TIMED_PASSES: usize = wgpu::QUERY_SET_MAX_QUERIES as usize / 2;

#[derive(Default)]
pub(crate) struct PassTimings {
    requested: bool,
    buffers: Option<TimingBuffers>,
    pending: Option<Receiver<Option<PassGpuTimingSample>>>,
    /// The generation `pending` was recorded under.
    pending_generation: u64,
    latest: Option<PassGpuTimingSample>,
    readback_failed: bool,
    /// Bumped when the pass configuration changes (a render graph switch, a pass enabled or
    /// disabled, a new order): samples recorded before it describe another pass set.
    generation: u64,
}

impl PassTimings {
    /// Forgets the latest sample and ignores the readback in flight: both were recorded with the
    /// previous pass configuration. The in-flight receiver is kept (its callback still unmaps the
    /// readback buffer before the buffer is reused) and its sample is dropped when it arrives.
    pub(crate) fn invalidate(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.latest = None;
        self.readback_failed = false;
    }

    pub(crate) fn request(&mut self) -> PassGpuTimings<'_> {
        if let Some(pending) = &self.pending {
            match pending.try_recv() {
                Ok(_) if self.pending_generation != self.generation => {
                    self.pending = None;
                }
                Ok(sample) => {
                    self.readback_failed = sample.is_none();
                    self.latest = sample;
                    self.pending = None;
                }
                Err(TryRecvError::Disconnected) => {
                    if self.pending_generation == self.generation {
                        self.latest = None;
                        self.readback_failed = true;
                    }
                    self.pending = None;
                }
                Err(TryRecvError::Empty) => {}
            }
        }
        self.requested = self.pending.is_none();

        match &self.latest {
            Some(sample) => PassGpuTimings::Sample(sample),
            None if self.readback_failed => {
                PassGpuTimings::Unavailable("timestamp readback failed")
            }
            None => PassGpuTimings::Pending,
        }
    }

    pub(crate) fn is_requested(&self) -> bool {
        self.requested
    }

    pub(crate) fn begin(
        &mut self,
        gpu: &GpuContext,
        pass_count: usize,
    ) -> Option<TimingCapture<'_>> {
        self.requested = false;
        if !(1..=MAX_TIMED_PASSES).contains(&pass_count) {
            return None;
        }
        let count = pass_count as u32 * 2;
        if self
            .buffers
            .as_ref()
            .is_none_or(|buffers| buffers.queries.count() < count)
        {
            self.buffers = Some(TimingBuffers::new(gpu.device(), count.next_power_of_two()));
        }
        self.pending_generation = self.generation;
        Some(TimingCapture {
            buffers: self.buffers.as_ref().expect("timing buffers initialized"),
            passes: Vec::with_capacity(pass_count),
            pending: &mut self.pending,
            ms_per_tick: f64::from(gpu.queue().get_timestamp_period()) / 1e6,
        })
    }
}

struct TimingBuffers {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
}

impl TimingBuffers {
    fn new(device: &wgpu::Device, count: u32) -> Self {
        let label = Some("raa.pipeline.pass-timings");
        let size = u64::from(count * wgpu::QUERY_SIZE);
        let buffer = |usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label,
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        Self {
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label,
                ty: wgpu::QueryType::Timestamp,
                count,
            }),
            resolve: buffer(wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC),
            readback: buffer(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
        }
    }
}

pub(crate) struct TimingCapture<'a> {
    buffers: &'a TimingBuffers,
    passes: Vec<String>,
    pending: &'a mut Option<Receiver<Option<PassGpuTimingSample>>>,
    ms_per_tick: f64,
}

impl TimingCapture<'_> {
    pub(crate) fn begin_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        self.marker(encoder, None, Some(self.passes.len() as u32 * 2));
    }

    pub(crate) fn end_pass(&mut self, encoder: &mut wgpu::CommandEncoder, name: &str) {
        self.marker(encoder, Some(self.passes.len() as u32 * 2 + 1), None);
        self.passes.push(name.to_owned());
    }

    fn marker(&self, encoder: &mut wgpu::CommandEncoder, begin: Option<u32>, end: Option<u32>) {
        // WebGPU exposes timestamps on pass boundaries. A feature pass may encode several
        // render/compute passes and copies, so bracket it with empty timestamped passes.
        let _marker = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("raa.pipeline.timestamp"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &self.buffers.queries,
                beginning_of_pass_write_index: begin,
                end_of_pass_write_index: end,
            }),
        });
    }

    pub(crate) fn resolve(self, encoder: &mut wgpu::CommandEncoder, frame: u64, size: (u32, u32)) {
        let count = self.passes.len() as u32 * 2;
        let bytes = u64::from(count * wgpu::QUERY_SIZE);
        let buffers = self.buffers;
        encoder.resolve_query_set(&buffers.queries, 0..count, &buffers.resolve, 0);
        encoder.copy_buffer_to_buffer(&buffers.resolve, 0, &buffers.readback, 0, bytes);

        // Register on the same command buffer so failed or skipped submissions never map
        // stale data. The buffer is reused only after its previous callback has unmapped it.
        let (send, receive) = sync_channel(1);
        let readback = buffers.readback.clone();
        let ms_per_tick = self.ms_per_tick;
        let names = self.passes;
        encoder.map_buffer_on_submit(
            &buffers.readback,
            wgpu::MapMode::Read,
            0..bytes,
            move |result| {
                let sample = result.ok().and_then(|()| {
                    let passes = readback
                        .slice(0..bytes)
                        .get_mapped_range()
                        .ok()
                        .map(|data| measured_passes(names, &data, ms_per_tick));
                    readback.unmap();
                    passes.map(|passes| PassGpuTimingSample {
                        frame,
                        size,
                        passes,
                    })
                });
                let _ = send.send(sample);
            },
        );
        *self.pending = Some(receive);
    }
}

fn measured_passes(
    names: Vec<String>,
    data: &[u8],
    ms_per_tick: f64,
) -> Vec<(String, Option<f64>)> {
    names
        .into_iter()
        .zip(data.chunks_exact(16))
        .map(|(name, pair)| {
            let begin = u64::from_le_bytes(pair[..8].try_into().unwrap());
            let end = u64::from_le_bytes(pair[8..].try_into().unwrap());
            let ms = end
                .checked_sub(begin)
                .filter(|_| begin != 0 && ms_per_tick.is_finite() && ms_per_tick > 0.0)
                .map(|ticks| ticks as f64 * ms_per_tick);
            (name, ms)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::measured_passes;

    #[test]
    fn timestamps_use_tick_period_and_preserve_quantized_zero() {
        let names = ["measured", "quantized", "missing", "reversed"]
            .map(String::from)
            .to_vec();
        let data: Vec<_> = [10_u64, 30, 31, 31, 0, 0, 40, 39]
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect();
        let measured = measured_passes(names, &data, 0.125);
        assert_eq!(
            measured,
            vec![
                ("measured".into(), Some(2.5)),
                ("quantized".into(), Some(0.0)),
                ("missing".into(), None),
                ("reversed".into(), None),
            ]
        );
    }
}
