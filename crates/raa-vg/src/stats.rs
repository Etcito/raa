//! Asynchronous cull statistics (S31, module raa-vg-resolve `stats.rs`): a bounded pool of three
//! staging samples reads the `CullState` counters of the main view and of the four cascades once
//! per completely submitted invocation, the pattern of `raa-lit/src/light_stats.rs`. A capture
//! copies the counters into a free sample; `observe_submission` marks the captured samples with
//! their invocation and maps them when the submission was complete; the map callback parses them
//! and publishes the newest one. A full pool skips the capture: nothing grows and nothing waits.
//!
//! At most one submitted sample may be outstanding at a time: a capture is skipped while a map
//! is still in flight. A page that changes the scene and waits for two newer published frames
//! (the S33 `settle` rule) therefore never reads counters older than the change twice, and a
//! still-mapped buffer only pauses new captures; it never stalls rendering or a later frame.

use std::sync::{Arc, Mutex, MutexGuard};

use raa_core::GpuContext;

use crate::culling::CullWork;

/// Staging samples in the pool.
const POOL_SAMPLES: usize = 3;
/// Views per sample: the main view, then the four cascades.
const VIEWS: usize = 5;
/// Bytes of one `CullState` (`pass_view.wgsl`).
const STATE_SIZE: u64 = 64;
/// Bytes of one staging sample: one `CullState` per view.
const SAMPLE_SIZE: u64 = STATE_SIZE * VIEWS as u64;

/// The counters of one view, read from its `CullState`.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct CullSample {
    pub clusters: u32,
    pub triangles: u32,
    pub software: u32,
    pub dropped: u32,
    /// Hierarchy entries (instance roots, nodes, clusters) the frustum rejected.
    pub frustum_culled: u32,
    /// Entries the first phase rejected against the previous frame's pyramid.
    pub occlusion_rejected: u32,
    /// Clusters the second phase recovered from those (drawn after all).
    pub occlusion_recovered: u32,
}

/// The counters of one completely submitted invocation. `shadow_clusters` sums the four
/// cascades' clusters and `dropped` sums the dropped counts of all five views.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct VgCullTotals {
    pub invocation: u64,
    pub main: CullSample,
    pub shadow_clusters: u32,
    pub dropped: u32,
}

/// Reads the counters of one 64-byte `CullState` (`CullStateRead` in `pass_view.wgsl`): the
/// clusters are the software plus the hardware count, then triangles, dropped entries, the
/// frustum and first-phase occlusion rejections, and the clusters the second phase drew (the
/// counts past `sw_base` / `hw_base`).
pub fn parse_state(bytes: &[u8; 64]) -> CullSample {
    let mut words = [0u32; 16];
    for (word, chunk) in words.iter_mut().zip(bytes.chunks_exact(4)) {
        *word = u32::from_le_bytes(chunk.try_into().unwrap_or([0; 4]));
    }
    let [
        _,
        _,
        _,
        _,
        software,
        hardware,
        sw_base,
        hw_base,
        triangles,
        dropped,
        _,
        frustum_culled,
        occlusion_rejected,
        ..,
    ] = words;
    CullSample {
        clusters: software.saturating_add(hardware),
        triangles,
        software,
        dropped,
        frustum_culled,
        occlusion_rejected,
        occlusion_recovered: software
            .saturating_sub(sw_base)
            .saturating_add(hardware.saturating_sub(hw_base)),
    }
}

/// Reads the five views of one sample into the totals of `invocation`.
fn read_totals(bytes: &[u8], invocation: u64) -> Option<VgCullTotals> {
    let mut views = [CullSample::default(); VIEWS];
    for (index, view) in views.iter_mut().enumerate() {
        let start = index * STATE_SIZE as usize;
        let chunk: &[u8; 64] = bytes.get(start..start + 64)?.try_into().ok()?;
        *view = parse_state(chunk);
    }
    let [main, cascades @ ..] = views;
    Some(VgCullTotals {
        invocation,
        main,
        shadow_clusters: cascades
            .iter()
            .fold(0, |sum, view| sum.saturating_add(view.clusters)),
        dropped: views
            .iter()
            .fold(0, |sum, view| sum.saturating_add(view.dropped)),
    })
}

#[derive(Copy, Clone, Default, PartialEq, Eq)]
enum Phase {
    #[default]
    Free,
    /// Copied this frame; no submission observed yet.
    Awaiting,
    /// Submitted completely and mapped, or about to be; the callback frees it.
    Mapping,
}

/// The pool state, shared with the map callbacks.
#[derive(Default)]
struct Shared {
    samples: [Phase; POOL_SAMPLES],
    latest: Option<VgCullTotals>,
}

/// The three-sample asynchronous readback of the cull counters (see the module docs).
pub struct VgStatsReadback {
    shared: Arc<Mutex<Shared>>,
    staging: [wgpu::Buffer; POOL_SAMPLES],
}

impl VgStatsReadback {
    /// Creates the three staging samples of 240 bytes each (`MAP_READ | COPY_DST`).
    pub fn new(gpu: &GpuContext) -> VgStatsReadback {
        VgStatsReadback {
            shared: Arc::default(),
            staging: std::array::from_fn(|_| {
                gpu.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("raa-vg stats readback"),
                    size: SAMPLE_SIZE,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            }),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Shared> {
        self.shared.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copies the main view's counters, then the four cascades' (zeroed when `cascades` is
    /// `None`), into a free sample awaiting submission. A frame is skipped while an earlier map
    /// is in flight or every sample is busy: nothing grows and nothing waits.
    pub fn capture(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        main: &CullWork,
        cascades: Option<&[CullWork; 4]>,
    ) {
        let slot = {
            let mut shared = self.lock();
            // One submitted sample in flight at a time: the next capture waits for its map, so
            // no page ever reads two readbacks older than a change it just made.
            if shared.samples.contains(&Phase::Mapping) {
                return;
            }
            let Some(slot) = shared
                .samples
                .iter()
                .position(|phase| *phase == Phase::Free)
            else {
                return;
            };
            shared.samples[slot] = Phase::Awaiting;
            slot
        };
        let buffer = &self.staging[slot];
        encoder.copy_buffer_to_buffer(main.state(), 0, buffer, 0, STATE_SIZE);
        match cascades {
            Some(works) => {
                for (index, work) in works.iter().enumerate() {
                    let offset = STATE_SIZE * (index as u64 + 1);
                    encoder.copy_buffer_to_buffer(work.state(), 0, buffer, offset, STATE_SIZE);
                }
            }
            None => encoder.clear_buffer(buffer, STATE_SIZE, Some(STATE_SIZE * 4)),
        }
    }

    /// Gives every sample awaiting submission the id `invocation`. A complete submission maps
    /// them; an incomplete one frees them, since their counters were never submitted. The maps
    /// are requested, not waited for: on the web the browser resolves them without a device
    /// poll, and a buffer that is still mapped is never reused or waited on.
    pub fn observe_submission(&mut self, invocation: u64, complete: bool) {
        let mut mapped = 0u8;
        {
            let mut shared = self.lock();
            for (slot, phase) in shared.samples.iter_mut().enumerate() {
                if *phase != Phase::Awaiting {
                    continue;
                }
                if complete {
                    *phase = Phase::Mapping;
                    mapped |= 1 << slot;
                } else {
                    *phase = Phase::Free;
                }
            }
        }
        for (slot, buffer) in self.staging.iter().enumerate() {
            if mapped & (1 << slot) == 0 {
                continue;
            }
            let shared = Arc::clone(&self.shared);
            let mapping = buffer.clone();
            buffer.map_async(wgpu::MapMode::Read, .., move |result| {
                let totals = result.ok().and_then(|()| {
                    let bytes = mapping.get_mapped_range(..).ok()?;
                    read_totals(&bytes, invocation)
                });
                mapping.unmap();
                let mut shared = shared.lock().unwrap_or_else(|e| e.into_inner());
                shared.samples[slot] = Phase::Free;
                if let Some(totals) = totals
                    && shared
                        .latest
                        .is_none_or(|latest| totals.invocation > latest.invocation)
                {
                    shared.latest = Some(totals);
                }
            });
        }
    }

    /// The newest fully read and completely submitted invocation.
    pub fn latest(&self) -> Option<VgCullTotals> {
        self.lock().latest
    }

    /// True while a capture awaits its submission or its map has not called back. A capture is
    /// only recorded when no other map is in flight, so a still-mapped buffer never holds more
    /// than that one sample.
    pub fn pending(&self) -> bool {
        self.lock()
            .samples
            .iter()
            .any(|phase| *phase != Phase::Free)
    }

    /// The GPU memory of the staging samples.
    pub fn allocated_bytes(&self) -> u64 {
        POOL_SAMPLES as u64 * SAMPLE_SIZE
    }

    #[cfg(test)]
    fn busy(&self) -> usize {
        self.lock()
            .samples
            .iter()
            .filter(|phase| **phase != Phase::Free)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::culling::CullPipelines;
    use raa_core::testing::{skip, software_gpu};

    /// A 64-byte `CullState` with the given words set (index, value) and the rest 0.
    fn state_bytes(words: &[(usize, u32)]) -> [u8; 64] {
        let mut bytes = [0u8; 64];
        for &(word, value) in words {
            bytes[word * 4..word * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    /// Words: 4 sw_count, 5 hw_count, 6 sw_base, 7 hw_base, 8 triangles, 9 dropped,
    /// 11 frustum_culled, 12 occlusion_rejected.
    #[test]
    fn parse_state_reads_the_counters() {
        let sample = parse_state(&state_bytes(&[
            (4, 5),
            (5, 7),
            (6, 4),
            (7, 5),
            (8, 900),
            (9, 2),
            (11, 30),
            (12, 6),
        ]));
        assert_eq!(
            sample,
            CullSample {
                clusters: 12,
                triangles: 900,
                software: 5,
                dropped: 2,
                frustum_culled: 30,
                occlusion_rejected: 6,
                occlusion_recovered: 3,
            }
        );
    }

    #[test]
    fn capture_keeps_three_samples_and_skips_the_fourth() {
        let test = "capture_keeps_three_samples_and_skips_the_fourth";
        let Some(gpu) = software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let pipelines = CullPipelines::new(&gpu);
        let work = CullWork::new(&gpu, &pipelines, "raa-vg stats test", 16, 16);
        let mut readback = VgStatsReadback::new(&gpu);
        for _ in 0..4 {
            let mut encoder = gpu
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            readback.capture(&mut encoder, &work, None);
            gpu.queue().submit(Some(encoder.finish()));
        }
        assert_eq!(readback.busy(), 3);
        assert!(readback.pending());
        assert_eq!(readback.latest(), None);
    }
}
