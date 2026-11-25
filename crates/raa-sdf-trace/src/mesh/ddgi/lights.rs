//! Which probes an edit of a local light can change: the diff of the packed light records of two
//! frames, a conservative world bound per light, and the readback of the light mask every probe
//! records while it traces (the lights that reach the surfaces it sees).

use std::sync::mpsc::{Receiver, TryRecvError, sync_channel};

use raa_core::GpuContext;
use wgpu::{Buffer, BufferUsages as Usage};

use super::super::pass::buffer;

/// One packed world-space light record (raa-shader `LIGHT_GPU_SIZE`): position + range, radiance,
/// direction + outer cone, inner cone, kind, width, height.
pub(super) const LIGHT_BYTES: usize = 64;
pub(super) const MAX_LIGHTS: usize = 256;
pub(super) type LightMask = [u32; 8];

/// Where a light can add more than the mask threshold: a sphere, everywhere or nowhere.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct LightBound {
    pub centre: [f32; 3],
    /// Metres; infinite for a directional or unbounded light, negative for a light that is off.
    pub radius: f32,
}

impl LightBound {
    pub(super) fn reaches(&self, p: [f32; 3], margin: f32) -> bool {
        if self.radius < 0.0 {
            return false;
        }
        if !self.radius.is_finite() {
            return true;
        }
        let d2: f32 = (0..3).map(|i| (p[i] - self.centre[i]).powi(2)).sum();
        d2 <= (self.radius + margin).powi(2)
    }
}

fn float(record: &[u8], at: usize) -> f32 {
    f32::from_le_bytes(record[at..at + 4].try_into().expect("four bytes"))
}

fn peak(record: &[u8]) -> f32 {
    float(record, 16)
        .max(float(record, 20))
        .max(float(record, 24))
}

/// The bound of a record: its range, shortened to the distance where a point or spot light's
/// diffuse contribution (at most radiance / (pi d^2)) falls under `threshold`.
pub(super) fn bound(record: &[u8], threshold: f32) -> LightBound {
    let centre = [float(record, 0), float(record, 4), float(record, 8)];
    let peak = peak(record);
    if !(peak > 0.0) {
        return LightBound {
            centre,
            radius: -1.0,
        };
    }
    let kind = u32::from_le_bytes(record[52..56].try_into().expect("four bytes"));
    let range = float(record, 12);
    let mut radius = if kind == 2 || !(range > 0.0) {
        f32::INFINITY
    } else {
        range
    };
    if kind < 2 && threshold > 0.0 {
        radius = radius.min((peak / (std::f32::consts::PI * threshold)).sqrt());
    }
    LightBound { centre, radius }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct LightChange {
    pub index: usize,
    /// Anything but the radiance changed: the probes it reaches are not the recorded ones.
    pub moved: bool,
    /// The radiance grew: it may now pass the mask threshold where it did not.
    pub brighter: bool,
    pub old: LightBound,
    pub new: LightBound,
}

/// The records that differ between two frames (`previous` and `now` hold `MAX_LIGHTS` records).
pub(super) fn diff(previous: &[u8], now: &[u8], threshold: f32) -> Vec<LightChange> {
    let mut changes = Vec::new();
    for index in 0..MAX_LIGHTS
        .min(previous.len() / LIGHT_BYTES)
        .min(now.len() / LIGHT_BYTES)
    {
        let at = index * LIGHT_BYTES;
        let (a, b) = (&previous[at..at + LIGHT_BYTES], &now[at..at + LIGHT_BYTES]);
        if a == b {
            continue;
        }
        let (old, new) = (bound(a, threshold), bound(b, threshold));
        if old.radius < 0.0 && new.radius < 0.0 {
            // Off before and after: no probe holds its light.
            continue;
        }
        changes.push(LightChange {
            index,
            moved: a[..16] != b[..16] || a[28..] != b[28..],
            brighter: peak(b) > peak(a),
            old,
            new,
        });
    }
    changes
}

pub(super) fn mask_has(mask: &LightMask, index: usize) -> bool {
    mask[index >> 5] >> (index & 31) & 1 == 1
}

/// Asynchronous copy of the probe records (4 rows each: the light mask is the last two); one in
/// flight at a time.
pub(super) struct MaskReadback {
    buffer: Buffer,
    bytes: u64,
    pending: Option<Receiver<Option<Vec<u8>>>>,
}

impl MaskReadback {
    pub(super) fn new(gpu: &GpuContext, probes: u32) -> Self {
        let bytes = 64 * u64::from(probes);
        Self {
            buffer: buffer(
                gpu.device(),
                "gi.ddgi.masks",
                bytes,
                Usage::MAP_READ | Usage::COPY_DST,
            ),
            bytes,
            pending: None,
        }
    }

    pub(super) fn idle(&self) -> bool {
        self.pending.is_none()
    }

    /// The light masks of a finished readback, one per slot.
    pub(super) fn poll(&mut self) -> Option<Vec<LightMask>> {
        let receiver = self.pending.as_ref()?;
        let bytes = match receiver.try_recv() {
            Ok(bytes) => bytes,
            Err(TryRecvError::Empty) => return None,
            Err(TryRecvError::Disconnected) => None,
        };
        self.pending = None;
        let bytes = bytes?;
        Some(
            bytes
                .chunks_exact(64)
                .map(|record| {
                    std::array::from_fn(|i| {
                        u32::from_le_bytes(
                            record[32 + 4 * i..36 + 4 * i]
                                .try_into()
                                .expect("four bytes"),
                        )
                    })
                })
                .collect(),
        )
    }

    /// Records the copy of the probe records at byte `offset` of `probes` and its map.
    pub(super) fn copy(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        probes: &Buffer,
        offset: u64,
    ) {
        if self.pending.is_some() {
            return;
        }
        let bytes = self.bytes;
        encoder.copy_buffer_to_buffer(probes, offset, &self.buffer, 0, bytes);
        let (send, receive) = sync_channel(1);
        let readback = self.buffer.clone();
        encoder.map_buffer_on_submit(&self.buffer, wgpu::MapMode::Read, 0..bytes, move |result| {
            let data = result.ok().and_then(|()| {
                let data = readback
                    .slice(0..bytes)
                    .get_mapped_range()
                    .ok()
                    .map(|data| data.to_vec());
                readback.unmap();
                data
            });
            let _ = send.send(data);
        });
        self.pending = Some(receive);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(position: [f32; 3], range: f32, radiance: f32, kind: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        for value in [
            position[0],
            position[1],
            position[2],
            range,
            radiance,
            radiance,
            radiance,
            0.0,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for value in [0.0f32, -1.0, 0.0, 0.5, 0.7] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&kind.to_le_bytes());
        bytes.extend_from_slice(&[0; 8]);
        assert_eq!(bytes.len(), LIGHT_BYTES);
        bytes
    }

    #[test]
    fn bound_is_the_range_shortened_by_the_threshold() {
        let b = bound(&record([1.0, 2.0, 3.0], 6.0, 4.0, 0), 1e-3);
        assert_eq!(b.centre, [1.0, 2.0, 3.0]);
        assert_eq!(b.radius, 6.0);
        // 0.01 / (pi d^2) = 1e-3 at d = 1.78 m
        let dim = bound(&record([0.0; 3], 6.0, 0.01, 0), 1e-3);
        assert!((dim.radius - 1.784).abs() < 1e-2, "{dim:?}");
        assert!(dim.reaches([2.5, 0.0, 0.0], 1.0) && !dim.reaches([3.0, 0.0, 0.0], 1.0));
        assert!(bound(&record([0.0; 3], 6.0, 0.0, 0), 1e-3).radius < 0.0);
        assert!(!bound(&record([0.0; 3], 6.0, 0.0, 0), 1e-3).reaches([0.0; 3], 10.0));
        assert!(bound(&record([0.0; 3], 6.0, 1.0, 2), 1e-3).reaches([1e6, 0.0, 0.0], 0.0));
    }

    #[test]
    fn diff_names_the_changed_lights_and_what_changed() {
        let mut a = vec![0u8; LIGHT_BYTES * MAX_LIGHTS];
        a[LIGHT_BYTES..2 * LIGHT_BYTES].copy_from_slice(&record([0.0; 3], 6.0, 1.0, 0));
        a[3 * LIGHT_BYTES..4 * LIGHT_BYTES].copy_from_slice(&record([5.0, 0.0, 0.0], 6.0, 0.0, 0));
        assert!(diff(&a, &a, 1e-3).is_empty());
        let mut b = a.clone();
        // Light 1 gets dimmer in place, light 3 stays off but moves, light 7 appears.
        b[LIGHT_BYTES..2 * LIGHT_BYTES].copy_from_slice(&record([0.0; 3], 6.0, 0.5, 0));
        b[3 * LIGHT_BYTES..4 * LIGHT_BYTES].copy_from_slice(&record([6.0, 0.0, 0.0], 6.0, 0.0, 0));
        b[7 * LIGHT_BYTES..8 * LIGHT_BYTES].copy_from_slice(&record([0.0, 3.0, 0.0], 4.0, 2.0, 1));
        let changes = diff(&a, &b, 1e-3);
        assert_eq!(changes.len(), 2);
        assert_eq!(
            (changes[0].index, changes[0].moved, changes[0].brighter),
            (1, false, false)
        );
        assert_eq!(
            (changes[1].index, changes[1].moved, changes[1].brighter),
            (7, true, true)
        );
        assert!(changes[1].old.radius < 0.0 && changes[1].new.radius == 4.0);
        let mut mask = [0u32; 8];
        mask[1] = 1 << 3;
        assert!(mask_has(&mask, 35) && !mask_has(&mask, 3));
    }
}
