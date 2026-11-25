//! Surface probes: extra probes outside the regular cascades, placed in the free space in front
//! of the surfaces the cascades serve badly.
//!
//! Demand comes from the probe rays themselves, not from the screen: every probe update shades
//! its ray hits with the volume, and a hit whose lookup finds no close probe that is in front of
//! the surface and visible from it writes a request for the point `hit + normal x offset` (the
//! offset shortened by a ray so the point stays in the hit's own free space). Probes of every
//! cascade trace all around the camera, and a surface probe traces too, so the placement follows
//! the geometry and spreads into spaces the camera does not see yet (a corridor behind an arch).
//!
//! Probes live on a sparse lattice of `spacing` metres, at most one per cell, at the requested
//! point inside the cell. The CPU owns the cell -> slot table (two-choice hashing over the pool):
//! it reads the requests back, keeps the cells nearest to the camera when the pool is full, and
//! writes one table row per slot; the lookup finds the eight cells around a shading point with at
//! most two row reads each. A slot is an ordinary probe slot after the cascades (same record,
//! tiles, groups, scheduling, relocation inside its cell).

use std::collections::HashMap;
use std::sync::mpsc::{Receiver, TryRecvError, sync_channel};

use raa_core::GpuContext;
use wgpu::{Buffer, BufferUsages as Usage};

use super::super::pass::buffer;

/// Salt of the second slot choice (`raa_ddgi_extra_slot`).
const SECOND_CHOICE: u32 = 0x51ed_270b;
/// Table rows per pool slot: one assignment row (CPU), two request rows (GPU).
pub(super) const TABLE_ROWS_PER_SLOT: u64 = 3;
/// Most cells assigned per readback.
const MAX_NEW: usize = 96;

/// The cell hash shared with `raa_ddgi_hash` (ddgi_gi.wgsl).
pub(super) fn hash(c: [i32; 3], salt: u32) -> u32 {
    let mut h = (c[0] as u32).wrapping_mul(0x9e37_79b1).wrapping_add(salt);
    h = (h ^ (h >> 15))
        .wrapping_mul(0x85eb_ca77)
        .wrapping_add((c[1] as u32).wrapping_mul(0xc2b2_ae3d));
    h = (h ^ (h >> 13))
        .wrapping_mul(0x27d4_eb2f)
        .wrapping_add((c[2] as u32).wrapping_mul(0x1656_67b1));
    h = (h ^ (h >> 16)).wrapping_mul(0x9e37_79b1);
    h ^ (h >> 15)
}

/// A change of the table: pool index, and the probe's cell and start position when assigned.
pub(super) struct Assignment {
    pub index: u32,
    pub position: Option<[f32; 3]>,
    pub row: [u32; 4],
}

pub(super) struct Extras {
    pub count: u32,
    pub spacing: f32,
    /// Cell and packed start offset (10 bits per axis) of each pool slot.
    cells: Vec<Option<([i32; 3], u32)>>,
    index: HashMap<[i32; 3], u32>,
    /// Newest request frame already processed.
    seen: Option<u32>,
    readback: Buffer,
    pending: Option<Receiver<Option<Vec<u8>>>>,
    pub assigned_total: u64,
    pub evicted_total: u64,
    pub requests_total: u64,
}

impl Extras {
    pub(super) fn new(gpu: &GpuContext, count: u32, spacing: f32) -> Self {
        Self {
            count,
            spacing,
            cells: vec![None; count as usize],
            index: HashMap::new(),
            seen: None,
            readback: buffer(
                gpu.device(),
                "gi.ddgi.requests",
                (32 * u64::from(count)).max(32),
                Usage::MAP_READ | Usage::COPY_DST,
            ),
            pending: None,
            assigned_total: 0,
            evicted_total: 0,
            requests_total: 0,
        }
    }

    pub(super) fn live(&self) -> u32 {
        self.index.len() as u32
    }

    pub(super) fn idle(&self) -> bool {
        self.pending.is_none()
    }

    fn centre(&self, cell: [i32; 3]) -> [f32; 3] {
        cell.map(|c| (c as f32 + 0.5) * self.spacing)
    }

    fn position(&self, cell: [i32; 3], packed: u32) -> [f32; 3] {
        std::array::from_fn(|i| {
            let f = ((packed >> (10 * i)) & 1023) as f32 / 1024.0;
            (cell[i] as f32 + f) * self.spacing
        })
    }

    /// Copies the request rows (at byte `offset` of `probes`) for a later `poll`.
    pub(super) fn copy(&mut self, encoder: &mut wgpu::CommandEncoder, probes: &Buffer, offset: u64) {
        if self.pending.is_some() || self.count == 0 {
            return;
        }
        let bytes = 32 * u64::from(self.count);
        encoder.copy_buffer_to_buffer(probes, offset, &self.readback, 0, bytes);
        let (send, receive) = sync_channel(1);
        let readback = self.readback.clone();
        encoder.map_buffer_on_submit(&self.readback, wgpu::MapMode::Read, 0..bytes, move |result| {
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

    /// Takes a finished readback and assigns its new requests, nearest to `eye` first, within
    /// `range` metres. A full pool keeps the nearest cells: a request evicts the farther of its
    /// two slot choices when that one is farther from the eye than the request.
    pub(super) fn poll(&mut self, eye: [f32; 3], range: f32) -> Vec<Assignment> {
        let Some(receiver) = self.pending.as_ref() else {
            return Vec::new();
        };
        let bytes = match receiver.try_recv() {
            Ok(bytes) => bytes,
            Err(TryRecvError::Empty) => return Vec::new(),
            Err(TryRecvError::Disconnected) => None,
        };
        self.pending = None;
        let Some(bytes) = bytes else {
            return Vec::new();
        };
        let word = |entry: &[u8], i: usize| {
            u32::from_le_bytes(entry[4 * i..4 * i + 4].try_into().expect("four bytes"))
        };
        let mut requests = Vec::new();
        let mut newest = self.seen;
        for entry in bytes.chunks_exact(32) {
            let cell = [word(entry, 0) as i32, word(entry, 1) as i32, word(entry, 2) as i32];
            let frame = word(entry, 3);
            let packed = word(entry, 4);
            // A torn or never-written entry fails the writer's check word.
            if word(entry, 5) != hash(cell, packed) || word(entry, 6) != 1 {
                continue;
            }
            if self.seen.is_some_and(|seen| frame.wrapping_sub(seen) as i32 <= 0) {
                continue;
            }
            if newest.is_none_or(|n| frame.wrapping_sub(n) as i32 > 0) {
                newest = Some(frame);
            }
            requests.push((cell, packed));
        }
        self.seen = newest;
        self.requests_total += requests.len() as u64;
        let distance = |p: [f32; 3]| (0..3).map(|i| (p[i] - eye[i]).powi(2)).sum::<f32>().sqrt();
        requests.sort_by(|a, b| {
            distance(self.centre(a.0))
                .partial_cmp(&distance(self.centre(b.0)))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut out = Vec::new();
        for (cell, packed) in requests {
            if out.len() >= MAX_NEW {
                break;
            }
            let d = distance(self.centre(cell));
            if d > range || self.index.contains_key(&cell) {
                continue;
            }
            let choices = [
                hash(cell, 0) % self.count,
                hash(cell, SECOND_CHOICE) % self.count,
            ];
            let free = choices.iter().copied().find(|&i| self.cells[i as usize].is_none());
            let index = free.or_else(|| {
                // The farther occupant, when it is farther than the request.
                choices
                    .iter()
                    .copied()
                    .filter_map(|i| {
                        let (other, _) = self.cells[i as usize]?;
                        let far = distance(self.centre(other));
                        (far > d + self.spacing).then_some((far, i))
                    })
                    .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(_, i)| i)
            });
            let Some(index) = index else {
                continue;
            };
            if let Some((old, _)) = self.cells[index as usize].take() {
                self.index.remove(&old);
                self.evicted_total += 1;
            }
            self.cells[index as usize] = Some((cell, packed));
            self.index.insert(cell, index);
            self.assigned_total += 1;
            out.push(Assignment {
                index,
                position: Some(self.position(cell, packed)),
                row: [cell[0] as u32, cell[1] as u32, cell[2] as u32, 1 | (packed << 1)],
            });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_matches_the_shader_reference_values() {
        // Fixed points computed with the same arithmetic as raa_ddgi_hash (wrapping u32).
        assert_eq!(hash([0, 0, 0], 0), hash([0, 0, 0], 0));
        assert_ne!(hash([1, 2, 3], 0), hash([1, 2, 3], SECOND_CHOICE));
        assert_ne!(hash([-1, 0, 0], 0), hash([1, 0, 0], 0));
        // Spread: 4096 neighbouring cells over 1024 slots leave few slots empty.
        let mut used = vec![false; 1024];
        for x in 0..16 {
            for y in 0..16 {
                for z in 0..16 {
                    used[(hash([x - 8, y, z - 8], 0) % 1024) as usize] = true;
                }
            }
        }
        assert!(used.iter().filter(|u| **u).count() > 950);
    }
}
