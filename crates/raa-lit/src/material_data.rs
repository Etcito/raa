//! Material data words: one read-only storage buffer of `u32` words, bound in group 0
//! (binding 6), that material functions read through the prelude's
//! `raa_material_word(i)` / `raa_material_f32(i)`. A block of words is added from the host
//! and its base offset handed to a surface material ([`SurfaceMaterial::data`], written
//! to `SurfaceMaterialGpu.header.w` and read in WGSL by `raa_material_data_base(record)`),
//! so one material function, hence one pipeline, can shade many materials whose inputs are
//! data rather than WGSL text.
//!
//! Words `0..MATERIAL_DATA_RESERVED_WORDS` are reserved and stay zero: a material without
//! data (base 0) reads zeros there. Blocks are placed first-fit in the freed ranges, else
//! at the end; a removed block is zeroed. The buffer starts at
//! [`MATERIAL_DATA_INITIAL_WORDS`] and doubles (at least to what is needed) when a block
//! does not fit; growing creates a new buffer, so group 0 is bound again. The host keeps a
//! copy of every word, which a grown buffer is filled from.
//!
//! [`SurfaceMaterial::data`]: crate::SurfaceMaterial::data

use std::collections::BTreeMap;

use raa_core::{GpuContext, RaaError};

/// Words reserved (zero) at offset 0: the data base of a material without data.
pub const MATERIAL_DATA_RESERVED_WORDS: u32 = 4;
/// Capacity of the buffer before the first growth, in words (16 KiB).
pub const MATERIAL_DATA_INITIAL_WORDS: u32 = 4096;

fn data_error(code: &'static str, message: String) -> RaaError {
    RaaError::FrameGraph { code, message }
}

/// Placement of the blocks of words: live blocks and free ranges by base offset, and the
/// end of the highest block (`top`). Pure, no GPU.
#[derive(Clone, Debug)]
pub struct DataAllocator {
    blocks: BTreeMap<u32, u32>,
    free: BTreeMap<u32, u32>,
    top: u32,
}

impl Default for DataAllocator {
    fn default() -> Self {
        DataAllocator::new()
    }
}

impl DataAllocator {
    /// No block; the first one goes at [`MATERIAL_DATA_RESERVED_WORDS`].
    pub fn new() -> Self {
        DataAllocator {
            blocks: BTreeMap::new(),
            free: BTreeMap::new(),
            top: MATERIAL_DATA_RESERVED_WORDS,
        }
    }

    /// Places a block of `len` (> 0) words and returns its base: the lowest free range it
    /// fits in, else the end. `None` when the end would pass `u32::MAX`.
    pub fn alloc(&mut self, len: u32) -> Option<u32> {
        debug_assert!(len > 0);
        let fit = self
            .free
            .iter()
            .find(|&(_, &n)| n >= len)
            .map(|(&at, &n)| (at, n));
        let base = match fit {
            Some((at, n)) => {
                self.free.remove(&at);
                if n > len {
                    self.free.insert(at + len, n - len);
                }
                at
            }
            None => {
                let at = self.top;
                self.top = at.checked_add(len)?;
                at
            }
        };
        self.blocks.insert(base, len);
        Some(base)
    }

    /// Frees the block at `base` and returns its length; `None` for an unknown base. The
    /// range merges with its free neighbours, and the end comes down when it was last.
    pub fn free(&mut self, base: u32) -> Option<u32> {
        let len = self.blocks.remove(&base)?;
        let (mut at, mut n) = (base, len);
        if let Some((&prev, &pn)) = self.free.range(..at).next_back()
            && prev + pn == at
        {
            self.free.remove(&prev);
            at = prev;
            n += pn;
        }
        if let Some(&next) = self.free.get(&(at + n)) {
            self.free.remove(&(at + n));
            n += next;
        }
        if at + n == self.top {
            self.top = at;
        } else {
            self.free.insert(at, n);
        }
        Some(len)
    }

    /// Length of the live block at `base`.
    pub fn len_of(&self, base: u32) -> Option<u32> {
        self.blocks.get(&base).copied()
    }

    /// End of the highest live block (at least [`MATERIAL_DATA_RESERVED_WORDS`]).
    pub fn top(&self) -> u32 {
        self.top
    }

    /// Number of live blocks.
    pub fn blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Words held by live blocks.
    pub fn used_words(&self) -> u64 {
        self.blocks.values().map(|&n| u64::from(n)).sum()
    }
}

/// What [`LitRenderer::material_data_stats`](crate::LitRenderer::material_data_stats) reports.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MaterialDataStats {
    /// Live blocks.
    pub blocks: usize,
    /// Words held by live blocks.
    pub used_words: u64,
    /// End of the highest live block, in words.
    pub top: u32,
    /// Capacity of the GPU buffer, in words.
    pub capacity: u32,
}

/// The material data buffer of group 0 (binding 6) and the host copy of its words.
pub(crate) struct MaterialData {
    alloc: DataAllocator,
    words: Vec<u32>,
    buffer: wgpu::Buffer,
    capacity: u32,
}

fn create_buffer(gpu: &GpuContext, words: u32) -> wgpu::Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("raa.lit.material_data"),
        size: u64::from(words) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

impl MaterialData {
    pub(crate) fn new(gpu: &GpuContext) -> Self {
        MaterialData {
            alloc: DataAllocator::new(),
            words: vec![0; MATERIAL_DATA_RESERVED_WORDS as usize],
            buffer: create_buffer(gpu, MATERIAL_DATA_INITIAL_WORDS),
            capacity: MATERIAL_DATA_INITIAL_WORDS,
        }
    }

    /// The group-0 entry 6.
    pub(crate) fn bind_entry(&self) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry {
            binding: 6,
            resource: self.buffer.as_entire_binding(),
        }
    }

    pub(crate) fn allocator(&self) -> &DataAllocator {
        &self.alloc
    }

    /// Capacity of the GPU buffer, in words.
    pub(crate) fn capacity(&self) -> u32 {
        self.capacity
    }

    /// The live block at `base` as the host holds it.
    pub(crate) fn block(&self, base: u32) -> Option<&[u32]> {
        let n = self.alloc.len_of(base)?;
        self.words.get(base as usize..(base + n) as usize)
    }

    /// Stores `data` as a new block and returns its base; the boolean is true when the GPU
    /// buffer was created again (group 0 must be bound again). Refusals (nothing changes):
    /// `material.data_invalid` (no word), `material.data_full` (the buffer would pass the
    /// device's storage binding size).
    pub(crate) fn add(&mut self, gpu: &GpuContext, data: &[u32]) -> Result<(u32, bool), RaaError> {
        let len = u32::try_from(data.len())
            .ok()
            .filter(|&n| n > 0)
            .ok_or_else(|| {
                data_error(
                    "material.data_invalid",
                    format!(
                        "a material data block holds 1 to {} words, got {}",
                        u32::MAX,
                        data.len()
                    ),
                )
            })?;
        let limits = gpu.limits();
        let max_bytes =
            limits.max_storage_buffer_binding_size.min(limits.max_buffer_size);
        let max_words = u32::try_from(max_bytes / 4).unwrap_or(u32::MAX);
        let full = |need: u64| {
            data_error(
                "material.data_full",
                format!(
                    "material data needs {need} words; the device binds at most {max_words} (max_storage_buffer_binding_size {max_bytes} B)"
                ),
            )
        };
        let base = self
            .alloc
            .alloc(len)
            .ok_or_else(|| full(u64::from(self.alloc.top()) + u64::from(len)))?;
        let end = base + len;
        let mut grown = false;
        if end > self.capacity {
            if end > max_words {
                self.alloc.free(base);
                return Err(full(u64::from(end)));
            }
            let capacity = self
                .capacity
                .saturating_mul(2)
                .max(end.next_power_of_two())
                .min(max_words)
                .max(end);
            self.buffer = create_buffer(gpu, capacity);
            self.capacity = capacity;
            grown = true;
        }
        if self.words.len() < end as usize {
            self.words.resize(end as usize, 0);
        }
        self.words[base as usize..end as usize].copy_from_slice(data);
        if grown {
            gpu.queue()
                .write_buffer(&self.buffer, 0, &bytes(&self.words));
        } else {
            gpu.queue()
                .write_buffer(&self.buffer, u64::from(base) * 4, &bytes(data));
        }
        Ok((base, grown))
    }

    /// Overwrites the live block at `base` with `data` of the same length. Refusal
    /// `material.data_invalid` (unknown base, other length); nothing changes.
    pub(crate) fn set(
        &mut self,
        gpu: &GpuContext,
        base: u32,
        data: &[u32],
    ) -> Result<(), RaaError> {
        let len = self.alloc.len_of(base).ok_or_else(|| {
            data_error(
                "material.data_invalid",
                format!("no material data block at word {base}"),
            )
        })?;
        if data.len() != len as usize {
            return Err(data_error(
                "material.data_invalid",
                format!(
                    "the material data block at word {base} holds {len} words, got {}",
                    data.len()
                ),
            ));
        }
        self.words[base as usize..(base + len) as usize].copy_from_slice(data);
        gpu.queue()
            .write_buffer(&self.buffer, u64::from(base) * 4, &bytes(data));
        Ok(())
    }

    /// Frees and zeroes the block at `base`; an unknown base does nothing. Returns whether
    /// it was live.
    pub(crate) fn remove(&mut self, gpu: &GpuContext, base: u32) -> bool {
        let Some(len) = self.alloc.free(base) else {
            return false;
        };
        let range = base as usize..(base + len) as usize;
        self.words[range].fill(0);
        gpu.queue().write_buffer(
            &self.buffer,
            u64::from(base) * 4,
            &vec![0; len as usize * 4],
        );
        self.words
            .truncate(self.alloc.top().max(MATERIAL_DATA_RESERVED_WORDS) as usize);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_start_after_the_reserved_words_and_reuse_freed_ranges() {
        let mut a = DataAllocator::new();
        let b0 = a.alloc(10).unwrap();
        let b1 = a.alloc(5).unwrap();
        let b2 = a.alloc(7).unwrap();
        assert_eq!([b0, b1, b2], [4, 14, 19]);
        assert_eq!(a.top(), 26);
        assert_eq!(a.free(b1), Some(5));
        assert_eq!(a.free(b1), None);
        // First fit: a 3-word block goes in the freed range, a 6-word one at the end.
        assert_eq!(a.alloc(3), Some(14));
        assert_eq!(a.alloc(6), Some(26));
        assert_eq!(a.alloc(2), Some(17));
        assert_eq!(a.blocks(), 5);
        assert_eq!(a.used_words(), 10 + 3 + 2 + 7 + 6);
    }

    #[test]
    fn freed_neighbours_merge_and_the_end_comes_down() {
        let mut a = DataAllocator::new();
        let b: Vec<u32> = (0..4).map(|_| a.alloc(4).unwrap()).collect();
        a.free(b[1]);
        a.free(b[2]);
        // b1 and b2 merged: an 8-word block fits there.
        assert_eq!(a.alloc(8), Some(b[1]));
        a.free(b[1]);
        a.free(b[3]);
        // b1..b3 were the last blocks: the end is back after b0.
        assert_eq!(a.top(), b[0] + 4);
        a.free(b[0]);
        assert_eq!(a.top(), MATERIAL_DATA_RESERVED_WORDS);
        assert_eq!(a.blocks(), 0);
        assert_eq!(a.len_of(b[0]), None);
    }
}
