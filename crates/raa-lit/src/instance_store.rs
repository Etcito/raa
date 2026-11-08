//! Persistent host mirrors of the S1 instance and S10 extras records, addressed by stable
//! scene instance slot (`InstanceId::index()`).
//!
//! The two mirrors hold [`INSTANCE_GPU_SIZE`]-byte S1 records and
//! [`INSTANCE_EXTRA_GPU_SIZE`]-byte `InstanceExtraGpu` records. Dirty tracking works in
//! pages of 32 records (4096 B) per kind, so a frame that changes nothing uploads nothing
//! and a sparse change uploads only its pages. [`InstanceStore::sync`] fills the mirrors
//! from a scene [`RenderList`] and reuses a scratch record array, so steady-state frames
//! allocate nothing.

use std::collections::HashMap;
use std::ops::Range;

use raa_math::Mat4;
use raa_scene::RenderList;

use crate::extras::{INSTANCE_EXTRA_GPU_SIZE, InstanceExtra};
use crate::prep::INSTANCE_GPU_SIZE;
use crate::previous_instances::PreviousInstances;

/// Records per dirty page.
const PAGE_RECORDS: usize = 32;

/// Bytes of one dirty page (32 records of 128 B).
const PAGE_BYTES: u64 = (PAGE_RECORDS * INSTANCE_GPU_SIZE) as u64;

const _: () = assert!(INSTANCE_EXTRA_GPU_SIZE == INSTANCE_GPU_SIZE);

/// Which mirror of an [`InstanceStore`] a query addresses.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum StoreKind {
    /// The S1 instance records.
    Instances,
    /// The S10 `InstanceExtraGpu` records.
    Extras,
}

/// Writes one 128-byte S1 instance record: model @0..64, normal matrix columns as three
/// vec4 @64..112, material @112, tag @116, palette base @120, zero @124.
pub fn write_instance_record(
    out: &mut [u8; INSTANCE_GPU_SIZE],
    world: &Mat4,
    material: u32,
    tag: u32,
    palette_base: u32,
) {
    world.write_le_bytes(&mut out[0..64]);
    for (chunk, v) in out[64..112]
        .chunks_exact_mut(4)
        .zip(world.normal_matrix().as_flattened())
    {
        chunk.copy_from_slice(&v.to_le_bytes());
    }
    out[112..116].copy_from_slice(&material.to_le_bytes());
    out[116..120].copy_from_slice(&tag.to_le_bytes());
    out[120..124].copy_from_slice(&palette_base.to_le_bytes());
    out[124..128].fill(0);
}

/// Host mirrors of the per-slot instance and extras records with per-page dirty tracking.
#[derive(Clone, Debug)]
pub struct InstanceStore {
    capacity: u32,
    instances: Vec<u8>,
    extras: Vec<u8>,
    dirty_instances: Vec<bool>,
    dirty_extras: Vec<bool>,
    scratch: Vec<u8>,
    last_uploaded: u64,
    previous: PreviousInstances,
    live_slots: Vec<bool>,
    live_scratch: Vec<bool>,
}

impl InstanceStore {
    /// An empty store for `capacity` instance slots; no page is dirty.
    pub fn new(capacity: u32) -> Self {
        let pages = pages_for(capacity);
        let len = pages * PAGE_RECORDS * INSTANCE_GPU_SIZE;
        Self {
            capacity,
            instances: vec![0; len],
            extras: vec![0; len],
            dirty_instances: vec![false; pages],
            dirty_extras: vec![false; pages],
            scratch: Vec::with_capacity(INSTANCE_EXTRA_GPU_SIZE),
            last_uploaded: 0,
            previous: PreviousInstances::new(capacity),
            live_slots: vec![false; pages * PAGE_RECORDS],
            live_scratch: vec![false; pages * PAGE_RECORDS],
        }
    }

    /// Number of addressable slots.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Grows both mirrors to hold at least `slots` slots by doubling the capacity.
    ///
    /// Returns true when it grew: the caller re-creates its GPU buffers, and every page is
    /// marked dirty so the new buffers receive the whole mirrors.
    pub fn ensure(&mut self, slots: u32) -> bool {
        if slots <= self.capacity {
            return false;
        }
        let mut capacity = self.capacity.max(1);
        while capacity < slots {
            capacity = capacity.saturating_mul(2);
        }
        self.capacity = capacity;
        self.previous.ensure(capacity);
        let len = pages_for(capacity) * PAGE_RECORDS * INSTANCE_GPU_SIZE;
        self.instances.resize(len, 0);
        self.extras.resize(len, 0);
        let pages = pages_for(capacity);
        self.dirty_instances.resize(pages, true);
        self.dirty_extras.resize(pages, true);
        self.dirty_instances.fill(true);
        self.dirty_extras.fill(true);
        self.live_slots
            .resize(pages_for(capacity) * PAGE_RECORDS, false);
        self.live_scratch
            .resize(pages_for(capacity) * PAGE_RECORDS, false);
        true
    }

    /// Writes the instance record of `slot`; true when any byte changed.
    pub fn write(&mut self, slot: u32, record: &[u8; INSTANCE_GPU_SIZE]) -> bool {
        let Some(offset) = self.slot_offset(slot, INSTANCE_GPU_SIZE) else {
            return false;
        };
        let end = offset + INSTANCE_GPU_SIZE;
        if self.instances[offset..end] == record[..] {
            return false;
        }
        self.instances[offset..end].copy_from_slice(record);
        self.dirty_instances[slot as usize / PAGE_RECORDS] = true;
        true
    }

    /// Writes the extras record of `slot`; true when any byte changed.
    pub fn write_extra(&mut self, slot: u32, extra: &InstanceExtra) -> bool {
        let Some(offset) = self.slot_offset(slot, INSTANCE_EXTRA_GPU_SIZE) else {
            return false;
        };
        self.scratch.clear();
        extra.write_gpu(&mut self.scratch);
        let end = offset + INSTANCE_EXTRA_GPU_SIZE;
        if self.extras[offset..end] == self.scratch[..] {
            return false;
        }
        self.extras[offset..end].copy_from_slice(&self.scratch);
        self.dirty_extras[slot as usize / PAGE_RECORDS] = true;
        true
    }

    /// The 128-byte instance record of `slot`, or `None` past the capacity.
    pub fn record(&self, slot: u32) -> Option<&[u8]> {
        let offset = self.slot_offset(slot, INSTANCE_GPU_SIZE)?;
        Some(&self.instances[offset..offset + INSTANCE_GPU_SIZE])
    }

    /// The whole instance mirror, page-aligned.
    pub fn instance_bytes(&self) -> &[u8] {
        &self.instances
    }

    /// The whole extras mirror, page-aligned.
    pub fn extra_bytes(&self) -> &[u8] {
        &self.extras
    }

    /// Writes the record and extras of every render item by its stable slot.
    ///
    /// `tags[i]` is the tag of `list.items[i]`, `slots[i]` its stable instance slot, and a
    /// tag without an entry in `extras` gets `InstanceExtra::default()`.
    /// `palette_base(tag)` supplies the skin palette base (0 when not posed). Unchanged
    /// bytes leave their page clean.
    ///
    /// Handoff: `InstanceStore::sync(&mut self, list: &RenderList, tags: &[u32], slots: &[u32], extras: &HashMap<u32, InstanceExtra>, palette_base: &dyn Fn(u32) -> u32) -> ()`
    /// in `crates/raa-lit/src/instance_store.rs` attaches `PreviousInstances::stage` and
    /// `clear`; `ensure` grows both mirrors, and `previous_bytes`/`upload_previous` expose
    /// the record bytes and dirty upload to LitShared. Renderer preparation calls
    /// `previous_submitted` then `finish_previous_frame`. All methods pass the existing
    /// scene instance slot (`InstanceId::index()`), one per submesh, as the registry key.
    pub fn sync(
        &mut self,
        list: &RenderList,
        tags: &[u32],
        slots: &[u32],
        extras: &HashMap<u32, InstanceExtra>,
        palette_base: &dyn Fn(u32) -> u32,
    ) {
        let mut record = [0u8; INSTANCE_GPU_SIZE];
        self.live_scratch.fill(false);
        for (i, item) in list.items.iter().enumerate() {
            let Some(&slot) = slots.get(i) else {
                continue;
            };
            let tag = tags.get(i).copied().unwrap_or(0);
            write_instance_record(
                &mut record,
                &item.world,
                item.material.0,
                tag,
                palette_base(tag),
            );
            self.write(slot, &record);
            if (slot as usize) < self.live_scratch.len() {
                self.live_scratch[slot as usize] = true;
                self.previous.stage(slot, &item.world, true);
            }
            let extra = extras.get(&tag).copied().unwrap_or_default();
            self.write_extra(slot, &extra);
        }
        for slot in 0..self.live_slots.len() {
            if self.live_slots[slot] && !self.live_scratch[slot] {
                self.previous.clear(slot as u32);
            }
        }
        self.live_slots.copy_from_slice(&self.live_scratch);
    }

    /// Ensures the previous-instance mirror follows this store's current capacity.
    #[allow(dead_code)] // Consumed by the binding-8 buffer owner in the next share.
    pub(crate) fn ensure_previous(&mut self) -> bool {
        self.previous.ensure(self.capacity)
    }

    /// Binding-8 mirror bytes for buffer creation and inspection.
    #[allow(dead_code)] // Consumed by the binding-8 buffer owner in the next share.
    pub(crate) fn previous_bytes(&self) -> &[u8] {
        self.previous.bytes()
    }

    /// Uploads dirty binding-8 pages, returning the number of bytes written.
    #[allow(dead_code)] // Consumed by the binding-8 buffer owner in the next share.
    pub(crate) fn upload_previous(&mut self, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> u64 {
        self.previous.upload(queue, buffer)
    }

    /// Marks an existing scene-instance slot as submitted for this scene frame.
    #[allow(dead_code)] // Consumed by the submitted-frame renderer share.
    pub(crate) fn previous_submitted(&mut self, slot: u32) {
        self.previous.submitted(slot);
    }

    /// Commits submitted instance history once after all views have been recorded.
    #[allow(dead_code)] // Consumed by the submitted-frame renderer share.
    pub(crate) fn finish_previous_frame(&mut self) {
        self.previous.finish_frame();
    }

    /// Appends the dirty 4096 B page ranges of `kind`, ascending, into `out`.
    ///
    /// Each dirty page gives one range; [`upload`](Self::upload) coalesces neighbouring
    /// pages into one buffer write.
    pub fn dirty_ranges(&self, kind: StoreKind, out: &mut Vec<Range<u64>>) {
        out.clear();
        for (page, &dirty) in self.dirty_pages(kind).iter().enumerate() {
            if dirty {
                let begin = page as u64 * PAGE_BYTES;
                out.push(begin..begin + PAGE_BYTES);
            }
        }
    }

    /// Uploads only the dirty ranges into the two GPU buffers and clears the flags;
    /// returns the number of bytes written.
    pub fn upload(
        &mut self,
        queue: &wgpu::Queue,
        instances: &wgpu::Buffer,
        extras: &wgpu::Buffer,
    ) -> u64 {
        let mut written = 0;
        {
            let bytes = self.instances.as_slice();
            written += upload_pages(queue, instances, bytes, &mut self.dirty_instances);
        }
        {
            let bytes = self.extras.as_slice();
            written += upload_pages(queue, extras, bytes, &mut self.dirty_extras);
        }
        self.last_uploaded = written;
        written
    }

    /// Clears every dirty page of both kinds.
    pub fn clear_dirty(&mut self) {
        self.dirty_instances.fill(false);
        self.dirty_extras.fill(false);
    }

    /// Bytes written by the last [`upload`](Self::upload).
    pub fn last_uploaded_bytes(&self) -> u64 {
        self.last_uploaded
    }

    fn slot_offset(&self, slot: u32, record_size: usize) -> Option<usize> {
        if slot >= self.capacity {
            return None;
        }
        Some(slot as usize * record_size)
    }

    fn dirty_pages(&self, kind: StoreKind) -> &[bool] {
        match kind {
            StoreKind::Instances => &self.dirty_instances,
            StoreKind::Extras => &self.dirty_extras,
        }
    }
}

/// Number of 32-record pages needed for `capacity` slots.
fn pages_for(capacity: u32) -> usize {
    (capacity as usize).div_ceil(PAGE_RECORDS)
}

/// Writes the contiguous dirty runs of `dirty` from `bytes` and clears them.
fn upload_pages(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    bytes: &[u8],
    dirty: &mut [bool],
) -> u64 {
    let mut written = 0;
    let mut start = 0;
    while start < dirty.len() {
        if !dirty[start] {
            start += 1;
            continue;
        }
        let mut end = start + 1;
        while end < dirty.len() && dirty[end] {
            end += 1;
        }
        let begin = start * PAGE_BYTES as usize;
        let stop = end * PAGE_BYTES as usize;
        queue.write_buffer(buffer, begin as u64, &bytes[begin..stop]);
        written += (stop - begin) as u64;
        dirty[start..end].fill(false);
        start = end;
    }
    written
}
