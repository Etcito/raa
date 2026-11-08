//! S10 previous affine records, indexed by the existing scene instance slot per submesh.
//! This encoding is independent of the live binding-7 instance record.
use raa_math::Mat4;

const RECORD_BYTES: usize = 64;
const PAGE_RECORDS: usize = 32;

/// Writes three affine rows, validity at byte 48, and zero padding at bytes 52..64.
/// Invalid history still contains the supplied current model, never another slot's data.
///
/// Handoff: `write_previous_record(out: &mut [u8; 64], model: &Mat4, valid: bool) -> ()`
/// in `crates/raa-lit/src/previous_instances.rs` is attached by `mod previous_instances`
/// in `lib.rs`. The lifecycle share calls it for existing scene-instance slots, with
/// committed models when valid and current models on creation, reuse or unsupported history.
pub(crate) fn write_previous_record(out: &mut [u8; 64], model: &Mat4, valid: bool) {
    for row in 0..3 {
        for col in 0..4 {
            let offset = row * 16 + col * 4;
            out[offset..offset + 4].copy_from_slice(&model.cols[col][row].to_le_bytes());
        }
    }
    out[48..52].copy_from_slice(&u32::from(valid).to_le_bytes());
    out[52..64].fill(0);
}

/// Page-rounded host mirror of binding-8 records in ascending scene-instance slot order.
///
/// Handoff: `PreviousInstances::new(capacity: u32) -> Self` and
/// `PreviousInstances::bytes(&self) -> &[u8]` in `crates/raa-lit/src/previous_instances.rs`
/// are attached through `lib.rs`; the binding share initializes binding 8 from `bytes()`
/// and preserves binding 7. Capacity comes from the existing instance store, in records.
///
/// Handoff: `PreviousInstances::ensure(&mut self, capacity: u32) -> bool`,
/// `PreviousInstances::stage(&mut self, slot: u32, model: &Mat4, history_supported: bool)`,
/// `PreviousInstances::submitted(&mut self, slot: u32)`, and
/// `PreviousInstances::finish_frame(&mut self)` in `crates/raa-lit/src/previous_instances.rs`
/// are attached by `InstanceStore::sync` and its crate-visible forwarding methods in
/// `crates/raa-lit/src/instance_store.rs`. The registry is the existing scene-instance
/// slot (`InstanceId::index()`), one slot per submesh; draw order is not a key.
#[derive(Clone, Debug)]
pub(crate) struct PreviousInstances {
    records: Vec<u8>,
    staged: Vec<[u8; RECORD_BYTES]>,
    pending: Vec<bool>,
    submitted: Vec<bool>,
    dirty_pages: Vec<bool>,
}

impl PreviousInstances {
    pub(crate) fn new(capacity: u32) -> Self {
        let pages = (capacity as usize).div_ceil(PAGE_RECORDS);
        let slots = pages * PAGE_RECORDS;
        Self {
            records: vec![0; slots * RECORD_BYTES],
            staged: vec![[0; RECORD_BYTES]; slots],
            pending: vec![false; slots],
            submitted: vec![false; slots],
            dirty_pages: vec![false; pages],
        }
    }

    /// Grows by 32-record pages, preserving all existing records.
    pub(crate) fn ensure(&mut self, capacity: u32) -> bool {
        let pages = (capacity as usize).div_ceil(PAGE_RECORDS);
        if pages <= self.dirty_pages.len() {
            return false;
        }
        let slots = pages * PAGE_RECORDS;
        self.records.resize(slots * RECORD_BYTES, 0);
        self.staged.resize(slots, [0; RECORD_BYTES]);
        self.pending.resize(slots, false);
        self.submitted.resize(slots, false);
        self.dirty_pages.resize(pages, true);
        self.dirty_pages.fill(true);
        true
    }

    /// Stages this frame's model without replacing valid committed history.
    pub(crate) fn stage(&mut self, slot: u32, model: &Mat4, history_supported: bool) {
        let i = slot as usize;
        if i >= self.staged.len() {
            return;
        }
        write_previous_record(&mut self.staged[i], model, history_supported);
        self.pending[i] = true;
        let offset = i * RECORD_BYTES;
        let valid =
            u32::from_le_bytes(self.records[offset + 48..offset + 52].try_into().unwrap()) != 0;
        if !history_supported || !valid {
            let mut record = [0; RECORD_BYTES];
            write_previous_record(&mut record, model, false);
            self.write_record(i, &record);
        }
    }

    /// Immediately invalidates a removed or reused slot.
    pub(crate) fn clear(&mut self, slot: u32) {
        let i = slot as usize;
        if i >= self.pending.len() {
            return;
        }
        self.pending[i] = false;
        self.submitted[i] = false;
        self.staged[i] = [0; RECORD_BYTES];
        self.write_record(i, &[0; RECORD_BYTES]);
    }

    /// Marks a staged slot as participating in the submitted frame.
    pub(crate) fn submitted(&mut self, slot: u32) {
        if let Some(value) = self.submitted.get_mut(slot as usize) {
            *value = true;
        }
    }

    /// Commits submitted slots and schedules the next-frame catch-up upload.
    pub(crate) fn finish_frame(&mut self) {
        for i in 0..self.submitted.len() {
            if self.submitted[i] && self.pending[i] {
                let record = self.staged[i];
                self.write_record(i, &record);
            }
            self.submitted[i] = false;
            self.pending[i] = false;
        }
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        &self.records
    }

    /// Uploads adjacent dirty pages as one queue write and returns bytes written.
    #[allow(dead_code)] // Consumed by the binding-8 buffer owner in the next share.
    pub(crate) fn upload(&mut self, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> u64 {
        let page_bytes = PAGE_RECORDS * RECORD_BYTES;
        let mut written = 0;
        let mut page = 0;
        while page < self.dirty_pages.len() {
            if !self.dirty_pages[page] {
                page += 1;
                continue;
            }
            let start = page;
            page += 1;
            while page < self.dirty_pages.len() && self.dirty_pages[page] {
                page += 1;
            }
            let begin = start * page_bytes;
            let end = page * page_bytes;
            queue.write_buffer(buffer, begin as u64, &self.records[begin..end]);
            written += (end - begin) as u64;
            self.dirty_pages[start..page].fill(false);
        }
        written
    }

    fn write_record(&mut self, slot: usize, record: &[u8; RECORD_BYTES]) {
        let begin = slot * RECORD_BYTES;
        if self.records[begin..begin + RECORD_BYTES] == record[..] {
            return;
        }
        self.records[begin..begin + RECORD_BYTES].copy_from_slice(record);
        self.dirty_pages[slot / PAGE_RECORDS] = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_math::{Quat, Vec3};

    fn close(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }

    #[test]
    fn affine_fixture_rows_columns_validity_and_padding() {
        let fixture = include_str!("../tests/fixtures/previous-instances.txt");
        let mut cases = 0;
        for line in fixture
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
        {
            let mut fields = line.split_whitespace();
            let name = fields.next().unwrap();
            let v: Vec<f32> = fields.map(|v| v.parse().unwrap()).collect();
            assert_eq!(v.len(), 25, "{name}");
            let model = Mat4::from_trs(
                Vec3::new(v[0], v[1], v[2]),
                Quat::from_xyzw(v[3], v[4], v[5], v[6]),
                Vec3::new(v[7], v[8], v[9]),
            );
            for valid in [false, true] {
                let mut record = [0xff; 64];
                write_previous_record(&mut record, &model, valid);
                let mut reconstructed = Mat4::IDENTITY;
                for row in 0..3 {
                    for col in 0..4 {
                        let offset = row * 16 + col * 4;
                        let value =
                            f32::from_le_bytes(record[offset..offset + 4].try_into().unwrap());
                        close(value, v[10 + row * 4 + col]);
                        reconstructed.cols[col][row] = value;
                    }
                }
                assert_eq!(reconstructed, model, "{name}");
                let point = reconstructed.transform_point(Vec3::new(1.0, 2.0, 3.0));
                for (actual, expected) in [point.x, point.y, point.z].into_iter().zip(&v[22..]) {
                    close(actual, *expected);
                }
                assert_eq!(&record[48..52], &u32::from(valid).to_le_bytes());
                assert_eq!(&record[52..64], &[0; 12]);
            }
            cases += 1;
        }
        assert_eq!(cases, 4);
    }

    #[test]
    fn invalid_record_overwrites_valid_record_and_has_little_endian_translation() {
        let mut record = [0xff; 64];
        write_previous_record(&mut record, &Mat4::IDENTITY, true);
        let current = Mat4::from_translation(Vec3::new(1.0, -2.0, 0.5));
        write_previous_record(&mut record, &current, false);
        assert_eq!(&record[12..16], &[0, 0, 128, 63]);
        assert_eq!(&record[28..32], &[0, 0, 0, 192]);
        assert_eq!(&record[44..48], &[0, 0, 0, 63]);
        assert_eq!(&record[48..64], &[0; 16]);
    }

    fn valid(store: &PreviousInstances, slot: usize) -> u32 {
        u32::from_le_bytes(
            store.records[slot * 64 + 48..slot * 64 + 52]
                .try_into()
                .unwrap(),
        )
    }

    fn tx(store: &PreviousInstances, slot: usize) -> f32 {
        f32::from_le_bytes(
            store.records[slot * 64 + 12..slot * 64 + 16]
                .try_into()
                .unwrap(),
        )
    }

    #[test]
    fn new_reused_slots_and_submeshes_have_independent_history() {
        let mut store = PreviousInstances::new(3);
        let one = Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0));
        let two = Mat4::from_translation(Vec3::new(2.0, 0.0, 0.0));
        store.stage(0, &one, true);
        store.stage(1, &two, true);
        assert_eq!((valid(&store, 0), valid(&store, 1)), (0, 0));
        store.submitted(0);
        store.submitted(1);
        store.finish_frame();
        assert_eq!((valid(&store, 0), valid(&store, 1)), (1, 1));
        store.clear(0);
        assert_eq!(valid(&store, 0), 0);
        store.stage(0, &two, true);
        assert_eq!(valid(&store, 0), 0);
        assert_eq!(tx(&store, 0), 2.0);
        assert_eq!(tx(&store, 1), 2.0);
    }

    #[test]
    fn reorder_and_partial_submission_commit_only_participants() {
        let mut store = PreviousInstances::new(2);
        let a = Mat4::from_translation(Vec3::new(3.0, 0.0, 0.0));
        let b = Mat4::from_translation(Vec3::new(7.0, 0.0, 0.0));
        store.stage(0, &a, true);
        store.stage(1, &b, true);
        store.submitted(0);
        store.submitted(1);
        store.finish_frame();
        store.dirty_pages.fill(false);

        let moved_a = Mat4::from_translation(Vec3::new(4.0, 0.0, 0.0));
        let moved_b = Mat4::from_translation(Vec3::new(8.0, 0.0, 0.0));
        // A draw-list reorder changes traversal order only; slot identity remains stable.
        store.stage(1, &moved_b, true);
        store.stage(0, &moved_a, true);
        assert_eq!((tx(&store, 0), tx(&store, 1)), (3.0, 7.0));
        store.submitted(0);
        store.finish_frame();
        assert_eq!((tx(&store, 0), tx(&store, 1)), (4.0, 7.0));
        assert_eq!((valid(&store, 0), valid(&store, 1)), (1, 1));
        assert!(store.dirty_pages[0]);
        store.dirty_pages.fill(false);
        store.stage(0, &moved_a, true);
        store.submitted(0);
        store.finish_frame();
        assert!(store.dirty_pages.iter().all(|dirty| !dirty));
    }

    #[test]
    fn unsupported_history_growth_and_motion_catch_up() {
        let mut store = PreviousInstances::new(1);
        let initial = Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0));
        store.stage(0, &initial, false);
        store.submitted(0);
        store.finish_frame();
        assert_eq!(valid(&store, 0), 0);
        assert!(store.ensure(33));
        assert_eq!(store.bytes().len(), 64 * 64);
        assert_eq!(tx(&store, 0), 1.0);

        let moved = Mat4::from_translation(Vec3::new(9.0, 0.0, 0.0));
        store.stage(0, &moved, true);
        assert_eq!(valid(&store, 0), 0);
        store.submitted(0);
        store.finish_frame();
        assert_eq!(valid(&store, 0), 1);
        assert_eq!(tx(&store, 0), 9.0);
        store.dirty_pages.fill(false);
        store.stage(0, &moved, true);
        store.submitted(0);
        store.finish_frame();
        assert!(store.dirty_pages.iter().all(|dirty| !dirty));
    }

    #[test]
    fn inspection_includes_zeroed_page_padding_in_slot_order() {
        for (capacity, pages) in [(0, 0), (1, 1), (32, 1), (33, 2)] {
            let mut store = PreviousInstances::new(capacity);
            assert_eq!(store.bytes().len(), pages * 32 * 64);
            assert!(store.bytes().iter().all(|&b| b == 0));
            for slot in 0..capacity as usize {
                let model = Mat4::from_translation(Vec3::new(slot as f32, 0.0, 0.0));
                write_previous_record(
                    (&mut store.records[slot * 64..(slot + 1) * 64])
                        .try_into()
                        .unwrap(),
                    &model,
                    false,
                );
            }
            for (slot, record) in store.bytes().chunks_exact(64).enumerate() {
                if slot < capacity as usize {
                    assert_eq!(&record[12..16], &(slot as f32).to_le_bytes());
                } else {
                    assert_eq!(record, &[0; 64]);
                }
                assert_eq!(&record[48..64], &[0; 16]);
            }
        }
    }
}
