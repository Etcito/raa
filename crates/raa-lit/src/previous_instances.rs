//! S10 previous affine records, indexed by the existing scene instance slot per submesh.
//! This encoding is independent of the live binding-7 instance record.
#![allow(dead_code)] // Encoding lands before the lifecycle and binding consumers.

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
/// Handoff: the lifecycle share extends this type in this file with
/// `ensure(&mut self, capacity: u32) -> bool`,
/// `stage(&mut self, slot: u32, model: &Mat4, history_supported: bool) -> ()`,
/// `clear(&mut self, slot: u32) -> ()`, `submitted(&mut self, slot: u32) -> ()`,
/// `finish_frame(&mut self) -> ()` and
/// `upload(&mut self, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> u64`.
/// InstanceStore passes its existing live slots to ensure/stage/clear. LitShared recreates
/// the buffer only on ensure growth and uses bytes/upload. The renderer share acknowledges
/// submitted slots using submitted, then finish_frame once after all views; committed
/// records stay frozen across views. These lifecycle methods belong to the next share.
#[derive(Clone, Debug)]
pub(crate) struct PreviousInstances {
    records: Vec<u8>,
}

impl PreviousInstances {
    pub(crate) fn new(capacity: u32) -> Self {
        let pages = (capacity as usize).div_ceil(PAGE_RECORDS);
        Self {
            records: vec![0; pages * PAGE_RECORDS * RECORD_BYTES],
        }
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        &self.records
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
