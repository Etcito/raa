//! Contract tests for raa-det: tick boundaries and FNV-1a 32-bit vectors.
//!
//! The byte vectors are the independently known FNV-1a 32-bit test vectors
//! (the IETF draft FNV test suite appendix). The f32 vectors were computed
//! with an independent implementation (Python) hashing the same byte
//! sequence (the little-endian bit patterns), cross-checked against those
//! byte vectors.

use raa_det::{
    ANIM_TICK_HZ, DET_HASH, FNV1A32_OFFSET, FNV1A32_PRIME, TICK_SECONDS_F32, TICK_SECONDS_F64,
    Tick, det_hash_f32s, tick_seconds_f32, tick_seconds_f64, wrapping_advance,
};

const MAX_TICK: Tick = u32::MAX;

fn hash_bytes(bytes: &[u8]) -> u32 {
    let mut hash = DET_HASH;
    hash.update(bytes);
    hash.finish()
}

/// The canonical byte vectors, plus the documented constants.
#[test]
fn fnv1a32_byte_vectors() {
    assert_eq!(FNV1A32_OFFSET, 0x811c_9dc5);
    assert_eq!(FNV1A32_PRIME, 0x0100_0193);
    assert_eq!(hash_bytes(b""), 0x811c_9dc5);
    assert_eq!(hash_bytes(b"a"), 0xe40c_292c);
    assert_eq!(hash_bytes(b"foobar"), 0xbf9c_f968);
    assert_eq!(DET_HASH.finish(), FNV1A32_OFFSET);
}

/// Feeding is one byte stream: chunking never changes the state.
#[test]
fn fnv1a32_chunking_is_the_stream() {
    let mut chunked = DET_HASH;
    chunked.update(b"foo");
    chunked.update(b"ba");
    chunked.update(b"r");
    assert_eq!(chunked.finish(), hash_bytes(b"foobar"));
}

/// An `f32` hashes its bit pattern, whatever the value path (zeros, infinities,
/// payloads); the bit-pattern entry point and `f32` never disagree.
#[test]
fn f32_hashes_its_le_bits() {
    let values = [
        -0.0_f32,
        0.25,
        123.75,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc0_0000),
    ];
    for &value in &values {
        let mut bits = DET_HASH;
        bits.update(&value.to_bits().to_le_bytes());
        let mut direct = DET_HASH;
        direct.f32(value);
        assert_eq!(direct.finish(), bits.finish(), "{value:?}");
    }
}

/// Known f32 vectors: signed-zero bit patterns differ, input order and payloads
/// are visible in the state, and the constants pin the exact result.
#[test]
fn f32_le_vectors() {
    assert_eq!(det_hash_f32s(&[1.0]), 0x1b58_7698);
    assert_eq!(det_hash_f32s(&[1.0, 2.0]), 0xb82f_acd8);
    assert_eq!(det_hash_f32s(&[2.0, 1.0]), 0x37f0_7f08);
    assert_eq!(det_hash_f32s(&[0.0]), 0x4b95_f515);
    assert_eq!(det_hash_f32s(&[-0.0]), 0xcb95_2b95);
    assert_ne!(det_hash_f32s(&[0.0]), det_hash_f32s(&[-0.0]));
    assert_eq!(det_hash_f32s(&[f32::from_bits(0x7fc0_0000)]), 0x5ab9_7798);
    assert_eq!(det_hash_f32s(&[f32::from_bits(0xffc0_0000)]), 0xdaba_4118);
    assert_eq!(det_hash_f32s(&[1e-9, 65504.0]), 0x5aa8_1439);
}

/// The tick constants and their exact bit patterns.
#[test]
fn tick_constants() {
    assert_eq!(ANIM_TICK_HZ, 30);
    assert_eq!(TICK_SECONDS_F64.to_bits(), 0x3fa1_1111_1111_1111);
    assert_eq!(TICK_SECONDS_F32.to_bits(), 0x3d08_8889);
}

/// Boundary instants at the count edges of the u32 tick range.
#[test]
fn tick_seconds_boundaries() {
    assert_eq!(tick_seconds_f64(0), 0.0);
    assert_eq!(tick_seconds_f64(1), TICK_SECONDS_F64);
    assert_eq!(tick_seconds_f64(3), 0.1);
    assert_eq!(tick_seconds_f64(MAX_TICK), 143165576.5);
}

/// The f32 seconds contract: one f64 division, then one cast. The rounding
/// at the first tick that exceeds the f32 mantissa is pinned, so native and
/// wasm32 must agree bit for bit.
#[test]
fn tick_seconds_f32_contract() {
    assert_eq!(tick_seconds_f32(0).to_bits(), 0);
    assert_eq!(tick_seconds_f32(1), TICK_SECONDS_F32);
    assert_eq!(tick_seconds_f32(3).to_bits(), 0x3dcc_cccd);
    assert_eq!(tick_seconds_f32(1 << 24).to_bits(), 0x4908_8889);
}

/// Ticks wrap like u32, never saturate: the counter restarts at 0.
#[test]
fn ticks_wrap() {
    assert_eq!(wrapping_advance(0, 1), 1);
    assert_eq!(wrapping_advance(MAX_TICK, 1), 0);
    assert_eq!(wrapping_advance(2, MAX_TICK), 1);
}
