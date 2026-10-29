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

#[test]
fn q16_sim_repeatability_and_sensitivity() {
    for seed in [0, 1, 0x1234_5678, u32::MAX] {
        let hash = raa_det::selftest(seed, 60);
        assert_eq!(hash, raa_det::selftest(seed, 60));
        assert_ne!(hash, raa_det::selftest(seed.wrapping_add(1), 60));
        assert_ne!(hash, raa_det::selftest(seed, 59));
        assert_ne!(hash, raa_det::selftest(seed, 61));
    }
}

#[test]
fn q16_sim_zero_ticks() {
    for seed in [0, 1, 0x1234_5678, u32::MAX] {
        assert_eq!(raa_det::selftest(seed, 0), FNV1A32_OFFSET);
        assert_ne!(raa_det::selftest(seed, 1), FNV1A32_OFFSET);
    }
}

#[allow(dead_code)]
#[path = "../examples/selftest.rs"]
mod native_example;

#[test]
fn q16_canonical_cases_and_serialization() {
    assert_eq!(raa_det::DET_CASES, [(0, 0), (1, 60), (305419896, 600)]);
    let expected = [0x811c_9dc5, 0x935f_1896, 0x77ae_eadf];
    for ((seed, ticks), hash) in raa_det::DET_CASES.into_iter().zip(expected) {
        assert_eq!(raa_det::selftest(seed, ticks), hash);
        assert_eq!(raa_det::selftest(seed, ticks), hash);
    }
    let mut output = Vec::new();
    native_example::write_json(&mut output).unwrap();
    assert_eq!(
        String::from_utf8(output).unwrap(),
        concat!(
            "[{\"case\":\"det-selftest\",\"seed\":0,\"ticks\":0,\"hash\":\"811c9dc5\"},",
            "{\"case\":\"det-selftest\",\"seed\":1,\"ticks\":60,\"hash\":\"935f1896\"},",
            "{\"case\":\"det-selftest\",\"seed\":305419896,\"ticks\":600,\"hash\":\"77aeeadf\"}]\n"
        )
    );
}

use raa_det::{
    MathError, bezier, lerp, nlerp, normalize, quat_mul, quat_rotate, sign_align, trs_compose,
};

#[test]
fn q16_scalar_order_and_normalization() {
    // A fused evaluation would retain the small negative product residual.
    assert_eq!(
        lerp(-1.0, f32::EPSILON, 1.0 - f32::EPSILON)
            .unwrap()
            .to_bits(),
        0
    );
    assert_eq!(bezier(0.0, 2.0, 4.0, 6.0, 0.25).unwrap(), 1.5);
    assert_eq!(
        normalize([0.0, 3.0, 0.0, 4.0]).unwrap(),
        [0.0, 0.6, 0.0, 0.8]
    );
    assert_eq!(normalize([0.0; 4]), Err(MathError::ZeroLength));
    assert_eq!(normalize([f32::from_bits(1)]), Err(MathError::ZeroLength));
    assert_eq!(normalize([f32::MAX]), Err(MathError::NotFinite));
    assert_eq!(lerp(0.0, 1.0, f32::NAN), Err(MathError::NotFinite));
    assert_eq!(MathError::ZeroLength.code(), "det.zero_length");
}

#[test]
fn q16_antipodes_and_sign_tie() {
    let q = [0.0, 0.0, 0.0, 1.0];
    for u in [0.0, 0.25, 0.5, 1.0] {
        assert_eq!(nlerp(q, q.map(|x| -x), u).unwrap(), q);
    }
    let b = [1.0, -0.0, 0.0, 0.0];
    assert_eq!(
        sign_align(q, b).unwrap().map(f32::to_bits),
        b.map(f32::to_bits)
    );
    let mid = nlerp(q, b, 0.5).unwrap();
    assert!((mid[0] - core::f32::consts::FRAC_1_SQRT_2).abs() < 1e-7);
    assert_eq!(nlerp(q, [0.0; 4], 0.0), Err(MathError::ZeroLength));
}

#[test]
fn q16_composition_and_column_major_trs() {
    let a = [0.5; 4]; // Cyclic permutation: X -> Y -> Z -> X.
    let b = [1.0, 0.0, 0.0, 0.0];
    let v = [2.0, 3.0, 5.0];
    assert_eq!(quat_mul(a, b).unwrap(), [0.5, 0.5, -0.5, -0.5]);
    assert_eq!(
        quat_rotate(quat_mul(a, b).unwrap(), v).unwrap(),
        quat_rotate(a, quat_rotate(b, v).unwrap()).unwrap()
    );
    assert_ne!(quat_mul(a, b).unwrap(), quat_mul(b, a).unwrap());
    assert_eq!(
        trs_compose([7.0, 8.0, 9.0], a, [2.0, 3.0, 4.0]).unwrap(),
        [
            0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0, 0.0, 7.0, 8.0, 9.0, 1.0
        ]
    );
    assert_eq!(quat_rotate([0.0; 4], v), Err(MathError::NotUnit));
    assert_eq!(
        trs_compose([f32::INFINITY; 3], a, v),
        Err(MathError::NotFinite)
    );
    assert_eq!(quat_mul([f32::MAX; 4], [2.0; 4]), Err(MathError::NotFinite));
}
