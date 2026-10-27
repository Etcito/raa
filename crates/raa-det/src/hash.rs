//! FNV-1a 32-bit hashing over `f32` values, byte-identical on every target.
//!
//! Byte order: an `f32` enters as its IEEE-754 bit pattern through
//! `f32::to_bits().to_le_bytes()`, the least significant byte first; a value
//! sequence feeds in input order.
//! Output order: [`Fnv1a32::finish`] returns the bare 32-bit state, with no
//! finalisation and no byte swap; its canonical text is `hex8`
//! (`format!("{:08x}", state)`, 8 lowercase hex digits).

/// The FNV-1a 32-bit offset basis: the hash of no bytes.
pub const FNV1A32_OFFSET: u32 = 0x811c_9dc5;

/// The FNV-1a 32-bit prime.
pub const FNV1A32_PRIME: u32 = 0x0100_0193;

/// FNV-1a 32-bit state over `f32` values hashed as little-endian bit
/// patterns. Bit-equal inputs hash equally: signed zero (`0x00000000` and
/// `0x80000000`) and every NaN payload keep their bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fnv1a32 {
    state: u32,
}

/// The ready-to-use hasher: FNV-1a 32-bit seeded with [`FNV1A32_OFFSET`]
/// (the state before any byte). Seed from it with `let mut hash = DET_HASH;`.
pub const DET_HASH: Fnv1a32 = Fnv1a32 {
    state: FNV1A32_OFFSET,
};

impl Fnv1a32 {
    /// Feeds bytes in order: `state ^= byte; state *= FNV1A32_PRIME`.
    pub fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.state ^= u32::from(byte);
            self.state = self.state.wrapping_mul(FNV1A32_PRIME);
        }
    }

    /// Feeds one `f32` as the 4 little-endian bytes of its bit pattern.
    pub fn f32(&mut self, value: f32) {
        self.update(&value.to_bits().to_le_bytes());
    }

    /// Feeds `f32`s in input order.
    pub fn f32s(&mut self, values: &[f32]) {
        for &value in values {
            self.f32(value);
        }
    }

    /// The state after the last fed byte: no finalisation, no byte swap.
    /// The canonical text is `hex8`, 8 lowercase hex digits.
    pub fn finish(self) -> u32 {
        self.state
    }
}

/// One-shot hash: [`DET_HASH`] over the `f32`s' little-endian bit patterns
/// in input order (the pose-hash one-liner).
pub fn det_hash_f32s(values: &[f32]) -> u32 {
    let mut hash = DET_HASH;
    hash.f32s(values);
    hash.finish()
}
