# raa-det

The determinism brick for RAA: the contracted simulation tick and the
deterministic `f32` hash that pose hashes and selftests are built on. All
tick and hash operations are total; all helpers are allocation-free, with no
unsafe code, no third-party crates.

## Public items

- `ANIM_TICK_HZ` = 30: ticks per second on the sim path. A tick is one
  AClip v2 sample (`raa.clip` counts AClip frames); the reference
  implementation the sampling ports (`sketchbook/unified.ts`) runs AClip v2
  at a fixed 30 fps.
- `Tick` = `u32`: a tick counter, stored and read back as `u32` across the
  JS boundary too. Ticks wrap like `u32` (advance with `wrapping_advance`,
  never saturate) so distant clients stay identical through a wrap.
- `TICK_SECONDS_F64` (exactly 1/30), `TICK_SECONDS_F32` (the nearest `f32`,
  `0x3d088889`), `tick_seconds_f64`, `tick_seconds_f32` (one `f64`
  division, then one cast; deterministic, but compare ticks, not `f32`
  seconds, once the counter passes roughly 2^21 s).
- `DET_HASH`: the ready `Fnv1a32` hasher, seeded with the FNV-1a 32-bit
  offset basis `FNV1A32_OFFSET` (`0x811c9dc5`), stepped per byte with the
  prime `FNV1A32_PRIME` (`0x01000193`).
- `Fnv1a32`: `update(&[u8])` feeds bytes in order; `f32(value)` feeds the
  value's IEEE-754 bit pattern as 4 little-endian bytes, the least
  significant byte first; `f32s(&[f32])` does that for a sequence;
  `finish()` returns the 32-bit state.
- `det_hash_f32s(&[f32])`: seed, feed, finish in one call (the pose hash).

## Byte order and output order

Every `f32` enters through `f32::to_bits().to_le_bytes()`: the value feeds
by its bit pattern, least significant byte first, a sequence in input
order, and the first fed byte is the first XOR. Host endianness never
matters; wasm32 and native hash identically. Signed zero
(`0x00000000`, `0x80000000`) and every NaN payload keep their bits, so
bit-equal inputs hash equally and bit-different inputs differ.
`finish()` is the bare state: no finalisation, no byte swap. Its
canonical text is `hex8`: `format!("{:08x}", hash)`, 8 lowercase digits.

## Errors

Tick and hash operations cannot fail. Math helpers use the workspace
error convention: `Display` + `Error` with a stable `code()`, codes
prefixed `det.`.

## Tests

`cargo test -p raa-det`: `tests/det.rs` holds the tick boundary tests and
the FNV-1a vectors — the IETF draft test-suite byte vectors plus `f32`
vectors cross-checked with an independent implementation of the byte
sequence.

## Source-ordered math (Q16)

`fmath` (also re-exported at the root) provides `lerp`, `bezier`, generic
array `normalize`, quaternion `sign_align`, `nlerp`, `quat_mul`,
`quat_rotate`, and `trs_compose`. Each returns `Result<_, MathError>`.
Quaternions are xyzw; multiplication applies the right operand first.
TRS returns 16 column-major floats, translation at indices 12–14.
Rotation and TRS require squared quaternion length within 1e-3 of one;
TRS permits zero and negative scale. Interpolation permits extrapolation.

Arithmetic uses source-ordered f32 products, sums, subtraction and division,
with sqrt for normalization; no trig, fused operations or fast-math.
Normalization accumulates squares in component order, then multiplies by
the reciprocal square root. Zero/underflowed lengths give `det.zero_length`;
non-finite inputs/results give `det.not_finite`; invalid rotation lengths
give `det.not_unit`. There is no identity fallback. Sign alignment negates
only on a negative dot product; nlerp aligns before component interpolation.
Tests cover antipodes, normalization, composition, TRS layout and refusals.
