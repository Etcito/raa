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

## Sixteen-body simulation (Q16)

`selftest(seed: u32, ticks: Tick) -> u32` runs sixteen independent bodies
in index order, with no allocation, wall clock, workers or external inputs.
Initialization advances one wrapping u32 LCG per sample:
`state = state * 1664525 + 1013904223`, starting at the supplied seed.
Each sample is `((state >> 8) as f32 / 16777216) * 2 - 1` (implemented
with the exact reciprocal). For each body, samples are consumed in xyz
order for anchor, position offset (times 0.5), velocity (times 0.25),
rotation xyz and angular velocity xyz. Rotation w is 1 before normalization.
The angular increment is normalized `[omega * (0.5 * dt), 1]`.

Each integer tick uses `dt = TICK_SECONDS_F32`. Per axis, semi-implicit
Euler evaluates `a = (anchor - position) * 8 - velocity * 2`, then
`velocity += a * dt`, then `position += velocity * dt`. Rotation advances
by normalizing `quat_mul(rotation, increment)`, then nlerping from the
old rotation to that result with weight 0.5. Bounded initial state and the
damped fixed step keep the internal helper inputs valid.

One continuous DET_HASH stream accumulates **every completed tick**,
then bodies 0 through 15, then position xyz, velocity xyz, rotation xyzw,
and the 16 column-major components of unit-scale TRS, in that order.
That is 26 f32 values per body, 416 per tick; temporary arithmetic values
and initialization are not outputs. Each value uses the little-endian
bit encoding above. Zero ticks returns the empty-stream offset basis for
every seed. Tests cover repeatability, seed/tick sensitivity and zero ticks.

`DET_CASES` is the ordered `(seed, ticks)` sequence `(0, 0)`, `(1, 60)`,
`(305419896, 600)`. Run `cargo run -p raa-det --example selftest` to emit
one JSON array with exactly three records, computed through `selftest`,
including the zero-step case:

```json
[{"case":"det-selftest","seed":0,"ticks":0,"hash":"811c9dc5"},{"case":"det-selftest","seed":1,"ticks":60,"hash":"935f1896"},{"case":"det-selftest","seed":305419896,"ticks":600,"hash":"77aeeadf"}]
```

Each record has exactly `case`, `seed`, `ticks`, and `hash`, in that order.
Seed and ticks are u32 JSON numbers; hash is eight lowercase hexadecimal
digits, zero-padded without a prefix. Stdout contains only the array and
a trailing newline; diagnostics use stderr. Cases execute serially.
The tests pin all three native hashes and the example's exact serialized
document to detect changes in the simulation or cross-runtime contract.
