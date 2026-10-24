# raa-particles

GPU particles for the RAA lit renderer: emitters of four effect families, a fixed-step
compute simulation and two draw stages. Depends on raa-core, raa-math, raa-pipeline,
raa-lit and raa-shader; raa-brdf and raa-transmission are dev-dependencies for the
variants test.

## Public items

- `ParticleKind`: `Splash`, `Snow`, `Confetti`, `Sparkles`.
- `EmitterDesc`: kind, position, direction, spread, rate, lifetime, speed, size, gravity,
  drag, colours, emissive strength, spawn volume, ground plane, particle cap and seed.
  - `EmitterDesc::for_kind(kind)`: per-kind defaults (gravity 9.81 except snow 1.0 and
    sparkles 0; splash default rate 0, snow/confetti/sparkles emit continuously).
  - `EmitterDesc::validate()`: refuses with `particles.invalid`, naming the offending
    field. Every number is finite; direction non-zero; spread in `[0, pi]`; rate `>= 0`;
    lifetime `0 < min <= max`; speed `0 <= min <= max`; size `0 < min <= max`; drag
    `>= 0`; 1 to 8 colours with components `>= 0`; emissive `>= 0`; volume half extents
    `>= 0`; `1 <= max_particles <= 65536`.
- `pcg_hash(v)`, `spawn_sample(&desc, serial)` and `SpawnSample`: the seeded CPU twin of
  the simulation. A sample is a pure function of `(seed, serial)`; the spawn module
  documents the exact draw order and the WGSL mirrors it in `f32` with the same operation
  order.
- `PARTICLE_STRIDE` (48): the bytes of one particle state record.
- `ParticleFeature::install(gpu, pipeline, lit) -> Result<ParticleHandle, RaaError>`:
  creates the shared emitter state and adds the three S12 entries
  `particles.sim` (pass, order 50), `particles.opaque` (forward contributor, order 600,
  through `lit.add_contributor`) and `particles.blend` (pass, order 85). Refusals are
  atomic (nothing installed): `pipeline.busy` when the pipeline is borrowed,
  `pipeline.duplicate_pass` when any of the three entries already exists, and the draw
  module refusal with its code (`shader.slot_missing`).
- `ParticleHandle` (clones share one state):
  - `add(desc) -> Result<u32, RaaError>`: validates and adds under the lowest free id
    (dense ids, holes reused).
  - `set(id, desc)`, `remove(id)`, `burst(id, count)`: delegate to the shared state.
  - `count() -> u32`: particle slots in use, the sum over emitters of
    `min(max_particles, spawned so far)`, without a GPU readback.
  - Refusals: `particles.invalid` (validation first, naming the field),
    `particles.capacity` (more than 64 emitters or more than 262144 particles in total),
    `particles.unknown_emitter`, and `pipeline.busy` on a re-entrant call.
- Hidden builders `sim_module_wgsl()`, `opaque_module_wgsl(&composer)` and
  `blend_module_wgsl(&composer)` (the `tests/variants.rs` Q11 entry points). The draw
  builders are `PRELUDE + LIB + the slot winners snapshotted at install + the pass body`.

## Particle state record (48 B)

Little-endian `to_le_bytes`, no padding:

| offset | field |
|---|---|
| 0  | `pos.xyz` (f32 x3), `age` (f32) |
| 16 | `vel.xyz` (f32 x3), `life` (f32) |
| 32 | `size` (f32), `rotation` (f32) |
| 40 | `color_index` (u32), `kind` (u32) |

A dead slot has `age >= life`. Size is in metres; rotation is a roll angle in radians.

## Pass behaviour

- `particles.sim` (compute, order 50): spawns by atomic counter from the accumulated rate
  and bursts with the seeded PCG hash, then integrates gravity, drag and the ground bounce
  (restitution 0.3). Confetti flutters (spin plus lift from its orientation), snow drifts
  with curl noise, splash is ballistic and sparkles drift.
- `particles.opaque` (contributor, order 600): confetti and snow as camera-facing or
  oriented quads with alpha-to-coverage, shaded through the installed Shadow, Ao, Ambient
  and Direct slots (the winners are snapshotted at install).
- `particles.blend` (pass, order 85): splash with env fresnel, premultiplied, and sparkles
  additive emissive with a twinkle. It copies `hdr.src` to `hdr.dst`, applies a soft depth
  fade against `depth`, then swaps; without such emitters it records nothing.

## Fixed-step catch-up rule

The simulation clock advances in fixed 1/60 s steps towards the pipeline time, at most 240
steps per frame. The first advance starts at `max(0, t - 4 s)`, so the table is pre-warmed
and a fixed-time frame (which has `dt` = 0) shows the state at `t`. A frame further behind
than 240 steps jumps its clock forward and simulates only the last 240 steps; disabling
the pass freezes the clock.

## Testing

```
cargo test -p raa-particles
```

- `tests/desc.rs` (Q12, CPU only): every per-kind default validates and every rule is
  refused with `particles.invalid` and a message naming the field.
- `tests/spawn.rs` (CPU only): the CPU twin is deterministic per `(seed, serial)` and
  matches the reference PCG values.
- `tests/variants.rs` (Q11, CPU only): `sim_module_wgsl`, `opaque_module_wgsl` and
  `blend_module_wgsl` each validate with the slot set of an engine with the M4..M5
  features (raa-lit defaults, `raa_brdf::providers()` and the `screen` Transmission
  provider).
- Unit tests in `src/feature.rs`: handle ids, validation order and the capacity and
  unknown-emitter refusals (CPU); a `software_gpu` test installs on a `LitRenderer`, adds
  one emitter of each kind, checks the three pass entries and that a second install
  refuses `pipeline.duplicate_pass` (prints `SKIP` without an adapter).

```
cargo clippy -p raa-particles --all-targets -- -D warnings
cargo clippy -p raa-particles --target wasm32-unknown-unknown -- -D warnings
```
