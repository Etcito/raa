# raa-particles

GPU particles for the RAA lit renderer. This skeleton holds the emitter
description and its validation; the simulation and draw passes land in later
tasks.

## Public items

- `ParticleKind`: `Splash`, `Snow`, `Confetti`, `Sparkles`.
- `EmitterDesc`: kind, position, direction, spread, rate, lifetime, speed,
  size, gravity, drag, colours, emissive strength, spawn volume, ground plane,
  particle cap and seed.
- `EmitterDesc::for_kind(kind)`: per-kind defaults (gravity 9.81 except snow
  1.0 and sparkles 0).
- `EmitterDesc::validate()`: refuses with `particles.invalid`, naming the
  offending field. Every number is finite; direction non-zero; spread in
  `[0, pi]`; rate `>= 0`; lifetime `0 < min <= max`; speed `0 <= min <= max`;
  size `0 < min <= max`; drag `>= 0`; 1 to 8 colours with components `>= 0`;
  emissive `>= 0`; volume half extents `>= 0`; `1 <= max_particles <= 65536`.

## Testing

```
cargo test -p raa-particles
```

`tests/desc.rs` (Q12, CPU only) validates every per-kind default and checks
that each rule is refused with `particles.invalid` and a non-empty message
naming the field.
