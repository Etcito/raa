# raa-sdf-trace

The section 67 trace program: RAA's reader of the packed SDF program and its raytrace passes.
`src/program.rs` reads the file, `src/bvh.rs` and `src/cpu.rs` are the conservative CPU twin
and `src/wgsl.rs` generates the shared WGSL evaluator, the `trace.sdf` compute shader and the
`trace.composite` display shader.

## File layout

All words are little-endian. The file is `header`, `sources`, `code`, `data`, `materials`,
`instances` in that order:

| section | words | contents |
|---|---|---|
| header | 8 | `magic` (`SDTP`), version 1, source count, instance count, code words, data words, material count, reserved 0 |
| source | 12 each | code start/length, data start/length, `min` xyz, `max` xyz, Lipschitz bound, flags (bit 0 placeholder) |
| code | 1 each | opcode bits 0..7, data offset bits 8..31 |
| data | 1 each | `f32` payload |
| material | 8 each | linear rgb, roughness, metallic, three reserved zeros |
| instance | 32 each | `worldFromLocal` 3x4, `localFromWorld` 3x4, distance scale, source, material override (`i32`, -1 = none), key, deform kind, bend angle, bend length, packed axes |

A primitive instruction needs `localFromParent` 3x4, distance scale, material index and its
parameters: box and roundBox 4, sphere 1, capsule/cylinder/torus 2, cone/ellipsoid 3. A union,
subtract or intersect needs one word (the smooth factor). The data offset is relative to the
owning source's data span, so each source is a self-contained program.

Integer record slots keep their 32-bit patterns: the key is a `u32`, the material override an
`i32`, and the deform kind and packed axes are read as raw words, never through `f32`.

## Public items

- `parse(bytes)` and `TraceProgram::parse` return a `TraceProgram` with its sources, code,
  data, materials and instances.
- `Source::{code, data, placeholder}` reads a source's spans from the program.
- `opcode` names the eight primitive, three boolean and parameter word counts.
- `TraceError { code, path, message }` implements `Display` and `std::error::Error`; codes are
  `trace.format`, `trace.version`, `trace.limits`, `trace.source`, `trace.opcode`,
  `trace.data`, `trace.stack`, `trace.reference` and `trace.not_finite`. Truncation, extra
  words, a bad magic and a bad reserved word are `trace.format`.

`parse` checks the header counts and the exact word total before slicing, then validates every
source range, Lipschitz bound and bounds box, all geometric values (bounds, matrices, distance
scales, bend values and data) for finiteness, every instruction's opcode and data range, the
stack balance (one value left per source), and every instance/source/material reference. A
bend record also needs a positive length and a packed frame whose along axis is one of the six
signed axes (`toward` 7 means no aim); an incomplete frame is refused.

## Installation and frame passes

`TraceFeature::install(gpu, &mut pipeline)` adds `trace.sdf` (compute, order 55) and
`trace.composite` (display overlay, order 560) and returns a `TraceHandle` (a shared
`TraceScene`). Set the program with `set_program`, the instance records with
`update_instances` and the frame's 512-byte S10 `FrameGpu` (camera and inverse matrices, sun
direction/colour, ambient, viewport) in `scene.frame`; both passes read the same bytes.
`stats` reports the source, instance and BVH counts with the capped target size.

`trace.sdf` uploads the packed program sections, the instance and bend records, the BVH nodes
and order, and the materials, then dispatches 8x8 workgroups over `target_size(w, h)` (never
larger than 1024 on the longest side). It writes the linear colour to `Rgba16Float` and the
projection depth to `R32Float`.

`trace.composite` loads the normal frame's colour and depth attachments. It draws the trace
colour with the trace depth as `frag_depth` under RAA's `Less` depth test and discards misses,
so scene meshes occlude traced surfaces and the depth written here is seen by later overlays.

## Test

```
cargo test -p raa-sdf-trace --test program
cargo test -p raa-sdf-trace --test variants
```
