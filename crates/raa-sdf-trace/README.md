# raa-sdf-trace

The section 67 trace program: RAA's reader of the packed SDF program and its raytrace passes.
`src/program.rs` reads the file, `src/bvh.rs` and `src/cpu.rs` are the conservative CPU twin,
`src/wgsl.rs` generates the shared WGSL evaluator and owns the frozen internal volume ABI
(`VOLUME_ABI_VERSION`), and `src/pass.rs` schedules the world volume, the cone prepass, the
full-resolution trace and the `trace.composite` overlay.

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
`stats` reports the source, instance and BVH counts with the full target size.

`trace.sdf` uploads the packed program sections, the instance and bend records, the BVH nodes
and order, and the materials, then schedules the Claybook-style world volume and the trace:

1. **World volume.** The instance BVH root gives the world bounds; spacing is the longest
   extent divided by at most 1024 and the base dimensions are multiples of 16. The volume is
   an `R8Snorm` 3D texture with exactly five mip levels. Base distances are normalized to the
   padded diagonal range `R` and encoded in a packed `u32` buffer (four adjacent x bytes per
   word) by `sdf_stamp`, then copied into mip 0 with `bytesPerRow = align_up(width, 256)`,
   `rowsPerImage = height` and zero buffer offset; row padding is never sampled. Every coarser
   mip takes the minimum signed byte of its eight children (`sdf_mip`) and is copied the same
   way. The shader subtracts the half-diagonal voxel footprint and the roundoff allowance
   before quantizing, so every stored value bounds its whole cell, never an average.
2. **Cone prepass.** After all five mips are copied, `sdf_coarse` covers one texel per 8x8
   full-resolution pixel tile (`ceil(target/8)`), encloses every covered pixel footprint in
   its cone and stores `(resume distance, validity state, revision)`; state 1 is a certified
   empty prefix, state 2 forces exact fallback and there is no miss state.
3. **Full trace.** `sdf_trace` runs at the full render-target resolution (there is no image
   cap; 1024 is the longest volume axis only). It resumes from the coarse prefix only when the
   state and revision match, walks the mips chosen by clearance and pixel-cone size, then
   refines every candidate with the exact packed programs and retained BVH. Final hit
   distance, position and normal come from the exact evaluator, never from baked data;
   material lookup and named unsupported-shape placeholders are preserved.
4. **Composite.** `trace.composite` loads the frame colour and depth attachments and draws the
   trace colour with the trace depth as `frag_depth` under RAA's `Less` depth test, discarding
   misses, so meshes occlude traced surfaces and later overlays see the composed depth.

Program, instance, bend or bounds changes bump the scene revision; the pass rebuilds the
volume, stamps and copies mip 0..4 and publishes the revision to every uniform snapshot
*before* the cone prepass and full trace record. A rejected update keeps the previous program,
BVH and volume. A resize recreates the trace and coarse textures without restamping, and the
coarse prepass runs every frame because camera changes invalidate it.

## Limits

- Volume: at most 1024 voxels on the longest axis, at least 16 and multiples of 16 per axis,
  exactly five mips, `R8Snorm` sampled without filtering. A scene whose padded base buffer
  would exceed the default 128 MiB storage-binding limit is coarsened (spacing doubles) until
  it fits, never clipped; a scene without instances keeps a valid dummy volume.
- Trace output: the full render-target size (an empty frame collapses to 1x1), `Rgba16Float`
  colour and `R32Float` depth; the coarse prepass is `ceil(target/8)` `Rgba32Float`. The
  CPU twin `cpu::trace_ray` remains the exact reference for the GPU refinement.

## Test

```
cargo test -p raa-sdf-trace --test program
cargo test -p raa-sdf-trace --test variants
```
