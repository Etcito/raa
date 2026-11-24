# M23 SDF tracing

Every engine installs `trace.sdf` and `trace.composite`. Call
`engine.setTraceScene(program)` with a `Uint8Array` containing a complete little-endian
TraceProgram v1 or v2, as produced by MAMA's trace packer. GPU compute stamps the scene into an
R8Snorm world-space distance volume with five mip levels. Its longest axis is at most 1024
voxels; this limit does not apply to the image. A coarse cone prepass guides volume tracing
at the full render-target resolution. Hits and normals are refined against the exact packed
programs, and the composite writes colour and fragment depth into the normal frame so meshes
and traced surfaces occlude each other. Each view uses its current camera and light.

`engine.updateTraceInstances(words)` replaces the instance records without changing their
count. Supply a `Float32Array` with 32 words per instance; integer fields must retain their
raw 32-bit representations (use a shared `Uint32Array` view). Bend records remain associated
with their original instance slots. Introducing a bend without an existing matching record,
or removing a bend while its record remains, is rejected. Use `setTraceScene` to change
bend payloads or associations. Rejected programs and updates leave the previous scene intact.
Accepted program and instance changes invalidate the volume, which is rebuilt before tracing
the changed scene. `clearTraceScene()` removes it; `traceStats()` returns source, instance,
BVH node and placeholder counts and the full target width and height.

The header's eighth word is the bend record count. Following the 32-word instances are
16-word bend records, one per bent instance in instance order. Each is eight little-endian
f64 values: axis, signed angle in degrees, direction in degrees, start, end and center xyz.
These are the authoritative MAMA rail parameters; missing records are rejected and never
synthesized. The instance retains its bend kind, angle, length and packed signed axes,
including toward value 7 for no aim. Zero-angle identity bends carry no record.

Validation errors carry their `trace.*` code and describe the invalid field; an update with
an incorrect instance count returns `trace.instance_count`. The reader also reports specific
format, version, limit, reference and numeric errors. See `raa-sdf-trace` for the packed
format and supported shapes. Version 2 adds scalar sampled leaves; see
[the version 2 format](trace-program-v2.md). Producers must refuse unsupported geometry
with a named error instead of silently omitting it. Legacy v1 placeholders remain readable.

## Mesh tracing in a render graph

Mesh trace stages share the engine's current opaque triangle BVH, material library, local lights,
environment and available GI. Add `pass.gi.volume` when the reflection path should sample the
scene's voxel GI. The stages run on a single viewport.

- `mesh.reflection` applies geometry reflections to materials marked `traceMirror`. Connect the
  prepass `depth`, `normal` and `motion`, `ssr.hiz.hiz`, `ssr.trace.trace` as `screenTrace`, and the
  original forward HDR as `screenColor`. Connect the ordinary SSR composite to `color` to preserve
  SSR on other surfaces. Stable screen hits above `confidenceThreshold` can bypass BVH traversal.
  `reflectionBounces` controls the mirror budget.
- `mesh.reference` replaces its linked HDR `color` with a full-resolution path-traced image. Set
  `reflectionBounces`, `pathBounces`, `samples` and `seed` to control sampling.
  `lightingOutput` selects `combined`, `direct`, `indirect`, or `bounce`. With `bounce`,
  `lightingDepth` counts continuations from the camera hit, including mirror continuations.
  Use the same seed, sample count and bounce budgets to compare depth slices.
- `mesh.fitted-specular` uses the same screen inputs as `mesh.reflection` and supplies a complete
  path-assisted HDR image. Call `setMeshPathModel` with the validated fitted weights before
  applying the graph. The fit estimates the specular lobe on eligible non-refresh pixels and
  uses the traced path on refresh pixels.

Each stage publishes full-size `color` for the HDR chain and linear `radiance` for inspection.
`radiance` is half size for the reflection and fitted stages and full size for the reference.
Connect the desired stage's `color` to the downstream tone mapper or HDR effect. The graph
[reflection fixture](../../crates/raa-graph/tests/fixtures/geometry-reflections.json) shows a
complete SSR and geometry-reflection route. `meshTracingStats(nodeId)` reports completed GPU
counters for a graph instance. Graph settings are per instance. Detaching the graph returns
control to the existing `setMeshTracing` settings.
