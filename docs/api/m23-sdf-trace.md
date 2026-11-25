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
environment and available GI. The stages run on a single viewport.

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
  uses the traced path on refresh pixels. Its `lightingOutput` can show direct,
  indirect diffuse, rough specular, or the rough-grid term in isolation.

Both `mesh.reflection` and `mesh.fitted-specular` accept two optional, current-frame grid
inputs. Connect `pass.gi.spherical.lookup` to `sphericalLookup` for the live spherical
lookup buffer, or connect `pass.gi.volume.volumeAtlas` to `volumeAtlas` for the voxel
atlas. These inputs accept their named producers and enforce execution order. With both
selected, spherical coverage is used first and the volume can fill uncovered regions.

An authored link on either input selects only the linked grids for that mesh node. With
neither input linked, `inheritGi` defaults to true and takes the spherical and volume
producers present in the graph. Set `inheritGi: false` to keep both inputs disconnected
when removing the last explicit link. Forward shading keeps its separate grid inputs,
so this setting can preserve a forward-only spherical setup. Existing frozen plans
without either input retain their volume-only path. Freeze again to capture the new
routing. The reference integrator has no grid inputs.

The grid estimates remaining indirect lighting at the world-space surface reached by
the existing ray. Geometry reflections keep their first-ray screen-confidence policy and
BVH hit, and the reached surface's direct shading follows its existing path.

The fitted rough lobe convolves directional cache radiance with four GGX
samples at the reached surface. It applies the secondary material's Fresnel
and visibility there, followed by the original ray's throughput. The
spherical path reads the six-face HDR pool and honors the selected lighting
groups. The Volume path traces directions in the existing clipmap.

The existing 524-byte FP32 fit keeps its frozen input contract. The fit
used two 128×72 indoor views and valid roughness values from 0.25 to 0.78.
Its inputs were freshly traced first-bounce estimates. Grid estimates stay
outside the model input. With a fixed seed, `captureMeshLighting` can return
the exact first BVH triangle and world position, original pre-grid RGB,
and accepted grid RGB from the same fitted frame. Refresh pixels retain
their existing additional traced bounce. Use the isolated `direct` view
to compare direct lighting on matching captures.

Each stage publishes full-size `color` for the HDR chain and linear `radiance` for inspection.
`radiance` is half size for the reflection and fitted stages and full size for the reference.
Connect the desired stage's `color` to the downstream tone mapper or HDR effect. The graph
[reflection fixture](../../crates/raa-graph/tests/fixtures/geometry-reflections.json) shows a
complete SSR and geometry-reflection route. `meshTracingStats(nodeId)` reports completed GPU
counters for a graph instance. Graph settings are per instance. Detaching the graph returns
control to the existing `setMeshTracing` settings.

### Grid-conditioned secondary diffuse experiment

The opt-in `mesh.grid-diffuse` graph stage uses the geometry-reflections wiring.
Connect `pass.gi.spherical.lookup` to its `sphericalLookup` input. An explicit
spherical-only link leaves the volume atlas disconnected. The default
`analytic-directional` mode cosine-weights six solid-angle-weighted face means
from the same cache. `legacy` measures the existing projection with identical
graph wiring. `learned` runs the same directional lookup with a shared 8→8→1
ReLU gate compiled into the mesh WGSL pipeline.

Load the frozen 324-byte little-endian FP32 checkpoint with
`setMeshGridDiffuseModel` before applying a learned graph. The tensor order is
row-major W1[8,8], b1[8], W2[8], b2. Inputs per cell are the three normalized
query offsets, normal agreement, directional luminance-moment alignment,
floored trilinear factor, geometric support and validity. The correction
multiplies the existing base weights by `exp(clamp(logit, -4, 4))` before
normalization. The existing support blend uses the uncorrected weights and
requires two compatible, current cells. The receiver's diffuse material weight
is applied once, after interpolation. Each cache slot gains 288 bytes for the
six RGB face means in each of three lighting groups.

For offline captures, call `setMeshGridDiffuseCapture(seed, stride)` and read
`meshGridDiffuseCapture(nodeId)` after the GPU has submitted and mapped a
frame. Zero stride disables capture. A stable hash of the primary trace pixel
selects one in stride eligible diffuse secondary hits. A candidate count over
128 marks a truncated frame. The seed changes the two independent 32-ray
reference banks. The reference reads scene-linear irradiance divided by π at
the real secondary hit and uses the same sky, sun and local-light groups as
the grid. RGB values include grid intensity.

Each row has 128 words. Interpret the header triplets as FP32, with the
following layout.

| Word | Field |
|---:|---|
| 0 | World position XYZ |
| 3 | Dense triangle ID (u32) |
| 4 | Shading normal XYZ |
| 7 | Material ID (u32) |
| 8 | Geometric normal XYZ |
| 11 | Neighbor count (u32) |
| 12 | Traced reference bank A RGB |
| 15 | Directional support blend |
| 16 | Traced reference bank B RGB |
| 19 | Legacy support blend |
| 20 | Legacy RGB before support blend |
| 23 | Sum of base neighbor weights |
| 24 | Active mode RGB before support blend |
| 27 | Light-group mask (u32) |
| 28 | Analytic directional RGB before support blend |
| 31 | Primary trace pixel ordinal (u32, row-major at trace resolution) |
| 32 | Up to eight 12-word neighbors. Each contains 8 FP32 features, RGB and base weight. |
