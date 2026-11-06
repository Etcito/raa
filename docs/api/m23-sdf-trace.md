# M23 SDF tracing

Every engine installs `trace.sdf` and `trace.composite`. Call
`engine.setTraceScene(program)` with a `Uint8Array` containing a complete little-endian
TraceProgram v1, as produced by MAMA's trace packer. GPU compute stamps the scene into an
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
format and supported shapes; unsupported sources are placeholders.
