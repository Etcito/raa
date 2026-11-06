# M23 SDF tracing

Every engine installs `trace.sdf` and `trace.composite`. Call
`engine.setTraceScene(program)` with a `Uint8Array` containing a complete little-endian
TraceProgram v1, as produced by MAMA's trace packer. The compute pass traces at the frame
size capped to 1024 pixels on the longest side; the composite writes colour and fragment
depth into the normal frame, allowing meshes and traced surfaces to occlude each other.
The current camera and light are used for each view.

`engine.updateTraceInstances(words)` replaces the instance records without changing their
count. Supply a `Float32Array` with 32 words per instance; integer fields must retain their
raw 32-bit representations (use a shared `Uint32Array` view). Bend records remain associated
with their original instance slots. Introducing a bend without an existing matching record,
or removing a bend while its record remains, is rejected. Use `setTraceScene` to change
bend payloads or associations. Rejected programs and updates leave the previous scene intact.
`clearTraceScene()` removes it; `traceStats()` returns source, instance, BVH node and
placeholder counts and capped target width and height.

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
