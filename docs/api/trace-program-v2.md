# Sampled SDF leaves (SDTP version 2)

`Engine.setTraceScene(bytes)` accepts SDTP versions 1 and 2. Version 1 cannot contain
opcode 9. Version 2 preserves the eight-word header and the source, material, instance
and bend layouts. `dataWords` now includes scalar sampled resources after all source
instruction data. Resources are part of the uploaded `trace_data` storage buffer;
there is no normal texture, triangle conversion, or additional binding.

Opcode 9 pushes one sampled leaf and its material. Its 24 f32 words are:

| Words | Meaning |
| --- | --- |
| 0–11 | Three rows of the source-to-leaf affine transform |
| 12 | Positive distance multiplier |
| 13 | Material index |
| 14–16 | Position of the first grid sample in leaf coordinates |
| 17 | Positive, uniform sample spacing |
| 18–20 | Integer grid dimensions, each 2–2048 |
| 21 | Absolute offset into `trace_data`, in units of four f32 words |
| 22 | Leaf Lipschitz upper bound |
| 23 | Storage format: 0 = f32, 1 = packed IEEE float16 |

Resources contain finite signed distances in x-fast order
`x + nx * (y + ny * z)`. Format 1 packs two float16 distances per word, low half
first; an odd sample count has zero padding. WGSL expands IEEE halves with integer bit operations and f32 arithmetic; no f16
device feature is needed. Bounds must cover the quantized values, and validation
uses those values. Offsets are four-word aligned and must follow every source's
instruction data. The producer caps the complete data buffer at 128 MiB. Identical
resources may be shared by instructions with different transforms or materials.

Inside the grid, the evaluator interpolates the eight neighbouring distances.
Outside, it evaluates at the clamped point and adds the distance to that point.
Every boundary sample must be strictly positive, so that extension creates no
additional zero surface. This is a sampled implicit surface, with accuracy determined
by the producer's grid and underlying sampler, rather than an exact analytic SDF.

For each axis, let `Ma` be the largest absolute adjacent-sample difference divided
by spacing. Each partial derivative of trilinear interpolation is a convex combination
of edge slopes. `sqrt(Mx² + My² + Mz²) + 1` bounds both interpolation and the exterior
extension. The reader checks this bound against the actual resource, including float16
quantization. A source's bound
must also cover the distance multiplier times the Frobenius norm of the leaf's inverse
linear transform. Ordered hard and polynomial smooth CSG retain the maximum operand
bound. Trace distance is divided by the source bound before stepping; hit tolerance
uses the same normalization. Producers must also bound analytic operands and include
smooth CSG expansion in source bounds.

`PROGRAM_EVAL_WGSL` evaluates opcode 9 for both world-volume stamping (`sdf_stamp`,
then `sdf_mip`) and final tracing. Surface normals come from shader evaluations of the
SDF gradient. The public CPU query evaluator also understands the format; rendering
does not call it or compile shaders on the CPU. Naga validates the composed WGSL in
crate tests.

Replacing a program uploads its distances with the other program data and invalidates
the world volume. Instance updates preserve resources; `clearTraceScene` releases the
scene. Invalid dimensions, offsets, boundaries, values, versions and insufficient step
bounds are rejected before installation. Rejected programs preserve the current scene.
