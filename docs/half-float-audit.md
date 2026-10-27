# Half-float target audit (0.4.1)

The shafts march stores view depth in Rgba16Float alpha. `1e6` overflows
binary16, whose largest finite value is 65504. The shader now uses that exactly
representable finite sentinel and clamps geometry depth to [0, 65504] in both
march and composite. This retains metre-based relative bilateral weights without
an encoding change. Sky/sky comparisons subtract equal finite values; thin nearby
geometry receives finite, small sky weights. Depth discrimination beyond 65504 m
is intentionally saturated.

Other target writers across the crates were checked:

| Writers / data | Storage and finding |
| --- | --- |
| DOF prefilter, gathers, composite | Half-float RGB and coverage only. `FAR_DEPTH = 1e6` is used only in f32 local bilateral calculations, never stored. CoC and tile extrema (`BIG = 3e38`) use R32Float/Rg32Float. |
| SSR trace | Half-float hit UV, confidence and logarithmic cone mip. `NO_T = 3.4e38` is local traversal state, never output. Hi-Z is R32Float; colour mip/composite targets carry HDR colour. |
| Lit prepass | Half-float unit normals and roughness, no linear depth. Scene and shadow depth use Depth32Float. |
| GTAO, contact shadows | Half-float outputs contain visibility in [0, 1] and fixed zero/one channels. GTAO intermediate targets are R32Float. |
| IBL BRDF LUT | Packed binary16 scale/bias from bounded BRDF integration, no depth sentinel. |
| Sky/environment, IBL filter/downsample | Half-float radiance and alpha; no depth or large sentinel is stored. SH projection uses f32 buffers. |
| Forward, fur, particles, transmission, OIT resolve, bloom, fog, grading | HDR colour/coverage targets, no stored large depth sentinel. Transmission's `1e30` no-absorption distance is an f32 shading input. OIT node colour packing already clamps to 65504; node depth remains f32. Fog's sky distance 20000 is used in calculations, not stored as depth. |
| Pipeline fallback lighting textures | Binary16 unit/zero lighting values; no large sentinel. |

No additional instance of the depth-sentinel storage bug was found. HDR radiance
is not universally range-limited: extreme lighting, material functions or grading
can still exceed binary16's colour range. This audit does not change the engine's
HDR colour policy.
