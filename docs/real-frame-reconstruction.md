# Real-frame reconstruction pilot

The real-frame branch has an early guide encoder and area-visibility head, a scene-resolution
recurrent state, and late reflection and display-resolution heads. The area head can run
before forward shading. The late update reads current post-lighting HDR and optional
incident reflection, and advances once per compatible real render. No target-time
prediction is involved in this model package.

## FP32 runtime contract

Activations are contiguous NHWC in active-rectangle coordinates, with the channel
fastest and FP32 arithmetic. Every integer spatial gather clamps to that active
rectangle. The CPU reference represents the same tensors internally as NCHW.
Weight tensors are little-endian FP32 with 16-byte-aligned offsets, zero inter-tensor
padding and no padding inside tensors.

| Layer | Weight shape | Weight offset | Bias offset |
|---|---|---:|---:|
| `stem` | [12,12] | 0 | 576 |
| `spatial` | [12,3,3] | 624 | 1056 |
| `mix` | [12,12] | 1104 | 1680 |
| `area` | [9,26] | 1728 | 2672 |
| `recurrent` | [24,33] | 2720 | 5888 |
| `reflection` | [3,12] | 5984 | 6128 |
| `sr` | [8,12] | 6144 | 6528 |

Pointwise shapes are [output,input]. Depthwise shape is [channel,row,column].
The guide lanes are signed view normal XYZ, `log1p(metricDepth)/8`, roughness,
unjittered current-minus-previous UV motion XY, accepted history, reactive coverage,
current source-pixel jitter XY, and geometry-valid. Zero metric depth marks invalid
geometry. Motion Z rejects history at or above 0.5.

With source pixel centers indexed by integers, sample previous state bilinearly at
`p - motionUV * sourceSize + previousJitterPx - currentJitterPx`. Require a compatible,
successfully submitted previous real render of this view, the same source/display
extents, the producer's optional validity bit, no reset, no motion rejection, and a
previous coordinate in `[-0.5,size-0.5)`. The nearest prior depth must be positive
and differ by at most `max(0.02 m, 0.025 * currentDepth)`. The dot product of current
and nearest previous signed view normals must be at least 0.8. Rejected samples have
zero history weight. Multiply accepted history by `1 - clamp(reactive,0,1)` before
feeding the network and before mixing recurrent state. Cuts and incompatible extents
reset state, while compatible motion uses the accepted feature history.

The stem applies pointwise ReLU. Add the pointwise mix of the clamped depthwise
3×3 result to the stem and apply ReLU again. This is the shared guide tensor.

- Area reads the shared guide, gated warped state, center visibility, and
  `log1p(U)`. Its nine logits give a row-major 3×3 softmax. Apply those weights
  identically to the current weighted visible and total pairs `(S,U)`. Divide
  the results, saturate to [0,1], and use visibility one for `U <= 1e-8`.
  The pilot processes one selected slot per call.
- The late pointwise layer reads the shared guide, gated warped state,
  current `log1p(HDR.rgb)`, and `log1p(reflection.rgb)` plus reflection validity,
  source class divided by four and age capped at 32 then divided by 32.
  Invalid reflection uses zero input lanes. Split the output into 12 candidate
  and 12 retention channels. The next state is
  `lerp(tanh(candidate), rawWarpedState, sigmoid(retention)*historyGate)`.
- The reflection head produces a bounded log residual around current incident
  radiance. Valid RGB resolves as
  `max(0, expm1(log1p(rgb) + 0.5*tanh(logits)))`. Invalid radiance resolves to zero.
  Apply any material/environment replacement outside this incident signal.
- The display head emits eight coefficients per scene pixel. For each display
  pixel, compute `q = (displayPixel + 0.5)*sourceSize/displaySize - 0.5 + jitterPx`.
  At each of the four bilinear taps, compute
  `b * exp(0.5*tanh(c0+c1*(qx-tx)+c2*(qy-ty)+c3*scaleX+c4*scaleY))`,
  where `b` is the positive bilinear weight and each scale is source/display.
  Normalize the tap weights. Filter current HDR and coefficients 5–7 with the
  same weights. Output
  `max(0, expm1(log1p(filteredHdr) + 0.25*tanh(filteredCoefficients)))`.
  Mapping works at native and noninteger ratios and uses the actual display extent.

## CPU training, export and evidence

Use the Kong Python environment with PyTorch. All training is on CPU. Export a seeded
operator fixture, run the CPU checks, or fit actual Windows data:

```sh
python -B scripts/train-real-frame.py seeded out/real-frame-seeded
python -B scripts/train-real-frame.test.py
python -B scripts/train-real-frame.py fit path/to/windows-sequence/manifest.json \
  out/real-frame-fit --area-capture out/area-pilot-win/manifest.json
```

The seeded fixture exercises a 3×2 scene and 5×3 display over three recurrent steps.
Its weights are for operator parity. A trained package names its supervised heads,
captures, fit frame IDs and held-out frame IDs. The optional area capture reuses
Ticket 10 as a spatial auxiliary on the shared encoder. Those sparse earlier
captures don't supply recurrent history or display targets.

The real-frame loader expects `raa.gallery-temporal-sequence/1` with a hardware
Windows receipt, sequences with whole-trajectory train/test assignments and frames
with actual lossless f32le source/label planes. Every plane lists its HWC shape, size
and SHA-256. The `guide` plane is the 12-channel capture input contract, and the CPU
recomputes acceptance from current and reprojected previous geometry. Recurrent
evaluation uses the model's previous features, explicit cuts and recorded jitter.
Optional aligned, supervised planes add `areaPairs` with `areaVisibility`, or
`reflection` with `reflectionTarget`. `analyticDisplayHdr` supplies an analytic
comparison when the capture includes it.

Training uses deterministic short causal unrolls with short same-sequence burn-in,
event-weighted robust log-HDR error, relative linear energy, image edges and
motion-correspondence error on changes relative to the teacher. Held-out trajectories
enter the evaluation only after export. CPU reload must exactly reproduce the
validation tensors and full captured rollouts. `manifest.json` hashes the weights and
golden vectors independently. Its package SHA-256 covers sorted-key compact JSON
without the package hash followed by exact weights and golden bytes.

A pilot fit and GPU parity establish executability and train-set response.
The held-out report gives per-sequence metrics and available baselines. A shipping
profile also requires broader independent rooms, rigs, assets and builds, held-out
quality at matched GPU cost, and physical on/off input-to-correct-photon evidence
meeting the brief's zero-added-latency gate. The package is marked unverified.
