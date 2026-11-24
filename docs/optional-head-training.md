# Optional lighting head training

This CPU experiment fits one head at a time with the area pilot's fixed-frame
workflow. Each head uses an eight-channel pointwise ReLU layer and an independent
initialization. A sigmoid bounds AO visibility. Softplus gives the RGB heads
nonnegative scene-linear output and preserves HDR values. RGB, AO, and the existing
area visibility package have separate formats and can be evaluated independently.

## Run the CPU diagnostic

From the RAA checkout with CPU PyTorch available:

```sh
python3 -B scripts/generate-optional-head-diagnostic.py out/optional-head-diagnostic
for signal in indirect-diffuse indirect-specular ao-visibility; do
  python3 -B scripts/train-optional-head.py out/optional-head-diagnostic/manifest.json \
    "out/optional-head-diagnostic-$signal" --signal "$signal" || exit
done
python3 -B scripts/train-optional-head.test.py
```

Each output directory must be new. The generator produces two analytic grids
(7×5 and 5×3) with independently seeded noise and deterministic source masks.
The manifest and reports identify this data as `cpu-diagnostic`. The geometry,
reference signals, confidence values, and masks are synthetic.

## Capture contract

Write a manifest with `format: "raa.optional-head-capture/1"`,
`rowOrder: "top-down"`, `byteOrder: "little-endian"`, `exposure: 1`,
`jitter: [0, 0]`, a `source` object, and a nonempty `frames` array.
A Windows exporter uses `source.kind: "windows-render"` and includes
`fixture`, `buildId`, and `facts` with the Windows Chrome user agent in
`browser`, `hardware: true`, and `adapter.isFallbackAdapter: false`.
Record the actual adapter identity in `adapter`. A CPU fixture uses
`source.kind: "cpu-diagnostic"` and a named `fixture`. The trainer carries
the source object and manifest SHA-256 into the fit manifest and report.

Each frame has a unique string `id`, a `viewId`, a nonnegative integer `frameId`,
a nonnegative simulation `timeMs`, and `size: [width, height]` in native
pixels. Crop the input and reference to the same active rectangle. Use the
same frozen camera, geometry, and lighting state for the input and reference.
The `signals` object has one or more of the following keys. A fit uses the
frames containing its selected key.

| Signal key | Input channel order | Reference channel order |
| --- | --- | --- |
| `indirect-diffuse` | red, green, blue, guides, policy | red, green, blue |
| `indirect-specular` | red, green, blue, guides, roughness, policy | red, green, blue |
| `ao-visibility` | visibility, guides, policy | visibility |

Use isolated, scene-linear, unexposed indirect RGB contributions at the same
lighting stage for source and reference. Diffuse and specular each exclude
direct lighting and the other lobe. AO visibility is a scalar in [0, 1], where
one is fully visible. Roughness is perceptual material roughness in [0, 1].
Guides are `normalX`, `normalY`, `normalZ` in view space (each in [-1, 1])
and `metricDepth` in metres. Positive depth marks foreground, and zero marks
background.

Policy channels follow the guides (and roughness for specular), in this order:

| Channel | Meaning |
| --- | --- |
| `sourceValid` | Binary. The source value is usable for this pixel, including a valid reused value. |
| `sourceConfidence` | Confidence in [0, 1]. Write zero if confidence is unavailable. |
| `confidenceValid` | Binary. A confidence estimate is available for the valid source. |
| `freshlyTraced` | Binary. The valid source came from a trace at this pixel on this frame. |

For an invalid source, write zero RGB or AO visibility one. Background
uses those same neutral values in both planes, with all policy channels zero.
Foreground pixels with a missing source still need reference labels.

Each signal record contains `input` and `reference` plane descriptors with
`file` relative to the manifest directory, `dtype: "f32le"`,
`shape: [height, width, channels]`, `byteLength`, and lowercase
`sha256`. Pack finite FP32 lanes in contiguous top-down HWC order without
row padding. The descriptors and plane hashes follow the area pilot exporter
pattern.

List every stochastic seed contributing to the input in `inputSeeds`, including
seeds from reused samples. List two equally sized, nonempty reference batches
in `referenceMetadata.seedBatches`. All seeds are unsigned 32-bit integers.
The input and both reference batches for every head must be pairwise disjoint
within a frame. The exporter holds the scene state fixed and resets stochastic
and temporal state as needed to make the batches independent. Each batch
produces its own estimate, and the label combines both batches with equal
sample counts. Record:

- `estimator` and `settings` identifying the teacher and the settings needed
  to reproduce it. Include the signal stage and sampling settings, such as AO
  radius, where applicable.
- `batchRmse` between the two batch predictions over foreground pixels and
  output channels in the signal's units.
- `batchRmseTolerance` chosen for the reference estimator. The trainer rejects
  a frame when its reported batch RMSE exceeds the tolerance.

The diagnostic exporter is an executable example of the manifest and packing.
The trainer checks all listed seed streams and reference metadata. It checks
binary lengths, hashes, extents, signal ranges, and policy masks for the head
it fits.

## Read the fit

The report compares the raw source (including its specified fallback),
initialization, and fitted predictions. Each frame contributes equally to the
objective. MSE averages the foreground pixels and output channels in that
frame. Per-frame slices cover freshly traced, valid reused, and missing source
pixels, and separately partition pixels by confidence validity. These slices
support policy analysis without requiring a confidence threshold.

The fit directory contains a hashed little-endian FP32 weight blob, a tensor
and model manifest, and CPU input/output vectors for replay. Weight matrices
use output-channel-major row order. Background predictions in the vectors
use the signal's neutral value. The report records the selected checkpoint,
Adam settings, initialization seed and hash, and the input and teacher
provenance per frame.

With CPU PyTorch 2.15.0, 300 steps and a learning rate of 0.01, the analytic
diagnostics produced the following equally weighted foreground MSE:

| Head | Raw source | Initialization | Fitted |
| --- | ---: | ---: | ---: |
| Indirect diffuse RGB | 0.229088 | 0.314991 | 0.005106 |
| Indirect specular RGB | 0.195629 | 0.117793 | 0.003541 |
| AO visibility | 0.056511 | 0.030757 | 0.001286 |

Physical gallery quality and GPU cost remain unmeasured for these heads.
Use fixed gallery captures to compare per-head errors and mask slices before
changing encoder sharing or receptive field. An acceleration claim requires
GPU timing on the same gallery workload with the baseline and candidate,
including inference and the tracing work each policy performs.
