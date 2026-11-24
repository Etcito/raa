# Learned reflection and generated-frame head pilots

The two heads use separate training signals. Reflection extends the shared
real-frame model. Frame generation refines the existing causal target-guide
warp on the presentation clock. The analytic generated-frame path remains
available independently.

## Reflection

Use incident scene-linear RGB before Fresnel, environment replacement, or
material weighting. Capture the noisy input, the scene-resolution HDR the
late encoder reads, depth, normals, motion, and two independent converged
teacher batches at the same time and pose. The capture metadata carries the
sample seeds and convergence comparison.

The input plane stores [R,G,B,valid,source class,age]. Train the three
`reflection` outputs against the supported incident RGB teacher. Freeze the
stem, recurrent weights, area head, and display head from the fitted
`real-frame-v1` checkpoint. The exporter checks that all frozen bytes stay
identical. The reflection package retains the existing tensor ABI.

The experimental `neural.reflection-reconstruction` node consumes and
produces incident radiance. Provider coverage comes from the valid incident
input. A separate offscreen provider is required to reconstruct unsupported
offscreen hits.

## Generated-frame refinement

`generated-frame-v1` uses four FP32 operators, NHWC activations, clamp gathers,
and little-endian weight tensors at 16-byte boundaries.

| Layer | Weight shape | Weight offset | Bias offset |
|---|---:|---:|---:|
| `stem` | [12,14] | 0 | 672 |
| `spatial` | [12,3,3] | 720 | 1152 |
| `mix` | [12,12] | 1200 | 1776 |
| `head` | [3,12] | 1824 | 1968 |

The 14 inputs in order are log1p warped RGB, log1p fresh RGB, warp-valid,
fresh-valid, target standard Z, target-minus-source UV XY, target lighting
change, retained real-frame age capped at eight and divided by eight, and
motion reject. The two RGB triplets are zero if their validity is false.
Apply a pointwise ReLU stem, depthwise 3×3 and pointwise mix with a residual
ReLU, then three pointwise outputs.

Refine a valid unrefreshed warp with
`max(0, expm1(log1p(warp.rgb) + 0.25*tanh(logits)))`, clamped to finite
RGBA16F range at storage. A trusted fresh repair passes through exactly.
The admission checks still reject incomplete frames. The learned operator
runs only for an explicitly selected package on the target-guide prediction
path. Prediction retains the source real frame and its state. It advances
neither real reconstruction history nor authoritative simulation.

The capture auditor checks the retained real ID and age, cutoff clock,
accepted input receipt timestamps, revision, the timestamp of the separate
label render and the exact warp visibility decision against source depth and
target guides. The label contributes to loss and evaluation. Fresh input is
rounded to the runtime's RGBA16F storage precision. For the earlier exporter,
the explicit `--replay-warp` option derives the exact runtime warp from the
original hashed cutoff inputs and records changed-pixel counts in provenance.

## Reproduce

Use the Kong PyTorch environment. Both trainers run on CPU. The capture
receipts must identify a Windows Chrome hardware WebGPU adapter.

```sh
python -B scripts/train-neural-heads.test.py
python -B scripts/train-neural-heads.py generation-parity out/generated-frame-seeded
python -B scripts/train-neural-heads.py reflection path/to/reflection/manifest.json \
  out/reflection-fit --base out/real-frame-fit-96ebd55
python -B scripts/train-neural-heads.py frame-generation path/to/fg/manifest.json \
  out/generated-frame-fit
```

Seeded packages prove layouts and operators. Measured fit reports compare
the frozen learned output with the native incident signal or analytic warp
on the entire frame and on the head's active pixels. Train and held-out
groups are disjoint. The trainers score held-out data after saving the
weights. The package remains experimental pending matched-cost quality and
physical input-to-correct-scene-photon measurements.

### First generated-frame diagnostic

The corrected Windows Chrome RTX 3090 capture at `5db9c2d` supplied two
training trajectories and two held-out trajectories with independent target
rendering. Two 250-step CPU fits produced identical weights, manifest,
golden vectors and reports. The training objective moved from 0.001155 to
0.000235. CPU reload and trusted-fresh error were both zero.

| Held-out path | Analytic linear HDR RMSE | Learned linear HDR RMSE | Analytic log MAE | Learned log MAE |
|---|---:|---:|---:|---:|
| Short left | 0.001898 | 0.001708 | 0.000274 | 0.000383 |
| Wide left | 0.001698 | 0.001646 | 0.000250 | 0.000373 |

Each path has one eligible target packet. These measurements qualify a
deterministic pilot only. The held-out log-error regression and pending
physical latency gate keep the default analytic path and experimental gating.

### First incident reflection diagnostic

The Windows Chrome RTX 3090 capture at `8381760` supplied 12 training and
12 held-out frames. Each frame had at least 2,301 accepted screen hits.
Independent teacher batches agreed within 0.00363 log-HDR RMSE on those
hits. Two 250-step CPU fits reproduced weights, manifest, golden vectors and
report byte for byte. The train objective moved from 0.22985 to 0.15035,
and captured-frame CPU outputs matched exactly after reload.

| Held-out mean over accepted screen hits | Native incident input | Fitted reflection |
|---|---:|---:|
| Linear HDR RMSE | 0.08845 | 0.07797 |
| Log-HDR MAE | 0.05062 | 0.04441 |

The combined package keeps the trained area, display, and recurrent weights
exactly as exported by the previous fit. The supported class is typed SSR
screen hits. Mesh-traced offscreen hits, volume and sky fallbacks use their
existing renderer paths.

The learned generated-frame dispatch used the fitted `5db9c2d` package in
the Windows A/B fixture at `246c570`. It changed 2,831 warped pixels while
preserving all 132 fresh repairs and the real-frame history count. Turning
the model off restored the analytic image byte for byte. The held-out quality
figures cover the target-guide actor-motion profile. A later graph integration
also permits the fitted residual on compatible camera-only predictions. That
profile has its own input distribution and requires separate quality and
latency qualification.
