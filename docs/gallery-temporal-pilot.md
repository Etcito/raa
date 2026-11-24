# Gallery temporal capture and prediction pilot

The first fit uses one 12-frame overfit replay. The independent 12-frame held-out
replay has its own camera path, object trajectories, panel event, seed range,
sealed skylight and vertex-authored wall colors. Keep the split before cropping or
choosing any checkpoint. This pilot measures pipeline and fitting correctness on
two gallery variants.

## Current real-frame target

The student produces consecutive 160 × 90 real views at recorded simulation times.
The dataset stores the actual pre-tonemap linear HDR, signed view normals with
roughness, metric view depth, unjittered current-minus-previous UV motion and
producer rejection. Depth zero means no geometry. Guide channels are declared in
the manifest in NHWC order. Motion and current jitter are in independent units.
Jitter in the manifest is source pixels, positive right and down. History is
conservatively rejected when the reprojected pixel leaves the previous rect or its
depth fails the correspondence test. Masks retain that distinction, plus the
renderer’s SSR offscreen classification and the current authored light event.
The reactive input marks visible geometry on the panel revision and conservatively
marks mirror coverage while the reflector moves. The SSR hit tap carries source
diagnostics. Incident RGB and area-visibility supervision are separate captures.

The teacher uses its own engine and the same recorded pose, material, transforms,
lights and simulation time. Native 320 × 180 mesh-path samples use independent
subpixel rays, fixed exposure, two scattering and two reflection bounces. Each
reference takes two disjoint seed batches of at least 64 spp each. Increase both
batches through 256 spp each until their log-HDR RMSE is at most 0.045. The saved
linear HDR is the average of the two batches. This is a higher-fidelity target for
the raster student, with the corresponding estimator difference. All planes are
little-endian FP32 with shape, byte length and SHA-256 in the manifest.

Use the direct driver on the native Windows RTX 3090. Build from the pinned
checkout with Kong. The optional final argument selects `train`, `test`, `both` or `reflection`.

```powershell
node scripts/capture-area-pilot-direct.mjs $Chrome dist/global-illumination out/gallery-sequence gallery-sequence both
```

The submitted frame ID binds the HDR, guides and timing to one view. The browser
records input cutoff, submit and canvas handoff timestamps, the queue fence and
readback completion separately. These observations do not measure physical
scanout.

## Screen incident teacher

The `reflection` run uses the typed `ssr.trace` and `ssr.incident` stages. It saves
both raw RGBA diagnostic textures and a packed FP32 `reflection` tensor. A lane
is eligible when the trace publishes screen source class 1, a valid geometric
hit, and positive incident confidence. The packed tensor stores three linear
incident channels, eligibility, source class and real-frame age.

Each disjoint high-sample batch supplies its own radiance pyramid at the same
pose and time. The capture maps the unjittered 2× HDR to the student grid, applies
the SSR 13-tap mip filter and samples the saved hit UV and cone level. Their
average is `reflectionTarget`. The manifest records disagreement on eligible
RGB samples. Misses are zero-filled and masked. This conditional reference
estimates radiance visible at the screen hit. Physical radiance along a BVH
reflection ray and offscreen or volume fallback remain separate estimators.

## Separate causal prediction target

Train a distinct decoder against the fully rendered HDR for the predicted
presentation time T. In each input packet record the last eligible real-frame ID,
source time and age, target tick and phase, input cutoff and accepted event IDs,
view, camera and transforms, light revisions and rollback generation.

Reproduce the engine's read-only cutoff snapshot and produce optional target
depth, motion and reactive guides from that authorized state only. Previous
completed real HDR and those target-time guides are the predictor's inputs.
Render full target HDR later for supervised comparison. A post-cutoff correction
or fresh control can appear in that label and must remain absent from the causal
packet. Distinguish scheduled events whose timing was known at cutoff from fresh
input and rollback. Score exposed pixels, offscreen subject recovery, lighting
response and skips. Compare against the last eligible real frame and the existing
causal warp, then test real-frame priority, expired deadlines and stale generations.

The predictive page replays two rightward training trajectories and two separate
leftward evaluation trajectories. Its cutoff includes the retained native HDR
and standard-Z depth, target motion and depth, the changed-light mask, and
bounded sparse refresh samples. The full target HDR submits only after the
candidate is presented. The packet records per-pixel warp and repair validity
under the native shader guards, and checks fresh samples against the independent
target before export.

Build the predictive page with `scripts/build-predictive-presentation.sh` and
the SDF Modeler engine WASM inputs. To stage a fitted model for the comparison,
set `RAA_GENERATED_FRAME_MODEL_DIR` to its manifest and weights directory.

```powershell
node scripts/capture-area-pilot-direct.mjs $Chrome dist/predictive-presentation out/generated-dataset --prediction dataset
node scripts/capture-area-pilot-direct.mjs $Chrome dist/predictive-presentation out/generated-ab --prediction learned
```

The learned check compares the same submitted source and cutoff with the model
off, on, then off. It verifies byte-identical displayed repairs, a changed warped
region, stable real-frame history, and fallback when the repair budget is short.
Capture-only
readbacks and the high-sample teacher stay outside the runtime prediction queue.
The current-frame data must not be fed to a prediction model as a target-frame
input. Keep learned features opt-in and disabled by default until the physical
zero-added-latency gate in the SDFM plan is measured.
