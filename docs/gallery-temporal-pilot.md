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
checkout with Kong. The optional final argument selects `train`, `test` or `both`.

```powershell
node scripts/capture-area-pilot-direct.mjs $Chrome dist/global-illumination out/gallery-sequence gallery-sequence both
```

The submitted frame ID binds the HDR, guides and timing to one view. The browser
records input cutoff, submit and canvas handoff timestamps, the queue fence and
readback completion separately. These observations do not measure physical
scanout.

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

Reuse the sibling RAA browser presentation fixture for this path. Capture-only
readbacks and the high-sample teacher stay outside the runtime prediction queue.
The current-frame data must not be fed to a prediction model as a target-frame
input. Keep learned features opt-in and disabled by default until the physical
zero-added-latency gate in the SDFM plan is measured.
