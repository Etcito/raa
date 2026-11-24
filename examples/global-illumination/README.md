# Neural rendering baseline

Build the global-illumination page with `bash scripts/build-global-illumination.sh` in the Kong environment. Open the staged page in Chrome on the PC and display under test. Select **Measure keyboard latency** and press Space each time the status asks for the next event. Keep the canvas focused. Select **Download latency data** after the trial finishes.

The managed GPU operation is `baseline` on `dist/global-illumination`. It defaults to two fresh scripted trials. Each runs a 240-frame warm-up, then renders 96 frames with identical simulation times and event IDs. The existing 2×2 stratified area shadow and analytic emitter remain active at native resolution, with fixed exposure and the proposed denoiser, upscaler and predictor off. Scripted input is excluded from key latency distributions.

The operation accepts `mode` (`scripted` or `keyboard`), `captureId`, `repeat`, `width`, `height`, `remainingMs`, and the operator-supplied `loadClass`, `refreshHz` and `powerMode`. Use a different load class label for each test condition. The harness reports observed animation-frame intervals separately from the configured refresh rate. Each accepted keyboard input keeps its browser event timestamp and is joined to the attempt that first applied the event by view, trial and present ID. An accepted input that reaches no attempt keeps a null present ID.

Run `node scripts/neural-baseline-latency.mjs neural-baseline.json` to summarize by load class. GPU submission and canvas handoff timestamps come from CPU-side observations. When available, a matching Long Animation Frame entry supplies the browser's document presentation timestamp. The next animation callback is an estimate of a rendering opportunity. Event Timing entries are reported separately for matched keys and can be absent on fast events. The page cannot confirm that the affected WebGPU canvas frame appeared at a particular display refresh.

## Fixed area-light training capture

On the NoMoreChat Windows GPU setup, provision the checkout with Kong. From PowerShell
with Git Bash and the Cargo.lock version of wasm-bindgen on PATH, run:

```powershell
kong use
kong env --shell ps1 | Invoke-Expression
bash scripts/build-global-illumination.sh
node scripts/capture-area-pilot.mjs dist/global-illumination out/area-pilot-win
```

The Node entry point uses the managed `NMC_GPU_CLIENT` and its execution credentials
to publish the built page and run `area-pilot` on Windows Chrome. The output path must
be new. A complete capture writes `manifest.json` after checking every downloaded
array's byte length and SHA-256. Pass that manifest to `scripts/train-area-pilot.py`.
The Node workflow has no browser launcher or npm install step.

The fixed 65×49 Cornell sequence delivers static frame 0 and every frame 14–18 around
the blocker step at frame 16, with one view, one slot and zero jitter. The native
little-endian trace and guides retain their GPU texel bits. The packed FP32 input
accepts the previous raw pair by motion and depth and resets it at the blocker step.
The offline teacher freezes pose and time, checks an exact same-seed repeat at every
label, and accumulates independent batches over complete 2×2 strata. The label is
the ratio of the combined sums, with visibility 1 for zero total weight. A frame
fails after 256 samples per batch if independent-batch visibility RMSE exceeds 0.02.

When managed execution credentials are unavailable, the same built page can be run
locally on the Windows RTX 3090 with Kong's Windows Node and installed Chrome:

```powershell
node scripts/capture-area-pilot-direct.mjs 'C:\Program Files\Google\Chrome\Application\chrome.exe' dist/global-illumination out/area-pilot-win
```

The direct driver uses a temporary Chrome profile and loopback-only HTTP and DevTools.
The manifest records Chrome's native renderer, the WebGPU adapter identity, the build
receipt hash and the verified capture array hashes.

## Spherical GI in the interactive gallery

Choose **Spherical** under **Indirect grid** while global illumination is on.
The grid replaces diffuse indirect light where ready captures can support the surface.
The selected volume solver supplies the remaining pixels. Choose **Volume** to free
the spherical capture resources. The live statistics show occupied cells, ready faces,
tracked lighting and GPU memory.

`Engine.setSphericalGi(patch)` accepts a boolean `enabled`, an integer
`sphereSubdivision` from 0 to 6, and a finite `exponentialPower` from 0 to 2.
Subdivision gives `2 ** sphereSubdivision` angular cells along each axis of a
cube face. Consecutive shell widths grow by `2 ** exponentialPower`, with 0
giving uniform widths. `clipStart` and `zMax` are finite radial boundaries in world
units. `clipStart` is nonnegative and `zMax` is strictly greater than `clipStart`.
Shell boundaries must remain distinct in float32. The final shell ends at `zMax`
and its center is half its width inside that boundary.

Run the comparison with the direct Chrome driver on Windows.

```powershell
node scripts/capture-area-pilot-direct.mjs 'C:\Program Files\Google\Chrome\Application\chrome.exe' dist/global-illumination out/gallery-spherical-win gallery-spherical
```

The evidence includes linear RGBA16F captures, separate sky, sun and local-light
contributions, comparison with both volume solvers, grid and light invalidation,
and hot control checks. Add the final argument `live` and use a fresh output
directory to check the same build at port 8081.
