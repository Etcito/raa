# Browser predictive presentation fixture

The fixture captures immutable cutoff snapshots through the Etcito Game Engine rollback binding.
Its single native-size scene has a moving unlit rectangle, a disoccluded background and an
independently refreshed emissive strip. The host builds exact analytic target depth, primary
motion and repair samples for this scene. RAA warps the retained real HDR, dispatches sparse
repair, applies its ordinary tone mapper and presents the selected canvas image. The DOM HUD
uses the current cutoff target once per display opportunity.

Build with the RAA Kong environment, the Cargo.lock version of wasm-bindgen, and an engine
Wasm brick compatible with the supplied engine checkout.

```sh
eval "$(kong env)"
bash scripts/build-predictive-presentation.sh ../etcito-game-engine /path/to/ege.wasm
```

Open the generated page in native Windows Chrome or run the existing direct Windows evidence
driver. Its argument paths are Windows paths. Use a new output directory.

```powershell
node scripts/capture-area-pilot-direct.mjs $Chrome dist/predictive-presentation out/prediction-win --prediction
```

The driver requires native RTX 3090 WebGPU and saves browser and build provenance. The fixture
checks expired deadlines, budget overflow, real-history isolation, HUD selection, pixel agreement
with a late full target render, and flash fallback. Its deadline is a verification allowance.
Physical latency and queue headroom require separate measurement.

Browser prediction is opt-in through `preparePrediction()`. A host that supports it can call
`submitPrediction()` with native-grid typed arrays from its cutoff snapshot, then call
`presentReady()` and update its HUD. An unsupported canvas can keep using `render(nowMs)`.
The fixture's analytic guide and repair producer applies only to its authored geometry. General
game guide generation, GPU-resident guide input, and predictive decoder training remain separate
work. The browser bridge uploads dense arrays and should be profiled before production use.
