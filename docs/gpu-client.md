# Managed lights-page jobs

The [lights page](../examples/lights/) uses the shared
[page wrapper](../web/common/nmc-gpu.js) and the `M2-PAGE-JOBS` contract below.
Build with `bash scripts/build-lights.sh`; it produces `dist/lights/` from this
checkout's package, runner, wrapper and discovered scene modules.

Implementation status: [lights-job.mjs](../web/common/lights-job.mjs) currently
implements `select` and the read-only `check` preflight. The registered `clean`,
`render`, `fresh` and `judge` commands are pending transport implementation in
this checkout. Their required behavior is documented here as the stable contract.

## Commands and ordering

The registered names in `.nmc/commands.json` are `m2-lights-build`,
`m2-lights-select`, `m2-lights-clean`, `m2-lights-render`, `m2-lights-fresh`,
`m2-lights-judge` and `m2-lights-command-check`. The chain builds before selecting,
selects before cleaning/rendering, then validates freshness before judgment.

```sh
bash scripts/build-lights.sh
node web/common/lights-job.mjs select --page dist/lights --out out/lights
node web/common/lights-job.mjs clean --page dist/lights --out out/lights
node web/common/lights-job.mjs render --page dist/lights --out out/lights
node web/common/lights-job.mjs fresh --page dist/lights --out out/lights
node web/common/lights-job.mjs judge --page dist/lights --out out/lights
```

`select` imports the produced build and refuses invalid/empty selections.
`clean` removes that output directory's previous generated captures/results,
retaining or regenerating selection. `render` publishes its own `dist/lights`
through `NMC_GPU_CLIENT`, then uses the returned `page_ref` for metadata-selected
pairs. `fresh` requires successful results and captures from that invocation.
`judge` runs `python3 NMC_JUDGE_CLIENT <capture> --description <metadata description>
--min-score 6` for every pair and fails on rejection. Subprocesses use no shell.

`NMC_GPU_WAIT_S` defaults to 60 seconds; unavailable clients fail. Independent
CPU cases use at most `NMC_CPUS` processes (default 1). GPU jobs use managed Chrome,
collect shader/device errors and accept no software fallback. Model tasks author
commands; programmatic execution runs `render` and `judge`.

## Metadata and page protocol

Built `scenes.json` is an array of `{id:string, milestone:string, file:string}`;
`file` is a relative scene-module path within the page. The default export has
`id`, `milestone`, `variants`, async `build(engine, lib, variant)` and optional
`frame(engine, timeMs, variant)`, following the existing S14 scene convention.
[main.js](../examples/lights/main.js) discovers modules from that index; later
scene modules extend discovery without a second hard-coded preset registry.

Each variant has `id: 'on' | 'off'`, a nonempty `description`, `judge: true`,
`timeMs: number`, optional `passes: {disable?: string[], order?: string[]}`, and:

```ts
selection: {
  camera: {
    eye: [number, number, number], target: [number, number, number],
    fovYDeg: number, near: number, far: number
  },
  subjects: Array<{min: [number, number, number], max: [number, number, number]}>
}
```

Coordinates are world metres. Selection requires at least one subject and validates
ids, descriptions and box/view-volume overlap, including enclosing floors with no
corner inside the screen. The metadata owns preset/variant ids: point, spot,
directional, area, many-lights, attached-light and motion, each on/off. Descriptions
state that local lights cast no shadow yet and explain prepass tile coverage.

The built `nmc.gpu.json` declares `render`; the page exposes `window.nmcGpu.ops`.
The wrapper returns `{json: object, files: Array<{name: string, base64: string}>}`;
it explicitly supplies `capture.png`. The managed client takes no screenshot on
the page's behalf. Assets use relative URLs and the published page reference.

## Outputs and functional results

`selection.json` contains `{pairs: Array<{scene, variant, description, timeMs}>}`;
the current selector also records the absolute `page` path. The transport contract
writes `run.json` as `{pageRef: string, pairs: Array<{scene: string, variant: string,
capture: string, result: string}>, complete: boolean}`. Each returned capture goes
to `out/lights/<scene>--<variant>/capture.png`, beside its textual result.
`dist/` and `out/` are ignored; captures/binaries remain there and in the managed
capture store, never committed. No reference image or output-hash check is used.

The many-lights operation retains device/shader setup and returns both on/off runs,
adapter identity, Chrome/Windows metadata and GPU errors. Its fixed camera uses
1920×1080, DPR 1, 64 lights, 60 warm-up frames and 300 measured frames; on requires
`fpsMean >= 60`. Animation-frame intervals produce `frameIntervalMsMedian`,
`frameIntervalMsP95`, `fpsMean`, `fpsMedian`, `fpsP95`; CPU submission timings keep
their existing meanings. Unmeasured calls retain their existing result keys.
