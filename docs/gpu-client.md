# Managed lights-page jobs

The [lights page](../examples/lights/) uses the shared
[page wrapper](../web/common/nmc-gpu.js) and the `M2-PAGE-JOBS` contract below.
Build with `bash scripts/build-lights.sh`; it produces `dist/lights/` from this
checkout's package, runner, wrapper and discovered scene modules.

Implementation status: [lights-job.mjs](../web/common/lights-job.mjs) implements
`select`, read-only `check`, `clean`, `render`, `fresh` and fail-closed `judge`.
Judgment calls `fresh` first, joins run pairs to selection metadata by scene and
variant, then judges primary captures and attached-light lifecycle captures.

## Commands and ordering

The registered names in `.nmc/commands.json` are `m2-lights-build`,
`m2-lights-functional`,
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
`clean` removes only the attempt named by `.lights-current.json` and its `run.json`,
retaining selection, unrelated files, sibling runs and archives. It is idempotent.
`render` invalidates the previous completion before requesting a client, publishes its own `dist/lights`
through `NMC_GPU_CLIENT`, then uses the returned `page_ref` for metadata-selected
pairs. Each input is exactly `{args: {scene, variant}}`. `fresh` requires successful
results and captures from that invocation. `judge` calls `fresh` before joining
run pairs to selection pairs by scene/variant. It invokes
`python3 "$NMC_JUDGE_CLIENT" <capture> --description <description> --min-score 6`
for every selected primary capture and each attached-light lifecycle capture.
Each result records its scene, variant, relative capture path, description, score
and judge exit code in `judge.json`. A missing verdict, failed process, score
outside [6, 10], incomplete prerequisite or missing lifecycle image leaves the
batch incomplete and rejects the command. The nine lifecycle stages are judged
in the order produced by the page, after their scene/variant primary capture.
Subprocesses use argument arrays without a shell. There is no local browser fallback.

`NMC_GPU_WAIT_S` defaults to 60 seconds for client availability, publication and
waiting for a GPU slot. The installed CLI reports queue/running transitions on
stderr. A separate 300-second running deadline rejects and terminates a hung
producer; queue time is not charged to that deadline. No spawn timeout wraps the
GPU client's entire lifetime. A timeout never accepts partial files. Producers
run sequentially, using at most one independent CPU worker; the build sets
`CARGO_BUILD_JOBS` from `NMC_CPUS` (default 1). Jobs require observed Chrome on
Windows, DPR 1 and a hardware adapter. Software/fallback identities and any
shader/device errors reject. Model tasks do not publish, render or judge.

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

`selection.json` contains `{page: string, pairs: Array<{scene, variant, description,
timeMs}>}`. The absolute page path and ordered pairs must still match the produced
metadata when cleaning, rendering or checking freshness. `check` writes nothing.
`run.json` contains the `RunRecord`:

```ts
{
  pageRef: string,
  pairs: Array<{scene: string, variant: 'on'|'off', capture: string, result: string}>,
  complete: boolean, buildId: string, runId: string, commit: string,
  startedAt: string, completedAt: string|null
}
```

Build provenance comes from `build.json`, written only after staging succeeds:
`buildId` is a UUID for that actual build invocation, `commit` is Git's verified
`HEAD`, and `builtAt` is a UTC ISO timestamp. It also records the producing
worktree, builder and tracked dirty state, so uncommitted staged changes are not
misrepresented as a new commit. If the host supplies `NMC_GPU_REVISION`, it must
equal Git `HEAD`. Missing Git/build provenance fails; a timestamp never substitutes
for a commit. No assumed `NMC_RUN_ID` or `NMC_COMMIT` variables are used.

The run's `pageRef` is the actual host-issued publication reference. A completed
run's `runId` is its first producer's actual host `job_id` (as a string); before
that producer returns, the incomplete record carries its publication reference
as the available host context. Each producer's real `job_id` and `client_id` are
retained in its client wrapper and completion receipt. The client at
`$NMC_GPU_CLIENT` was inspected: `publish`
prints `page_ref`; `render input.json --page-ref REF -o DIR` writes
`result.json` as `{job_id, client_id, result: <operation json>}` and downloads the
page files separately. Publication JSON wrappers and `page_ref: REF` lines are
also accepted, following [lookdev-capture.mjs](../scripts/lookdev-capture.mjs).

Every invocation uses a new `.lights-runs/<attempt UUID>/s-<encoded scene>/<variant>/`
namespace under the selected output directory. Pair paths are relative, unique,
contained and symlink-free. The explicit `capture.png`, client `result.json`, all
supplementary lifecycle files and both measured runs remain intact. Arbitrary PNGs
cannot stand in for `capture.png`. PNG chunk CRCs and decoded scanline structure
are validated, including dimensions; empty, truncated or corrupt files fail.

The run starts incomplete after publication and gains completion only after all
selected producers exit successfully and every output passes validation.
`completedAt` stays null until success. Per-pair completion receipts bind the
output inventory and managed job identity to the run/build/commit and selected
pair. `fresh(page, out)` rechecks all these conditions, including current build
identity, without publishing or rendering. A failed producer cannot inherit an
older attempt's success, PNG or completion receipt.

`m2-lights-functional` runs `node --test web/common/lights-job.test.mjs` after the
lights build. The command-check entry follows the functional and judge entries.
`dist/` and `out/` are ignored; captures/binaries remain there and in the managed
capture store, never committed. No reference image or output-hash check is used.

The many-lights operation retains device/shader setup and returns both on/off runs,
adapter identity, Chrome/Windows metadata and GPU errors. Its fixed camera uses
1920×1080, DPR 1, 64 lights, 60 warm-up frames and 300 measured frames; on requires
`fpsMean >= 60`. Animation-frame intervals produce `frameIntervalMsMedian`,
`frameIntervalMsP95`, `fpsMean`, `fpsMedian`, `fpsP95`; CPU submission timings keep
their existing meanings. Raw `frameIntervalsMs` must contain exactly 300 positive
finite intervals for each run. Transport uses the staged runner's
`summarizeFrameIntervals` to validate every FPS/interval summary and checks both
runs' dimensions and warm-up/frame counts. Ordinary page evidence requires
1280×720, 30 warm-up frames and one capture frame. Upstream unmeasured runner
calls retain their existing result keys; browser/error fields are page evidence.
