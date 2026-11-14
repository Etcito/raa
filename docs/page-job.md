# The shared page capture job

## 1. Purpose

`web/common/page-job.mjs` is the one implementation of the page capture job. A page job
file is a configuration passed to `createPageJob`: it names the page, its stages and its
validators, and adds its own output lines through `report`. It never re-implements a
stage. `web/common/sky-weather-job.mjs` and `web/common/vg-job.mjs` are such
configurations; the stages, record validation, resumable store, freshness checks and
judge calls live only in `page-job.mjs`. A configuration built with `createPageJob`
exports the stage functions and ends its file with `runCli(job, import.meta.url)`, which
runs `main` only when the file is the process entry point.

## 2. The stages

The actions are `check`, `select`, `clean`, `render`, `fresh` and `judge` (`ACTIONS`), and
the stage names are the configuration's `stages` list.

**check** verifies the staged build without a GPU. `readBuildReceipt` reads
`<page>/build.json`, checks its eight `RECORD_FIELDS.buildReceipt` keys, requires status
`ready`, validates every staged `files` entry, requires `scenes.json`, and checks the
inputs are current. `loadScenes` imports only the modules listed by `scenes.json`, which
must be sorted by unique id and match its `milestone` entries. The configured
`validators(page)` are called with the resolved page, and
`validateInventory(namespaces, stage)` returns the cumulative inventory rows. `check`
returns those rows and writes nothing.

**select** freezes the built page. It runs the same verification and writes
`<out>/selection.json` (`RECORD_FIELDS.selection`) with a new `selectionId`, the `stage`,
the receipt's `buildId` and `preparationId`, its `inputs` and the inventory rows. Later
stages compare the selection against this frozen copy.

**clean** allocates one immutable attempt and deletes nothing. It validates the selection
against the current build and inventory, creates `<out>/attempts/<uuid>/`, copies the
frozen `selection.json` and `build.json` into it, writes `attempt.json` with status
`allocated`, and writes `<out>/current.json`, the pointer `{ selectionId, attemptId,
directory }`. A pointer or an allocated attempt is never a pass.

**render** is resumable. It resolves `<out>/current.json`, the frozen selection, build and
attempt, requires the attempt to be untouched, and reserves the run. It publishes the page
once (`publishGpuPage`, transport timeout 300000 ms) and runs one managed operation per
`(scenario, mode)` group with input `{ selectionId, attemptId, buildId, scenario, mode,
width: 1280, height: 720, remainingMs: 270000 }`. Each returned capture is checked as a
PNG, saved as `images/<filename>`, and listed in `attempts/<uuid>/run.json` together with
the jobs, kept images, status and errors. The store is
`$NMC_CACHE_DIR/<topic>/<stage>` (see section 4); every job that passes is saved there in
the attempt layout, so another worktree keeps it. A group is kept from an earlier attempt
only when its rows, identities, job and image bytes still check out, and the kept images
are listed in `run.json` under `kept`. Jobs whose last dispatch failed (the store's
`failed-jobs.json`) run first, then never-rendered jobs, then the rest. A blank capture
fails the attempt before the next job is dispatched. `<envPrefix>_RERENDER_ALL=1` keeps
nothing.

**fresh** verifies the current attempt's real GPU evidence and writes
`attempts/<uuid>/fresh.json` (`RECORD_FIELDS.fresh`). It checks the frozen attempt and
current pointer, that `run.json` is complete and passed, one job per selected group, the
operation provenance and adapter, the image inventory and bytes, the scene's required
facts, the declared pair measurements, exact pair coverage, and empty console, page and
GPU errors. It never publishes or renders.

**judge** judges every selected capture with `NMC_JUDGE_CLIENT` and writes
`attempts/<uuid>/judge.json` (`RECORD_FIELDS.judge`). A byte-identical image with the same
description reuses an earlier passed verdict (exit 0, score at least 6), including one from
the store. New images are judged with the row's `expectedVisibleBehavior`. A row may carry two optional
fields, passed to the judge and recorded in each verdict:

- `judgeReference`: an absolute `.png` path. It must exist when the judge starts, or the row
  fails with `judge: reference <path> is missing` and no judge process starts.
- `judgeAspect`: `look`, `functional` or `design` (`JUDGE_ASPECTS`).

`judgeCapture(capture, description, { killAt, heldConstant, reference, aspect })` then adds
`--reference <reference>` and `--aspect <aspect>` before `--min-score 6`. A row without the
fields passes exactly the argv and options it always did. Each verdict records `reference`
and `aspect` (null when absent), and a verdict is reused (own attempt or earlier attempt)
only when its reference and aspect also match; an entry without those keys counts as null.
Calls start only inside the budget; images left when it ends are named, and the next
run judges only those. A bounded pool of `JUDGE_CONCURRENCY` runs the calls, and each call
is killed after the budget plus the grace. Verdicts are written in inventory order as they
settle.

The record files are `<page>/build.json` (the build receipt), `<out>/selection.json`, the
pointer `<out>/current.json`, and under `<out>/attempts/<uuid>/`: `selection.json`,
`build.json`, `attempt.json`, `run.json`, `fresh.json`, `judge.json`,
`jobs/<index>/{input,result}.json` and `images/<filename>` (`attemptRecords`).

## 3. The configuration object of `createPageJob`

`createPageJob(config)` validates the object and returns the job; an invalid field is
refused before any work.

| entry | type | default | effect |
|---|---|---|---|
| `name` | nonblank string | required | File stem and stdout prefix; used in the usage text and the commands of the page. |
| `topic` | nonblank string | required | `dist/<topic>` in the usage text and the store directory `$NMC_CACHE_DIR/<topic>/<stage>`. |
| `stages` | non-empty array of distinct nonblank strings | required | Allowed `--stage` values and the cumulative inventory stages; every `scenes.json` `milestone` must be one of them. |
| `envPrefix` | nonblank string | required | Prefix of `_RERENDER_ALL`, `_JUDGE_GRACE_MS`, `_JUDGE_CONCURRENCY` and `_JUDGE_BUDGET_MS`. |
| `validators` | function, possibly async | required | Called with the resolved page; returns `{ validateInventory, validateRequiredFacts }`, used by `check`/`select` and every per-job check. |
| `validateMeasurement` | function | required | Validates each measurement record returned by a job and its match with the variant declaration. |
| `keepSameBuildOnly` | boolean | `false` | When true, a kept group must come from an attempt whose `buildKey` equals the current build's, so a changed source rebuilds it. |
| `judgeHeldConstant` | boolean | `false` | When true, the row's `heldConstant` list is passed to the judge as `--held-constant`. |
| `report` | function `(action, stage, result, inventory) -> string[]` | the built-in `defaultReport` | Prints the action's stdout lines; `main` refuses anything but an array of strings. |

## 4. Environment names

- `NMC_CACHE_DIR`: the shared store root. When non-blank the store is
  `$NMC_CACHE_DIR/<topic>/<stage>`; unset or blank, it is the resolved `--out` directory.
- `NMC_JUDGE_CLIENT`: path of the judge client, required by `judge`; there is no default
  and a missing value fails with `NMC_JUDGE_CLIENT is missing`.
- `<envPrefix>_RERENDER_ALL`: the value `1` keeps nothing, so every group renders again;
  unset keeps resumable images.
- `<envPrefix>_JUDGE_GRACE_MS`: milliseconds added after the judge budget end before a
  running judge is killed; default `JUDGE_GRACE_MS` = 150000 ms, must be non-negative.
- `<envPrefix>_JUDGE_CONCURRENCY`: number of judge calls at once; default
  `JUDGE_CONCURRENCY` = 4, an integer from 1 to 16.
- `<envPrefix>_JUDGE_BUDGET_MS`: milliseconds during which judge calls may start; default
  `JUDGE_BUDGET_MS` = 480000 ms, must be non-negative. Judge's `--budget-s N` overrides it
  with `N * 1000` ms.

## 5. The two existing configurations

| | `web/common/sky-weather-job.mjs` | `web/common/vg-job.mjs` |
|---|---|---|
| `name` | `sky-weather-job` | `vg-job` |
| `stages` | `foundation`, `atmosphere`, `weather` | `bake`, `passes` |
| `topic` | `sky-weather` | `virtual-geometry` |
| `envPrefix` | `SKY_WEATHER` | `VG` |
| `validators` | static imports of `validateInventory` and `validateRequiredFacts` from `examples/sky-weather/lib/inventory.js` | loaded from the built page's `lib/inventory.js` (`validators(page)`), which must export both functions |
| `validateMeasurement` | `validateFeatureMeasurement` from `examples/sky-weather/lib/measure.js` | the same static import |
| `keepSameBuildOnly` | off (default) | `true` |
| `judgeHeldConstant` | off (default) | `true` |
| `report` | the built-in `defaultReport` | its own `report`: one `<row id> <row filename>` line per row on `check`, `vg-job render: kept <k>, rendered <r>` on `render`, then `vg-job <action>: ok (<stage>, <n> rows)` |

Sky-weather command line:

```sh
node web/common/sky-weather-job.mjs <check|select|clean|render|fresh|judge> --page dist/sky-weather --out out/sky-weather/<stage> --stage <foundation|atmosphere|weather>
```

Virtual-geometry command line:

```sh
node web/common/vg-job.mjs <check|select|clean|render|fresh|judge> --page dist/virtual-geometry --out out/virtual-geometry/<stage> --stage <bake|passes>
```

`judge` also accepts `--budget-s N`. A failure prints `<name>: <message>` on stderr and
exits 1.

## 6. How to add a page

1. Build the page under `dist/<topic>/` and write its `build.json` receipt (the eight
   `RECORD_FIELDS.buildReceipt` keys) and a sorted `scenes.json` as the last build step.
2. Provide the page's validators: a module exporting `validateInventory(namespaces,
   stage)` and `validateRequiredFacts(row, facts, factNames)`, either imported statically
   or loaded from the built page as `vg-job.mjs` does.
3. Provide a `validateMeasurement(measurement)` function.
4. Create `web/common/<name>.mjs` with `createPageJob`, export what the page needs, and end
   with `runCli(job, import.meta.url)`.
5. Register the page's command entries, letting the prefix drive the four environment
   names.

A configuration example:

```js
#!/usr/bin/env node
import { createPageJob, runCli } from './page-job.mjs';
import { validateInventory, validateRequiredFacts } from '../../examples/sky-weather/lib/inventory.js';
import { validateFeatureMeasurement } from '../../examples/sky-weather/lib/measure.js';

const job = createPageJob({
  name: 'my-page-job', topic: 'my-page', stages: ['base', 'passes'], envPrefix: 'MY_PAGE',
  validators: () => ({ validateInventory, validateRequiredFacts }),
  validateMeasurement: validateFeatureMeasurement,
});

export const { check, select, clean, render, fresh, judge, parseArgs, main, STAGES } = job;
runCli(job, import.meta.url);
```

## 7. Tests

- `web/common/sky-weather-job.test.mjs` pins the stage behaviour of the configuration: the
  build receipt and inventory, `select`, non-destructive `clean` and its pointer, the CLI,
  render with publication, captures and failures, freshness over kept and new images,
  required facts, measurements, pair coverage, the resumable store, blank captures,
  judge verdict reuse, concurrency, budget and dispatch order.
- `web/common/page-job.test.mjs` pins the configuration seam and the options: configuration
  refusals, `STAGES`/`USAGE`, `parseArgs`, `storeRoot`, `check`, `select`/`clean`/`render`
  records, the store save, `keepSameBuildOnly` (`false` by default), `judgeHeldConstant`,
  `report` and `runCli`.
- `web/common/vg-job.test.mjs` pins the virtual-geometry configuration: the stages and
  usage, `storeRoot`, the validators loaded from the built page, `readBuildReceipt` and the
  report lines.

Run them with:

```sh
node --test web/common/page-job.test.mjs web/common/vg-job.test.mjs web/common/sky-weather-job.test.mjs
```

## Follow-up: lights-job.mjs

`web/common/lights-job.mjs` (P1, documented in `docs/gpu-client.md`) is not migrated onto
`page-job.mjs` because its stages do not map without a behaviour change:

- It selects by milestone into `{ page, pairs, milestone }` and has no stage (`selectFor`).
- Its `clean` deletes the current attempt (`cleanFor`), where the page job never deletes.
- Its attempts live under `.lights-runs/<uuid>` behind `.lights-current.json`
  (`currentAttempt`).
- It renders every pair in one managed job (`renderFor`).
- Its build identity is the git commit (`buildIdentity`, `commitAt`).
- Its judge also judges the attached-light lifecycle captures (`ATTACHED_STAGES`,
  `judgeFor`).

A migration would have to decide the record layout of `out/lights`, how the `m2-lights-*`
command lines map onto the shared actions, whether each pair becomes its own job, and how
the lifecycle captures become inventory rows. It is not part of this milestone.
