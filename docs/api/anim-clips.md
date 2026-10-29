# Rig loading and clip sampling

The animation wasm brick exports `loadRig(json)`, `loadClip(rig, json)` and
`sampleClip(clip, tick)`. JSON inputs are strings in S16 `raa.rig` v1 and S17
`raa.clip` v2 format. Validation, rotation preparation, analytic inverse binds
and sampling are delegated to `raa-anim`; this API does not reinterpret documents.

```js
const rig = anim.loadRig(rigJson);
const clip = anim.loadClip(rig, clipJson);
rig.free(); // The clip retains its own reference to the rig.
const { local, model, skin, rootDelta } = anim.sampleClip(clip, 48);
clip.free();
// Returned arrays remain usable after freeing the handles.
```

Handles are opaque wasm objects, not numeric IDs or JSON. Do not use a handle
again after `free()`. A clip remains tied to the rig used to validate it.
`types/clips.d.ts` describes the additive `RaaAnimClipsApi` surface, which can be
intersected with `RaaAnimApi` from `types/brick.d.ts`.

## Validation and limits

`MAX_BONES = 256`: exactly 256 bones is allowed if all S16 rules hold. More than
256 is rejected with `rig.cap_bones`, never truncated. Parent ordering, one root,
unique names, rest transforms, roles, IK chains, limits and attachments retain
S16 validation. No ragdoll fields are accepted.

Inclusive `CLIP_CAPS` are 256 tracks per clip, 4096 keys per track and 1024 events
per clip. Exceeding any count gives `clip.cap`, without truncation. There is no
additional `lengthTicks` cap; S17's numeric representation, integrality, range,
and key/event timing constraints still apply. Rig-name matching, track targets,
unique bone/channel pairs, key order, channels, interpolation and handles are
validated by `raa-anim`.

Document and sampling refusals throw `Error` with the original `code`, `path`
(JSON pointer), and `message` from `raa-anim` (`json.*`, `rig.*`, or `clip.*`).
`sampleClip` rejects non-number, non-finite, fractional, negative or above-u32
ticks with `TypeError`, `code: "api.invalid_input"`; it never coerces or wraps
JavaScript ticks. Valid ticks are integers from 0 through 4294967295.

## Samples

Each call returns a plain object containing fresh `Float32Array` copies:

| Field | Layout |
| --- | --- |
| `local` | 10 floats per bone: t.xyz, r.xyzw, s.xyz |
| `model` | 16 column-major floats per bone: parent model * local TRS |
| `skin` | 16 column-major floats per bone: model * analytic inverse bind |
| `rootDelta` | 3 floats: extracted translation delta xyz |

All bone arrays use S16 bone order. Untracked channels retain rest values.
Sampling uses deterministic Rust f32 arithmetic. CONSTANT, LINEAR and BEZIER
retain S17 evaluation order; Bezier dt handles are ignored. Rotation keys are
sign-aligned at load and sampled quaternions normalized. Loop poses address
`tick % lengthTicks` when length is positive, with track values clamped before
the first and after the last key. The API does not dispatch clip events.

## Stateless root motion

`rootMotion: "none"` gives a zero delta. For `xz` or `xyz`, selected root local
translation axes are restored to their rig rest values; unselected components
remain sampled normally. The root is the bone with no parent. Extraction is
independent of call order: previous means tick minus one, not the last API call.

Delta is zero at tick 0. For positive loop length L, let P(s) be selected root
translation sampled without wrapping on [0,L], including the actual endpoint.
With q = floor(t/L) and r = t mod L:

```
U(t) = P(r) + q * (P(L) - P(0))
delta(t) = U(t) - U(t-1), for t > 0
```

The implementation uses equivalent adjacent local differences without
accumulating q in f32. Every seam yields P(L)-P(L-1), while the pose wraps to
tick 0. Non-looping deltas use clamped adjacent ticks and are zero after the
endpoint. Any zero-duration clip permitted by S17 produces zero delta.

## Selftest integration

`clips::register` is discovered by the existing build registry and registers
the brick's `selftest("clip-sample", seed, ticks)` case. The native worker is
`clips::clip_sample_selftest(seed, ticks)`. Inputs retain the dispatcher's u32
checks; `det-selftest` keeps its existing behavior.

The test loads `rig-basic` and the linear/constant, Bezier and rotation-flip
fixtures. For each offset in `[0,ticks)`, it samples all three clips in that
order at `seed.wrapping_add(offset)`, computes root deltas, checks finite output,
and feeds skin matrices in bone/component order into DET_HASH. Zero ticks
samples all three fixtures once at `seed`, through the same runtime path.
The returned record is exactly
`{case, seed, ticks, hash}`, with an eight-digit lowercase hexadecimal hash.
The hash is API data; acceptance executes cases and does not compare hashes
across native, Node or browser targets. Canonical cases are `(0,0)`, `(1,60)`
and `(305419896,600)`.

`cargo test -p raa-anim-js clip_sample` executes these inputs and checks sampled
local transforms and root deltas against the committed independent
f64 reference at its listed ticks (or its clamped endpoint), within 1e-6.
Every sampled model and skin matrix must be finite. Registry coverage runs with
`cargo test -p raa-anim-js --test api_registry`. Neither check compares hashes.

Browser operation integration uses the existing
`window.nmcGpu.ops.run({case, seeds, ticks})` on `examples/anim-brick` after a
local brick build. Both `det-selftest` and `clip-sample` accept the same request
shape: a nonempty array of u32 seeds and a u32 tick count. Other cases and
invalid inputs are rejected before dispatch. Invoke the canonical cases
separately, in order:

```js
for (const [seed, ticks] of [[0, 0], [1, 60], [305419896, 600]]) {
  const result = await window.nmcGpu.ops.run({
    case: 'clip-sample', seeds: [seed], ticks,
  });
  console.log(result.results); // [{case: 'clip-sample', seed, ticks, hash}]
}
```

Successful execution and the returned record shape are the checks; retain the
hash as API data.

The managed capture command is `node scripts/anim-capture.mjs clip-sample`
(`--case clip-sample` is equivalent). It requires `NMC_GPU_CLIENT`, publishes
`dist/anim-brick/` after staging the built brick and shared page helper, and calls
the declared `render` operation separately for the three canonical requests above.
It first publishes and runs a blank diagnostic page to check the client protocol.
The zero-tick request then runs first; any failure stops the sequence. Each client call
has a 60-second limit. Software/fallback adapters and malformed records fail.
Only after all requests succeed does it write
`proofs/results/anim/clip-sample.json`, preserving the capture envelope
`{adapter, hardware, adapterText, records}` and each dispatcher's exact
`{case, seed, ticks, hash}` record in canonical order. No hash values are compared.
`node scripts/anim-capture.mjs --case det-selftest` retains its existing local
Chrome runner, stdout envelope and exit codes.

`.nmc/commands.json` prepares the build and capture for programmatic execution.
The build sets `npm_config_cache=/tmp/raa-5426-npm-cache` because the default
cache is read-only. Model cards do not execute GPU jobs. Client-GPU capture
acceptance remains pending until that execution succeeds; no evidence record is
created by this card. The diagnostic operation returns JSON and no screenshot.
`node --test scripts/anim-capture.test.mjs` covers CLI parsing/usage, staging,
output location and failure handling with a stubbed client in temporary
directories. It also exercises page validation and real staged WASM dispatch
without GPU setup. Stub records are never acceptance evidence.
