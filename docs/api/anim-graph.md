# RAA API: the S19 animation graph

The animation graph loads `raa.animgraph` v1 documents (S19) against the rig and clip
handles of the [clip API](anim-clips.md) and steps them one deterministic tick at a
time. The runtime is `crates/raa-anim-graph` (sim path, raa-det arithmetic); the
JavaScript binding is `crates/raa-anim-js/src/api/graph.rs`. Declarations are in
`packages/raa-anim/types/graph.d.ts` as `RaaAnimGraphApi`, which the brick build
aggregates into `types/all.d.ts`. Intersect it with `RaaAnimApi`, as `RaaAnimClipsApi`
is intersected for clips.

Access: the generated module `pkg/raa_anim_js.js` exports `loadRig`, `loadClip`,
`loadGraph`, `AnimationGraph` and `AnimationGraphState`. The `loadRaaAnim()` facade
currently returns only `brickInfo` and `selftest`, and the package `exports` map
exposes only `.` and the wasm file. The graph calls are therefore not yet reachable
from a published package import.

## Loading

```js
const rig = loadRig(rigJson);
const clips = [loadClip(rig, idleJson), loadClip(rig, waddleJson)];
const graph = loadGraph(graphJson, rig, clips);
rig.free();        // the graph keeps its own copy of the rig and clips
clips.forEach((clip) => clip.free());
```

`loadGraph(json, rig, clips)` borrows the rig and clip handles, so they stay usable
after loading. Each graph clip reference names a clip by its S17 `name`, and those
names must be unique across the supplied handles. All clips must belong to `rig`.
Refusals throw `Error` with `code`, `path` (JSON pointer) and `message`.

## Document (S19)

Top-level fields are `format` (`raa.animgraph`), `version` (1), `name`, `rig`, `params`,
`blendSpaces`, `layers`, `machines` and `rootMotion`. Unknown fields are refused.

- `params`: `{name, type, default}` where `type` is `float`, `int`, `bool` or `trigger`.
  Triggers take no default.
- `blendSpaces`: `{name, params, points, sync}`. One parameter gives a 1D space over
  neighbouring points. Two give a 2D space: barycentric weights in the containing
  Delaunay triangle, precomputed at load, with the nearest hull edge outside the hull.
  Each point is `{clip, at}`.
- `layers`: `{name, machine, mask, weight, additive}`. `mask` is `null` or a list of
  bones.
- `machines`: `{name, entry, states, transitions}`. A state is `{name, motion, loop,
  speedParam}`, where `motion` names a clip or a blend space. A transition is `{from,
  to, when, blendTicks, exitTick, priority}`, where `from` may be `*`.
- `when` conditions are `{param, op, value}`. Operators are `trigger`, `==`, `!=`, `>`,
  `<`, `>=` and `<=`. `trigger` applies only to trigger parameters.
- `rootMotion`: `{bone, mode}` with `mode` as `none`, `xz` or `xyz`.

## Typed parameters

`graph.step(state, params)` takes a plain object keyed by declared parameter name.

| Declared type | JavaScript value |
| --- | --- |
| `float` | finite number, stored as f32 |
| `int` | integer number in the i32 range |
| `bool` | `true` or `false` |
| `trigger` | `true` pulses for this tick; `false` is ignored |

`params` may be omitted or `null`. An omitted name keeps its last applied value, and
a trigger applies to exactly one tick unless it is given again. A trigger is consumed
only by the transition that fires on it. Pending pulses survive failed conditions and
exit gates. An unknown name, a value of the wrong type, or a non-integer `int` throws
`graph.param`, with path `/params` for names and `/params/<index>` for typed values.
A parameter object that is not a plain object throws `TypeError` with code
`api.invalid_input`.

## Tick order

Each `graph.step` runs these phases in order. A refused call changes nothing.

1. Parameters: trigger pulses fold into the pending state, and float, int and bool
   values are read as given.
2. Transitions: each machine evaluates once, current state first and then `*`, by
   ascending `priority` and then file order. The first match fires.
3. Motion: every layer not entered this tick advances its current phase by this tick's
   speed. Fading sources stay frozen.
4. Crossfades: the destination weight is `elapsed / blendTicks`.
5. Layers: composed in document order over the rig rest pose, with masks, weights and
   additive deltas.
6. Outputs: the root delta of the clip ticks crossed this tick, extracted axes restored
   to rest, and events.
7. Matrices: S16 model and skin matrices for the composed local pose.

Transitions use the pre-tick local clock. `exitTick` gates inclusively, and `null`
means no gate. A non-transitioning state's clock advances by one each tick, and a
destination starts at clock and phase zero. A transition never chains into another
within the same tick.

## Tick and fade boundaries

A transition that fires at tick `t` enters its destination at clock and phase zero,
with a new fade at elapsed zero. Its entry tick therefore shows the frozen source. The
fade reaches weight one `blendTicks` calls later and is then removed. A zero
`blendTicks` switches immediately. Interrupting a fade discards the previous fade and
starts from the current destination at its current phase, even if the old fade had not
completed.

An entered state reports a zero root delta and its tick-0 events. A fading source
keeps its phase, so it moves nothing while it fades out.

## Phase synchronization

- An unsynchronized blend space samples each point's clip at the state's clip tick.
  Its period is the lcm of the point lengths for looping states (the longest length
  when not looping or above 2^24).
- A `sync` blend space uses one normalized phase in [0, 1] shared by all points. Per
  tick it advances by `speed / D`, where `D = sum(weight * L)`, and each point samples
  `floor(phase * L)`. The weight changes never move the phase.
- Each tick a state advances by `speed` ticks, where `speed` is `speedParam` when set
  and 1 otherwise. A zero period or `D` keeps progress at 0.
- Looping states wrap progress modulo the period. Others clamp it.

## Masks and additive layers

Layers apply in document order. A mask of `null` selects every bone. A list selects the
named bones and all their descendants, expanded at load. Unknown bone names are
refused with `graph.mask`. Composition starts from the rig rest local pose, so
a bone that no layer selects keeps its rest value. A masked-out bone keeps whatever
earlier layers left on it.

A non-additive layer blends toward its sampled pose by its weight: translation and scale
component-wise, rotation by shortest-path nlerp. An additive layer adds its delta from
the motion's first-key reference, which is the same motion sampled at progress zero.
Its rotation delta is `sample * conjugate(reference)`, pre-multiplied in the parent
frame.

## Root motion

`rootMotion.bone` names the bone whose translation track gives per-tick deltas. In
`xz` mode the X and Z axes are removed from the pose and reported as the delta; `xyz`
removes all three. Removed axes return to rig rest, and other components keep their
samples. `rootDelta.r` is always identity.

Loop seams are crossed through the unwrapped P(L), so the delta over a seam is
continuous, and a non-looping clip clamps to its last key. Deltas, not positions, are
blended by blend weight, then by crossfade weight, so a frozen source moves nothing and
a weight change never jumps.

## Events

Events come from clip keys crossed in `(previous sample, current sample]`. Each machine
reports them once per tick from its first weighted layer, using the dominant
contribution of its current state (the first on ties), in playback direction. Looping
states deliver each event at most once per tick. Events are ordered by layer, then by
crossing offset, then by clip file order. Each event is `{name, tick, layer, clip}`,
where `layer` and `clip` are zero-based indices and `tick` is the clip tick.

## Stepping and snapshots

```js
const state = graph.newState();
const out = graph.step(state, { speed: 1.5 }); // {skin, rootDelta: {t, r}, events}
const bytes = state.save();   // RGS2 snapshot as a Uint8Array
state.restore(bytes);         // replaces the state only if the whole snapshot validates
```

`out.skin` holds 16 floats per rig bone in bone order (the S18 `setSkinPose` layout).
Snapshots bind the graph's canonical text and asset indices, but not rig or clip data,
so restoring needs the same assets. An incompatible snapshot or a state from another
graph throws with `graph.schema` at `/state`. In every refusal the live state is unchanged.
Events need no separate cursor: the tick, local clocks and phases encode them. Rollback
is save, restore and replay, which reproduces the same outputs.

## Caps (GRAPH_CAPS)

Limits are inclusive. Excess is refused with `graph.cap`, never truncated. S19 names
GRAPH_CAPS without numbers, so these are implementation limits. They are not exposed
to JavaScript.

| Limit | Value |
| --- | --- |
| `max_params` | 64 |
| `max_blend_spaces` | 64 |
| `max_points_per_space` | 64 |
| `max_layers` | 16 |
| `max_machines` | 16 |
| `max_states_per_machine` | 256 |
| `max_transitions_per_machine` | 1024 |
| `max_conditions_per_transition` | 8 |
| `max_mask_bones` | 256 (S16 `MAX_BONES`) |

## Errors

| Code | Refused when |
| --- | --- |
| `graph.format` | `format` is not `raa.animgraph` |
| `graph.version` | `version` is not 1 |
| `graph.schema` | JSON shape, fields or values are invalid; a snapshot or state is foreign or incompatible (`/state`) |
| `graph.param` | a parameter name is unknown or its value has the wrong type |
| `graph.clip_unknown` | a clip name is missing, duplicated or for another rig, or a track bone is not in the rig |
| `graph.state_unknown` | an entry, transition or machine names an unknown state or machine |
| `graph.blend_space` | a blend space is invalid, such as degenerate 2D points |
| `graph.mask` | a mask or `rootMotion.bone` names an unknown bone |
| `graph.cap` | a count exceeds GRAPH_CAPS |

Rig, clip and JSON failures keep their `rig.*`, `clip.*` and `json.*` codes. Errors
are `Error` objects with `code`, `path` and `message`. A non-array `clips`, a
non-handle element or a non-object `params` throws `TypeError` with `api.invalid_input`.

## Runnable graph-run example

`graph-run` runs the shared 600-tick penguin script from `scenario_data.rs` and its
timeline in `crates/raa-anim-graph/src/fixtures/penguin.timeline.json`. Seed is record
metadata only. Ticks above 600 run 600, and 0 runs none. A rollback mismatch aborts the
call instead of returning a record.

```js
import { loadRaaAnim } from '@etcito/raa-anim';

const anim = await loadRaaAnim();
const record = anim.selftest('graph-run', 0, 600);
console.log(record); // { case: 'graph-run', seed: 0, ticks: 600, hash: '<8 hex digits>' }
```

The script's key moments are speed 1.5 at tick 40 and 4 at tick 100, a jump at 150, a
dive at 240, a grab at 360, a climb at 420, a hand push from 470 to 520, and emotes at
300, 450, 550 and 580. `cargo test -p raa-anim-graph --test scenario` checks that the
base layer enters 13 states, the emote layer 5, 49 clip events fire, the root travels
about 16.725 along +Z in `xz` mode, and a rollback at tick 151 replays 448 ticks
identically. The hash digests the full ordered trace. It is API data, not a
cross-platform acceptance value.
