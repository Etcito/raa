# RAA API m31: temporal anti-aliasing

Temporal anti-aliasing (TAA) is the pipeline pass `taa` (S43). It is configured through the
existing pass controls (`passes`, `setPassEnabled`); there is no settings object and no new
method or declaration. `packages/raa/index.d.ts` and `docs/API.md` stay v0. Everything in
`docs/API.md` and the earlier `docs/api/mNN-*.md` stays valid.

## Rendering model

| order | name | role | what it does |
|---|---|---|---|
| 100 | `taa` | pass (compute) | Resolves the jittered frame against a ping-pong history after `particles.blend` (85) and before `dof` (110). |

The pass runs at render resolution 1:1: it reads `hdr.src()`, the prepass depth and motion
targets and the frame uniform, and writes `hdr.dst()` and its own `Rgba16Float` history, then
swaps. It is installed disabled.

### Disabling with `setPassEnabled`

- `setPassEnabled('taa', true)`: the pass enables; the frame uniform carries the camera jitter
  and accumulation is on from the next frame.
- `setPassEnabled('taa', false)`: the pass stops resolving and the frame is byte-identical to a
  frame without a temporal consumer. The history is dropped; re-enabling starts a fresh one.

The codes are `pipeline.unknown_pass` and `pipeline.required_pass`; `passes()` lists the row in
execution order like any other pass.

## Camera jitter

While `taa` is enabled the pass applies an 8-sample Halton(2, 3) sub-pixel jitter to the
projection, moving the sample within a pixel. The jitter is applied only while the pass is
enabled: with it disabled the projection and the frame uniform are unchanged (S43). A change of
consumer invalidates the camera history of the next frame.

## Motion and history

- Motion comes from the prepass motion target: the current unjittered NDC minus the previous
  position, as a UV displacement.
- Sky pixels (depth 1) are reprojected from the camera direction through the previous
  view-projection, so a moving camera keeps the background stable.
- History is rejected for instances without object history (new, re-posed or skinned instances,
  and vertex-function materials): their motion is camera-only, so they cannot smear.

## Per-viewport history

Each split viewport owns its own `taa` instance and history (`crates/raa-js/src/api/m07_pipeline.rs`
`ViewPost`), so views never share accumulated frames. A camera cut on one viewport invalidates
only that viewport's history.

## Invalidation

The history is dropped on:

- resize;
- disabling the pass;
- `setCamera` with different values or `setOrbit` (a camera cut);
- a viewport camera change (`setViewportCamera`);
- incomplete frames (a view not submitted this invocation);
- a viewport camera cut, which calls the view's `TaaHandle::invalidate`.

`orbit()` keeps the history: it is a continuous camera move, not a cut.
