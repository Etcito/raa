# tools/anim-ref — independent f64 S17/S16 reference

`gen-clip-samples.mjs` is an independent JavaScript reference for S17 sampling and S16 pose
matrices. It reads the four canonical inputs `rig-basic.json`, `clip-bezier.json`,
`clip-linear-constant.json` and `clip-rotation-flip.json` from
`crates/raa-anim/tests/fixtures/`, never imports Rust or wasm sampling results, and writes
`crates/raa-anim/tests/fixtures/samples.json`.

Regenerate with `node tools/anim-ref/gen-clip-samples.mjs`. The output is deterministic (no
timestamps, hashes or environment data), one compact JSON line plus the final LF. The generator
runs sequentially and starts no workers; any future independent-case workers must read NMC_CPUS (default 1) and never exceed it.

## Sampling (f64)

1. Address a tick: a looping clip uses `tick % lengthTicks`; otherwise the tick is unchanged.
2. Clamp to the first key before the first key and to the last key after the last key.
3. Between keys `a` and `b`, `u = (tick - a.tick) / (b.tick - a.tick)` in f64. `CONSTANT` returns
   `a.v`; `LINEAR` returns `a.v + (b.v - a.v) * u`; `BEZIER` is evaluated per component as
   `((1-u)^3)*p0 + (3*(1-u)^2*u)*p1 + (3*(1-u)*u^2)*p2 + u^3*p3` with `p0 = a.v`, `p1 = a.v + a.hr.dv`, `p2 = b.v + b.hl.dv`, `p3 = b.v`, in that order.
4. Rotation keys are sign-aligned to the previous key (`dot(a, b) < 0` negates `b`); a flipped key also flips its handle `dv` offsets. Handle `dt` is retained in the input and ignored by
   sampling. Rotation results are normalized with `sqrt`.
5. A bone without a track keeps its rest value. All canonical clips use `rootMotion: "none"`; another mode is refused by this fixture generator rather than sampled incompletely.

The record covers every key endpoint, three interior ticks per key interval (half and two quarters), ticks clamped before the first and after the last key, and `0`/`lengthTicks` loop
boundaries. The f64 values deliberately differ in the last bits from the runtime f32 path.

## Pose (S16)

Rest and sampled quaternions are normalized first. A bone model matrix is
`parent_model * TRS(local)`; `inverse_bind` is the analytic `S^-1 * R(conjugate) * T(-t)` composed with the parent inverse; `skin = model(pose) * inverse_bind`. Matrices are column-major f64 arrays
with the same product and sum order as the Rust reference.

## Record shape

`{format:"raa.anim.samples", version:1, precision:"f64", generatedBy, rig:{name,file},
clips:[{name,file,lengthTicks,loop,rootMotion,ticks,tracks:[{bone,channel,values}],poses}]}`.
`values` and `poses` are aligned with the clip's sorted `ticks`; each pose has `tick`, per-bone
`local` TRS, and column-major `model` and `skin` matrices.

## Sketchbook cross-check

When `SKETCHBOOK_DIR` is set (a directory containing `unified.ts`, or the file itself), the generator imports that actual source and compares every compatible track sample at every recorded
tick. Handles are passed as per-component `[dt, dv]` pairs, so nonzero ignored `dt` values are
exercised. Comparison uses the raw per-component sample before rotation normalization because
`sampleTrack` neither normalizes nor sign-aligns. An unreadable source, missing `sampleTrack`, bad
width or any component outside 1e-9 relative tolerance fails the run; nothing is skipped.

`AClip`/`ATrack` mapping: a clip maps to `{version:2, name, duration:lengthTicks, fps:24,
layers:[{id:"base", name:"base", factor:1, visible:true, collapsed:false, tracks}]}`; a track maps
to `ATrack {id:"<bone>:<channel>", label:"<bone>.<channel>", channelNames:["X","Y","Z"(,"W")],
target:{kind:"bone-rot"|"bone-pos"|"bone-scale", boneName:bone}, keyframes:[{time:tick, v:prepared
values, interp, hl/hr:[[dt,dv],...]}]}`. Rotation keys and their handle `dv` use the sign-aligned
runtime values prepared in step 4; the source `dt` stays in each pair.
