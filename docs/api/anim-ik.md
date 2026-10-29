# Visual inverse kinematics

The animation brick's `ik` API applies visual IK to an S17 pose using an
`AnimationRig` loaded with `loadRig`. It returns a corrected pose and rebuilt
S16 model and skin matrices. Declarations are in
`packages/raa-anim/types/ik.d.ts` as `RaaAnimIkApi`; intersect this additive
surface with `RaaAnimApi` and the clips API. See S16 for rig chains and limits.

The caller supplies targets derived from deterministic simulation state, such
as contact points, grab points, and look targets. IK is a visual correction:
the simulation never consumes a corrected IK pose. Inputs are read only, and
each call returns fresh arrays; calls do not mutate the input pose or rig.

## Lookdev integration

Lookdev pages initialize the animation package and provide it as `lib.anim` to
scene `build` and `frame` functions. The M20 scene imports its metadata without
initializing WASM; during `build`, it uses `lib.anim.loadRig` and
`lib.anim.ik`. Its `ik-on` variant applies those corrections, while `ik-off`
uses the same rig, sampled source pose, mesh, targets, camera, and lighting
without applying IK. Targets still come from simulation state, and corrected
skin matrices are used only for rendering.

## Pose and coordinates

Pass either an S17 `local` array or an object containing `local`. It must be an
Array or `Float32Array` with exactly 10 values per rig bone, in S16 bone order:
translation xyz, quaternion xyzw, scale xyz. A clip sample can be passed
directly. Targets and ground-hit points are three-component vectors in the
world/model coordinate frame of the pose, before the scene instance transform.
The returned `local`, `model`, and `skin` are `Float32Array` copies in rig bone
order. Matrix data is 16 column-major floats per bone; `skin` is
`model(pose) * inverse_bind`, ready for the S18 skinning API.

All changed rotations respect the rig's S16 XYZ Euler-degree limits relative
to rest. Unchanged bones retain their input values. `twoBone` uses the chain's
configured pole when `pole` is omitted; `null` disables it. An explicit pole
is a vector in the root bone's parent space. Pole direction selects the bend
side. Targets are world-space positions. `lookAt.axis` is a direction in the
bone's local coordinates; `lookAt` turns it toward the target.

## Solvers

All methods take `(rig, pose, ...)` and return `IkPoseOutput`.

- `twoBone(rig, pose, chain, target, pole?, weight?)` requires an S16
  `twoBone` chain of exactly three root-to-tip bones. `weight` defaults to 1
  and is clamped to [0, 1].
- `ccd(rig, pose, chain, target, iterations?, tolerance?)` and
  `fabrik(...)` require their matching S16 solver chain, with 2–8 direct
  parent-to-child bones. Iterations default to `ikChains[].iterations` and
  must be an integer from 0 through 16. Tolerance defaults to 0.001 world
  units; it must be finite, and negative values act as zero. Iterative output
  includes `converged`, `passes`, and final `distance` in world units. These
  report the solve result; an unconverged solve still returns its best pose.
- `footPlacement(rig, pose, legs, hits)` takes one named `twoBone` leg per
  `{point, normal}` hit. The arrays must have equal lengths; legs must be
  distinct and share a pelvis. Normals are normalized and must be finite,
  nonzero, and face upward (+Y). The solver plants each foot using its input
  height, aligns it to the hit normal, and may lower the shared pelvis (never
  raises it); `drop` reports that world-unit displacement. Empty arrays yield
  an unchanged pose and zero drop.
- `handReach(rig, pose, chain, target, weight?)` uses the named chain's S16
  solver, with weight default 1 and clamped to [0, 1]. Weight 0 preserves the
  pose; iterative chains use a fixed 1e-4 world-unit stopping distance.
- `lookAt(rig, pose, bone, target, axis, maxDeg?)` rotates the named bone
  toward the target by at most `maxDeg`, default 180 degrees. The cap is in
  degrees (not radians); negative values act as zero. A zero axis or target at
  the bone origin leaves the pose unchanged.

## Results and errors

Every output has `local`, `model`, and `skin`. Only iterative solvers add
`converged`, `passes`, and `distance`; only foot placement adds `drop`. Arrays
are fresh and can be retained after later calls. Input conversion/type errors
throw `TypeError` with `code: "api.invalid_input"` and `path: "/pose"`.
Solver errors are `Error` objects carrying `code`, `path`, and `message`:

| Code | Meaning |
| --- | --- |
| `ik.chain` | Missing/incompatible chain, invalid iteration count above 16, or invalid leg pairing/hit count. |
| `ik.target_not_finite` | Non-finite target, pole-derived solve value, weight, tolerance, angle, or invalid ground hit point/normal. |
| `ik.bone_unknown` | Named bone (or a referenced chain bone) does not exist. |

Finite weights are clamped rather than refused. A malformed pose length,
non-vector target/axis/hit, or non-number argument is rejected during JS input
conversion as `api.invalid_input`.

## Apply a corrected pose to a skinned instance

The example samples animation first, derives the hand target from simulation
state, solves visual reach, then passes the corrected skin matrices to the
existing S18 `setSkinPose`. If a mesh uses only a prefix of the rig's joints,
trim to the mesh's required joint count (`1 + max(mesh.joints)` over all four
slots, including zero-weight slots):

```js
const sampled = anim.sampleClip(clip, tick);
// The simulation owns this target; IK output is never fed back to simulation.
const grabTarget = simulation.grabPoint();
const corrected = ik.handReach(rig, sampled, 'armR', grabTarget, 0.8);

// meshJointCount is 1 + max(mesh.joints), as required by S18.
const meshSkin = corrected.skin.slice(0, 16 * meshJointCount);
engine.setSkinPose(instance, meshSkin);
engine.render(timeMs);
```

This composes with `setSkinPose` because both use S16 skin matrices in bone
order, 16 column-major floats per joint, mapping rest mesh space to posed mesh
space. Do not pass `model` matrices or apply the scene instance transform to
these matrices; the renderer applies that transform after skinning.
