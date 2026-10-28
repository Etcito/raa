# RAA API m18: per-instance GPU skinning (S18)

Additive to [m07 meshes and pipeline](m07-meshes-hdr.md). Declarations live in
`packages/raa/types/m18-skinning.d.ts`; `scripts/build-package.sh` appends them to
the package's `index.d.ts`, merging `MeshDesc` and `Engine`. The source facade also
exposes these additions. The consumed m07 declaration remains unchanged.

## Typed mesh inputs

`addMesh(desc)` and `updateMesh(mesh, desc)` accept the existing rest geometry plus:

```ts
interface MeshDesc {
  joints?: Uint16Array;   // 4n, S16 bone indices in vertex order
  weights?: Float32Array; // 4n, matching influence slots
}
```

Here n is the rest vertex count. Supply both arrays or neither; neither creates a
rigid mesh. Each vertex has four slots. Every index must be below 256, including
zero-weight slots. Weights must be finite and non-negative, with each vertex's
sum within 1e-4 of 1; upload validates rather than renormalises them. Unused slots
should use joint 0 and weight 0. Plain arrays and other typed-array kinds are
refused for these mesh fields. Existing S13 validation still applies.

`updateMesh` replaces skin attributes along with geometry and clears previous
poses on that mesh's instances; omitting both arrays removes skinning. A refused
description leaves the mesh unchanged. Posing never remeshes: topology, indices
and edge connectivity stay fixed.

## Poses and matrix layout

```ts
interface Engine {
  setSkinPose(instance: number, matrices: Float32Array): void;
  setSkinPoses(instances: Uint32Array, matrices: Float32Array): void;
}
```

These methods take instance handles, including those returned by `addInstance`
and `addMeshInstance`. Each instance owns its pose, even when instances share a
mesh; all sub-meshes of an instance use that pose. Calls replace pose data and
the next `render` draws it. An instance without a pose draws its rest mesh.

Each joint matrix is the S16 skin matrix `model(pose) * inverse_bind`, mapping
rest mesh space to posed mesh space. Build bone model matrices as
`parent_model * TRS(local)` in S16 bone order; use the rest model inverse for
`inverse_bind`. Do not include the instance's node/world transform: the renderer
applies that afterward. Matrices use column vectors and column-major storage:
16 floats per joint, columns consecutive, translation at offsets 12, 13, 14.
The input contains final skin matrices, not local TRS or raw bone model matrices.

The required joint count is **J = 1 + max(mesh.joints)** over all four slots per
vertex, even zero-weight slots. Supply exactly `16 * J` finite floats in joint
index order, including gaps. A larger rig's unused trailing matrices must be
trimmed; counts above 256 are refused. Identity matrices restore rest without
replacing the mesh; there is no public pose-clear method.

For a batch, concatenate slices in `instances` order, with no stride header or
padding. Slice i has `16 * J_i` floats and starts at `16 * sum(J_k, k < i)`;
total length is `16 * sum(J_i)`. Mixed joint counts are supported: J values 1
and 3 need slices `[0, 16)` and `[16, 64)`, for 64 floats total. The 256 cap is
per instance, not per batch. Repeated handles use their last slice. Every handle,
skin and matrix slice is validated before any pose update, so an invalid batch
leaves all poses unchanged.

## Feature toggling and posed overlays

The enabled-by-default compute pass `skin` runs at order 15, before shadow and
camera draws. Its feature-bit index is 20; existing S12 bits 0–19 retain their
meaning. Inspect it with `engine.passes()` and toggle with
`engine.setPassEnabled('skin', false)` / `true`. Keep it before draws when
customising pass order.

Disabling draws rest geometry with a valid frame; it retains stored poses, which
are recomputed when re-enabled. Pose setters can update stored poses while the
pass is disabled. Prepass, forward, shadows and transparent draws use each
instance's posed geometry, as do editor wireframe and selection outlines.
Wireframe edge segments follow the pose without changing edge connectivity.
Normals and tangent xyz use the blended matrix's upper 3x3 and are renormalised;
tangent handedness, UVs and colours stay unchanged. Culling uses conservative
posed bounds formed from the union of joint-transformed rest bounds.

## Errors

Engine validation throws `RaaError` with a stable `code`:

| Code | Cause |
|---|---|
| `api.invalid_argument` | Wrong typed-array kind for a mesh skin field. |
| `skin.length_mismatch` | Only one skin array supplied, or either length differs from 4n. |
| `skin.weights_sum` | Non-finite/negative weight or per-vertex sum outside tolerance. |
| `skin.joint_range` | Mesh joint index >= 256, or a single pose omits a referenced joint. |
| `skin.cap_joints` | A single pose contains more than 256 joint matrices. |
| `skin.pose_length` | Empty, truncated, non-finite or extra single-pose data; wrong batch total or non-finite batch slice. |
| `skin.not_skinned` | Instance's mesh has no skin attributes. |
| `api.unknown_handle` | Unknown or removed instance (or mesh for mesh operations). |

All methods also reject after disposal (`api.disposed`). Feature installation or
frame reentrancy failures retain `api.feature_missing` or `pipeline.busy`.
Pipeline toggle/order errors follow m07. A failed pose validation changes nothing;
an empty array is not a request to clear a pose.

## Two instances, one mesh, different poses

```ts
import { loadRaa } from '@etcito/raa';
const raa = await loadRaa();
const engine = await raa.createEngine(document.querySelector('canvas')!);
const material = engine.addMaterial({ color: [0.3, 0.7, 1] });
const mesh = engine.addMesh({
  positions: new Float32Array([-0.4, 0, 0, 0.4, 0, 0, 0, 1, 0]),
  normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1]),
  indices: new Uint32Array([0, 1, 2]),
  joints: new Uint16Array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
  weights: new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]),
});
const left = engine.addInstance(engine.addNode(null, [-0.7, 0, 0]), mesh, material);
const right = engine.addInstance(engine.addNode(null, [0.7, 0, 0]), mesh, material);
const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
function pose(dx: number): Float32Array {
  const data = new Float32Array([...identity, ...identity]); // J = 2
  data[16 + 12] = dx; // joint 1 moves the tip in mesh-space X
  return data;
}
engine.setSkinPose(left, pose(-0.25)); // right still rests
engine.setSkinPoses(new Uint32Array([left, right]),
  new Float32Array([...pose(-0.25), ...pose(0.25)])); // 32 + 32 floats
engine.setWireframe('overlay');
engine.setSelectionOutline(right, { color: [1, 0.6, 0.1] });
engine.setCamera({ eye: [0, 1, 4], target: [0, 0.5, 0] });
engine.render(0);
```
