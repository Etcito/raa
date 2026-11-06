# RAA API m26: instance LOD, ray queries and statistics

The declarations are in `packages/raa/types/m26-instances.d.ts`; the Engine fragment is
appended to `packages/raa/index.d.ts` in sorted order by `scripts/build-package.sh`. A mesh
detail chain pairs each level's sub-meshes with the full mesh's sub-meshes by index.

```ts
interface Engine {
  meshLodError(full: MeshLodGeometry, level: MeshLodGeometry): number
  setMeshLods(mesh: number, levels: MeshLodLevel[]): void
  clearMeshLods(mesh: number): boolean
  setLodOptions(options: LodOptions): void
  setInstanceLodBias(instance: number, bias: number): void
  queryInstancesRay(origin: Vec3 | Float32Array, dir: Vec3 | Float32Array,
                    max?: number): InstanceRayHit[]
  instanceStats(): InstanceStats
}
```

## Methods

- `meshLodError(full, level): number` takes geometry objects with `positions: Float32Array`
  and `indices: Uint32Array`; normals and other attributes are not read. It returns the
  two-sided Euclidean vertex-to-triangle distance in object units: the largest distance from
  either mesh's vertices to the nearest triangle on the other mesh. Refusals: `lod.invalid`
  if either mesh has no triangles or any position is non-finite; `api.invalid_argument` for
  malformed geometry arguments; `api.disposed` after engine disposal.
- `setMeshLods(mesh, levels): void` registers 1..7 `{mesh, error}` entries, ordered from
  finest to coarsest. Errors must be positive, finite and strictly increasing. Each level
  must have the same number of sub-meshes as the full mesh, paired by sub-mesh index. Refusals:
  `lod.unknown_mesh` for an unknown mesh handle; `lod.too_many_levels` above seven coarser
  levels; `lod.invalid` for an invalid error, chain or sub-mesh layout; `api.invalid_argument`
  for malformed arguments; `api.disposed` after engine disposal. A refusal does not leave a
  newly registered partial chain.
- `clearMeshLods(mesh): boolean` removes all chains registered for the mesh's sub-meshes and
  returns `true` if at least one chain existed, otherwise `false`. Refusal: `lod.unknown_mesh`
  for an unknown mesh handle; `api.disposed` after engine disposal.
- `setLodOptions(options): void` accepts any subset of `{thresholdPx, hysteresis, bias,
  shadowBias}`. `thresholdPx` must be positive and finite; `hysteresis` is 0..0.9; `bias` and
  `shadowBias` must be finite. Omitted values keep their previous settings. Camera selection
  uses `thresholdPx`; shadow selection uses one shadow-map texel as its fixed threshold and
  `shadowBias` (default 0). Positive bias selects coarser levels sooner. Refusal:
  `api.invalid_argument` for malformed or out-of-range values; `api.disposed` after engine
  disposal.
- `setInstanceLodBias(instance, bias): void` sets a whole-number bias from -3 to 3; positive
  values select coarser levels sooner. Refusals: `api.invalid_argument` for a non-integer or
  out-of-range bias; `api.unknown_handle` for an unknown instance; `api.disposed` after
  engine disposal.
- `queryInstancesRay(origin, dir, max?): InstanceRayHit[]` accepts xyz arrays or
  `Float32Array`s. Each result is `{instance, t}`, where the ray is `origin + t * dir` and
  `dir` need not be normalized. It returns each intersected instance once, using its nearest
  hit, ordered by ascending `t` then instance handle. `max` is a whole number from 1 to 16384
  and defaults to 64. Refusal: `api.invalid_argument` for malformed vectors or `max`;
  `api.disposed` after engine disposal.
- `instanceStats(): InstanceStats` returns the most recently prepared frame's
  `{instances, camera, cascades, lods, uploadedBytes}`. `cascades` has four caster counts;
  `lods` has eight camera-level counts, for levels 0 through 7. Posed instances count at level
  0. `uploadedBytes` is the number of bytes uploaded for changed persistent instance pages.
  Refusal: `api.disposed` after engine disposal.

## Drawing instances at scale

Instance records persist at stable scene slots, and the renderer uploads only changed 4096 B
pages. The instance index chooses camera-visible instances and the casters in each shadow
cascade, so an off-screen tree can still cast a shadow. Each instance selects its own LOD by
projected geometric error with hysteresis. Level silhouettes are measured against the full
mesh using the two-sided vertex-to-triangle metric. Posed (skinned) instances always draw the
full-resolution mesh in the camera and every cascade. Shading stays the same across levels:
each uses the same material and per-pixel SDF field.
