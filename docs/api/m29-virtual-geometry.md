# RAA API m29: virtual geometry

The declarations are in `packages/raa/types/m29-virtual-geometry.d.ts`; the package build
appends them to `packages/raa/index.d.ts`. Virtualized geometry draws the clustered mesh of a
requested instance through the cull, raster, resolve, G-buffer and shade passes instead of the
forward path. The cluster data comes from the [m29 cluster bake](m29-cluster-bake.md) API.

```ts
type VirtualGeometryDebugView = 'none' | 'clusters' | 'lod' | 'raster' | 'occlusion'
interface VirtualGeometrySettings { /* the patch: every key optional */ }
interface ResolvedVirtualGeometry { /* every key */ }
interface InstanceVirtualGeometry { requested; path; pending; code; message }
interface VirtualGeometryStats { /* counters of the last published readback */ }
interface Engine {
  setVirtualGeometry(settings: VirtualGeometrySettings): void
  virtualGeometry(): ResolvedVirtualGeometry
  setInstanceVirtualGeometry(instance: number, enabled: boolean | null): void
  instanceVirtualGeometry(instance: number): InstanceVirtualGeometry
  virtualGeometryStats(): VirtualGeometryStats
}
```

## Settings

`setVirtualGeometry(settings)` merges a `VirtualGeometrySettings` patch onto the current values.

| key | type | default | domain |
|---|---|---|---|
| `enabled` | boolean | `false` | `true` needs an Engine created with `msaa: 1` |
| `errorPixels` | number, pixels of screen-space error | `1` | finite, in [0.25, 64] |
| `softwareRaster` | boolean | `true` | small clusters may go to software raster |
| `occlusion` | boolean | `true` | `false` runs the first cull phase only |
| `shadows` | boolean | `true` | virtualized casters enter the four cascades |
| `runtimeBake` | boolean | `true` | a requested instance with no cluster data starts a bake |
| `instanceDefault` | boolean | `true` | the request of an instance whose value is `null` |
| `maxVisibleClusters` | number, clusters | `262144` | integer in [1024, 4194304] |
| `debugView` | `VirtualGeometryDebugView` | `'none'` | one of the five views below |

Merge rules:

- An omitted key (`undefined`) keeps its value.
- `null`, a value of the wrong type, an unknown key, a non-integer `maxVisibleClusters`, an
  unknown `debugView` or an out-of-domain value throws `vg.invalid` with the message
  `<key>: <reason>`, for example `errorPixels: must be finite in [0.25, 64]`.
- A non-object argument throws `vg.invalid` with the message `virtualGeometry: expected an object`.
- A refusal changes nothing: the other keys of the same patch are not applied.
- `errorPixels` is the screen-space threshold of the level of detail, in pixels of screen-space
  error. A cluster's error is in metres of object space; it is projected to pixels before the
  comparison.

`virtualGeometry()` returns a fresh `ResolvedVirtualGeometry` with every key of the table.

- `enabled` toggles the five passes `vg.cull`, `vg.raster`, `vg.resolve`, `vg.gbuffer` and
  `vg.shade` through `setPassEnabled`. With `enabled` off no vg pass is recorded and no vg
  buffer or texture exists.
- `enabled: true` needs an Engine created with `msaa: 1` (VG_MSAA): virtualized pixels are
  single-sample. Otherwise `setVirtualGeometry` throws `vg.invalid` with the message
  `msaa: virtualized geometry needs msaa 1`. There is no separate code, and the settings stay
  unchanged.
- Changing `enabled` resets the occlusion history.

## Debug views

| `debugView` | shows |
|---|---|
| `none` | the shaded frame |
| `clusters` | a hashed colour per cluster |
| `lod` | a colour per DAG level |
| `raster` | hardware or software raster of each cluster |
| `occlusion` | the clusters recovered by the second occlusion phase |

## Path selection

An instance takes the virtual path in a submitted frame only when all of these hold:

1. `enabled` is true;
2. the five vg passes are enabled, through `setPassEnabled`;
3. the invocation submits exactly one viewport;
4. the instance qualifies: its request resolves to true and it has no refusal.

Otherwise it takes the forward path. `inactiveReason` in the stats reports the first failing
condition in this order: `disabled`, `passDisabled`, `multiView`, `noInstance`.

## Instance requests and codes

`setInstanceVirtualGeometry(instance, enabled)` sets the request of one instance group:
`true`, `false`, or `null` for the default request (`instanceDefault`). `false` is never refused.

The three instance codes:

| code | instance |
|---|---|
| `vg.skinned_unsupported` | skinned |
| `vg.deformed_unsupported` | deformed |
| `vg.transparent_material` | transparent material |

- An explicit `true` on such an instance throws its code from `setInstanceVirtualGeometry`.
- Under the default request (`null`) the same instance is not thrown. The code is reported as
  state in `instanceVirtualGeometry(instance).code` with its message, and the path is `forward`.
- Skinned, deformed, transparent and fur instances stay on forward beside virtualized
  instances, correctly depth-sorted with them.

The mesh codes `vg.mesh_vertex_colors`, `vg.mesh_empty`, `vg.mesh_too_large` and `vg.invalid`
come from the bake of a mesh's clusters. An automatic runtime bake (`runtimeBake` on) records
the code in `meshClusters(mesh)`; each instance that uses the mesh reports it in its `code` and
`message`, and is not drawn virtually. Mesh codes are not counted in `refused`. An explicit
`bakeMeshClusters` call throws the same codes.

`instanceVirtualGeometry(instance)` returns a fresh `InstanceVirtualGeometry`:

- `requested`: the explicit request, or `null` when the instance follows `instanceDefault`.
- `path`: `virtual` or `forward`, for the last submitted frame.
- `pending`: true when the instance is requested and eligible, but at least one of its meshes
  has no ready cluster data yet: state `none` (a runtime bake is started when `runtimeBake` is
  true) or `baking`. Such an instance draws on forward meanwhile and switches to `virtual` once
  every mesh is `ready`. `pending` is false on `virtual`, for a non-requested instance and for a
  refused instance.
- `code` and `message`: the refusal, or `null`. They are `null` when the instance is pending,
  virtual or not requested.

## Single view

Virtualized geometry draws one viewport. An invocation that submits more than one view runs
every instance on the forward path, and the stats report `inactiveReason` `multiView`.

## Vertex format limits

The virtual path reads the 16 B packed vertex of the cluster section. Therefore:

- there is no tangent: material functions receive the S10 default tangent of the normal;
- there are no vertex colours: a mesh with a colour stream is refused with
  `vg.mesh_vertex_colors`.

## Stats

`virtualGeometryStats()` returns a fresh `VirtualGeometryStats`, with the keys in this order:

| key | type and meaning |
|---|---|
| `active` | boolean; true when the path is taken |
| `inactiveReason` | `disabled`, `passDisabled`, `multiView`, `noInstance` or `null` (exactly when active) |
| `virtualInstances` | instance groups on the virtual path |
| `forwardInstances` | instance groups on the forward path |
| `refused` | fresh object: requested instance groups refused per instance code; absent codes are not listed |
| `bakingMeshes` | distinct meshes of the listed instances in state `baking` |
| `readyMeshes` | distinct meshes of the listed instances in state `ready` |
| `drawnClusters` | clusters, main view, both occlusion phases |
| `drawnTriangles` | triangles, main view, both occlusion phases |
| `hardwareClusters` | the part of `drawnClusters` rasterised in hardware |
| `softwareClusters` | the part of `drawnClusters` rasterised in software |
| `shadowClusters` | clusters drawn into the four cascades, summed; 0 when none is drawn |
| `droppedClusters` | clusters dropped at `maxVisibleClusters`, main view and cascades |
| `statsFrameIndex` | the Engine render invocation the counters come from |
| `pending` | true while a readback is outstanding |

The counters are read back asynchronously:

- The counters and `statsFrameIndex` are `null` before the first published readback.
- `pending` is true while a readback is outstanding.
- While inactive, the latest published values are kept.
- A readback is published only after its frame is completely submitted.
- Stats are never a GPU fence: no call waits for the GPU.

## Resize and camera cuts

Resizing the surface, changing `enabled`, and a camera discontinuity of view 0 reset the
occlusion history: the first frame after one of them tests against no previous depth.

## Disposal

`dispose()` drops the virtualized-geometry feature with the Engine state. Every later call on
the disposed Engine throws `api.disposed`.

## Not provided

- Skinned, deformed, transparent and fur instances: they stay on the forward path.
- Streaming or paging of cluster data.
