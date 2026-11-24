# RAA API m32: render graph

The declarations are in `packages/raa/types/m32-render-graph.d.ts`. A render graph defines which
lit-pipeline passes run, in which branches, with every parameter, live on one view. Scene content
(meshes, lights, materials, sky, time of day, weather, cameras) is never in a graph. The document
format is `raa.render-graph` version 1 (S50); the catalog is `raa.render-graph-catalog` version 1
(S51) and is the one source of node types, ports, parameters and defaults. While a graph is set,
the graph owns the live pipeline; `setRenderGraph(null)` restores the configuration from before
the graph.

```ts
interface RenderGraphNode { id: string; type: string; v?: number; params?: Record<string, number | boolean | string | number[]>; inputs?: RenderGraphComputePort[]; outputs?: RenderGraphComputePort[]; math?: unknown; wgsl?: string }
interface RenderGraphComputePort { key: string; type: 'image2d' | 'depth2d' | 'buffer'; format?: string; size?: { scale: number } | { width: number; height: number } }
interface RenderGraphLink { from: string; to: string; history?: boolean }
interface RenderGraphDoc { format: 'raa.render-graph'; version: 1; name: string; nodes: RenderGraphNode[]; links: RenderGraphLink[]; editor?: unknown }
interface RenderGraphIssue { code: string; severity: 'error' | 'warning'; nodeId?: string; port?: string; param?: string; link?: number; message: string; fix: string }
interface RenderGraphResult { ok: boolean; issues: RenderGraphIssue[]; rebuilt: number }
interface RenderGraphStats { frameIndex: number | null; nodes: { id: string; type: string; ran: boolean; resources: { port: string; width: number; height: number; layers: number; mips: number; format: string; bytes: number }[] }[] }
interface Engine {
  renderGraphCatalog(): RenderGraphCatalog
  setRenderGraph(graph: RenderGraphDoc | string | null): RenderGraphResult
  renderGraph(): RenderGraphDoc
  renderGraphActive(): boolean
  renderGraphIssues(): RenderGraphIssue[]
  renderGraphStats(): RenderGraphStats
  renderGraphOwnedSetters(): string[]
}
```

## Engine methods

| method | returns | refusals |
|---|---|---|
| `renderGraphCatalog()` | a fresh `RenderGraphCatalog` | `api.disposed` |
| `setRenderGraph(graph)` | a fresh `RenderGraphResult`; invalid graphs are reported in `issues`, never thrown | throws `api.invalid_argument`, `api.disposed` |
| `renderGraph()` | a fresh `RenderGraphDoc` | `api.disposed` |
| `renderGraphActive()` | boolean | `api.disposed` |
| `renderGraphIssues()` | a fresh `RenderGraphIssue[]` | `api.disposed` |
| `renderGraphStats()` | a fresh `RenderGraphStats` | `api.disposed` |
| `renderGraphOwnedSetters()` | a fresh `string[]` | `api.disposed` |

### `renderGraphCatalog()`

`renderGraphCatalog(): RenderGraphCatalog` returns the S51 JSON of every node type the engine
knows, in catalog order (category rank geometry, lighting, gi, shadows, post, compute, view,
output; then `defaultOrder`; then `id` bytewise). `categories` lists the eight category names in
rank order and `portTypes` the seven port-type names; `graphVersion` is 1. A node whose `pass` is
not installed in this engine has `available: false`. Every default is the feature descriptor
default, read through the `get()` of the matching feature, never a retyped number. There are no
parameters and no defaults of the call itself; the returned object is fresh, so changing it
changes nothing.

### `setRenderGraph(graph)`

`setRenderGraph(graph: RenderGraphDoc | string | null): RenderGraphResult` validates the document,
plans it and installs it atomically. A `string` is parsed as UTF-8 JSON first. On success `ok` is
true, `issues` holds only warnings and `rebuilt` counts the features the graph rebuilt. On a
refusal `ok` is false, `issues` holds the S52 errors and the previous graph (or the no-graph
configuration) keeps rendering. Nothing changes before validation finishes.

Defaults of the document: `version` 1, `name` 1..64 characters, an absent node `v` means the
catalog's current version, an absent `params` key takes the catalog default, an absent `history`
is false, an absent `editor` is not written back. RG_CAPS: up to 256 nodes, 1024 links, 1048576
bytes of document text and 65536 bytes of `wgsl` per node.

Codes: a value that is not an object, a string or `null` throws `api.invalid_argument`; a disposed
Engine throws `api.disposed`. A refused graph returns any of the S52 codes below, each carrying its
`nodeId`, `port`, `param` or `link`; `setRenderGraph` never throws for an invalid graph.

### `renderGraph()`

`renderGraph(): RenderGraphDoc` returns the active document canonicalised in the S50 key order
with its parameters as given; `editor` is returned verbatim. With no graph set it returns the
export of today's configuration: one node per enabled pass and contributor in execution order,
type `pass.<entry>`, id the entry name in lower camel case (`ao.gtao` -> `aoGtao`,
`sky.cloud-lighting` -> `skyCloudLighting`, `shadow.csm` -> `shadowCsm`), every parameter
written; links are the colour chain between consecutive enabled chain passes plus
`tonemap.color -> out.color`, and there is one `output.display` node `out`. G-buffer and slot
inputs stay unlinked (default sources). The exported document passed back to `setRenderGraph` is
accepted and leaves the configuration unchanged.

### `renderGraphActive()`

`renderGraphActive(): boolean` is true while a graph is set. It is false when no graph was ever
set or after `setRenderGraph(null)`; a refused `setRenderGraph` keeps the previous answer.

### `renderGraphIssues()`

`renderGraphIssues(): RenderGraphIssue[]` returns the issues of the last accepted or refused
`setRenderGraph` (errors and warnings, in S52 order), and an empty list before any call. Viewports
turned on while a graph is active keep today's per-view post pipelines and append a
`graph.multi_view` warning here.

### `renderGraphStats()`

`renderGraphStats(): RenderGraphStats` returns the S53 `NodeStats` of the latest rendered frame:
per node `id`, `type`, `ran`, and per output port `port`, `width`, `height`, `layers`, `mips`,
`format` and `bytes`. `frameIndex` is the Engine render invocation; it is null before the first
frame.

### `renderGraphOwnedSetters()`

`renderGraphOwnedSetters(): string[]` returns the setters the active graph owns, in this order:
`setPassEnabled`, `setPassOrder`, `setToneMapping`, `setShadows`, `setAo`, `setContactShadows`,
`setBloom`, `setTransparency`, `setSsr`, `setDof`, `setFog`, `setSunShafts`, `setGrading`,
`applyEnvironment`, `setCel`, `setVirtualGeometry`, `setAreaLightShading`,
`setAreaLightShadows`, `setGlobalIllumination`, `setGiQuality`, `setSphericalGi`. It returns that list whether or
not a graph is set.

## Runtime lighting

- `pass.gi.spherical` runs the existing camera-centred capture pass on the shared mesh BVH.
  Its `lookup` buffer feeds `pass.forward.sphericalLookup` through the forward lighting
  bind group. This link accepts the engine's spherical lookup only and resolves by default
  when the spherical node is present. The lookup has no compute or history tap. Capture
  uses the primary view and the scene's geometry and lights.
- The spherical node exposes `sphereSubdivision`, `exponentialPower`, `clipStart`, `zMax`,
  `channelMask` and the `debug` views from `setSphericalGi`. Node presence enables capture.
  Set `zMax` above `clipStart`. The `spherical_gi` frozen graph uses the same pass and handles.
- `pass.gi.volume.volumeAtlas` feeds `pass.gi.screen.volumeAtlas`. Set the screen node's
  `indirect` to `probes` or `restir` and use its `debugView` for volume and screen diagnostics.
  Both modes publish `pass.gi.screen.gi` for `pass.forward.gi`. Spherical GI can coexist with
  the existing volume and screen passes. Forward shading blends the spherical result with
  its existing diffuse fallback according to spherical coverage.
- `pass.ao.gtao` owns `radius`, `intensity`, `power`, `slices`, `steps` and
  `indirectDiffuse`. The last setting uses the same diffuse AO shader control as `setAo`.
- `pass.forward.lightingOutput` selects `combined`, `direct` or `indirect` in the raster
  shader. Use the mesh reference node's `lightingOutput` and `lightingDepth` for an exact
  bounce, since the forward pass has no per-bounce signal.

## Document format

Top-level keys, a closed set:

| key | type | meaning |
|---|---|---|
| `format` | `'raa.render-graph'` | the format name |
| `version` | `1` | the format version |
| `name` | string, 1..64 characters | the graph name |
| `nodes` | `RenderGraphNode[]` | the nodes |
| `links` | `RenderGraphLink[]` | the links |
| `editor` | any JSON value, optional | editor layout; raa ignores it and `renderGraph()` returns it verbatim |

Node keys, a closed set: `id` (`/^[A-Za-z][A-Za-z0-9_]{0,63}$/`, no `.`, unique), `type` (a
catalog node-type id), optional `v` (integer >= 1: the node-type version the parameters were
written for), optional `params` (catalog parameter keys of the type), and for the type
`compute.wgsl` also `inputs`, `outputs`, `math` and `wgsl`.

Link keys, a closed set: `from` and `to`, each `"<nodeId>.<portKey>"` split at the first `.`
(port keys `/^[a-z][A-Za-z0-9]{0,31}$/`), and optional `history` (boolean, default false). `from`
names an output port and `to` an input port; an input port takes at most one link.

Parameter values by kind: `number` a JSON number finite as f32; `int` an integer in
[-2^31, 2^31); `bool`; `enum` one of the parameter's option strings; `vec2`/`vec3`/`vec4` arrays
of 2/3/4 numbers finite as f32; `string`. Colours are `vec3` linear RGB with unit `linear-rgb`;
the kind `color` is never used. An absent parameter takes the catalog default.

Canonical text is pretty JSON with the keys in the orders above, nodes and links in document
order, node parameters in catalog key order, plus a final LF. `renderGraph()` returns the active
document canonicalised with its parameters as given; the no-graph export writes every parameter.

Versions: a node whose `v` is lower than the catalog version runs the type's upgrade once per
version step and adds the warning `graph.node_upgraded`; an `v` higher than the catalog version is
`graph.node_version`. An unknown `format`, `version`, node type, port or parameter is a refusal
naming it.

Example (the Simple forward starter, abbreviated):

```json
{ "format": "raa.render-graph", "version": 1, "name": "Simple forward",
  "nodes": [ { "id": "shadowCsm", "type": "pass.shadow.csm" }, { "id": "prepass", "type": "pass.prepass" },
    { "id": "lightTiles", "type": "pass.light.tiles" }, { "id": "forward", "type": "pass.forward" },
    { "id": "skyBackground", "type": "pass.sky.background" },
    { "id": "tonemap", "type": "pass.tonemap", "params": { "mapper": "agx", "exposureEv": 0 } },
    { "id": "out", "type": "output.display" } ],
  "links": [ { "from": "prepass.depth", "to": "lightTiles.depth" }, { "from": "shadowCsm.shadowMap", "to": "forward.shadowMap" },
    { "from": "lightTiles.lightTiles", "to": "forward.lightTiles" }, { "from": "forward.color", "to": "tonemap.color" },
    { "from": "tonemap.color", "to": "out.color" } ] }
```

## Issue codes

Stages rank parse 0, structure 1, budget 2, plan 3, apply 4. A stage that holds an error stops
validation after that stage, and every error of the stage is reported. Issues are sorted by stage
rank, then the node's document index (issues without a node first), then the link index (none
first), then `code` bytewise. `message` states what is wrong with the actual values; `fix` states
one concrete action.

| code | stage | fields | condition |
|---|---|---|---|
| `graph.json` | parse | – | the text is not JSON; the message comes from the JSON parser |
| `graph.format` | parse | – | `format` absent or not `raa.render-graph` |
| `graph.version` | parse | – | `version` is not 1 |
| `graph.schema` | parse | – | unknown, missing or mistyped key; the message names the JSON pointer |
| `graph.too_large` | parse | – | RG_CAPS exceeded |
| `graph.node_id` | structure | `nodeId` | invalid or duplicate id |
| `graph.node_type` | structure | `nodeId` | unknown node type |
| `graph.node_version` | structure | `nodeId` | `v` newer than the catalog, or an upgrade failed |
| `graph.param_unknown` | structure | `nodeId`, `param` | unknown parameter key |
| `graph.param_kind` | structure | `nodeId`, `param` | wrong parameter kind |
| `graph.param_range` | structure | `nodeId`, `param` | outside `min`/`max` or the options |
| `graph.port_unknown` | structure | `nodeId`, `port`, `link` | a link names an unknown port |
| `graph.link_type` | structure | `nodeId`, `port`, `link` | output and input types differ, or the output format is not accepted by the input |
| `graph.input_taken` | structure | `nodeId`, `port`, `link` | a second link into one input |
| `graph.input_missing` | structure | `nodeId`, `port` | a required input unlinked, or no default source present and not `absentOk` |
| `graph.cycle` | structure | `nodeId` | a cycle not broken by a history link; the first node of the cycle in document order |
| `graph.history` | structure | `link` | a history link into a port without `history`, or out of a non-image port |
| `graph.output` | structure | – | zero or several `output.display` nodes |
| `graph.required_node` | structure | – | `pass.prepass`, `pass.forward` or `pass.tonemap` missing; the message names it |
| `graph.single_instance` | structure | `nodeId` | a second node of a single-instance type |
| `graph.fixed_wiring` | structure | `nodeId`, `port`, `link` | a link the current wiring cannot honour |
| `graph.unavailable` | structure | `nodeId` | the node's pass is not installed in this engine |
| `graph.multi_view` | structure | – | more than one viewport is active |
| `graph.wgsl` | structure | `nodeId` | a compute node's WGSL fails validation; the message carries the user line |
| `graph.budget` | budget | `nodeId` | a stage's bindings exceed RG_LIMITS; the message names stage, binding kind, count and limit |
| `graph.engine_option` | plan | `nodeId`, `param` | an Engine-bound parameter differs from the engine; the fix names the createEngine option |
| `graph.order` | apply | – | the pipeline refused the order |
| `graph.apply` | apply | `nodeId`, `param` | a feature refused a value; the message comes from the feature |
| `graph.frozen_unknown` | plan | – | unknown frozen name |
| `graph.frozen_stale` | plan | – | a frozen node version differs from the catalog |
| `graph.unreachable` (warning) | plan | `nodeId` | the node reaches neither `output.display` nor a side-effect node; it is not executed |
| `graph.node_upgraded` (warning) | structure | `nodeId` | parameters upgraded from an older `v` |

## Ownership

While a graph is active a call to one of the owned setters is validated and accepted exactly as
today, but the graph owns the live pipeline: at the next frame the executor restores the graph's
values and keeps the call's result in the no-graph configuration that `setRenderGraph(null)`
restores. With no graph set the owned setters behave exactly as before. The owned setters, in the
order of `renderGraphOwnedSetters()`:

1. `setPassEnabled`
2. `setPassOrder`
3. `setToneMapping`
4. `setShadows`
5. `setAo`
6. `setContactShadows`
7. `setBloom`
8. `setTransparency`
9. `setSsr`
10. `setDof`
11. `setFog`
12. `setSunShafts`
13. `setGrading`
14. `applyEnvironment`
15. `setCel`
16. `setVirtualGeometry`
17. `setAreaLightShading`
18. `setAreaLightShadows`
19. `setGlobalIllumination`
20. `setGiQuality`

Scene setters (`setSky`, `setTimeOfDay`, `setWeather`, `setEnvironmentLighting`, lights, meshes,
cameras) are never owned.

## `null` and the no-graph configuration

On the first activation a graph takes a snapshot of the no-graph configuration: the enabled pass
set, the pass order and every owned feature value read through its getter. `setRenderGraph(null)`
clears the graph, restores that snapshot in feature-rank order and returns
`{ ok: true, issues: [], rebuilt }`: the engine then renders exactly as an engine that never had a
graph. A refused `setRenderGraph` never takes or restores the snapshot; the previous graph keeps
rendering.

## Single view: `graph.multi_view`

A render graph drives one view. `setRenderGraph` with more than one active viewport is refused
with `graph.multi_view` and changes nothing. Viewports turned on while a graph is already active
keep today's per-view post pipelines and add a `graph.multi_view` warning to
`renderGraphIssues()`.

## Starters

`renderGraphCatalog().starters` returns `{ name, file, json }` entries: `json` is the complete
document text, `name` its display name and `file` the catalog-relative file it was loaded from.
Pass `json` (or the parsed object) to `setRenderGraph` to install a starter.

| Starter | Pipeline |
|---|---|
| Area inference pilot | Stratified area visibility wired through the area mask. The engine can apply a compatible registered pilot package to this branch. |
| Custom denoise | Full GI stages with a custom WGSL denoiser on the GI resolve branch. |
| Debug | A grid of render taps including depth, normals, AO, area shadow, GI and final color. |
| Full GI | Volume and screen irradiance, area-light shadows, screen-space GI, contact shadows, fog and TAA. |
| Full GI stages | Explicit probe, accumulation, filtering, area shadow and screen-space GI stages. |
| Real-frame neural reconstruction | Area, SSR incident radiance, temporal resolve and frame-generation routing. Empty model IDs keep native or analytic processing. Frame generation also needs a compatible presentation host and opt-in. |
| Simple forward | Shadow cascades, prepass, light tiles, forward shading, sky background and AgX tone mapping. |

`presets` lists the compute-node presets and `frozen` the names of the frozen pipelines for the
same installation.

### Experimental neural routing

Area reconstruction consumes and produces stratified visibility pairs between
`area.trace.raw` and `area.mask.src`. Reflection reconstruction consumes and
produces RGBA32F incident radiance between `ssr.incident.incident` and
`ssr.replace.incident`. Its fitted head refines accepted screen hits and
preserves the surrounding native signal and confidence. Temporal SR owns the
scene-resolution color, depth, motion and previous display history at the
temporal resolve boundary. These three nodes share the recurrent features
when wired together and require the same registered package ID. Register
the package with `registerNeuralModel` before selecting the graph. Each
selected head must appear in the package's `trainedHeads`;
`neuralReconstructionStats` reports selected and executed heads separately.

`neural.frame-generation.color` connects to the same real HDR endpoint as
the tonemap input. It is a presentation branch with no real-frame output or
history link. A saved graph can store the ID of a package registered with
`registerGeneratedFrameModel`. An empty ID selects analytic reprojection.
The Game camera profile uses a retained native HDR/depth image and the
motion derived from both cameras. It requires a static scene, a contained
target frustum, the same camera eye and valid render-graph history. The
GPU decoder only refines the accepted warp. The host presents an eligible
candidate before its deadline or uses the available real image.
`generatedFrameModelStats` counts learned work and learned presentations.
The fitted weights were trained on causal target-guide actor motion.
The camera profile uses optical guides and remains experimental. Physical
input-to-display latency and camera quality require separate qualification.
