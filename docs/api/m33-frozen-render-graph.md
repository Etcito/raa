# RAA API m33: frozen render graph (S60)

Additive to [m32 render graph](m32-render-graph.md). Declarations live in
`packages/raa/types/m33-frozen-render-graph.d.ts`; `scripts/build-package.sh` appends them to the
package's `index.d.ts`, merging `Engine` and `EngineOptions`. The method is in
`crates/raa-js/src/api/m33_frozen_render_graph.rs`.

```ts
interface EngineOptions {
  pipeline?: string
}
interface Engine {
  setFrozenRenderGraph(name: string): RenderGraphResult
}
```

## Frozen pipelines

A frozen pipeline is a starter graph written as Rust literals: `crates/raa-graph/src/frozen/`
holds one module `<name>.rs` (with `pub fn plan() -> GraphPlan`) and the committed source
`<name>.json` (`SOURCE_JSON`). The `raa-graph` build registers all of them in `frozen_all()`.
The three starters are `simple_forward`, `full_gi` and `debug`; `renderGraphCatalog().frozen`
lists the names this engine knows.

The plan is applied **as written**. The method does not replan the frozen execution: it reads the
literal plan from the registry, refuses a stale literal, and installs it through
`GraphExecutor::set_plan` with the parsed, canonical saved source as the active document. A frozen
pipeline is therefore owned by the render graph exactly like an interpreted one: the graph owns the
live pipeline while it is set, and `setRenderGraph(null)` restores the configuration from before
the first graph.

## `setFrozenRenderGraph(name)`

`setFrozenRenderGraph(name: string): RenderGraphResult` selects the registry entry `name` and
applies it. On success `ok` is true, `issues` holds warnings only and `rebuilt` counts the features
the plan rebuilt; `renderGraphActive()` is true and `renderGraph()` returns the saved document
canonicalised in the S50 key order. Nothing changes before the selection succeeds.

Codes:

| code | stage | when |
|---|---|---|
| `graph.frozen_unknown` | plan | no frozen pipeline has this name |
| `graph.frozen_stale` | plan | a plan node's type is missing from the catalog, or its generated version differs from the catalog's |
| `api.invalid_argument` | – | `name` is not a string (thrown) |
| `api.disposed` | – | the Engine was disposed (thrown) |

A refusal returns `{ ok: false, issues }` and keeps the previous graph (or the no-graph
configuration) rendering; the issues are also what `renderGraphIssues()` reports. A stale literal
is never upgraded: the node type's upgrade function is not run and the plan is not edited.

## `createEngine(canvas, { pipeline })`

`createEngine(canvas, { pipeline: '<name>' })` removes `pipeline` from the options, creates the
engine, then calls `setFrozenRenderGraph` once. A refusal disposes the new engine and rejects the
promise with the issue codes; without `pipeline` the call keeps its v0 behaviour. Engine creation
itself is unchanged (`EngineOptions.pipeline` is the only added option); the engine's msaa and
viewport count still constrain a graph that the engine cannot run.
