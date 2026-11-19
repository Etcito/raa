// S60 additions: frozen render-graph pipelines. See docs/api/m33-frozen-render-graph.md.
// scripts/build-package.sh appends this file to index.d.ts, and scripts/typecheck-types.mjs
// stages it the same way; the `EngineOptions` and `Engine` interfaces below merge with the v0
// declarations of the same names.

/**
 * Options of {@link RaaModule.createEngine}. Every field is optional.
 */
export interface EngineOptions {
  /**
   * Name of a frozen render-graph pipeline to install after the engine is created, for example
   * `simple_forward`, `full_gi` or `debug` (S60). The wrapper removes this key before calling the
   * Engine constructor, then calls {@link Engine.setFrozenRenderGraph}; a refusal disposes the new
   * engine and rejects the promise with the issue codes. Absent = no pipeline, and the promise
   * keeps today's behaviour.
   */
  pipeline?: string;
}

export interface Engine {
  /**
   * Applies the frozen pipeline `name` as a literal plan through the executor. The active document
   * is the parsed, canonical saved source, so {@link Engine.renderGraphActive} is true and
   * {@link Engine.renderGraph} returns that document. An absent name reports
   * `graph.frozen_unknown`; a frozen node type the catalog lacks or a node generated for another
   * type version reports `graph.frozen_stale`, without upgrading the literal. A refusal leaves the
   * previous graph (or the no-graph configuration) rendering. Returns a fresh
   * `RenderGraphResult`; throws `api.invalid_argument` for a non-string and `api.disposed` after
   * dispose.
   */
  setFrozenRenderGraph(name: string): RenderGraphResult;
}
