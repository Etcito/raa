# Freezing a render graph to Rust (S60)

A frozen pipeline is a `raa.render-graph` document whose `GraphPlan` is written as Rust literals.
It is the same plan the interpreter produces, executed through `GraphExecutor::set_plan`; there is
no special executor. `simple_forward`, `full_gi` and `debug` were frozen first, from
`crates/raa-graph/graphs/simple-forward.json`, `full-gi.json` and `debug.json`. A later freeze is
**one card**: it adds `crates/raa-graph/src/frozen/<name>.rs` and `<name>.json` and runs the
existing validation. It edits no registry, `lib.rs`, `build.rs`, API file or test.

## Procedure

Write `<name>` for the freeze name and `<graph.json>` for the source document.

1. **Export and validate the input.** Export the graph from the editor (`exportGraph`, 2-space
   JSON plus LF) or take a starter from `crates/raa-graph/graphs/`. The generator plans it with
   `plan_from_json` (the catalog, `GraphLimits::CORE`, every pass installed, one viewport, msaa 4),
   so a document that does not validate and plan is refused with its issue codes, nodes, ports,
   parameters and fixes. Fix the document first.
2. **Choose the name.** It must match RG_FREEZE_NAME, `^[a-z][a-z0-9_]{0,47}$`: lower-case start,
   then lower-case letters, digits and `_`, at most 48 characters (`simple_forward`, not
   `Simple-Forward`). The name is also the file stem and the registry key.
3. **Copy the source to its sibling JSON.** The module includes it with
   `include_str!("<name>.json")`, so the file must sit beside the module:

   ```
   cp <graph.json> crates/raa-graph/src/frozen/<name>.json
   ```

   For a starter this is the starter's text (`graphs/simple-forward.json` becomes
   `frozen/simple_forward.json`). The copy is committed unchanged: it is the source of truth that
   the plan is checked against.
4. **Invoke the actual generator.** It prints the module to stdout:

   ```
   cargo run -p raa-graph --bin raa-graph-freeze -- crates/raa-graph/graphs/simple-forward.json simple_forward
   ```

   Redirect to a temporary file, then move it, so a refusal never leaves an empty module:

   ```
   cargo run -q -p raa-graph --bin raa-graph-freeze -- <graph.json> <name> > /tmp/<name>.rs
   mv /tmp/<name>.rs crates/raa-graph/src/frozen/<name>.rs
   ```

   The module holds `pub const SOURCE_JSON`, one `WGSL_<NODE_ID_UPPER>` constant per compute node
   and `pub fn plan() -> GraphPlan` with every node, link, resource and parameter as literals
   (parameters in catalog key order, defaults written out, `view.*` nodes that do not reach
   `output.display` dropped). No local path appears in it.
5. **Format the generated module.** The generator's text is not rustfmt output:

   ```
   rustfmt --edition 2024 --config skip_children=true crates/raa-graph/src/frozen/<name>.rs
   ```

6. **Registry discovery is automatic.** `crates/raa-graph/build.rs` lists `src/frozen/*.rs` by
   file stem in C-locale order and writes `frozen_all()` in `$OUT_DIR/registry.rs`: `(name, plan,
   SOURCE_JSON)`. Cargo reruns the build script when the directory changes. `raa_graph::frozen_plan(name)`,
   `raa_graph::frozen_names()` and `renderGraphCatalog().frozen` then include `<name>`.
7. **Validate equality and versions.** Run the existing test:

   ```
   cargo test -p raa-graph --test frozen
   ```

   It asserts, for every `frozen_all()` entry, that `plan_from_json(SOURCE_JSON) == plan()` and that
   every node's `type_version` equals the catalog's current version. The same test covers the
   `graph.frozen_unknown` and `graph.frozen_stale` selection refusals. Finish with `cargo check`.
8. **Rebuild the package** so the new plan reaches the WASM and `@etcito/raa`:
   `bash scripts/build-package.sh`.
9. **Select the frozen pipeline.**
   - JavaScript: `createEngine(canvas, { pipeline: '<name>' })`, or on a live engine
     `engine.setFrozenRenderGraph('<name>')`, which returns `{ ok, issues, rebuilt }`. The factory
     removes `pipeline` before creating the engine, calls `setFrozenRenderGraph` once, and on a
     refusal disposes the engine and rejects with the issue codes and messages.
   - Rust: `let plan = raa_graph::frozen_plan("<name>")`, then `GraphExecutor::set_plan(plan, doc,
     handles)`. The JS API (`select_frozen` in `crates/raa-js/src/api/m33_frozen_render_graph.rs`)
     does the same after the stale check. See [m33](../api/m33-frozen-render-graph.md).

## Refusals

- `graph.frozen_unknown` (plan stage): no frozen pipeline has that name. Nothing is applied.
- `graph.frozen_stale` (plan stage): a plan node's type is missing from the catalog, or its
  generated `type_version` differs from the catalog's. The literal is **never upgraded**; the node
  type's `upgrade` function does not run on it. `cargo test -p raa-graph --test frozen` fails for
  the same reason.

**A changed node version requires regeneration.** When a node type's version rises (a changed
default or meaning), the committed literal and its JSON are stale. Regenerate from the sibling
JSON (the source is preserved, not the plan): run steps 4 and 5 with
`crates/raa-graph/src/frozen/<name>.json` as `<graph.json>`, then run the frozen test. The
interpreter upgrades the document's older `v` with a warning; the regenerated plan carries the new
version.

**Source preservation.** The generator only reads its input and never writes `<name>.json`. A
frozen module is replaced only by regenerating it; the JSON stays what the author exported.

**Atomic refusal.** The generator checks the name and plans the whole input before it prints
anything. A refused name, unreadable file or invalid graph writes diagnostics to stderr, exits
nonzero and leaves stdout empty, so no partial module exists (hence the temporary file in step 4).
At run time a refused selection changes nothing: the previous graph, or the no-graph
configuration, keeps rendering.

## Paired render requirement

A freeze is accepted only with a paired render: the frozen pipeline and the interpreted graph of
the same source render the same RG_RUN frame, side by side (`frozen-<name>-left-interpreted-right`
in the `frozen` stage of the render-graph page). Each half runs on its own freshly created engine
(`isolatedPair` of `examples/render-graph/lib/runtime.js`, which builds isolated pipeline engines).
Each half's capture-time facts are immutable (deep-frozen at
capture): the graph frame index (240), the 241 render and camera times, and the seeds (weather
seed, weather changing, time playing, fresh engine with zero earlier renders, source). The halves
must carry identical frame and seed evidence or the pair fails. The evidence is judged like every
render of the graph; there is no pixel metric, hash or CPU/GPU comparison.
