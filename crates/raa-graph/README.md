# raa-graph

The render graph of the lit pipeline: the `raa.render-graph` document format (S50),
the node catalog (S51), validation (S52), the plan (S53) and the executor that applies
a plan to the pipeline and its feature handles. Scene content (meshes, lights,
materials, sky, time of day, weather, cameras) is never part of a graph.

The crate is device-free except for the executor: `raa-pipeline` never depends on it.

## Layout

- `src/catalog.rs`: the node-type, port, parameter and budget types of the catalog.
- `src/format.rs`: the `GraphDoc`, `NodeDoc`, `LinkDoc` and compute declaration types.

## Extension directories

The crate grows by new files, never by editing an earlier milestone's file:

- `src/nodes/`: node families, one file each.
- `src/exec_ext/`: executor extensions.
- `src/frozen/`: frozen pipelines written as Rust.
- `graphs/`: starter graphs (JSON).
- `presets/`: compute presets (JSON).

The generated registry in `OUT_DIR` lists their contents in sorted file-stem order; a
missing directory registers nothing.

## JSON

All JSON goes through `raa_loader::json` (`parse`, `write_pretty`, `format_number`),
the one JSON implementation of the workspace. The crate uses no serde.
