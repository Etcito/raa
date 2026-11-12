//! Sliced cluster bake: `ClusterBake` advances the DAG build in bounded work units, `bake()`
//! runs it to the end, and the progress and status types report it.
//! Adapted from solstice (MIT), Copyright (c) 2026 SamG-Coder:
//! `crates/solstice-builder/src/dag.rs`, lines 92-123, revision 7e465b5.
