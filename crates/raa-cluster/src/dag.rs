//! Cluster DAG builder: the weld, the per-level partition, group, simplify and re-split loop,
//! and the packing of sibling sets into leaves and packed clusters.
//! Adapted from solstice (MIT), Copyright (c) 2026 SamG-Coder:
//! `crates/solstice-builder/src/dag.rs`, revision 7e465b5.
