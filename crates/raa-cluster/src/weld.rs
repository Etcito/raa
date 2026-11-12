//! Grid weld of the input mesh: positions snap to the 16-bit quantisation grid, coincident
//! vertices merge, collapsed triangles drop, and octahedral normals are packed.
//! Adapted from solstice (MIT), Copyright (c) 2026 SamG-Coder:
//! `crates/solstice-builder/src/dag.rs`, lines 419-533, revision 7e465b5.
