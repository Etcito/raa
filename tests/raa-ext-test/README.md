# raa-ext-test

Purpose: extensibility is tested, not claimed. This crate adds a new geometry
kind (`star`) to RAA from outside, without editing any RAA crate.

Public items:
- `StarProvider`: `GeometryProvider` for kind `star`, params `points`, `outerRadius`,
  `innerRadius`, `halfHeight`; a closed, non-convex star prism with hard edges.
- `TintPass`: added by the next task.

Test: `cargo test -p raa-ext-test`

Rule: nothing in the workspace depends on this crate, and it depends only on `raa-*` crates.
