---
name: strict-check-whole-workspace
description: Use before finishing any raa change: scripts/check.sh (cargo fmt --all --check plus clippy on every crate) must be green on the whole workspace, including reds inherited from merged work.
---

# The strict check covers the whole workspace

`scripts/check.sh` runs `cargo fmt --all --check` and clippy across all crates (raa-sky, raa-vg, raa-cluster, ...). It fails on files you never touched, for example formatting or clippy issues brought in when another line of work was merged.

## Rules
- Measure first. List the inherited formatting reds with `cargo fmt --all --check 2>&1 | grep '^Diff in'`, and run clippy before you start, so you know which files beyond your own need edits.
- Make sure your branch contains the current main before you judge what is left. A branch behind main shows reds that are already fixed there.
- Do not treat inherited diffs as a baseline and do not suppress lints. The gate is the strict script, not a count relative to the base.
- If a test in another file fails (for example a layout expectation in `crates/raa-sky/tests/variants.rs`), read the panic text before you change anything.
- GPU tests can fail in a sandbox with no adapter. Use the measured output to tell a missing device apart from a real defect before you edit the test.
