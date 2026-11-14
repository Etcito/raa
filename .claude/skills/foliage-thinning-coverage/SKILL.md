---
name: foliage-thinning-coverage
description: Use when changing raa-cluster thin.rs (pieces, enlargement, placement) or the foliage bake tests: thinned blades must stay upright and cover the patch footprint, and tests must measure both.
---

# Foliage thinning must keep coverage and orientation

`thin()` in raa-cluster replaces many thin blades with fewer, larger pieces. One regression produced flat, tilted diamond plates with large empty gaps, and the bake tests still passed because they never measured the thinned output.

## Rules
- Thinned pieces must stay upright (same up axis as the source blades) and must cover the source patch footprint without obvious holes.
- Any change to enlargement or placement needs a test that measures footprint coverage and piece orientation on the thinned output. A test that only counts triangles or checks that the call succeeds is not enough. Prefer a property test over several inputs.
- Test the input the scene really uses. The virtual-geometry page builds its foliage in `lib/content.js`, so check which generator feeds the scene before you fix against a different one.
- Index sites in thin.rs must not panic on degenerate pieces. Give each one a defined fallback and a test.
- Re-run the full test list after your last edit. Fixing geometry without re-running the coverage test on the final file has already shipped a wrong fix.
