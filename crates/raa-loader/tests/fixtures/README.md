# Loader test fixtures

- `published/*.raa.prims.json`: copies of the valid raa.prims v0 fixtures of
  `raa-assets-tools/fixtures/valid` (raa-assets-tools commit `eb8ed4d`). They are the
  published format, unchanged, and every one must parse and build.
  `tests/published.rs` checks them.
- `royal-fiesta/bean.prims.json`: a copy of `royal-fiesta/fixtures/assets/bean.prims.json`
  (royal-fiesta commit `2b53335`). It is NOT valid raa.prims v0 (its `budget` is a number,
  its transforms have no `r` and `s`, its ids and emissive values do not match the schema,
  and its colliders use flat halfHeight/radius keys). The loader must refuse it, at its
  budget, and the test only checks that refusal.

Keep these files byte-identical to their sources; regenerate by copying, not by editing.
