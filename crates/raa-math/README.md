# raa-math

Small f32 math for RAA, with no dependency. Conventions: right-handed, Y up, camera looks
down -Z; matrices are column-major with column vectors (`M * v`). It compiles for the host
and for `wasm32-unknown-unknown`.

## Public items

- `Vec3 { x, y, z }` (Copy, Clone, Debug, PartialEq, Default): `new`, `splat`, consts
  `ZERO`, `ONE`, `X`, `Y`, `Z`, `dot`, `cross` (`X.cross(Y) == Z`), `length`,
  `length_squared`, `normalize_or_zero` (`ZERO` for a zero or non-finite length), `min`,
  `max`, `abs`, `lerp(a, b, t)` (= `a + (b - a) * t`), `to_array`, `from_array`,
  `is_finite`; operators `+`, `-`, unary `-`, `* f32`, `/ f32`, `* Vec3`
  (component-wise), `+=`, `-=`.
- `Vec4 { x, y, z, w }` (same derives): `new`, `from_vec3(v, w)`, `xyz`, `dot`,
  `to_array`; operators `+`, `-`, `* f32`.
- `Quat { x, y, z, w }` (same derives, stored `[x, y, z, w]`): `IDENTITY`, `from_xyzw`,
  `from_array`, `to_array`, `from_axis_angle(axis, radians)` (axis normalised; zero axis
  gives `IDENTITY`), `from_rotation_y`, `mul` (Hamilton; `a.mul(b)` applies `b` first, also
  `*`), `rotate(v)`, `normalize` (`IDENTITY` for zero length), `length`, `conjugate`,
  `slerp(a, b, t)` (shortest path).
- `Mat4 { cols: [[f32; 4]; 4] }` (same derives, column-major, `cols[c][r]`): `IDENTITY`,
  `from_cols`, `from_translation`, `from_scale`, `from_quat`, `from_trs(t, r, s)`
  (= `T * R * S`), `mul` (also `*`), `mul_vec4`, `transform_point` (w = 1, divides by w
  when it is not 1), `transform_vector` (w = 0), `transpose`, `determinant`, `inverse`
  (`None` when `abs(det) < 1e-12`), `normal_matrix` (inverse-transpose of the upper 3x3 as
  3 columns with w = 0; identity when singular), `perspective_rh_zo(fov_y, aspect, near,
  far)` (clip depth [0, 1]: `-near` maps to 0, `-far` to 1), `look_at_rh(eye, target, up)`
  (eye to origin, target on -Z), `to_cols_array` (16 floats, column-major),
  `write_le_bytes(out)` (64 little-endian bytes; `out` must hold 64 bytes, debug-asserted,
  release writes only what fits).
- `Aabb { min, max }` (same derives): `EMPTY` (min +inf, max -inf), `from_points(&[[f32; 3]])`
  (`EMPTY` for no points), `include(p)`, `union`, `intersects` (touching counts),
  `center`, `extent` (= `max - min`), `diagonal` (length of `extent`, 0 for `EMPTY`),
  `transform(&m)` (bounds of the 8 transformed corners; `EMPTY` stays `EMPTY`).
- `approx_eq(a, b, eps) -> bool`: true when `abs(a - b) <= eps`.

## How to test

```sh
cargo test -p raa-math
cargo clippy -p raa-math --all-targets -- -D warnings
cargo clippy -p raa-math --lib --target wasm32-unknown-unknown -- -D warnings
```

Unit tests live next to each type; `tests/math.rs` covers TRS composition, inverse round
trips, quaternion vs matrix rotation, perspective depth, `look_at_rh` orientation and
`Aabb::transform`.
