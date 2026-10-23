# raa-matfn

Built-in WGSL material functions of the RAA lit renderer and the parameter tables that map
JS parameters into the material record (`SurfaceMaterialGpu.p`, S10/S11).

## Public items

- `ParamKind` (`Float`, `Vec2`, `Vec3`, `Vec4`, `Color`) and `ParamKind::components`.
- `ParamSpec { name, kind, default, slot }`: `default` is in the unit JS sends (Color
  defaults are sRGB [0, 1], converted to linear when mapping); `slot` is
  `(p index, first component)`; parameters of one function never overlap.
- `BuiltinFn { name, wgsl, params, flags, class }`.
- `BUILTINS`: the built-in table:
  - `raa.inflatable` (`color`, `stripeColor`, `stripeWidth`, `seamDepth`, `seamCount`):
    glossy inflated plastic, meridian seams bend the normal inwards (pillow look), stripes
    along uv.x, clear coat 1 (roughness 0.04), base roughness 0.45.
  - `raa.candy` (`color`, `stripeColor`, `stripeCount`, `swirl`): helical stripes twisted
    by `swirl`, clear coat 1 (roughness 0.05), base roughness 0.35.
- `register_builtins(&mut MaterialFnRegistry)`: registers every row; refuses with
  `material.fn_duplicate` (registering nothing) when a built-in name is already present.

Material functions read their parameters from `in.record.p[i]`; their helpers are prefixed
`<fnname>_` (e.g. `inflatable_stripe`), never `raa_`.

## Testing

```sh
cargo test -p raa-matfn
```
