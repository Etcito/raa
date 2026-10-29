# RAA API m20: player tint

Additions of S23 (Player Color material node) to `@etcito/raa`: a per-material tint mode and the
reset of an instance tint. Types: `packages/raa/types/m20-player-tint.d.ts` (appended to
`index.d.ts` when the package is staged; `Engine` merges with the v0 declaration). Everything in
v0 and `docs/api/m07`..`m18` stays valid. Failures throw a `RaaError` with a `code`; a refusal
changes nothing.

## Exports

- `TintMode`: `'multiply' | 'graph'`.
- `setMaterialTintMode(material, mode)`, `clearInstanceTint(instance)`: methods of `Engine`.

## Tint modes

`setInstanceTint(instance, rgba)` (m15) stores a per-instance tint. How it reaches the shaded
colour depends on the tint mode of the instance's material:

- `multiply` (the default, today's shader): after the material function, the tint rgb multiplies
  the base colour and the tint alpha multiplies the opacity.
- `graph`: the material graph reads the tint itself (the MAMA `input.player_color` node). The
  passes are composed without the post-material colour multiply; the tint alpha still multiplies
  the opacity.

`setMaterialTintMode(material, mode)` stores the mode of a surface material handle (from
`addSurfaceMaterial`). The draws of a `graph` material carry the variant flag
`raa_shader::VARIANT_FLAG_TINT_GRAPH`, so they use their own pipelines composed without the
multiply; `multiply` materials keep today's pipelines, pixel for pixel. Setting the mode a
material already has does nothing. raa-sdfm calls it right after
`addSurfaceMaterial` for materials whose MAMA WGSL uses `mat_instance_tint`.

```js
const mat = engine.addSurfaceMaterial(desc);   // desc.function holds the player_color node
engine.setMaterialTintMode(mat, 'graph');
engine.setInstanceTint(bean, PALETTE[player]); // the graph shows the player colour
engine.clearInstanceTint(bean);                // back to the node's preview colour
```

`clearInstanceTint(instance)` resets the tint to `[1, 1, 1, 1]` and the set flag to 0. Squash,
jiggle and instance parameters are kept. It validates the instance like `setInstanceTint`.

## WGSL contract (S23)

The prelude declares two module-scope values that material function code may read:

```wgsl
var<private> mat_instance_tint: vec4<f32>;     // the instance tint (linear rgb, a)
var<private> mat_instance_tint_set: f32;       // 1 after setInstanceTint, 0 after clearInstanceTint
```

Each pass (prepass, forward, transparent, fur shells) sets both from the instance record before it
calls `material_main`, so they hold the shaded instance's values. The flag is 0 for an instance
whose tint was never set. The `input.player_color` node emits:

```wgsl
let c = select(<preview colour>, mat_instance_tint.rgb, mat_instance_tint_set > 0.5);
```

A material using them should be in tint mode `graph`; in `multiply` the tint would be applied a
second time after the material.

## Error codes

| Code | Thrown by | When |
| --- | --- | --- |
| `tint.material_range` | `setMaterialTintMode` | `material` is not a live surface material handle |
| `tint.mode_unknown` | `setMaterialTintMode` | `mode` is not exactly `'multiply'` or `'graph'` |
| `api.unknown_handle` | `clearInstanceTint` | `instance` is not a live instance handle |
