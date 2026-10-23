# RAA API m10: materials

Additions of chain milestone 4 to `@etcito/raa`: WGSL material functions, layered surfaces (clear
coat, sheen), fur shells and emissive bloom. Types: `packages/raa/types/m10-materials.d.ts`
(appended to `index.d.ts` when the package is staged; `Engine` merges with the v0 declaration).
Everything in v0 and `docs/api/m07`..`m09` stays valid. Failures throw a `RaaError` with a `code`.

## Exports

- `ParamKind`: `'scalar' | 'vec3' | 'color'`.
- `MaterialFunctionParam`: `{ name, kind, default }`; `default` has 1 number (scalar) or 3.
- `MaterialFunctionInfo`: `{ name, builtin, params }`.
- `SurfaceBlend`: `'opaque' | 'transparent' | 'transmissive'`.
- `SurfaceMaterialDesc`: the material description, below.
- `BloomSettings`: `{ strength?, radius? }`.
- `addMaterialFunction(name, wgsl)`, `materialFunctions()`, `addSurfaceMaterial(desc)`,
  `setSurfaceMaterial(id, desc)`, `setBloom(settings)`: methods of `Engine`.

## Surface materials

`addSurfaceMaterial(desc)` returns a handle for `addInstance` and sub-mesh materials;
`setSurfaceMaterial(id, desc)` replaces it (omitted fields take defaults; a refusal changes
nothing). Colours are sRGB [0, 1] except `emissive`, which is linear and unbounded above (HDR).

| field | default | unit / range |
|---|---|---|
| `function` | `raa.standard` | material function name |
| `params` | function defaults | name to number or number[] |
| `blend` | `opaque` | `opaque`, `transparent`, `transmissive` |
| `doubleSided` | false | disables culling |
| `castShadow` | true | |
| `color` | [0.8, 0.8, 0.8] | sRGB |
| `roughness` | 0.5 | [0, 1] |
| `metallic` | 0 | [0, 1] |
| `emissive` | [0, 0, 0] | linear, >= 0 |
| `opacity` | 1 | [0, 1] |
| `clearcoat` | 0 | [0, 1] |
| `clearcoatRoughness` | 0.1 | [0, 1] |
| `sheenColor` | [0, 0, 0] | sRGB |
| `sheenRoughness` | 0.5 | [0, 1] |
| `transmission` | 0 | [0, 1] |
| `ior` | 1.5 | [1, 3] |
| `thickness` | 0 | metres |
| `attenuationColor` | [1, 1, 1] | sRGB |
| `attenuationDistance` | Infinity | metres |
| `furLength` | 0 | metres; > 0 enables fur |
| `furDensity` | 2000 | strands per cm2 |
| `furDroop` | 0 | [0, 1] |
| `furShells` | 32 | integer 1..64 |

Fur is drawn by the `fur.shells` contributor of the forward pass with alpha-to-coverage.
`setPassEnabled('fur.shells', false)` leaves fur materials drawn as their base surface only.

## Built-in material functions

Seven functions, listed by `materialFunctions()` with `builtin: true`. Colour params are sRGB.

| name | params (default) |
|---|---|
| `raa.standard` | none; the `SurfaceMaterialDesc` fields alone |
| `raa.unlit` | `color` color (1,1,1), `strength` scalar (1) |
| `raa.checker` | `colorA` color (0.9,0.9,0.9), `colorB` color (0.1,0.1,0.1), `scale` scalar (4) |
| `raa.stripes` | `colorA` color (0.9,0.9,0.9), `colorB` color (0.1,0.1,0.1), `scale` scalar (8), `angle` scalar (0) |
| `raa.dots` | `colorA` color (0.9,0.9,0.9), `colorB` color (0.1,0.1,0.1), `scale` scalar (6), `radius` scalar (0.3) |
| `raa.noise` | `colorA` color (0.2,0.2,0.2), `colorB` color (0.8,0.8,0.8), `scale` scalar (3), `octaves` scalar (4) |
| `raa.gradient` | `colorA` color (1,1,1), `colorB` color (0,0,0), `direction` vec3 (0,1,0) |

## User WGSL functions (S11)

`addMaterialFunction(name, wgsl)` registers source that must define
`fn material_main(in: MaterialInput) -> MaterialOutput`. Rules, each refused with
`material.fn_invalid`: no `@group`, `@binding`, `@vertex`, `@fragment`, `@compute`,
`var<storage`, `var<uniform`, `var<workgroup`, `enable` or `override`; no identifier starting
`raa_`; at most 65536 bytes; the composed forward variant must validate (the message carries the
naga line:col). A name already registered is `material.fn_duplicate`. Helpers available:
`raa_hash3`, `raa_noise3`, `raa_fbm3`, `raa_voronoi3`, `raa_triplanar_weights`,
`raa_perturb_normal`, `raa_luma`, `raa_srgb_to_linear`, and `raa_default_output(in)`.

```js
engine.addMaterialFunction('animated-checker', `
fn material_main(in: MaterialInput) -> MaterialOutput {
  var o = raa_default_output(in);
  let p = floor(in.uv * 8.0 + vec2<f32>(in.time * 0.5, 0.0));
  let c = select(0.15, 0.9, (i32(p.x) + i32(p.y)) % 2 == 0);
  o.base_color = vec3<f32>(c);
  return o;
}`);
const id = engine.addSurfaceMaterial({ function: 'animated-checker' });
```

### Positional parameters

A function's params are delivered to the shader as `in.record.p[0]..p[7]` (`p0..p7`, vec4 each):
parameter `i` of `materialFunctions()` goes to `p[i]`; a `scalar` in `.x`, a `vec3` or `color`
in `.xyz` (colours converted from sRGB to linear). User functions currently declare none, so
`params` for them is refused with `material.unknown_param`.

## Bloom

`setBloom({ strength?, radius? })` merges over the current values: `strength` default 0.04,
`radius` default 0.85, both in [0, 1]. Bloom is up to 8 mips with a Karis average on the first
downsample. Emissive above 1 blooms.

## Pass names for `setPassEnabled`

- `fur.shells` (order 500): fur shells; disabled, fur materials show no strands.
- `bloom` (order 120): disabled, emissive no longer glows.

## Refusal codes

| code | meaning |
|---|---|
| `material.unknown_fn` | `function` names no registered material function |
| `material.unknown_param` | `params` key not a parameter of the function |
| `material.param_type` | `params` value has the wrong arity or is non-finite |
| `material.invalid` | a desc field is out of range, non-finite or inconsistent |
| `material.blend_unavailable` | `transparent` or `transmissive` while no transparency feature is installed |
| `material.fn_invalid` | user WGSL breaks an S11 rule or fails validation |
| `material.fn_duplicate` | function name already registered |
| `bloom.invalid` | `setBloom` field out of range or non-finite |
| `api.invalid_argument` | an argument has the wrong JS type |
