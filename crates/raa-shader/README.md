# raa-shader

WGSL composition and validation for the lit renderer (ARCHITECTURE S11). A composed module is
`PRELUDE` (S10 structs and the bindings of groups 0-1), then `LIB` (helpers), one material
function, a passthrough `material_vertex` when the function has none, the winning provider
of each slot, and the entry points (`vs_main`, `fs_main`) of one pass kind. Every composed
module can be checked on the CPU with naga (WGSL front-end + validator, core WebGPU
capabilities), so a broken shader fails `cargo test` without a GPU. Depends on `raa-core`
and `naga` (`wgsl-in`).

## Public items

- `PassKind` (`Prepass`, `Forward`, `Shadow`, `Transparent`, `FurShell`; `ALL`).
- `SlotId` (`ALL`, `fn_name`) and `SlotProvider { slot, priority, provider, wgsl }`. A
  provider's WGSL defines exactly the slot function, plus private helpers named
  `raa_<provider>_*`.
- `MaterialFnId` (`STANDARD` = 0) and `VariantKey { material_fn, pass, msaa, flags }`.
- `MaterialFnRegistry`:
  - `new` registers `raa.standard` as id 0.
  - `register_builtin` stores engine functions. Only a duplicate name is refused.
  - `register_user` stores user WGSL from JS after checking the user rules below.
  - Lookups: `get` (returns a `MaterialFnEntry { name, wgsl, has_vertex, builtin }`),
    `id_of`, `names`, `len`.
- `Composer`:
  - `new` starts with no providers.
  - `add_slot` installs a provider. Two providers with the same priority on one slot are
    refused with `shader.slot_conflict`.
  - `winner` returns the provider that is used for a slot.
  - `slots_hash` is FNV-1a over the winners.
  - `compose` builds the module of a `VariantKey`.
  - `validate` runs naga's parser and validator. Messages start with `line:col`.
- `ShaderError { code, message }`, `code()`. Codes: `shader.parse`, `shader.validate`,
  `shader.slot_conflict`, `shader.slot_missing`, `shader.unknown_fn`, `material.fn_invalid`,
  `material.fn_duplicate`.
- WGSL sources: `PRELUDE_WGSL`, `LIB_WGSL`, `PASS_PREPASS_WGSL`, `PASS_FORWARD_WGSL`,
  `PASS_SHADOW_WGSL`, `PASS_TRANSPARENT_WGSL`, `PASS_FURSHELL_WGSL`.

## Slot names

The highest priority wins. The default providers (priority 0) are installed by raa-lit.

| `SlotId` | function | default |
|---|---|---|
| `Deform` | `raa_slot_deform(p, n, x: InstanceExtraGpu, t) -> VertexMaterialOutput` | identity |
| `Shadow` | `raa_slot_shadow(world_pos, n, frag) -> f32` | 1 |
| `Ao` | `raa_slot_ao(frag) -> f32` | 1 |
| `Ambient` | `raa_slot_ambient(s: MaterialOutput, c: ShadeCtx) -> vec3<f32>` | S1 PbrLite ambient |
| `Direct` | `raa_slot_direct(s, c, l, radiance) -> vec3<f32>` | S1 GGX + Lambert |
| `Transmission` | `raa_slot_transmission(s, c) -> vec3<f32>` | 0 |
| `Fog` | `raa_slot_fog(color, world_pos, c) -> vec3<f32>` | identity |

## User material functions

`register_user(name, wgsl, composer)` refuses `material.fn_duplicate` when `name` is
already registered. Otherwise it applies the rules below in this order. A failed rule gives
`material.fn_invalid`, and the message names the rule:

1. `size limit`: the source is at most 65536 bytes.
2. `forbidden attribute`, `forbidden address space`, `forbidden keyword`: the source uses
   none of `@group`, `@binding`, `@vertex`, `@fragment`, `@compute`, `var<storage`,
   `var<uniform`, `var<workgroup`, `enable` or `override`. These are found by a token scan,
   so comments do not count. The message gives the line:col.
3. `parse`: the source parses (naga's message, line:col in the user source). naga cannot
   parse the user source on its own because it uses the prelude structs and LIB helpers.
   So the source is parsed after PRELUDE and LIB, and only items whose span is inside the
   user source count as its definitions.
4. `reserved prefix`: the source defines no module-scope `fn`, `struct`, `const` or `var`
   whose name starts with `raa_`. Redefining a LIB helper or prelude global, such as
   `raa_fbm3`, is reported the same way.
5. `material_main signature`: the source defines
   `fn material_main(in: MaterialInput) -> MaterialOutput`.
6. `composed Forward variant (msaa 4) does not validate`: the Forward variant built with
   the given composer passes naga. The message keeps naga's text and the line:col in the
   composed module, and adds `(user source L:C)` when the error is inside the user source.
   If a slot has no provider in the composer, the `shader.slot_missing` error is returned
   unchanged.

A refused function leaves the registry unchanged. An accepted one gets the next id.
`has_vertex` is true when the source defines `material_vertex`.

## How to test

```sh
cargo test -p raa-shader --quiet
cargo clippy -p raa-shader --all-targets -- -D warnings
```

The tests are CPU-only:

- `tests/compose.rs`: section order, conflict, missing slot, unknown function, priority.
- `tests/variants.rs` (Q11): the standard function and a vertex function, for every pass
  kind at MSAA 1 and 4, validate with the default providers.
- `tests/user_fn.rs` (Q12): each user rule is refused with its reason; valid functions,
  `has_vertex` and duplicate names are also covered.
