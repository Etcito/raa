# RAA API m25: material data words

Inputs a material function reads as data instead of WGSL text. `Engine.addMaterialData`
stores a block of `u32` words in one engine-wide read-only storage buffer and returns its base
word offset; a surface material carries that base as `SurfaceMaterialDesc.data`, and its
material function reads the words from the prelude accessors below. One registered function,
hence one compiled pipeline per pass and variant, then shades any number of materials whose
differences are data: the pipeline count no longer follows the content. Types:
`addMaterialData`, `setMaterialData`, `removeMaterialData` and `SurfaceMaterialDesc.data` in
`packages/raa/index.d.ts`. Failures throw a `RaaError` with a `code`; a refusal changes nothing.

RAA gives the words no meaning: their layout is the material function's own contract.

## Methods

- `addMaterialData(words): number`: `words` is a `Uint32Array` (stored as is), a
  `Float32Array` (each value stored as its IEEE bits, read back by `raa_material_f32`) or an
  `Int32Array` (two's complement bits), at least one word. Returns the block's base word
  offset. Blocks never start in the 4 reserved words at offset 0; a freed range is reused
  (first fit). The buffer starts at 4096 words and grows (doubling) when a block does not fit.
- `setMaterialData(base, words): void`: overwrites the live block at `base` with exactly as
  many words as it holds.
- `removeMaterialData(base): void`: frees the block; its words read zero. Removing an unknown
  or already removed base does nothing. The range may be handed out again.
- `SurfaceMaterialDesc.data?: number`: the base of the material's block (`addSurfaceMaterial`,
  `setSurfaceMaterial`); absent = 0, the reserved zero words.

```js
const base = engine.addMaterialData(new Float32Array([0.8, 0.2, 0.1]));
engine.addMaterialFunction('tinted', `fn material_main(in: MaterialInput) -> MaterialOutput {
  var o = raa_default_output(in);
  let b = raa_material_data_base(in.record);
  o.base_color = vec3<f32>(raa_material_f32(b), raa_material_f32(b + 1u), raa_material_f32(b + 2u));
  return o;
}`);
const red = engine.addSurfaceMaterial({ function: 'tinted', data: base });
// Another colour, same function and pipeline:
const blue = engine.addSurfaceMaterial({ function: 'tinted', data: engine.addMaterialData(new Float32Array([0.1, 0.2, 0.8])) });
engine.removeMaterialData(base); // `red` now reads zeros
```

## WGSL

```wgsl
fn raa_material_data_base(record: SurfaceMaterialGpu) -> u32   // in.record of MaterialInput / VertexMaterialInput
fn raa_material_word(i: u32) -> u32                            // absolute word index; 0 past the end of the buffer
fn raa_material_f32(i: u32) -> f32                             // bitcast<f32>(raa_material_word(i))
```

The words are binding 6 of the lit group 0 (`var<storage, read> raa_material_data:
array<u32>`, vertex and fragment stages), after the frame uniform, materials, instance extras
and the texture atlas, sampler and rects. The base is `SurfaceMaterialGpu.header.w`. A user
function still declares no binding: it only calls the accessors. Indices are not checked
against the block: a function reading past its block reads its neighbours' words (or 0 past
the buffer).

## Error codes

| Code | Thrown by | When |
| --- | --- | --- |
| `material.data_invalid` | `addMaterialData`, `setMaterialData` | no word; setMaterialData of an unknown base or with another length |
| `material.data_full` | `addMaterialData` | the buffer would pass the device's `maxStorageBufferBindingSize` |
| `api.invalid_argument` | all three, `addSurfaceMaterial`, `setSurfaceMaterial` | `words` of another type; a base or `data` that is not a whole number >= 0 |
