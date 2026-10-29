# RAA API m22: material textures

Images that material functions sample. `Engine.addTexture` uploads an rgba image into one
raa-lit texture atlas and returns its id; `raa_texture(id, uv)` (raa-shader prelude) samples it
in material WGSL. Types: `TextureData`, `addTexture` and `removeTexture` in
`packages/raa/index.d.ts`. Failures throw a `RaaError` with a `code`; a refusal changes nothing.

## Methods

- `addTexture({ width, height, data }): number`: `data` holds `width * height * 4` rgba
  components, rows top first. A `Uint8Array` or `Uint8ClampedArray` is stored as is; a
  `Float32Array` is clamped to [0, 1] (NaN as 0) and quantized to 8 bits, round(x * 255).
  The id is the lowest free one, from 1.
- `removeTexture(id): void`: frees the texture. Removing an unknown or already removed id
  does nothing. The id may be handed out again by a later `addTexture`.

```js
const base = engine.addTexture({ width: 256, height: 256, data: rgba8 });
const mat = engine.addSurfaceMaterial({
  function: { name: 'baked', wgsl: `fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    o.base_color = raa_srgb_to_linear(raa_texture(${base}u, in.uv).rgb);
    return o;
  }` },
});
engine.removeTexture(base);   // the material now samples (0, 0, 0, 0)
```

## WGSL

```wgsl
fn raa_texture(id: u32, uv: vec2<f32>) -> vec4<f32>
```

`uv` (0, 0) is the top-left corner of the image; it is clamped to [0, 1] (clamp to edge) and
filtered linearly. Each image is packed into a 4096 x 4096 rgba8unorm atlas (shelf packing);
the sample position stays half a texel inside the image, so neighbours never bleed in. No
mipmaps: the sample is level 0, so the function may be called in non-uniform control flow.
Texels are returned as stored (bytes / 255): decode sRGB colour maps with `raa_srgb_to_linear`,
tangent-space normal maps with `* 2.0 - 1.0` and `raa_perturb_normal`. An unknown or removed
id returns (0, 0, 0, 0).

The atlas, its sampler and the rect of each id are bindings 3, 4 and 5 of the lit group 0
(after the frame uniform, materials and instance extras). The 4096 atlas is allocated by the
first `addTexture`; before that a 1 x 1 placeholder is bound.

## Error codes

| Code | Thrown by | When |
| --- | --- | --- |
| `texture.invalid` | `addTexture` | `data.length` is not `width * height * 4`, or a side is 0 or over 4096 |
| `texture.atlas_full` | `addTexture` | no room left in the atlas, or 1023 textures are live |
| `api.invalid_argument` | `addTexture` | not an object, a side that is not a whole number, `data` of another type |
