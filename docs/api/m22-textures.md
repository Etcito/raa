# RAA API m22: material textures

Images that material functions sample. `Engine.addTexture` uploads an rgba image into one
raa-lit texture atlas and returns its id; `raa_texture(id, uv)` (raa-shader prelude) samples it
in material WGSL. Types: `TextureData`, `addTexture` and `removeTexture` in
`packages/raa/index.d.ts`. Failures throw a `RaaError` with a `code`; a refusal changes nothing.

## Methods

- `addTexture({ width, height, data, mips?, srgb? }): number`: `data` holds `width * height * 4`
  rgba components, rows top first. A `Uint8Array` or `Uint8ClampedArray` is stored as is; a
  `Float32Array` is clamped to [0, 1] (NaN as 0) and quantized to 8 bits, round(x * 255).
  The id is the lowest free one, from 1.
  - `mips?: boolean`: default true. `true` uploads the mip chain (see Mip chain); `false`
    uploads level 0 only, and the image is always sampled at level 0.
  - `srgb?: boolean`: default false. `true` averages the chain in linear light and re-encodes
    it, for sRGB colour images; `false` averages the stored bytes. Level 0 is the same either way.
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

## Mip chain

With `mips` true (the default) an image is uploaded with its mip chain:
`levels = min(5, 1 + floor(log2(min side)))`, so at most 5 levels (`ATLAS_MIP_LEVELS`, levels
0 to 4); a 1 x 1 image has one level. Level k is made from level k - 1 by a 2 x 2 box average.
The rect of each image starts on a multiple of 16 texels (`ATLAS_ALIGN`) and reserves its sides
rounded up to 16, so the packer loses up to 15 texels of capacity per side for each image. The
4096 x 4096 atlas takes about 64 MiB at level 0 and about 85 MiB with the chain; the first
`addTexture` allocates it with the chain.

## WGSL

```wgsl
fn raa_texture(id: u32, uv: vec2<f32>) -> vec4<f32>
fn raa_texture_grad(id: u32, uv: vec2<f32>, duv_dx: vec2<f32>, duv_dy: vec2<f32>) -> vec4<f32>
```

`uv` (0, 0) is the top-left corner of the image; it is clamped to [0, 1] (clamp to edge) and
filtered linearly. Each image is packed into a 4096 x 4096 rgba8unorm atlas (shelf packing);
the sample position stays half a texel inside the image, so neighbours never bleed in.
`raa_texture` samples level 0 and may be called anywhere.
Texels are returned as stored (bytes / 255): decode sRGB colour maps with `raa_srgb_to_linear`,
tangent-space normal maps with `* 2.0 - 1.0` and `raa_perturb_normal`. An unknown or removed
id returns (0, 0, 0, 0).

The atlas, its sampler and the rect of each id are bindings 3, 4 and 5 of the lit group 0
(after the frame uniform, materials and instance extras). The 4096 atlas is allocated by the
first `addTexture`; before that a 1 x 1 placeholder is bound.

`raa_texture_grad(id, uv, duv_dx, duv_dy)` samples the mip chain at the level that matches the
screen-space rate of change of `uv`, given as `duv_dx = dpdx(uv)` and `duv_dy = dpdy(uv)` of
the unwrapped uv. The level is clamped to the image's levels: a `mips: false` image is always
sampled at level 0, and an unknown or removed id returns (0, 0, 0, 0). The function takes no
derivative itself, so it may be called in non-uniform control flow, but its derivative
arguments must come from uniform control flow: take `dpdx` and `dpdy` once at the top of
`material_main`, before any branch, from the unwrapped uv, and pass them in. Derivatives of
`fract(uv)` jump at every tile seam and pick the wrong level there. A caller that cannot take
derivatives keeps `raa_texture`. `id` stands for the id `addTexture` returned, written into the
WGSL as in the JS example above.

```wgsl
fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    let uv = in.uv * 4.0;        // unwrapped, tiled: 4 repeats of the image
    let duv_dx = dpdx(uv);       // taken at the top, before any branch
    let duv_dy = dpdy(uv);
    let c = raa_texture_grad(id, fract(uv), duv_dx, duv_dy);
    o.base_color = raa_srgb_to_linear(c.rgb);
    return o;
}
```

## Error codes

| Code | Thrown by | When |
| --- | --- | --- |
| `texture.invalid` | `addTexture` | `data.length` is not `width * height * 4`, or a side is 0 or over 4096 |
| `texture.atlas_full` | `addTexture` | no room left in the atlas, or 1023 textures are live |
| `api.invalid_argument` | `addTexture` | not an object, a side that is not a whole number, `data` of another type |
| `api.invalid_argument` | `addTexture` | `mips` or `srgb` is not a boolean (message names the key); nothing changes |
