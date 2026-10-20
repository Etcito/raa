# raa-material

Simple materials for RAA: a linear-colour `Material`, its 48-byte GPU layout (S1
`MaterialGpu`) and a `MaterialLibrary` addressed by dense `raa_core::MaterialId`s. It depends
only on raa-core and compiles for the host and for `wasm32-unknown-unknown`.

## Public items

- `ShadingModel { Flat, PbrLite }`: Flat is unlit `base_color + emissive`; PbrLite is
  ambient + Lambert diffuse + GGX specular under one directional light. `gpu_value()` gives
  the `params.z` value (0 Flat, 1 PbrLite).
- `Material { base_color, roughness, metallic, emissive, model }`:
  - `flat(color_linear)`, `pbr(color_linear, roughness, metallic)`,
    `from_srgb(color_srgb, roughness, metallic, emissive_linear)` (PbrLite).
  - `Default`: linear grey 0.5, roughness 0.6, metallic 0, PbrLite.
  - `validate()`: every value finite, colour channels >= 0, roughness and metallic in
    [0, 1]. A refusal has code `material.invalid` and a message naming the field
    (`base_color`, `emissive`, `roughness` or `metallic`) and the rule.
  - `write_gpu(out)`: writes 48 bytes little-endian into `out[..48]`: `base_color` with
    a = 1 @0, `emissive` with w = 0 @16, `params` @32 = (roughness clamped to [0.04, 1],
    metallic, model, 0). `out` must hold at least `MATERIAL_GPU_SIZE` (48) bytes; a shorter
    slice trips a debug assertion and writes nothing in release (no panic).
- `MATERIAL_GPU_SIZE` = 48.
- `srgb_to_linear(c)`: exact piecewise sRGB EOTF (`c / 12.92` for `c <= 0.04045`, else
  `((c + 0.055) / 1.055)^2.4`).
- `MaterialLibrary`: `new()`, `add(m)` (validates, ids dense from 0), `set(id, m)`
  (validates; unknown id gives `material.unknown_id`), `get(id)`, `len()`, `is_empty()`,
  `is_dirty()` (set by `add` and `set`), `write_gpu_all(&mut Vec<u8>)` (clears the vector,
  keeps its capacity, appends 48 bytes per material in id order, clears the dirty flag).
- `MaterialError { code, message }` with `code()`, `Display` and `std::error::Error`.

## Linear colour rule

Colours are linear RGB everywhere in Rust and on the GPU. Only colours coming from asset
files are sRGB; convert them once at load with `Material::from_srgb` (or
`srgb_to_linear`). `emissive` is always linear.

## How to test

```sh
cargo test -p raa-material
cargo clippy -p raa-material --all-targets -- -D warnings
cargo clippy -p raa-material --target wasm32-unknown-unknown -- -D warnings
```
