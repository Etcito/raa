# raa-transmission

Transmission and refraction for the RAA lit renderer: Beer-Lambert absorption
through a thickness, refraction of the view ray, rough-transmission mip
selection and Schlick Fresnel.

## Public items

- `absorption`: CPU reference functions using the same formulas as the WGSL
  `Transmission` slot `screen`:
  `sigma`, `path_length`, `transmittance`, `transmittance_for`, `refract`,
  `transmission_lod`, `schlick_fresnel`, `NO_ABSORPTION_DISTANCE`.

## Testing

```
cargo test -p raa-transmission
```

`tests/absorption.rs` checks the CPU Beer-Lambert reference (no GPU needed).
