# raa-transmission

Transmission and refraction for the RAA lit renderer: Beer-Lambert absorption
through a thickness, refraction of the view ray, rough-transmission mip
selection and Schlick Fresnel.

## Public items

- `absorption`: CPU reference functions using the same formulas as the WGSL
  `Transmission` slot `screen`:
  `sigma`, `path_length`, `transmittance`, `transmittance_for`, `refract`,
  `transmission_lod`, `schlick_fresnel`, `NO_ABSORPTION_DISTANCE`.
- `TransmissionFeature::install(gpu, pipeline, lit)`: registers every
  `TRANSMISSIVE_BUILTINS` row as a built-in material function, adds the
  `Transmission` slot provider `screen` (priority 10) and the pass
  `opaque.copy` (order 70). Refusals (nothing installed): `pipeline.busy`,
  `pipeline.duplicate_pass` (also a second install), and the shader codes
  `shader.slot_conflict` / `material.fn_duplicate`.
- `TRANSMISSIVE_BUILTINS`: `raa.ice`, `raa.glass`, `raa.slime`, `raa.water`
  (flags 0, class Opaque; raa-js pushes the table into
  `raa_matfn::ExtraBuiltins`).
- `SCREEN_PROVIDER`: the provider name `screen`.

## Testing

```
cargo test -p raa-transmission
```

`tests/absorption.rs` checks the CPU Beer-Lambert reference (no GPU needed).
`tests/variants.rs` (Q11, CPU only) composes and naga-validates every
transmissive built-in for Transparent (MSAA 1) and Forward (MSAA 1 and 4) over
the raa-lit default providers, the raa-brdf `layered` providers and `screen`,
and validates `copy.wgsl`. The `feature` unit tests install the feature on a
software adapter (printing `SKIP` without one) and check its refusals.
