# raa-ibl

Image-based lighting derived from the procedural sky cube (S10 group 1 bindings 3-5).

## Purpose

`src/sh.rs` is the CPU reference of the SH9 irradiance projection that the GPU reduction
reproduces: cube texel direction (wgpu face order +X,-X,+Y,-Y,+Z,-Z), exact texel solid angle
(Driscoll), real SH basis order 2, cosine-lobe convolution (A0 = pi, A1 = 2pi/3, A2 = pi/4)
divided by pi. A constant-radiance environment `c` evaluates to `c` in every direction.
Coefficients are `[[f32; 4]; 9]` (rgb, w = 0), the shape `LightingResources::set_env` takes.

`src/wgsl/ibl_lib.wgsl` holds the IBL contract functions; `src/wgsl/ambient.wgsl` the
`raa_slot_ambient` of provider `ibl` (`src/provider.rs`: slot `Ambient`, priority 10, source
IBL_LIB_WGSL followed by ambient.wgsl). Without an environment (`EnvSh.info.w < 0.5`) it
returns the S1 PbrLite ambient term; otherwise SH diffuse x base x (1 - metallic) with
Jimenez multi-bounce AO, plus prefiltered specular along the Lagarde dominant direction x
(F0 A + B), both with Fdez-Aguera multiscatter energy compensation, the specular term with
Lagarde specular occlusion, each scaled by its `EnvSh.info` intensity.

## Public items

- `IBL_LIB_WGSL: &str`: `raa_ibl_irradiance(n)`, `raa_ibl_specular(r, roughness)`,
  `raa_ibl_dfg(n_dot_v, roughness) -> vec2` and `raa_specular_occlusion(n_dot_v, ao,
  roughness)`; they read only prelude globals (`raa_env_sh`, `raa_env_specular`,
  `raa_brdf_lut`, `raa_linear_clamp`) so other providers reuse them.
- The `ibl` Ambient slot provider (`provider::ambient_provider`, `pub(crate)`, installed by
  the IBL feature).

The SH helpers are `pub(crate)`.

## Test

`cargo test -p raa-ibl` (`tests/sh.rs`: CPU SH9; `tests/variants.rs`: the `ibl` ambient
provider with raa-lit's defaults x standard fn x every pass kind, naga-validated).
