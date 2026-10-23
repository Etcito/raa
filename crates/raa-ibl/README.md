# raa-ibl

Image-based lighting derived from the procedural sky cube (S10 group 1 bindings 3-5).

## Purpose

`IblFeature::install` prepares a pipeline and its lit renderer for IBL; the returned
`IblHandle` is driven every frame by the `sky.env` pass of raa-sky:

- **BRDF LUT** (`src/lut.rs`, `src/wgsl/brdf_lut.wgsl`): 128x128 Rg16Float split-sum scale A
  and bias B (GGX, height-correlated Smith), computed once on the GPU at install. Core WebGPU
  has no Rg16Float storage texture, so the compute pass writes `pack2x16float(A, B)` into a
  storage buffer copied into the texture. Set with `LightingResources::set_brdf_lut`.
- **Filter** (`src/filter.rs`): from the SKY_ENV source cube (256 px, Rgba16Float, 6 mips,
  usage TEXTURE_BINDING | STORAGE_BINDING | COPY_SRC) it generates source mips 1..5 by box
  downsample, copies mip 0 into the IBL-owned output cube, computes output mips 1..5 by GGX
  importance sampling (1024 samples per texel, PDF-based source mip), and projects the SH9
  irradiance by compute reduction with solid-angle weights. The 9 coefficients are copied
  into a ring of 3 staging buffers for asynchronous readback.
- **SH9** (`src/sh.rs`, the CPU reference the GPU reduction reproduces): cube texel direction
  (wgpu face order +X,-X,+Y,-Y,+Z,-Z), exact texel solid angle (Driscoll), real SH basis
  order 2, cosine-lobe convolution (A0 = pi, A1 = 2pi/3, A2 = pi/4) divided by pi. A
  constant-radiance environment `c` evaluates to `c` in every direction. Coefficients are
  `[[f32; 4]; 9]` (rgb, w = 0), the shape `LightingResources::set_env` takes.
- **Ambient provider** `ibl` (`src/provider.rs`: slot `Ambient`, priority 10, source
  IBL_LIB_WGSL followed by `src/wgsl/ambient.wgsl`). Without an environment
  (`EnvSh.info.w < 0.5`) it returns the S1 PbrLite ambient term; otherwise SH diffuse x base
  x (1 - metallic) with Jimenez multi-bounce AO, plus prefiltered specular along the Lagarde
  dominant direction x (F0 A + B), both with Fdez-Aguera multiscatter energy compensation,
  the specular term with Lagarde specular occlusion, each scaled by its `EnvSh.info`
  intensity.

### Asynchronous SH

The SH9 reaches the lighting through a GPU readback, never in the frame that filtered:

1. `filter` records into the frame's encoder and submits nothing (the encoder belongs to
   the frame).
2. The next `apply_pending` (in `sky.env`'s prepare, so after that encoder was submitted)
   calls `map_async` on every staging buffer written since the last call.
3. The map callback stores the newest coefficients (older ones are dropped). On wasm the
   browser drives the callbacks; native callers poll the device. A later `apply_pending`
   takes them and calls `set_env(output cube, sh, [specular, diffuse, 5, 1])`.

Until the first SH lands the env stays cleared (`info.w = 0`), so the ambient provider
falls back to the flat S1 ambient instead of black. When all 3 staging buffers are still
in flight, `filter` skips the copy and the next `apply_pending` returns true so the caller
filters again. `apply_pending` must not run between `filter` and the submission of its
encoder (a buffer mapped while a pending copy targets it fails validation at submit).

## Public items

- `IblFeature::install(gpu, pipeline: &Rc<RefCell<Pipeline>>, lit: &mut LitRenderer) ->
  Result<IblHandle, RaaError>`: adds the `ibl` Ambient provider (refusals keep their
  `add_slot` code, e.g. `shader.slot_conflict` on a second install), builds and sets the
  BRDF LUT, creates the filter; `pipeline.busy` when the pipeline is borrowed. Nothing is
  installed on a refusal.
- `#[derive(Clone)] IblHandle` (clones share one `Rc<RefCell<..>>` state: filter,
  intensities, last applied SH, changed flag):
  - `filter(&self, gpu, encoder, cube: &wgpu::Texture, lighting: &mut LightingResources) ->
    Result<(), RaaError>`: records the filter of `cube`. A cube that is not SKY_ENV with the
    usages above records nothing, clears the env and is refused with `ibl.invalid_source`.
  - `apply_pending(&self, lighting: &mut LightingResources) -> bool`: maps pending
    readbacks, applies a new SH or changed intensities with `set_env`, returns true when a
    readback was skipped (filter again).
  - `set_intensity(&self, diffuse: f32, specular: f32)`: default (1, 1); each value is
    stored when finite and >= 0, ignored otherwise (raa-js refuses it first); applied at the
    next `apply_pending`.
  - `sh(&self) -> Option<[[f32; 4]; 9]>`: the SH9 last applied (None while the env is
    cleared).
- `IBL_LIB_WGSL: &str`: `raa_ibl_irradiance(n)`, `raa_ibl_specular(r, roughness)`,
  `raa_ibl_dfg(n_dot_v, roughness) -> vec2` and `raa_specular_occlusion(n_dot_v, ao,
  roughness)`; they read only prelude globals (`raa_env_sh`, `raa_env_specular`,
  `raa_brdf_lut`, `raa_linear_clamp`) so other providers (raa-brdf) reuse them.

The SH, filter, LUT and provider helpers are `pub(crate)`.

## Test

`cargo test -p raa-ibl`:

- `tests/sh.rs`: CPU SH9 (constant cube, sky gradient brighter up, solid angles, basis).
- `tests/variants.rs`: the `ibl` ambient provider with raa-lit's defaults x standard fn x
  every pass kind, and the filter compute shaders, naga-validated (CPU only).
- `tests/gpu_ibl.rs` (native, `raa_core::testing::software_gpu()`; prints `SKIP ...` without
  an adapter): `constant_cube_filters_to_its_colour` filters a cube cleared to
  (0.4, 0.6, 0.8), polls until the SH lands (at most 50 polls) and checks it at +Y and -Y
  within 2e-2.
- Unit tests in `src/` (readback ring, LUT shader validation and CPU values, LUT build).
