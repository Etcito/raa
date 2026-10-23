# raa-ibl

Image-based lighting derived from the procedural sky cube (S10 group 1 bindings 3-5).

## Purpose

`src/sh.rs` is the CPU reference of the SH9 irradiance projection that the GPU reduction
reproduces: cube texel direction (wgpu face order +X,-X,+Y,-Y,+Z,-Z), exact texel solid angle
(Driscoll), real SH basis order 2, cosine-lobe convolution (A0 = pi, A1 = 2pi/3, A2 = pi/4)
divided by pi. A constant-radiance environment `c` evaluates to `c` in every direction.
Coefficients are `[[f32; 4]; 9]` (rgb, w = 0), the shape `LightingResources::set_env` takes.

## Public items

None yet (the SH helpers are `pub(crate)`); items are listed here as they are added.

## Test

`cargo test -p raa-ibl`
