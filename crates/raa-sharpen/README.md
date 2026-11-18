# raa-sharpen

Contrast-adaptive sharpening on the HDR chain: the pass `sharpen` (S47, order 195, disabled by
default). It runs after `cel.resolve` and before `tonemap`. This share holds the constants, the
settings and the CPU kernel twins; the shaders and the feature come in later shares.

## Constants (`src/lib.rs`)

`SHARPEN_PASS_NAME` = `"sharpen"`, `SHARPEN_ORDER` = 195, `SHARPEN_WEIGHT` = 0.2,
`COMPRESS_LIMIT` = 0.999, `PARAMS_SIZE` = 16.

## Settings (`SharpenDesc`, `src/desc.rs`)

`enabled` (default false) and `sharpness` (default 0.9, finite in 0..=1). `validate` refuses
anything else with `RaaError::FrameGraph { code: "post.invalid", message }`, the message being
``"`sharpness` must be in [0, 1], got <v>"``.

## Kernel (`src/kernel.rs`)

The kernel is solstice's contrast-adaptive sharpening. Solstice sharpens display values; the RAA
chain is HDR, so every tap goes through a reversible range compression `c / (1 + max(c))` first
and the result through its inverse `d / (1 - min(max(d), 0.999))`. The result stays inside the
range of its neighbours, so high-contrast edges get no halo; sharpness 0 returns the centre.

Adapted from solstice@7e465b5 `crates/solstice-render/shaders/post/tonemap.wgsl`, MIT licence,
Copyright (c) 2026 SamG-Coder.

## Cel lines

Sharpening also sharpens cel lines when both passes are on. Sharpening is off by default.
