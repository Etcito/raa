# raa-grading

HDR colour grading before tonemap (S12 `grading`, order 130): white balance, exposure,
log-space contrast, ASC CDL lift/gamma/gain, split toning, luma-preserving saturation and
vibrance, and vignette. Depends on raa-core, raa-pipeline and raa-shader.

Public items:
- `GradingDesc { temperature_k, tint, exposure_ev, contrast, saturation, vibrance, lift, gamma, gain, shadows_tint, highlights_tint, split_balance, vignette }`: the grade, `Copy`. `Default` is the `neutral` preset: 6500 K, tint 0, 0 EV, contrast 1, saturation 1, vibrance 0, lift 0, gamma 1, gain 1, both tints 0.5 (no tint), split balance 0, vignette 0.
- `GradingDesc::preset(name)`: `neutral`, `candy`, `ice`, `jungle`, `neon`; `None` for any other name.
- `GradingDesc::validate()`: refuses with code `post.invalid` and a message naming the field (e.g. `gamma[1]`). Rules: every value finite; temperature_k in [1500, 15000]; tint in [-1, 1]; exposure_ev in [-10, 10]; contrast in (0, 4]; saturation in [0, 4]; vibrance in [-1, 1]; gamma > 0; gain and both tints >= 0; split_balance in [-1, 1]; vignette in [0, 1]. Lift has no range beyond finite.
- `white_balance_matrix(temperature_k, tint) -> [[f32; 3]; 3]`: CAT02 in LMS against D65, identity at 6500 K and tint 0. Added by the grading feature task; not in this crate yet.
- `GradingFeature::install(gpu, pipeline) -> Result<GradingHandle, RaaError>`: the FeaturePass that runs fullscreen `hdr.src` to `hdr.dst` in this order: white balance, exposure, log-space contrast, lift/gamma/gain (ASC CDL in log), split toning by luminance, luma-preserving saturation and vibrance, vignette (cos^4 natural times the artistic amount); then swaps. Added by the grading feature task; not in this crate yet.
- `GradingHandle` (`Clone`): `set(desc)` and `get()`. Added by the grading feature task; not in this crate yet.

Test: `cargo test -p raa-grading`. `tests/desc.rs` checks that every preset validates, that unknown names give `None`, and that each field refuses NaN and an out-of-range value with `post.invalid` naming it.
