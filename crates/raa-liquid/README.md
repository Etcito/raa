# raa-liquid

Animated liquid surfaces of the RAA lit renderer. The crate owns the material functions
`raa.liquid.water` and `raa.liquid.slime` (Transmissive) and `raa.liquid.mud` (Opaque,
wet clearcoat), with Gerstner vertex stages, ripple rings from instance impacts, scrolling
detail normals, shore and crest foam, and refraction through the `Transmission` slot.
Depends on raa-core, raa-shader, raa-lit and raa-matfn; raa-transmission and raa-brdf are
dev-dependencies for the variants test of the built-in functions.

This skeleton ships the shared Gerstner CPU twin and the parameter contract every WGSL file
of the crate follows; the built-in material functions and their `install` follow in the
next task.

## Public items

- `WAVE_COUNT` (4): number of waves packed in the material record's `p` array.
- `wave_height(params, x, z, t) -> f32`: sum of the rest-position heights
  `A sin(k (D.x x + D.y z - speed t))` of the four waves.

`params` is `[[f32; 4]; 8]`: `p[i]` for `i` in `0..4` is wave `i` as `(amplitude A in m,
wavelength L in m, direction angle theta in rad in XZ from +X towards +Z, steepness Q in
[0, 1])`; `p[4]` holds the four speeds in m/s; `p[5..8]` are function-specific and are not
read by the twin. A wave with `L <= 0`, `A == 0` or any non-finite field contributes 0, so
the result is never NaN.

The `wave_height` rustdoc also fixes the horizontal Gerstner displacement
`dxz = D Qi A cos(phase)` with `Qi = clamp(Q / (k A WAVE_COUNT), 0, 1)`, the displaced
position and the analytic normal from the GPU Gems 1 chapter 1 partial derivatives, so the
WGSL vertex stages and the CPU twin stay identical.

## Testing

```
cargo test -p raa-liquid
```

`tests/waves.rs` (CPU only) checks on a grid that the height is finite with degenerate waves
(`L = 0`, NaN amplitude and other non-finite fields), that one wave is periodic in time with
period `L / speed` and along its direction with period `L`, that four waves with commensurate
periods are periodic, and that the height is bounded by the sum of the amplitudes.

The crate must also stay clean for the web target:

```
cargo clippy -p raa-liquid --target wasm32-unknown-unknown -- -D warnings
```
