# raa-brdf

Layered BRDF (base, sheen, clearcoat) of the lit renderer (S11 `Direct` slot) and its CPU
reference.

## Purpose

`raa-brdf` holds the layered `Direct` model. This crate has the CPU reference that the
tests use; the WGSL provider (`providers()`, `install`, `BRDF_WGSL`) comes in a later task
and mirrors `src/cpu.rs` line by line: the same operations, guards and fit coefficients.

The model, for a unit normal `n`, view `v` and light `l` (`mu = max(n.v, 1e-4)`, as in the
shading context):

- **Base**: Lambert x (1 - metallic), plus GGX (`alpha = roughness^2`, roughness clamped to
  [0.03, 1]) with height-correlated Smith visibility, Schlick Fresnel and the multiscatter
  compensation `1 + F0 (1 / E(mu) - 1)`. The Lambert lobe is weighted by
  `1 - (F0 A + B) (1 + F0 (1 / E - 1))`, which is what the compensated specular leaves.
  Without this weight a white dielectric reflects up to about 1.5 at grazing view.
- **Sheen**: Charlie NDF (Estevez and Kulla 2017) x Neubelt visibility, sheen roughness
  clamped to [0.07, 1]. The base is scaled by `1 - max(sheen_color) min(E_sheen, 1)`.
  Neubelt over-reflects at grazing view (`E_sheen` reaches about 5.7). Where `E_sheen > 1`
  the sheen lobe is divided by `E_sheen`, so a white sheen reflects at most what it receives.
- **Clearcoat**: GGX with F0 0.04 and Kelemen visibility `1 / (4 (l.h)^2)` on the geometric
  normal. Base and sheen are attenuated by `1 - clearcoat Fc(n.v)`. The reference takes one
  normal; in WGSL the coat uses the geometric normal and the other layers the shading normal.

With clearcoat 0 the result is exactly the under-coat result. With sheen 0 too, it is exactly
the base result.

### Fits

Both fits come from this crate's own minimax fitting of numerical integrals. The integrals
use Gauss-Legendre quadrature over the exact `n.l > 0` half-vector domain, graded toward the
lobe peak and the horizon (converged below 1e-6). Fit and check points:

| Fit | Points | Range | Extra check |
|---|---|---|---|
| `ggx_dfg_fit` | 1189 | roughness [0.03, 1] x n.v [1e-4, 1] | 400 random points |
| `sheen_albedo_fit` | 1131 | sheen roughness [0.07, 1] x n.v [1e-4, 1] | 300 random points |

Both fits are shifted so they over-estimate the albedo they approximate. That shift keeps
the layered albedo at or below 1.

| Fit | Form (`s = mu / (mu + r^2)`, `y = mu / r`) | Over-estimate |
|---|---|---|
| `E = A + B` | `P(r, s)`, degree 5 x 5, x 1.006, clamped to [0.25, 1] | 0.08 % to 1.1 % |
| `B` | `(1 - mu)^5 Q(sqrt r, s) + 0.006`, degree 4 x 4, clamped to `E` | 0.0006 to 0.012 |
| `E_sheen` | `exp(P(r, min(y, 3)) + sqrt(1 - mu^2) (c0 + c1 r + c2 r^2)) / r`, degree 3 x 4 | 0.4 % to 6.8 % |

The GGX integral is the one in the `raa-ibl` DFG LUT. For `E_sheen`, the three.js sheen fit
does not match Charlie x Neubelt (0.027 against 0.141 at roughness 1, n.v 1), which is why
the crate has its own fit.

## Public items

- `DirectTerms`: `base_color`, `metallic`, `roughness`, `f0`, `clearcoat`,
  `clearcoat_roughness`, `sheen_color` and `sheen_roughness`. These are the `MaterialOutput`
  fields of the same names, plus `f0` from the shading context.
- `f0_from_ior(base_color, metallic, ior)`: same as `raa_pass_f0`.
- `eval_direct(&terms, n, v, l) -> [f32; 3]`: the full model. It returns the reflected
  radiance per unit irradiance x cos (`f(v, l) n.l`), and 0 below the horizon.
- `eval_under_coat`: the model without the clearcoat (base and sheen).
- `eval_base`: the base layer alone.
- `ggx_dfg_fit(n_dot_v, roughness) -> [A, B]` and `sheen_albedo_fit(n_dot_v, sheen_roughness)`:
  the albedo fits.
- Constants:
  - `MIN_ROUGHNESS` (0.03)
  - `MIN_SHEEN_ROUGHNESS` (0.07)
  - `MIN_N_DOT_V` (1e-4)
  - `CLEARCOAT_F0` (0.04)

## Test

Run `cargo test -p raa-brdf`:

- `tests/energy.rs` (CPU white furnace): integrates `eval_direct` over the hemisphere for a
  white base on this grid:
  - roughness {0.05, 0.25, 0.5, 0.75, 1}, also used as the clearcoat and sheen roughness
  - n.v {0.1, 0.5, 1}
  - metallic {0, 1}
  - clearcoat {0, 1}
  - sheen {0, 1} (white sheen color)

  The integration is deterministic stratified (midpoint) sampling of the half-vector domain
  over the exact `n.l > 0` region, with polar strata graded toward the lobe peak and the
  horizon. It is within 1e-4 of the same quadrature with 4x finer strata. A second test
  checks that it integrates `n.l / pi` to 1 within 2e-4. The test asserts:
  - albedo <= 1 + 1e-3 everywhere;
  - clearcoat 0 gives exactly the under-coat result, and with sheen 0 exactly `eval_base`,
    on every sample;
  - the base alone keeps at least 0.97 of the energy.

  It uses `NMC_CPUS` threads when that is above 1.
- `src/cpu.rs`:
  - the fits stay above reference values of their integrals, within the stated bounds;
  - `f0_from_ior` matches the pass;
  - degenerate directions give finite, non-negative values.
