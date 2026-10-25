//! Per-instance squash-and-stretch and jiggle of the lit renderer (S10 `InstanceExtra`, S11 `Deform` slot).
//!
//! Squash scales a point along a unit axis `a` by amount `s` (clamped to `s >= -0.9`):
//! `p' = (p.a) a (1 + s) + (p - (p.a) a) / sqrt(1 + s)`. The determinant is `(1 + s) / (1 + s) = 1`,
//! so the volume is preserved. Normals use the inverse transpose: the parallel part divided by
//! `1 + s`, the perpendicular part times `sqrt(1 + s)`, renormalised ([`squash_normal`]).
//!
//! The jiggle offset is weighted by height: `w(y) = smoothstep(0, 1, clamp(y / h, 0, 1))^2`
//! ([`jiggle_offset_weight`]), where `y` is the height of the vertex along the node's up axis
//! and `h` the instance's `height_m`; the vertex moves by `offset * w(y)` and its normal is tilted
//! by the gradient of that weight.
//!
//! The spring ([`JiggleSprings`]) lags the mass behind the node's world translation: each step the
//! translation delta is subtracted from the lag `x`, then `a = -k x - 2 zeta sqrt(k) v` is integrated
//! by semi-implicit Euler in sub-steps of at most 1/120 s. The lag is expressed in node-local axes,
//! multiplied by `amount`, and clamped to half of `height_m`. The state is stored per tag in a
//! tag-sorted vector, so a given input sequence always gives bit-identical outputs.
//!
//! The `Deform` provider and the `install` function that registers it with the lit renderer come
//! in the next task of this module.

mod jiggle;
mod squash;

pub use jiggle::{JiggleConfig, JiggleSprings};
pub use squash::{jiggle_offset_weight, squash_normal, squash_point};
