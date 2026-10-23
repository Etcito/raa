# raa-bloom

Physically based bloom for the lit pipeline (BLOOM values: up to 8 mips, strength 0.04, radius 0.85).

Public items:
- `BloomDesc { strength, radius }` with `Default` and `validate` (`bloom.invalid`, message names the field).
- `mip_sizes(width, height) -> ([(u32, u32); 8], usize)`: level i is the HDR size halved i+1 times (min 1); levels are kept while both sides are >= 8 px, at most 8; 0 levels means the feature passes HDR through.

Test: `cargo test -p raa-bloom`.
