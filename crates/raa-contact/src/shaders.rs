//! WGSL source of the `shadow.contact` compute pass and the size of its uniform.
//!
//! Standalone (no `crate::` paths) so tests/variants.rs can compile it with `#[path]`.

/// Compute module, entry `contact` (8x8), group 0 only: 0 depth, 1 `ContactParams`,
/// 2 Rgba16Float storage output (value in .r, 1 = lit).
pub(crate) const CONTACT_WGSL: &str = include_str!("wgsl/contact.wgsl");

/// Byte size of `ContactParams`: proj @0, inv_proj @64, viewport @128, sun_view @144,
/// params @160.
pub(crate) const PARAMS_SIZE: u64 = 176;
