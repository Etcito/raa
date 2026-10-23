//! Screen-space contact shadows (S12 `shadow.contact`, order 45).
//!
//! A compute pass marches CONTACT = 24 jittered steps from each pixel's view position toward
//! the sun over at most 0.35 m and counts the pixel occluded when the depth buffer lies in
//! front of the ray by less than the thickness (0.05 m). The result (1 = lit) reaches the
//! lighting through S10 group 1 binding 8. [`ContactDesc`] holds the tunable values.

mod desc;

pub use desc::ContactDesc;
