//! Small math types (vectors, matrices, quaternions) with no heavy dependency.
//!
//! f32 everywhere, column-major matrices with column vectors, right-handed, Y up.

mod aabb;
mod mat4;
mod quat;
mod vec;

pub use aabb::Aabb;
pub use mat4::Mat4;
pub use quat::Quat;
pub use vec::{Vec3, Vec4};

/// True when `a` and `b` differ by at most `eps` (absolute difference).
pub fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}
