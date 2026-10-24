//! Seeded CPU twin of the particle spawn hash used by the `particles.sim`
//! compute pass.
//!
//! # Hash
//!
//! [`pcg_hash`] is the PCG-RXS-M-XS 32-bit output function of Jarzynski and
//! Olano, *Hash Functions for GPU Rendering* (JCGT 2020):
//!
//! ```text
//! state = input * 747796405 + 2891336453        (wrapping u32)
//! word  = ((state >> ((state >> 28) + 4)) ^ state) * 277803737
//! hash  = (word >> 22) ^ word
//! ```
//!
//! # Spawn recipe
//!
//! A sample is a pure function of `(desc.seed, serial)`. The WGSL simulation
//! mirrors every step below in `f32` arithmetic, with the same operation
//! order (no reassociation), so CPU and GPU agree bit for bit.
//!
//! 1. `h0 = pcg_hash(seed ^ pcg_hash(serial))`.
//! 2. Draw `u_i = (h_i >> 8) as f32 / 16777216.0` (24 bits, so `u` is in
//!    `[0, 1)`), then advance `h_{i+1} = pcg_hash(h_i)`. Draws are consumed
//!    in this order:
//!    - `u0, u1, u2`: position, one per axis `x, y, z`;
//!    - `u3`: cone polar angle, `u4`: cone azimuth;
//!    - `u5`: speed, `u6`: life, `u7`: size;
//!    - `u8`: rotation.
//! 3. `position = desc.position + (2u - 1) * volume_half` per axis.
//! 4. Direction: normalize `desc.direction`, build an orthonormal basis
//!    `(t, b, n)` with the branchless method of Duff et al. 2017 (the same
//!    basis as the S10 `default_tangent`). With
//!    `cos_theta = mix(1.0, cos(spread_rad), u3)` and `phi = 2pi * u4`,
//!    the local cone direction is
//!    `l = (sin_theta * cos(phi), sin_theta * sin(phi), cos_theta)` and
//!    `dir = t * l.x + b * l.y + n * l.z`. A half-angle of `pi` therefore
//!    covers the whole sphere.
//! 5. `speed = mix(desc.speed[0], desc.speed[1], u5)`,
//!    `life = mix(desc.lifetime_s[0], desc.lifetime_s[1], u6)`,
//!    `size = mix(desc.size[0], desc.size[1], u7)`.
//!    `mix(a, b, t) = a * (1 - t) + b * t`, exactly as WGSL `mix`.
//! 6. `velocity = dir * speed`; `rotation = 2pi * u8`.
//! 7. `color_index = h9 % colors.len()`, where `h9` is the hash state left
//!    after the nine draws. An empty palette (only possible for an
//!    unvalidated description) yields index 0.
//!
//! Unvalidated descriptions never panic: a zero-length direction falls back
//! to `[0, 1, 0]`.
//!
//! # Particle state record
//!
//! The GPU state record is 48 bytes ([`PARTICLE_STRIDE`]) per particle:
//!
//! | offset | field |
//! |---|---|
//! | 0  | `pos.xyz` (f32 x3), `age` (f32) |
//! | 16 | `vel.xyz` (f32 x3), `life` (f32) |
//! | 32 | `size` (f32), `rotation` (f32) |
//! | 40 | `color_index` (u32), `kind` (u32) |
//!
//! Fields are little-endian, written with `to_le_bytes`, no padding.

/// The GPU particle state record stride in bytes ([`PARTICLE_STRIDE`]).
pub const PARTICLE_STRIDE: u64 = 48;

/// PCG-RXS-M-XS 32-bit hash (Jarzynski and Olano 2020).
pub fn pcg_hash(v: u32) -> u32 {
    let state = v.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

/// One deterministic spawn sample. See the [module docs](self) for the recipe.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SpawnSample {
    /// Spawn position in world space.
    pub position: [f32; 3],
    /// Initial velocity in m/s.
    pub velocity: [f32; 3],
    /// Lifetime in seconds.
    pub life: f32,
    /// Particle size in metres.
    pub size: f32,
    /// Rotation around the view axis in radians.
    pub rotation: f32,
    /// Index into [`crate::EmitterDesc::colors`].
    pub color_index: u32,
}

/// Deterministic spawn sample for `(desc.seed, serial)`; mirrored by the
/// `particles.sim` WGSL. See the [module docs](self) for the exact recipe.
pub fn spawn_sample(desc: &crate::EmitterDesc, serial: u32) -> SpawnSample {
    let mut h = pcg_hash(desc.seed ^ pcg_hash(serial));

    let u0 = draw(&mut h);
    let u1 = draw(&mut h);
    let u2 = draw(&mut h);
    let u3 = draw(&mut h);
    let u4 = draw(&mut h);
    let u5 = draw(&mut h);
    let u6 = draw(&mut h);
    let u7 = draw(&mut h);
    let u8 = draw(&mut h);
    let color_hash = h;

    let position = [
        desc.position[0] + (2.0 * u0 - 1.0) * desc.volume_half[0],
        desc.position[1] + (2.0 * u1 - 1.0) * desc.volume_half[1],
        desc.position[2] + (2.0 * u2 - 1.0) * desc.volume_half[2],
    ];

    let n = normalize_or(desc.direction, [0.0, 1.0, 0.0]);
    let (t, b) = orthonormal_basis(n);
    let cos_spread = desc.spread_rad.cos();
    let cos_theta = mix(1.0, cos_spread, u3);
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = 2.0 * std::f32::consts::PI * u4;
    let (sin_phi, cos_phi) = phi.sin_cos();
    let local = [sin_theta * cos_phi, sin_theta * sin_phi, cos_theta];
    let direction = [
        t[0] * local[0] + b[0] * local[1] + n[0] * local[2],
        t[1] * local[0] + b[1] * local[1] + n[1] * local[2],
        t[2] * local[0] + b[2] * local[1] + n[2] * local[2],
    ];

    let speed = mix(desc.speed[0], desc.speed[1], u5);
    let life = mix(desc.lifetime_s[0], desc.lifetime_s[1], u6);
    let size = mix(desc.size[0], desc.size[1], u7);
    let rotation = 2.0 * std::f32::consts::PI * u8;

    let color_count = desc.colors.len();
    let color_index = if color_count == 0 {
        0
    } else {
        color_hash % color_count as u32
    };

    SpawnSample {
        position,
        velocity: [
            direction[0] * speed,
            direction[1] * speed,
            direction[2] * speed,
        ],
        life,
        size,
        rotation,
        color_index,
    }
}

fn draw(h: &mut u32) -> f32 {
    let u = (*h >> 8) as f32 / 16_777_216.0;
    *h = pcg_hash(*h);
    u
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

fn normalize_or(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len_sq <= 0.0 || !len_sq.is_finite() {
        return fallback;
    }
    let inv = 1.0 / len_sq.sqrt();
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

/// Branchless orthonormal basis around `n` (Duff et al. 2017).
fn orthonormal_basis(n: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let sign = if n[2] >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (sign + n[2]);
    let b = n[0] * n[1] * a;
    (
        [1.0 + sign * n[0] * n[0] * a, sign * b, -sign * n[0]],
        [b, sign + n[1] * n[1] * a, -n[1]],
    )
}
