//! Fixed-tick sixteen-body spring and rotation workload; see the README
//! for the initialization, integration and hash stream contracts.

use crate::{DET_HASH, TICK_SECONDS_F32, Tick, nlerp, normalize, quat_mul, trs_compose};

/// Canonical `(seed, ticks)` fixtures, in native/Node/browser record order.
pub const DET_CASES: [(u32, Tick); 3] = [(0, 0), (1, 60), (305419896, 600)];

#[derive(Clone, Copy)]
struct Body {
    position: [f32; 3],
    velocity: [f32; 3],
    anchor: [f32; 3],
    rotation: [f32; 4],
    increment: [f32; 4],
}

// Wrapping LCG includes seed zero. The upper 24 bits convert exactly to f32.
fn sample(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    ((*state >> 8) as f32 * (1.0 / 16_777_216.0)) * 2.0 - 1.0
}

fn body(state: &mut u32) -> Body {
    let anchor = core::array::from_fn(|_| sample(state));
    let position = core::array::from_fn(|i| anchor[i] + sample(state) * 0.5);
    let velocity = core::array::from_fn(|_| sample(state) * 0.25);
    let rotation = normalize([sample(state), sample(state), sample(state), 1.0])
        .expect("bounded quaternion with nonzero w");
    let half_step = 0.5 * TICK_SECONDS_F32;
    let increment = normalize([
        sample(state) * half_step,
        sample(state) * half_step,
        sample(state) * half_step,
        1.0,
    ])
    .expect("bounded angular increment with nonzero w");
    Body {
        position,
        velocity,
        anchor,
        rotation,
        increment,
    }
}

/// Hash every completed tick of sixteen deterministically seeded bodies.
/// Uses semi-implicit Euler at ANIM_TICK_HZ with spring stiffness 8 and
/// damping 2, plus normalized quaternion multiplication and nlerp.
/// Stream order: tick, body 0..15, position xyz, velocity xyz, rotation
/// xyzw, then the 16 column-major unit-scale TRS components. Zero ticks
/// hashes no values and returns FNV1A32_OFFSET for every seed.
pub fn selftest(seed: u32, ticks: Tick) -> u32 {
    let mut state = seed;
    let mut bodies: [Body; 16] = core::array::from_fn(|_| body(&mut state));
    let mut hash = DET_HASH;
    for _ in 0..ticks {
        for body in &mut bodies {
            for axis in 0..3 {
                let spring = (body.anchor[axis] - body.position[axis]) * 8.0;
                let acceleration = spring - body.velocity[axis] * 2.0;
                body.velocity[axis] += acceleration * TICK_SECONDS_F32;
                body.position[axis] += body.velocity[axis] * TICK_SECONDS_F32;
            }
            let next = normalize(
                quat_mul(body.rotation, body.increment).expect("bounded quaternion product"),
            )
            .expect("nonzero quaternion product");
            body.rotation = nlerp(body.rotation, next, 0.5).expect("nearby unit rotations");
            let model = trs_compose(body.position, body.rotation, [1.0; 3])
                .expect("bounded position and normalized rotation");
            hash.f32s(&body.position);
            hash.f32s(&body.velocity);
            hash.f32s(&body.rotation);
            hash.f32s(&model);
        }
    }
    hash.finish()
}
