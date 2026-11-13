// S32 depth helper: the one place that knows RAA's depth convention.
//
// RAA is standard-Z: clip depth runs 0..1 from the near plane to the far plane, the depth
// buffer is cleared to 1 and the depth test is `Less`. Solstice is reversed-Z, so every
// shader adapted from solstice calls these helpers instead of hand-flipping a comparison,
// a min/max reduction or an atomic depth. Item names and bodies follow S32.
//
// This module declares no binding, no entry point and depends on nothing else (not even the
// prelude), so a pass-private module can include it alone.

const RAA_DEPTH_FAR: f32 = 1.0;

// True when `a` is nearer than `b` under standard-Z (smaller depth wins).
fn raa_depth_nearer(a: f32, b: f32) -> bool {
    return a < b;
}

// The nearer of two depths.
fn raa_depth_nearest(a: f32, b: f32) -> f32 {
    return min(a, b);
}

// The farther of two depths.
fn raa_depth_farthest(a: f32, b: f32) -> f32 {
    return max(a, b);
}

// True where `d` is the far plane or an empty pixel.
fn raa_depth_is_far(d: f32) -> bool {
    return d >= 1.0;
}

// Order-preserving standard-Z key: larger is nearer, 0 is far or empty. It orders exactly
// like the reversed-Z bit pattern the solstice visibility write compares as an integer.
fn raa_depth_key(d: f32) -> u32 {
    return bitcast<u32>(1.0 - clamp(d, 0.0, 1.0));
}

// Inverse of `raa_depth_key`.
fn raa_depth_from_key(k: u32) -> f32 {
    return 1.0 - bitcast<f32>(k);
}
