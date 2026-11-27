//! Pure whole-object billboard selection for `m6-render-contract`.
//!
//! The eight baked directions are object-local camera positions around +Y at
//! angle `k * pi/4`: `direction = (sin(angle), 0, cos(angle))`. Their ranks are
//! +Z=0, +X+Z=1, +X=2, +X-Z=3, -Z=4, -X-Z=5, -X=6, -X+Z=7. A view selects the
//! direction with the largest dot product against its normalized object-local
//! horizontal camera direction; equal maxima choose the lowest rank, and a zero
//! horizontal direction selects rank 0.
//!
//! Level selection is stateless: the caller passes the level chosen for the
//! previous frame. Automatic LOD1 entry is below `0.9 * switch_pixels * 2^bias`
//! and LOD0 re-entry is above `1.1 * switch_pixels * 2^bias`, which gives the
//! 10% hysteresis band. A forced level bypasses size and hysteresis but still
//! passes validation.

use raa_core::RaaError;

use crate::convert::api_error;

/// Stable code of every refusal from [`select_billboard`].
pub const BILLBOARD_INVALID: &str = "billboard.invalid";

/// Two dot products this close are a mathematical tie; the lowest rank wins.
///
/// The direction table is built from `FRAC_1_SQRT_2` while a midpoint camera is
/// built from `sin`/`cos`, so an exact tie differs by a few ULPs in `f32`.
const TIE_EPSILON: f32 = 1e-6;

/// The eight directions in rank order.
const DIRECTIONS: [[f32; 3]; 8] = [
    [0.0, 0.0, 1.0],
    [
        std::f32::consts::FRAC_1_SQRT_2,
        0.0,
        std::f32::consts::FRAC_1_SQRT_2,
    ],
    [1.0, 0.0, 0.0],
    [
        std::f32::consts::FRAC_1_SQRT_2,
        0.0,
        -std::f32::consts::FRAC_1_SQRT_2,
    ],
    [0.0, 0.0, -1.0],
    [
        -std::f32::consts::FRAC_1_SQRT_2,
        0.0,
        -std::f32::consts::FRAC_1_SQRT_2,
    ],
    [-1.0, 0.0, 0.0],
    [
        -std::f32::consts::FRAC_1_SQRT_2,
        0.0,
        std::f32::consts::FRAC_1_SQRT_2,
    ],
];

fn invalid(message: impl Into<String>) -> RaaError {
    api_error(BILLBOARD_INVALID, message)
}

/// Handoff: "Prepare billboard groups for each camera" calls [`select_billboard`]
/// once per view to choose the whole-object level and the baked direction.
///
/// Today this function validates `camera_local`, `projected_diameter_px`,
/// `switch_pixels`, `bias`, `previous` and `forced`, then returns
/// `(level, direction)` with level 0 or 1 and direction 0..7. Direction uses the
/// normalized object-local horizontal camera position. Level uses the previous
/// level's hysteresis band unless `forced` is set.
///
/// # Errors
///
/// Every scalar input must be finite, `switch_pixels` must be positive, `bias`
/// must be in -3..=3, and `previous`/`forced` must be 0 or 1. Otherwise the
/// refusal carries [`BILLBOARD_INVALID`].
pub fn select_billboard(
    camera_local: [f32; 3],
    projected_diameter_px: f32,
    switch_pixels: f32,
    bias: i8,
    previous: u8,
    forced: Option<u8>,
) -> Result<(u8, u8), RaaError> {
    validate(
        camera_local,
        projected_diameter_px,
        switch_pixels,
        bias,
        previous,
        forced,
    )?;
    let direction = direction_rank(camera_local);
    let level = match forced {
        Some(level) => level,
        None => automatic_level(projected_diameter_px, switch_pixels, bias, previous),
    };
    Ok((level, direction))
}

fn validate(
    camera_local: [f32; 3],
    projected_diameter_px: f32,
    switch_pixels: f32,
    bias: i8,
    previous: u8,
    forced: Option<u8>,
) -> Result<(), RaaError> {
    if !camera_local.iter().all(|value| value.is_finite()) {
        return Err(invalid("camera_local must contain only finite numbers"));
    }
    if !projected_diameter_px.is_finite() {
        return Err(invalid("projected_diameter_px must be finite"));
    }
    if !switch_pixels.is_finite() || switch_pixels <= 0.0 {
        return Err(invalid("switch_pixels must be finite and positive"));
    }
    if !(-3..=3).contains(&bias) {
        return Err(invalid("bias must be in -3..=3"));
    }
    if previous > 1 {
        return Err(invalid("previous must be 0 or 1"));
    }
    if let Some(level) = forced
        && level > 1
    {
        return Err(invalid("forced must be 0 or 1"));
    }
    Ok(())
}

/// Automatic level from the previous level's 10% hysteresis band.
fn automatic_level(projected_diameter_px: f32, switch_pixels: f32, bias: i8, previous: u8) -> u8 {
    let threshold = switch_pixels * 2.0_f32.powi(i32::from(bias));
    if previous == 0 {
        u8::from(projected_diameter_px < threshold * 0.9)
    } else {
        u8::from(projected_diameter_px <= threshold * 1.1)
    }
}

/// Rank 0..7 of the normalized object-local horizontal camera direction.
fn direction_rank(camera_local: [f32; 3]) -> u8 {
    let (x, z) = (camera_local[0], camera_local[2]);
    let length = x.hypot(z);
    if !length.is_finite() || length == 0.0 {
        return 0;
    }
    let (nx, nz) = (x / length, z / length);
    let dot = |direction: &[f32; 3]| direction[0] * nx + direction[2] * nz;
    let mut best_rank = 0;
    let mut best = dot(&DIRECTIONS[0]);
    for (rank, direction) in DIRECTIONS.iter().enumerate().skip(1) {
        let value = dot(direction);
        if value > best + TIE_EPSILON {
            best = value;
            best_rank = rank as u8;
        }
    }
    best_rank
}

#[cfg(test)]
mod tests {
    use super::*;

    const SWITCH: f32 = 100.0;

    fn select(projected: f32, previous: u8) -> (u8, u8) {
        select_billboard([0.0, 0.0, 1.0], projected, SWITCH, 0, previous, None)
            .expect("valid input")
    }

    #[test]
    fn cardinal_camera_directions_have_their_rank() {
        let cases = [
            ([0.0, 0.0, 1.0], 0),
            ([1.0, 0.0, 1.0], 1),
            ([1.0, 0.0, 0.0], 2),
            ([1.0, 0.0, -1.0], 3),
            ([0.0, 0.0, -1.0], 4),
            ([-1.0, 0.0, -1.0], 5),
            ([-1.0, 0.0, 0.0], 6),
            ([-1.0, 0.0, 1.0], 7),
        ];
        for (camera, rank) in cases {
            assert_eq!(direction_rank(camera), rank, "camera {camera:?}");
        }
        assert_eq!(direction_rank([5.0, 0.0, 5.0]), 1, "magnitude is ignored");
        assert_eq!(direction_rank([-2.0, 0.0, 2.0]), 7);
    }

    #[test]
    fn all_eight_midpoints_choose_the_lower_rank() {
        for k in 0..8_u8 {
            let angle = (f32::from(k) + 0.5) * std::f32::consts::FRAC_PI_4;
            let camera = [angle.sin(), 0.0, angle.cos()];
            let expected = if k == 7 { 0 } else { k };
            assert_eq!(direction_rank(camera), expected, "midpoint after {k}");
        }
    }

    #[test]
    fn zero_horizontal_direction_selects_zero() {
        assert_eq!(direction_rank([0.0, 0.0, 0.0]), 0);
        assert_eq!(direction_rank([0.0, 12.0, 0.0]), 0);
        assert_eq!(
            select_billboard([0.0, 5.0, 0.0], 500.0, SWITCH, 0, 0, None),
            Ok((0, 0))
        );
    }

    #[test]
    fn vertical_component_does_not_change_facing() {
        assert_eq!(direction_rank([1.0, 0.0, 1.0]), 1);
        assert_eq!(direction_rank([1.0, 400.0, 1.0]), 1);
        assert_eq!(direction_rank([1.0, -400.0, 1.0]), 1);
    }

    #[test]
    fn automatic_lod_uses_the_nine_percent_hysteresis() {
        assert_eq!(select(89.0, 0), (1, 0));
        assert_eq!(select(90.0, 0), (0, 0), "entry is strictly below 90");
        assert_eq!(select(500.0, 0), (0, 0));

        assert_eq!(select(109.0, 1), (1, 0));
        assert_eq!(select(110.0, 1), (1, 0), "re-entry is strictly above 110");
        assert_eq!(select(111.0, 1), (0, 0));
        assert_eq!(select(20.0, 1), (1, 0));
    }

    #[test]
    fn bias_scales_the_switch_threshold() {
        let at = |projected, bias, previous| {
            select_billboard([0.0, 0.0, 1.0], projected, SWITCH, bias, previous, None)
                .expect("valid input")
        };
        assert_eq!(at(179.0, 1, 0), (1, 0));
        assert_eq!(at(180.0, 1, 0), (0, 0));
        assert_eq!(at(220.0, 1, 1), (1, 0));
        assert_eq!(at(221.0, 1, 1), (0, 0));

        assert_eq!(at(44.0, -1, 0), (1, 0));
        assert_eq!(at(45.0, -1, 0), (0, 0));
        assert_eq!(at(55.0, -1, 1), (1, 0));
        assert_eq!(at(56.0, -1, 1), (0, 0));

        assert_eq!(at(719.0, 3, 0), (1, 0));
        assert_eq!(at(720.0, 3, 0), (0, 0));
        assert_eq!(at(11.0, -3, 0), (1, 0));
        assert_eq!(at(12.0, -3, 0), (0, 0));
    }

    #[test]
    fn forced_level_bypasses_size_and_hysteresis() {
        assert_eq!(
            select_billboard([0.0, 0.0, 1.0], 10_000.0, SWITCH, 0, 0, Some(1)),
            Ok((1, 0))
        );
        assert_eq!(
            select_billboard([1.0, 0.0, 0.0], 0.0, SWITCH, 0, 1, Some(0)),
            Ok((0, 2))
        );
        assert_eq!(
            select_billboard([0.0, 0.0, 1.0], 50.0, SWITCH, 3, 1, Some(0)),
            Ok((0, 0)),
            "forced ignores bias too"
        );
    }

    #[test]
    fn invalid_inputs_are_refused_with_billboard_invalid() {
        let cases = [
            ([f32::NAN, 0.0, 1.0], 100.0, SWITCH, 0, 0, None),
            ([0.0, f32::INFINITY, 1.0], 100.0, SWITCH, 0, 0, None),
            ([0.0, 0.0, f32::NEG_INFINITY], 100.0, SWITCH, 0, 0, None),
            ([0.0, 0.0, 1.0], f32::NAN, SWITCH, 0, 0, None),
            ([0.0, 0.0, 1.0], 100.0, 0.0, 0, 0, None),
            ([0.0, 0.0, 1.0], 100.0, -1.0, 0, 0, None),
            ([0.0, 0.0, 1.0], 100.0, f32::INFINITY, 0, 0, None),
            ([0.0, 0.0, 1.0], 100.0, SWITCH, -4, 0, None),
            ([0.0, 0.0, 1.0], 100.0, SWITCH, 4, 0, None),
            ([0.0, 0.0, 1.0], 100.0, SWITCH, 0, 2, None),
            ([0.0, 0.0, 1.0], 100.0, SWITCH, 0, 0, Some(2)),
        ];
        for (camera, projected, switch, bias, previous, forced) in cases {
            let error = select_billboard(camera, projected, switch, bias, previous, forced)
                .expect_err("input must be refused");
            assert_eq!(error.code(), BILLBOARD_INVALID, "input {camera:?}");
        }
    }

    #[test]
    fn nonuniform_transform_normal_matrix_is_inverse_transpose() {
        use raa_math::{Mat4, Quat, Vec3};

        let model = Mat4::from_trs(
            Vec3::new(10.0, 20.0, 30.0),
            Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            Vec3::new(2.0, 4.0, 0.5),
        );
        let normal_matrix = model.normal_matrix();
        let object_normal = Vec3::new(1.0, 2.0, 3.0);
        let transformed = Vec3::new(
            normal_matrix[0][0] * object_normal.x
                + normal_matrix[1][0] * object_normal.y
                + normal_matrix[2][0] * object_normal.z,
            normal_matrix[0][1] * object_normal.x
                + normal_matrix[1][1] * object_normal.y
                + normal_matrix[2][1] * object_normal.z,
            normal_matrix[0][2] * object_normal.x
                + normal_matrix[1][2] * object_normal.y
                + normal_matrix[2][2] * object_normal.z,
        )
        .normalize_or_zero();
        let length = (36.5_f32).sqrt();
        let expected = [6.0 / length, 0.5 / length, -0.5 / length];
        for (actual, wanted) in transformed.to_array().iter().zip(expected) {
            assert!(
                (actual - wanted).abs() < 1e-5,
                "normal {actual} differs from {wanted}"
            );
        }
    }
}
