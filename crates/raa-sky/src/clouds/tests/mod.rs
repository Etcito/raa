//! Focused tests of the M2 cloud backend.

mod history;
mod noise;
mod passes;
mod resources;
mod shaders;

use raa_math::Mat4;
use raa_scene::{Camera, DirectionalLight};

use super::super::CloudFrame;
use crate::time_of_day::TimeOfDayState;
use crate::weather::Weather;

/// A marchable-looking frame; tests change single fields with struct update syntax.
fn test_frame(view: u32, invocation: u64, size: (u32, u32)) -> CloudFrame {
    CloudFrame {
        invocation,
        view,
        size,
        camera: Camera::default(),
        previous_view_proj: Mat4::IDENTITY,
        history_valid: true,
        weather: Weather::default(),
        sky: TimeOfDayState::default().sky,
        dominant: DirectionalLight::default(),
        animation_seconds: 0.0,
        wind_offset: [0.0, 0.0],
        wind_velocity: [0.0, 0.0],
        evolution: 0.0,
        weather_enabled: true,
        cloud_shadows: true,
        physical: true,
    }
}
