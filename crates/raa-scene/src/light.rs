//! The two lights the forward renderer shades with: one directional light and an
//! ambient term.

use raa_math::Vec3;

/// A sun-like light. The direction is the travel direction of the light (from the light
/// towards the scene), normalised.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DirectionalLight {
    /// Travel direction of the light, normalised.
    pub direction: Vec3,
    /// Linear rgb colour.
    pub color: [f32; 3],
    /// Colour multiplier.
    pub intensity: f32,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        DirectionalLight {
            direction: Vec3::new(-0.4, -1.0, -0.3).normalize_or_zero(),
            color: [1.0, 1.0, 1.0],
            intensity: 3.0,
        }
    }
}

/// A uniform ambient term that keeps dark sides readable.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Ambient {
    /// Linear rgb colour.
    pub color: [f32; 3],
    /// Colour multiplier.
    pub intensity: f32,
}

impl Default for Ambient {
    fn default() -> Self {
        Ambient {
            color: [0.6, 0.65, 0.75],
            intensity: 0.3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_math::approx_eq;

    #[test]
    fn default_light_direction_is_normalised_and_points_down() {
        let light = DirectionalLight::default();
        assert!(approx_eq(light.direction.length(), 1.0, 1e-6));
        assert!(light.direction.y < 0.0);
        assert_eq!(light.color, [1.0, 1.0, 1.0]);
        assert_eq!(light.intensity, 3.0);
    }

    #[test]
    fn default_ambient_is_the_contract_value() {
        let ambient = Ambient::default();
        assert_eq!(ambient.color, [0.6, 0.65, 0.75]);
        assert_eq!(ambient.intensity, 0.3);
    }
}
