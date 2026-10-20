//! Perspective camera and an orbit controller that drives it.

use raa_math::{Mat4, Vec3};

/// Pitch limit of [`OrbitController`]: 89 degrees, so the view never looks straight
/// along the up axis (where `look_at` degenerates).
const MAX_PITCH_RAD: f32 = 89.0 * std::f32::consts::PI / 180.0;

/// A right-handed perspective camera looking from `eye` at `target`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Camera {
    /// Camera position in world space.
    pub eye: Vec3,
    /// Point the camera looks at.
    pub target: Vec3,
    /// Up direction used to orient the view.
    pub up: Vec3,
    /// Vertical field of view in radians.
    pub fov_y_rad: f32,
    /// Near clip distance (> 0).
    pub near: f32,
    /// Far clip distance (> near).
    pub far: f32,
    /// Viewport width divided by height.
    pub aspect: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            eye: Vec3::new(0.0, 2.0, 6.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov_y_rad: 60.0_f32.to_radians(),
            near: 0.1,
            far: 200.0,
            aspect: 16.0 / 9.0,
        }
    }
}

impl Camera {
    /// World-to-view matrix (`Mat4::look_at_rh`).
    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye, self.target, self.up)
    }

    /// View-to-clip matrix with clip depth [0, 1] (`Mat4::perspective_rh_zo`).
    pub fn projection(&self) -> Mat4 {
        Mat4::perspective_rh_zo(self.fov_y_rad, self.aspect, self.near, self.far)
    }

    /// `projection * view`.
    pub fn view_proj(&self) -> Mat4 {
        self.projection() * self.view()
    }

    /// Sets the aspect from a viewport size in pixels. Zero sizes count as 1, so the
    /// aspect stays finite and positive.
    pub fn set_viewport(&mut self, width: u32, height: u32) {
        self.aspect = width.max(1) as f32 / height.max(1) as f32;
    }
}

/// Orbits a camera around a target on a sphere of radius `distance`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct OrbitController {
    /// Point orbited around.
    pub target: Vec3,
    /// Distance from the target to the eye, kept in `[min_distance, max_distance]`.
    pub distance: f32,
    /// Rotation around Y in radians; 0 puts the eye on +Z of the target.
    pub yaw_rad: f32,
    /// Elevation in radians, kept in +-89 degrees.
    pub pitch_rad: f32,
    /// Smallest allowed distance.
    pub min_distance: f32,
    /// Largest allowed distance.
    pub max_distance: f32,
}

impl OrbitController {
    /// A controller with yaw and pitch 0 and distance limits [0.1, 1000]. The distance
    /// is clamped to those limits (a non-finite distance becomes the minimum).
    pub fn new(target: Vec3, distance: f32) -> Self {
        let min_distance = 0.1;
        let max_distance = 1000.0;
        let distance = if distance.is_finite() {
            distance.clamp(min_distance, max_distance)
        } else {
            min_distance
        };
        OrbitController {
            target,
            distance,
            yaw_rad: 0.0,
            pitch_rad: 0.0,
            min_distance,
            max_distance,
        }
    }

    /// Adds the deltas to yaw and pitch, then clamps pitch to +-89 degrees. Non-finite
    /// deltas are ignored.
    pub fn rotate(&mut self, d_yaw: f32, d_pitch: f32) {
        if d_yaw.is_finite() {
            self.yaw_rad += d_yaw;
        }
        if d_pitch.is_finite() {
            self.pitch_rad = (self.pitch_rad + d_pitch).clamp(-MAX_PITCH_RAD, MAX_PITCH_RAD);
        }
    }

    /// Multiplies the distance by `factor`, then clamps it to
    /// `[min_distance, max_distance]`. Non-positive or non-finite factors are ignored.
    pub fn zoom(&mut self, factor: f32) {
        if !(factor.is_finite() && factor > 0.0) {
            return;
        }
        self.distance = (self.distance * factor).clamp(self.min_distance, self.max_distance);
    }

    /// Moves the target in the camera plane: `dx * distance` along the camera right
    /// vector and `dy * distance` along the camera up vector. The distance is unchanged.
    /// Non-finite deltas are ignored.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        if !(dx.is_finite() && dy.is_finite()) {
            return;
        }
        let (right, up) = self.basis();
        self.target += right * (dx * self.distance) + up * (dy * self.distance);
    }

    /// Places the camera: `eye = target + distance * (cos p sin y, sin p, cos p cos y)`,
    /// looking at the target with up = Y.
    pub fn apply(&self, cam: &mut Camera) {
        cam.eye = self.target + self.offset_dir() * self.distance;
        cam.target = self.target;
        cam.up = Vec3::Y;
    }

    /// Unit vector from the target to the eye.
    fn offset_dir(&self) -> Vec3 {
        let (sy, cy) = self.yaw_rad.sin_cos();
        let (sp, cp) = self.pitch_rad.sin_cos();
        Vec3::new(cp * sy, sp, cp * cy)
    }

    /// Camera right and up vectors (unit length, orthogonal to the view direction).
    fn basis(&self) -> (Vec3, Vec3) {
        let (sy, cy) = self.yaw_rad.sin_cos();
        // Horizontal, so it is well defined at every clamped pitch.
        let right = Vec3::new(cy, 0.0, -sy);
        let forward = -self.offset_dir();
        let up = right.cross(forward);
        (right, up)
    }
}
