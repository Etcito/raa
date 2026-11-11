//! The private cloud compute ABI (contract `P2-CLOUD-GPU`).
//!
//! `CloudGpu` is the 192-byte little-endian group-2 binding-0 record of every cloud march
//! module; its first 64 bytes are the Solstice `CloudUniform` twin
//! (`crates/solstice-render/src/clouds/mod.rs:44-60` at revision
//! 7e465b599c13da165948ebaa44df824c36c3ae07, MIT, Copyright (c) 2026 SamG-Coder) and the
//! rest is the RAA camera/history ABI. [`CloudGpu::to_bytes`] is the only serialization
//! boundary and writes explicit little-endian words, never an unsafe cast.
//!
//! [`cloud_sky_parameters`] builds the 256-byte `PhysicalSkyGpu` record the cloud passes
//! receive from the accepted frame alone, so the backend never reads M1 state.

use raa_math::{Mat4, Vec3};

use super::super::CloudFrame;
use crate::atmosphere::{PhysicalSkyGpu, atmospheric_radius_km};

/// Ground covered by the shadow map, in metres along each side.
pub(super) const SHADOW_EXTENT: f32 = 12_000.0;
/// Texels along one side of the ground shadow map.
pub(super) const SHADOW_SIZE: u32 = 256;
/// The panorama, laid out like the sky-view table (`solstice clouds/mod.rs:31-35`).
pub(super) const PANORAMA_SIZE: (u32, u32) = (192, 108);

// Keeps the panorama size reachable until the panorama card allocates its target.
const _: (u32, u32) = PANORAMA_SIZE;

/// The group-2 binding-0 cloud uniform of every cloud march module.
#[repr(C)]
#[derive(Clone, Debug, PartialEq)]
pub(super) struct CloudGpu {
    /// @0: wind drift in metres (x/z).
    pub(super) wind_offset: [f32; 2],
    /// @8: cloud-boil phase in metres.
    pub(super) evolution: f32,
    /// @12: sky coverage in [0, 1].
    pub(super) coverage: f32,
    /// @16: 0 stratus, 0.5 stratocumulus, 1 cumulus.
    pub(super) cloud_type: f32,
    /// @20: cloud-base height in metres.
    pub(super) base: f32,
    /// @24: cloud-top height in metres.
    pub(super) top: f32,
    /// @28: optical thickness relative to ordinary cloud.
    pub(super) density: f32,
    /// @32: high cirrus coverage in [0, 1].
    pub(super) cirrus: f32,
    /// @36: rain intensity in [0, 1].
    pub(super) rain: f32,
    /// @40: ground covered by the shadow map, in metres.
    pub(super) shadow_extent: f32,
    /// @44: 1 when the physical model renders weather, else 0.
    pub(super) enabled: f32,
    /// @48: shadow-map corner in world metres.
    pub(super) shadow_origin: [f32; 2],
    /// @56: unit wind direction in x/z.
    pub(super) wind_direction: [f32; 2],
    /// @64: camera right * tan(fovY/2) * aspect.
    pub(super) right: [f32; 4],
    /// @80: camera up * tan(fovY/2).
    pub(super) up: [f32; 4],
    /// @96: camera forward, w = 1 when history is usable.
    pub(super) forward: [f32; 4],
    /// @112: half-resolution width, height, invocation & 255, 0.
    pub(super) screen: [f32; 4],
    /// @128: previous camera view-projection, column-major.
    pub(super) previous_view_proj: [[f32; 4]; 4],
}

/// An orthonormal camera basis; the scale is applied when the lanes are written.
struct Basis {
    right: Vec3,
    up: Vec3,
    forward: Vec3,
}

/// The camera basis, falling back to forward -Z and up +Y when the frame's is degenerate.
fn camera_basis(frame: &CloudFrame) -> Basis {
    let camera = &frame.camera;
    let forward = (camera.target - camera.eye).normalize_or_zero();
    let up = camera.up.normalize_or_zero();
    let right = forward.cross(up).normalize_or_zero();
    if forward == Vec3::ZERO || right == Vec3::ZERO {
        return Basis {
            right: Vec3::X,
            up: Vec3::Y,
            forward: -Vec3::Z,
        };
    }
    Basis {
        right,
        up: right.cross(forward),
        forward,
    }
}

impl CloudGpu {
    /// Size in bytes of the cloud uniform.
    pub(super) const SIZE: usize = 192;

    /// Builds the record from a validated frame and its committed history matrix.
    pub(super) fn from_frame(frame: &CloudFrame, history: Option<Mat4>) -> CloudGpu {
        let weather = frame.weather;
        let texel = SHADOW_EXTENT / SHADOW_SIZE as f32;
        let snap = |v: f32| (v / texel).floor() * texel - SHADOW_EXTENT * 0.5;
        let basis = camera_basis(frame);
        let wind = frame.wind_velocity;
        let length = (wind[0] * wind[0] + wind[1] * wind[1]).sqrt();
        let wind_direction = if length > 0.0 && length.is_finite() {
            [wind[0] / length, wind[1] / length]
        } else {
            [1.0, 0.0]
        };
        let tan_half = (frame.camera.fov_y_rad * 0.5).tan();
        let right = basis.right * (tan_half * frame.camera.aspect);
        let up = basis.up * tan_half;
        CloudGpu {
            wind_offset: frame.wind_offset,
            evolution: frame.evolution,
            coverage: weather.coverage,
            cloud_type: weather.cloud_type,
            base: weather.base,
            top: weather.base + weather.thickness,
            density: weather.density,
            cirrus: weather.cirrus,
            rain: weather.rain,
            shadow_extent: SHADOW_EXTENT,
            enabled: if frame.physical && frame.weather_enabled {
                1.0
            } else {
                0.0
            },
            shadow_origin: [snap(frame.camera.eye.x), snap(frame.camera.eye.z)],
            wind_direction,
            right: [right.x, right.y, right.z, 0.0],
            up: [up.x, up.y, up.z, 0.0],
            forward: [
                basis.forward.x,
                basis.forward.y,
                basis.forward.z,
                if history.is_some() { 1.0 } else { 0.0 },
            ],
            screen: [
                (frame.size.0 as f32 * 0.5).ceil(),
                (frame.size.1 as f32 * 0.5).ceil(),
                (frame.invocation & 255) as f32,
                0.0,
            ],
            previous_view_proj: history.unwrap_or(Mat4::IDENTITY).cols,
        }
    }

    /// Explicit little-endian bytes of the 192-byte record.
    pub(super) fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut out = [0u8; Self::SIZE];
        put_f32x2(&mut out, 0, self.wind_offset);
        put_f32(&mut out, 8, self.evolution);
        put_f32(&mut out, 12, self.coverage);
        put_f32(&mut out, 16, self.cloud_type);
        put_f32(&mut out, 20, self.base);
        put_f32(&mut out, 24, self.top);
        put_f32(&mut out, 28, self.density);
        put_f32(&mut out, 32, self.cirrus);
        put_f32(&mut out, 36, self.rain);
        put_f32(&mut out, 40, self.shadow_extent);
        put_f32(&mut out, 44, self.enabled);
        put_f32x2(&mut out, 48, self.shadow_origin);
        put_f32x2(&mut out, 56, self.wind_direction);
        put_f32x4(&mut out, 64, self.right);
        put_f32x4(&mut out, 80, self.up);
        put_f32x4(&mut out, 96, self.forward);
        put_f32x4(&mut out, 112, self.screen);
        for (i, column) in self.previous_view_proj.into_iter().enumerate() {
            put_f32x4(&mut out, 128 + i * 16, column);
        }
        out
    }
}

/// The 256-byte S10 sky record of a cloud frame; every unspecified lane is the default.
pub(super) fn cloud_sky_parameters(frame: &CloudFrame) -> PhysicalSkyGpu {
    let defaults = PhysicalSkyGpu::default();
    let sky = frame.sky;
    let light = frame.dominant;
    let toward = (-light.direction).normalize_or_zero();
    let toward = if toward == Vec3::ZERO {
        Vec3::Y
    } else {
        toward
    };
    PhysicalSkyGpu {
        viewer: [
            frame.camera.eye.x,
            frame.camera.eye.y,
            frame.camera.eye.z,
            atmospheric_radius_km(frame.camera.eye.y),
        ],
        sun: [sky.sun.x, sky.sun.y, sky.sun.z, sky.moon_lit],
        moon: [sky.moon.x, sky.moon.y, sky.moon.z, defaults.moon[3]],
        pole: [sky.pole.x, sky.pole.y, sky.pole.z, defaults.pole[3]],
        equinox: [
            sky.equinox.x,
            sky.equinox.y,
            sky.equinox.z,
            defaults.equinox[3],
        ],
        dominant: [toward.x, toward.y, toward.z, light.intensity],
        daylight: [
            light.color[0],
            light.color[1],
            light.color[2],
            light.intensity,
        ],
        ozone: [
            defaults.ozone[0],
            defaults.ozone[1],
            defaults.ozone[2],
            frame.weather.rain,
        ],
        planet: [
            defaults.planet[0],
            defaults.planet[1],
            frame.animation_seconds as f32,
            defaults.planet[3],
        ],
        flags: [1, 0, 0, 0],
        ..defaults
    }
}

fn put_f32(out: &mut [u8; CloudGpu::SIZE], offset: usize, value: f32) {
    out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_f32x2(out: &mut [u8; CloudGpu::SIZE], offset: usize, values: [f32; 2]) {
    for (i, value) in values.into_iter().enumerate() {
        put_f32(out, offset + i * 4, value);
    }
}

fn put_f32x4(out: &mut [u8; CloudGpu::SIZE], offset: usize, values: [f32; 4]) {
    for (i, value) in values.into_iter().enumerate() {
        put_f32(out, offset + i * 4, value);
    }
}

// Contract checks until the ground-shadow card uploads these records.
const _: fn(&CloudFrame, Option<Mat4>) -> CloudGpu = CloudGpu::from_frame;
const _: fn(&CloudGpu) -> [u8; 192] = CloudGpu::to_bytes;
const _: fn(&CloudFrame) -> PhysicalSkyGpu = cloud_sky_parameters;
