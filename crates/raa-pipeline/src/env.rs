//! `EnvGpu` (S10 group 1 binding 11): sky, background, clouds, fog and aerial parameters.

use raa_core::CLEAR_COLOR;

/// S10 `EnvGpu` uniform, one named `vec4<f32>` per field; `reserved` @192..256 is written as
/// zeros. Feature crates edit it through `LightingResources::env_mut`, which marks it changed.
#[derive(Clone, Debug, PartialEq)]
pub struct EnvGpu {
    /// @0: angular radius (rad), intensity, 0, 0.
    pub sun_disc: [f32; 4],
    /// @16: zenith rgb, w 0.
    pub zenith: [f32; 4],
    /// @32: horizon rgb, w 0.
    pub horizon: [f32; 4],
    /// @48: ground rgb, w 0.
    pub ground: [f32; 4],
    /// @64: intensity, horizon softness, background mode (0 constant / 1 sky), 0.
    pub sky_params: [f32; 4],
    /// @80: display-referred background rgb, w 0.
    pub background: [f32; 4],
    /// @96: coverage, density, altitude m, scale m.
    pub clouds: [f32; 4],
    /// @112: wind x, wind z, silver lining, 0.
    pub clouds2: [f32; 4],
    /// @128: cloud rgb, w 0.
    pub cloud_color: [f32; 4],
    /// @144: density, height falloff, base height, start distance.
    pub fog: [f32; 4],
    /// @160: fog rgb, inscatter strength.
    pub fog_color: [f32; 4],
    /// @176: strength, distance scale, 0, 0.
    pub aerial: [f32; 4],
}

impl Default for EnvGpu {
    /// Everything zero except `background` = CLEAR_COLOR linear rgb; background mode 0
    /// (constant), so an empty frame equals the v0 clear colour.
    fn default() -> Self {
        EnvGpu {
            sun_disc: [0.0; 4],
            zenith: [0.0; 4],
            horizon: [0.0; 4],
            ground: [0.0; 4],
            sky_params: [0.0; 4],
            background: [
                CLEAR_COLOR.r as f32,
                CLEAR_COLOR.g as f32,
                CLEAR_COLOR.b as f32,
                0.0,
            ],
            clouds: [0.0; 4],
            clouds2: [0.0; 4],
            cloud_color: [0.0; 4],
            fog: [0.0; 4],
            fog_color: [0.0; 4],
            aerial: [0.0; 4],
        }
    }
}

impl EnvGpu {
    /// Size in bytes of the uniform (S10).
    pub const SIZE: usize = 256;

    /// Little-endian S10 bytes (fields in declaration order @0..192, zeros @192..256).
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let fields = [
            &self.sun_disc,
            &self.zenith,
            &self.horizon,
            &self.ground,
            &self.sky_params,
            &self.background,
            &self.clouds,
            &self.clouds2,
            &self.cloud_color,
            &self.fog,
            &self.fog_color,
            &self.aerial,
        ];
        let mut out = [0u8; Self::SIZE];
        let values = fields.iter().flat_map(|field| field.iter());
        for (chunk, value) in out.chunks_exact_mut(4).zip(values) {
            chunk.copy_from_slice(&value.to_le_bytes());
        }
        out
    }
}
