//! Physical atmosphere data and the private S10 `PhysicalSkyGpu` uniform.
//!
//! The constants, exponential density profile, phase functions, ray/sphere
//! intersection and LUT parameterizations are adapted from Solstice revision
//! `7e465b599c13da165948ebaa44df824c36c3ae07`:
//! `crates/solstice-render/shaders/common/atmosphere.wgsl:4-102`. The viewer
//! metres-to-kilometres conversion is the one at
//! `crates/solstice-render/src/sky.rs:104-108`. Adapted source retains the
//! Solstice MIT notice and `Copyright (c) 2026 SamG-Coder`.
//!
//! All atmospheric distances are kilometres. `PhysicalSkyGpu` is the sixteen-lane
//! S10 group-2 binding-0 record; [`PhysicalSkyGpu::to_bytes`] is its only
//! serialization boundary and always writes explicit little-endian words.
/*
MIT License

Copyright (c) 2026 SamG-Coder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/// Ground radius at sea level, in kilometres.
pub(crate) const GROUND_RADIUS_KM: f32 = 6360.0;
/// Top of the atmosphere, in kilometres.
pub(crate) const TOP_RADIUS_KM: f32 = 6460.0;

/// Rayleigh scattering coefficients, per kilometre.
pub(crate) const RAYLEIGH_SCATTERING: [f32; 3] = [5.802e-3, 13.558e-3, 33.1e-3];
/// Rayleigh density scale height, in kilometres.
pub(crate) const RAYLEIGH_SCALE_HEIGHT_KM: f32 = 8.0;

/// Mie scattering coefficient, per kilometre (before `mieMultiplier`).
pub(crate) const MIE_SCATTERING: f32 = 14.0e-3;
/// Mie absorption coefficient, per kilometre.
pub(crate) const MIE_ABSORPTION: f32 = 4.40e-3;
/// Mie density scale height, in kilometres.
pub(crate) const MIE_SCALE_HEIGHT_KM: f32 = 1.6;
/// Cornette-Shanks asymmetry parameter.
pub(crate) const MIE_G: f32 = 0.8;

/// Ozone absorption coefficients, per kilometre.
pub(crate) const OZONE_ABSORPTION: [f32; 3] = [0.650e-3, 1.881e-3, 0.085e-3];

/// The one metres-to-kilometres viewer conversion of the sky feature.
///
/// Solstice `sky.rs:104-108`: a viewer height at or below one metre counts as
/// one metre; the result is the distance from the planet centre in kilometres.
pub(crate) fn atmospheric_radius_km(height_m: f32) -> f32 {
    GROUND_RADIUS_KM + height_m.max(1.0) * 0.001
}

/// The S10 private physical-sky uniform (group 2, binding 0, 256 bytes).
///
/// Exactly sixteen 16-byte lanes. `viewer` is the world position in metres with
/// the already-converted viewer radius in kilometres in `w`; directions are
/// unit vectors toward the body/light. See the architecture S10 table for every
/// component meaning.
#[repr(C)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PhysicalSkyGpu {
    /// @0: viewer world xyz metres, viewer radius kilometres.
    pub(crate) viewer: [f32; 4],
    /// @16: direction toward the sun, moon lit fraction.
    pub(crate) sun: [f32; 4],
    /// @32: direction toward the moon, moon radiance.
    pub(crate) moon: [f32; 4],
    /// @48: celestial pole xyz, star intensity.
    pub(crate) pole: [f32; 4],
    /// @64: equinox xyz, night glow.
    pub(crate) equinox: [f32; 4],
    /// @80: direction toward the dominant light, raw intensity.
    pub(crate) dominant: [f32; 4],
    /// @96: daylight linear rgb, baseline intensity.
    pub(crate) daylight: [f32; 4],
    /// @112: sun radius, sun multiplier, moon radius, pixel angle.
    pub(crate) discs: [f32; 4],
    /// @128: ground albedo rgb, Mie multiplier.
    pub(crate) ground_albedo: [f32; 4],
    /// @144: Rayleigh coefficients rgb, scale height kilometres.
    pub(crate) rayleigh: [f32; 4],
    /// @160: Mie scattering, absorption, scale height kilometres, g.
    pub(crate) mie: [f32; 4],
    /// @176: ozone coefficients rgb, rain.
    pub(crate) ozone: [f32; 4],
    /// @192: ground radius kilometres, top radius kilometres, animation seconds, 0.
    pub(crate) planet: [f32; 4],
    /// @208: physical, cloud screen valid, cloud panorama valid, 0.
    pub(crate) flags: [u32; 4],
    /// @224: reserved words, serialized as zeros.
    pub(crate) reserved: [[f32; 4]; 2],
}

impl Default for PhysicalSkyGpu {
    /// Source defaults: physical atmosphere on, white daylight, sea-level viewer,
    /// clear-day ground albedo and the Solstice atmosphere constants.
    fn default() -> Self {
        Self {
            viewer: [0.0, 0.0, 0.0, atmospheric_radius_km(1.0)],
            sun: [0.0, 1.0, 0.0, 1.0],
            moon: [0.0, -1.0, 0.0, 0.0],
            pole: [0.0, 1.0, 0.0, 1.0],
            equinox: [1.0, 0.0, 0.0, 1.2e-4],
            dominant: [0.0, 1.0, 0.0, 1.0],
            daylight: [1.0, 1.0, 1.0, 1.0],
            discs: [0.003, 8.0, 0.0075, 0.0],
            ground_albedo: [0.3, 0.3, 0.3, 1.0],
            rayleigh: [
                RAYLEIGH_SCATTERING[0],
                RAYLEIGH_SCATTERING[1],
                RAYLEIGH_SCATTERING[2],
                RAYLEIGH_SCALE_HEIGHT_KM,
            ],
            mie: [MIE_SCATTERING, MIE_ABSORPTION, MIE_SCALE_HEIGHT_KM, MIE_G],
            ozone: [
                OZONE_ABSORPTION[0],
                OZONE_ABSORPTION[1],
                OZONE_ABSORPTION[2],
                0.0,
            ],
            planet: [GROUND_RADIUS_KM, TOP_RADIUS_KM, 0.0, 0.0],
            flags: [1, 0, 0, 0],
            reserved: [[0.0; 4]; 2],
        }
    }
}

impl PhysicalSkyGpu {
    /// Size in bytes of the S10 uniform.
    pub(crate) const SIZE: usize = 256;

    /// Little-endian S10 bytes: thirteen `vec4<f32>` lanes at @0..208, `uvec4<u32>`
    /// flags at @208 and the two reserved zero lanes at @224/@240.
    #[allow(dead_code)] // consumed by the atmosphere resource/share cards (7428+)
    pub(crate) fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut out = [0u8; Self::SIZE];
        let lanes = [
            self.viewer,
            self.sun,
            self.moon,
            self.pole,
            self.equinox,
            self.dominant,
            self.daylight,
            self.discs,
            self.ground_albedo,
            self.rayleigh,
            self.mie,
            self.ozone,
            self.planet,
        ];
        for (lane, values) in lanes.into_iter().enumerate() {
            put_f32x4(&mut out, lane * 16, values);
        }
        put_u32x4(&mut out, 208, self.flags);
        // S10 lanes 14 and 15 are reserved padding: the declared record keeps them,
        // but the ABI always uploads zeros so placeholder content cannot leak.
        debug_assert_eq!(self.reserved, [[0.0; 4]; 2]);
        out
    }
}

fn put_f32x4(out: &mut [u8; PhysicalSkyGpu::SIZE], offset: usize, values: [f32; 4]) {
    for (i, value) in values.into_iter().enumerate() {
        let at = offset + i * 4;
        out[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }
}

fn put_u32x4(out: &mut [u8; PhysicalSkyGpu::SIZE], offset: usize, values: [u32; 4]) {
    for (i, value) in values.into_iter().enumerate() {
        let at = offset + i * 4;
        out[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_at(bytes: &[u8; 256], offset: usize) -> f32 {
        f32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("four bytes"))
    }

    fn u32_at(bytes: &[u8; 256], offset: usize) -> u32 {
        u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("four bytes"))
    }

    #[test]
    fn radius_boundaries_follow_the_solstice_conversion() {
        assert_eq!(atmospheric_radius_km(-100.0), 6360.001);
        assert_eq!(atmospheric_radius_km(0.0), 6360.001);
        assert_eq!(atmospheric_radius_km(1.0), 6360.001);
        assert!((atmospheric_radius_km(1000.0) - 6361.0).abs() < 1e-6);
        assert!((atmospheric_radius_km(10_000.0) - 6370.0).abs() < 1e-4);
        // NaN is ignored by `f32::max`, so it takes the one-metre floor.
        assert_eq!(atmospheric_radius_km(f32::NAN), 6360.001);
        assert!(atmospheric_radius_km(f32::INFINITY).is_infinite());
    }

    #[test]
    fn lanes_serialize_at_the_s10_offsets_in_little_endian() {
        let gpu = PhysicalSkyGpu {
            viewer: [11.0, 12.0, 13.0, 6360.5],
            sun: [21.0, 22.0, 23.0, 0.25],
            moon: [31.0, 32.0, 33.0, 0.5],
            pole: [41.0, 42.0, 43.0, 0.75],
            equinox: [51.0, 52.0, 53.0, 1.5],
            dominant: [61.0, 62.0, 63.0, 2.5],
            daylight: [71.0, 72.0, 73.0, 3.5],
            discs: [81.0, 82.0, 83.0, 84.0],
            ground_albedo: [91.0, 92.0, 93.0, 94.0],
            rayleigh: [101.0, 102.0, 103.0, 104.0],
            mie: [111.0, 112.0, 113.0, 114.0],
            ozone: [121.0, 122.0, 123.0, 124.0],
            planet: [131.0, 132.0, 133.0, 134.0],
            flags: [1, 2, 3, 4],
            reserved: [[0.0; 4]; 2],
        };
        let bytes = gpu.to_bytes();
        assert_eq!(bytes.len(), 256);
        assert_eq!(&bytes[0..4], &11.0f32.to_le_bytes());
        for (offset, lane) in [
            (0, gpu.viewer),
            (16, gpu.sun),
            (32, gpu.moon),
            (48, gpu.pole),
            (64, gpu.equinox),
            (80, gpu.dominant),
            (96, gpu.daylight),
            (112, gpu.discs),
            (128, gpu.ground_albedo),
            (144, gpu.rayleigh),
            (160, gpu.mie),
            (176, gpu.ozone),
            (192, gpu.planet),
        ] {
            for (i, value) in lane.into_iter().enumerate() {
                assert_eq!(f32_at(&bytes, offset + i * 4).to_bits(), value.to_bits());
            }
        }
        for (i, value) in gpu.flags.into_iter().enumerate() {
            assert_eq!(u32_at(&bytes, 208 + i * 4), value);
        }
        assert!(bytes[224..256].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn default_uses_source_constants_and_zero_reserved_words() {
        let gpu = PhysicalSkyGpu::default();
        assert_eq!(gpu.viewer[3], 6360.001);
        assert_eq!(gpu.rayleigh[0..3], RAYLEIGH_SCATTERING);
        assert_eq!(gpu.rayleigh[3], RAYLEIGH_SCALE_HEIGHT_KM);
        assert_eq!(
            gpu.mie,
            [MIE_SCATTERING, MIE_ABSORPTION, MIE_SCALE_HEIGHT_KM, MIE_G]
        );
        assert_eq!(gpu.ozone[0..3], OZONE_ABSORPTION);
        assert_eq!(gpu.planet, [GROUND_RADIUS_KM, TOP_RADIUS_KM, 0.0, 0.0]);
        assert_eq!(gpu.reserved, [[0.0; 4]; 2]);
        let bytes = gpu.to_bytes();
        assert_eq!(f32_at(&bytes, 224), 0.0);
        assert_eq!(f32_at(&bytes, 240), 0.0);
        assert_eq!(f32_at(&bytes, 252), 0.0);
    }
}
