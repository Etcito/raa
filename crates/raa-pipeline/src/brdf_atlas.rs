//! The BRDF/LTC atlas of group 1 binding 5 (contract brdf-ltc-atlas): one `Rgba16Float`
//! texture of 128 x 192 texels holding the split-sum BRDF LUT and the two 64 x 64 linearly
//! transformed cosine tables of three.js r180.

use raa_core::GpuContext;

use crate::lighting::f16_texel;

/// Format of the BRDF/LTC atlas (S10 binding 5).
pub const BRDF_ATLAS_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Width of the atlas in texels.
pub const BRDF_ATLAS_WIDTH: u32 = 128;
/// Height of the atlas in texels.
pub const BRDF_ATLAS_HEIGHT: u32 = 192;
/// Side of the BRDF split-sum region at origin (0, 0).
pub const BRDF_LUT_SIZE: u32 = 128;
/// Side of each LTC table region.
pub const LTC_TABLE_SIZE: u32 = 64;
/// Origin (x, y) of the LTC1 region.
pub const LTC1_ORIGIN: [u32; 2] = [0, 128];
/// Origin (x, y) of the LTC2 region.
pub const LTC2_ORIGIN: [u32; 2] = [64, 128];

/// Channels of one atlas texel.
const CHANNELS: usize = 4;
/// Texels of one 64 x 64 LTC table.
const LTC_TEXELS: usize = (LTC_TABLE_SIZE * LTC_TABLE_SIZE) as usize;
/// Data rows of one LTC table.
const LTC_ROWS: usize = LTC_TABLE_SIZE as usize;
/// Channel values of one LTC row (64 texels x 4 channels).
const LTC_ROW_VALUES: usize = LTC_TABLE_SIZE as usize * CHANNELS;
/// Bytes of one `Rgba16Float` texel.
const TEXEL_BYTES: u32 = CHANNELS as u32 * 2;
/// Hex digits per channel value.
const HEX_DIGITS: usize = 4;

/// The three.js r180 LTC tables: 128 data lines of 1024 hex digits. Lines 1-64 are
/// `LTC_MAT_1` rows j = 0..63, lines 65-128 are `LTC_MAT_2` rows j = 0..63; texel (i, j)
/// starts at hex group (j*64 + i)*4 and each group is the IEEE binary16 bit pattern of one
/// channel in r, g, b, a order. The file header keeps the paper citation, the source and
/// the three.js MIT licence.
const LTC_TABLES: &str = include_str!("ltc_tables.txt");

/// Parses [`LTC_TABLES`] into the LTC1 then LTC2 tables, each 16384 IEEE binary16 bit
/// patterns indexed `(j*64 + i)*4 + channel`. A malformed hex group decodes as zero.
pub fn ltc_table_halfs() -> [Vec<u16>; 2] {
    let mut tables = [
        vec![0u16; LTC_TEXELS * CHANNELS],
        vec![0u16; LTC_TEXELS * CHANNELS],
    ];
    let mut line_index = 0usize;
    for line in LTC_TABLES.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let table = line_index / LTC_ROWS;
        if table >= tables.len() {
            break;
        }
        let base = (line_index % LTC_ROWS) * LTC_ROW_VALUES;
        for (index, group) in line.as_bytes().chunks_exact(HEX_DIGITS).enumerate() {
            if index >= LTC_ROW_VALUES {
                break;
            }
            let value = std::str::from_utf8(group)
                .ok()
                .and_then(|digits| u16::from_str_radix(digits, 16).ok())
                .unwrap_or(0);
            tables[table][base + index] = value;
        }
        line_index += 1;
    }
    tables
}

/// Handoff: "raa-ibl and raa-ssr: BRDF LUT written into and read from the atlas region"
/// takes over the BRDF region: `create_brdf_atlas` creates the 128 x 192 `Rgba16Float`
/// atlas, writes both LTC regions from [`LTC_TABLES`] and fills every texel of the BRDF
/// region with `brdf_fill`; `ltc_table_halfs` returns the parsed LTC1 and LTC2 tables.
///
/// The returned texture is `TEXTURE_BINDING | COPY_DST`, never panics, and is written in
/// place: the BRDF region of `BRDF_LUT_SIZE` x `BRDF_LUT_SIZE` at origin (0, 0) holds
/// `(brdf_fill[0], brdf_fill[1], 0, 0)`, LTC1 sits at [`LTC1_ORIGIN`] and LTC2 at
/// [`LTC2_ORIGIN`].
pub fn create_brdf_atlas(gpu: &GpuContext, label: &str, brdf_fill: [f32; 2]) -> wgpu::Texture {
    let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: BRDF_ATLAS_WIDTH,
            height: BRDF_ATLAS_HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: BRDF_ATLAS_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let brdf_texel = f16_texel(&[brdf_fill[0], brdf_fill[1], 0.0, 0.0]);
    let brdf_region = brdf_texel.repeat((BRDF_LUT_SIZE * BRDF_LUT_SIZE) as usize);
    gpu.queue().write_texture(
        texture.as_image_copy(),
        &brdf_region,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(BRDF_LUT_SIZE * TEXEL_BYTES),
            rows_per_image: Some(BRDF_LUT_SIZE),
        },
        wgpu::Extent3d {
            width: BRDF_LUT_SIZE,
            height: BRDF_LUT_SIZE,
            depth_or_array_layers: 1,
        },
    );
    for (table, origin) in ltc_table_halfs().iter().zip([LTC1_ORIGIN, LTC2_ORIGIN]) {
        let bytes: Vec<u8> = table.iter().flat_map(|half| half.to_le_bytes()).collect();
        gpu.queue().write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: origin[0],
                    y: origin[1],
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(LTC_TABLE_SIZE * TEXEL_BYTES),
                rows_per_image: Some(LTC_TABLE_SIZE),
            },
            wgpu::Extent3d {
                width: LTC_TABLE_SIZE,
                height: LTC_TABLE_SIZE,
                depth_or_array_layers: 1,
            },
        );
    }
    texture
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decodes an IEEE binary16 bit pattern to `f32`.
    fn f16_value(bits: u16) -> f32 {
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exponent = ((bits >> 10) & 0x1f) as i32;
        let mantissa = f32::from(bits & 0x03ff);
        match exponent {
            0 => sign * mantissa * 2.0f32.powi(-24),
            31 if mantissa == 0.0 => sign * f32::INFINITY,
            31 => f32::NAN,
            exponent => sign * (1.0 + mantissa / 1024.0) * 2.0f32.powi(exponent - 15),
        }
    }

    /// The `(r, g, b, a)` texel (i, j) of a parsed table, decoded from f16.
    fn texel(table: &[u16], i: usize, j: usize) -> [f32; 4] {
        let base = (j * LTC_ROWS + i) * CHANNELS;
        std::array::from_fn(|channel| f16_value(table[base + channel]))
    }

    #[test]
    fn file_holds_two_tables_of_16384_halfs() {
        let data: Vec<&str> = LTC_TABLES
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .collect();
        assert_eq!(data.len(), 2 * LTC_ROWS, "128 data lines");
        for line in &data {
            assert_eq!(line.len(), 1024, "one line of 1024 hex digits");
            assert!(
                line.bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
                "lowercase hex digits only"
            );
        }
        let tables = ltc_table_halfs();
        assert_eq!(tables[0].len(), 16384);
        assert_eq!(tables[1].len(), 16384);
        assert!(tables[0].iter().any(|half| *half != 0));
        assert!(tables[1].iter().any(|half| *half != 0));
    }

    #[test]
    fn first_texels_match_the_three_js_source() {
        let tables = ltc_table_halfs();
        // LTC_MAT_1 and LTC_MAT_2 both start with 1, 0, 0, <fourth>.
        let expected = [[1.0, 0.0, 0.0, 2e-05], [1.0, 0.0, 0.0, 0.0]];
        for (table, want) in tables.iter().zip(expected) {
            let got = texel(table, 0, 0);
            for (channel, (got, want)) in got.iter().zip(want).enumerate() {
                assert!(
                    (got - want).abs() <= 1e-3,
                    "first texel channel {channel}: {got} != {want}"
                );
            }
        }
    }

    #[test]
    fn ltc2_magnitude_and_fresnel_stay_in_range() {
        let tables = ltc_table_halfs();
        for j in 0..LTC_ROWS {
            for i in 0..LTC_ROWS {
                let [magnitude, fresnel, _, _] = texel(&tables[1], i, j);
                assert!(
                    (0.0..=1.05).contains(&magnitude),
                    "LTC2 ({i}, {j}) magnitude {magnitude}"
                );
                assert!(
                    (0.0..=1.05).contains(&fresnel),
                    "LTC2 ({i}, {j}) fresnel {fresnel}"
                );
            }
        }
    }

    #[test]
    fn ltc1_texels_have_a_positive_determinant() {
        let tables = ltc_table_halfs();
        for j in 0..LTC_ROWS {
            for i in 0..LTC_ROWS {
                let [r, g, b, a] = texel(&tables[0], i, j);
                let determinant = r * a - g * b;
                assert!(
                    determinant > 0.0,
                    "LTC1 ({i}, {j}) = ({r}, {g}, {b}, {a}): determinant {determinant}"
                );
            }
        }
    }

    /// The context is leaked like in tests/resources.rs (GL driver teardown).
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn brdf_atlas_builds() {
        let Some(gpu) = raa_core::testing::software_gpu() else {
            raa_core::testing::skip(
                "brdf_atlas_builds",
                "software_gpu returned None; see the software_gpu line above",
            );
            return;
        };
        let scope = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
        let atlas = create_brdf_atlas(&gpu, "test.brdf_atlas", [1.0, 0.0]);
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
        let error = raa_core::testing::block_on(scope.pop());
        assert!(error.is_none(), "validation error: {error:?}");
        assert_eq!(atlas.format(), BRDF_ATLAS_FORMAT);
        assert_eq!(atlas.format(), wgpu::TextureFormat::Rgba16Float);
        assert_eq!(
            (atlas.width(), atlas.height()),
            (BRDF_ATLAS_WIDTH, BRDF_ATLAS_HEIGHT)
        );
        assert_eq!((atlas.width(), atlas.height()), (128, 192));
        drop(atlas);
        std::mem::forget(gpu);
    }
}
