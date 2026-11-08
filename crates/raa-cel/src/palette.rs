//! Six-row warm/cool palette and diffuse band thresholds.
pub const CEL_THRESHOLDS: [f32; 3] = [0.05, 0.32, 0.68];
#[derive(Clone, Copy, Debug)]
pub struct RampRow {
    pub base: f32,
    pub light: f32,
    pub offset: [f32; 3],
}
pub const CEL_RAMP: [RampRow; 6] = [
    RampRow {
        base: 0.18,
        light: 0.00000000,
        offset: [0.15380392, 0.14792157, 0.21372549],
    },
    RampRow {
        base: 0.38,
        light: 0.00000000,
        offset: [0.16870588, 0.12752941, 0.12745098],
    },
    RampRow {
        base: 0.65,
        light: 0.00000000,
        offset: [0.13470588, 0.06509804, 0.04019608],
    },
    RampRow {
        base: 1.00000000,
        light: 0.00000000,
        offset: [0.00000000, 0.00000000, 0.00000000],
    },
    RampRow {
        base: 0.72,
        light: 0.28,
        offset: [0.04109804, -0.03286275, -0.04641569],
    },
    RampRow {
        base: 0.4,
        light: 0.6,
        offset: [-0.08784314, -0.08666667, -0.01262745],
    },
];
/// Diffuse-only band selection; no curvature or material-id input.
pub fn cel_step(n_dot_l: f32) -> usize {
    1 + CEL_THRESHOLDS.iter().filter(|&&t| n_dot_l > t).count()
}
pub fn palette_color(anchor: [f32; 3], light: [f32; 3], band: usize) -> [f32; 3] {
    let row = CEL_RAMP[band.min(5)];
    let mut a = anchor.map(|v| v.clamp(0.0, 1.0));
    if a.iter().copied().fold(0.0, f32::max) < 0.06 {
        a = [0.12, 0.12, 0.14];
    }
    let warm = [1.0, 0.9, 0.74];
    std::array::from_fn(|i| {
        (a[i] * row.base
            + row.offset[i]
            + (0.7 * warm[i] + 0.3 * light[i].clamp(0.0, 1.0)) * row.light)
            .clamp(0.0, 1.0)
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bands_obey_strict_thresholds() {
        assert_eq!(cel_step(-1.0), 1);
        for (i, t) in CEL_THRESHOLDS.into_iter().enumerate() {
            assert_eq!(cel_step(t), i + 1);
            assert_eq!(cel_step(t + 0.00001), i + 2);
        }
    }
    #[test]
    fn palette_reproduces_anchor_and_warm_cool_rows() {
        let anchor = [121.0 / 255.0, 96.0 / 255.0, 75.0 / 255.0];
        let expected = [
            [61, 55, 68],
            [89, 69, 61],
            [113, 79, 59],
            [121, 96, 75],
            [169, 125, 95],
            [179, 154, 140],
        ];
        for (band, rgb) in expected.into_iter().enumerate() {
            let got = palette_color(anchor, [1.0, 0.9, 0.74], band);
            for i in 0..3 {
                assert!((got[i] - rgb[i] as f32 / 255.0).abs() < 1e-6);
            }
        }
    }
}
