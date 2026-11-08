//! SDTP v2 scalar sampled resources; trilinear interpolation and bounded exterior extension.
use crate::program::{DATA, TraceError, TraceProgram};
use raa_math::Vec3;

pub(crate) fn validate(program: &TraceProgram, d: &[f32], path: &str) -> Result<f32, TraceError> {
    let bad = |s: &str| TraceError::at(DATA, path, s);
    if d[12] <= 0.0 {
        return Err(bad("sampled distance multiplier must be positive"));
    }
    let h = d[17];
    if h <= 0.0 {
        return Err(bad("sample spacing must be positive"));
    }
    let mut size = [0usize; 3];
    for a in 0..3 {
        let n = d[18 + a];
        if !(2.0..=2048.0).contains(&n) || n.fract() != 0.0 {
            return Err(bad("invalid sampled dimensions"));
        }
        size[a] = n as usize;
    }
    if d[21] < 0.0 || d[21].fract() != 0.0 || d[21] > 8_388_608.0 || !(d[23] == 0.0 || d[23] == 1.0)
    {
        return Err(bad("invalid sampled resource offset or reserved word"));
    }
    let offset = d[21] as usize * 4;
    let count: usize = size.iter().product();
    let word_count = if d[23] == 1.0 {
        count.div_ceil(2)
    } else {
        count
    };
    let end = offset
        .checked_add(word_count)
        .ok_or_else(|| bad("sample resource overflows"))?;
    if program
        .sources
        .iter()
        .any(|s| offset < (s.data_start as usize + s.data_len as usize))
    {
        return Err(bad("sample resource overlaps instruction data"));
    }
    let values = program
        .data
        .get(offset..end)
        .ok_or_else(|| bad("sample resource is truncated"))?;
    if !(0..count).any(|i| sample_value(values, i, d[23]) < 0.0) {
        return Err(bad("sampled leaf has no interior"));
    }
    let [nx, ny, nz] = size;
    let mut slopes = [0.0f64; 3];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let i = x + nx * (y + ny * z);
                let v = sample_value(values, i, d[23]);
                if !v.is_finite() {
                    return Err(bad("non-finite sampled distance"));
                }
                if (x == 0 || y == 0 || z == 0 || x + 1 == nx || y + 1 == ny || z + 1 == nz)
                    && v <= 0.0
                {
                    return Err(bad("sampled leaf reaches its domain boundary"));
                }
                for (a, stride, at, n) in [(0, 1, x, nx), (1, nx, y, ny), (2, nx * ny, z, nz)] {
                    if at + 1 < n {
                        slopes[a] = slopes[a].max(
                            (f64::from(sample_value(values, i + stride, d[23])) - f64::from(v))
                                .abs()
                                / f64::from(h),
                        );
                    }
                }
            }
        }
    }
    let bound = (slopes.iter().map(|v| v * v).sum::<f64>().sqrt() + 1.0) as f32;
    if d[22] < bound {
        return Err(bad("sampled Lipschitz bound is too small"));
    }
    let norm = d[..12]
        .chunks_exact(4)
        .flat_map(|r| r[..3].iter())
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt();
    Ok(d[22] * d[12] * norm)
}

pub(crate) fn distance(data: &[f32], d: &[f32], p: Vec3) -> f32 {
    let min = Vec3::new(d[14], d[15], d[16]);
    let h = d[17];
    let size = [d[18] as usize, d[19] as usize, d[20] as usize];
    let offset = d[21] as usize * 4;
    let p = p.to_array();
    let min = min.to_array();
    let q: [f32; 3] =
        std::array::from_fn(|a| ((p[a] - min[a]) / h).clamp(0.0, (size[a] - 1) as f32));
    let i: [usize; 3] = std::array::from_fn(|a| (q[a].floor() as usize).min(size[a] - 2));
    let t: [f32; 3] = std::array::from_fn(|a| q[a] - i[a] as f32);
    let mut value = 0.0;
    for k in 0..8 {
        let xyz = [k & 1, (k >> 1) & 1, k >> 2];
        let w = (0..3)
            .map(|a| if xyz[a] == 0 { 1.0 - t[a] } else { t[a] })
            .product::<f32>();
        value += w * sample_value(
            &data[offset..],
            i[0] + xyz[0] + size[0] * (i[1] + xyz[1] + size[1] * (i[2] + xyz[2])),
            d[23],
        );
    }
    value
        + (0..3)
            .map(|a| (p[a] - min[a] - q[a] * h).powi(2))
            .sum::<f32>()
            .sqrt()
}

fn sample_value(data: &[f32], i: usize, format: f32) -> f32 {
    if format == 0.0 {
        data[i]
    } else {
        half::f16::from_bits((data[i / 2].to_bits() >> (16 * (i % 2))) as u16).to_f32()
    }
}
