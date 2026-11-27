//! CPU twin of `wgsl/water.wgsl`: the shared water surface (height, analytic gradient, normal),
//! the dielectric helpers and the packing of the water body into `FrameGpu.reserved[0..4]`.
//! Same constants and the same evaluation order as the WGSL, so the tests (the refracted-grid
//! check of the caustics) see the normal and the caustics the GPU computes.

/// The WGSL module (pure functions over [`WaterBody`] values; no bindings).
pub const WATER_WGSL: &str = include_str!("wgsl/water.wgsl");

/// Number of directional sines of the ripple.
pub const WATER_WAVES: u32 = 8;
const RATIO: f32 = 1.27;
const ANGLE_STEP: f32 = 2.399_963_2;

/// One still water body: a horizontal surface at `level` over the XZ footprint `lo..hi`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaterBody {
    pub level: f32,
    /// RMS slope of the ripple (|grad h|).
    pub slope: f32,
    /// Ripple frequency in 1/m: the longest wavelength is 2 / frequency.
    pub frequency: f32,
    /// Animation phase in seconds (0 = static).
    pub time: f32,
    pub lo: [f32; 2],
    pub hi: [f32; 2],
    pub ior: f32,
    /// Beer-Lambert absorption coefficients per metre (linear RGB).
    pub sigma: [f32; 3],
    /// Artistic bend of the sun under the water (raa_water_sun): 0 straight, 1 physical
    /// refraction, more steeper (bounded at half a straight run past the vertical).
    pub sun_bend: f32,
    /// Artistic distortion of what is seen through the surface (raa_water_transmit_dir):
    /// 1 physical refraction, more wobbles more, 0 undistorted.
    pub distortion: f32,
}

impl Default for WaterBody {
    fn default() -> Self {
        Self {
            level: 0.0,
            slope: 0.0,
            frequency: 1.0,
            time: 0.0,
            lo: [0.0; 2],
            hi: [0.0; 2],
            ior: 1.33,
            sigma: [0.0; 3],
            sun_bend: 1.0,
            distortion: 1.0,
        }
    }
}

impl WaterBody {
    /// `FrameGpu.reserved[0..4]` as 16 floats (layout in water.wgsl).
    pub fn frame_lanes(&self, caustics: bool) -> [f32; 16] {
        [
            self.level,
            self.slope,
            self.frequency,
            self.time,
            self.lo[0],
            self.lo[1],
            self.hi[0],
            self.hi[1],
            self.ior,
            self.sigma[0],
            self.sigma[1],
            self.sigma[2],
            if caustics { 1.0 } else { 0.0 },
            self.sun_bend,
            self.distortion,
            1.0,
        ]
    }

    pub fn inside(&self, x: f32, z: f32) -> bool {
        x >= self.lo[0] && z >= self.lo[1] && x <= self.hi[0] && z <= self.hi[1]
    }

    /// (h, dh/dx, dh/dz) at (x, z), as `raa_water_height_gradient`.
    pub fn height_gradient(&self, x: f32, z: f32) -> [f32; 3] {
        let mut out = [0.0f32; 3];
        if self.slope <= 0.0 || self.frequency <= 0.0 {
            return out;
        }
        let norm = norm();
        for i in 0..WATER_WAVES {
            let fi = i as f32;
            let angle = 0.4 + ANGLE_STEP * fi;
            let (dz, dx) = angle.sin_cos();
            let k = std::f32::consts::PI * self.frequency * RATIO.powf(fi);
            let s = self.slope * RATIO.powf(-0.5 * fi) / norm;
            let phase = std::f32::consts::TAU * (0.618_034 * (fi + 1.0)).fract()
                - (9.81 * k).sqrt() * self.time;
            let arg = k * (dx * x + dz * z) + phase;
            out[0] += s / k * arg.sin();
            out[1] += s * arg.cos() * dx;
            out[2] += s * arg.cos() * dz;
        }
        out
    }

    /// Light concentration of the sun on the submerged point `p`, as `raa_water_caustic`
    /// (same order: entry point, three fixed-point steps, slope and curvature, first-order
    /// Jacobian, cap 1 / 0.15).
    pub fn caustic(&self, sun: [f32; 3], p: [f32; 3]) -> f32 {
        if self.slope <= 0.0
            || self.frequency <= 0.0
            || p[1] >= self.level
            || !self.inside(p[0], p[2])
        {
            return 1.0;
        }
        let eta = 1.0 / self.ior.max(1.0);
        let depth = self.level - p[1];
        let Some(flat) = refract(sun, [0.0, 1.0, 0.0], eta) else {
            return 1.0;
        };
        if flat[1] > -1e-3 {
            return 1.0;
        }
        let land = |gx: f32, gz: f32| {
            let t = refract(sun, normalize([-gx, 1.0, -gz]), eta).unwrap_or([0.0, -1.0, 0.0]);
            let l = depth / (-t[1]).max(0.05);
            [t[0] * l, t[2] * l]
        };
        let mut x = [
            p[0] - flat[0] * depth / -flat[1],
            p[2] - flat[2] * depth / -flat[1],
        ];
        for _ in 0..3 {
            let [_, gx, gz] = self.height_gradient(x[0], x[1]);
            let h = land(gx, gz);
            x = [p[0] - h[0], p[2] - h[1]];
        }
        let norm = norm();
        let (mut g, mut h) = ([0.0f32; 2], [0.0f32; 3]);
        for i in 0..WATER_WAVES {
            let fi = i as f32;
            let (dz, dx) = (0.4 + ANGLE_STEP * fi).sin_cos();
            let k = std::f32::consts::PI * self.frequency * RATIO.powf(fi);
            let s = self.slope * RATIO.powf(-0.5 * fi) / norm;
            let phase = std::f32::consts::TAU * (0.618_034 * (fi + 1.0)).fract()
                - (9.81 * k).sqrt() * self.time;
            let arg = k * (dx * x[0] + dz * x[1]) + phase;
            g[0] += s * arg.cos() * dx;
            g[1] += s * arg.cos() * dz;
            let c = s * k * arg.sin();
            h[0] -= c * dx * dx;
            h[1] -= c * dx * dz;
            h[2] -= c * dz * dz;
        }
        let e = 0.01;
        let h0 = land(g[0], g[1]);
        let h1 = land(g[0] + e * h[0], g[1] + e * h[1]);
        let h2 = land(g[0] + e * h[1], g[1] + e * h[2]);
        let a = [e + h1[0] - h0[0], h1[1] - h0[1]];
        let b = [h2[0] - h0[0], e + h2[1] - h0[1]];
        1.0 / ((a[0] * b[1] - a[1] * b[0]).abs() / (e * e)).max(0.15)
    }

    /// Mean (flat-water) horizontal run per metre of depth from a submerged receiver up to where the sun's path
    /// enters the water, as `raa_water_sun` without the ripple (bent by `sun_bend`); None when the sun is below
    /// the horizon of the water.
    pub fn sun_run(&self, sun: [f32; 3]) -> Option<[f32; 2]> {
        let t = refract(sun, [0.0, 1.0, 0.0], 1.0 / self.ior.max(1.0))?;
        if t[1] > -1e-3 {
            return None;
        }
        let y = sun[1].min(-1e-3);
        let straight = [sun[0] / y, sun[2] / y];
        let refracted = [t[0] / t[1], t[2] / t[1]];
        let b = self.sun_bend;
        let mut run = [
            straight[0] + b * (refracted[0] - straight[0]),
            straight[1] + b * (refracted[1] - straight[1]),
        ];
        let ss = straight[0] * straight[0] + straight[1] * straight[1];
        if run[0] * straight[0] + run[1] * straight[1] < -0.5 * ss {
            run = [-0.5 * straight[0], -0.5 * straight[1]];
        }
        Some(run)
    }

    /// Unit upward normal at (x, z), as `raa_water_normal`.
    pub fn normal(&self, x: f32, z: f32) -> [f32; 3] {
        let [_, gx, gz] = self.height_gradient(x, z);
        normalize([-gx, 1.0, -gz])
    }
}

fn norm() -> f32 {
    (0.5 * (0..WATER_WAVES)
        .map(|i| RATIO.powf(-(i as f32)))
        .sum::<f32>())
    .sqrt()
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / l, v[1] / l, v[2] / l]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Unpolarised Fresnel reflectance, as `raa_water_fresnel`.
pub fn fresnel(cos_i: f32, eta: f32) -> f32 {
    let c = cos_i.clamp(0.0, 1.0);
    let sin_t2 = eta * eta * (1.0 - c * c);
    if sin_t2 >= 1.0 {
        return 1.0;
    }
    let cos_t = (1.0 - sin_t2).sqrt();
    let rs = (eta * c - cos_t) / (eta * c + cos_t);
    let rp = (c - eta * cos_t) / (c + eta * cos_t);
    0.5 * (rs * rs + rp * rp)
}

/// Refraction of `d` through `n` (dot(d, n) < 0), as `raa_water_refract`; None on TIR.
pub fn refract(d: [f32; 3], n: [f32; 3], eta: f32) -> Option<[f32; 3]> {
    let c = -dot(d, n);
    let k = 1.0 - eta * eta * (1.0 - c * c);
    if k < 0.0 {
        return None;
    }
    let f = eta * c - k.sqrt();
    Some(normalize([
        eta * d[0] + f * n[0],
        eta * d[1] + f * n[1],
        eta * d[2] + f * n[2],
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool() -> WaterBody {
        WaterBody {
            level: 1.01,
            slope: 0.12,
            frequency: 1.6,
            lo: [-11.14, -6.14],
            hi: [19.25, 6.05],
            sigma: [0.478, 0.128, 0.083],
            ..WaterBody::default()
        }
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let w = pool();
        let h = 1e-3;
        for &(x, z) in &[(0.3, -1.7), (5.0, 2.25), (-9.6, 0.35), (12.0, -4.0)] {
            let [_, gx, gz] = w.height_gradient(x, z);
            let fx = (w.height_gradient(x + h, z)[0] - w.height_gradient(x - h, z)[0]) / (2.0 * h);
            let fz = (w.height_gradient(x, z + h)[0] - w.height_gradient(x, z - h)[0]) / (2.0 * h);
            assert!((gx - fx).abs() < 2e-3, "{gx} {fx}");
            assert!((gz - fz).abs() < 2e-3, "{gz} {fz}");
        }
    }

    #[test]
    fn rms_slope_is_the_parameter() {
        let w = pool();
        let mut sum = 0.0f64;
        let n = 400;
        for i in 0..n {
            for j in 0..n {
                let [_, gx, gz] = w.height_gradient(i as f32 * 0.0731, j as f32 * 0.0619);
                sum += f64::from(gx * gx + gz * gz);
            }
        }
        let rms = (sum / f64::from(n * n)).sqrt() as f32;
        assert!((rms - w.slope).abs() < 0.15 * w.slope, "rms {rms}");
    }

    #[test]
    fn flat_and_static_defaults() {
        let w = WaterBody {
            slope: 0.0,
            ..pool()
        };
        assert_eq!(w.normal(1.0, 2.0), [0.0, 1.0, 0.0]);
        let lanes = pool().frame_lanes(true);
        assert_eq!(lanes[15], 1.0);
        assert_eq!(lanes[12], 1.0);
        assert_eq!(pool().frame_lanes(false)[12], 0.0);
        assert_eq!(lanes[13], 1.0);
        assert_eq!(lanes[14], 1.0);
    }

    /// Flat-floor check of `raa_water_caustic`: splat the sun refracted through a fine grid of
    /// the surface onto a floor 1 m down (Wallace's area ratio, brute force) and compare the
    /// histogram mean (energy conservation, ~1) and the presence of bright lines.
    #[test]
    fn refracted_grid_concentrates_light_and_conserves_it() {
        let w = WaterBody {
            slope: 0.1,
            frequency: 4.0,
            ..pool()
        };
        let sun = normalize([0.3, -0.9, 0.2]);
        let eta = 1.0 / w.ior;
        let (n, cell, depth) = (600usize, 0.005f32, 1.0f32);
        let mut bins = vec![0u32; 100 * 100];
        let mut landed = 0u32;
        for j in 0..n {
            for i in 0..n {
                let (x, z) = (i as f32 * cell, j as f32 * cell);
                let t = refract(sun, w.normal(x, z), eta).unwrap();
                let (hx, hz) = (x + t[0] * depth / -t[1], z + t[2] * depth / -t[1]);
                // Floor window [0.5, 2.5) in 2 cm bins, away from the grid edges.
                let (bx, bz) = (((hx - 0.5) / 0.02).floor(), ((hz - 0.5) / 0.02).floor());
                if (0.0..100.0).contains(&bx) && (0.0..100.0).contains(&bz) {
                    bins[bz as usize * 100 + bx as usize] += 1;
                    landed += 1;
                }
            }
        }
        let per_bin = (0.02 / cell) * (0.02 / cell);
        let mean = landed as f32 / (bins.len() as f32 * per_bin);
        let max = *bins.iter().max().unwrap() as f32 / per_bin;
        assert!((mean - 1.0).abs() < 0.1, "mean {mean}");
        assert!(max > 2.0, "max {max}");
        // The per-receiver formula (shader twin) at the bin centres follows the histogram.
        let (mut sa, mut sb, mut sab, mut saa, mut sbb) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for (idx, &count) in bins.iter().enumerate() {
            let (bx, bz) = ((idx % 100) as f32, (idx / 100) as f32);
            let p = [
                0.5 + 0.02 * (bx + 0.5),
                w.level - depth,
                0.5 + 0.02 * (bz + 0.5),
            ];
            let a = f64::from(w.caustic(sun, p));
            let b = f64::from(count as f32 / per_bin);
            sa += a;
            sb += b;
            sab += a * b;
            saa += a * a;
            sbb += b * b;
        }
        let m = bins.len() as f64;
        let corr = (sab - sa * sb / m) / ((saa - sa * sa / m) * (sbb - sb * sb / m)).sqrt();
        let analytic_mean = sa / m;
        assert!(
            (analytic_mean - 1.0).abs() < 0.15,
            "analytic mean {analytic_mean}"
        );
        assert!(corr > 0.6, "correlation {corr}");
    }

    /// The strip lit next to a wall under 1 m of water at a 75 degree sun: tan 15 - tan(refracted)
    /// = 0.07 m when physical, 6x that bent (0.42, clamped to 1.5 x tan 15 = 0.40), none straight.
    #[test]
    fn sun_bend_widens_the_lit_strip() {
        let e = 75f32.to_radians();
        let sun = [-e.cos(), -e.sin(), 0.0];
        let straight = 1.0 / e.tan();
        let strip = |b: f32| {
            straight
                - WaterBody {
                    sun_bend: b,
                    ..pool()
                }
                .sun_run(sun)
                .unwrap()[0]
        };
        assert!(strip(0.0).abs() < 1e-5);
        assert!((strip(1.0) - 0.0696).abs() < 2e-3, "{}", strip(1.0));
        assert!((strip(6.0) - 1.5 * straight).abs() < 1e-4, "{}", strip(6.0));
        assert!((strip(4.0) - 4.0 * strip(1.0)).abs() < 1e-3);
    }

    #[test]
    fn fresnel_and_refraction() {
        // Normal incidence air -> water: ((1 - 1.33) / (1 + 1.33))^2 = 0.02006.
        assert!((fresnel(1.0, 1.0 / 1.33) - 0.02006).abs() < 1e-4);
        // Water -> air beyond the critical angle reflects totally.
        assert_eq!(fresnel(0.3, 1.33), 1.0);
        let d = normalize([0.26, -0.966, 0.0]);
        let t = refract(d, [0.0, 1.0, 0.0], 1.0 / 1.33).unwrap();
        // Snell: sin_t = sin_i / 1.33
        assert!((t[0] - d[0] / 1.33).abs() < 1e-5);
        assert!(refract([0.9, 0.436, 0.0], [0.0, -1.0, 0.0], 1.33).is_none());
    }

    #[test]
    fn wgsl_declares_the_shared_functions() {
        for name in [
            "fn raa_water_unpack(",
            "fn raa_water_height_gradient(",
            "fn raa_water_normal(",
            "fn raa_water_fresnel(",
            "fn raa_water_refract(",
            "fn raa_water_sun_entry(",
            "fn raa_water_caustic(",
        ] {
            assert!(WATER_WGSL.contains(name), "{name}");
        }
        assert!(WATER_WGSL.contains("const RAA_WATER_WAVES: u32 = 8u;"));
        assert!(WATER_WGSL.contains("const RAA_WATER_RATIO: f32 = 1.27;"));
    }
}
