//! M2-TILE-SELECTION CPU fixture oracle; estimator specification is in the crate README.

/// Exact strongest 127 of at most 256 qualified, finite, nonnegative scores.
/// Unused output slots are zero; eligibility is independent of score and handle.
///
/// Handoff: coverage consumes `pub(crate) fn select_tile(candidates: &[(u64, f32)], out: &mut [u64; 127]) -> (usize, u32)`
/// in `crates/raa-lit/src/tile_selection.rs`, attached by `#[cfg(test)] pub(crate) mod tile_selection;`
/// in `crates/raa-lit/src/lib.rs`. It passes qualified `(LightHandle, score)` pairs from
/// the `LocalLights` registry, using the README's depth, bounds and estimator rules.
/// Handoff: GPU selection consumes `pub(crate) fn select_tile(candidates: &[(u64, f32)], out: &mut [u64; 127]) -> (usize, u32)`
/// in `crates/raa-lit/src/tile_selection.rs`, attached by the test module in `lib.rs` as
/// a specification for its compute entry point's fixed network. GPU indices
/// are dense `LocalLights::write_gpu` indices in ascending stable-handle order, including
/// after removal; the CPU fixture is never called by or compared against a GPU run.
/// Handoff: order-35 recording consumes `pub(crate) fn select_tile(candidates: &[(u64, f32)], out: &mut [u64; 127]) -> (usize, u32)`
/// in `crates/raa-lit/src/tile_selection.rs`, attached by the test module in `lib.rs`;
/// it attaches selection per covered tile/view and sums the returned dropped-reference
/// counts for submitted views keyed by scene-frame invocation, with no CPU registry mutation.
pub(crate) fn select_tile(candidates: &[(u64, f32)], out: &mut [u64; 127]) -> (usize, u32) {
    use raa_shader::{MAX_LOCAL_LIGHTS, TILE_MAX_LIGHTS};
    assert!(candidates.len() <= MAX_LOCAL_LIGHTS);
    let mut entries = [(false, 0.0, 0_u64); MAX_LOCAL_LIGHTS];
    for (entry, &(handle, score)) in entries.iter_mut().zip(candidates) {
        assert!(score.is_finite() && score >= 0.0);
        *entry = (true, score, handle);
    }
    let before = |a: (bool, f32, u64), b: (bool, f32, u64)| {
        (a.0 && !b.0) || (a.0 == b.0 && (a.1 > b.1 || (a.1 == b.1 && a.2 < b.2)))
    };
    let mut width = 2;
    while width <= MAX_LOCAL_LIGHTS {
        let mut stride = width / 2;
        while stride != 0 {
            for i in 0..MAX_LOCAL_LIGHTS {
                let j = i ^ stride;
                if j > i {
                    let swap = if i & width == 0 {
                        before(entries[j], entries[i])
                    } else {
                        before(entries[i], entries[j])
                    };
                    if swap {
                        entries.swap(i, j);
                    }
                }
            }
            stride /= 2;
        }
        width *= 2;
    }
    let count = candidates.len().min(TILE_MAX_LIGHTS as usize);
    out.fill(0);
    for (slot, entry) in out.iter_mut().zip(&entries[..count]) {
        *slot = entry.2;
    }
    (count, (candidates.len() - count) as u32)
}

#[cfg(test)]
mod tests {
    use super::select_tile;
    use crate::light_math::{
        area_light_emits, representative_point, spot_factor, windowed_falloff,
    };
    use raa_math::{Mat4, Vec3};

    fn numbers(text: &str) -> Vec<f32> {
        text.split(',').map(|s| s.parse().unwrap()).collect()
    }

    fn handles(text: &str) -> Vec<u64> {
        if text == "-" {
            Vec::new()
        } else if let Some((a, b)) = text.split_once("..") {
            let (a, b): (u64, u64) = (a.parse().unwrap(), b.parse().unwrap());
            if a <= b {
                (a..=b).collect()
            } else {
                (b..=a).rev().collect()
            }
        } else {
            text.split(',').map(|s| s.parse().unwrap()).collect()
        }
    }

    #[test]
    fn ranking_text_fixtures() {
        let mut dropped_sum = 0;
        for line in include_str!("../tests/fixtures/tile-selection.txt").lines() {
            let f: Vec<_> = line.split_whitespace().collect();
            if f.is_empty() || f[0].starts_with('#') {
                continue;
            }
            match f[0] {
                "rank" => {
                    let mut candidates: Vec<(u64, f32)> = if f[2] == "-" {
                        Vec::new()
                    } else if let Some((mode, count)) = f[2].split_once('/') {
                        (1..=count.parse::<u64>().unwrap())
                            .map(|h| (h, if mode == "ramp" { h as f32 } else { 0.0 }))
                            .collect()
                    } else {
                        f[2].split(',')
                            .map(|s| {
                                let (h, score) = s.split_once(':').unwrap();
                                (h.parse().unwrap(), score.parse().unwrap())
                            })
                            .collect()
                    };
                    let expected = handles(f[3]);
                    let dropped: u32 = f[4].parse().unwrap();
                    dropped_sum += select_tile(&candidates, &mut [0; 127]).1;
                    for _ in 0..3 {
                        let mut out = [u64::MAX; 127];
                        assert_eq!(
                            select_tile(&candidates, &mut out),
                            (expected.len(), dropped),
                            "{line}"
                        );
                        assert_eq!(&out[..expected.len()], expected, "{line}");
                        assert!(out[expected.len()..].iter().all(|&h| h == 0));
                        candidates.reverse();
                        if !candidates.is_empty() {
                            candidates.rotate_left(1);
                        }
                    }
                }
                "sum" => assert_eq!(dropped_sum, f[1].parse::<u32>().unwrap()),
                "tile" => {
                    let v = numbers(f[2]); // viewport xy, tile xy, depth samples
                    let nearest = v[4..]
                        .iter()
                        .copied()
                        .filter(|d| d.is_finite() && (0.0..1.0).contains(d))
                        .reduce(f32::min);
                    let actual = nearest.map(|z| {
                        let centre = |axis: usize| {
                            let lo = v[axis + 2] * raa_shader::TILE_SIZE as f32;
                            (lo + (lo + 16.0).min(v[axis])) * 0.5 / v[axis]
                        };
                        let mut inverse = Mat4::from_translation(Vec3::new(2.0, 3.0, 4.0));
                        inverse.cols[3][3] = 2.0;
                        inverse.transform_point(Vec3::new(
                            2.0 * centre(0) - 1.0,
                            1.0 - 2.0 * centre(1),
                            z,
                        ))
                    });
                    if f[3] == "-" {
                        assert!(actual.is_none(), "{line}");
                    } else {
                        let want = numbers(f[3]);
                        assert!(
                            (actual.unwrap() - Vec3::new(want[0], want[1], want[2])).length()
                                < 1e-6,
                            "{line}"
                        );
                    }
                }
                "bounds" => {
                    let v = numbers(f[3]); // centre xyz, range, full width/height
                    let p = Vec3::new(v[0], v[1], v[2]);
                    let radius = v[3]
                        + if f[2] == "area" {
                            v[4].hypot(v[5]) * 0.5
                        } else {
                            0.0
                        };
                    let covered = f.get(5) != Some(&"empty");
                    let reaches = covered
                        && (f[2] == "directional"
                            || [p.x, p.y, p.z]
                                .iter()
                                .all(|&c| c + radius >= 0.0 && c - radius <= 1.0));
                    assert_eq!(reaches, f[4] == "yes", "{line}");
                }
                "score" => {
                    let v = numbers(f[3]); // surface xyz, range, width/height, RGB*intensity, cone cosines
                    let surface = Vec3::new(v[0], v[1], v[2]);
                    let point = if f[2] == "area" {
                        representative_point(Vec3::ZERO, Vec3::Z, [v[4], v[5]], surface)
                    } else {
                        Vec3::ZERO
                    };
                    let delta = surface - point;
                    let mut factor = windowed_falloff(delta.length(), v[3]);
                    match f[2] {
                        "directional" => factor = 1.0,
                        "spot" => factor *= spot_factor(delta.normalize_or_zero().z, v[9], v[10]),
                        "area" if !area_light_emits(Vec3::ZERO, Vec3::Z, surface) => factor = 0.0,
                        _ => {}
                    }
                    let luminance = 0.2126 * f64::from(v[6])
                        + 0.7152 * f64::from(v[7])
                        + 0.0722 * f64::from(v[8]);
                    let raw = luminance * f64::from(factor);
                    let score = if raw.is_nan() {
                        0.0
                    } else {
                        raw.clamp(0.0, f64::from(f32::MAX)) as f32
                    };
                    let expected: f32 = f[4].parse().unwrap();
                    assert!(score.is_finite() && score >= 0.0, "{line}");
                    assert!(
                        (score - expected).abs() <= 1e-6_f32.max(expected * 1e-5),
                        "{line}: {score}"
                    );
                }
                _ => panic!("unknown fixture: {line}"),
            }
        }
        assert_eq!(
            (
                raa_shader::TILE_SIZE,
                raa_shader::TILE_MAX_LIGHTS,
                raa_shader::TILE_WORDS
            ),
            (16, 127, 128)
        );
    }
}
