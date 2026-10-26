use raa_ssr::SsrDesc;
use raa_ssr::hiz::{mip_count, next_size, reduce_min};

struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn reduce_once(src: &[f32], w: u32, h: u32) -> (Vec<f32>, u32, u32) {
    let mut dst = Vec::new();
    let (dw, dh) = reduce_min(src, w, h, &mut dst);
    (dst, dw, dh)
}

#[test]
fn mip_count_1280x720_is_11() {
    assert_eq!(mip_count(1280, 720), 11);
}

#[test]
fn mip_count_1x1_is_1() {
    assert_eq!(mip_count(1, 1), 1);
    assert_eq!(mip_count(2, 1), 2);
}

#[test]
fn next_size_halves_and_keeps_one() {
    assert_eq!(next_size(5, 3), (2, 1));
    assert_eq!(next_size(1, 1), (1, 1));
    assert_eq!(next_size(1280, 720), (640, 360));
}

#[test]
fn odd_corner_takes_the_extra_column_and_row() {
    // 5x3: destination (1, 0) covers columns 2..=4 and rows 0..=2, a 3x3 footprint.
    let mut src = vec![1.0f32; 15];
    src[2 * 5 + 4] = 0.25;
    let (dst, dw, dh) = reduce_once(&src, 5, 3);
    assert_eq!((dw, dh), (2, 1));
    assert_eq!(dst, vec![1.0, 0.25]);
}

#[test]
fn extra_row_alone_reaches_its_destination() {
    let mut src = vec![1.0f32; 15];
    src[2 * 5] = 0.5;
    let (dst, _, _) = reduce_once(&src, 5, 3);
    assert_eq!(dst, vec![0.5, 1.0]);
}

#[test]
fn every_source_sample_reaches_a_destination_texel() {
    for (w, h) in [(5, 3), (3, 5), (7, 7), (1, 4), (4, 1), (1, 1)] {
        for cell in 0..w * h {
            let mut src = vec![1.0f32; (w * h) as usize];
            src[cell as usize] = 0.0;
            let (dst, _, _) = reduce_once(&src, w, h);
            assert!(dst.contains(&0.0), "{w}x{h} cell {cell} lost");
        }
    }
}

#[test]
fn pyramid_last_level_is_the_global_min() {
    let mut rng = Lcg(7);
    for (w, h) in [(37, 23), (1, 9), (64, 64), (5, 3), (1280, 720)] {
        let level0: Vec<f32> = (0..w * h).map(|_| rng.next_f32()).collect();
        let global = level0.iter().copied().fold(f32::INFINITY, f32::min);
        let mut cur = level0;
        let mut next = Vec::new();
        let (mut cw, mut ch) = (w, h);
        for _ in 1..mip_count(w, h) {
            let (nw, nh) = reduce_min(&cur, cw, ch, &mut next);
            std::mem::swap(&mut cur, &mut next);
            (cw, ch) = (nw, nh);
        }
        assert_eq!((cw, ch), (1, 1), "{w}x{h} ends at 1x1");
        assert_eq!(cur, vec![global], "{w}x{h} last level");
    }
}

#[test]
fn reuses_the_output_vector() {
    let src = vec![0.5f32; 16];
    let mut dst = Vec::with_capacity(64);
    let cap = dst.capacity();
    reduce_min(&src, 4, 4, &mut dst);
    reduce_min(&src, 4, 4, &mut dst);
    assert_eq!(dst.len(), 4);
    assert_eq!(dst.capacity(), cap);
}

#[test]
fn desc_defaults_validate() {
    let d = SsrDesc::default();
    assert_eq!(
        (d.intensity, d.max_roughness, d.thickness_m),
        (1.0, 0.6, 0.15)
    );
    d.validate().expect("defaults");
}

#[test]
fn desc_boundaries_are_accepted() {
    for (intensity, max_roughness, thickness_m) in [(0.0, 0.0, 1e-6), (2.0, 1.0, 10.0)] {
        SsrDesc {
            intensity,
            max_roughness,
            thickness_m,
        }
        .validate()
        .expect("boundary accepted");
    }
}

#[test]
fn desc_refusals_name_the_field_with_post_invalid() {
    let d = SsrDesc::default();
    let bad = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
    let out_of_range = [
        (
            "intensity",
            SsrDesc {
                intensity: -0.1,
                ..d
            },
        ),
        (
            "intensity",
            SsrDesc {
                intensity: 2.1,
                ..d
            },
        ),
        (
            "max_roughness",
            SsrDesc {
                max_roughness: -0.1,
                ..d
            },
        ),
        (
            "max_roughness",
            SsrDesc {
                max_roughness: 1.1,
                ..d
            },
        ),
        (
            "thickness_m",
            SsrDesc {
                thickness_m: 0.0,
                ..d
            },
        ),
        (
            "thickness_m",
            SsrDesc {
                thickness_m: -0.1,
                ..d
            },
        ),
        (
            "thickness_m",
            SsrDesc {
                thickness_m: 10.1,
                ..d
            },
        ),
    ];
    for (field, desc) in out_of_range {
        let e = desc.validate().expect_err("refused");
        assert_eq!(e.code(), "post.invalid");
        assert!(e.to_string().contains(field), "{e}");
    }
    for v in bad {
        for (field, desc) in [
            ("intensity", SsrDesc { intensity: v, ..d }),
            (
                "max_roughness",
                SsrDesc {
                    max_roughness: v,
                    ..d
                },
            ),
            (
                "thickness_m",
                SsrDesc {
                    thickness_m: v,
                    ..d
                },
            ),
        ] {
            let e = desc.validate().expect_err("refused");
            assert_eq!(e.code(), "post.invalid");
            assert!(e.to_string().contains(field), "{e}");
        }
    }
}
