use raa_bloom::BloomDesc;

#[test]
fn defaults_are_bloom_values() {
    let d = BloomDesc::default();
    assert_eq!((d.strength, d.radius), (0.04, 0.85));
    d.validate().expect("default");
}

#[test]
fn boundaries_accepted() {
    for (s, r) in [(0.0, 0.0), (1.0, 1.0)] {
        BloomDesc {
            strength: s,
            radius: r,
        }
        .validate()
        .expect("ok");
    }
}

#[test]
fn every_refusal_has_code_and_message() {
    let d = BloomDesc::default();
    for v in [-0.1, 1.1, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for (field, desc) in [
            ("strength", BloomDesc { strength: v, ..d }),
            ("radius", BloomDesc { radius: v, ..d }),
        ] {
            let e = desc.validate().expect_err("refused");
            assert_eq!(e.code(), "bloom.invalid");
            let m = e.to_string();
            assert!(!m.is_empty() && m.contains(field), "{m}");
        }
    }
}
