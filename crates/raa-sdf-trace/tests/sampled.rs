//! Sampled-resource ABI and shader validation. No GPU or CPU/GPU comparisons.
use raa_sdf_trace::{VERSION, distance, parse, wgsl};
fn fixture(compact: bool) -> Vec<u8> {
    let mut d = vec![
        1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -2.0, -2.0, -2.0,
        2.0, 3.0, 3.0, 3.0, 6.0, 3.0, 0.0,
    ];
    for z in [-2.0f32, 0.0, 2.0] {
        for y in [-2.0f32, 0.0, 2.0] {
            for x in [-2.0f32, 0.0, 2.0] {
                d.push((x * x + y * y + z * z).sqrt() - 1.0);
            }
        }
    }
    if compact {
        let samples = d.split_off(24);
        d[23] = 1.0;
        for pair in samples.chunks(2) {
            let a = u32::from(half::f16::from_f32(pair[0]).to_bits());
            let b = u32::from(half::f16::from_f32(*pair.get(1).unwrap_or(&0.0)).to_bits());
            d.push(f32::from_bits(a | (b << 16)));
        }
    }
    let mut w = vec![
        u32::from_le_bytes(*b"SDTP"),
        VERSION,
        1,
        1,
        1,
        d.len() as u32,
        1,
        0,
        0,
        1,
        0,
        24,
    ];
    w.extend([-2.0f32, -2.0, -2.0, 2.0, 2.0, 2.0, 6.0].map(f32::to_bits));
    w.push(0);
    w.push(9);
    w.extend(d.iter().map(|v| v.to_bits()));
    w.extend([0.7f32, 0.2, 0.1, 0.5, 0.0, 0.0, 0.0, 0.0].map(f32::to_bits));
    for _ in 0..2 {
        w.extend(
            [
                1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ]
            .map(f32::to_bits),
        );
    }
    w.extend([1.0f32.to_bits(), 0, u32::MAX, 7, 0, 0, 0, 0]);
    w.into_iter().flat_map(u32::to_le_bytes).collect()
}
fn set(bytes: &mut [u8], word: usize, value: u32) {
    bytes[word * 4..word * 4 + 4].copy_from_slice(&value.to_le_bytes());
}
#[test]
fn sampled_resource_keeps_sign_bounds_and_material() {
    for compact in [false, true] {
        let p = parse(&fixture(compact)).unwrap();
        assert_eq!(p.sources.len(), 1);
        assert!(distance(&p, 0, [0.0; 3]) < 0.0);
        assert!(distance(&p, 0, [4.0, 0.0, 0.0]) > 0.0);
        assert_eq!(raa_sdf_trace::material(&p, 0, [0.0; 3]), 0);
        assert_eq!(p.data.len(), if compact { 38 } else { 51 });
    }
}
#[test]
fn malformed_resources_and_v1_opcode_are_refused() {
    for (word, value) in [
        (1, 1),
        (21 + 17, 0),
        (21 + 12, 0),
        (21 + 23, 2.0f32.to_bits()),
        (21 + 18, 1.0f32.to_bits()),
        (21 + 21, 5.0f32.to_bits()),
        (21 + 22, 0.5f32.to_bits()),
        (21 + 24, (-1.0f32).to_bits()),
        (21 + 25, f32::NAN.to_bits()),
        (21 + 24 + 13, 1.0f32.to_bits()),
        (18, 1.0f32.to_bits()),
    ] {
        let mut bytes = fixture(false);
        set(&mut bytes, word, value);
        assert!(parse(&bytes).is_err(), "word {word}");
    }
    let mut compact = fixture(true);
    set(&mut compact, 21 + 24, 0x7e007e00);
    assert!(parse(&compact).is_err());
    let mut bytes = fixture(false);
    bytes.truncate(bytes.len() - 4);
    assert!(parse(&bytes).is_err());
}
#[test]
fn sampled_shared_evaluator_and_volume_shaders_validate() {
    let p = parse(&fixture(false)).unwrap();
    for code in [
        wgsl::eval_module(&p),
        wgsl::trace_module(&p),
        wgsl::volume_module(&p),
    ] {
        let module = naga::front::wgsl::parse_str(&code).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
    }
}
