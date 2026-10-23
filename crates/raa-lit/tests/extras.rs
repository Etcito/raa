use raa_lit::{INSTANCE_EXTRA_GPU_SIZE, InstanceExtra};

fn f32_at(b: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

#[test]
fn default_bytes_match_s10() {
    let mut out = Vec::new();
    InstanceExtra::default().write_gpu(&mut out);
    assert_eq!(out.len(), 128);
    let mut want = vec![1.0f32; 4];
    want.extend([0.0, 1.0, 0.0, 0.0]);
    want.extend([0.0; 24]);
    let bytes: Vec<u8> = want.iter().flat_map(|x| x.to_le_bytes()).collect();
    assert_eq!(out, bytes);
}

#[test]
fn distinct_values_land_at_their_offsets() {
    let mut x = InstanceExtra {
        tint: [1.0, 2.0, 3.0, 4.0],
        squash: [5.0, 6.0, 7.0, 8.0],
        jiggle: [9.0, 10.0, 11.0, 12.0],
        params: [[0.0; 4]; 4],
    };
    for (i, p) in x.params.iter_mut().enumerate() {
        for (j, v) in p.iter_mut().enumerate() {
            *v = 13.0 + (i * 4 + j) as f32;
        }
    }
    let mut out = vec![0xAA];
    x.write_gpu(&mut out);
    let b = &out[1..];
    assert_eq!(b.len(), INSTANCE_EXTRA_GPU_SIZE);
    for k in 0..28 {
        assert_eq!(f32_at(b, k * 4), (k + 1) as f32, "float {k}");
    }
    assert!(b[112..128].iter().all(|&v| v == 0));
}

#[test]
fn record_is_128_bytes() {
    let mut out = Vec::new();
    InstanceExtra::default().write_gpu(&mut out);
    InstanceExtra::default().write_gpu(&mut out);
    assert_eq!(out.len(), 2 * 128);
}
