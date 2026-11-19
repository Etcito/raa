use raa_bloom::filters::{
    BOX_WEIGHTS, TENT, composite, down, down_first, half_size, karis_weight, luma, up,
};

fn assert_rgb(actual: [f32; 3], expected: [f32; 3]) {
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
    }
}

#[test]
fn luma_and_karis_weight() {
    assert!((luma([1.0, 1.0, 1.0]) - 1.0).abs() <= 1e-6);
    assert!((karis_weight(0.5, [1.0, 1.0, 1.0]) - 0.25).abs() <= 1e-6);
}

#[test]
fn plain_downsample_preserves_constant_colour() {
    assert_eq!(BOX_WEIGHTS.iter().sum::<f32>(), 1.0);
    let boxes = [[2.0, 4.0, 8.0]; 5];
    assert_rgb(down(&boxes), [2.0, 4.0, 8.0]);
}

#[test]
fn first_downsample_renormalizes_karis_weights() {
    let boxes = [[3.0, 3.0, 3.0], [0.0; 3], [0.0; 3], [0.0; 3], [0.0; 3]];
    // The bright inner box has weight 0.5 / 4 = 0.125. The four black boxes
    // each keep weight 0.125, so the normalized result is 0.375 / 0.625 = 0.6.
    assert_rgb(down_first(&boxes), [0.6, 0.6, 0.6]);
}

#[test]
fn tent_upsample_has_expected_impulse_and_constant_response() {
    let mut impulse = [[0.0; 3]; 9];
    impulse[4] = [1.0; 3];
    assert_rgb(up(&impulse), [0.25; 3]);
    assert_eq!(TENT.iter().sum::<f32>(), 1.0);
    assert_rgb(up(&[[2.0, 4.0, 8.0]; 9]), [2.0, 4.0, 8.0]);
}

#[test]
fn composite_and_half_size_match_shader_rules() {
    assert_rgb(composite([1.0; 3], [3.0; 3], 0.25), [1.5; 3]);
    assert_eq!(half_size((1280, 720)), (640, 360));
    assert_eq!(half_size((1, 1)), (1, 1));
}
