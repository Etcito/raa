//! Material atlas mip chain (S48, P5-ATLAS-MIPS), CPU side: the box downsample
//! `raa_lit::downsample_rgba8` that builds each mip level from the one above. Every expected
//! value is derived by hand in its test, from the rules of the module doc of `mipgen`.

use raa_lit::downsample_rgba8;

/// Flattens texels (rgba, rows top first) into rgba8 bytes.
fn flat<const N: usize>(texels: &[[u8; 4]; N]) -> Vec<u8> {
    texels.iter().flatten().copied().collect()
}

/// Texel `(x, y)` of a `width`-texel-wide rgba8 image.
fn texel(rgba: &[u8], width: usize, x: usize, y: usize) -> [u8; 4] {
    let at = (y * width + x) * 4;
    <[u8; 4]>::try_from(&rgba[at..at + 4]).unwrap()
}

/// A 4 x 4 image in rows top first; texel `(x, y)` is entry `y * 4 + x`.
const FOUR_BY_FOUR: [[u8; 4]; 16] = [
    [0, 0, 0, 0],
    [255, 255, 255, 255],
    [10, 20, 30, 40],
    [11, 21, 31, 41],
    [0, 0, 0, 0],
    [255, 255, 255, 255],
    [12, 22, 32, 42],
    [13, 23, 33, 43],
    [255, 0, 0, 255],
    [255, 0, 0, 255],
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [255, 0, 0, 255],
    [255, 0, 0, 255],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
];

/// A 3 x 5 image (3 wide, 5 high) in rows top first; texel `(x, y)` is entry `y * 3 + x`.
const THREE_BY_FIVE: [[u8; 4]; 15] = [
    [0, 0, 7, 1],
    [255, 0, 7, 2],
    [20, 0, 7, 3],
    [0, 0, 7, 4],
    [255, 0, 7, 5],
    [40, 40, 7, 6],
    [0, 80, 7, 7],
    [255, 80, 7, 8],
    [60, 80, 7, 9],
    [0, 120, 7, 10],
    [255, 120, 7, 11],
    [80, 120, 7, 12],
    [0, 160, 7, 13],
    [255, 160, 7, 14],
    [100, 160, 7, 15],
];

#[test]
fn linear_blocks_average_with_half_up_rounding() {
    // Block (0,0): r, g, b, a = 0, 255, 0, 255 in the pattern of the task: (0 + 255 + 0 + 255 +
    // 2) / 4 = 128 in every channel.
    // Block (1,0): r 10..13 sum 46 -> 48 / 4 = 12; g 20..23 sum 86 -> 22; b 30..33 sum 126 ->
    // 32; a 40..43 sum 166 -> 42.
    // Block (0,1): four texels [255, 0, 0, 255]: (1020 + 2) / 4 = 255 and 0, 0, 255.
    // Block (1,1): r 1, 5, 9, 13 sum 28 -> 7; g 2, 6, 10, 14 sum 32 -> 8; b 3..15 sum 36 -> 9;
    // a 4..16 sum 40 -> 10.
    let (out, w, h) = downsample_rgba8(&flat(&FOUR_BY_FOUR), 4, 4, false);
    assert_eq!((w, h), (2, 2));
    assert_eq!(
        out,
        flat(&[
            [128, 128, 128, 128],
            [12, 22, 32, 42],
            [255, 0, 0, 255],
            [7, 8, 9, 10],
        ])
    );
}

#[test]
fn srgb_block_averages_in_linear_light() {
    // Two black and two white texels: decoded 0, 0, 1, 1, mean 0.5, re-encoded as
    // 1.055 * 0.5^(1/2.4) - 0.055 = 0.73536 (x 255 = 187.52), which quantizes to 188.
    // Alpha stays on the linear rule: (0 + 255 + 0 + 255 + 2) / 4 = 128.
    let block = flat(&[
        [0, 0, 0, 0],
        [255, 255, 255, 255],
        [0, 0, 0, 0],
        [255, 255, 255, 255],
    ]);
    let (out, w, h) = downsample_rgba8(&block, 2, 2, true);
    assert_eq!((w, h), (1, 1));
    assert_eq!(out, vec![188, 188, 188, 128]);
    let (linear, _, _) = downsample_rgba8(&block, 2, 2, false);
    assert_eq!(linear, vec![128, 128, 128, 128]);
    // The same two blocks of the 4 x 4 image sRGB: block (0,0) as above; block (0,1) is
    // [255, 0, 0, 255] four times, r decodes to 1 and re-encodes to 1 (255), g and b stay 0,
    // alpha 255.
    let (srgb_4x4, _, _) = downsample_rgba8(&flat(&FOUR_BY_FOUR), 4, 4, true);
    assert_eq!(&srgb_4x4[0..4], &[188, 188, 188, 128]);
    assert_eq!(&srgb_4x4[8..12], &[255, 0, 0, 255]);
}

#[test]
fn odd_sides_repeat_the_last_column_and_row() {
    // 3 x 5 gives 2 x 3. Output (0,0) taps (0,0), (1,0), (0,1), (1,1): linear r (0 + 255 + 0 +
    // 255 + 2) / 4 = 128, g 0, b (28 + 2) / 4 = 7, a (1 + 2 + 4 + 5 + 2) / 4 = 3.
    // Output (1,0) taps (2,0), (2,0), (2,1), (2,1): r (120 + 2) / 4 = 30, g (80 + 2) / 4 = 20,
    // b 7, a (18 + 2) / 4 = 5.
    // Output (1,2) taps (2,4) four times (x 3 -> 2, y 5 -> 4): the source texel exactly.
    let (out, w, h) = downsample_rgba8(&flat(&THREE_BY_FIVE), 3, 5, false);
    assert_eq!((w, h), (2, 3));
    assert_eq!(
        out,
        flat(&[
            [128, 0, 7, 3],
            [30, 20, 7, 5],
            [128, 100, 7, 9],
            [70, 100, 7, 11],
            [128, 160, 7, 14],
            [100, 160, 7, 15],
        ])
    );
    // sRGB: output (0,0) has r taps 0, 255, 0, 255 (mean 0.5, 188) and g 0, b 7, alpha 3 as
    // above. Output (0,2) has g constant 160, and output (1,2) has four equal taps: decode and
    // re-encode of equal values return the value itself, so those channels are the source's.
    let (srgb, _, _) = downsample_rgba8(&flat(&THREE_BY_FIVE), 3, 5, true);
    assert_eq!(texel(&srgb, 2, 0, 0), [188, 0, 7, 3]);
    // Output (0,2) taps (0,4), (1,4) twice: r 188, g 160 (constant), b 7, a 14.
    assert_eq!(texel(&srgb, 2, 0, 2), [188, 160, 7, 14]);
    assert_eq!(texel(&srgb, 2, 1, 2), THREE_BY_FIVE[14]);
    assert_eq!(texel(&out, 2, 1, 2), THREE_BY_FIVE[14]);
}

#[test]
fn each_level_is_made_from_the_one_above() {
    // Level 1 is the 2 x 2 image of the first test: [128 x4], [12, 22, 32, 42], [255, 0, 0, 255],
    // [7, 8, 9, 10]. Level 2 averages those four texels: r (402 + 2) / 4 = 101, g (158 + 2) / 4
    // = 40, b (169 + 2) / 4 = 42, a (435 + 2) / 4 = 109.
    let (level1, w1, h1) = downsample_rgba8(&flat(&FOUR_BY_FOUR), 4, 4, false);
    assert_eq!((w1, h1), (2, 2));
    let (level2, w2, h2) = downsample_rgba8(&level1, w1, h1, false);
    assert_eq!((w2, h2), (1, 1));
    assert_eq!(level2, vec![101, 40, 42, 109]);
    // A 1 x 1 level is its own next level: every tap is the same texel.
    let (level3, w3, h3) = downsample_rgba8(&level2, w2, h2, false);
    assert_eq!((w3, h3), (1, 1));
    assert_eq!(level3, level2);
}

#[test]
fn short_source_and_zero_sides_return_empty_with_the_output_size() {
    let src = flat(&FOUR_BY_FOUR);
    assert_eq!(
        downsample_rgba8(&src[..63], 4, 4, false),
        (Vec::new(), 2, 2)
    );
    assert_eq!(downsample_rgba8(&src, 0, 4, true), (Vec::new(), 0, 2));
    assert_eq!(downsample_rgba8(&src, 4, 0, false), (Vec::new(), 2, 0));
    assert_eq!(downsample_rgba8(&[], 3, 5, true), (Vec::new(), 2, 3));
    // A single texel is returned unchanged.
    assert_eq!(
        downsample_rgba8(&[9, 8, 7, 6], 1, 1, false),
        (vec![9, 8, 7, 6], 1, 1)
    );
}
