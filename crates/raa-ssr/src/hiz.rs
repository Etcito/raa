//! CPU twin of the Hi-Z depth pyramid that the GPU reduction builds from `depth`.
//!
//! The rule the WGSL reduction mirrors:
//! - Level 0 is the depth target, `width x height`, row-major `f32`.
//! - Level `i + 1` has size `max(1, w / 2) x max(1, h / 2)` of level `i`; the pyramid has
//!   [`mip_count`] levels and its last level is 1x1.
//! - Each destination texel is the minimum of its 2x2 source footprint.
//! - When a source dimension is odd, the last destination column (or row) also takes the
//!   extra source column (or row), so no depth sample is lost. A 5x3 level reduces to 2x1:
//!   destination (1, 0) covers source columns 2..=4 and rows 0..=2, a 3x3 footprint.

/// Number of pyramid levels from `width x height` down to 1x1: `floor(log2(max(w, h))) + 1`.
pub fn mip_count(width: u32, height: u32) -> u32 {
    width.max(height).max(1).ilog2() + 1
}

/// Size of the level after `width x height`: `max(1, w / 2) x max(1, h / 2)`.
pub fn next_size(width: u32, height: u32) -> (u32, u32) {
    ((width / 2).max(1), (height / 2).max(1))
}

/// Reduces `src` (row-major, `width x height`) by the rule above into `dst`, which is cleared
/// and refilled; returns the size of the new level. Reusing `dst` across levels keeps the
/// pyramid free of per-level allocations once its capacity has grown.
pub fn reduce_min(src: &[f32], width: u32, height: u32, dst: &mut Vec<f32>) -> (u32, u32) {
    assert!(width > 0 && height > 0, "hiz: level must be non-empty");
    assert_eq!(
        src.len(),
        width as usize * height as usize,
        "hiz: source length must be width x height"
    );
    let (dw, dh) = next_size(width, height);
    let stride = width as usize;
    dst.clear();
    dst.reserve(dw as usize * dh as usize);
    for y in 0..dh {
        let (y0, y1) = footprint(y, dh, height);
        for x in 0..dw {
            let (x0, x1) = footprint(x, dw, width);
            let mut m = f32::INFINITY;
            for row in y0..=y1 {
                let line = &src[row * stride..(row + 1) * stride];
                for &v in &line[x0..=x1] {
                    m = m.min(v);
                }
            }
            dst.push(m);
        }
    }
    (dw, dh)
}

/// Inclusive source range of destination index `i` (of `dst_len`) along a source of `src_len`.
fn footprint(i: u32, dst_len: u32, src_len: u32) -> (usize, usize) {
    let start = 2 * i;
    let end = if i + 1 == dst_len {
        src_len - 1
    } else {
        start + 1
    };
    (start as usize, end as usize)
}
