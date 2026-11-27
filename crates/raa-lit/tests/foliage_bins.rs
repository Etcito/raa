#[allow(dead_code)]
#[path = "../src/foliage_bins.rs"]
mod foliage_bins;
#[allow(dead_code)]
#[allow(clippy::wrong_self_convention)]
#[path = "../src/foliage_resources.rs"]
mod foliage_resources;

use foliage_bins::{dispatch_size, needs_bin_retry, rectangle_tile_bounds, tiles_per_axis};

// Closed-form pixel rectangles specify the conservative boxes the shader fills.
fn rectangle_tiles(resolution: u32, min: [u32; 2], max: [u32; 2]) -> Vec<[u32; 2]> {
    let (lo, hi) =
        rectangle_tile_bounds(resolution, min.map(|v| v as f32), max.map(|v| v as f32)).unwrap();
    let mut tiles = Vec::new();
    for y in lo[1]..=hi[1] {
        for x in lo[0]..=hi[0] {
            tiles.push([x, y]);
        }
    }
    tiles
}

#[test]
fn dispatch_and_shader_validate() {
    assert_eq!([0, 1, 1, 2, 2], [0, 1, 64, 65, 128].map(dispatch_size));
    raa_shader::Composer::validate(include_str!("../src/wgsl/foliage_bins.wgsl")).unwrap();
}

#[test]
fn rectangles_cover_enclosed_and_partial_edge_tiles() {
    assert_eq!(tiles_per_axis(33), 3);
    assert_eq!(
        rectangle_tiles(33, [31, 31], [33, 33]),
        vec![[1, 1], [2, 1], [1, 2], [2, 2]]
    );
    let enclosing = rectangle_tiles(48, [0, 0], [48, 48]);
    assert_eq!(enclosing.len(), 9);
    assert!(enclosing.contains(&[1, 1]));
    assert_eq!(rectangle_tiles(33, [32, 32], [33, 33]), vec![[2, 2]]);
}

#[test]
fn overflow_requires_retry_until_capacity_fits() {
    let counts = [0, 2, 5, 1];
    assert!(needs_bin_retry(&counts, 4));
    assert!(!needs_bin_retry(&counts, 5));
    assert!(!needs_bin_retry(&[0, 0, 0, 0], 4));
}
