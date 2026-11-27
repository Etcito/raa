//! Billboard bake-target layout (M6, contract m6-bake-targets-api): dimensions, tile
//! origins, gutters, refusals and atlas capacity planning, through the public layout API.

use raa_lit::{ATLAS_SIZE, MAX_TEXTURES, ShelfPacker, billboard_layout, billboard_tile_rect};

#[test]
fn extents_and_columns_follow_the_resolution() {
    for (res, columns) in [(64u32, 8u32), (128, 8), (256, 8), (512, 4)] {
        let layout = billboard_layout(res).unwrap();
        let pitch = res + 2;
        assert_eq!(layout.resolution, res);
        assert_eq!(layout.extent, [columns * pitch, (8 / columns) * pitch]);
    }
    assert_eq!(billboard_layout(64).unwrap().extent, [528, 66]);
    assert_eq!(billboard_layout(512).unwrap().extent, [2056, 1028]);
}

#[test]
fn tile_origins_run_across_a_row_then_the_next() {
    let l = billboard_layout(128).unwrap();
    assert_eq!(l.tiles[0], [1, 1, 128, 128]);
    assert_eq!(l.tiles[1], [131, 1, 128, 128]);
    assert_eq!(l.tiles[7], [7 * 130 + 1, 1, 128, 128]);
    let l = billboard_layout(512).unwrap();
    assert_eq!(l.tiles[3], [3 * 514 + 1, 1, 512, 512]);
    assert_eq!(l.tiles[4], [1, 515, 512, 512]);
    assert_eq!(l.tiles[7], [3 * 514 + 1, 515, 512, 512]);
    for res in [64, 128, 256, 512] {
        let l = billboard_layout(res).unwrap();
        for k in 0..8u8 {
            assert_eq!(billboard_tile_rect(res, k).unwrap(), l.tiles[k as usize]);
        }
    }
}

#[test]
fn every_tile_keeps_a_one_texel_gutter_inside_the_extent() {
    for res in [64, 128, 256, 512] {
        let l = billboard_layout(res).unwrap();
        let padded: Vec<[u32; 4]> = l
            .tiles
            .iter()
            .map(|t| [t[0] - 1, t[1] - 1, t[0] + t[2] + 1, t[1] + t[3] + 1])
            .collect();
        for (i, a) in padded.iter().enumerate() {
            assert!(a[0] < a[2] && a[2] <= l.extent[0] && a[3] <= l.extent[1]);
            for b in &padded[i + 1..] {
                let overlap = a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3];
                assert!(!overlap, "padded tiles {a:?} and {b:?} overlap at {res}");
            }
        }
        let area: u32 = padded.iter().map(|p| (p[2] - p[0]) * (p[3] - p[1])).sum();
        assert_eq!(
            area,
            l.extent[0] * l.extent[1],
            "gutters tile the extent at {res}"
        );
    }
}

#[test]
fn invalid_resolution_and_direction_are_refused() {
    for res in [0, 32, 100, 1024] {
        assert_eq!(
            billboard_layout(res).unwrap_err().code(),
            "billboard.invalid"
        );
        assert_eq!(
            billboard_tile_rect(res, 0).unwrap_err().code(),
            "billboard.invalid"
        );
    }
    assert_eq!(
        billboard_tile_rect(64, 8).unwrap_err().code(),
        "billboard.invalid"
    );
    assert_eq!(
        billboard_tile_rect(64, 255).unwrap_err().code(),
        "billboard.invalid"
    );
}

#[test]
fn atlas_capacity_fails_and_partial_reservations_are_freed() {
    // Three images of the 512 extent (2056 x 1028, 2064 x 1040 aligned) per publication:
    // one per shelf row in the 4096 atlas, so three rows hold three images, a fourth fails.
    let extent = billboard_layout(512).unwrap().extent;
    let mut packer = ShelfPacker::new(ATLAS_SIZE, MAX_TEXTURES);
    let first: Vec<u32> = (0..3)
        .map(|_| packer.alloc(extent[0], extent[1]).unwrap().0)
        .collect();
    let mut second = Vec::new();
    let mut failure = None;
    for _ in 0..3 {
        match packer.alloc(extent[0], extent[1]) {
            Ok((id, _)) => second.push(id),
            Err(e) => {
                failure = Some(e);
                break;
            }
        }
    }
    assert_eq!(
        failure.expect("the second publication does not fit").code(),
        "texture.atlas_full"
    );
    for id in second {
        assert!(packer.free(id));
    }
    assert_eq!(
        packer.len(),
        first.len(),
        "only the first publication stays"
    );
    for id in first {
        assert!(packer.free(id));
    }
    assert!(packer.is_empty());
}
