//! The per-vertex colour streams of the virtualized-geometry resolve: packing, the pool texels and
//! the refusals of `VgScene::set_cluster_colors`. CPU only.

use std::collections::BTreeMap;

use raa_vg::resolve_module_wgsl;
use raa_vg::scene::{VG_COLOR_WIDTH, VG_WHITE, VgScene, pack_vertex_colors, pool_color_texels};

#[test]
fn pack_vertex_colors_packs_rgba8_and_refuses_bad_streams() {
    let packed = pack_vertex_colors(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 1.0]).expect("rgba");
    assert_eq!(packed, vec![0xFF00_00FF, 0xFF00_8000]);
    assert!(
        pack_vertex_colors(&[1.0, 0.0, 0.0]).is_err(),
        "rgb without alpha"
    );
    assert!(
        pack_vertex_colors(&[f32::NAN, 0.0, 0.0, 1.0]).is_err(),
        "non-finite"
    );
}

#[test]
fn pool_texels_place_each_stream_at_its_section_base_and_keep_others_white() {
    let sections = BTreeMap::from([((7, 0), (0, 2)), ((7, 1), (2, 3)), ((9, 0), (5, 1))]);
    let red = pack_vertex_colors(&[1.0, 0.0, 0.0, 1.0]).unwrap()[0];
    let green = pack_vertex_colors(&[0.0, 1.0, 0.0, 1.0]).unwrap()[0];
    let colors = BTreeMap::from([
        ((7, 1), vec![red, green, red]),
        // A stream of the wrong length stays white.
        ((9, 0), vec![green, green]),
        // A section the pool does not hold is ignored.
        ((3, 0), vec![red]),
    ]);
    let texels = pool_color_texels(6, &sections, &colors);
    assert_eq!(texels, vec![VG_WHITE, VG_WHITE, red, green, red, VG_WHITE]);
}

#[test]
fn set_cluster_colors_refuses_bad_streams_and_removes() {
    let mut scene = VgScene::new();
    assert!(scene.set_cluster_colors(1, 0, Some(&[1.0, 0.0])).is_err());
    scene
        .set_cluster_colors(1, 0, Some(&[1.0, 0.0, 0.0, 1.0]))
        .expect("valid stream");
    scene.set_cluster_colors(1, 0, None).expect("removal");
    assert!(scene.colors().is_none(), "no texture before an upload");
}

#[test]
fn resolve_multiplies_by_the_vertex_colour_texture_of_the_same_width() {
    let source = resolve_module_wgsl();
    assert!(source.contains(&format!("const VG_COLOR_WIDTH: u32 = {VG_COLOR_WIDTH}u;")));
    assert!(
        source.contains(
            "encode_srgb(interp3(b, vertex_color(i0), vertex_color(i1), vertex_color(i2)))"
        )
    );
}
