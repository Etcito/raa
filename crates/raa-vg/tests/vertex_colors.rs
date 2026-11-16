//! The per-vertex attribute streams of the virtualized-geometry resolve: the colour and shading
//! streams, the pool texels and the refusals of `VgScene::set_cluster_colors` /
//! `VgScene::set_cluster_shading`. CPU only.

use std::collections::BTreeMap;

use raa_vg::resolve_module_wgsl;
use raa_vg::scene::{
    VG_COLOR_WIDTH, VG_NO_UV, VG_WHITE, VgScene, VgShading, pack_vertex_colors, pool_color_texels,
};

#[test]
fn colours_keep_full_f32_precision_and_refuse_bad_streams() {
    // A field-shaded part keeps its rest point in the colour: 2.5 and -0.125 must survive, which
    // the former rgba8 packing clamped to 1 and 0.
    let packed = pack_vertex_colors(&[2.5, -0.125, 0.3, 0.75]).expect("rgba");
    assert_eq!(packed, vec![[2.5, -0.125, 0.3, 0.75]]);
    assert!(
        pack_vertex_colors(&[1.0, 0.0, 0.0]).is_err(),
        "rgb without alpha"
    );
    assert!(
        pack_vertex_colors(&[f32::NAN, 0.0, 0.0, 1.0]).is_err(),
        "non-finite"
    );
    // White is 1.0f32 = 0x3F800000 in every channel.
    assert_eq!(VG_WHITE, [0x3F80_0000; 4]);
}

#[test]
fn pool_texels_place_colours_then_shading_uvs_at_each_section_base() {
    let sections = BTreeMap::from([((7, 0), (0, 2)), ((7, 1), (2, 2)), ((9, 0), (4, 1))]);
    let colors = BTreeMap::from([
        ((7, 1), vec![[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]),
        // A stream of the wrong length stays white.
        ((9, 0), vec![[0.0; 4], [0.0; 4]]),
        // A section the pool does not hold is ignored.
        ((3, 0), vec![[0.0; 4]]),
    ]);
    // Section (7, 0) has shading colours and uvs; the shading colour of (7, 1) replaces its
    // vertex colour, and it has no uv.
    let shading = BTreeMap::from([
        (
            (7, 0),
            VgShading {
                colors: Some(vec![[2.0, -3.0, 0.5, 0.25], [0.0, 0.0, 0.0, -1.0]]),
                uvs: Some(vec![[0.5, -0.5], [1.0, 0.0]]),
            },
        ),
        (
            (7, 1),
            VgShading {
                colors: Some(vec![[4.0, 4.0, 4.0, 4.0], [5.0, 5.0, 5.0, 5.0]]),
                uvs: None,
            },
        ),
    ]);
    let texels = pool_color_texels(5, &sections, &colors, &shading);
    let rgba = |c: [f32; 4]| c.map(f32::to_bits);
    assert_eq!(texels.len(), 10, "two texels per vertex");
    assert_eq!(texels[0], rgba([2.0, -3.0, 0.5, 0.25]));
    assert_eq!(texels[1], [0.5f32.to_bits(), (-0.5f32).to_bits(), 1, 0]);
    assert_eq!(texels[2], rgba([0.0, 0.0, 0.0, -1.0]));
    assert_eq!(texels[3], [1.0f32.to_bits(), 0, 1, 0]);
    assert_eq!(texels[4], rgba([4.0; 4]));
    assert_eq!(texels[5], VG_NO_UV);
    assert_eq!(texels[6], rgba([5.0; 4]));
    assert_eq!(texels[8], VG_WHITE);
    assert_eq!(texels[9], VG_NO_UV);
}

#[test]
fn set_cluster_streams_refuse_bad_lengths_and_remove() {
    let mut scene = VgScene::new();
    assert!(scene.set_cluster_colors(1, 0, Some(&[1.0, 0.0])).is_err());
    scene
        .set_cluster_colors(1, 0, Some(&[1.0, 0.0, 0.0, 1.0]))
        .expect("valid stream");
    scene.set_cluster_colors(1, 0, None).expect("removal");
    assert!(scene.set_cluster_shading(1, 0, None, Some(&[1.0])).is_err());
    assert!(
        scene
            .set_cluster_shading(1, 0, Some(&[f32::INFINITY, 0.0, 0.0, 1.0]), None)
            .is_err()
    );
    scene
        .set_cluster_shading(1, 0, Some(&[1.0, 2.0, 3.0, 4.0]), Some(&[0.5, 0.5]))
        .expect("valid shading");
    scene
        .set_cluster_shading(1, 0, None, None)
        .expect("removal");
    assert!(scene.colors().is_none(), "no texture before an upload");
}

#[test]
fn resolve_reads_two_attribute_texels_per_vertex_of_the_same_width() {
    let source = resolve_module_wgsl();
    assert!(source.contains(&format!("const VG_COLOR_WIDTH: u32 = {VG_COLOR_WIDTH}u;")));
    assert!(source.contains("let flat = index * 2u + slot;"));
    assert!(source.contains("texture_storage_2d<rgba32float, write>"));
}
