use super::fixture::*;
#[test]
fn volume_layout_encloses_the_bounds_with_five_aligned_mips() {
    let layout = VolumeLayout::enclose([-4.0, -2.0, 0.5], [12.0, 6.0, 4.5]).expect("bounds");
    assert_eq!(layout.base, [VOLUME_MAX_AXIS, 512, 256]);
    assert!((layout.spacing - 16.0 / VOLUME_MAX_AXIS as f32).abs() < 1e-6);
    assert!(layout.base.iter().all(|axis| axis % VOLUME_AXIS_ALIGN == 0));
    let diagonal = layout.spacing
        * (0..3)
            .map(|a| (layout.base[a] * layout.base[a]) as f32)
            .sum::<f32>()
            .sqrt();
    assert!((layout.range - diagonal).abs() < 1e-3);
    for axis in 0..3 {
        let reach = layout.origin[axis] + layout.base[axis] as f32 * layout.spacing;
        assert!(reach >= [-4.0, -2.0, 0.5][axis] && reach >= [12.0, 6.0, 4.5][axis]);
    }
    for level in 0..VOLUME_MIPS {
        let mip = layout.mip(level);
        assert_eq!(mip.level, level);
        assert_eq!(
            mip.size,
            [
                layout.base[0] >> level,
                layout.base[1] >> level,
                layout.base[2] >> level
            ]
        );
        assert_eq!(mip.bytes_per_row % VOLUME_ROW_ALIGN, 0);
        assert!(mip.bytes_per_row >= mip.size[0]);
        assert_eq!(mip.pitch_words, mip.bytes_per_row / 4);
        assert_eq!(mip.rows_per_image, mip.size[1]);
        assert_eq!(
            mip.dispatch,
            [
                mip.size[0].div_ceil(32),
                mip.size[1].div_ceil(8),
                mip.size[2]
            ]
        );
        let uniform = layout.uniform(level);
        assert_eq!(uniform.len(), VOLUME_UNIFORM_BYTES as usize);
        assert_eq!(
            f32::from_le_bytes(uniform[12..16].try_into().unwrap()),
            layout.spacing
        );
        assert_eq!(
            u32::from_le_bytes(uniform[48..52].try_into().unwrap()),
            level
        );
        assert_eq!(
            u32::from_le_bytes(uniform[52..56].try_into().unwrap()),
            mip.pitch_words
        );
        assert_eq!(
            f32::from_le_bytes(uniform[32..36].try_into().unwrap()),
            layout.range
        );
    }
    assert!(layout.buffer_bytes() >= layout.mip(0).bytes_per_row as u64 * 512 * 256);
}

#[test]
fn volume_layout_refuses_empty_or_degenerate_bounds() {
    assert!(VolumeLayout::enclose([0.0; 3], [0.0; 3]).is_none());
    assert!(VolumeLayout::enclose([1.0, 0.0, 0.0], [0.0, 1.0, 1.0]).is_none());
    assert!(VolumeLayout::enclose([0.0; 3], [1.0, f32::NAN, 1.0]).is_none());
}

#[test]
fn volume_bounds_follow_accepted_instance_updates() {
    let mut scene = TraceScene::default();
    assert!(scene.volume_layout().is_none());
    scene
        .set_program(&Fixture::minimal().bytes())
        .expect("valid");
    let before = scene.volume_layout().expect("bounds");
    let mut moved = instance(0, u32::MAX);
    moved[3] = 10.0f32.to_bits();
    scene
        .update_instances(&instance_floats(moved))
        .expect("same count");
    let after = scene.volume_layout().expect("bounds");
    assert!(after.origin[0] > before.origin[0]);
    scene.clear();
    assert!(scene.volume_layout().is_none());
}
