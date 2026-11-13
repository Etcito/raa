//! The S32 `PassView` layout, the view-slot constants and the plane extraction and builders
//! shared with the cull and shadow paths. The GPU test only allocates the slot buffer; no pass
//! is recorded, so it reads no depth or colour input.

#![cfg(not(target_arch = "wasm32"))]

use raa_core::testing::{skip, software_gpu};
use raa_math::{Mat4, Vec3, Vec4};
use raa_vg::{
    CASCADE_BASE, GI_BASE, GI_SLOTS, InstanceFilter, MAIN_FIRST, MAIN_SECOND, PASS_VIEW_SIZE,
    PassView, SLOT_STRIDE, VG_VIEW_SLOTS, ViewSlots,
};

fn f32_at(bytes: &[u8], at: usize) -> f32 {
    f32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
}

fn floats_at(bytes: &[u8], at: usize, n: usize) -> Vec<f32> {
    (0..n).map(|i| f32_at(bytes, at + i * 4)).collect()
}

fn matrix(base: f32) -> Mat4 {
    let mut cols = [[0.0; 4]; 4];
    for (c, col) in cols.iter_mut().enumerate() {
        for (r, value) in col.iter_mut().enumerate() {
            *value = base + (c * 4 + r) as f32;
        }
    }
    Mat4 { cols }
}

fn sample() -> PassView {
    let index = |i: usize| 33.0 + (i * 4) as f32;
    let planes = std::array::from_fn(|i| {
        let b = index(i);
        Vec4::new(b, b + 1.0, b + 2.0, b + 3.0)
    });
    let receiver = std::array::from_fn(|i| {
        let b = 125.0 + (i * 4) as f32;
        Vec4::new(b, b + 1.0, b + 2.0, b + 3.0)
    });
    PassView {
        raster_view_proj: matrix(1.0),
        occlusion_view: matrix(17.0),
        planes,
        camera_position: Vec3::new(101.0, 102.0, 103.0),
        lod_threshold: 104.0,
        viewport: [105.0, 106.0],
        proj_scale: [107.0, 108.0],
        hzb_size: [109.0, 110.0],
        near: 111.0,
        lod_scale: 112.0,
        is_ortho: 113,
        phase: 114,
        occlusion_enabled: 115,
        allow_software: 116,
        eye_position: Vec3::new(117.0, 118.0, 119.0),
        eye_lod_scale: 120.0,
        eye_near: 121.0,
        receiver_plane_count: 122,
        instance_filter: 123,
        stamp: 124,
        receiver_planes: receiver,
        eye_light: Vec3::new(149.0, 150.0, 151.0),
    }
}

#[test]
fn write_le_bytes_puts_every_field_at_its_s32_offset() {
    let view = sample();
    let mut bytes = [0u8; PASS_VIEW_SIZE];
    view.write_le_bytes(&mut bytes);
    assert_eq!(PASS_VIEW_SIZE, 432);
    assert!(PASS_VIEW_SIZE <= SLOT_STRIDE as usize);

    let cols = |m: &Mat4| m.cols.iter().flatten().copied().collect::<Vec<_>>();
    assert_eq!(floats_at(&bytes, 0, 16), cols(&view.raster_view_proj));
    assert_eq!(floats_at(&bytes, 64, 16), cols(&view.occlusion_view));
    let planes = view
        .planes
        .iter()
        .flat_map(|p| p.to_array())
        .collect::<Vec<_>>();
    assert_eq!(floats_at(&bytes, 128, 24), planes);
    assert_eq!(floats_at(&bytes, 224, 3), vec![101.0, 102.0, 103.0]);
    assert_eq!(f32_at(&bytes, 236), 104.0);
    assert_eq!(floats_at(&bytes, 240, 2), vec![105.0, 106.0]);
    assert_eq!(floats_at(&bytes, 248, 2), vec![107.0, 108.0]);
    assert_eq!(floats_at(&bytes, 256, 2), vec![109.0, 110.0]);
    assert_eq!(f32_at(&bytes, 264), 111.0);
    assert_eq!(f32_at(&bytes, 268), 112.0);
    assert_eq!(u32_at(&bytes, 272), 113);
    assert_eq!(u32_at(&bytes, 276), 114);
    assert_eq!(u32_at(&bytes, 280), 115);
    assert_eq!(u32_at(&bytes, 284), 116);
    assert_eq!(floats_at(&bytes, 288, 3), vec![117.0, 118.0, 119.0]);
    assert_eq!(f32_at(&bytes, 300), 120.0);
    assert_eq!(f32_at(&bytes, 304), 121.0);
    assert_eq!(u32_at(&bytes, 308), 122);
    assert_eq!(u32_at(&bytes, 312), 123);
    assert_eq!(u32_at(&bytes, 316), 124);
    let receiver = view
        .receiver_planes
        .iter()
        .flat_map(|p| p.to_array())
        .collect::<Vec<_>>();
    assert_eq!(floats_at(&bytes, 320, 24), receiver);
    assert_eq!(floats_at(&bytes, 416, 3), vec![149.0, 150.0, 151.0]);
    assert_eq!(floats_at(&bytes, 428, 1), vec![0.0]);

    let mut short = [0u8; 228];
    view.write_le_bytes(&mut short);
    assert_eq!(f32_at(&short, 224), 101.0);
    view.write_le_bytes(&mut [0u8; 3]);
}

#[test]
fn view_slot_constants_match_the_s32_table() {
    assert_eq!(
        (
            VG_VIEW_SLOTS,
            MAIN_FIRST,
            MAIN_SECOND,
            CASCADE_BASE,
            GI_BASE,
            GI_SLOTS,
            SLOT_STRIDE
        ),
        (11, 0, 1, 2, 6, 5, 512)
    );
    assert_eq!(ViewSlots::offset(0), 0);
    assert_eq!(ViewSlots::offset(5), 2560);
}

#[test]
fn view_slots_allocate_one_stride_per_slot_and_ignore_out_of_range_writes() {
    let test = "view_slots_allocate_one_stride_per_slot_and_ignore_out_of_range_writes";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let slots = ViewSlots::new(&gpu);
    assert_eq!(
        slots.allocated_bytes(),
        u64::from(VG_VIEW_SLOTS * SLOT_STRIDE)
    );
    assert_eq!(slots.buffer().size(), slots.allocated_bytes());
    slots.write(&gpu, MAIN_FIRST, &sample());
    slots.write(&gpu, VG_VIEW_SLOTS, &sample());
}

fn inside(planes: &[Vec4; 6], p: Vec3) -> bool {
    planes
        .iter()
        .all(|plane| plane.x * p.x + plane.y * p.y + plane.z * p.z + plane.w >= -1.0e-4)
}

#[test]
fn perspective_planes_accept_in_front_and_reject_outside_points() {
    let eye = Vec3::new(0.0, 0.0, 5.0);
    let view_proj = Mat4::perspective_rh_zo(60f32.to_radians(), 16.0 / 9.0, 0.1, 100.0)
        * Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
    let planes = PassView::frustum_planes(view_proj);
    assert!(inside(&planes, Vec3::ZERO), "the target is in front");
    assert!(!inside(&planes, Vec3::new(0.0, 0.0, 10.0)), "behind");
    assert!(!inside(&planes, Vec3::new(0.0, 0.0, -200.0)), "beyond far");
    assert!(!inside(&planes, Vec3::new(1000.0, 0.0, 0.0)), "far left");
}

#[test]
fn orthographic_planes_accept_the_centre_and_reject_each_side() {
    let camera = raa_scene::Camera {
        eye: Vec3::new(0.0, 0.0, 8.0),
        target: Vec3::ZERO,
        up: Vec3::Y,
        near: 0.0,
        far: 16.0,
        aspect: 1.0,
        ortho_height: Some(16.0),
        ..raa_scene::Camera::default()
    };
    let planes = PassView::frustum_planes(camera.view_proj());
    assert!(inside(&planes, Vec3::ZERO));
    for p in [
        Vec3::new(-9.0, 0.0, 0.0),
        Vec3::new(9.0, 0.0, 0.0),
        Vec3::new(0.0, -9.0, 0.0),
        Vec3::new(0.0, 9.0, 0.0),
        Vec3::new(0.0, 0.0, 9.0),
        Vec3::new(0.0, 0.0, -9.0),
    ] {
        assert!(!inside(&planes, p), "{p:?} is outside");
    }
}

#[test]
fn shadowing_keeps_only_light_facing_planes_and_drawing_sets_the_filter() {
    let view = PassView::perspective(
        Mat4::IDENTITY,
        Mat4::IDENTITY,
        Vec3::ZERO,
        [1.0, 1.0],
        [1.0, 1.0],
        0.1,
        1.0,
    );
    let planes = [
        Vec4::new(1.0, 0.0, 0.0, 0.0),
        Vec4::new(-1.0, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0, 0.0, 0.0),
        Vec4::new(0.0, -1.0, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        Vec4::new(0.0, 0.0, -1.0, 0.0),
    ];
    let shadow = view.shadowing(&planes, Vec3::new(1.0, 1.0, 1.0));
    assert_eq!(shadow.receiver_plane_count, 3);
    assert_eq!(
        &shadow.receiver_planes[..3],
        &[planes[0], planes[2], planes[4]]
    );
    assert_eq!(shadow.receiver_planes[3], Vec4::new(0.0, 0.0, 0.0, 0.0));
    assert_eq!(view.drawing(InstanceFilter::Gi).instance_filter, 3);
}
