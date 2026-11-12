//! Serialization of the private cloud uniform and sky record (contract `P2-CLOUD-GPU`).

use raa_math::{Mat4, Vec3};
use raa_scene::Camera;

use super::super::super::CloudFrame;
use super::super::history::Accumulation;
use super::super::resources::{CloudGpu, cloud_sky_parameters};
use super::test_frame;

fn f32_at(bytes: &[u8; 192], at: usize) -> f32 {
    f32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"))
}

fn frame_at(eye: Vec3) -> CloudFrame {
    CloudFrame {
        camera: Camera {
            eye,
            ..Camera::default()
        },
        ..test_frame(0, 1, (640, 360))
    }
}

fn sample_cloud_gpu() -> CloudGpu {
    let mut next = 1.0_f32;
    let mut value = || {
        let v = next;
        next += 1.0;
        v
    };
    CloudGpu {
        wind_offset: [value(), value()],
        evolution: value(),
        coverage: value(),
        cloud_type: value(),
        base: value(),
        top: value(),
        density: value(),
        cirrus: value(),
        rain: value(),
        shadow_extent: value(),
        enabled: value(),
        shadow_origin: [value(), value()],
        wind_direction: [value(), value()],
        right: [value(), value(), value(), value()],
        up: [value(), value(), value(), value()],
        forward: [value(), value(), value(), value()],
        screen: [value(), value(), value(), value()],
        previous_view_proj: [
            [value(), value(), value(), value()],
            [value(), value(), value(), value()],
            [value(), value(), value(), value()],
            [value(), value(), value(), value()],
        ],
    }
}

#[test]
fn cloud_gpu_offsets_match_the_contract() {
    let bytes = sample_cloud_gpu().to_bytes();
    assert_eq!(bytes.len(), 192, "the record is 192 bytes");
    for i in 0..48 {
        assert_eq!(
            f32_at(&bytes, i * 4),
            (i + 1) as f32,
            "field word {i} is not at byte {}",
            i * 4
        );
    }
}

#[test]
fn shadow_origin_snaps_to_whole_texels() {
    let gpu = CloudGpu::from_frame(&frame_at(Vec3::new(100.0, 2.0, -50.0)), None, false);
    assert_eq!(gpu.shadow_origin, [-5906.25, -6093.75]);
}

#[test]
fn disabled_or_legacy_frames_serialize_enabled_zero() {
    let frame = test_frame(0, 1, (640, 360));
    let disabled = CloudFrame {
        weather_enabled: false,
        ..frame
    };
    let legacy = CloudFrame {
        physical: false,
        ..frame
    };
    assert_eq!(CloudGpu::from_frame(&disabled, None, false).enabled, 0.0);
    assert_eq!(CloudGpu::from_frame(&legacy, None, false).enabled, 0.0);
}

#[test]
fn zero_wind_uses_the_x_axis() {
    let gpu = CloudGpu::from_frame(&test_frame(0, 1, (640, 360)), None, false);
    assert_eq!(gpu.wind_direction, [1.0, 0.0]);
}

#[test]
fn history_none_serializes_identity_and_flag_zero() {
    let frame = test_frame(0, 1, (640, 360));
    let none = CloudGpu::from_frame(&frame, None, false);
    assert_eq!(none.previous_view_proj, Mat4::IDENTITY.cols);
    assert_eq!(none.forward[3], 0.0);

    let committed = Mat4::from_translation(Vec3::new(3.0, 4.0, 5.0));
    let accumulation = Accumulation { previous: committed, count: 1 };
    let some = CloudGpu::from_frame(&frame, Some(accumulation), false);
    assert_eq!(some.previous_view_proj, committed.cols);
    assert_eq!(some.forward[3], 1.0);
}

#[test]
fn forward_w_carries_the_accumulated_frame_count() {
    let frame = test_frame(0, 300, (640, 360));
    let accumulation = Accumulation { previous: Mat4::IDENTITY, count: 37 };
    let gpu = CloudGpu::from_frame(&frame, Some(accumulation), false);
    assert_eq!(gpu.forward[3], 37.0);
    assert_eq!(gpu.screen[2], 37.0, "an unchanged view walks the sequence by its count");
    let moving = Accumulation { previous: Mat4::IDENTITY, count: 1 };
    let gpu = CloudGpu::from_frame(&frame, Some(moving), false);
    assert_eq!(gpu.screen[2], 44.0, "a moving view keeps the invocation (300 & 255)");
}

#[test]
fn screen_lane_holds_scale_flags_jitter_and_samples() {
    let frame = CloudFrame { cloud_scale: 0.75, cloud_samples: 120, ..test_frame(0, 9, (1281, 721)) };
    assert_eq!(CloudGpu::from_frame(&frame, None, false).screen, [0.75, 0.0, 9.0, 120.0]);
    let animated = CloudFrame { animated: true, ..frame };
    assert_eq!(CloudGpu::from_frame(&animated, None, true).screen[1], 3.0);
    assert_eq!(super::super::resources::screen_size((1281, 721), 0.5), (641, 361));
    assert_eq!(super::super::resources::screen_size((1281, 721), 1.0), (1281, 721));
    assert_eq!(super::super::resources::screen_size((100, 3), 0.25), (25, 1));
}

#[test]
fn cloud_sky_parameters_convert_the_viewer_height_once() {
    let frame = CloudFrame {
        camera: Camera {
            eye: Vec3::new(0.0, 2000.0, 6.0),
            ..Camera::default()
        },
        ..test_frame(0, 1, (640, 360))
    };
    let sky = cloud_sky_parameters(&frame);
    assert_eq!(sky.viewer[1], 2000.0);
    assert_eq!(sky.viewer[3], 6362.0);
}

