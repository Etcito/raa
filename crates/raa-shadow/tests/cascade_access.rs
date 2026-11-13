//! `cascade_frame` matches the splits and matrices the CSM pass computes (CPU), and a
//! `ShadowHandle` exposes its cascade targets only while `shadow.csm` is bound (software GPU,
//! SKIP without an adapter).

use raa_math::{Mat4, Vec3};
use raa_shadow::{cascade_frame, cascade_matrix, cascade_splits};

const NEAR: f32 = 0.1;
const FAR: f32 = 200.0;
const DISTANCE: f32 = 60.0;
const RESOLUTION: u32 = 2048;
const FOV_Y: f32 = 60.0 * std::f32::consts::PI / 180.0;
const ASPECT: f32 = 16.0 / 9.0;

fn camera() -> Mat4 {
    Mat4::look_at_rh(Vec3::new(0.0, 5.0, 10.0), Vec3::ZERO, Vec3::Y)
}

fn sun() -> Vec3 {
    Vec3::new(-0.4, -1.0, -0.3).normalize_or_zero()
}

#[test]
fn cascade_frame_matches_the_split_and_matrix_calls() {
    let view = camera();
    let (matrices, splits) =
        cascade_frame(&view, FOV_Y, ASPECT, NEAR, FAR, DISTANCE, sun(), RESOLUTION);
    let expected_splits = cascade_splits(NEAR, FAR.min(DISTANCE), 0.75, 4);
    assert_eq!(splits, expected_splits);
    let mut previous = NEAR;
    for (i, matrix) in matrices.iter().enumerate() {
        let expected = cascade_matrix(
            &view,
            FOV_Y,
            ASPECT,
            (previous, expected_splits[i]),
            sun(),
            RESOLUTION,
        );
        assert_eq!(*matrix, expected, "cascade {i}");
        previous = expected_splits[i];
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod gpu {
    use std::cell::RefCell;
    use std::rc::Rc;

    use raa_core::testing::{skip, software_gpu};
    use raa_core::{FrameGraph, OffscreenTarget, wgpu};
    use raa_lit::{LitOptions, LitRenderer};
    use raa_pipeline::Pipeline;
    use raa_shadow::{ShadowFeature, ShadowHandle};

    use super::{ASPECT, DISTANCE, FAR, FOV_Y, NEAR, RESOLUTION, camera, sun};

    const TEST: &str = "cascade_targets_need_a_bound_view";
    const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    const SIZE: (u32, u32) = (32, 32);

    #[test]
    fn cascade_targets_need_a_bound_view() {
        let Some(gpu) = software_gpu() else {
            skip(TEST, "software_gpu returned None");
            return;
        };
        let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, SURFACE, false)));
        let mut lit =
            LitRenderer::new(&gpu, &pipeline, LitOptions::default()).expect("lit renderer");
        let shadow: ShadowHandle =
            ShadowFeature::install(&gpu, &pipeline, &mut lit).expect("shadow feature");

        // No view was set: nothing prepared, so there is nothing to read.
        assert!(shadow.cascade_targets().is_none(), "no view set");

        shadow.set_view(&camera(), FOV_Y, ASPECT, NEAR, FAR, sun());
        assert!(
            shadow.cascade_targets().is_none(),
            "a view alone does not prepare the cascades"
        );

        // One empty frame prepares the pass (the lit passes draw nothing).
        let mut graph = FrameGraph::new(SURFACE);
        Pipeline::install(&pipeline, &mut graph);
        graph.compile().expect("compiles");
        let target = OffscreenTarget::new(&gpu, SIZE.0, SIZE.1, SURFACE);
        let commands = graph
            .execute(&gpu, target.view(), SIZE)
            .expect("frame runs");
        gpu.queue().submit([commands]);

        let prepared = shadow.cascade_targets().expect("bound after prepare");
        assert_eq!(prepared.count, 4);
        assert_eq!(prepared.resolution, RESOLUTION);
        assert_eq!(prepared.layer_views.len(), 4);
        assert_eq!(
            prepared.splits,
            raa_shadow::cascade_splits(NEAR, FAR.min(DISTANCE), 0.75, 4)
        );
        assert_eq!(prepared.sun_dir, sun());

        pipeline
            .borrow_mut()
            .set_enabled("shadow.csm", false)
            .expect("disabled");
        assert!(
            shadow.cascade_targets().is_none(),
            "a disabled pass drops its cascades"
        );
        std::mem::forget(graph);
        std::mem::forget(pipeline);
    }
}
