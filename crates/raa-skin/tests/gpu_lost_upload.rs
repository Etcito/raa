//! A device the browser lost must refuse, not panic: the lobby rebuild uploaded the start-screen
//! penguin's `skin.rest` buffer with `create_buffer_init` after headless Chrome had destroyed the
//! GPUDevice, wgpu panicked, and the page's frame loop stopped (Runtime.evaluate then timed out).
//! Uploading through a plain buffer and a queue write keeps that path panic-free, and
//! `GpuContext::ensure_live` reports the loss as `core.gpu`.

#![cfg(not(target_arch = "wasm32"))]

use std::time::{Duration, Instant};

use raa_core::testing::{skip, software_gpu};
use raa_core::{GpuContext, wgpu};
use raa_skin::{SkinMesh, SkinVertexInputs};

const TEST: &str = "skin_upload_after_a_destroyed_device_does_not_panic";

fn fixture() -> SkinMesh {
    let rest = raa_mesh::from_js_arrays(
        &[2., 0., 1.].repeat(4),
        &[1., 0., 0.].repeat(4),
        Some(&[0.25, 0.75].repeat(4)),
        Some(&[0., 1., 0., -1.].repeat(4)),
        None,
        &[0, 1, 2, 1, 2, 3],
        &[],
    )
    .unwrap();
    SkinMesh::new(rest, &[0, 1, 2, 3].repeat(4), &[1., 0., 0., 0.].repeat(4)).unwrap()
}

fn wait_lost(gpu: &GpuContext) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !gpu.is_lost() && Instant::now() < deadline {
        let _ = gpu.device().poll(wgpu::PollType::Poll);
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn skin_upload_after_a_destroyed_device_does_not_panic() {
    let Some(gpu) = software_gpu() else {
        skip(
            TEST,
            "software_gpu returned None; see the software_gpu line above",
        );
        return;
    };
    let mesh = fixture();
    let before = SkinVertexInputs::new(&gpu, &mesh).expect("inputs before loss");
    assert!(gpu.ensure_live().is_ok());

    gpu.device().destroy();
    wait_lost(&gpu);
    assert!(gpu.is_lost(), "destroyed device was not reported lost");

    // The old path created a mapped-at-creation `skin.rest` buffer here and wgpu panicked on the
    // dead device; the upload now refuses with the same typed loss the engine reports.
    let error = SkinVertexInputs::new(&gpu, &mesh)
        .err()
        .expect("an upload on a destroyed device must refuse");
    assert_eq!(error.code(), "core.gpu");
    assert!(
        error.message().starts_with("device lost (destroyed)"),
        "{error}"
    );
    let engine = gpu
        .ensure_live()
        .expect_err("a destroyed device must refuse");
    assert_eq!(engine.code(), "core.gpu");
    eprintln!(
        "gpu_lost_upload: adapter `{}`, upload of {} vertices after destroy: {error}",
        gpu.summary().device,
        before.vertices
    );
    // Same teardown caveat as the other GPU tests: leaking the context avoids a driver
    // teardown segfault (WSL's D3D12 GL adapter) after all work has completed.
    std::mem::forget(gpu);
}
