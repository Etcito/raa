//! GPU readback: a ClearPass into an OffscreenTarget reads back CLEAR_COLOR.
//! Skips with the SKIP line when no software adapter is available.

#![cfg(not(target_arch = "wasm32"))]

use raa_core::testing::{skip, software_gpu};
use raa_core::{BACKBUFFER, CLEAR_COLOR, ClearPass, FrameGraph, OffscreenTarget, non_empty};

const SIZE: u32 = 64;

fn unorm8(v: f64) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[test]
fn clear_pass_into_offscreen_target_reads_back_clear_color() {
    let Some(gpu) = software_gpu() else {
        skip(
            "clear_pass_into_offscreen_target_reads_back_clear_color",
            "software_gpu returned None; see the software_gpu line above",
        );
        return;
    };
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let target = OffscreenTarget::new(&gpu, SIZE, SIZE, format);

    let mut graph = FrameGraph::new(format);
    graph.add_pass(Box::new(ClearPass::new("clear", BACKBUFFER, CLEAR_COLOR)));
    graph.compile().expect("graph compiles");
    let commands = graph
        .execute(&gpu, target.view(), (SIZE, SIZE))
        .expect("graph executes");
    gpu.queue().submit([commands]);
    let pixels = target.read_rgba8(&gpu).expect("readback succeeds");

    assert_eq!(pixels.len(), (SIZE * SIZE * 4) as usize);
    let expected = [
        unorm8(CLEAR_COLOR.r),
        unorm8(CLEAR_COLOR.g),
        unorm8(CLEAR_COLOR.b),
        unorm8(CLEAR_COLOR.a),
    ];
    for (i, px) in pixels.chunks_exact(4).enumerate() {
        for c in 0..4 {
            assert!(
                px[c].abs_diff(expected[c]) <= 1,
                "pixel {i} is {px:?}, expected {expected:?} (±1); adapter {}",
                gpu.summary().device
            );
        }
    }
    assert!(!non_empty(&pixels, expected));
    eprintln!(
        "gpu_readback: adapter `{}` ({}, {}), pixel 0 = {:?}",
        gpu.summary().device,
        gpu.summary().backend,
        if gpu.summary().is_hardware() {
            "hardware"
        } else {
            "software"
        },
        &pixels[..4]
    );
    drop(graph);
    drop(target);
    // Tearing down a wgpu instance that enumerated Mesa's D3D12 GL adapter (WSL) segfaults
    // inside the driver, even without creating a device. Leak the context at the end of the
    // test so the process exits cleanly; all GPU work has completed by now.
    std::mem::forget(gpu);
}
