//! IBL filter on a software GPU (SKIP without an adapter): a constant SKY_ENV cube is
//! filtered through `IblFeature::install` / `IblHandle::filter`, the device is polled and
//! `apply_pending` called until the SH9 readback lands; the coefficients evaluated at +Y and
//! -Y give the cube's colour (cosine-convolved / pi, so a constant c gives c).

#![cfg(not(target_arch = "wasm32"))]

// The CPU SH evaluation is crate-private; the test compiles its source file.
#[path = "../src/sh.rs"]
mod sh;

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::testing::{block_on, skip, software_gpu};
use raa_core::{GpuContext, wgpu};
use raa_ibl::IblFeature;
use raa_lit::{LitOptions, LitRenderer};
use raa_pipeline::Pipeline;

const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// SKY_ENV: 256 px faces, HDR_FORMAT, 6 mips.
const ENV_SIZE: u32 = 256;
const ENV_MIPS: u32 = 6;
const COLOUR: [f64; 3] = [0.4, 0.6, 0.8];
const TOLERANCE: f64 = 2e-2;
const MAX_POLLS: usize = 50;

/// A SKY_ENV cube with the filter's source usages (plus RENDER_ATTACHMENT for the clear).
fn env_cube(gpu: &GpuContext) -> wgpu::Texture {
    gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("gpu_ibl.env"),
        size: wgpu::Extent3d {
            width: ENV_SIZE,
            height: ENV_SIZE,
            depth_or_array_layers: 6,
        },
        mip_level_count: ENV_MIPS,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}

/// Clears mip 0 of every face to COLOUR (one render pass per face).
fn clear_faces(encoder: &mut wgpu::CommandEncoder, cube: &wgpu::Texture) {
    for face in 0..6 {
        let view = cube.create_view(&wgpu::TextureViewDescriptor {
            label: Some("gpu_ibl.face"),
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: face,
            array_layer_count: Some(1),
            ..Default::default()
        });
        let [r, g, b] = COLOUR;
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("gpu_ibl.clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a: 1.0 }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        drop(pass);
    }
}

#[test]
fn constant_cube_filters_to_its_colour() {
    let test = "constant_cube_filters_to_its_colour";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, SURFACE, false)));
    let mut lit = LitRenderer::new(&gpu, &pipeline, LitOptions::default()).expect("lit renderer");
    let handle = IblFeature::install(&gpu, &pipeline, &mut lit).expect("install");
    assert_eq!(
        handle.sh(),
        None,
        "the env stays cleared until the first SH"
    );
    let again = IblFeature::install(&gpu, &pipeline, &mut lit);
    assert_eq!(
        again.err().map(|e| e.code()),
        Some("shader.slot_conflict"),
        "a second install keeps the add_slot refusal code"
    );

    let cube = env_cube(&gpu);
    let mut encoder = gpu
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    clear_faces(&mut encoder, &cube);
    handle
        .filter(
            &gpu,
            &mut encoder,
            &cube,
            pipeline.borrow_mut().lighting_mut(),
        )
        .expect("filter");
    gpu.queue().submit([encoder.finish()]);

    let mut coefficients = None;
    for _ in 0..MAX_POLLS {
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
        let retry = handle.apply_pending(pipeline.borrow_mut().lighting_mut());
        assert!(!retry, "one filter never fills the staging ring");
        coefficients = handle.sh();
        if coefficients.is_some() {
            break;
        }
    }
    let error = block_on(errors.pop());
    assert!(error.is_none(), "GPU validation error: {error:?}");
    let coefficients = coefficients.expect("the SH9 readback lands within 50 polls");

    for (name, n) in [("+Y", [0.0, 1.0, 0.0]), ("-Y", [0.0, -1.0, 0.0])] {
        let got = sh::evaluate(&coefficients, n);
        for (c, (g, want)) in got.iter().zip(COLOUR).enumerate() {
            assert!(
                (g - want).abs() <= TOLERANCE,
                "{name} channel {c}: {g} vs {want} (sh {coefficients:?})"
            );
        }
    }

    // Leaked like the other GPU tests: dropping the device after creating Rgba16Float
    // textures crashed the GL backend of the WSL driver at process exit.
    std::mem::forget(handle);
    std::mem::forget(lit);
    std::mem::forget(pipeline);
    std::mem::forget(cube);
    std::mem::forget(gpu);
}
