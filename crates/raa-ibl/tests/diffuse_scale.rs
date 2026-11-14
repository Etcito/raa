//! IBL diffuse scale on a software GPU (SKIP without an adapter). A SKY_ENV cube cleared to a
//! constant colour is filtered through `IblFeature::install` / `IblHandle::filter`, and
//! `apply_pending` is polled until the first SH9 readback lands. The published `EnvSh.info`,
//! read through `IblHandle::published_info`, must equal the authored diffuse times the scale;
//! invalid scales are ignored, and scale 1 restores the authored value.
//!
//! Initialisation: the filter reads only the SKY_ENV cube, which this test clears itself (one
//! render pass per face, mip 0). No mesh pass writes depth and no G-buffer or HDR input is
//! read, so nothing else is initialised. The pipeline is the lit renderer's own.

#![cfg(not(target_arch = "wasm32"))]

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
const MAX_POLLS: usize = 50;
const EPSILON: f32 = 1e-6;

/// A SKY_ENV cube with the filter's source usages (plus RENDER_ATTACHMENT for the clear).
fn env_cube(gpu: &GpuContext) -> wgpu::Texture {
    gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("diffuse_scale.env"),
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
            label: Some("diffuse_scale.face"),
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: face,
            array_layer_count: Some(1),
            ..Default::default()
        });
        let [r, g, b] = COLOUR;
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("diffuse_scale.clear"),
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

fn near(got: f32, want: f32) -> bool {
    (got - want).abs() <= EPSILON
}

#[test]
fn published_info_carries_the_diffuse_scale() {
    let test = "published_info_carries_the_diffuse_scale";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, SURFACE, false)));
    let mut lit = LitRenderer::new(&gpu, &pipeline, LitOptions::default()).expect("lit renderer");
    let handle = IblFeature::install(&gpu, &pipeline, &mut lit).expect("install");
    assert_eq!(
        handle.published_info(),
        None,
        "nothing is published before the first SH"
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

    // The apply_pending that maps the first readback also publishes the environment.
    for _ in 0..MAX_POLLS {
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
        let retry = handle.apply_pending(pipeline.borrow_mut().lighting_mut());
        assert!(!retry, "one filter never fills the staging ring");
        if handle.sh().is_some() {
            break;
        }
    }
    let error = block_on(errors.pop());
    assert!(error.is_none(), "GPU validation error: {error:?}");
    let first = handle
        .published_info()
        .expect("the first publication lands within 50 polls");
    assert!(near(first[1], 1.0), "authored diffuse: {first:?}");
    assert!(near(handle.diffuse_scale(), 1.0), "default scale is 1");

    // Authored diffuse 0.8 with scale 0.25 publishes 0.2; the specular is unchanged.
    handle.set_intensity(0.8, 1.0);
    handle.set_diffuse_scale(0.25);
    handle.apply_pending(pipeline.borrow_mut().lighting_mut());
    let scaled = handle.published_info().expect("published");
    assert!(near(scaled[1], 0.2), "diffuse x scale: {scaled:?}");
    assert!(near(scaled[0], 1.0), "specular: {scaled:?}");

    // Out-of-range and non-finite scales are ignored: the stored scale and the published
    // info stay as they were.
    for invalid in [1.5, -0.1, f32::NAN] {
        handle.set_diffuse_scale(invalid);
        assert!(
            near(handle.diffuse_scale(), 0.25),
            "{invalid} is ignored, scale {}",
            handle.diffuse_scale()
        );
    }
    handle.apply_pending(pipeline.borrow_mut().lighting_mut());
    assert_eq!(
        handle.published_info(),
        Some(scaled),
        "invalid scales change nothing"
    );

    // Scale 1 restores the authored diffuse value.
    handle.set_diffuse_scale(1.0);
    handle.apply_pending(pipeline.borrow_mut().lighting_mut());
    let restored = handle.published_info().expect("published");
    assert!(
        near(restored[1], 0.8),
        "scale 1 restores the authored value: {restored:?}"
    );

    // Leaked like the other GPU tests: dropping the device after creating Rgba16Float
    // textures crashed the GL backend of the WSL driver at process exit.
    std::mem::forget(handle);
    std::mem::forget(lit);
    std::mem::forget(pipeline);
    std::mem::forget(cube);
    std::mem::forget(gpu);
}
