//! `GtaoFeature::install_instance` (pass `ao.gtao#<instance>`), `GtaoHandle::copy_camera_from`
//! and the bound depth/normal inputs of an instance pass.
//!
//! Software GPU only: prints `SKIP` when no software adapter exists, like the other GPU tests.
#![cfg(not(target_arch = "wasm32"))]

use std::{cell::RefCell, rc::Rc};

use raa_core::testing::{block_on, skip, software_gpu};
use raa_core::{FrameGraph, GpuContext, OffscreenTarget, RaaError, wgpu};
use raa_gtao::{AoDesc, GtaoFeature};
use raa_math::Mat4;
use raa_pipeline::{
    GraphFrame, ImageDesc, ImageDim, Pipeline, PrepareCtx, PublishedImage, RecordCtx, ResourceTable,
};

const SIZE: u32 = 32;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Clears the depth target to 1.0 (far; no mesh pass writes depth here) before the wide
/// instance records, and on the second frame binds a separate 32x32 Depth32Float view as its
/// `depth` input in `before_prepare`.
struct Binding {
    frame: u32,
    depth: wgpu::TextureView,
}

impl Binding {
    fn new(gpu: &GpuContext) -> Self {
        let depth = gpu
            .device()
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("test.gtao.depth_input"),
                size: wgpu::Extent3d {
                    width: SIZE,
                    height: SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        Self { frame: 0, depth }
    }
}

impl GraphFrame for Binding {
    fn before_prepare(&mut self, pass: &str, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        if pass == "ao.gtao#wide" && self.frame == 1 {
            let desc = ImageDesc::new(
                ImageDim::D2,
                wgpu::TextureFormat::Depth32Float,
                SIZE,
                SIZE,
                1,
                1,
            );
            ctx.resources.bind_input(
                pass,
                "depth",
                PublishedImage {
                    view: self.depth.clone(),
                    desc,
                },
            );
        }
        Ok(())
    }

    fn before_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if pass == "ao.gtao#wide" {
            let view = if self.frame == 1 {
                &self.depth
            } else {
                ctx.targets.depth()
            };
            drop(ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("test.gtao.depth_far"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            }));
        }
        Ok(())
    }

    fn end(&mut self, _table: &ResourceTable) {
        self.frame += 1;
    }
}

/// f16 bits of the first texel of `texture` (Rgba16Float, one row copied).
fn first_texel_bits(gpu: &GpuContext, texture: &wgpu::Texture) -> u16 {
    let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("test.gtao.readback"),
        size: u64::from(SIZE) * 8,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device().create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SIZE * 8),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: SIZE,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue().submit([encoder.finish()]);
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let mapped = slice.get_mapped_range().unwrap();
    let bits = u16::from_le_bytes([mapped[0], mapped[1]]);
    drop(mapped);
    buffer.unmap();
    bits
}

#[test]
fn instances_bound_inputs_and_lighting() {
    let name = "instances_bound_inputs_and_lighting";
    let Some(gpu) = software_gpu() else {
        skip(name, "no software adapter");
        return;
    };
    let scope = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, FORMAT, false)));

    // (1) The instance pass is listed at order 40, refuses a duplicate and can be reinstalled.
    GtaoFeature::install_instance(&gpu, &pipeline, "wide").unwrap();
    let listed = |pass: &str| {
        pipeline
            .borrow()
            .passes()
            .iter()
            .find(|p| p.name == pass)
            .map(|p| (p.order, p.enabled))
    };
    assert_eq!(listed("ao.gtao#wide"), Some((40, true)));
    assert_eq!(
        GtaoFeature::install_instance(&gpu, &pipeline, "wide")
            .err()
            .unwrap()
            .code(),
        "pipeline.duplicate_pass"
    );
    pipeline.borrow_mut().remove("ao.gtao#wide").unwrap();
    assert_eq!(listed("ao.gtao#wide"), None);
    let wide = GtaoFeature::install_instance(&gpu, &pipeline, "wide").unwrap();

    // (2) Two instances keep independent descriptors.
    let narrow = GtaoFeature::install_instance(&gpu, &pipeline, "narrow").unwrap();
    let desc = |radius_m: f32| AoDesc {
        radius_m,
        ..AoDesc::default()
    };
    wide.set(desc(0.3)).unwrap();
    narrow.set(desc(2.5)).unwrap();
    assert_eq!(wide.get().radius_m, 0.3);
    assert_eq!(narrow.get().radius_m, 2.5);

    // (3) `b` (wide) copies `a`'s (narrow) camera, so its pass dispatches.
    let proj = Mat4::perspective_rh_zo(1.0, 1.0, 0.1, 100.0);
    let inv_proj = proj.inverse().unwrap();
    narrow.set_camera(&proj, &inv_proj);
    wide.copy_camera_from(&narrow);

    // (4) Snapshot group 1 before the frames: an instance never sets AO.
    let before = pipeline.borrow().lighting().slot_view(7);

    pipeline
        .borrow_mut()
        .add_graph_frame("binding", Box::new(Binding::new(&gpu)))
        .unwrap();
    let output = OffscreenTarget::new(&gpu, SIZE, SIZE, FORMAT);
    for _ in 0..2 {
        let mut graph = FrameGraph::new(FORMAT);
        Pipeline::install(&pipeline, &mut graph);
        graph.compile().unwrap();
        gpu.queue()
            .submit([graph.execute(&gpu, output.view(), (SIZE, SIZE)).unwrap()]);
    }
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();

    // (3) The tap is published and the dispatch wrote AO 1 (far depth, f16 0x3c00); a texture
    // that was never written stays zero-initialised.
    let bits = {
        let published = pipeline
            .borrow()
            .resources()
            .get("ao.gtao#wide", "ao")
            .expect("ao.gtao#wide publishes `ao`");
        first_texel_bits(&gpu, published.view.texture())
    };
    assert_eq!(bits, 0x3c00, "ao.gtao#wide did not dispatch");

    // (4) The instance never replaced the group-1 AO slot.
    let after = pipeline.borrow().lighting().slot_view(7);
    assert_eq!(before, after);

    let error = block_on(scope.pop());
    assert!(error.is_none(), "validation error: {error:?}");
}
