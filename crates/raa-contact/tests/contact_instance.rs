#![cfg(not(target_arch = "wasm32"))]

//! `ContactFeature::install_instance`: its own pass name, state and depth input.

use std::{cell::RefCell, rc::Rc};

use raa_contact::{ContactDesc, ContactFeature};
use raa_core::testing::{block_on, skip, software_gpu};
use raa_core::{FrameGraph, GpuContext, OffscreenTarget, RaaError, wgpu};
use raa_math::{Mat4, Vec3};
use raa_pipeline::{
    GraphFrame, ImageDesc, ImageDim, Pipeline, PrepareCtx, PublishedImage, RecordCtx,
};

const SIZE: u32 = 32;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

fn pipeline(gpu: &GpuContext) -> Rc<RefCell<Pipeline>> {
    Rc::new(RefCell::new(Pipeline::new(gpu, FORMAT, false)))
}

#[test]
fn named_passes_and_descriptors() {
    let Some(gpu) = software_gpu() else {
        skip("named_passes_and_descriptors", "no software adapter");
        return;
    };
    let pipeline = pipeline(&gpu);
    let a = ContactFeature::install_instance(&gpu, &pipeline, "a").unwrap();
    let b = ContactFeature::install_instance(&gpu, &pipeline, "b").unwrap();
    let passes = pipeline.borrow().passes();
    let b_info = passes
        .iter()
        .find(|p| p.name == "shadow.contact#b")
        .unwrap();
    let a_info = passes
        .iter()
        .find(|p| p.name == "shadow.contact#a")
        .unwrap();
    assert_eq!((b_info.order, b_info.enabled), (45, true));
    assert_eq!((a_info.order, a_info.enabled), (45, true));
    assert_eq!(
        ContactFeature::install_instance(&gpu, &pipeline, "b")
            .err()
            .unwrap()
            .code(),
        "pipeline.duplicate_pass"
    );
    let want = ContactDesc {
        distance_m: 2.0,
        thickness_m: 0.2,
        intensity: 0.5,
    };
    b.set(want).unwrap();
    assert_eq!(b.get(), want);
    assert_eq!(a.get(), ContactDesc::default());
    pipeline.borrow_mut().remove("shadow.contact#b").unwrap();
    let passes = pipeline.borrow().passes();
    assert!(!passes.iter().any(|p| p.name == "shadow.contact#b"));
    let _b = ContactFeature::install_instance(&gpu, &pipeline, "b").unwrap();
}

/// Binds a separate D2 Depth32Float view as `shadow.contact#b`'s `depth` input and clears
/// it to 1.0 (far) before that instance records; no mesh pass writes a depth buffer here.
struct Binding {
    depth: wgpu::TextureView,
}

impl Binding {
    fn new(gpu: &GpuContext) -> Self {
        let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("test.contact.depth"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self { depth }
    }
}

impl GraphFrame for Binding {
    fn before_prepare(&mut self, pass: &str, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        if pass == "shadow.contact#b" {
            let desc = ImageDesc::new(
                ImageDim::D2,
                wgpu::TextureFormat::Depth32Float,
                SIZE,
                SIZE,
                1,
                1,
            );
            let view = self.depth.clone();
            ctx.resources
                .bind_input(pass, "depth", PublishedImage { view, desc });
        }
        Ok(())
    }

    fn before_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if pass == "shadow.contact#b" {
            let _clear = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("test.contact.clear"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        Ok(())
    }
}

#[test]
fn camera_copy_and_depth_input() {
    let Some(gpu) = software_gpu() else {
        skip("camera_copy_and_depth_input", "no software adapter");
        return;
    };
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let pipeline = pipeline(&gpu);
    let src = ContactFeature::install_instance(&gpu, &pipeline, "src").unwrap();
    let b = ContactFeature::install_instance(&gpu, &pipeline, "b").unwrap();
    src.set_camera(
        &Mat4::IDENTITY,
        &Mat4::IDENTITY,
        &Mat4::IDENTITY,
        Vec3::new(0.0, -1.0, 0.0),
    );
    b.copy_camera_from(&src);
    pipeline
        .borrow_mut()
        .add_graph_frame("binding", Box::new(Binding::new(&gpu)))
        .unwrap();
    let before = pipeline.borrow().lighting().slot_view(8);
    let output = OffscreenTarget::new(&gpu, SIZE, SIZE, FORMAT);
    let mut graph = FrameGraph::new(FORMAT);
    Pipeline::install(&pipeline, &mut graph);
    graph.compile().unwrap();
    for _ in 0..2 {
        let command = graph.execute(&gpu, output.view(), (SIZE, SIZE)).unwrap();
        gpu.queue().submit([command]);
    }
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let after = pipeline.borrow().lighting().slot_view(8);
    assert_eq!(before, after, "the instance must not write group-1 slot 8");
    let p = pipeline.borrow();
    assert!(p.resources().get("shadow.contact#b", "contact").is_some());
    assert!(p.resources().get("shadow.contact#b", "output").is_some());
    assert!(block_on(errors.pop()).is_none(), "GPU validation error");
}
