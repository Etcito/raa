//! Multi-instance dof (contract rg-install-instance) and the bound HDR chain (contract
//! rg-hdr-chain-binding), native only. `DofFeature::install_instance` adds `dof#<instance>`
//! beside `dof` with the same order; with both passes installed, two frames run over a chain
//! bound to two test textures around each pass record. Without a software adapter each test
//! prints `SKIP`.
//!
//! Inputs the passes read are initialised here, since the test pipeline holds only the dof
//! passes: before each record the test clears the depth to 1.0 (far; no mesh pass writes
//! depth) and clears the bound source to a constant colour, and it sets an identity view with
//! a 1.0 rad perspective as the camera, so the CoC, gather and composite passes all run.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use raa_core::testing::{skip, software_gpu};
use raa_core::{FrameGraph, GpuContext, OffscreenTarget, RaaError, wgpu};
use raa_dof::DofFeature;
use raa_math::Mat4;
use raa_pipeline::{GraphFrame, HDR_FORMAT, Pipeline, RecordCtx};

/// Edge of the square frame and of the test textures, in pixels.
const SIZE: u32 = 16;

/// A surface-sized `HDR_FORMAT` texture usable as a bound chain texture (the wiring pool's
/// usages), with its default view.
fn chain_texture(gpu: &GpuContext, label: &str) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HDR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Binds the HDR chain to two test textures (`views[0]` the source, `views[1]` the
/// destination) around every enabled pass record, and counts the records it wraps.
struct BindChain {
    /// Kept alive beside the views they were created with.
    textures: [wgpu::Texture; 2],
    views: [wgpu::TextureView; 2],
    wrapped: Rc<Cell<u32>>,
}

impl GraphFrame for BindChain {
    fn before_record(&mut self, _pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        {
            let _clear = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("test.clear_chain"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.views[0],
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.5,
                            g: 0.25,
                            b: 0.1,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: ctx.targets.depth(),
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
        ctx.targets.hdr().bind(&self.views[0], &self.views[1]);
        assert!(ctx.targets.hdr().is_bound());
        assert!(ctx.targets.hdr().src_texture() == &self.textures[0]);
        self.wrapped.set(self.wrapped.get() + 1);
        Ok(())
    }

    fn after_record(&mut self, _pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        ctx.targets.hdr().unbind();
        Ok(())
    }
}

#[test]
fn install_instance_adds_an_enabled_pass_beside_dof_and_refuses_a_repeat() {
    let name = "install_instance_adds_an_enabled_pass_beside_dof_and_refuses_a_repeat";
    let Some(gpu) = software_gpu() else {
        skip(name, "no adapter");
        return;
    };
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, format, false)));
    DofFeature::install(&gpu, &pipeline).unwrap();
    DofFeature::install_instance(&gpu, &pipeline, "b").unwrap();

    let passes = pipeline.borrow().passes();
    let names: Vec<&str> = passes.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, ["dof", "dof#b"]);
    assert_eq!(passes[0].order, passes[1].order);
    assert!(passes[1].enabled);

    let Err(err) = DofFeature::install_instance(&gpu, &pipeline, "b") else {
        panic!("a repeated instance was accepted");
    };
    assert_eq!(err.code(), "pipeline.duplicate_pass");

    assert!(pipeline.borrow_mut().remove("dof#b").is_ok());
    let names: Vec<String> = pipeline
        .borrow()
        .passes()
        .into_iter()
        .map(|p| p.name)
        .collect();
    assert_eq!(names, ["dof"]);
}

#[test]
fn two_dof_passes_run_over_a_bound_chain_for_two_frames() {
    let name = "two_dof_passes_run_over_a_bound_chain_for_two_frames";
    let Some(gpu) = software_gpu() else {
        skip(name, "no adapter");
        return;
    };
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let output = OffscreenTarget::new(&gpu, SIZE, SIZE, format);
    let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, format, false)));
    let (a, a_view) = chain_texture(&gpu, "test.chain_a");
    let (b, b_view) = chain_texture(&gpu, "test.chain_b");
    let wrapped = Rc::new(Cell::new(0));
    let bind = BindChain {
        textures: [a, b],
        views: [a_view, b_view],
        wrapped: wrapped.clone(),
    };
    pipeline
        .borrow_mut()
        .add_graph_frame("test.bind_chain", Box::new(bind))
        .unwrap();
    let dof = DofFeature::install(&gpu, &pipeline).unwrap();
    let instance = DofFeature::install_instance(&gpu, &pipeline, "b").unwrap();

    let mut graph = FrameGraph::new(format);
    Pipeline::install(&pipeline, &mut graph);
    graph.compile().unwrap();
    let proj = Mat4::perspective_rh_zo(1.0, 1.0, 0.1, 100.0);
    for _ in 0..2 {
        dof.set_camera(&Mat4::IDENTITY, &proj);
        instance.set_camera(&Mat4::IDENTITY, &proj);
        let commands = graph.execute(&gpu, output.view(), (SIZE, SIZE)).unwrap();
        gpu.queue().submit([commands]);
    }
    // Two enabled passes in each of the two frames.
    assert_eq!(wrapped.get(), 4);
}
