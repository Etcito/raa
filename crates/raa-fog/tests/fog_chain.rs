//! Fog accepts a graph-bound HDR chain on consecutive frames.
#![cfg(not(target_arch = "wasm32"))]

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::testing::{block_on, skip, software_gpu};
use raa_core::{FrameGraph, OffscreenTarget, RaaError, wgpu};
use raa_fog::FogFeature;
use raa_lit::{LitOptions, LitRenderer};
use raa_pipeline::{GraphFrame, Pipeline, RecordCtx};

const SIZE: (u32, u32) = (32, 32);

struct BindFog {
    views: [wgpu::TextureView; 2],
}

impl GraphFrame for BindFog {
    fn before_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if pass == "fog" {
            // There is no mesh pass in this pipeline: initialize depth to far and the
            // bound HDR input to black before fog samples them. Fog reads no G-buffer.
            let _clear = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fog test inputs"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.views[0],
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
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
            ctx.targets.hdr().bind(&self.views[0], &self.views[1]);
        }
        Ok(())
    }

    fn after_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if pass == "fog" {
            ctx.targets.hdr().unbind();
        }
        Ok(())
    }
}

#[test]
fn fog_records_with_bound_hdr_chain() {
    let Some(gpu) = software_gpu() else {
        skip(
            "fog_records_with_bound_hdr_chain",
            "software_gpu returned None",
        );
        return;
    };
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let format = wgpu::TextureFormat::Rgba8Unorm;
    // Keep the lit renderer's mesh passes on a separate pipeline so this pipeline
    // contains only fog; the renderer supplies its frame bind group and layout.
    let lit_pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, format, false)));
    let mut lit = LitRenderer::new(
        &gpu,
        &lit_pipeline,
        LitOptions {
            max_instances: 1,
            max_materials: 1,
            msaa: 1,
        },
    )
    .expect("lit renderer");
    let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, format, false)));
    FogFeature::install(&gpu, &pipeline, &mut lit).expect("fog install");
    let texture = || {
        gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("fog test HDR"),
            size: wgpu::Extent3d {
                width: SIZE.0,
                height: SIZE.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    };
    let textures = [texture(), texture()];
    let views = textures
        .each_ref()
        .map(|t| t.create_view(&Default::default()));
    pipeline
        .borrow_mut()
        .add_graph_frame("bind fog", Box::new(BindFog { views }))
        .expect("graph frame");
    let mut graph = FrameGraph::new(format);
    Pipeline::install(&pipeline, &mut graph);
    graph.compile().expect("compile");
    let output = OffscreenTarget::new(&gpu, SIZE.0, SIZE.1, format);
    for _ in 0..2 {
        let commands = graph.execute(&gpu, output.view(), SIZE).expect("frame");
        gpu.queue().submit([commands]);
    }
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");
    let error = block_on(errors.pop());
    assert!(error.is_none(), "GPU validation error: {error:?}");
}
