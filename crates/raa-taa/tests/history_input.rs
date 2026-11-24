#![cfg(not(target_arch = "wasm32"))]

//! The `taa` pass reads a bound `history` input (contract `rg-taa-history-input`). The rig is
//! an empty pipeline with the TAA pass enabled and no lit renderer: the test graph frame
//! clears the depth to 1.0 (far), the motion and the HDR source to zero before the resolve
//! records, and zeroes the bound history texture when it creates it. No pixel is read back.

use std::{cell::RefCell, rc::Rc};

use raa_core::testing::{block_on, skip, software_gpu};
use raa_core::{FrameGraph, GpuContext, OffscreenTarget, RaaError, wgpu};
use raa_pipeline::{
    GraphFrame, HDR_FORMAT, ImageDesc, ImageDim, Pipeline, PublishedImage, RecordCtx,
};
use raa_taa::{TaaFeature, TaaHandle};

const SIZE: (u32, u32) = (64, 64);
const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// `FrameGpu` (S10) is 768 bytes; the pass binds it as its frame uniform.
const FRAME_BYTES: usize = 768;
/// Bytes per texel of `Rgba16Float`.
const HDR_TEXEL_BYTES: u32 = 8;

fn cleared(view: &wgpu::TextureView) -> Option<wgpu::RenderPassColorAttachment<'_>> {
    Some(wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            store: wgpu::StoreOp::Store,
        },
    })
}

/// Clears the inputs the taa pass reads before it records: depth to 1.0, motion and the HDR
/// source to zero. With `image`, binds it to the `taa` `history` input every frame.
struct BindHistory {
    image: Option<(wgpu::TextureView, ImageDesc)>,
}

impl GraphFrame for BindHistory {
    fn before_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if pass != "taa" {
            return Ok(());
        }
        let colors = [
            cleared(ctx.targets.hdr().src()),
            cleared(ctx.targets.motion()),
        ];
        drop(ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("taa history test clear"),
            color_attachments: &colors,
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
        }));
        if let Some((view, desc)) = &self.image {
            ctx.resources.bind_input(
                "taa",
                "history",
                PublishedImage {
                    view: view.clone(),
                    desc: *desc,
                },
            );
        }
        Ok(())
    }
}

/// A frame-size `Rgba16Float` history image (TEXTURE_BINDING | COPY_DST), zeroed by upload,
/// and its descriptor `ImageDesc::new(D2, HDR_FORMAT, w, h, 1, 1)`.
fn history_image(gpu: &GpuContext) -> (wgpu::TextureView, ImageDesc) {
    let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("taa history test input"),
        size: wgpu::Extent3d {
            width: SIZE.0,
            height: SIZE.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HDR_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let zeros = vec![0u8; (SIZE.0 * SIZE.1 * HDR_TEXEL_BYTES) as usize];
    gpu.queue().write_texture(
        texture.as_image_copy(),
        &zeros,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(SIZE.0 * HDR_TEXEL_BYTES),
            rows_per_image: Some(SIZE.1),
        },
        wgpu::Extent3d {
            width: SIZE.0,
            height: SIZE.1,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&Default::default());
    let desc = ImageDesc::new(ImageDim::D2, HDR_FORMAT, SIZE.0, SIZE.1, 1, 1);
    (view, desc)
}

/// An empty pipeline with the taa pass enabled, the zeroed frame uniform and the clearing
/// graph frame; `image` is the history input the graph frame binds each frame.
struct Rig {
    pipeline: Rc<RefCell<Pipeline>>,
    taa: TaaHandle,
    graph: FrameGraph,
    output: OffscreenTarget,
}

impl Rig {
    fn new(gpu: &GpuContext, image: Option<(wgpu::TextureView, ImageDesc)>) -> Rig {
        let pipeline = Rc::new(RefCell::new(Pipeline::new(gpu, SURFACE, false)));
        let frame = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("taa history test frame"),
            size: FRAME_BYTES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue().write_buffer(&frame, 0, &[0u8; FRAME_BYTES]);
        let taa = TaaFeature::install(gpu, &pipeline, &frame).expect("taa install");
        pipeline
            .borrow_mut()
            .set_enabled("taa", true)
            .expect("enable taa");
        pipeline
            .borrow_mut()
            .add_graph_frame("history", Box::new(BindHistory { image }))
            .expect("graph frame");
        let mut graph = FrameGraph::new(SURFACE);
        Pipeline::install(&pipeline, &mut graph);
        graph.compile().expect("compile");
        Rig {
            pipeline,
            taa,
            graph,
            output: OffscreenTarget::new(gpu, SIZE.0, SIZE.1, SURFACE),
        }
    }

    fn record(&mut self, gpu: &GpuContext) -> wgpu::CommandBuffer {
        self.graph
            .execute(gpu, self.output.view(), self.output.size())
            .expect("frame")
    }

    /// Records and submits one frame; the result is not read back.
    fn frame(&mut self, gpu: &GpuContext) {
        let commands = self.record(gpu);
        gpu.queue().submit([commands]);
        self.graph.notify_submitted();
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
    }
}

#[test]
fn bound_history_input_frames_count_and_reset_on_invalidate() {
    let test = "bound_history_input_frames_count_and_reset_on_invalidate";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let mut rig = Rig::new(&gpu, Some(history_image(&gpu)));

    // Three frames with a bound history input: each returns Ok and counts one frame.
    for _ in 0..3 {
        rig.frame(&gpu);
    }
    assert_eq!(rig.taa.accumulated_frames(), 3, "three bound frames count");
    assert!(
        rig.taa.history_valid(),
        "history is valid after three frames"
    );

    // invalidate ignores the bound history: the next frame restarts the count at 1.
    rig.taa.invalidate();
    rig.frame(&gpu);
    assert_eq!(
        rig.taa.accumulated_frames(),
        1,
        "the frame after invalidate restarts the count"
    );

    let error = block_on(errors.pop());
    assert!(error.is_none(), "GPU validation error: {error:?}");
}

#[test]
fn without_a_bound_history_input_the_pass_keeps_rendering() {
    let test = "without_a_bound_history_input_the_pass_keeps_rendering";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let mut rig = Rig::new(&gpu, None);

    for _ in 0..2 {
        rig.frame(&gpu);
    }
    assert_eq!(
        rig.taa.accumulated_frames(),
        2,
        "two own-history frames count"
    );
    rig.taa.invalidate();
    rig.frame(&gpu);
    assert_eq!(
        rig.taa.accumulated_frames(),
        1,
        "invalidate restarts the count"
    );
    assert!(
        rig.pipeline
            .borrow()
            .resources()
            .input("taa", "history")
            .is_none(),
        "no history input is bound in this rig"
    );

    let error = block_on(errors.pop());
    assert!(error.is_none(), "GPU validation error: {error:?}");
}

#[test]
fn discarded_frame_preserves_history_and_retry_advances_once() {
    if !cfg!(target_os = "windows") {
        return;
    }
    let gpu = block_on(GpuContext::headless(&raa_core::GpuOptions {
        allow_software: false,
        ..Default::default()
    }))
    .expect("Windows hardware GPU");
    let mut rig = Rig::new(&gpu, None);
    rig.frame(&gpu);
    let pending = rig.record(&gpu);
    let written = rig
        .pipeline
        .borrow()
        .resources()
        .get("taa", "history")
        .unwrap()
        .view;
    drop(pending);
    assert_eq!(rig.taa.accumulated_frames(), 1);
    rig.frame(&gpu);
    let retried = rig
        .pipeline
        .borrow()
        .resources()
        .get("taa", "history")
        .unwrap()
        .view;
    assert_eq!(written, retried, "retry writes the same uncommitted target");
    rig.graph.notify_submitted();
    assert_eq!(rig.taa.accumulated_frames(), 2);
}
