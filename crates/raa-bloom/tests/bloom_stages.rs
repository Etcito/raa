#![cfg(not(target_arch = "wasm32"))]

use std::{cell::RefCell, future::Future, rc::Rc, task::Context};

use raa_bloom::stages::{BloomStageKind, install_stage};
use raa_core::testing::{skip, software_gpu};
use raa_core::{FrameGraph, OffscreenTarget, RaaError, wgpu};
use raa_pipeline::{FeaturePass, Pipeline, RecordCtx};

const WIDTH: u32 = 32;
const ROW_BYTES: u32 = WIDTH * 8;

struct Setup;

impl FeaturePass for Setup {
    fn name(&self) -> &str {
        "test.setup"
    }
    fn order(&self) -> u32 {
        30
    }
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let hdr = ctx.targets.hdr();
        let pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("test.setup"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: hdr.dst(),
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 2.0,
                        g: 2.0,
                        b: 2.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        drop(pass);
        hdr.swap();
        Ok(())
    }
}

struct Probe(Rc<wgpu::Buffer>);

impl FeaturePass for Probe {
    fn name(&self) -> &str {
        "test.probe"
    }
    fn order(&self) -> u32 {
        150
    }
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        ctx.encoder.copy_texture_to_buffer(
            ctx.targets.hdr().src_texture().as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &self.0,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(ROW_BYTES),
                    rows_per_image: Some(WIDTH),
                },
            },
            wgpu::Extent3d {
                width: WIDTH,
                height: WIDTH,
                depth_or_array_layers: 1,
            },
        );
        Ok(())
    }
}

fn half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exp = i32::from((bits >> 10) & 0x1f);
    let frac = f32::from(bits & 0x3ff);
    match exp {
        0 => sign * frac * 2.0_f32.powi(-24),
        31 => f32::NAN,
        _ => sign * (1.0 + frac / 1024.0) * 2.0_f32.powi(exp - 15),
    }
}

fn poll<T>(future: impl Future<Output = T>) -> T {
    let mut future = std::pin::pin!(future);
    let mut context = Context::from_waker(std::task::Waker::noop());
    loop {
        if let std::task::Poll::Ready(value) = future.as_mut().poll(&mut context) {
            return value;
        }
        std::thread::yield_now();
    }
}

#[test]
fn bloom_levels_install_and_publish_without_graph_inputs() {
    let Some(gpu) = software_gpu() else {
        skip("bloom_stages", "no software adapter");
        return;
    };
    let pipeline = Rc::new(RefCell::new(Pipeline::new(
        &gpu,
        wgpu::TextureFormat::Rgba8Unorm,
        false,
    )));
    let down = install_stage(&gpu, &pipeline, BloomStageKind::Down, "d1").unwrap();
    let up = install_stage(&gpu, &pipeline, BloomStageKind::Up, "u1").unwrap();
    assert_eq!(down.entry(), "bloom.down#d1");
    assert_eq!(up.entry(), "bloom.up#u1");
    assert_eq!(
        install_stage(&gpu, &pipeline, BloomStageKind::Down, "d1")
            .err()
            .unwrap()
            .code(),
        "pipeline.duplicate_pass"
    );

    let mut graph = FrameGraph::new(wgpu::TextureFormat::Rgba8Unorm);
    Pipeline::install(&pipeline, &mut graph);
    graph.compile().unwrap();
    let target = OffscreenTarget::new(&gpu, 64, 64, wgpu::TextureFormat::Rgba8Unorm);
    gpu.queue()
        .submit([graph.execute(&gpu, target.view(), (64, 64)).unwrap()]);
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let published = down.published();
    assert_eq!(published.len(), 1);
    assert_eq!(published[0].0, "level");
    assert_eq!(
        (
            published[0].1.width,
            published[0].1.height,
            published[0].1.format
        ),
        (32, 32, wgpu::TextureFormat::Rgba16Float)
    );
    assert!(up.published().is_empty());
    assert!(!down.ran());
    assert!(!up.ran());

    assert_eq!(
        down.set_values(&[("karis".into(), 0.5)])
            .err()
            .unwrap()
            .code(),
        "post.invalid"
    );
    assert!(down.set_values(&[("karis".into(), 1.0)]).is_ok());
    assert_eq!(
        up.set_values(&[("strength".into(), -1.0)])
            .err()
            .unwrap()
            .code(),
        "post.invalid"
    );
    assert_eq!(
        down.set_values(&[("strength".into(), 0.5)])
            .err()
            .unwrap()
            .code(),
        "post.invalid"
    );
}

#[test]
fn composite_without_a_level_preserves_the_hdr_chain() {
    let Some(gpu) = software_gpu() else {
        skip(
            "composite_without_a_level_preserves_the_hdr_chain",
            "no software adapter",
        );
        return;
    };
    let pipeline = Rc::new(RefCell::new(Pipeline::new(
        &gpu,
        wgpu::TextureFormat::Rgba8Unorm,
        false,
    )));
    pipeline.borrow_mut().add(Box::new(Setup)).unwrap();
    let up = install_stage(&gpu, &pipeline, BloomStageKind::Up, "u").unwrap();
    up.set_composite(true);
    let buffer = Rc::new(gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("test.bloom.stage.readback"),
        size: u64::from(ROW_BYTES * WIDTH),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    pipeline
        .borrow_mut()
        .add(Box::new(Probe(buffer.clone())))
        .unwrap();
    let mut graph = FrameGraph::new(wgpu::TextureFormat::Rgba8Unorm);
    Pipeline::install(&pipeline, &mut graph);
    graph.compile().unwrap();
    let target = OffscreenTarget::new(&gpu, WIDTH, WIDTH, wgpu::TextureFormat::Rgba8Unorm);
    let error_scope = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let commands = graph.execute(&gpu, target.view(), (WIDTH, WIDTH)).unwrap();
    gpu.queue().submit([commands]);
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let mapped = slice.get_mapped_range().unwrap();
    for pixel in mapped.chunks_exact(8) {
        for channel in pixel.chunks_exact(2).take(3) {
            let value = half(u16::from_le_bytes([channel[0], channel[1]]));
            assert!((value - 2.0).abs() < 1e-2, "channel was {value}");
        }
        let alpha = half(u16::from_le_bytes([pixel[6], pixel[7]]));
        assert!((alpha - 1.0).abs() < 1e-2, "alpha was {alpha}");
    }
    drop(mapped);
    buffer.unmap();
    let error = poll(error_scope.pop());
    assert!(error.is_none(), "validation error: {error:?}");
    assert!(!up.ran());
}
