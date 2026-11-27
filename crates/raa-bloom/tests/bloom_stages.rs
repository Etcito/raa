#![cfg(not(target_arch = "wasm32"))]

use std::{
    cell::{Cell, RefCell},
    future::Future,
    rc::Rc,
    task::Context,
};

use raa_bloom::stages::{BloomStageKind, install_stage};
use raa_core::testing::{skip, software_gpu};
use raa_core::{FrameGraph, OffscreenTarget, RaaError, wgpu};
use raa_pipeline::{
    FeaturePass, ImageDesc, ImageDim, Pipeline, PrepareCtx, PublishedImage, RecordCtx,
};

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
    for pixel in mapped.as_chunks::<8>().0 {
        for channel in pixel.as_chunks::<2>().0.iter().take(3) {
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

fn clear(encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, value: f64) {
    let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("test.bloom.feed"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: value,
                    g: value,
                    b: value,
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
}

/// Binds a `base` level (cleared to `value`) and a black `src` level to `bloom.up#u`, and
/// clears the HDR chain to black, so the composite at strength 1 shows the level alone.
struct Feed {
    value: Rc<Cell<f64>>,
    base: (wgpu::Texture, wgpu::TextureView),
    src: (wgpu::Texture, wgpu::TextureView),
}

fn level(gpu: &raa_core::GpuContext, size: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("test.bloom.level"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: raa_pipeline::HDR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

impl FeaturePass for Feed {
    fn name(&self) -> &str {
        "test.feed"
    }
    fn order(&self) -> u32 {
        30
    }
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        for (port, (texture, view)) in [("base", &self.base), ("src", &self.src)] {
            let desc = ImageDesc::new(
                ImageDim::D2,
                raa_pipeline::HDR_FORMAT,
                texture.width(),
                texture.height(),
                1,
                1,
            );
            ctx.resources.bind_input(
                "bloom.up#u",
                port,
                PublishedImage {
                    view: view.clone(),
                    desc,
                },
            );
        }
        Ok(())
    }
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        clear(ctx.encoder, &self.base.1, self.value.get());
        clear(ctx.encoder, &self.src.1, 0.0);
        let hdr = ctx.targets.hdr();
        clear(ctx.encoder, hdr.dst(), 0.0);
        hdr.swap();
        Ok(())
    }
}

/// `prepare` allocates a new level texture every frame: the composite must sample this
/// frame's level, not the first frame's (a bloom frozen at frame 1 made a two-level and a
/// five-level chain render the same image).
#[test]
fn composite_samples_the_current_frame_level() {
    let Some(gpu) = software_gpu() else {
        skip(
            "composite_samples_the_current_frame_level",
            "no software adapter",
        );
        return;
    };
    let pipeline = Rc::new(RefCell::new(Pipeline::new(
        &gpu,
        wgpu::TextureFormat::Rgba8Unorm,
        false,
    )));
    let value = Rc::new(Cell::new(1.0));
    pipeline
        .borrow_mut()
        .add(Box::new(Feed {
            value: value.clone(),
            base: level(&gpu, WIDTH / 2),
            src: level(&gpu, WIDTH / 4),
        }))
        .unwrap();
    let up = install_stage(&gpu, &pipeline, BloomStageKind::Up, "u").unwrap();
    up.set_values(&[("strength".into(), 1.0)]).unwrap();
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
    for frame_value in [1.0_f32, 3.0] {
        value.set(f64::from(frame_value));
        let commands = graph.execute(&gpu, target.view(), (WIDTH, WIDTH)).unwrap();
        gpu.queue().submit([commands]);
        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let mapped = slice.get_mapped_range().unwrap();
        for pixel in mapped.as_chunks::<8>().0 {
            for channel in pixel.as_chunks::<2>().0.iter().take(3) {
                let seen = half(u16::from_le_bytes([channel[0], channel[1]]));
                assert!(
                    (seen - frame_value).abs() < 1e-2,
                    "frame with level {frame_value}: composite showed {seen}"
                );
            }
        }
        drop(mapped);
        buffer.unmap();
        assert!(up.ran());
    }
}
