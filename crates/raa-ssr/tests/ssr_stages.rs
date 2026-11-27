#![cfg(not(target_arch = "wasm32"))]

use std::{cell::RefCell, rc::Rc};

use raa_core::testing::{skip, software_gpu};
use raa_core::{FrameGraph, OffscreenTarget, RaaError, wgpu};
use raa_pipeline::{FeaturePass, Pipeline, RecordCtx};
use raa_ssr::stages::{SsrStageKind, install_stage, stage_entry};

const SIZE: u32 = 32;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

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
        let depth = ctx
            .targets
            .depth_attachment_input(ctx.resources, self.name());
        let pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("test.setup"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: hdr.dst(),
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth,
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
        drop(pass);
        hdr.swap();
        Ok(())
    }
}

struct ConstantSetup;

impl FeaturePass for ConstantSetup {
    fn name(&self) -> &str {
        "test.constant_setup"
    }
    fn order(&self) -> u32 {
        30
    }
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let hdr = ctx.targets.hdr();
        let depth = ctx
            .targets
            .depth_attachment_input(ctx.resources, self.name());
        let pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("test.constant_setup"),
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
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth,
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
                    bytes_per_row: Some(256),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
        );
        Ok(())
    }
}

fn half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exp = i32::from((bits >> 10) & 31);
    let frac = f32::from(bits & 1023);
    if exp == 0 {
        sign * frac * 2f32.powi(-24)
    } else {
        sign * (1.0 + frac / 1024.0) * 2f32.powi(exp - 15)
    }
}

#[test]
fn hiz_and_trace_stage_contract() {
    let Some(gpu) = software_gpu() else {
        skip("hiz_and_trace_stage_contract", "no software adapter");
        return;
    };
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, FORMAT, false)));
    pipeline.borrow_mut().add(Box::new(Setup)).unwrap();
    let h = install_stage(&gpu, &pipeline, SsrStageKind::Hiz, "h").unwrap();
    let t = install_stage(&gpu, &pipeline, SsrStageKind::Trace, "t").unwrap();
    let entries = pipeline
        .borrow()
        .passes()
        .into_iter()
        .map(|p| p.name)
        .collect::<Vec<_>>();
    assert!(entries.contains(&stage_entry(SsrStageKind::Hiz, "h")));
    assert!(entries.contains(&stage_entry(SsrStageKind::Trace, "t")));
    assert_eq!(
        install_stage(&gpu, &pipeline, SsrStageKind::Hiz, "h")
            .err()
            .unwrap()
            .code(),
        "pipeline.duplicate_pass"
    );

    let output = OffscreenTarget::new(&gpu, SIZE, SIZE, FORMAT);
    let mut graph = FrameGraph::new(FORMAT);
    Pipeline::install(&pipeline, &mut graph);
    graph.compile().unwrap();
    gpu.queue()
        .submit([graph.execute(&gpu, output.view(), (SIZE, SIZE)).unwrap()]);
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();

    let hiz = h.published();
    assert_eq!(hiz.len(), 1);
    assert_eq!(hiz[0].0, "hiz");
    assert_eq!(hiz[0].1.format, wgpu::TextureFormat::R32Float);
    assert_eq!((hiz[0].1.width, hiz[0].1.height), (SIZE, SIZE));
    assert_eq!(hiz[0].1.mips, raa_ssr::hiz::mip_count(SIZE, SIZE));
    let trace = t.published();
    assert_eq!(trace.len(), 2);
    assert_eq!(trace[1].0, "hitMeta");
    assert_eq!(trace[1].1.format, wgpu::TextureFormat::Rgba32Float);
    assert_eq!(trace[0].0, "trace");
    assert_eq!(trace[0].1.format, wgpu::TextureFormat::Rgba16Float);
    assert_eq!((trace[0].1.width, trace[0].1.height), (SIZE, SIZE));
    assert!(h.ran());
    assert!(!t.ran());
    assert!(
        raa_core::testing::block_on(errors.pop()).is_none(),
        "GPU validation error"
    );

    assert_eq!(
        t.set_values(&[("maxRoughness".into(), 1.5)])
            .unwrap_err()
            .code(),
        "post.invalid"
    );
    assert_eq!(
        t.set_values(&[("maxSteps".into(), 32.5)])
            .unwrap_err()
            .code(),
        "post.invalid"
    );
    assert!(t.set_values(&[("maxSteps".into(), 32.0)]).is_ok());
    assert_eq!(
        h.set_values(&[("anything".into(), 1.0)])
            .unwrap_err()
            .code(),
        "post.invalid"
    );
}

#[test]
fn composite_without_trace_advances_hdr_unchanged() {
    let Some(gpu) = software_gpu() else {
        skip(
            "composite_without_trace_advances_hdr_unchanged",
            "no software adapter",
        );
        return;
    };
    let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, FORMAT, false)));
    let buffer = Rc::new(gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("test.composite_readback"),
        size: 256 * u64::from(SIZE),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    pipeline.borrow_mut().add(Box::new(ConstantSetup)).unwrap();
    let stage = install_stage(&gpu, &pipeline, SsrStageKind::Composite, "c").unwrap();
    assert!(
        pipeline
            .borrow()
            .passes()
            .iter()
            .any(|x| x.name == "ssr.composite#c")
    );
    assert!(stage.set_values(&[("intensity".into(), 0.5)]).is_ok());
    assert_eq!(
        stage
            .set_values(&[("intensity".into(), -1.0)])
            .unwrap_err()
            .code(),
        "post.invalid"
    );
    {
        let mut p = pipeline.borrow_mut();
        p.add(Box::new(Probe(buffer.clone()))).unwrap();
    }
    let target = OffscreenTarget::new(&gpu, SIZE, SIZE, FORMAT);
    let mut graph = FrameGraph::new(FORMAT);
    Pipeline::install(&pipeline, &mut graph);
    graph.compile().unwrap();
    gpu.queue()
        .submit([graph.execute(&gpu, target.view(), (SIZE, SIZE)).unwrap()]);
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let mapped = slice.get_mapped_range().unwrap();
    for (i, px) in mapped.as_chunks::<2>().0.iter().enumerate() {
        let value = half(u16::from_le_bytes([px[0], px[1]]));
        let expected = if i % 4 == 3 { 1.0 } else { 2.0 };
        assert!((value - expected).abs() < 1e-2, "channel {i}: {value}");
    }
}

// Use a controlled hit so identity is measured independently of the Hi-Z traversal.
// The HDR sample is red-only and the receiver has non-unit Fresnel.
struct SplitFixture(wgpu::TextureView);

impl FeaturePass for SplitFixture {
    fn name(&self) -> &str {
        "test.split_fixture"
    }

    fn order(&self) -> u32 {
        31
    }

    fn prepare(&mut self, ctx: &mut raa_pipeline::PrepareCtx<'_>) -> Result<(), RaaError> {
        let image = raa_pipeline::PublishedImage {
            view: self.0.clone(),
            desc: raa_pipeline::ImageDesc::new(
                raa_pipeline::ImageDim::D2,
                wgpu::TextureFormat::Rgba16Float,
                SIZE,
                SIZE,
                1,
                1,
            ),
        };
        for entry in ["ssr.composite#c", "ssr.incident#i"] {
            ctx.resources.bind_input(entry, "trace", image.clone());
        }
        Ok(())
    }

    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let hdr = ctx.targets.hdr();
        let clear = |r, g, b, a| wgpu::Color { r, g, b, a };
        let attachments = [
            (&self.0, clear(0.5, 0.5, 0.5, 0.0)),
            (ctx.targets.normal(), clear(0.0, 0.0, -1.0, 0.3)),
            (ctx.targets.material(), clear(0.737, 0.737, 0.737, 1.0)),
            (hdr.dst(), clear(2.0, 0.0, 0.0, 1.0)),
        ]
        .map(|(view, value)| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(value),
                    store: wgpu::StoreOp::Store,
                },
            })
        });
        drop(ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("test.split_fixture"),
            color_attachments: &attachments,
            ..Default::default()
        }));
        hdr.swap();
        ctx.resources.bind_input(
            "ssr.incident#i",
            "color",
            raa_pipeline::PublishedImage {
                view: hdr.src().clone(),
                desc: raa_pipeline::ImageDesc::new(
                    raa_pipeline::ImageDim::D2,
                    wgpu::TextureFormat::Rgba16Float,
                    SIZE,
                    SIZE,
                    1,
                    1,
                ),
            },
        );
        Ok(())
    }
}

struct BindIncident;

impl raa_pipeline::GraphFrame for BindIncident {
    fn before_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if pass == "ssr.replace#r" {
            let image = ctx.resources.get("ssr.incident#i", "incident").unwrap();
            ctx.resources.bind_input(pass, "incident", image);
        }
        Ok(())
    }
}

#[test]
fn split_reflection_matches_legacy_composite() {
    let Some(gpu) = software_gpu() else {
        skip("split_reflection_matches_legacy_composite", "no adapter");
        return;
    };
    let render = |split| {
        let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
        let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, FORMAT, false)));
        let buffer = Rc::new(gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("test.split_readback"),
            size: 256 * u64::from(SIZE),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        let trace = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("test.synthetic_hit"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        {
            let mut p = pipeline.borrow_mut();
            p.add(Box::new(Setup)).unwrap();
            p.add(Box::new(SplitFixture(
                trace.create_view(&Default::default()),
            )))
            .unwrap();
            p.add(Box::new(Probe(buffer.clone()))).unwrap();
        }
        let camera = raa_ssr::SsrFeature::install(&gpu, &pipeline).unwrap();
        camera.set_camera(&raa_math::Mat4::IDENTITY, &raa_math::Mat4::IDENTITY);
        pipeline.borrow_mut().set_enabled("ssr", false).unwrap();
        if split {
            let incident = install_stage(&gpu, &pipeline, SsrStageKind::Incident, "i").unwrap();
            let replace = install_stage(&gpu, &pipeline, SsrStageKind::Replace, "r").unwrap();
            replace.copy_camera_from(&camera);
            replace.set_values(&[("intensity".into(), 1.0)]).unwrap();
            pipeline
                .borrow_mut()
                .add_graph_frame("test.bind_incident", Box::new(BindIncident))
                .unwrap();
            assert!(!incident.ran());
        } else {
            let composite = install_stage(&gpu, &pipeline, SsrStageKind::Composite, "c").unwrap();
            composite.copy_camera_from(&camera);
            composite.set_values(&[("intensity".into(), 1.0)]).unwrap();
        }
        let target = OffscreenTarget::new(&gpu, SIZE, SIZE, FORMAT);
        let mut graph = FrameGraph::new(FORMAT);
        Pipeline::install(&pipeline, &mut graph);
        graph.compile().unwrap();
        gpu.queue()
            .submit([graph.execute(&gpu, target.view(), (SIZE, SIZE)).unwrap()]);
        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        assert!(raa_core::testing::block_on(errors.pop()).is_none());
        let mapped = slice.get_mapped_range().unwrap();
        mapped.to_vec()
    };
    let legacy = render(false);
    let split = render(true);
    if let Some(pixel) = split
        .chunks_exact(8)
        .zip(legacy.chunks_exact(8))
        .position(|(split, legacy)| split != legacy)
    {
        panic!(
            "screen source differs from legacy at ({}, {})",
            pixel % SIZE as usize,
            pixel / SIZE as usize
        );
    }
    let center = ((SIZE / 2) * 256 + (SIZE / 2) * 8) as usize;
    let red = half(u16::from_le_bytes([split[center], split[center + 1]]));
    assert!(
        (red - 2.5).abs() < 0.01,
        "red after one Fresnel response: {red}"
    );
}
