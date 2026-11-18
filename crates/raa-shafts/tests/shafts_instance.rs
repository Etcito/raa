//! `ShaftsFeature::install_instance` (pass `sunshafts#<instance>`) and the bound HDR chain
//! of the `sunshafts` pass (contract rg-hdr-chain-binding).
//!
//! Software GPU only: each test prints `SKIP` when no software adapter exists.
#![cfg(not(target_arch = "wasm32"))]

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use raa_core::testing::{block_on, skip, software_gpu};
use raa_core::{FrameGraph, GpuContext, OffscreenTarget, RaaError, wgpu};
use raa_math::{Mat4, Vec3};
use raa_pipeline::{GraphFrame, Pipeline, RecordCtx, ResourceTable};
use raa_shafts::ShaftsFeature;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const POOL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// One Rgba16Float row of 32 px is exactly COPY_BYTES_PER_ROW_ALIGNMENT.
const ROW: u32 = 32 * 8;
const SIZE: u32 = 32;
const FRAMES: usize = 2;

fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((bits >> 10) & 0x1f);
    let frac = f32::from(bits & 0x3ff);
    match exp {
        0 => sign * frac * 2f32.powi(-24),
        _ => sign * (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
    }
}

fn pool_texture(gpu: &GpuContext, label: &str) -> wgpu::Texture {
    gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: POOL_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

/// Binds the chain to two pool textures around each `sunshafts` record. Before the pass:
/// clears depth to 1.0 (far; no mesh pass writes depth here) and clears the bound source to
/// `2 + frame`. After the pass: copies the bound destination to `readback`, then unbinds.
/// The pair alternates each frame, so frame 2 binds the textures in the opposite order.
struct PoolChain {
    textures: [wgpu::Texture; 2],
    views: [wgpu::TextureView; 2],
    frame: usize,
    readback: Rc<wgpu::Buffer>,
    bound_frames: Rc<Cell<u32>>,
}

impl PoolChain {
    fn pair(&self) -> (usize, usize) {
        (self.frame % 2, (self.frame + 1) % 2)
    }
}

impl GraphFrame for PoolChain {
    fn before_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if pass != "sunshafts" {
            return Ok(());
        }
        let (src, dst) = self.pair();
        let value = 2.0 + self.frame as f64;
        let color = wgpu::Color {
            r: value,
            g: value,
            b: value,
            a: 1.0,
        };
        let depth = ctx.targets.depth();
        drop(ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("test.depth_far"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
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
        drop(ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("test.bound_src"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.views[src],
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(color),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        }));
        let hdr = ctx.targets.hdr();
        hdr.bind(&self.views[src], &self.views[dst]);
        if hdr.is_bound() {
            self.bound_frames.set(self.bound_frames.get() + 1);
        }
        Ok(())
    }

    fn after_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if pass != "sunshafts" {
            return Ok(());
        }
        let (_, dst) = self.pair();
        ctx.encoder.copy_texture_to_buffer(
            self.textures[dst].as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(ROW),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: SIZE,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        ctx.targets.hdr().unbind();
        Ok(())
    }

    fn end(&mut self, _table: &ResourceTable) {
        self.frame += 1;
    }
}

#[test]
fn install_instance_lists_refuses_duplicates_and_removes() {
    let name = "install_instance_lists_refuses_duplicates_and_removes";
    let Some(gpu) = software_gpu() else {
        skip(name, "no software adapter");
        return;
    };
    let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, FORMAT, false)));
    ShaftsFeature::install(&gpu, &pipeline).unwrap();
    ShaftsFeature::install_instance(&gpu, &pipeline, "b").unwrap();
    let listed = |pipeline: &Rc<RefCell<Pipeline>>, pass: &str| {
        pipeline
            .borrow()
            .passes()
            .iter()
            .find(|info| info.name == pass)
            .map(|info| (info.order, info.enabled))
    };
    assert_eq!(listed(&pipeline, "sunshafts"), Some((69, true)));
    assert_eq!(listed(&pipeline, "sunshafts#b"), Some((69, true)));
    let err = ShaftsFeature::install_instance(&gpu, &pipeline, "b")
        .err()
        .unwrap();
    assert_eq!(err.code(), "pipeline.duplicate_pass");
    assert_eq!(pipeline.borrow().passes().len(), 2);
    pipeline.borrow_mut().remove("sunshafts#b").unwrap();
    assert_eq!(listed(&pipeline, "sunshafts#b"), None);
    assert_eq!(listed(&pipeline, "sunshafts"), Some((69, true)));
}

#[test]
fn bound_chain_two_frames_composite_into_bound_dst() {
    let name = "bound_chain_two_frames_composite_into_bound_dst";
    let Some(gpu) = software_gpu() else {
        skip(name, "no software adapter");
        return;
    };
    let scope = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let target = OffscreenTarget::new(&gpu, SIZE, SIZE, FORMAT);
    let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, FORMAT, false)));
    let readback = Rc::new(gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("test.readback"),
        size: u64::from(ROW),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
    let bound_frames = Rc::new(Cell::new(0));
    let textures = [
        pool_texture(&gpu, "test.pool.a"),
        pool_texture(&gpu, "test.pool.b"),
    ];
    let views = textures
        .each_ref()
        .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()));
    let chain = PoolChain {
        textures,
        views,
        frame: 0,
        readback: readback.clone(),
        bound_frames: bound_frames.clone(),
    };
    let handle = ShaftsFeature::install(&gpu, &pipeline).unwrap();
    let view = Mat4::look_at_rh(
        Vec3::new(0.0, 2.0, 5.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    );
    handle.set_camera(&view, &Mat4::perspective_rh_zo(1.0, 1.0, 0.1, 100.0));
    handle.set_sun(Vec3::new(0.3, -1.0, 0.2), [5.0, 5.0, 5.0]);
    pipeline
        .borrow_mut()
        .add_graph_frame("pool", Box::new(chain))
        .unwrap();
    for frame in 0..FRAMES {
        let mut graph = FrameGraph::new(FORMAT);
        Pipeline::install(&pipeline, &mut graph);
        graph.compile().unwrap();
        let commands = graph.execute(&gpu, target.view(), (SIZE, SIZE)).unwrap();
        gpu.queue().submit([commands]);
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let expected = 2.0 + frame as f32;
        {
            let mapped = slice.get_mapped_range().unwrap();
            for px in mapped.chunks_exact(8).take(SIZE as usize) {
                let r = f16_to_f32(u16::from_le_bytes([px[0], px[1]]));
                assert!((r - expected).abs() < 1e-2, "frame {frame}: got {r}");
            }
        }
        readback.unmap();
    }
    assert_eq!(bound_frames.get(), FRAMES as u32);
    let error = block_on(scope.pop());
    assert!(error.is_none(), "validation error: {error:?}");
}
