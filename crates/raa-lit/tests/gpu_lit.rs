//! The lit renderer on a software GPU (SKIP without an adapter): a raa_geom `DebugCube`
//! uploaded through `LitRenderer`, prepared from a scene and drawn by the pipeline's
//! `prepass` and `forward` passes (no tonemap) with MSAA 4 and 1. The HDR chain source
//! after the frame is read back: some texel shows the cube (differs from the background)
//! and the corners show the cleared background.

#![cfg(not(target_arch = "wasm32"))]

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use raa_core::testing::{skip, software_gpu};
use raa_core::{FrameGraph, GpuContext, MaterialId, OffscreenTarget, RaaError, wgpu};
use raa_geom::builtin::DebugCube;
use raa_geom::{GeometryProvider, MeshData, ParamValue, Params, Quality};
use raa_lit::{AttributedMesh, ForwardContributor, LitHandle, LitOptions, LitRenderer};
use raa_math::{Quat, Vec3};
use raa_pipeline::{PassRole, Pipeline};
use raa_scene::{RenderList, Scene, Transform};

const SIZE: (u32, u32) = (64, 64);
const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// Display-referred background (CLEAR_COLOR rgb), the default of the pipeline's `EnvGpu`.
const BACKGROUND: [f32; 3] = [0.05, 0.06, 0.08];

/// Decodes an IEEE 754 binary16 value.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((bits >> 10) & 0x1f);
    let mant = f32::from(bits & 0x3ff);
    match exp {
        0 => sign * mant * 2f32.powi(-24),
        31 if mant == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + mant / 1024.0) * 2f32.powi(exp - 15),
    }
}

/// Reads an Rgba16Float texture of `size` back as rgba f32 texels, top row first.
fn read_rgba16f(gpu: &GpuContext, texture: &wgpu::Texture, size: (u32, u32)) -> Vec<[f32; 4]> {
    let row_bytes = size.0 as usize * 8;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
    let padded = row_bytes.div_ceil(align) * align;
    let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_lit.readback"),
        size: (padded * size.1 as usize) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded as u32),
                rows_per_image: Some(size.1),
            },
        },
        wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue().submit([encoder.finish()]);
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");
    rx.recv().expect("map callback").expect("map");
    let mapped = slice.get_mapped_range().expect("mapped range");
    let mut texels = Vec::with_capacity((size.0 * size.1) as usize);
    for row in mapped.chunks_exact(padded) {
        for px in row[..row_bytes].chunks_exact(8) {
            texels.push(std::array::from_fn(|c| {
                f16_to_f32(u16::from_le_bytes([px[2 * c], px[2 * c + 1]]))
            }));
        }
    }
    texels
}

/// The DebugCube (half 0.5) as an attributed mesh (S10 default uv, tangent, colour).
fn cube() -> AttributedMesh {
    let mut mesh = MeshData::new();
    let params = Params::new().with("half", ParamValue::Scalar(0.5));
    DebugCube
        .generate(&params, Quality::LOD0, &mut mesh)
        .expect("debug cube");
    AttributedMesh::from_mesh(mesh)
}

/// A contributor that logs its name into `log` each time the forward pass records it.
struct Logger {
    name: &'static str,
    order: u32,
    log: Rc<RefCell<Vec<&'static str>>>,
    prepared: Rc<Cell<u32>>,
}

impl ForwardContributor for Logger {
    fn name(&self) -> &str {
        self.name
    }

    fn order(&self) -> u32 {
        self.order
    }

    fn prepare(
        &mut self,
        _lit: &LitHandle,
        _ctx: &mut raa_pipeline::PrepareCtx<'_>,
    ) -> Result<(), RaaError> {
        self.prepared.set(self.prepared.get() + 1);
        Ok(())
    }

    fn record(&self, _pass: &mut wgpu::RenderPass<'_>, _lit: &LitHandle) -> Result<(), RaaError> {
        self.log.borrow_mut().push(self.name);
        Ok(())
    }
}

/// A pipeline with a `LitRenderer` of `msaa` samples, a scene with one cube at the origin
/// seen by the default camera, and the frame graph running the pipeline.
struct Rig {
    pipeline: Rc<RefCell<Pipeline>>,
    lit: LitRenderer,
    scene: Scene,
    list: RenderList,
    tags: Vec<u32>,
    graph: FrameGraph,
    backbuffer: OffscreenTarget,
}

impl Rig {
    fn new(gpu: &GpuContext, msaa: u32) -> Rig {
        let pipeline = Rc::new(RefCell::new(Pipeline::new(gpu, SURFACE, false)));
        let opts = LitOptions {
            msaa,
            ..LitOptions::default()
        };
        let mut lit = LitRenderer::new(gpu, &pipeline, opts).expect("lit renderer");
        let id = lit.upload(gpu, &cube()).expect("upload");
        let mesh = lit.submesh_ids(id).expect("sub-meshes")[0];
        let mut scene = Scene::with_capacity(1, 1);
        let rotation = Quat::from_axis_angle(Vec3::Y, 0.6);
        let node = scene
            .add_node(None, Transform::from_trs(Vec3::ZERO, rotation, Vec3::ONE))
            .expect("node");
        scene
            .add_instance(node, mesh, MaterialId(0))
            .expect("instance");
        scene.camera.set_viewport(SIZE.0, SIZE.1);
        let mut graph = FrameGraph::new(SURFACE);
        Pipeline::install(&pipeline, &mut graph);
        graph.compile().expect("compile");
        Rig {
            pipeline,
            lit,
            scene,
            list: RenderList::with_capacity(1),
            tags: Vec::with_capacity(1),
            graph,
            backbuffer: OffscreenTarget::new(gpu, SIZE.0, SIZE.1, SURFACE),
        }
    }

    /// Prepares and renders one frame, then waits for the GPU.
    fn frame(&mut self, gpu: &GpuContext) {
        let counts = self
            .lit
            .prepare(gpu, &mut self.scene, &mut self.list, &mut self.tags)
            .expect("prepare");
        assert_eq!(
            (counts.instances, counts.draw_calls, counts.triangles),
            (1, 1, 12)
        );
        let commands = self
            .graph
            .execute(gpu, self.backbuffer.view(), SIZE)
            .expect("execute");
        gpu.queue().submit([commands]);
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
    }

    /// The HDR chain source (what the next pass after forward reads).
    fn hdr_src(&self, gpu: &GpuContext) -> Vec<[f32; 4]> {
        let p = self.pipeline.borrow();
        let targets = p.targets().expect("targets after a frame");
        assert_eq!(targets.size(), SIZE);
        read_rgba16f(gpu, targets.hdr().src_texture(), SIZE)
    }
}

fn is_background(t: &[f32; 4]) -> bool {
    t.iter().zip(BACKGROUND).all(|(a, b)| (a - b).abs() <= 1e-3)
}

/// Renders the cube twice with `msaa` samples and checks the HDR source.
fn cube_reaches_hdr(test: &str, msaa: u32) {
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let mut rig = Rig::new(&gpu, msaa);
    assert_eq!(rig.lit.handle().msaa(), msaa);
    rig.frame(&gpu);
    rig.frame(&gpu);
    let texels = rig.hdr_src(&gpu);
    let error = raa_core::testing::block_on(errors.pop());
    assert!(error.is_none(), "GPU validation error: {error:?}");

    let (w, h) = (SIZE.0 as usize, SIZE.1 as usize);
    for corner in [0, w - 1, (h - 1) * w, h * w - 1] {
        assert!(
            is_background(&texels[corner]),
            "corner {corner} is the background: {:?}",
            texels[corner]
        );
    }
    let centre = texels[(h / 2) * w + w / 2];
    assert!(!is_background(&centre), "the cube covers the centre");
    assert!(centre.iter().all(|c| c.is_finite()));
    let covered = texels.iter().filter(|t| !is_background(t)).count();
    assert!(covered > 16 && covered < w * h / 2, "{covered} cube texels");
}

#[test]
fn forward_msaa_4_draws_the_cube_into_hdr() {
    cube_reaches_hdr("forward_msaa_4_draws_the_cube_into_hdr", 4);
}

#[test]
fn forward_msaa_1_draws_the_cube_into_hdr() {
    cube_reaches_hdr("forward_msaa_1_draws_the_cube_into_hdr", 1);
}

#[test]
fn lit_renderer_registers_prepass_and_forward_once() {
    let test = "lit_renderer_registers_prepass_and_forward_once";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, SURFACE, false)));
    let _lit = LitRenderer::new(&gpu, &pipeline, LitOptions::default()).expect("lit renderer");
    let rows: Vec<_> = pipeline
        .borrow()
        .passes()
        .into_iter()
        .map(|p| (p.name, p.role, p.order, p.required, p.enabled))
        .collect();
    assert_eq!(
        rows,
        [
            ("prepass".to_string(), PassRole::Pass, 30, true, true),
            ("forward".to_string(), PassRole::Pass, 60, true, true),
        ]
    );
    let again = LitRenderer::new(&gpu, &pipeline, LitOptions::default());
    assert_eq!(
        again.err().map(|e| e.code()),
        Some("pipeline.duplicate_pass")
    );
    assert_eq!(pipeline.borrow().passes().len(), 2, "nothing half-added");
    let off = pipeline.borrow_mut().set_enabled("forward", false);
    assert_eq!(off.err().map(|e| e.code()), Some("pipeline.required_pass"));
}

#[test]
fn forward_records_enabled_contributors_by_order() {
    let test = "forward_records_enabled_contributors_by_order";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let mut rig = Rig::new(&gpu, 4);
    let log = Rc::new(RefCell::new(Vec::new()));
    let prepared = Rc::new(Cell::new(0));
    for (name, order) in [("late", 20), ("early", 10), ("off", 15)] {
        let logger = Logger {
            name,
            order,
            log: Rc::clone(&log),
            prepared: Rc::clone(&prepared),
        };
        rig.lit
            .add_contributor(Box::new(logger))
            .expect("contributor");
    }
    rig.pipeline
        .borrow_mut()
        .set_enabled("off", false)
        .expect("disable a contributor");
    let names: Vec<_> = rig
        .pipeline
        .borrow()
        .passes()
        .into_iter()
        .map(|p| p.name)
        .collect();
    assert_eq!(names, ["prepass", "forward", "early", "off", "late"]);
    rig.frame(&gpu);
    assert_eq!(*log.borrow(), ["early", "late"]);
    assert_eq!(prepared.get(), 2, "only enabled contributors are prepared");
    let texels = rig.hdr_src(&gpu);
    assert!(texels.iter().any(|t| !is_background(t)));
}
