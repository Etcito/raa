//! The linked-list OIT on a software GPU (SKIP without an adapter): two translucent quads
//! covering the same pixels at local z = -0.3 and +0.3 of instances at the SAME world
//! translation, so raa-lit's back-to-front sort ties and falls back to tag order. Frame A
//! submits them with tags (1, 2), frame B with the tags swapped. A probe pass at order 150
//! copies the centre row of hdr.src after `transparent.oit`: both frames give bit-identical
//! pixels, the centre equals the CPU `resolve_cpu` result over the background and a pixel
//! outside the quads is the background.

#![cfg(not(target_arch = "wasm32"))]

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::testing::{skip, software_gpu};
use raa_core::{FrameGraph, GpuContext, MaterialId, OffscreenTarget, RaaError, wgpu};
use raa_lit::{
    DrawClass, FLAG_TRANSPARENT, LitOptions, LitRenderer, SurfaceMaterial, SurfaceModel,
};
use raa_math::Vec3;
use raa_mesh::{AttributedMesh, SubMesh};
use raa_oit::node::{END, f16_to_f32};
use raa_oit::resolve_cpu::resolve_pixel;
use raa_oit::{DEFAULT_MAX_FRAGMENTS, OitFeature, OitNode};
use raa_pipeline::{FeaturePass, Pipeline, RecordCtx};
use raa_scene::{InstanceId, RenderList, Scene, Transform};

const SIZE: (u32, u32) = (64, 64);
const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// Display-referred background (CLEAR_COLOR rgb), the default of the pipeline's `EnvGpu`.
const BACKGROUND: [f32; 3] = [0.05, 0.06, 0.08];
/// One Rgba16Float row of 64 px (512 B, a multiple of COPY_BYTES_PER_ROW_ALIGNMENT).
const ROW: u32 = SIZE.0 * 8;
/// The row the probe copies (through the centre of the quads).
const PROBE_Y: u32 = SIZE.1 / 2;
/// Half extent of both quads (m).
const HALF: f32 = 1.0;
const RED: [f32; 3] = [1.0, 0.0, 0.0];
const BLUE: [f32; 3] = [0.0, 0.0, 1.0];
const ALPHA: f32 = 0.5;

/// Copies row `PROBE_Y` of hdr.src() into a readback buffer after the OIT resolve.
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
            wgpu::TexelCopyTextureInfo {
                texture: ctx.targets.hdr().src_texture(),
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: PROBE_Y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.0,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(ROW),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: SIZE.0,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        Ok(())
    }
}

/// A CCW quad facing +Z (normal +Z) of half extent `HALF` in the plane local z = `z`.
fn quad(z: f32) -> AttributedMesh {
    let mut m = AttributedMesh::default();
    m.base.positions = vec![
        [-HALF, -HALF, z],
        [HALF, -HALF, z],
        [HALF, HALF, z],
        [-HALF, HALF, z],
    ];
    m.base.normals = vec![[0.0, 0.0, 1.0]; 4];
    m.base.indices = vec![0, 1, 2, 0, 2, 3];
    m.submeshes = vec![SubMesh {
        first_index: 0,
        index_count: 6,
        material: None,
    }];
    m
}

/// Flat (exact shading: base + emissive), transparent, opacity `ALPHA`.
fn translucent(rgb: [f32; 3]) -> SurfaceMaterial {
    SurfaceMaterial {
        model: SurfaceModel::Flat,
        base_color: rgb,
        opacity: ALPHA,
        flags: FLAG_TRANSPARENT,
        ..SurfaceMaterial::default()
    }
}

/// Pipeline + lit renderer (msaa 1) + OIT, the two quads at the origin seen from +Z, and
/// the probe's readback buffer.
struct Rig {
    lit: LitRenderer,
    scene: Scene,
    list: RenderList,
    tags: Vec<u32>,
    graph: FrameGraph,
    backbuffer: OffscreenTarget,
    readback: Rc<wgpu::Buffer>,
    /// (far quad at z = -0.3, blue; near quad at z = +0.3, red).
    instances: [InstanceId; 2],
}

impl Rig {
    fn new(gpu: &GpuContext) -> Rig {
        let pipeline = Rc::new(RefCell::new(Pipeline::new(gpu, SURFACE, false)));
        let opts = LitOptions {
            msaa: 1,
            ..LitOptions::default()
        };
        let mut lit = LitRenderer::new(gpu, &pipeline, opts).expect("lit renderer");
        let _oit = OitFeature::install(gpu, &pipeline, &mut lit).expect("oit install");
        let readback = Rc::new(gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_oit.readback"),
            size: u64::from(ROW),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        pipeline
            .borrow_mut()
            .add(Box::new(Probe(readback.clone())))
            .expect("probe");

        let mut scene = Scene::with_capacity(2, 2);
        let mut instances = Vec::new();
        for (z, rgb) in [(-0.3, BLUE), (0.3, RED)] {
            let id = lit.upload(gpu, &quad(z)).expect("upload");
            let mesh = lit.submesh_ids(id).expect("sub-meshes")[0];
            let material = lit.materials_mut().add(translucent(rgb)).expect("material");
            let node = scene.add_node(None, Transform::IDENTITY).expect("node");
            instances.push(scene.add_instance(node, mesh, material).expect("instance"));
        }
        scene.camera.eye = Vec3::new(0.0, 0.0, 5.0);
        scene.camera.target = Vec3::ZERO;
        scene.camera.up = Vec3::Y;
        scene.camera.set_viewport(SIZE.0, SIZE.1);

        let mut graph = FrameGraph::new(SURFACE);
        Pipeline::install(&pipeline, &mut graph);
        graph.compile().expect("compile");
        Rig {
            lit,
            scene,
            list: RenderList::with_capacity(2),
            tags: Vec::with_capacity(2),
            graph,
            backbuffer: OffscreenTarget::new(gpu, SIZE.0, SIZE.1, SURFACE),
            readback,
            instances: [instances[0], instances[1]],
        }
    }

    /// Tags the (far, near) instances, renders one frame and returns the materials of the
    /// Transparent draws in submission order and the probed row (rgba f16 bits per pixel).
    fn frame(&mut self, gpu: &GpuContext, tags: [u32; 2]) -> (Vec<MaterialId>, Vec<[u16; 4]>) {
        for (id, tag) in self.instances.into_iter().zip(tags) {
            self.scene.set_instance_tag(id, tag).expect("tag");
        }
        let counts = self
            .lit
            .prepare(gpu, &mut self.scene, &mut self.list, &mut self.tags)
            .expect("prepare");
        assert_eq!((counts.instances, counts.triangles), (2, 4));
        let order = self
            .lit
            .handle()
            .draws(DrawClass::Transparent)
            .iter()
            .map(|d| d.material)
            .collect();
        let commands = self
            .graph
            .execute(gpu, self.backbuffer.view(), SIZE)
            .expect("execute");
        gpu.queue().submit([commands]);
        let slice = self.readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
        rx.recv().expect("map callback").expect("map");
        let row = {
            let mapped = slice.get_mapped_range().expect("mapped range");
            mapped
                .chunks_exact(8)
                .map(|px| std::array::from_fn(|c| u16::from_le_bytes([px[2 * c], px[2 * c + 1]])))
                .collect()
        };
        self.readback.unmap();
        (order, row)
    }
}

fn rgb(px: &[u16; 4]) -> [f32; 3] {
    [0, 1, 2].map(|c| f16_to_f32(px[c]))
}

fn near(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

#[test]
fn two_quads_same_pixels_in_both_draw_orders() {
    let test = "two_quads_same_pixels_in_both_draw_orders";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let mut rig = Rig::new(&gpu);
    let (order_a, a) = rig.frame(&gpu, [1, 2]);
    let (order_b, b) = rig.frame(&gpu, [2, 1]);
    let error = pollster::block_on(errors.pop());
    assert!(error.is_none(), "GPU validation error: {error:?}");

    assert_eq!(order_a.len(), 2);
    assert_eq!(
        order_a.iter().rev().collect::<Vec<_>>(),
        order_b.iter().collect::<Vec<_>>(),
        "swapping the tags swaps the submission order"
    );
    assert_eq!(a.len(), SIZE.0 as usize);
    assert_eq!(a, b, "the submission order changed the pixels");

    // CPU reference: the near red fragment over the far blue one over the background (the
    // list holds them in either order; only the depths decide).
    let nodes = [
        OitNode::pack(BLUE, ALPHA, 0.6, END),
        OitNode::pack(RED, ALPHA, 0.4, 0),
    ];
    let expected = resolve_pixel(&nodes, 1, BACKGROUND, DEFAULT_MAX_FRAGMENTS);
    let centre = rgb(&a[SIZE.0 as usize / 2]);
    assert!(
        near(centre, expected, 1e-3),
        "centre {centre:?}, CPU resolve {expected:?}"
    );
    let outside = rgb(&a[2]);
    assert!(
        near(outside, BACKGROUND, 1e-3),
        "outside the quads {outside:?}, background {BACKGROUND:?}"
    );
}
