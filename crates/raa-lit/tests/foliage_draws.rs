//! Foliage draws (M7): the prepass and forward traversal selects the additive foliage
//! pipeline and binds group 2 only for associated materials, while ordinary materials keep
//! the unchanged pipeline path.
//!
//! Upstream anchors: `LitHandle::pipeline_for_foliage`
//! (`crates/raa-lit/src/handle.rs:2186`) returns the cached foliage pipeline, and
//! `LitHandle::bind_foliage` (`crates/raa-lit/src/handle.rs:2222`) binds its group 2.
//! The traversal itself is `draw_lit` in `crates/raa-lit/src/passes.rs`.

#![cfg(not(target_arch = "wasm32"))]

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::testing::{block_on, skip, software_gpu};
use raa_core::{FrameGraph, GpuContext, MaterialId, OffscreenTarget, wgpu};
use raa_geom::builtin::DebugCube;
use raa_geom::{GeometryProvider, MeshData, ParamValue, Params, Quality};
use raa_lit::foliage_resources::{
    FOLIAGE_MODE_POM, FOLIAGE_MODE_SPOM, FoliageParams, FoliageResources, PackedFoliageLayers,
};
use raa_lit::{AttributedMesh, LitOptions, LitRenderer, PipelineSpec, SurfaceMaterial};
use raa_math::{Quat, Vec3};
use raa_pipeline::Pipeline;
use raa_scene::{RenderList, Scene, Transform};
use raa_shader::{MaterialFnId, PassKind, VariantKey};

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
        label: Some("foliage_draws.readback"),
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
        for px in row[..row_bytes].as_chunks::<8>().0 {
            texels.push(std::array::from_fn(|c| {
                f16_to_f32(u16::from_le_bytes([px[2 * c], px[2 * c + 1]]))
            }));
        }
    }
    texels
}

/// The DebugCube (half 0.25) as an attributed mesh (S10 default uv, tangent, colour).
fn cube() -> AttributedMesh {
    let mut mesh = MeshData::new();
    let params = Params::new().with("half", ParamValue::Scalar(0.25));
    DebugCube
        .generate(&params, Quality::LOD0, &mut mesh)
        .expect("debug cube");
    AttributedMesh::from_mesh(mesh)
}

/// Validated impostor parameters of `mode` for one small generation.
fn params(mode: u32) -> FoliageParams {
    FoliageParams {
        min: [-1.0, -1.0, -1.0],
        max: [1.0, 1.0, 1.0],
        resolution: 16,
        mode,
        translucency: 0.3,
        generation: 7,
        cutoff: 0.5,
        view_count: 2,
        texture_resolution: 64,
    }
}

/// The forward pipeline spec the foliage lookup refuses before any association check.
fn forward_spec() -> PipelineSpec {
    PipelineSpec {
        color_targets: vec![Some(wgpu::ColorTargetState {
            format: raa_pipeline::HDR_FORMAT,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })],
        depth: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        cull: Some(wgpu::Face::Back),
        alpha_to_coverage: false,
        extra_layout: None,
    }
}

fn is_background(t: &[f32; 4]) -> bool {
    t.iter().zip(BACKGROUND).all(|(a, b)| (a - b).abs() <= 1e-3)
}

/// A pipeline with a `LitRenderer` and one small cube each for an ordinary material
/// (outside the source-root bounds) and two NPC impostor materials (SPOM and POM, inside).
struct Rig {
    pipeline: Rc<RefCell<Pipeline>>,
    lit: LitRenderer,
    scene: Scene,
    list: RenderList,
    tags: Vec<u32>,
    slots: Vec<u32>,
    graph: FrameGraph,
    backbuffer: OffscreenTarget,
    ordinary: MaterialId,
    spom: MaterialId,
    pom: MaterialId,
    mesh: raa_lit::MeshId,
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
        let ordinary = MaterialId(0);
        let spom = lit
            .materials_mut()
            .add(SurfaceMaterial::default())
            .expect("spom material");
        let pom = lit
            .materials_mut()
            .add(SurfaceMaterial::default())
            .expect("pom material");
        lit.handle()
            .set_foliage(spom, Some(Rc::new(foliage(gpu, FOLIAGE_MODE_SPOM))))
            .expect("spom association");
        lit.handle()
            .set_foliage(pom, Some(Rc::new(foliage(gpu, FOLIAGE_MODE_POM))))
            .expect("pom association");

        let mut scene = Scene::with_capacity(3, 3);
        for (x, material) in [(2.0, ordinary), (-0.6, spom), (0.6, pom)] {
            let node = scene
                .add_node(
                    None,
                    Transform::from_trs(Vec3::new(x, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE),
                )
                .expect("node");
            scene.add_instance(node, mesh, material).expect("instance");
        }
        scene.camera.set_viewport(SIZE.0, SIZE.1);
        let mut graph = FrameGraph::new(SURFACE);
        Pipeline::install(&pipeline, &mut graph);
        graph.compile().expect("compile");
        Rig {
            pipeline,
            lit,
            scene,
            list: RenderList::with_capacity(3),
            tags: Vec::with_capacity(3),
            slots: Vec::with_capacity(3),
            graph,
            backbuffer: OffscreenTarget::new(gpu, SIZE.0, SIZE.1, SURFACE),
            ordinary,
            spom,
            pom,
            mesh,
        }
    }

    /// Prepares and renders one frame, then waits for the GPU.
    fn frame(&mut self, gpu: &GpuContext) {
        self.lit
            .prepare(
                gpu,
                &mut self.scene,
                &mut self.list,
                &mut self.tags,
                &mut self.slots,
            )
            .expect("prepare");
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

/// Full-coverage packed arrays (`Rgba8Unorm` albedo/coverage, `Rgba16Float`
/// normal/depth) with one finite identity view per generation view.
fn layers(gpu: &GpuContext, p: &FoliageParams) -> PackedFoliageLayers {
    let res = p.texture_resolution;
    let views = p.view_count;
    let extent = wgpu::Extent3d {
        width: res,
        height: res,
        depth_or_array_layers: views,
    };
    let texture = |label: &str, format, bytes_per_texel: u32, fill: u8| {
        let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let bytes = vec![fill; (res * res * views * bytes_per_texel) as usize];
        gpu.queue().write_texture(
            texture.as_image_copy(),
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(res * bytes_per_texel),
                rows_per_image: Some(res),
            },
            extent,
        );
        texture
    };
    let mut identity = [0.0f32; 16];
    identity[0] = 1.0;
    identity[5] = 1.0;
    identity[10] = 1.0;
    identity[15] = 1.0;
    let submission = gpu.queue().submit(std::iter::empty());
    PackedFoliageLayers {
        albedo_coverage: texture(
            "foliage_draws.albedo_coverage",
            wgpu::TextureFormat::Rgba8Unorm,
            4,
            0xff,
        ),
        normal_depth: texture(
            "foliage_draws.normal_depth",
            wgpu::TextureFormat::Rgba16Float,
            8,
            0,
        ),
        views: vec![(identity, [0.1, 10.0]); views as usize],
        submission,
    }
}

/// An initialized impostor generation of `mode` with full coverage in every texel.
fn foliage(gpu: &GpuContext, mode: u32) -> FoliageResources {
    let p = params(mode);
    let mut resources = FoliageResources::new(gpu, p).expect("foliage resources");
    let packed = layers(gpu, &p);
    let volume = vec![0u8; 4 * (p.resolution * p.resolution * p.resolution) as usize];
    resources
        .initialize(gpu, Some(&packed), &volume)
        .expect("initialize foliage resources");
    resources
}

fn invalid_code() -> &'static str {
    raa_core::RaaError::FrameGraph {
        code: "foliage.invalid",
        message: String::new(),
    }
    .code()
}

#[test]
fn mixed_ordinary_and_foliage_materials_draw_through_their_pipelines() {
    let test = "mixed_ordinary_and_foliage_materials_draw_through_their_pipelines";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let mut rig = Rig::new(&gpu, 1);
    let handle = rig.lit.handle();
    assert_eq!(handle.foliage_mode(rig.ordinary), None);
    assert_eq!(handle.foliage_mode(rig.spom), Some(FOLIAGE_MODE_SPOM));
    assert_eq!(handle.foliage_mode(rig.pom), Some(FOLIAGE_MODE_POM));

    rig.frame(&gpu);

    let texels = rig.hdr_src(&gpu);
    let error = block_on(errors.pop());
    assert!(error.is_none(), "GPU validation error: {error:?}");
    assert!(
        texels.iter().any(|t| !is_background(t)),
        "the mixed materials reach the HDR target"
    );
    // The ordinary cube is at x 2 (screen band 2) and the two impostor cubes at x -0.6
    // and 0.6 (band 1): both the ordinary and the foliage draws reach the target.
    let (w, h) = (SIZE.0 as usize, SIZE.1 as usize);
    for band in [1, 2] {
        let covered = (0..h)
            .flat_map(|y| {
                (0..w)
                    .filter(move |x| x * 3 / w == band)
                    .map(move |x| (x, y))
            })
            .filter(|&(x, y)| !is_background(&texels[y * w + x]))
            .count();
        assert!(covered > 0, "screen band {band} draws a cube");
    }
}

#[test]
fn detached_association_draws_as_ordinary_and_the_foliage_lookup_refuses_it() {
    let test = "detached_association_draws_as_ordinary_and_the_foliage_lookup_refuses_it";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let mut rig = Rig::new(&gpu, 4);
    let handle = rig.lit.handle();
    // An ordinary material is refused by the foliage-only entry point.
    let key = VariantKey {
        material_fn: MaterialFnId::STANDARD,
        pass: PassKind::Forward,
        msaa: 4,
        flags: 0,
    };
    let refused = handle
        .pipeline_for_foliage(&gpu, rig.mesh, key, &forward_spec(), rig.ordinary)
        .expect_err("an ordinary material has no foliage generation");
    assert_eq!(refused.code(), invalid_code());

    // Detaching one association clears it before release; the draw falls back to ordinary.
    handle.set_foliage(rig.spom, None).expect("detach spom");
    assert_eq!(handle.foliage_mode(rig.spom), None);
    assert_eq!(handle.foliage_mode(rig.pom), Some(FOLIAGE_MODE_POM));

    rig.frame(&gpu);

    let texels = rig.hdr_src(&gpu);
    let error = block_on(errors.pop());
    assert!(error.is_none(), "GPU validation error: {error:?}");
    assert!(
        texels.iter().any(|t| !is_background(t)),
        "the detached and associated materials still draw"
    );
}
