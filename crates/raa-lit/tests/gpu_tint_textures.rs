//! S23 tint mode and the material textures on a software GPU (SKIP without an adapter): one
//! cube drawn with an unlit user material function, the HDR source read back at the centre.
//! A `graph` material selects its own variant (no tint colour multiply) and reads the instance
//! tint once it is set; textures uploaded through `LitRenderer::add_texture` are sampled by
//! `raa_texture` in the material function.

#![cfg(not(target_arch = "wasm32"))]

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::testing::{block_on, skip, software_gpu};
use raa_core::{FrameGraph, GpuContext, MaterialId, OffscreenTarget, wgpu};
use raa_geom::builtin::DebugCube;
use raa_geom::{GeometryProvider, MeshData, ParamValue, Params, Quality};
use raa_lit::{
    AttributedMesh, FLAG_UNLIT, InstanceExtra, LitOptions, LitRenderer, SurfaceMaterial,
    SurfaceModel, TexelData, TintMode,
};
use raa_pipeline::Pipeline;
use raa_scene::{RenderList, Scene, Transform};
use raa_shader::VARIANT_FLAG_TINT_GRAPH;

const SIZE: (u32, u32) = (32, 32);
const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Decodes an IEEE 754 binary16 value (normal and zero values only).
fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((bits >> 10) & 0x1f);
    let mant = f32::from(bits & 0x3ff);
    if exp == 0 {
        return sign * mant * 2f32.powi(-24);
    }
    sign * (1.0 + mant / 1024.0) * 2f32.powi(exp - 15)
}

/// The rgb of the centre texel of an Rgba16Float texture of SIZE.
fn centre_rgb(gpu: &GpuContext, texture: &wgpu::Texture) -> [f32; 3] {
    let row = SIZE.0 as usize * 8;
    let padded = row.div_ceil(256) * 256;
    let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_tint_textures.readback"),
        size: (padded * SIZE.1 as usize) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device().create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded as u32),
                rows_per_image: Some(SIZE.1),
            },
        },
        wgpu::Extent3d {
            width: SIZE.0,
            height: SIZE.1,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue().submit([encoder.finish()]);
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let mapped = slice.get_mapped_range().expect("mapped range");
    let at = padded * (SIZE.1 as usize / 2) + 8 * (SIZE.0 as usize / 2);
    std::array::from_fn(|c| {
        f16_to_f32(u16::from_le_bytes([
            mapped[at + 2 * c],
            mapped[at + 2 * c + 1],
        ]))
    })
}

/// A lit renderer drawing one cube with material 1, an unlit material of the user function
/// `wgsl`, in front of the default camera.
struct Rig {
    pipeline: Rc<RefCell<Pipeline>>,
    lit: LitRenderer,
    scene: Scene,
    graph: FrameGraph,
    backbuffer: OffscreenTarget,
}

impl Rig {
    fn new(gpu: &GpuContext, wgsl: &str) -> Rig {
        let pipeline = Rc::new(RefCell::new(Pipeline::new(gpu, SURFACE, false)));
        let mut lit = LitRenderer::new(gpu, &pipeline, LitOptions::default()).expect("lit");
        let fid = lit.register_user_fn("probe", wgsl).expect("register");
        let material = lit
            .materials_mut()
            .add(SurfaceMaterial {
                model: SurfaceModel::Function(fid),
                flags: FLAG_UNLIT,
                ..SurfaceMaterial::default()
            })
            .expect("material");
        assert_eq!(material, MaterialId(1));
        let mut mesh = MeshData::new();
        let params = Params::new().with("half", ParamValue::Scalar(0.5));
        DebugCube
            .generate(&params, Quality::LOD0, &mut mesh)
            .expect("cube");
        let id = lit
            .upload(gpu, &AttributedMesh::from_mesh(mesh))
            .expect("upload");
        let sub = lit.submesh_ids(id).expect("sub-meshes")[0];
        let mut scene = Scene::with_capacity(1, 1);
        let node = scene.add_node(None, Transform::IDENTITY).expect("node");
        scene.add_instance(node, sub, material).expect("instance");
        scene.camera.set_viewport(SIZE.0, SIZE.1);
        let mut graph = FrameGraph::new(SURFACE);
        Pipeline::install(&pipeline, &mut graph);
        graph.compile().expect("compile");
        let backbuffer = OffscreenTarget::new(gpu, SIZE.0, SIZE.1, SURFACE);
        Rig {
            pipeline,
            lit,
            scene,
            graph,
            backbuffer,
        }
    }

    /// Renders one frame and returns the centre of the HDR source; no validation error.
    fn centre(&mut self, gpu: &GpuContext) -> [f32; 3] {
        let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
        let (mut list, mut tags) = (RenderList::with_capacity(1), Vec::new());
        let counts = self
            .lit
            .prepare(gpu, &mut self.scene, &mut list, &mut tags)
            .expect("prepare");
        assert_eq!(counts.draw_calls, 1);
        let commands = self
            .graph
            .execute(gpu, self.backbuffer.view(), SIZE)
            .expect("execute");
        gpu.queue().submit([commands]);
        let error = block_on(errors.pop());
        assert!(error.is_none(), "GPU validation error: {error:?}");
        let p = self.pipeline.borrow();
        centre_rgb(gpu, p.targets().expect("targets").hdr().src_texture())
    }
}

fn assert_rgb(test: &str, got: [f32; 3], want: [f32; 3]) {
    let close = got.iter().zip(want).all(|(g, w)| (g - w).abs() <= 2e-3);
    assert!(close, "{test}: centre {got:?}, want {want:?}");
}

/// The S23 `input.player_color` node: the instance tint once set, else its preview colour.
const PLAYER_COLOR: &str = "fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    o.base_color = select(vec3<f32>(0.5, 0.25, 0.125), mat_instance_tint.rgb, mat_instance_tint_set > 0.5);
    o.emissive = vec3<f32>(0.0);
    return o;
}
";

#[test]
fn graph_mode_selects_its_variant_and_the_tint_reaches_the_node() {
    let test = "graph_mode_selects_its_variant_and_the_tint_reaches_the_node";
    let Some(gpu) = software_gpu() else {
        return skip(test, "software_gpu returned None");
    };
    let mut rig = Rig::new(&gpu, PLAYER_COLOR);
    let (lit, m) = (rig.lit.handle(), MaterialId(1));
    let tint = [0.25, 0.75, 0.5, 1.0];
    let tinted = InstanceExtra {
        tint,
        tint_set: 1,
        ..InstanceExtra::default()
    };
    // Default (multiply): the node's colour (the tint once set) times the tint, as before S23.
    assert_eq!(lit.material_variant_flags(m), 0);
    assert_rgb(test, rig.centre(&gpu), [0.5, 0.25, 0.125]);
    rig.lit.set_instance_extra(0, tinted);
    assert_rgb(test, rig.centre(&gpu), [0.0625, 0.5625, 0.25]);
    // Graph: its own variant, no multiply, the node shows the tint.
    lit.set_material_tint_mode(m, TintMode::Graph);
    assert_eq!(lit.material_variant_flags(m), VARIANT_FLAG_TINT_GRAPH);
    assert_eq!(lit.material_variant_flags(MaterialId(0)), 0);
    assert_rgb(test, rig.centre(&gpu), [0.25, 0.75, 0.5]);
    // Tint not set (clearInstanceTint): the node's preview colour.
    rig.lit.set_instance_extra(
        0,
        InstanceExtra {
            tint,
            ..InstanceExtra::default()
        },
    );
    assert_rgb(test, rig.centre(&gpu), [0.5, 0.25, 0.125]);
    lit.set_material_tint_mode(m, TintMode::Multiply);
    assert_eq!(lit.material_tint_mode(m), TintMode::Multiply);
    rig.lit.set_instance_extra(0, tinted);
    assert_rgb(test, rig.centre(&gpu), [0.0625, 0.5625, 0.25]);
}

/// Base colour = texture 2 at uv (0, 0.5), plus half the blue of texture 2 at uv (3, 0.5)
/// (clamped to the right texel), plus unknown texture 7.
const SAMPLER: &str = "fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    let left = raa_texture(2u, vec2<f32>(0.0, 0.5)).rgb;
    let right = raa_texture(2u, vec2<f32>(3.0, 0.5)).rgb;
    o.base_color = left + vec3<f32>(0.0, 0.0, 0.5 * right.b) + raa_texture(7u, in.uv).rgb;
    o.emissive = vec3<f32>(0.0);
    return o;
}
";

#[test]
fn uploaded_texture_is_sampled_through_raa_texture() {
    let test = "uploaded_texture_is_sampled_through_raa_texture";
    let Some(gpu) = software_gpu() else {
        return skip(test, "software_gpu returned None");
    };
    let mut rig = Rig::new(&gpu, SAMPLER);
    // Id 1, a white neighbour packed on the same shelf (never bleeds into texture 2).
    let white = vec![255u8; 3 * 2 * 4];
    assert_eq!(
        rig.lit
            .add_texture(&gpu, 3, 2, TexelData::Rgba8(&white))
            .unwrap(),
        1
    );
    // Id 2, 2 x 1 Float32: left (-1, 0.5, 0) clamps to (0, 128, 0); right (0.2, 0, 2) to (51, 0, 255).
    let texels = [-1.0, 0.5, 0.0, 1.0, 0.2, 0.0, 2.0, 1.0];
    assert_eq!(
        rig.lit
            .add_texture(&gpu, 2, 1, TexelData::Float(&texels))
            .unwrap(),
        2
    );
    let rect = rig.lit.texture_rect(2).expect("rect");
    assert_eq!((rect.x, rect.y, rect.width, rect.height), (3, 0, 2, 1));
    // left + 0.5 * right.b: (0, 128/255, 0.5); the unknown id samples zero.
    assert_rgb(test, rig.centre(&gpu), [0.0, 128.0 / 255.0, 0.5]);
    let bad = rig
        .lit
        .add_texture(&gpu, 2, 2, TexelData::Rgba8(&white[..15]));
    assert_eq!(bad.unwrap_err().code(), "texture.invalid");
    // Removed: zero, and a second removal changes nothing.
    assert!(rig.lit.remove_texture(&gpu, 2).unwrap());
    assert!(!rig.lit.remove_texture(&gpu, 2).unwrap());
    assert_eq!(rig.lit.texture_rect(2), None);
    assert_rgb(test, rig.centre(&gpu), [0.0, 0.0, 0.0]);
}
