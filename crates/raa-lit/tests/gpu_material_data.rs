//! Material data words on a software GPU (SKIP without an adapter): one cube drawn with an
//! unlit user material function that reads its base colour from the material's data block
//! (`raa_material_data_base(in.record)`, `raa_material_f32`), the HDR source read back at the
//! centre. One function shades any block: the material only changes its `data` base; a
//! block that outgrows the buffer rebinds group 0; a removed block reads zero.

#![cfg(not(target_arch = "wasm32"))]

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::testing::{block_on, skip, software_gpu};
use raa_core::{FrameGraph, GpuContext, MaterialId, OffscreenTarget, wgpu};
use raa_geom::builtin::DebugCube;
use raa_geom::{GeometryProvider, MeshData, ParamValue, Params, Quality};
use raa_lit::{
    AttributedMesh, FLAG_UNLIT, LitOptions, LitRenderer, MATERIAL_DATA_INITIAL_WORDS,
    MATERIAL_DATA_RESERVED_WORDS, MaterialFnId, SurfaceMaterial, SurfaceModel,
};
use raa_pipeline::Pipeline;
use raa_scene::{RenderList, Scene, Transform};

const SIZE: (u32, u32) = (32, 32);
const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Base colour = words base + 1, + 2, + 3 as f32 (word base + 0 is a u32 that scales them).
const READER: &str = "fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    let b = raa_material_data_base(in.record);
    let k = f32(raa_material_word(b));
    o.base_color = k * vec3<f32>(raa_material_f32(b + 1u), raa_material_f32(b + 2u), raa_material_f32(b + 3u));
    o.emissive = vec3<f32>(0.0);
    return o;
}
";

fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((bits >> 10) & 0x1f);
    let mant = f32::from(bits & 0x3ff);
    if exp == 0 {
        return sign * mant * 2f32.powi(-24);
    }
    sign * (1.0 + mant / 1024.0) * 2f32.powi(exp - 15)
}

fn centre_rgb(gpu: &GpuContext, texture: &wgpu::Texture) -> [f32; 3] {
    let row = SIZE.0 as usize * 8;
    let padded = row.div_ceil(256) * 256;
    let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_material_data.readback"),
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

struct Rig {
    pipeline: Rc<RefCell<Pipeline>>,
    lit: LitRenderer,
    scene: Scene,
    graph: FrameGraph,
    backbuffer: OffscreenTarget,
    fid: MaterialFnId,
}

impl Rig {
    fn new(gpu: &GpuContext) -> Rig {
        let pipeline = Rc::new(RefCell::new(Pipeline::new(gpu, SURFACE, false)));
        let mut lit = LitRenderer::new(gpu, &pipeline, LitOptions::default()).expect("lit");
        let fid = lit
            .register_user_fn("data_reader", READER)
            .expect("register");
        let material = lit
            .materials_mut()
            .add(Self::material(fid, 0))
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
            fid,
        }
    }

    fn material(fid: MaterialFnId, data: u32) -> SurfaceMaterial {
        SurfaceMaterial {
            model: SurfaceModel::Function(fid),
            flags: FLAG_UNLIT,
            data,
            ..SurfaceMaterial::default()
        }
    }

    /// Points material 1 at the data block `base`.
    fn use_data(&mut self, base: u32) {
        let m = Self::material(self.fid, base);
        self.lit
            .materials_mut()
            .set(MaterialId(1), m)
            .expect("set material");
    }

    fn centre(&mut self, gpu: &GpuContext) -> [f32; 3] {
        let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
        let (mut list, mut tags, mut slots) =
            (RenderList::with_capacity(1), Vec::new(), Vec::new());
        let counts = self
            .lit
            .prepare(gpu, &mut self.scene, &mut list, &mut tags, &mut slots)
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

fn block(k: u32, rgb: [f32; 3]) -> Vec<u32> {
    let mut w = vec![k];
    w.extend(rgb.iter().map(|x| x.to_bits()));
    w
}

#[test]
fn one_function_shades_from_each_materials_data_block() {
    let test = "one_function_shades_from_each_materials_data_block";
    let Some(gpu) = software_gpu() else {
        return skip(test, "software_gpu returned None");
    };
    let mut rig = Rig::new(&gpu);
    // No data: base 0, the reserved zero words.
    assert_rgb(test, rig.centre(&gpu), [0.0, 0.0, 0.0]);
    let a = rig
        .lit
        .add_material_data(&gpu, &block(1, [0.25, 0.5, 0.125]))
        .unwrap();
    let b = rig
        .lit
        .add_material_data(&gpu, &block(2, [0.125, 0.0625, 0.25]))
        .unwrap();
    assert_eq!(
        (a, b),
        (
            MATERIAL_DATA_RESERVED_WORDS,
            MATERIAL_DATA_RESERVED_WORDS + 4
        )
    );
    rig.use_data(a);
    assert_rgb(test, rig.centre(&gpu), [0.25, 0.5, 0.125]);
    rig.use_data(b);
    assert_rgb(test, rig.centre(&gpu), [0.25, 0.125, 0.5]);
    // Overwritten in place; another length is refused and changes nothing.
    rig.lit
        .set_material_data(&gpu, b, &block(1, [0.5, 0.5, 0.5]))
        .unwrap();
    assert_rgb(test, rig.centre(&gpu), [0.5, 0.5, 0.5]);
    let e = rig.lit.set_material_data(&gpu, b, &[1, 2]).unwrap_err();
    assert_eq!(e.code(), "material.data_invalid");
    assert_eq!(
        rig.lit.add_material_data(&gpu, &[]).unwrap_err().code(),
        "material.data_invalid"
    );
    assert_eq!(rig.lit.material_data(b), Some(block(1, [0.5, 0.5, 0.5])));
    // A block past the initial capacity grows the buffer (group 0 bound again); the earlier
    // blocks are still there.
    let mut big = vec![0u32; MATERIAL_DATA_INITIAL_WORDS as usize];
    big[..4].copy_from_slice(&block(1, [0.0, 0.75, 0.25]));
    let c = rig.lit.add_material_data(&gpu, &big).unwrap();
    let stats = rig.lit.material_data_stats();
    assert!(
        stats.capacity >= c + MATERIAL_DATA_INITIAL_WORDS,
        "{stats:?}"
    );
    assert_eq!(stats.blocks, 3);
    rig.use_data(c);
    assert_rgb(test, rig.centre(&gpu), [0.0, 0.75, 0.25]);
    rig.use_data(a);
    assert_rgb(test, rig.centre(&gpu), [0.25, 0.5, 0.125]);
    // Removed: its words read zero; a second removal does nothing.
    assert!(rig.lit.remove_material_data(&gpu, a).unwrap());
    assert!(!rig.lit.remove_material_data(&gpu, a).unwrap());
    assert_rgb(test, rig.centre(&gpu), [0.0, 0.0, 0.0]);
    assert_eq!(rig.lit.material_data(a), None);
}
