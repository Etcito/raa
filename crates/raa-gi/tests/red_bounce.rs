//! Red indirect light next to a red-albedo voxel, end to end on a software GPU (SKIP without an
//! adapter): the GI volume voxelizes a sunlit red wall, `gi.screen` gathers its bounce, and the
//! `gi` providers of the `Ao` and `Indirect` slots put that bounce onto the white floor beside it
//! in the forward pass.
//!
//! The pipeline is the lit renderer's (the required `prepass` and `forward`) plus the GI feature
//! (`gi.volume`, `gi.screen`, `gi.history`); no other feature is installed. Every shared input is
//! set up here:
//! - depth and the S12 normal target are written by the lit prepass from the two boxes below (it
//!   clears depth to 1.0 = far, so the pixels outside the boxes are sky pixels the passes skip);
//!   the HDR chain is cleared and written by forward, and `gi.history` copies it for the next
//!   frame's screen traces;
//! - no IBL is installed, so the env cube is the pipeline dummy (a dim grey-blue that GI misses
//!   read, neutral rather than red), and the scene's flat ambient is set to 0, so the floor's
//!   forward colour away from direct light is the `Indirect` term alone;
//! - gtao is not installed, so S10 binding 7 is the 1x1 dummy of 1. The `gi` Ao provider loaded
//!   it at the fragment's pixel without clamping, an out-of-bounds load that robust backends
//!   (this software adapter among them) answer with 0. `min(s.ao, c.ao)` was then 0 and the
//!   floor showed no bounce at all: this test failed with R = G = B = 0 at every floor pixel.
//! - the sun travels along (-1, 0.1, 0), so it shines on the wall's inner face (+x) and not on
//!   the floor (+y faces away from it, in the forward pass and in the voxel lighting alike).
//!
//! Spatial extent: at quality 64/3, level 0 spans 64 x 0.2 m = 12.8 m per axis around the
//! camera's snapped origin. For the eye (1.5, 1.6, 3.0) that origin is (-32, -24, -24) voxels, so
//! level 0 covers x in [-6.4, 6.4], y in [-4.8, 8.0] and z in [-4.8, 8.0] m: the camera, the floor
//! (x, z in [-3, 3], y in [-0.4, 0]) and the wall (x in [-1.5, -1.1], y in [0, 2], z in [-2, 2])
//! all lie inside it, and every probe ray toward the wall stays inside it.
//!
//! The asserted property is qualitative, as S37 asks of GPU results (no reference image): the
//! floor beside the wall carries indirect light whose red exceeds 1.5 times its green and blue
//! (the misses add the dummy env's neutral grey-blue on top of the red bounce).

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::testing::{skip, software_gpu};
use raa_core::{FrameGraph, GpuContext, MaterialId, OffscreenTarget};
use raa_gi::{GiFeature, GiQualityPatch, GiSettingsPatch};
use raa_lit::{AttributedMesh, LitOptions, LitRenderer, SurfaceMaterial};
use raa_math::{Quat, Vec3, Vec4};
use raa_pipeline::Pipeline;
use raa_scene::{Camera, DirectionalLight, RenderList, Scene, Transform};

const SIZE: (u32, u32) = (128, 72);
const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// Frames rendered before the read: the first allocates and builds every level, the relight
/// schedule [0, 1, 0, 2] of 64/3 then updates each level-0 voxel every fourth frame
/// (checkerboard), and the screen probes blend in their history.
const FRAMES: usize = 40;

/// A 1 m cube centred at the origin: four vertices per face (hard edges, outward normals) and
/// 12 counter-clockwise triangles; scaled per instance into the floor and the wall.
fn cube() -> AttributedMesh {
    let mut mesh = AttributedMesh::from_mesh(Default::default());
    for axis in 0..3 {
        for sign in [-0.5_f32, 0.5] {
            let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
            let mut normal = [0.0_f32; 3];
            normal[axis] = sign.signum();
            let first = mesh.base.positions.len() as u32;
            for (a, b) in [(-0.5_f32, -0.5_f32), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)] {
                let mut p = [0.0_f32; 3];
                p[axis] = sign;
                p[u] = a;
                p[v] = b;
                mesh.base.positions.push(p);
                mesh.base.normals.push(normal);
            }
            let quad = if sign > 0.0 {
                [0u32, 1, 2, 0, 2, 3]
            } else {
                [0u32, 2, 1, 0, 3, 2]
            };
            mesh.base.indices.extend(quad.map(|i| first + i));
        }
    }
    mesh.submeshes[0].index_count = mesh.base.indices.len() as u32;
    mesh
}

/// IEEE half to f32 (the HDR chain is Rgba16Float).
fn half_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    match exponent {
        0 => sign * mantissa * 2f32.powi(-24),
        31 => f32::NAN,
        _ => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15),
    }
}

/// Reads an Rgba16Float 2D texture (COPY_SRC) into rows of linear rgba.
fn read_rgba16(gpu: &GpuContext, texture: &wgpu::Texture) -> Vec<[f32; 4]> {
    let (width, height) = (texture.width(), texture.height());
    let row = (width * 8).div_ceil(256) * 256;
    let staging = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("red_bounce.readback"),
        size: u64::from(row * height),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device().create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue().submit([encoder.finish()]);
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");
    let data = staging
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback")
        .to_vec();
    let mut texels = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let offset = (y * row + x * 8) as usize;
            let channel = |i: usize| {
                half_to_f32(u16::from_le_bytes([
                    data[offset + 2 * i],
                    data[offset + 2 * i + 1],
                ]))
            };
            texels.push([channel(0), channel(1), channel(2), channel(3)]);
        }
    }
    texels
}

/// The pixel of a world point through `camera` (column-major view-projection, y down in pixels).
fn pixel_of(camera: &Camera, point: Vec3) -> (u32, u32) {
    let clip = camera
        .view_proj()
        .mul_vec4(Vec4::new(point.x, point.y, point.z, 1.0));
    let (x, y) = (clip.x / clip.w, clip.y / clip.w);
    let px = ((x * 0.5 + 0.5) * SIZE.0 as f32) as u32;
    let py = ((0.5 - y * 0.5) * SIZE.1 as f32) as u32;
    (px.min(SIZE.0 - 1), py.min(SIZE.1 - 1))
}

/// Adds one instance of the unit cube scaled to `size` and centred at `center`.
fn add_box(
    scene: &mut Scene,
    mesh: raa_core::MeshId,
    center: Vec3,
    size: Vec3,
    material: MaterialId,
) {
    let node = scene
        .add_node(None, Transform::from_trs(center, Quat::IDENTITY, size))
        .expect("node");
    scene.add_instance(node, mesh, material).expect("instance");
}

#[test]
fn floor_beside_a_sunlit_red_wall_receives_red_indirect_light() {
    let test = "floor_beside_a_sunlit_red_wall_receives_red_indirect_light";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, SURFACE, false)));
    let options = LitOptions {
        msaa: 4,
        ..LitOptions::default()
    };
    let mut lit = LitRenderer::new(&gpu, &pipeline, options).expect("lit renderer");
    let handle =
        GiFeature::install(&gpu, &pipeline, &mut lit, None, None).expect("the feature installs");
    let id = lit.upload(&gpu, &cube()).expect("upload");
    let mesh = lit.submesh_ids(id).expect("sub-meshes")[0];
    let mut material = |base_color: [f32; 3]| {
        lit.materials_mut()
            .add(SurfaceMaterial {
                base_color,
                roughness: 0.9,
                ..SurfaceMaterial::default()
            })
            .expect("material")
    };
    // Linear albedos: the red wall reflects red only; the floor is a neutral white.
    let red = material([0.7, 0.02, 0.02]);
    let white = material([0.8, 0.8, 0.8]);

    let mut scene = Scene::with_capacity(2, 2);
    add_box(
        &mut scene,
        mesh,
        Vec3::new(0.0, -0.2, 0.0),
        Vec3::new(6.0, 0.4, 6.0),
        white,
    );
    add_box(
        &mut scene,
        mesh,
        Vec3::new(-1.3, 1.0, 0.0),
        Vec3::new(0.4, 2.0, 4.0),
        red,
    );
    let mut camera = Camera {
        eye: Vec3::new(1.5, 1.6, 3.0),
        target: Vec3::new(-0.8, 0.2, -0.5),
        near: 0.1,
        far: 30.0,
        ..Camera::default()
    };
    camera.set_viewport(SIZE.0, SIZE.1);
    scene.camera = camera;
    scene.ambient.intensity = 0.0;
    let sun = DirectionalLight {
        direction: Vec3::new(-1.0, 0.1, 0.0).normalize_or_zero(),
        color: [1.0, 1.0, 1.0],
        intensity: 4.0,
    };
    scene.light = sun;
    handle.set_sun(&sun);
    handle
        .set(GiSettingsPatch {
            enabled: Some(true),
            ..GiSettingsPatch::default()
        })
        .expect("enabled");
    handle
        .set_quality(GiQualityPatch {
            resolution: Some(64.0),
            levels: Some(3.0),
            probe_rays: Some(8.0),
        })
        .expect("64/3/8 is a listed quality");

    let mut graph = FrameGraph::new(SURFACE);
    Pipeline::install(&pipeline, &mut graph);
    graph.compile().expect("compile");
    let backbuffer = OffscreenTarget::new(&gpu, SIZE.0, SIZE.1, SURFACE);
    let (mut list, mut tags, mut slots) = (RenderList::with_capacity(2), Vec::new(), Vec::new());
    let mut stats = None;
    for _ in 0..FRAMES {
        handle.set_view(&camera, SIZE, 1);
        stats = Some(handle.prepare_frame(&gpu, &mut scene, &mut lit));
        lit.prepare(&gpu, &mut scene, &mut list, &mut tags, &mut slots)
            .expect("lit prepare");
        let commands = graph
            .execute(&gpu, backbuffer.view(), SIZE)
            .expect("execute");
        gpu.queue().submit([commands]);
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
    }
    let stats = stats.expect("frames ran");
    assert!(stats.active, "{stats:?}");
    assert_eq!(stats.voxelized_forward_instances, 2, "{stats:?}");
    assert!(!stats.pending_rebuild, "{stats:?}");

    let hdr = {
        let pipeline = pipeline.borrow();
        let targets = pipeline.targets().expect("targets after a frame");
        read_rgba16(&gpu, targets.hdr().src_texture())
    };
    // Floor points 0.3 to 0.5 m from the wall's inner face (x = -1.1), out of the sun.
    for point in [
        Vec3::new(-0.8, 0.0, 0.0),
        Vec3::new(-0.7, 0.0, 0.6),
        Vec3::new(-0.6, 0.0, -0.6),
    ] {
        let (x, y) = pixel_of(&camera, point);
        let [r, g, b, _] = hdr[(y * SIZE.0 + x) as usize];
        println!("floor {point:?} at pixel ({x}, {y}): rgb {r:.4}, {g:.4}, {b:.4}");
        assert!(
            r > 0.01,
            "no indirect light on the floor at {point:?}, pixel ({x}, {y}): rgb {r}, {g}, {b}"
        );
        assert!(
            r > 1.5 * g.max(b),
            "the floor at {point:?}, pixel ({x}, {y}) is not lit red by the wall: rgb {r}, {g}, {b}"
        );
    }
}
