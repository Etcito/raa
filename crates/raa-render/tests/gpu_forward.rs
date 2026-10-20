//! Native GPU test of the forward renderer: one DebugCube into an OffscreenTarget.
//! Skips with the SKIP line when no software adapter is available.

#![cfg(not(target_arch = "wasm32"))]

use raa_core::testing::{skip, software_gpu};
use raa_core::{FrameGraph, GpuContext, MeshId, OffscreenTarget, non_empty};
use raa_geom::builtin::DebugCube;
use raa_geom::{GeometryProvider, MeshData, ParamValue, Params, Quality};
use raa_material::Material;
use raa_math::{Quat, Vec3};
use raa_render::{Renderer, RendererOptions};
use raa_scene::{RenderList, Scene, Transform};

const SIZE: u32 = 128;
/// CLEAR_COLOR written as-is into an Rgba8Unorm target.
const CLEAR_RGBA8: [u8; 4] = [13, 15, 20, 255];

fn pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * SIZE + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
}

fn within(a: [u8; 4], b: [u8; 4], tol: u8) -> bool {
    a.iter().zip(b.iter()).all(|(x, y)| x.abs_diff(*y) <= tol)
}

fn adapter_line(test: &str, gpu: &GpuContext) {
    let s = gpu.summary();
    eprintln!(
        "{test}: adapter `{}` ({}, {})",
        s.device,
        s.backend,
        if s.is_hardware() {
            "hardware"
        } else {
            "software"
        }
    );
}

fn debug_cube() -> MeshData {
    let params = Params::new().with("half", ParamValue::Scalar(0.5));
    let mut mesh = MeshData::new();
    DebugCube
        .generate(&params, Quality::LOD0, &mut mesh)
        .expect("debug cube generates");
    mesh
}

fn scene_with_instance(mesh: MeshId, material: raa_core::MaterialId) -> Scene {
    let mut scene = Scene::new();
    let rotation = Quat::from_rotation_y(30.0_f32.to_radians());
    let node = scene
        .add_node(None, Transform::from_trs(Vec3::ZERO, rotation, Vec3::ONE))
        .expect("node added");
    scene
        .add_instance(node, mesh, material)
        .expect("instance added");
    scene.camera.eye = Vec3::new(1.5, 1.5, 3.0);
    scene.camera.target = Vec3::ZERO;
    scene.camera.set_viewport(SIZE, SIZE);
    scene
}

#[test]
fn forward_renders_a_lit_debug_cube_into_an_offscreen_target() {
    const NAME: &str = "forward_renders_a_lit_debug_cube_into_an_offscreen_target";
    let Some(gpu) = software_gpu() else {
        skip(
            NAME,
            "software_gpu returned None; see the software_gpu line above",
        );
        return;
    };
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let target = OffscreenTarget::new(&gpu, SIZE, SIZE, format);
    let mut graph = FrameGraph::new(format);
    let mut renderer =
        Renderer::new(&gpu, format, false, RendererOptions::default()).expect("renderer");
    renderer.install(&mut graph);
    graph.compile().expect("graph compiles");

    let mesh = renderer
        .upload_mesh(&gpu, &debug_cube())
        .expect("mesh uploads");
    let material = renderer
        .materials_mut()
        .add(Material::pbr([0.8, 0.3, 0.2], 0.4, 0.0))
        .expect("material added");
    let mut scene = scene_with_instance(mesh, material);
    let mut list = RenderList::with_capacity(4);

    let counts = renderer
        .prepare(&gpu, &mut scene, &mut list)
        .expect("prepare succeeds");
    let commands = graph
        .execute(&gpu, target.view(), (SIZE, SIZE))
        .expect("graph executes");
    gpu.queue().submit([commands]);
    let pixels = target.read_rgba8(&gpu).expect("readback succeeds");

    assert_eq!(pixels.len(), (SIZE * SIZE * 4) as usize);
    assert!(non_empty(&pixels, CLEAR_RGBA8), "nothing drawn");
    let centre = pixel(&pixels, SIZE / 2, SIZE / 2);
    let corner = pixel(&pixels, 0, 0);
    assert!(
        !within(centre, CLEAR_RGBA8, 2),
        "centre {centre:?} is the clear colour"
    );
    assert!(
        within(corner, CLEAR_RGBA8, 2),
        "corner {corner:?} is not the clear colour {CLEAR_RGBA8:?}"
    );
    assert_eq!(counts.instances, 1);
    assert_eq!(counts.draw_calls, 1);
    assert_eq!(counts.triangles, 12);
    adapter_line(NAME, &gpu);
    eprintln!("{NAME}: centre {centre:?}, corner {corner:?}, counts {counts:?}");

    drop(renderer);
    // Tearing down a wgpu instance that enumerated Mesa's D3D12 GL adapter (WSL) segfaults
    // inside the driver. Leak the GPU objects at the end of the test (as m1-boot does);
    // all GPU work has completed by now.
    std::mem::forget(graph);
    std::mem::forget(target);
    std::mem::forget(gpu);
}

#[test]
fn unknown_mesh_is_refused() {
    const NAME: &str = "unknown_mesh_is_refused";
    let Some(gpu) = software_gpu() else {
        skip(
            NAME,
            "software_gpu returned None; see the software_gpu line above",
        );
        return;
    };
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let mut graph = FrameGraph::new(format);
    let mut renderer =
        Renderer::new(&gpu, format, false, RendererOptions::default()).expect("renderer");
    renderer.install(&mut graph);
    graph.compile().expect("graph compiles");

    let mut scene = scene_with_instance(MeshId(99), raa_core::MaterialId(0));
    let mut list = RenderList::with_capacity(4);
    let err = renderer
        .prepare(&gpu, &mut scene, &mut list)
        .expect_err("unknown mesh must be refused");
    assert_eq!(err.code(), "core.resource");
    // Display prefixes the `core.resource` code; the message carries the render code.
    let text = err.message();
    assert!(
        text.starts_with("render.unknown_mesh"),
        "message `{text}` does not start with render.unknown_mesh"
    );
    adapter_line(NAME, &gpu);

    drop(renderer);
    // See the first test: leak the GPU objects to avoid the WSL driver teardown crash.
    std::mem::forget(graph);
    std::mem::forget(gpu);
}
