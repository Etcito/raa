//! Q8: the extension crate adds a geometry provider and a render pass from outside RAA.
//! The provider registers next to the 8 prims and loads through PrimsLoader; the pass
//! orders after `forward` and tints the backbuffer (GPU test skips without an adapter).

use raa_core::wgpu;
use raa_core::{BACKBUFFER, CLEAR_COLOR, ClearPass, FrameGraph};
use raa_ext_test::{StarProvider, TINT, TintPass};
use raa_geom::analysis::{analyze, passes};
use raa_geom::{GeometryRegistry, MeshData, ParamValue, Params, Quality};
use raa_loader::prims::PrimsLoader;
use raa_loader::{ColliderShape, LoadContext, LoaderRegistry, NoBooleans};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

fn registry() -> GeometryRegistry {
    let mut g = GeometryRegistry::new();
    raa_prims::register_all(&mut g).expect("register_all");
    g.register(Box::new(StarProvider)).expect("star registers");
    g
}

fn star_text() -> String {
    concat!(
        "{\n",
        "  \"format\": \"raa.prims\",\n",
        "  \"version\": 0,\n",
        "  \"id\": \"star-asset\",\n",
        "  \"name\": \"Star\",\n",
        "  \"nodes\": [\n",
        "    {\n",
        "      \"id\": \"star\",\n",
        "      \"kind\": \"star\",\n",
        "      \"params\": {\n",
        "        \"points\": 5,\n",
        "        \"outerRadius\": 0.8,\n",
        "        \"innerRadius\": 0.35,\n",
        "        \"halfHeight\": 0.2\n",
        "      }\n",
        "    }\n",
        "  ],\n",
        "  \"materials\": {},\n",
        "  \"anchors\": {}\n",
        "}\n",
    )
    .to_string()
}

fn loaders() -> LoaderRegistry {
    let mut l = LoaderRegistry::new();
    l.register(Box::new(PrimsLoader))
        .expect("PrimsLoader registers");
    l
}

fn ctx(g: &GeometryRegistry, strict_kinds: bool) -> LoadContext<'_> {
    LoadContext {
        geometry: g,
        booleans: &NoBooleans,
        quality: Quality::LOD0,
        strict_kinds,
    }
}

#[test]
fn star_registers_next_to_the_eight_prims_and_passes_analysis() {
    let g = registry();
    let kinds = g.kinds();
    assert_eq!(kinds.len(), 9, "kinds: {kinds:?}");
    assert_eq!(kinds.last().copied(), Some("star"));

    let params = Params::new()
        .with("points", ParamValue::Scalar(5.0))
        .with("outerRadius", ParamValue::Scalar(0.8))
        .with("innerRadius", ParamValue::Scalar(0.35))
        .with("halfHeight", ParamValue::Scalar(0.2));
    let mut mesh = MeshData::new();
    for q in [Quality::LOW, Quality::LOD0, Quality::HIGH] {
        g.generate("star", &params, q, &mut mesh)
            .unwrap_or_else(|e| panic!("star at {}: {e}", q.segments));
        let report = analyze(&mesh);
        if let Err(msg) = passes(&report) {
            panic!("star at {} fails analysis: {msg}", q.segments);
        }
    }
}

#[test]
fn loader_accepts_star_when_kinds_are_not_strict() {
    let g = registry();
    let asset = loaders()
        .load(star_text().as_bytes(), &ctx(&g, false))
        .unwrap_or_else(|e| panic!("star asset loads: {e}"));
    assert_eq!(asset.meshes.len(), 1);
    assert_eq!(asset.meshes[0].kind, "star");
    assert_eq!(asset.meshes[0].mesh.triangle_count(), 40);
    assert_eq!(asset.colliders.len(), 1);
    assert!(
        matches!(asset.colliders[0].shape, ColliderShape::ConvexHull { .. }),
        "collider {:?}",
        asset.colliders[0].shape
    );
}

#[test]
fn loader_refuses_star_when_kinds_are_strict() {
    let g = registry();
    let e = loaders()
        .load(star_text().as_bytes(), &ctx(&g, true))
        .expect_err("strict kinds refuse star");
    assert_eq!(e.code, "kind.unknown", "{e}");
    assert!(!e.message.is_empty());
}

#[test]
fn tint_pass_runs_after_forward_in_the_graph_cpu() {
    let mut graph = FrameGraph::new(FORMAT);
    graph.add_pass(Box::new(ClearPass::new("forward", BACKBUFFER, CLEAR_COLOR)));
    graph
        .add_pass_after("forward", Box::new(TintPass::new()))
        .expect("forward exists");
    graph.compile().expect("graph compiles");
    assert_eq!(graph.pass_names(), vec!["forward", "ext-tint"]);

    let e = graph
        .add_pass_after("missing", Box::new(TintPass::new()))
        .expect_err("unknown pass refused");
    assert_eq!(e.code(), "framegraph.unknown_pass");
}

/// Exact sRGB OETF of a linear value in [0, 1], as an 8-bit channel.
fn srgb_u8(linear: f64) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn tint_pass_after_renderer_install_gpu() {
    use raa_core::testing::{skip, software_gpu};
    use raa_core::{OffscreenTarget, non_empty};
    use raa_render::{Renderer, RendererOptions};
    use raa_scene::{RenderList, Scene};

    const NAME: &str = "tint_pass_after_renderer_install_gpu";
    const SIZE: u32 = 64;
    let Some(gpu) = software_gpu() else {
        skip(
            NAME,
            "software_gpu returned None; see the software_gpu line above",
        );
        return;
    };

    let mut renderer =
        Renderer::new(&gpu, FORMAT, true, RendererOptions::default()).expect("renderer");
    let mut graph = FrameGraph::new(FORMAT);
    renderer.install(&mut graph);
    graph
        .add_pass_after("forward", Box::new(TintPass::new()))
        .expect("forward exists");
    graph.compile().expect("graph compiles");
    assert_eq!(graph.pass_names(), vec!["forward", "ext-tint"]);

    let mut scene = Scene::new();
    let mut list = RenderList::with_capacity(1);
    renderer
        .prepare(&gpu, &mut scene, &mut list)
        .expect("prepare succeeds");

    let target = OffscreenTarget::new(&gpu, SIZE, SIZE, FORMAT);
    let commands = graph
        .execute(&gpu, target.view(), (SIZE, SIZE))
        .expect("graph executes");
    gpu.queue().submit([commands]);
    let pixels = target.read_rgba8(&gpu).expect("readback succeeds");
    assert_eq!(pixels.len(), (SIZE * SIZE * 4) as usize);

    let clear = [CLEAR_COLOR.r, CLEAR_COLOR.g, CLEAR_COLOR.b];
    let clear_rgba8 = [srgb_u8(clear[0]), srgb_u8(clear[1]), srgb_u8(clear[2]), 255];
    assert!(non_empty(&pixels, clear_rgba8), "tint not visible");

    let a = f64::from(TINT[3]);
    let mut expected = [0u8; 3];
    for (i, e) in expected.iter_mut().enumerate() {
        *e = srgb_u8(clear[i] + (f64::from(TINT[i]) - clear[i]) * a);
    }
    for (x, y) in [(SIZE / 2, SIZE / 2), (0, 0)] {
        let i = ((y * SIZE + x) * 4) as usize;
        let px = &pixels[i..i + 3];
        let ok = px.iter().zip(expected).all(|(p, e)| p.abs_diff(e) <= 2);
        assert!(ok, "pixel ({x}, {y}) {px:?} != expected {expected:?} (+-2)");
    }
    eprintln!(
        "{NAME}: adapter `{}` ({}), expected {expected:?}",
        gpu.summary().device,
        if gpu.summary().is_hardware() {
            "hardware"
        } else {
            "software"
        }
    );

    drop(renderer);
    // Same WSL driver-teardown workaround as raa-render's gpu_forward: leak the GPU objects
    // once all GPU work has completed.
    std::mem::forget(graph);
    std::mem::forget(target);
    std::mem::forget(gpu);
}
