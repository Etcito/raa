//! S12 motion from the prepass. The CPU half mirrors the WGSL formula against
//! `tests/fixtures/motion.txt`; the software-GPU half drives the public renderer and
//! pipeline and reads the Rg16Float motion target back through a compute copy.

use raa_shader::PASS_PREPASS_WGSL;

/// One fixture case: jittered clip positions and validity inputs plus the expected UV
/// displacement.
struct Case {
    name: String,
    current_clip: [f32; 3],
    previous_clip: [f32; 3],
    current_jitter: [f32; 2],
    previous_jitter: [f32; 2],
    history_valid: bool,
    instance_valid: bool,
    expected: [f32; 2],
}

fn number(token: &str, line: &str) -> f32 {
    token
        .parse()
        .unwrap_or_else(|_| panic!("`{token}` is not a number in `{line}`"))
}

fn parse_case(line: &str) -> Case {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    assert_eq!(tokens.len(), 15, "fixture columns in `{line}`");
    let f = |i: usize| number(tokens[i], line);
    Case {
        name: tokens[0].to_string(),
        current_clip: [f(1), f(2), f(3)],
        previous_clip: [f(4), f(5), f(6)],
        current_jitter: [f(7), f(8)],
        previous_jitter: [f(9), f(10)],
        history_valid: f(11) != 0.0,
        instance_valid: f(12) != 0.0,
        expected: [f(13), f(14)],
    }
}

/// True for a finite f32, the same guard the WGSL uses.
fn finite(v: f32) -> bool {
    (v - v) == 0.0
}

/// CPU twin of `raa_pass_motion`: remove both jitters, then encode the NDC delta as a UV
/// displacement with positive X right and positive Y down.
fn motion(case: &Case) -> [f32; 2] {
    if !case.history_valid || !case.instance_valid {
        return [0.0; 2];
    }
    let (cw, pw) = (case.current_clip[2], case.previous_clip[2]);
    if !(cw > 0.0) || !(pw > 0.0) {
        return [0.0; 2];
    }
    let current = [
        case.current_clip[0] / cw - case.current_jitter[0],
        case.current_clip[1] / cw - case.current_jitter[1],
    ];
    let previous = [
        case.previous_clip[0] / pw - case.previous_jitter[0],
        case.previous_clip[1] / pw - case.previous_jitter[1],
    ];
    if !current.iter().chain(previous.iter()).all(|&v| finite(v)) {
        return [0.0; 2];
    }
    [
        (current[0] - previous[0]) * 0.5,
        (previous[1] - current[1]) * 0.5,
    ]
}

#[test]
fn fixture_right_down_camera_and_invalid_cases() {
    let fixture = include_str!("fixtures/motion.txt");
    let mut names = Vec::new();
    for line in fixture.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let case = parse_case(line);
        let got = motion(&case);
        assert!(
            (got[0] - case.expected[0]).abs() <= 1e-6 && (got[1] - case.expected[1]).abs() <= 1e-6,
            "{}: got {got:?}, expected {:?}",
            case.name,
            case.expected
        );
        names.push(case.name);
    }
    for required in [
        "right",
        "down",
        "up",
        "camera_right",
        "jitter_only",
        "zero_motion",
        "invalid_history",
        "new_instance",
        "invalid_current_w",
        "invalid_previous_w",
        "not_finite",
    ] {
        assert!(names.iter().any(|n| n == required), "missing {required}");
    }
}

#[test]
fn prepass_source_writes_location_two_after_normal_and_material() {
    let source = PASS_PREPASS_WGSL;
    for text in [
        "@location(0) normal: vec4<f32>",
        "@location(1) material: vec4<f32>",
        "@location(2) motion: vec2<f32>",
        "out.motion = raa_pass_motion(v);",
        "raa_frame.unjittered_view_proj",
        "raa_frame.previous_view_proj",
        "raa_frame.camera_jitter.zw",
        "raa_frame.temporal.z",
        "previous.previous_valid",
        "RAA_FLAG_VERTEX_FN",
        "(current_ndc.x - previous_ndc.x) * 0.5",
        "(previous_ndc.y - current_ndc.y) * 0.5",
    ] {
        assert!(source.contains(text), "prepass is missing `{text}`");
    }
    assert!(source.contains("fn vs_main("), "vs_main");
    assert!(source.contains("fn fs_main("), "fs_main");
}

#[cfg(not(target_arch = "wasm32"))]
mod software_gpu_motion {
    use std::cell::RefCell;
    use std::rc::Rc;

    use raa_core::testing::{block_on, skip, software_gpu};
    use raa_core::{FrameGraph, GpuContext, MaterialId, MeshId, OffscreenTarget, wgpu};
    use raa_geom::builtin::DebugCube;
    use raa_geom::{GeometryProvider, MeshData, ParamValue, Params, Quality};
    use raa_lit::{AttributedMesh, LitOptions, LitRenderer};
    use raa_math::{Quat, Vec3};
    use raa_pipeline::Pipeline;
    use raa_scene::{NodeId, RenderList, Scene, Transform};

    const SIZE: (u32, u32) = (64, 64);
    const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    const EPS: f32 = 1e-4;

    /// Copies the motion target into a mappable storage buffer with textureLoad. The S12
    /// target itself only carries RENDER_ATTACHMENT | TEXTURE_BINDING, so it is never a
    /// copy source.
    const READBACK: &str = "
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<vec2<f32>>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(src);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    out[id.y * size.x + id.x] = textureLoad(src, vec2<i32>(i32(id.x), i32(id.y)), 0).xy;
}
";

    /// Reads an Rg16Float texture as two f32 lanes per texel, top row first.
    fn read_motion(gpu: &GpuContext, texture: &wgpu::Texture) -> Vec<[f32; 2]> {
        let device = gpu.device();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("motion.readback"),
            source: wgpu::ShaderSource::Wgsl(READBACK.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("motion.readback"),
            layout: None,
            module: &module,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let layout = pipeline.get_bind_group_layout(0);
        let bytes = u64::from(SIZE.0) * u64::from(SIZE.1) * 8;
        let storage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("motion.readback.storage"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("motion.readback.map"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("motion.readback"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: storage.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("motion.readback"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("motion.readback"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(SIZE.0.div_ceil(8), SIZE.1.div_ceil(8), 1);
        }
        encoder.copy_buffer_to_buffer(&storage, 0, &readback, 0, bytes);
        gpu.queue().submit([encoder.finish()]);
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
        rx.recv().expect("map callback").expect("map");
        let mapped = slice.get_mapped_range().expect("mapped range");
        let texels = mapped
            .chunks_exact(8)
            .map(|px| {
                [
                    f32::from_le_bytes([px[0], px[1], px[2], px[3]]),
                    f32::from_le_bytes([px[4], px[5], px[6], px[7]]),
                ]
            })
            .collect();
        drop(mapped);
        readback.unmap();
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

    /// A pipeline with a lit renderer (MSAA 4) and one scene, run without tone mapping.
    struct Rig {
        pipeline: Rc<RefCell<Pipeline>>,
        lit: LitRenderer,
        scene: Scene,
        list: RenderList,
        tags: Vec<u32>,
        slots: Vec<u32>,
        graph: FrameGraph,
        backbuffer: OffscreenTarget,
        mesh: MeshId,
        node: Option<NodeId>,
    }

    impl Rig {
        fn new(gpu: &GpuContext) -> Rig {
            let pipeline = Rc::new(RefCell::new(Pipeline::new(gpu, SURFACE, false)));
            let mut lit =
                LitRenderer::new(gpu, &pipeline, LitOptions::default()).expect("lit renderer");
            let id = lit.upload(gpu, &cube()).expect("upload");
            let mesh = lit.submesh_ids(id).expect("sub-meshes")[0];
            let mut scene = Scene::with_capacity(1, 1);
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
                slots: Vec::with_capacity(1),
                graph,
                backbuffer: OffscreenTarget::new(gpu, SIZE.0, SIZE.1, SURFACE),
                mesh,
                node: None,
            }
        }

        fn add_cube(&mut self) {
            let node = self
                .scene
                .add_node(None, Transform::IDENTITY)
                .expect("node");
            self.scene
                .add_instance(node, self.mesh, MaterialId(0))
                .expect("instance");
            self.node = Some(node);
        }

        fn move_cube(&mut self, x: f32) {
            let t = Transform::from_trs(Vec3::new(x, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE);
            self.scene
                .set_local(self.node.expect("cube"), t)
                .expect("local");
        }

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

        fn motion(&self, gpu: &GpuContext) -> Vec<[f32; 2]> {
            let p = self.pipeline.borrow();
            let targets = p.targets().expect("targets after a frame");
            assert_eq!(targets.size(), SIZE);
            read_motion(gpu, targets.motion_texture())
        }
    }

    fn all_zero(motion: &[[f32; 2]]) -> bool {
        motion
            .iter()
            .all(|m| m[0].abs() <= EPS && m[1].abs() <= EPS)
    }

    #[test]
    fn motion_target_gets_signed_displacement_from_the_prepass() {
        let test = "motion_target_gets_signed_displacement_from_the_prepass";
        let Some(gpu) = software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
        let mut rig = Rig::new(&gpu);

        // No geometry: the prepass clears motion to zero.
        rig.frame(&gpu);
        assert!(all_zero(&rig.motion(&gpu)), "empty scene");

        // A newly introduced instance has no submitted history: zero motion.
        rig.add_cube();
        rig.frame(&gpu);
        assert!(all_zero(&rig.motion(&gpu)), "new instance");

        // A settled frame is zero once history exists.
        rig.frame(&gpu);
        assert!(all_zero(&rig.motion(&gpu)), "settled frame");

        // The cube moves +X: its covered pixels move right (positive motion.x).
        rig.move_cube(0.5);
        rig.frame(&gpu);
        let moving = rig.motion(&gpu);
        let max_x = moving.iter().map(|m| m[0]).fold(f32::MIN, f32::max);
        assert!(max_x > 0.01, "rightward motion {max_x}");
        assert!(
            moving.iter().all(|m| m[1].abs() <= 0.01),
            "no vertical motion"
        );

        // The camera moves +X instead: the cube shifts left (negative motion.x).
        rig.scene.camera.eye = Vec3::new(0.5, 2.0, 6.0);
        rig.scene.camera.target = Vec3::new(0.5, 0.0, 0.0);
        rig.frame(&gpu);
        let camera = rig.motion(&gpu);
        let min_x = camera.iter().map(|m| m[0]).fold(f32::MAX, f32::min);
        assert!(min_x < -0.01, "camera motion {min_x}");

        // Camera and geometry both settle: zero again.
        rig.frame(&gpu);
        assert!(all_zero(&rig.motion(&gpu)), "settled camera");

        let error = block_on(errors.pop());
        assert!(error.is_none(), "GPU validation error: {error:?}");
    }
}
