//! The bounce of the views pair (examples/render-graph/scenes/views.js), measured end to end on a
//! software GPU (SKIP without an adapter): the GI volume voxelizes the sunlit red wall, `gi.screen`
//! gathers its bounce, and the forward pass puts that bounce onto the surfaces beside the wall.
//!
//! This is a measurement with a qualitative assertion, not a golden image: it asserts only points
//! that the views camera sees and the sun does not light, and it prints the GI state so that a
//! failure can be placed. It changes no raa-gi code; it records that the engine GI carries the red
//! bounce at the page's scale.
//!
//! What is reproduced, from views.js and lib/scene.js:
//! - the floor: 20 x 20 m with its top at y = 0, linear grey 0.5;
//! - the red left wall: x in [-5.1, -4.9], y in [0, 4], z in [-5, 10], albedo the linear value
//!   of sRGB (0.6, 0.1, 0.1);
//! - the white back wall: x in [-5, 10], y in [0, 4], z in [-5.1, -4.9], albedo the linear value
//!   of sRGB (0.9, 0.9, 0.88);
//! - the bounce box: centre (-3.9, 0.6, 0.6), half extent 0.6, the same albedo as the back wall;
//! - the sun of `bounceSunDirection()` (elevation 35 and azimuth 110 degrees), colour
//!   [1, 0.96, 0.9], intensity 3;
//! - the camera of `cameraAt(RG_CAPTURE_T_S)`: eye (9 sin 1, 3, 9 cos 1), target (0, 1, 0), 50
//!   degrees vertical field of view, near 0.1 m, far 200 m, at 16:9.
//!
//! The page passes sRGB colours to the JS API, which converts them to linear: the albedos here are
//! those linear values (`srgb_to_linear`). Ambient is 0. No sky is installed (the env cube is the
//! pipeline dummy). The GI settings are the Full GI starter's: `giVolume` resolution 96, levels 5,
//! probe rays 8, intensity 1, environment diffuse 0, local light shadows and voxel full relight on,
//! voxel rays 3; `giScreen` probes with screen traces and reflections. SSGI and gtao are not
//! installed, so S10 binding 7 holds the 1x1 dummy, as in red_bounce.rs.
//!
//! Every shared input is set up here: the lit prepass writes depth and the S12 normal target from
//! the four boxes, and the HDR chain is written by forward. The frame loop runs 240 frames (the
//! page's RG_WARMUP_FRAMES) at the one camera, so the volume and the probes settle before the read.
//!
//! Spatial extent: level 0 has 0.2 m voxels, so a 96-voxel level spans 19.2 m. The test prints each
//! level's origin, voxel size and span, and whether the red wall and the bounce box lie inside it.
//!
//! The asserted property is qualitative. In the forward HDR, red must exceed 0.01 and 1.3 times the
//! larger of green and blue at each asserted point. The `gi.screen` output (the published `gi` tap,
//! read by a compute pass, because a tap is a view) is printed beside it: that is the indirect light
//! alone.

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::testing::{skip, software_gpu};
use raa_core::{FrameGraph, GpuContext, MaterialId, OffscreenTarget};
use raa_gi::clipmap;
use raa_gi::{DebugView, GiFeature, GiQualityPatch, GiSettingsPatch, IndirectMode};
use raa_lit::{AttributedMesh, LitOptions, LitRenderer, SurfaceMaterial};
use raa_math::{Quat, Vec3, Vec4};
use raa_pipeline::Pipeline;
use raa_scene::{Camera, DirectionalLight, RenderList, Scene, Transform};

const SIZE: (u32, u32) = (320, 180);
const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// RG_WARMUP_FRAMES of the page: 240 frames before the captured one.
const FRAMES: usize = 240;
/// Full GI starter: `giVolume.resolution` 96 and `levels` 5.
const RESOLUTION: u32 = 96;
const LEVELS: u32 = 5;
/// Full GI starter `giVolume.probeRays` 8.
const PROBE_RAYS: f64 = 8.0;

/// A 1 m cube centred at the origin: four vertices per face (hard edges, outward normals) and
/// 12 counter-clockwise triangles; scaled per instance into the floor, the walls and the box.
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
        label: Some("views_bounce.readback"),
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

const READ_VIEW_WGSL: &str = r#"
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> texels: array<vec4<f32>>;

@compute @workgroup_size(8, 8)
fn read_texels(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(source);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    texels[id.y * size.x + id.x] = textureLoad(source, vec2<i32>(id.xy), 0);
}
"#;

/// Reads a published 2D float view into rows of linear rgba. A tap is a `TextureView`, which no
/// copy command can read, so a compute pass loads every texel into a storage buffer.
fn read_view(gpu: &GpuContext, view: &wgpu::TextureView, width: u32, height: u32) -> Vec<[f32; 4]> {
    let device = gpu.device();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("views_bounce.read_view"),
        source: wgpu::ShaderSource::Wgsl(READ_VIEW_WGSL.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("views_bounce.read_view"),
        layout: None,
        module: &module,
        entry_point: Some("read_texels"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let bytes = u64::from(width) * u64::from(height) * 16;
    let storage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("views_bounce.texels"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("views_bounce.texels.readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("views_bounce.read_view"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: storage.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }
    encoder.copy_buffer_to_buffer(&storage, 0, &staging, 0, bytes);
    gpu.queue().submit([encoder.finish()]);
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");
    let data = staging
        .slice(..)
        .get_mapped_range()
        .expect("mapped readback")
        .to_vec();
    data.as_chunks::<16>()
        .0
        .iter()
        .map(|texel| {
            let channel = |i: usize| {
                f32::from_le_bytes([
                    texel[4 * i],
                    texel[4 * i + 1],
                    texel[4 * i + 2],
                    texel[4 * i + 3],
                ])
            };
            [channel(0), channel(1), channel(2), channel(3)]
        })
        .collect()
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

/// The sRGB transfer curve, as the JS API applies to material colours.
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// `bounceSunDirection()` of views.js: the sun's travel direction at elevation 35 and azimuth 110
/// degrees from +Z.
fn bounce_sun_direction() -> Vec3 {
    let elevation = 35.0_f32.to_radians();
    let azimuth = 110.0_f32.to_radians();
    Vec3::new(
        -azimuth.sin() * elevation.cos(),
        -elevation.sin(),
        -azimuth.cos() * elevation.cos(),
    )
}

/// The views camera: `cameraAt(RG_CAPTURE_T_S)` of lib/scene.js, at 16:9.
fn views_camera() -> Camera {
    let a = 0.6_f32 + 0.1 * 4.0;
    let mut camera = Camera {
        eye: Vec3::new(9.0 * a.sin(), 3.0, 9.0 * a.cos()),
        target: Vec3::new(0.0, 1.0, 0.0),
        fov_y_rad: 50.0_f32.to_radians(),
        near: 0.1,
        far: 200.0,
        ..Camera::default()
    };
    camera.set_viewport(SIZE.0, SIZE.1);
    camera
}

/// Whether the box `min`..`max` lies inside the box `lo`..`hi`.
fn inside(lo: [f32; 3], hi: [f32; 3], min: [f32; 3], max: [f32; 3]) -> bool {
    (0..3).all(|axis| min[axis] >= lo[axis] && max[axis] <= hi[axis])
}

#[test]
fn views_back_wall_corner_and_box_face_receive_red_indirect_light() {
    let test = "views_back_wall_corner_and_box_face_receive_red_indirect_light";
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
                roughness: 0.8,
                ..SurfaceMaterial::default()
            })
            .expect("material")
    };
    let floor = material([0.5, 0.5, 0.5]);
    let red = material([0.6, 0.1, 0.1].map(srgb_to_linear));
    let white = material([0.9, 0.9, 0.88].map(srgb_to_linear));

    let mut scene = Scene::with_capacity(4, 4);
    add_box(
        &mut scene,
        mesh,
        Vec3::new(0.0, -0.2, 0.0),
        Vec3::new(20.0, 0.4, 20.0),
        floor,
    );
    add_box(
        &mut scene,
        mesh,
        Vec3::new(-5.0, 2.0, 2.5),
        Vec3::new(0.2, 4.0, 15.0),
        red,
    );
    add_box(
        &mut scene,
        mesh,
        Vec3::new(2.5, 2.0, -5.0),
        Vec3::new(15.0, 4.0, 0.2),
        white,
    );
    let bounce_center = Vec3::new(-3.9, 0.6, 0.6);
    add_box(
        &mut scene,
        mesh,
        bounce_center,
        Vec3::new(1.2, 1.2, 1.2),
        white,
    );
    let camera = views_camera();
    scene.camera = camera;
    scene.ambient.intensity = 0.0;
    let sun = DirectionalLight {
        direction: bounce_sun_direction(),
        color: [1.0, 0.96, 0.9],
        intensity: 3.0,
    };
    scene.light = sun;
    handle.set_sun(&sun);
    handle
        .set(GiSettingsPatch {
            enabled: Some(true),
            intensity: Some(1.0),
            environment_diffuse: Some(0.0),
            local_light_shadows: Some(true),
            reflections: Some(true),
            screen_traces: Some(true),
            indirect: Some(IndirectMode::Probes),
            debug_view: Some(DebugView::None),
            history_max: Some(128.0),
            history_min: Some(1.0),
            clamp_gamma: Some(3.0),
            spatial_passes: Some(4.0),
            restir_m_cap: Some(30.0),
            voxel_full_relight: Some(true),
            voxel_history_max: Some(64.0),
            voxel_rays: Some(3.0),
            ..GiSettingsPatch::default()
        })
        .expect("enabled");
    handle
        .set_quality(GiQualityPatch {
            resolution: Some(f64::from(RESOLUTION)),
            levels: Some(f64::from(LEVELS)),
            probe_rays: Some(PROBE_RAYS),
        })
        .expect("96/5/8 is a listed quality");

    let mut graph = FrameGraph::new(SURFACE);
    Pipeline::install(&pipeline, &mut graph);
    graph.compile().expect("compile");
    let backbuffer = OffscreenTarget::new(&gpu, SIZE.0, SIZE.1, SURFACE);
    let (mut list, mut tags, mut slots) = (RenderList::with_capacity(4), Vec::new(), Vec::new());
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
    println!("GI stats: {stats:?}");
    assert!(stats.active, "{stats:?}");
    assert_eq!(stats.voxelized_forward_instances, 4, "{stats:?}");
    assert!(!stats.pending_rebuild, "{stats:?}");

    // The clipmap: each level's origin snaps to the camera (feature.rs), so the walls and the box
    // are checked against each level's grid. Printed only: at this camera level 0 holds none of
    // them, and level 1 holds all of them.
    let boxes = [
        ("red wall", [-5.1, 0.0, -5.0], [-4.9, 4.0, 10.0]),
        ("back wall", [-5.0, 0.0, -5.1], [10.0, 4.0, -4.9]),
        ("bounce box", [-4.5, 0.0, 0.0], [-3.3, 1.2, 1.2]),
    ];
    for level in 0..2 {
        let voxel = clipmap::voxel_size(level);
        let origin = clipmap::snapped_origin(camera.eye, voxel, RESOLUTION);
        let lo = origin.map(|o| o as f32 * voxel);
        let hi = origin.map(|o| (o as f32 + RESOLUTION as f32) * voxel);
        println!(
            "level {level}: origin {origin:?} voxels, voxel size {voxel} m, spans {lo:?} .. {hi:?}"
        );
        for (name, min, max) in boxes {
            println!(
                "  {name} inside level {level}: {}",
                inside(lo, hi, min, max)
            );
        }
    }

    let (hdr, tap_width, tap_height, tap) = {
        let pipeline = pipeline.borrow();
        let targets = pipeline.targets().expect("targets after a frame");
        let hdr = read_rgba16(&gpu, targets.hdr().src_texture());
        let published = pipeline
            .resources()
            .get("gi.screen", "gi")
            .expect("gi.screen publishes its gi output");
        println!("gi.screen gi tap: {:?}", published.desc);
        let texels = read_view(
            &gpu,
            &published.view,
            published.desc.width,
            published.desc.height,
        );
        (hdr, published.desc.width, published.desc.height, texels)
    };
    // (pixel, forward HDR rgb, gi.screen rgb) of a world point through the views camera.
    let sample = |point: Vec3| {
        let (x, y) = pixel_of(&camera, point);
        let [r, g, b, _] = hdr[(y * SIZE.0 + x) as usize];
        let (tx, ty) = (x * tap_width / SIZE.0, y * tap_height / SIZE.1);
        let [gr, gg, gb, _] = tap[(ty * tap_width + tx) as usize];
        ((x, y), [r, g, b], [gr, gg, gb])
    };

    // Asserted, each lit by the bounce alone. The back wall's front face faces +z, away from the
    // sun, so the sun does not reach its corner beside the red wall. The bounce box's +z face also
    // faces away from the sun. The three wall points are 0.3, 0.4 and 0.6 m from the red wall's
    // inner face (x = -4.9) and 0.05 m in front of the wall.
    //
    // Not asserted: the brief's three floor points beside the red wall (x in -4.6 .. -4.3, z in
    // 0.6 .. 0.9). From this camera the bounce box hides them, and they lie in the box's sun shadow.
    // That shadow needs the sun's shadow map, which this test does not install, so the test cannot
    // cast it. The points asserted here stand in for the floor.
    let asserted = [
        ("back wall corner", Vec3::new(-4.6, 0.5, -4.85)),
        ("back wall corner", Vec3::new(-4.5, 1.0, -4.85)),
        ("back wall corner", Vec3::new(-4.3, 0.5, -4.85)),
        (
            "bounce box +z face",
            bounce_center + Vec3::new(0.0, 0.0, 0.6),
        ),
    ];
    let mut failures = Vec::new();
    for (name, point) in asserted {
        let (pixel, [r, g, b], [gr, gg, gb]) = sample(point);
        println!(
            "{name} {point:?} at pixel {pixel:?}: HDR rgb {r:.4}, {g:.4}, {b:.4}, red over \
             max(green, blue) {:.3}; gi.screen rgb {gr:.4}, {gg:.4}, {gb:.4}",
            r / g.max(b)
        );
        assert!(
            r > 0.01,
            "no red indirect light on the {name} at {point:?}, pixel {pixel:?}: rgb {r}, {g}, {b}"
        );
        if r <= 1.3 * g.max(b) {
            failures.push(format!(
                "{name} {point:?} pixel {pixel:?}: rgb {r}, {g}, {b}"
            ));
        }
    }
    assert!(failures.is_empty(), "not lit red by the wall: {failures:?}");
}
