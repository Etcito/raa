//! A thin-walled box with one green wall, end to end on a software GPU (SKIP without an adapter):
//! the box of the GI page's Cornell rows at their x10 scale (canonical millimetres / 100), whose
//! walls are single quads backed outside by 0.8 m slabs of the wall's colour, lit by one white
//! area light 0.4 m under the ceiling (CORNELL_LIGHT_Y 5.088 of
//! examples/global-illumination/scenes/lighting.js). The GI volume voxelizes the box, and the
//! `gi` providers of the `Indirect`, `Ao` and `LocalVisibility` slots shade the forward pass.
//! Three qualitative properties of the failed lighting-stage judgment (card 8045) are checked,
//! with no reference image (S37: GPU results are judged, never compared to a reference):
//! - the indirect light on the floor beside the green wall (GI on minus GI off; the white light
//!   and the white floor make the direct term neutral) has a green channel above its red;
//! - the front face of the green and red walls' backing, just outside the box at its open front,
//!   sees only the black environment and receives no indirect light. The screen integration once
//!   averaged probes on other surfaces into such pixels, an orange glow around the Cornell box;
//! - points the light reaches unobstructed keep their direct light with GI on. The volume
//!   visibility of the area light once offset its rays by 1.5 finest voxels while the box lies in
//!   the 0.4 m level, so grazing rays shadowed their own wall in contour blotches.
//!
//! The pipeline is the lit renderer's (`prepass` and `forward`), the light tiling (`light.tiles`,
//! so the forward pass sees the area light) and the GI feature (`gi.volume`, `gi.screen`,
//! `gi.history`); nothing else is installed. Every shared input is set up here:
//! - depth and the S12 normal target come from the lit prepass of the meshes below; it clears
//!   depth to 1.0 = far, so the pixels outside the box are sky pixels every gi pass skips; the HDR
//!   chain is cleared and written by forward, and `gi.history` copies it for the next frame's
//!   screen traces;
//! - the env cube is a black 1x1 cube (`LightingResources::set_env`), so a ray that misses the
//!   volume brings no light; the sun has intensity 0 and the flat ambient is 0, so every light in
//!   the image comes from the area light, directly or through GI;
//! - gtao is not installed, so S10 binding 7 is the dummy of 1.
//!
//! Spatial extent, at the default quality 96/5 the page uses: level i has voxels of 0.2 x 2^i m
//! and spans 96 of them around the camera's snapped origin. For the eye (2.78, 2.73, -8.0), level
//! 0 covers z in [-17.6, 1.6] and level 1 covers x in [-19.2, 19.2], y in [-19.2, 19.2] and z in
//! [-28.8, 9.6] m. The camera and the whole box with its slabs (x in [-0.8, 6.36], y in [-0.8,
//! 6.29], z in [0, 6.39]) lie in level 1, the box's back two thirds only there: the setting of the
//! judged image, where the walls are traced in 0.4 m voxels.

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::testing::{skip, software_gpu};
use raa_core::{FrameGraph, GpuContext, OffscreenTarget};
use raa_gi::{GiFeature, GiQualityPatch, GiSettingsPatch};
use raa_lit::{
    AttributedMesh, LitOptions, LitRenderer, LocalLight, LocalLightKind, SurfaceMaterial,
    install_local_lights,
};
use raa_math::{Quat, Vec3, Vec4};
use raa_pipeline::Pipeline;
use raa_scene::{Camera, DirectionalLight, RenderList, Scene, Transform};

const SIZE: (u32, u32) = (640, 360);
const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// Frames before the read: the first allocates and builds every level, the 96/5 relight
/// schedule [0, 1, 0, 2, 0, 3, 0, 4] updates each level-1 voxel every 16th frame (checkerboard),
/// and the screen probes blend in their history.
const FRAMES: usize = 64;
/// Linear albedos of the canonical Cornell box, the white made neutral.
const WHITE: [f32; 3] = [0.8, 0.8, 0.8];
const RED: [f32; 3] = [0.570, 0.043, 0.044];
const GREEN: [f32; 3] = [0.105, 0.378, 0.076];
/// The light's centre height of the GI page (CORNELL_LIGHT_Y, 0.4 m under the ceiling) and the
/// canonical one (1 mm under it).
const LIGHT_Y: f32 = 5.088;
const CANONICAL_LIGHT_Y: f32 = 5.487;
/// The room's centre: every quad faces it.
const CENTRE: [f32; 3] = [2.78, 2.744, 2.796];

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Appends one quad as two triangles, its normal and winding turned towards `towards` (the
/// `quad` of scenes/lighting.js).
fn add_quad(mesh: &mut AttributedMesh, corners: [[f32; 3]; 4], towards: [f32; 3]) {
    let mid = [0, 1, 2].map(|i| corners.iter().map(|c| c[i]).sum::<f32>() / 4.0);
    let mut normal = cross(sub(corners[1], corners[0]), sub(corners[2], corners[0]));
    let length = dot(normal, normal).sqrt();
    normal = normal.map(|v| v / length);
    let flip = dot(normal, sub(towards, mid)) < 0.0;
    if flip {
        normal = normal.map(|v| -v);
    }
    let first = mesh.base.positions.len() as u32;
    for corner in corners {
        mesh.base.positions.push(corner);
        mesh.base.normals.push(normal);
    }
    let indices = if flip {
        [0u32, 2, 1, 0, 3, 2]
    } else {
        [0u32, 1, 2, 0, 2, 3]
    };
    mesh.base.indices.extend(indices.map(|i| first + i));
}

/// Appends an axis-aligned box (centre, half extents) with outward faces.
fn add_box(mesh: &mut AttributedMesh, center: [f32; 3], half: [f32; 3]) {
    for axis in 0..3 {
        for sign in [-1.0_f32, 1.0] {
            let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
            let mut corners = [center; 4];
            for (corner, (a, b)) in
                corners
                    .iter_mut()
                    .zip([(-1.0_f32, -1.0_f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)])
            {
                corner[axis] += sign * half[axis];
                corner[u] += a * half[u];
                corner[v] += b * half[v];
            }
            let mut outside = center;
            outside[axis] += 2.0 * sign * half[axis];
            add_quad(mesh, corners, outside);
        }
    }
}

/// Appends a block: its top quad at `height` and four sides dropping each top edge to y 0.
fn add_block(mesh: &mut AttributedMesh, top: [[f32; 3]; 4], height: f32) {
    let cx = top.iter().map(|p| p[0]).sum::<f32>() / 4.0;
    let cz = top.iter().map(|p| p[2]).sum::<f32>() / 4.0;
    let raised = top.map(|p| [p[0], height, p[2]]);
    add_quad(mesh, raised, [cx, height + 1.0, cz]);
    for i in 0..4 {
        let (p, q) = (raised[i], raised[(i + 1) % 4]);
        let outside = [p[0] + q[0] - cx, height / 2.0, p[2] + q[2] - cz];
        add_quad(mesh, [p, q, [q[0], 0.0, q[2]], [p[0], 0.0, p[2]]], outside);
    }
}

fn empty() -> AttributedMesh {
    AttributedMesh::from_mesh(Default::default())
}

fn finished(mut mesh: AttributedMesh) -> AttributedMesh {
    mesh.submeshes[0].index_count = mesh.base.indices.len() as u32;
    mesh
}

/// The box's three meshes with their albedos: white (floor, ceiling, back wall, both blocks and
/// their slabs), red (the wall at x about 5.5 and its slab) and green (the wall at x 0 and its
/// slab). The slabs are 0.8 m thick, outside the box, and the front (z 0) is open. The tall
/// block's corner x 3.14 m is canonical Cornell data (314 mm), not an approximation of pi.
#[allow(clippy::approx_constant)]
fn cornell_meshes() -> [(AttributedMesh, [f32; 3]); 3] {
    let mut white = empty();
    add_quad(
        &mut white,
        [
            [5.528, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 5.592],
            [5.496, 0.0, 5.592],
        ],
        CENTRE,
    );
    add_quad(
        &mut white,
        [
            [5.56, 5.488, 0.0],
            [5.56, 5.488, 5.592],
            [0.0, 5.488, 5.592],
            [0.0, 5.488, 0.0],
        ],
        CENTRE,
    );
    add_quad(
        &mut white,
        [
            [5.496, 0.0, 5.592],
            [0.0, 0.0, 5.592],
            [0.0, 5.488, 5.592],
            [5.56, 5.488, 5.592],
        ],
        CENTRE,
    );
    add_block(
        &mut white,
        [
            [1.30, 1.65, 0.65],
            [0.82, 1.65, 2.25],
            [2.40, 1.65, 2.72],
            [2.90, 1.65, 1.14],
        ],
        1.65,
    );
    add_block(
        &mut white,
        [
            [4.23, 3.30, 2.47],
            [2.65, 3.30, 2.96],
            [3.14, 3.30, 4.56],
            [4.72, 3.30, 4.06],
        ],
        3.30,
    );
    add_box(&mut white, [2.78, -0.4, 2.796], [2.78, 0.4, 2.796]);
    add_box(&mut white, [2.78, 5.888, 2.796], [2.78, 0.4, 2.796]);
    add_box(&mut white, [2.78, 2.744, 5.992], [2.78, 2.744, 0.4]);
    let mut red = empty();
    add_quad(
        &mut red,
        [
            [5.528, 0.0, 0.0],
            [5.496, 0.0, 5.592],
            [5.56, 5.488, 5.592],
            [5.56, 5.488, 0.0],
        ],
        CENTRE,
    );
    add_box(&mut red, [5.96, 2.744, 2.796], [0.4, 2.744, 2.796]);
    let mut green = empty();
    add_quad(
        &mut green,
        [
            [0.0, 0.0, 5.592],
            [0.0, 0.0, 0.0],
            [0.0, 5.488, 0.0],
            [0.0, 5.488, 5.592],
        ],
        CENTRE,
    );
    add_box(&mut green, [-0.4, 2.744, 2.796], [0.4, 2.744, 2.796]);
    [
        (finished(white), WHITE),
        (finished(red), RED),
        (finished(green), GREEN),
    ]
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
        label: Some("thin_wall_bleed.readback"),
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

/// The pixel of a world point through `camera`, or None outside the frame or behind the eye.
fn pixel_of(camera: &Camera, point: Vec3) -> Option<(u32, u32)> {
    let clip = camera
        .view_proj()
        .mul_vec4(Vec4::new(point.x, point.y, point.z, 1.0));
    if clip.w <= 0.0 {
        return None;
    }
    let (x, y) = (clip.x / clip.w, clip.y / clip.w);
    let px = (x * 0.5 + 0.5) * SIZE.0 as f32;
    let py = (0.5 - y * 0.5) * SIZE.1 as f32;
    let inside = (0.0..SIZE.0 as f32).contains(&px) && (0.0..SIZE.1 as f32).contains(&py);
    inside.then_some((px as u32, py as u32))
}

/// The HDR texel of a world point, with the point's name in the panic message when it is out of
/// the frame.
fn texel(hdr: &[[f32; 4]], camera: &Camera, name: &str, point: Vec3) -> [f32; 3] {
    let (x, y) =
        pixel_of(camera, point).unwrap_or_else(|| panic!("{name} {point:?} is out of frame"));
    let [r, g, b, _] = hdr[(y * SIZE.0 + x) as usize];
    println!("{name} {point:?} at pixel ({x}, {y}): rgb {r:.5}, {g:.5}, {b:.5}");
    [r, g, b]
}

/// The camera of both renders: the Cornell camera, outside the open front, looking along +z.
fn camera() -> Camera {
    let mut camera = Camera {
        eye: Vec3::new(2.78, 2.73, -8.0),
        target: Vec3::new(2.78, 2.73, 0.0),
        fov_y_rad: 39.3_f32.to_radians(),
        near: 0.05,
        far: 200.0,
        ..Camera::default()
    };
    camera.set_viewport(SIZE.0, SIZE.1);
    camera
}

/// Renders the box for `FRAMES` frames with GI enabled or disabled and the light's centre at
/// `light_y`, and returns the HDR (linear rgba per pixel, row-major).
fn render(gpu: &GpuContext, gi_enabled: bool, light_y: f32) -> Vec<[f32; 4]> {
    let pipeline = Rc::new(RefCell::new(Pipeline::new(gpu, SURFACE, false)));
    let options = LitOptions {
        msaa: 1,
        ..LitOptions::default()
    };
    let mut lit = LitRenderer::new(gpu, &pipeline, options).expect("lit renderer");
    install_local_lights(gpu, &pipeline, &mut lit).expect("light tiling installs");
    let handle =
        GiFeature::install(gpu, &pipeline, &mut lit, None, None).expect("the feature installs");

    let mut scene = Scene::with_capacity(3, 3);
    for (mesh, base_color) in cornell_meshes() {
        let id = lit.upload(gpu, &mesh).expect("upload");
        let submesh = lit.submesh_ids(id).expect("sub-meshes")[0];
        let material = lit
            .materials_mut()
            .add(SurfaceMaterial {
                base_color,
                roughness: 1.0,
                ..SurfaceMaterial::default()
            })
            .expect("material");
        let node = scene
            .add_node(
                None,
                Transform::from_trs(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE),
            )
            .expect("node");
        scene
            .add_instance(node, submesh, material)
            .expect("instance");
    }
    // The only light: the Cornell area light (1.30 m along x, 1.05 m along z, radiance x area),
    // white, emitting downward, its centre at `light_y`.
    lit.local_lights_mut()
        .add(LocalLight {
            kind: LocalLightKind::Area,
            position: [2.78, light_y, 2.795],
            direction: [0.0, -1.0, 0.0],
            color: [1.0, 1.0, 1.0],
            intensity: 18.4 * 1.365,
            range: 30.0,
            size: [1.05, 1.30],
            ..LocalLight::default()
        })
        .expect("area light");

    let camera = camera();
    scene.camera = camera;
    scene.ambient.intensity = 0.0;
    let sun = DirectionalLight {
        direction: Vec3::new(0.0, -1.0, 0.0),
        color: [1.0, 1.0, 1.0],
        intensity: 0.0,
    };
    scene.light = sun;
    handle.set_sun(&sun);
    handle
        .set(GiSettingsPatch {
            enabled: Some(gi_enabled),
            ..GiSettingsPatch::default()
        })
        .expect("settings");
    handle
        .set_quality(GiQualityPatch {
            resolution: Some(96.0),
            levels: Some(5.0),
            probe_rays: Some(8.0),
        })
        .expect("96/5/8 is a listed quality");

    // A black environment: a ray that misses the volume brings no light.
    let black = gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("thin_wall_bleed.black_env"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let black_view = black.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });
    pipeline
        .borrow_mut()
        .lighting_mut()
        .set_env(black_view, [[0.0; 4]; 9], [0.0; 4]);

    let mut graph = FrameGraph::new(SURFACE);
    Pipeline::install(&pipeline, &mut graph);
    graph.compile().expect("compile");
    let backbuffer = OffscreenTarget::new(gpu, SIZE.0, SIZE.1, SURFACE);
    let (mut list, mut tags, mut slots) = (RenderList::with_capacity(3), Vec::new(), Vec::new());
    let mut stats = None;
    for _ in 0..FRAMES {
        handle.set_view(&camera, SIZE, 1);
        handle.set_local_light_count(lit.local_lights().active_count());
        stats = Some(handle.prepare_frame(gpu, &mut scene, &mut lit));
        lit.prepare(gpu, &mut scene, &mut list, &mut tags, &mut slots)
            .expect("lit prepare");
        let commands = graph
            .execute(gpu, backbuffer.view(), SIZE)
            .expect("execute");
        gpu.queue().submit([commands]);
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
    }
    let stats = stats.expect("frames ran");
    assert_eq!(stats.active, gi_enabled, "{stats:?}");
    if gi_enabled {
        assert_eq!(stats.voxelized_forward_instances, 3, "{stats:?}");
        assert!(!stats.pending_rebuild, "{stats:?}");
    }
    let pipeline = pipeline.borrow();
    let targets = pipeline.targets().expect("targets after a frame");
    read_rgba16(gpu, targets.hdr().src_texture())
}

#[test]
fn green_wall_bleeds_green_and_no_light_leaks_outside_the_box() {
    let test = "green_wall_bleeds_green_and_no_light_leaks_outside_the_box";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let on = render(&gpu, true, LIGHT_Y);
    let off = render(&gpu, false, LIGHT_Y);
    let camera = camera();

    // The floor beside the green wall, lit directly by the white light (neutral on the white
    // floor) and indirectly by the green wall and the white room: its indirect light (GI on
    // minus GI off) is green above red.
    let mut inside_green = f32::MAX;
    for point in [
        Vec3::new(0.3, 0.0, 3.6),
        Vec3::new(0.25, 0.0, 3.9),
        Vec3::new(0.6, 0.0, 3.2),
    ] {
        let lit_on = texel(&on, &camera, "floor beside the green wall", point);
        let lit_off = texel(&off, &camera, "floor beside the green wall (GI off)", point);
        let indirect = [0, 1, 2].map(|c| lit_on[c] - lit_off[c]);
        println!("indirect at {point:?}: {indirect:?}");
        assert!(
            indirect[1] > 1.0e-3,
            "no indirect light on the floor beside the green wall at {point:?}: {indirect:?}"
        );
        assert!(
            indirect[1] > indirect[0],
            "the indirect light beside the green wall is not green at {point:?}: {indirect:?}"
        );
        inside_green = inside_green.min(indirect[1]);
    }

    // The front faces of the green and red walls' slabs (z 0), 0.08 m outside the walls' front
    // edges: they face the black environment, so with GI on they stay black beside the lit room.
    for (name, point) in [
        ("green slab front", Vec3::new(-0.08, 1.4, 0.0)),
        ("green slab front", Vec3::new(-0.08, 2.7, 0.0)),
        ("green slab front", Vec3::new(-0.08, 4.4, 0.0)),
        ("red slab front", Vec3::new(5.64, 1.4, 0.0)),
        ("red slab front", Vec3::new(5.64, 2.7, 0.0)),
        ("red slab front", Vec3::new(5.64, 4.4, 0.0)),
    ] {
        let [r, g, b] = texel(&on, &camera, name, point);
        let leak = r.max(g).max(b);
        assert!(
            leak <= 0.02 * inside_green,
            "indirect light reaches the {name} outside the box at {point:?}: rgb {r}, {g}, {b} \
             (the green bounce inside is at least {inside_green})"
        );
    }

    // Surfaces the light reaches unobstructed keep their direct light with GI on, with the light
    // at the canonical 5.487 (1 mm under the ceiling), where the volume visibility shadowed most
    // of the walls and the floor before its offsets followed the level the box is traced in.
    let on = render(&gpu, true, CANONICAL_LIGHT_Y);
    let off = render(&gpu, false, CANONICAL_LIGHT_Y);
    for (name, point) in [
        ("green wall", Vec3::new(0.0, 1.5, 3.0)),
        ("green wall", Vec3::new(0.0, 2.5, 4.4)),
        ("red wall", Vec3::new(5.53, 2.0, 1.2)),
        ("back wall", Vec3::new(1.4, 1.2, 5.592)),
        ("back wall", Vec3::new(2.4, 2.4, 5.592)),
        ("floor", Vec3::new(4.5, 0.0, 1.0)),
    ] {
        let lit_on = texel(&on, &camera, name, point);
        let lit_off = texel(&off, &camera, &format!("{name} (GI off)"), point);
        for channel in 0..3 {
            assert!(
                lit_on[channel] >= 0.9 * lit_off[channel],
                "the {name} at {point:?} lost its direct light with GI on: {lit_on:?} against \
                 {lit_off:?} with GI off"
            );
        }
    }
}
