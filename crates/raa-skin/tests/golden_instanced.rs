//! Golden: instanced vertex-stage skinning against the CPU reference, through the real
//! lit pipeline (prepass + forward, HDR target, software GPU).
//!
//! Two instances of one skinned tube (two joints, blended weights along its height) are
//! posed differently and drawn by the new path: one shared joint palette, ONE instanced
//! draw, zero compute dispatches. The reference renders the same scene with two static
//! meshes whose vertices were skinned on the CPU (`skin_cpu`: the S18 formula the old
//! compute kernel matched within `SKIN_PARITY_EPS`). The HDR texels must agree within a
//! small tolerance, for three pose pairs.

#![cfg(not(target_arch = "wasm32"))]

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::testing::{block_on, skip, software_gpu};
use raa_core::{FrameGraph, GpuContext, MaterialId, OffscreenTarget, RaaError, wgpu};
use raa_lit::{AttributedMesh, LitOptions, LitRenderer, RestMeshView};
use raa_math::{Mat4, Quat, Vec3};
use raa_pipeline::{FeaturePass, LightingResources, Pipeline, RecordCtx};
use raa_scene::{RenderList, Scene, Transform};
use raa_skin::{SkinMesh, SkinPass, SkinVertexInputs, skin_cpu};

const SIZE: (u32, u32) = (128, 128);
const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const SEGMENTS: usize = 12;
const RINGS: usize = 9;

/// A capped-free tube along +Y (radius 0.2, y in -0.45..0.45): joint 0 at the bottom,
/// joint 1 at the top, weights blending linearly with height.
fn tube() -> SkinMesh {
    let (mut p, mut n, mut uv, mut idx) = (vec![], vec![], vec![], vec![]);
    let (mut joints, mut weights) = (vec![], vec![]);
    for r in 0..RINGS {
        let t = r as f32 / (RINGS - 1) as f32;
        let y = -0.45 + 0.9 * t;
        for s in 0..=SEGMENTS {
            let a = s as f32 / SEGMENTS as f32 * std::f32::consts::TAU;
            let (x, z) = (a.cos(), a.sin());
            p.extend_from_slice(&[0.2 * x, y, 0.2 * z]);
            n.extend_from_slice(&[x, 0.0, z]);
            uv.extend_from_slice(&[s as f32 / SEGMENTS as f32, t]);
            joints.extend_from_slice(&[0u16, 1, 0, 0]);
            weights.extend_from_slice(&[1.0 - t, t, 0.0, 0.0]);
        }
    }
    let row = (SEGMENTS + 1) as u32;
    for r in 0..(RINGS - 1) as u32 {
        for s in 0..SEGMENTS as u32 {
            let (a, b) = (r * row + s, (r + 1) * row + s);
            idx.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    let rest = raa_mesh::from_js_arrays(&p, &n, Some(&uv), None, None, &idx, &[]).unwrap();
    SkinMesh::new(rest, &joints, &weights).unwrap()
}

/// Skin matrix rotating by `angle` about +Z around `pivot`, then translating by `t`.
fn joint(pivot: Vec3, angle: f32, t: Vec3) -> Mat4 {
    let r = Quat::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), angle);
    let to = Mat4::from_trs(pivot + t, r, Vec3::ONE);
    let back = Mat4::from_trs(
        Vec3::new(-pivot.x, -pivot.y, -pivot.z),
        Quat::IDENTITY,
        Vec3::ONE,
    );
    to.mul(&back)
}

fn pose(bottom: Mat4, top: Mat4) -> Vec<f32> {
    [bottom.to_cols_array(), top.to_cols_array()].concat()
}

/// The CPU-skinned mesh of `skin` in `matrices`, as a static lit mesh.
fn cpu_mesh(skin: &SkinMesh, matrices: &[f32]) -> AttributedMesh {
    let cpu = skin_cpu(skin, Some(matrices)).unwrap();
    let p: Vec<f32> = cpu.positions.iter().flatten().copied().collect();
    let n: Vec<f32> = cpu.normals.iter().flatten().copied().collect();
    let t: Vec<f32> = cpu.tangents.iter().flatten().copied().collect();
    let mut rest = skin.rest.clone();
    rest.ensure_tangents();
    let mut s1 = Vec::new();
    rest.write_stream1_bytes(&mut s1);
    let uv: Vec<f32> = s1
        .chunks_exact(40)
        .flat_map(|v| [0, 4].map(|o| f32::from_le_bytes(v[o..o + 4].try_into().unwrap())))
        .collect();
    let mut idx = Vec::new();
    rest.write_index_bytes(&mut idx);
    let idx: Vec<u32> = idx
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let mut mesh = raa_mesh::from_js_arrays(&p, &n, Some(&uv), Some(&t), None, &idx, &[]).unwrap();
    mesh.ensure_tangents();
    mesh
}

struct SkinNode(Rc<RefCell<SkinPass>>);

impl FeaturePass for SkinNode {
    fn name(&self) -> &str {
        raa_skin::SKIN_PASS
    }
    fn order(&self) -> u32 {
        raa_skin::SKIN_ORDER
    }
    fn set_enabled(&mut self, enabled: bool, lighting: &mut LightingResources) {
        self.0.borrow_mut().set_enabled(enabled, lighting);
    }
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        self.0.borrow_mut().record(ctx)
    }
}

struct Rig {
    pipeline: Rc<RefCell<Pipeline>>,
    lit: LitRenderer,
    scene: Scene,
    list: RenderList,
    tags: Vec<u32>,
    slots: Vec<u32>,
    graph: FrameGraph,
    backbuffer: OffscreenTarget,
}

impl Rig {
    fn new(gpu: &GpuContext, skin: Option<Rc<RefCell<SkinPass>>>, capacity: usize) -> Rig {
        let pipeline = Rc::new(RefCell::new(Pipeline::new(gpu, SURFACE, false)));
        let opts = LitOptions {
            msaa: 1,
            ..LitOptions::default()
        };
        let lit = LitRenderer::new(gpu, &pipeline, opts).expect("lit renderer");
        if let Some(skin) = skin {
            lit.handle().set_skin_views(skin.borrow().views());
            pipeline
                .borrow_mut()
                .add(Box::new(SkinNode(skin)))
                .expect("skin pass");
        }
        let mut scene = Scene::with_capacity(capacity, capacity);
        scene.camera.set_viewport(SIZE.0, SIZE.1);
        scene.camera.eye = Vec3::new(0.3, 0.4, 2.0);
        scene.camera.target = Vec3::ZERO;
        let mut graph = FrameGraph::new(SURFACE);
        Pipeline::install(&pipeline, &mut graph);
        graph.compile().expect("compile");
        Rig {
            pipeline,
            lit,
            scene,
            list: RenderList::with_capacity(capacity),
            tags: Vec::with_capacity(capacity),
            slots: Vec::with_capacity(capacity),
            graph,
            backbuffer: OffscreenTarget::new(gpu, SIZE.0, SIZE.1, SURFACE),
        }
    }

    fn place(&mut self, mesh: raa_core::MeshId, x: f32, tag: u32) {
        let node = self
            .scene
            .add_node(
                None,
                Transform::from_trs(Vec3::new(x, 0.0, 0.0), Quat::IDENTITY, Vec3::ONE),
            )
            .expect("node");
        let id = self
            .scene
            .add_instance(node, mesh, MaterialId(0))
            .expect("instance");
        self.scene.set_instance_tag(id, tag).expect("tag");
    }

    /// One frame; returns (draw calls, HDR texels).
    fn frame(&mut self, gpu: &GpuContext) -> (u32, Vec<[f32; 4]>) {
        let counts = self
            .lit
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
            .expect("poll");
        let p = self.pipeline.borrow();
        let hdr = read_rgba16f(gpu, p.targets().expect("targets").hdr().src_texture());
        (counts.draw_calls, hdr)
    }
}

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

fn read_rgba16f(gpu: &GpuContext, texture: &wgpu::Texture) -> Vec<[f32; 4]> {
    let row_bytes = SIZE.0 as usize * 8;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
    let padded = row_bytes.div_ceil(align) * align;
    let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("golden.readback"),
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
    encoder.map_buffer_on_submit(&buffer, wgpu::MapMode::Read, .., |_| {});
    gpu.queue().submit([encoder.finish()]);
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let mapped = buffer.get_mapped_range(..).expect("mapped").to_vec();
    let mut texels = Vec::with_capacity((SIZE.0 * SIZE.1) as usize);
    for row in mapped.chunks_exact(padded) {
        for px in row[..row_bytes].chunks_exact(8) {
            texels.push(std::array::from_fn(|c| {
                f16_to_f32(u16::from_le_bytes([px[2 * c], px[2 * c + 1]]))
            }));
        }
    }
    texels
}

fn rest_view(gpu: &GpuContext, skin: &SkinMesh, edges: u32) -> RestMeshView {
    let n = skin.rest.vertex_count() as u64;
    let b = |size: u64| {
        gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    };
    RestMeshView::new(
        b(n * 24),
        b(n * 40),
        b(4 * 3),
        Some(b(u64::from(edges) * 24)),
    )
    .unwrap()
}

#[test]
fn instanced_skinning_matches_the_cpu_reference_in_three_poses() {
    let test = "instanced_skinning_matches_the_cpu_reference_in_three_poses";
    let Some(gpu) = software_gpu() else {
        skip(
            test,
            "software_gpu returned None; the golden needs an adapter",
        );
        return;
    };
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let skin_mesh = tube();
    let bottom = Vec3::new(0.0, -0.45, 0.0);
    let centre = Vec3::new(0.0, 0.0, 0.0);
    let pose_pairs = [
        (
            pose(Mat4::IDENTITY, joint(centre, 0.6, Vec3::ZERO)),
            pose(Mat4::IDENTITY, joint(centre, -0.9, Vec3::ZERO)),
        ),
        (
            pose(
                joint(bottom, 0.3, Vec3::ZERO),
                joint(centre, 1.1, Vec3::new(0.0, 0.1, 0.0)),
            ),
            pose(joint(bottom, -0.2, Vec3::ZERO), Mat4::IDENTITY),
        ),
        (
            pose(
                joint(bottom, 0.0, Vec3::new(0.0, 0.15, 0.0)),
                joint(centre, -0.5, Vec3::new(0.1, 0.2, 0.0)),
            ),
            pose(
                joint(centre, 0.4, Vec3::ZERO),
                joint(centre, 0.4, Vec3::ZERO),
            ),
        ),
    ];

    // New path: one rest mesh, two posed instances, one palette.
    let skin = Rc::new(RefCell::new(SkinPass::new(&gpu)));
    let mut new = Rig::new(&gpu, Some(Rc::clone(&skin)), 2);
    let lit_id = new.lit.upload(&gpu, &skin_mesh.rest).expect("upload");
    let mesh = new.lit.submesh_ids(lit_id).expect("submeshes")[0];
    let inputs = Rc::new(SkinVertexInputs::new(&gpu, &skin_mesh).expect("inputs"));
    for (tag, x) in [(1, -0.45), (2, 0.45)] {
        new.place(mesh, x, tag);
        skin.borrow_mut()
            .add_instance(
                tag,
                Rc::clone(&inputs),
                rest_view(&gpu, &skin_mesh, inputs.edges),
            )
            .expect("skin instance");
    }

    for (k, (a, b)) in pose_pairs.iter().enumerate() {
        skin.borrow_mut().set_pose(&gpu, 1, a).expect("pose a");
        skin.borrow_mut().set_pose(&gpu, 2, b).expect("pose b");
        let (draws, got) = new.frame(&gpu);
        let stats = skin.borrow().last_stats();
        assert_eq!(draws, 1, "both posed instances in ONE instanced draw");
        assert_eq!((stats.dispatches, stats.palette_writes), (0, 1));

        // Reference: two static CPU-skinned meshes.
        let mut reference = Rig::new(&gpu, None, 2);
        for (tag, x, m) in [(1, -0.45, a), (2, 0.45, b)] {
            let id = reference
                .lit
                .upload(&gpu, &cpu_mesh(&skin_mesh, m))
                .expect("ref");
            let sub = reference.lit.submesh_ids(id).expect("sub")[0];
            reference.place(sub, x, tag);
        }
        let (_, want) = reference.frame(&gpu);

        let background = want[0];
        let covered = want.iter().filter(|t| **t != background).count();
        assert!(covered > 200, "pose {k}: the tubes cover {covered} texels");
        let mut worst = 0.0f32;
        let mut off = 0usize;
        for (g, w) in got.iter().zip(&want) {
            let d = g
                .iter()
                .zip(w)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f32::max);
            worst = worst.max(d);
            if d > 0.02 {
                off += 1;
            }
        }
        let got_covered = got.iter().filter(|t| **t != got[0]).count();
        eprintln!(
            "golden pose {k}: {covered} covered texels (new path {got_covered}), {off} differ > 0.02, worst {worst}"
        );
        if std::env::var_os("RAA_GOLDEN_DUMP").is_some() {
            for (name, img) in [("got", &got), ("want", &want)] {
                let mut ppm = format!("P3\n{} {}\n255\n", SIZE.0, SIZE.1);
                for t in img.iter() {
                    for c in &t[..3] {
                        ppm.push_str(&format!("{} ", (c.clamp(0.0, 1.0) * 255.0) as u8));
                    }
                    ppm.push('\n');
                }
                let path = format!("/root/nmc/temp/raa-instskin2/golden_{k}_{name}.ppm");
                std::fs::write(path, ppm).unwrap();
            }
        }
        assert!(
            off * 100 <= covered,
            "pose {k}: {off} of {covered} texels differ by more than 0.02 (worst {worst})"
        );
    }
    // Editor overlays: compute runs only for the outlined instance, and stops with it.
    let handle = new.lit.handle();
    handle.set_outline(1, Some([1.0, 0.5, 0.0]));
    let (draws, _) = new.frame(&gpu);
    let stats = skin.borrow().last_stats();
    assert_eq!(draws, 1);
    assert_eq!(
        (stats.dispatches, stats.overlays),
        (2, 1),
        "vertices + edges of tag 1"
    );
    handle.set_wireframe(raa_lit::WireframeMode::Overlay, None);
    new.frame(&gpu);
    assert_eq!(
        skin.borrow().last_stats().dispatches,
        4,
        "wireframe on: both instances"
    );
    handle.set_wireframe(raa_lit::WireframeMode::Off, None);
    handle.set_outline(1, None);
    new.frame(&gpu);
    let stats = skin.borrow().last_stats();
    assert_eq!(
        (stats.dispatches, stats.overlays),
        (0, 0),
        "overlays off: no compute"
    );

    let error = block_on(errors.pop());
    assert!(error.is_none(), "GPU validation error: {error:?}");
    std::mem::forget(gpu);
}

/// Texels of `got` differing from `want` by more than 0.02 in any channel.
fn texels_off(got: &[[f32; 4]], want: &[[f32; 4]]) -> usize {
    got.iter()
        .zip(want)
        .filter(|(g, w)| g.iter().zip(w.iter()).any(|(x, y)| (x - y).abs() > 0.02))
        .count()
}

/// Regression (Royal Fiesta penguin, 2026-11-03): a material in tint mode `graph` (every
/// player-tinted character material) composes its entry points with the graph tint call,
/// so the posed variant must be derived from the composed pass, not from the raw pass
/// source; otherwise the rest pipeline is reused and posed instances draw in rest pose.
/// Two instances of one mesh, the pose changing every frame (setSkinPose per frame), must
/// match the CPU reference and differ from the rest pose.
#[test]
fn graph_tint_posed_instances_follow_their_pose_every_frame() {
    let test = "graph_tint_posed_instances_follow_their_pose_every_frame";
    let Some(gpu) = software_gpu() else {
        skip(
            test,
            "software_gpu returned None; the golden needs an adapter",
        );
        return;
    };
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let skin_mesh = tube();
    let centre = Vec3::new(0.0, 0.0, 0.0);
    let graph = |rig: &Rig| {
        rig.lit
            .handle()
            .set_material_tint_mode(MaterialId(0), raa_lit::TintMode::Graph);
    };

    let skin = Rc::new(RefCell::new(SkinPass::new(&gpu)));
    let mut new = Rig::new(&gpu, Some(Rc::clone(&skin)), 2);
    graph(&new);
    let lit_id = new.lit.upload(&gpu, &skin_mesh.rest).expect("upload");
    let mesh = new.lit.submesh_ids(lit_id).expect("submeshes")[0];
    let inputs = Rc::new(SkinVertexInputs::new(&gpu, &skin_mesh).expect("inputs"));
    for (tag, x) in [(1, -0.45), (2, 0.45)] {
        new.place(mesh, x, tag);
        skin.borrow_mut()
            .add_instance(
                tag,
                Rc::clone(&inputs),
                rest_view(&gpu, &skin_mesh, inputs.edges),
            )
            .expect("skin instance");
    }
    // Rest frame (no pose yet).
    let (_, rest) = new.frame(&gpu);

    for (k, angle) in [0.9f32, -0.7, 1.2].into_iter().enumerate() {
        let a = pose(Mat4::IDENTITY, joint(centre, angle, Vec3::ZERO));
        let b = pose(Mat4::IDENTITY, joint(centre, -angle, Vec3::ZERO));
        skin.borrow_mut().set_pose(&gpu, 1, &a).expect("pose a");
        skin.borrow_mut().set_pose(&gpu, 2, &b).expect("pose b");
        let (draws, got) = new.frame(&gpu);
        assert_eq!(
            draws, 1,
            "frame {k}: both posed instances in ONE instanced draw"
        );

        let mut reference = Rig::new(&gpu, None, 2);
        graph(&reference);
        for (tag, x, m) in [(1, -0.45, &a), (2, 0.45, &b)] {
            let id = reference
                .lit
                .upload(&gpu, &cpu_mesh(&skin_mesh, m))
                .expect("ref");
            let sub = reference.lit.submesh_ids(id).expect("sub")[0];
            reference.place(sub, x, tag);
        }
        let (_, want) = reference.frame(&gpu);
        let covered = want.iter().filter(|t| **t != want[0]).count();
        let moved = texels_off(&want, &rest);
        let off = texels_off(&got, &want);
        let from_rest = texels_off(&got, &rest);
        eprintln!(
            "graph tint frame {k}: {covered} covered, pose moves {moved} texels, new path {off} off reference, {from_rest} off rest"
        );
        assert!(
            moved > 100,
            "frame {k}: the pose must move the tubes ({moved})"
        );
        assert!(
            off * 100 <= covered,
            "frame {k}: {off} of {covered} texels differ from the CPU reference (rest pose drawn?)"
        );
    }
    let error = block_on(errors.pop());
    assert!(error.is_none(), "GPU validation error: {error:?}");
    std::mem::forget(gpu);
}

/// Structural counts per frame for N = 1..200 posed instances of one mesh and material
/// (what examples/instanced-skin shows in the browser): one draw per pass, one palette
/// write, zero compute dispatches, whatever N.
#[test]
fn draws_dispatches_and_uploads_per_frame_are_constant_in_n() {
    let test = "draws_dispatches_and_uploads_per_frame_are_constant_in_n";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let skin_mesh = tube();
    for n in [1u32, 2, 10, 60, 200] {
        let skin = Rc::new(RefCell::new(SkinPass::new(&gpu)));
        let mut rig = Rig::new(&gpu, Some(Rc::clone(&skin)), n as usize);
        let lit_id = rig.lit.upload(&gpu, &skin_mesh.rest).expect("upload");
        let mesh = rig.lit.submesh_ids(lit_id).expect("submeshes")[0];
        let inputs = Rc::new(SkinVertexInputs::new(&gpu, &skin_mesh).expect("inputs"));
        for tag in 0..n {
            rig.place(mesh, (tag % 20) as f32 * 0.1 - 1.0, tag);
            let rest = rest_view(&gpu, &skin_mesh, inputs.edges);
            skin.borrow_mut()
                .add_instance(tag, Rc::clone(&inputs), rest)
                .expect("skin instance");
        }
        for frame in 0..2 {
            for tag in 0..n {
                let a = 0.1 * (tag as f32) + frame as f32;
                let m = pose(Mat4::IDENTITY, joint(Vec3::ZERO, a.sin(), Vec3::ZERO));
                skin.borrow_mut().set_pose(&gpu, tag, &m).expect("pose");
            }
            let (draws, _) = rig.frame(&gpu);
            let s = skin.borrow().last_stats();
            if frame == 1 {
                eprintln!(
                    "N={n:3}: draws/pass={draws} compute dispatches={} palette writes={} posed={} palette matrices={}",
                    s.dispatches,
                    s.palette_writes,
                    s.posed,
                    skin.borrow().views().palette_matrices()
                );
            }
            assert_eq!(
                (draws, s.dispatches, s.palette_writes, s.posed),
                (1, 0, 1, n),
                "N = {n}"
            );
        }
        assert_eq!(
            skin.borrow().views().overlay_views(),
            0,
            "no per-instance buffers"
        );
    }
    let error = block_on(errors.pop());
    assert!(error.is_none(), "GPU validation error: {error:?}");
    std::mem::forget(gpu);
}
