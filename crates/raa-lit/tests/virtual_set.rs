//! The virtualized-instance set (P3-LIT-SEAM). CPU tests cover `VirtualSet` membership,
//! generations and clearing. The software-GPU test (SKIP without an adapter) checks that a
//! member leaves the camera draw lists and the cascade caster section while the rest stay,
//! and that frame feature bits are stored by id.
//!
//! The GPU test records no pass, so no depth or HDR input is read: it only prepares
//! frames. Every shared input it needs is set up by the renderer itself (the uploaded cube,
//! the scene and its camera, the pipeline's default `EnvGpu`). The scene and both cascade
//! boxes are inside a 16 m box, with the camera inside it too.

#![cfg(not(target_arch = "wasm32"))]

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::testing::{skip, software_gpu};
use raa_core::{GpuContext, MaterialId, MeshId, wgpu};
use raa_geom::builtin::DebugCube;
use raa_geom::{GeometryProvider, MeshData, ParamValue, Params, Quality};
use raa_lit::{
    AttributedMesh, DrawClass, FLAG_TRANSPARENT, FrameCounts, InstanceExtra, InstanceShape,
    LitDraw, LitMeshId, LitOptions, LitRenderer, MaterialFnId, PipelineSpec, SurfaceMaterial,
    SurfaceModel, VirtualSet,
};
use raa_math::{Mat4, Quat, Vec3};
use raa_pipeline::Pipeline;
use raa_scene::{Camera, InstanceId, NodeId, RenderList, Scene, Transform};
use raa_shader::{PassKind, VariantKey};

const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// Triangles of one DebugCube.
const CUBE_TRIANGLES: u64 = 12;

/// Adds an instance of `mesh` under a new node at `at`; returns the node and the instance.
fn add_at(scene: &mut Scene, mesh: MeshId, at: Vec3) -> (NodeId, InstanceId) {
    let node = scene
        .add_node(None, Transform::from_trs(at, Quat::IDENTITY, Vec3::ONE))
        .expect("node");
    let instance = scene
        .add_instance(node, mesh, MaterialId(0))
        .expect("instance");
    (node, instance)
}

/// Fresh scene instances for the CPU tests (their mesh is never drawn).
fn instances(n: usize) -> (Scene, Vec<InstanceId>) {
    let mut scene = Scene::new();
    let ids = (0..n)
        .map(|i| add_at(&mut scene, MeshId(0), Vec3::new(i as f32, 0.0, 0.0)).1)
        .collect();
    (scene, ids)
}

#[test]
fn new_set_is_empty_at_generation_zero() {
    let set = VirtualSet::new();
    let (_scene, ids) = instances(1);
    assert_eq!(set.len(), 0);
    assert!(set.is_empty());
    assert_eq!(set.generation(), 0);
    assert!(!set.contains(ids[0]));
    assert_eq!(VirtualSet::default().generation(), 0);
}

#[test]
fn add_counts_members_and_advances_the_generation() {
    let (_scene, ids) = instances(2);
    let mut set = VirtualSet::new();
    set.set(ids[0], true);
    assert!(set.contains(ids[0]));
    assert!(!set.contains(ids[1]));
    assert_eq!((set.len(), set.is_empty(), set.generation()), (1, false, 1));
    set.set(ids[1], true);
    assert_eq!((set.len(), set.generation()), (2, 2));
    assert!(set.contains(ids[1]));
}

#[test]
fn re_adding_a_member_keeps_its_generation() {
    let (_scene, ids) = instances(1);
    let mut set = VirtualSet::new();
    set.set(ids[0], true);
    let generation = set.generation();
    set.set(ids[0], true);
    assert_eq!(set.len(), 1);
    assert_eq!(set.generation(), generation);
    assert!(set.contains(ids[0]));
}

#[test]
fn remove_drops_the_member_and_ignores_non_members() {
    let (_scene, ids) = instances(2);
    let mut set = VirtualSet::new();
    set.set(ids[0], true);
    set.set(ids[1], true);
    set.set(ids[0], false);
    assert!(!set.contains(ids[0]));
    assert!(set.contains(ids[1]));
    assert_eq!((set.len(), set.generation()), (1, 3));
    // Removing a non-member, or a member twice, changes nothing.
    set.set(ids[0], false);
    set.set(ids[0], false);
    assert_eq!((set.len(), set.generation()), (1, 3));
    set.set(ids[1], false);
    assert!(set.is_empty());
    assert_eq!(set.generation(), 4);
}

#[test]
fn a_stale_generation_of_a_reused_slot_is_not_contained() {
    let (mut scene, ids) = instances(1);
    let stale = ids[0];
    let mut set = VirtualSet::new();
    set.set(stale, true);
    scene.remove_instance(stale).expect("remove");
    let node = scene
        .add_node(
            None,
            Transform::from_trs(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE),
        )
        .expect("node");
    let fresh = scene
        .add_instance(node, MeshId(0), MaterialId(0))
        .expect("instance");
    assert_eq!(
        fresh.to_bits() as u32,
        stale.to_bits() as u32,
        "slot is reused"
    );
    assert_ne!(fresh, stale);
    assert!(!set.contains(fresh));

    // Adding the fresh id replaces the stale member of its slot: one member, one change.
    let generation = set.generation();
    set.set(fresh, true);
    assert!(set.contains(fresh));
    assert!(!set.contains(stale));
    assert_eq!((set.len(), set.generation()), (1, generation + 1));

    // Removing the stale id does nothing to the fresh member.
    set.set(stale, false);
    assert!(set.contains(fresh));
    assert_eq!(set.len(), 1);
}

#[test]
fn clear_on_empty_keeps_the_generation_and_clear_on_members_advances_it() {
    let (_scene, ids) = instances(2);
    let mut set = VirtualSet::new();
    set.clear();
    assert_eq!(set.generation(), 0);
    assert!(set.is_empty());

    set.set(ids[0], true);
    set.set(ids[1], true);
    assert_eq!(set.generation(), 2);
    set.clear();
    assert!(set.is_empty());
    assert!(!set.contains(ids[0]));
    assert!(!set.contains(ids[1]));
    assert_eq!(set.generation(), 3);
    set.clear();
    assert_eq!(set.generation(), 3);
}

/// A 16 m orthographic box centred on the origin, as a cascade view-projection: the camera
/// sits 8 m up +Z looking down -Z, depth 0..16 m, x and y within +-8 m.
fn cascade_box() -> Mat4 {
    let camera = Camera {
        eye: Vec3::new(0.0, 0.0, 8.0),
        target: Vec3::ZERO,
        up: Vec3::Y,
        near: 0.0,
        far: 16.0,
        aspect: 1.0,
        ortho_height: Some(16.0),
        ..Camera::default()
    };
    camera.view_proj()
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

/// A pipeline with a single-sample `LitRenderer` and two cubes of one uploaded mesh inside
/// the 16 m box, seen by the default camera (eye 2 m up, 6 m back, inside the box).
struct Rig {
    lit: LitRenderer,
    scene: Scene,
    list: RenderList,
    tags: Vec<u32>,
    slots: Vec<u32>,
    a: InstanceId,
    mesh: MeshId,
    _pipeline: Rc<RefCell<Pipeline>>,
}

impl Rig {
    fn new(gpu: &GpuContext) -> Rig {
        let pipeline = Rc::new(RefCell::new(Pipeline::new(gpu, SURFACE, false)));
        let opts = LitOptions {
            msaa: 1,
            ..LitOptions::default()
        };
        let mut lit = LitRenderer::new(gpu, &pipeline, opts).expect("lit renderer");
        let uploaded = lit.upload(gpu, &cube()).expect("upload");
        let mesh = lit.submesh_ids(uploaded).expect("sub-meshes")[0];
        let mut scene = Scene::with_capacity(2, 2);
        let (_, a) = add_at(&mut scene, mesh, Vec3::new(-3.0, 0.0, -3.0));
        add_at(&mut scene, mesh, Vec3::new(3.0, 0.0, -3.0));
        Rig {
            lit,
            scene,
            list: RenderList::with_capacity(2),
            tags: Vec::with_capacity(2),
            slots: Vec::with_capacity(2),
            a,
            mesh,
            _pipeline: pipeline,
        }
    }

    /// Prepares one frame (no pass is recorded).
    fn prepare(&mut self, gpu: &GpuContext) -> FrameCounts {
        self.lit
            .prepare(
                gpu,
                &mut self.scene,
                &mut self.list,
                &mut self.tags,
                &mut self.slots,
            )
            .expect("prepare")
    }

    /// The camera draws of the opaque class, by value.
    fn opaque(&self) -> Vec<LitDraw> {
        self.lit.handle().draws(DrawClass::Opaque).to_vec()
    }

    /// Instances of the opaque camera draws.
    fn opaque_instances(&self) -> u32 {
        self.opaque().iter().map(|d| d.instance_count).sum()
    }

    /// Instances of the opaque caster draws of cascade `cascade`.
    fn cascade_instances(&self, cascade: usize) -> u32 {
        self.lit
            .handle()
            .cascade_draws(cascade, DrawClass::Opaque)
            .iter()
            .map(|d| d.instance_count)
            .sum()
    }
}

#[test]
fn virtual_members_leave_the_draw_lists_and_the_cascades() {
    let test = "virtual_members_leave_the_draw_lists_and_the_cascades";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let mut reference = Rig::new(&gpu);
    let mut subject = Rig::new(&gpu);

    // An empty set that was touched (a member added then removed) changes nothing.
    let a = subject.a;
    subject.lit.virtual_instances_mut().set(a, true);
    subject.lit.virtual_instances_mut().set(a, false);
    assert!(subject.lit.virtual_instances().is_empty());

    let base = reference.prepare(&gpu);
    let touched = subject.prepare(&gpu);
    assert_eq!(
        base,
        FrameCounts {
            instances: 2,
            draw_calls: 1,
            triangles: 2 * CUBE_TRIANGLES,
        }
    );
    assert_eq!(touched, base);
    assert_eq!(subject.opaque(), reference.opaque());
    assert_eq!(subject.opaque_instances(), 2);

    // One member: its persistent records are still written, but the draw lists lose it.
    subject.lit.virtual_instances_mut().set(a, true);
    let with_member = subject.prepare(&gpu);
    assert_eq!(with_member.instances, base.instances - 1);
    assert_eq!(with_member.draw_calls, base.draw_calls);
    assert_eq!(with_member.triangles, base.triangles - CUBE_TRIANGLES);
    assert_eq!(subject.opaque_instances(), reference.opaque_instances() - 1);

    // Cascade casters: the member is left out of cascade 0; the box covers both instances.
    let matrices = [cascade_box(); 4];
    reference
        .lit
        .handle()
        .select_casters(&gpu, &matrices, 1024)
        .expect("reference casters");
    subject
        .lit
        .handle()
        .select_casters(&gpu, &matrices, 1024)
        .expect("subject casters");
    assert_eq!(reference.cascade_instances(0), 2);
    assert_eq!(subject.cascade_instances(0), 1);

    // Frame feature bits: stored by id; 24 (tiling) and ids >= 64 are ignored.
    subject.lit.set_frame_feature(27, true);
    assert!(subject.lit.frame_feature(27));
    assert!(!subject.lit.frame_feature(24));
    assert!(!subject.lit.frame_feature(64));
    subject.prepare(&gpu);
    assert!(subject.lit.frame_feature(27));
}

/// The S32 `vg.shade` group-2 layout: the ids texture (Uint), three unfilterable float
/// textures and the 16-byte dynamic-offset `VgShadeGpu` uniform, all read by the fragment.
fn vg_shade_extra_layout(gpu: &GpuContext) -> wgpu::BindGroupLayout {
    let texture = |binding: u32, sample_type: wgpu::TextureSampleType| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let unfilterable = wgpu::TextureSampleType::Float { filterable: false };
    gpu.device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("test.vg_shade.group2"),
            entries: &[
                texture(0, wgpu::TextureSampleType::Uint),
                texture(1, unfilterable),
                texture(2, unfilterable),
                texture(3, unfilterable),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
            ],
        })
}

#[test]
fn eligibility_accessors_and_the_fullscreen_vg_shade_pipeline() {
    let test = "eligibility_accessors_and_the_fullscreen_vg_shade_pipeline";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let mut rig = Rig::new(&gpu);
    let handle = rig.lit.handle();
    let mesh = rig.mesh;

    // The persistent slot of a live instance is its scene index; it has no committed
    // previous row before the first submitted frame.
    let slot = handle.instance_slot(rig.a).expect("live instance slot");
    assert_eq!(slot, rig.a.to_bits() as u32);
    assert_eq!(handle.previous_instance_row(slot), None);

    // A default material on the uploaded cube: no pose, no deformation, opaque, no vertex
    // function, no colour stream.
    assert_eq!(
        handle.instance_eligibility(0, mesh, MaterialId(0)),
        InstanceShape {
            skinned: false,
            deformed: false,
            class: DrawClass::Opaque,
            vertex_fn: false,
            has_colors: false,
        }
    );
    assert_eq!(handle.material_fn_of(MaterialId(0)), MaterialFnId::STANDARD);

    // A transparent material draws in the Transparent class.
    let transparent = SurfaceMaterial {
        flags: FLAG_TRANSPARENT,
        ..SurfaceMaterial::default()
    };
    let transparent = rig.lit.materials_mut().add(transparent).expect("material");
    assert_eq!(
        handle.instance_eligibility(0, mesh, transparent).class,
        DrawClass::Transparent
    );

    // A non-default squash extra marks its tag deformed; clearing restores it.
    rig.lit.set_instance_extra(
        7,
        InstanceExtra {
            squash: [0.0, 1.0, 0.0, 0.5],
            ..InstanceExtra::default()
        },
    );
    assert!(handle.instance_eligibility(7, mesh, MaterialId(0)).deformed);
    assert!(!handle.instance_eligibility(8, mesh, MaterialId(0)).deformed);
    rig.lit.clear_instance_extra(7);
    assert!(!handle.instance_eligibility(7, mesh, MaterialId(0)).deformed);

    // A mesh uploaded with a colour stream reports has_colors.
    let mut coloured = cube();
    coloured.colors = Some(vec![[1.0, 0.0, 0.0, 1.0]; coloured.vertex_count()]);
    let coloured: LitMeshId = rig.lit.upload(&gpu, &coloured).expect("coloured upload");
    let coloured_mesh = rig.lit.submesh_ids(coloured).expect("sub-meshes")[0];
    assert!(
        handle
            .instance_eligibility(0, coloured_mesh, MaterialId(0))
            .has_colors
    );

    // A function material reports its registered function id.
    let source = "fn material_main(in: MaterialInput) -> MaterialOutput {
        return raa_default_output(in);
    }";
    let function = rig
        .lit
        .register_user_fn("test.probe", source)
        .expect("register function");
    let material = SurfaceMaterial {
        model: SurfaceModel::Function(function),
        ..SurfaceMaterial::default()
    };
    let material = rig.lit.materials_mut().add(material).expect("material");
    assert_eq!(handle.material_fn_of(material), function);

    // One prepared frame (no pass recorded) commits a previous row for the live slot.
    rig.prepare(&gpu);
    assert!(handle.previous_instance_row(slot).is_some());

    // The fullscreen VG shade pipeline composes, validates and builds against the S32
    // group-2 layout without a validation error.
    let spec = PipelineSpec {
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
        cull: None,
        alpha_to_coverage: false,
        extra_layout: Some(vg_shade_extra_layout(&gpu)),
    };
    let key = VariantKey {
        material_fn: MaterialFnId::STANDARD,
        pass: PassKind::VgShade,
        msaa: 1,
        flags: 0,
    };
    let scope = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let pipeline = handle.pipeline_for(&gpu, key, &spec);
    let error = raa_core::testing::block_on(scope.pop());
    assert!(error.is_none(), "VG shade pipeline validation: {error:?}");
    pipeline.expect("VG shade pipeline");
}
