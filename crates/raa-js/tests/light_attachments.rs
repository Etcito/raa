//! Local-light attachments and Engine object mutations (M2-LOCAL-STATE × S17).
//!
//! The renderer-owned registry is keyed by the stable `LightHandle`; its parents are the JS
//! instance-group handles resolved through `EngineState.instances` and each group's node,
//! never a compacted submesh slot. The CPU cases drive the same `state.rs` helpers the
//! Engine's `setTransform`, `setTransforms`, `removeInstance` and `removeNode` call.

#![cfg(not(target_arch = "wasm32"))]

#[allow(dead_code)]
#[path = "../src/convert.rs"]
mod convert;
#[allow(dead_code)]
#[path = "../src/feature_map.rs"]
mod feature_map;
#[allow(dead_code)]
#[path = "../src/state.rs"]
mod state;

use convert::HandleMap;
use raa_core::{MaterialId, MeshId};
use raa_lit::{LitMeshId, LocalLight, LocalLights};
use raa_math::{Quat, Vec3};
use raa_scene::{NodeId, Scene, Transform};
use state::{
    InstanceGroup, NodeParents, remove_instance_checked, remove_node_checked, set_local_checked,
};

fn node(scene: &mut Scene, parent: Option<NodeId>, local: Transform) -> NodeId {
    scene.add_node(parent, local).expect("scene node")
}

fn group(node: NodeId, tag: u32) -> InstanceGroup {
    InstanceGroup::new(node, LitMeshId(tag), tag, None)
}

fn translation(offset: Vec3) -> Transform {
    Transform::from_trs(offset, Quat::IDENTITY, Vec3::ONE)
}

fn close(a: [f32; 3], b: [f32; 3]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
}

fn position(lights: &LocalLights, handle: raa_lit::LightHandle) -> [f32; 3] {
    lights.get(handle).expect("light").position
}

#[test]
fn attachments_follow_the_instance_group_subtree_not_a_submesh_slot() {
    let mut scene = Scene::new();
    let root = node(&mut scene, None, translation(Vec3::new(10.0, 0.0, 0.0)));
    let child = node(
        &mut scene,
        Some(root),
        translation(Vec3::new(0.0, 2.0, 0.0)),
    );
    let outside = node(&mut scene, None, translation(Vec3::new(0.0, 0.0, 100.0)));
    let parents: NodeParents = vec![(root, None), (child, Some(root)), (outside, None)];
    let mut instances = HandleMap::new("instance");
    let at_child = instances.insert(group(child, 1));
    let sibling = instances.insert(group(child, 2));
    let at_outside = instances.insert(group(outside, 3));

    let mut lights = LocalLights::new();
    let followed = lights.add(LocalLight::default()).expect("light");
    let untouched = lights.add(LocalLight::default()).expect("light");
    scene.update_world();
    lights
        .attach(
            followed,
            at_child,
            [1.0, 0.0, 0.0],
            &scene.world(child).expect("child world"),
        )
        .expect("attach");
    lights
        .attach(
            untouched,
            at_outside,
            [0.0, 0.0, 1.0],
            &scene.world(outside).expect("outside world"),
        )
        .expect("attach");
    assert!(close(position(&lights, followed), [11.0, 2.0, 0.0]));

    set_local_checked(
        &mut scene,
        &instances,
        &parents,
        &mut lights,
        root,
        translation(Vec3::new(-10.0, 0.0, 4.0)),
    )
    .expect("validated move");
    assert!(close(position(&lights, followed), [-9.0, 2.0, 4.0]));
    assert!(close(position(&lights, untouched), [0.0, 0.0, 101.0]));
    assert_eq!(lights.attached_count(), 2);

    // A sibling group on the same node follows the same subtree world; the JS handle (not
    // the submesh instances) is the parent key.
    let other = lights.add(LocalLight::default()).expect("light");
    lights
        .attach(
            other,
            sibling,
            [0.0, 1.0, 0.0],
            &scene.world(child).expect("child world"),
        )
        .expect("attach");
    assert!(close(position(&lights, other), [-10.0, 3.0, 4.0]));
    assert_eq!(lights.attached_count(), 3);
}

#[test]
fn degenerate_transform_refuses_before_scene_or_registry_change() {
    let mut scene = Scene::new();
    let root = node(&mut scene, None, Transform::IDENTITY);
    let child = node(
        &mut scene,
        Some(root),
        translation(Vec3::new(0.0, 1.0, 0.0)),
    );
    let parents: NodeParents = vec![(root, None), (child, Some(root))];
    let mut instances = HandleMap::new("instance");
    let at_child = instances.insert(group(child, 1));
    let mut lights = LocalLights::new();
    let handle = lights.add(LocalLight::default()).expect("light");
    scene.update_world();
    lights
        .attach(
            handle,
            at_child,
            [0.0; 3],
            &scene.world(child).expect("child world"),
        )
        .expect("attach");
    let before_local = scene.local(child).expect("child");
    let before_world = scene.world(child).expect("child world");
    let before_light = *lights.get(handle).expect("light");

    // Scale z -> 0 maps the local direction [0, 0, -1] to the zero vector: refuse.
    let flat = Transform::from_trs(Vec3::ZERO, Quat::IDENTITY, Vec3::new(1.0, 1.0, 0.0));
    let err = set_local_checked(&mut scene, &instances, &parents, &mut lights, child, flat)
        .expect_err("degenerate attachment");
    assert_eq!(err.code(), "light.invalid_attachment");
    assert_eq!(scene.local(child).expect("child"), before_local);
    assert_eq!(scene.world(child).expect("child world"), before_world);
    assert_eq!(*lights.get(handle).expect("light"), before_light);
}

#[test]
fn sequential_batch_entries_keep_the_partial_application_contract() {
    let mut scene = Scene::new();
    let first = node(&mut scene, None, Transform::IDENTITY);
    let failing = node(&mut scene, None, Transform::IDENTITY);
    let later = node(&mut scene, None, Transform::IDENTITY);
    let parents: NodeParents = vec![(first, None), (failing, None), (later, None)];
    let mut instances = HandleMap::new("instance");
    let mut handles = [0_u32; 3];
    for (slot, id) in [first, failing, later].into_iter().enumerate() {
        handles[slot] = instances.insert(group(id, slot as u32 + 1));
    }
    let mut lights = LocalLights::new();
    let mut light_handles = [0_u64; 3];
    scene.update_world();
    for (slot, id) in [first, failing, later].into_iter().enumerate() {
        light_handles[slot] = lights.add(LocalLight::default()).expect("light");
        lights
            .attach(
                light_handles[slot],
                handles[slot],
                [0.0; 3],
                &scene.world(id).expect("world"),
            )
            .expect("attach");
    }

    let batch = [
        (first, translation(Vec3::new(1.0, 0.0, 0.0))),
        (
            failing,
            Transform::from_trs(Vec3::ZERO, Quat::IDENTITY, Vec3::new(1.0, 1.0, 0.0)),
        ),
        (later, translation(Vec3::new(3.0, 0.0, 0.0))),
    ];
    let mut applied = 0;
    for (node, local) in batch {
        match set_local_checked(&mut scene, &instances, &parents, &mut lights, node, local) {
            Ok(()) => applied += 1,
            Err(e) => {
                assert_eq!(e.code(), "light.invalid_attachment");
                break;
            }
        }
    }
    assert_eq!(applied, 1, "entries before the failure stay applied");
    assert_eq!(
        scene.local(first).expect("first"),
        translation(Vec3::new(1.0, 0.0, 0.0))
    );
    assert_eq!(scene.local(failing).expect("failing"), Transform::IDENTITY);
    assert_eq!(scene.local(later).expect("later"), Transform::IDENTITY);
    assert!(close(position(&lights, light_handles[0]), [1.0, 0.0, 0.0]));
    assert!(close(position(&lights, light_handles[1]), [0.0; 3]));
    assert!(close(position(&lights, light_handles[2]), [0.0; 3]));
}

#[test]
fn removing_a_group_or_node_detaches_at_the_last_valid_world() {
    // Group removal: the light keeps the world it had while attached.
    let mut scene = Scene::new();
    let root = node(&mut scene, None, Transform::IDENTITY);
    let child = node(
        &mut scene,
        Some(root),
        translation(Vec3::new(0.0, 2.0, 0.0)),
    );
    let parents: NodeParents = vec![(root, None), (child, Some(root))];
    let mut instances = HandleMap::new("instance");
    let at_child = instances.insert(group(child, 1));
    let mut lights = LocalLights::new();
    let handle = lights.add(LocalLight::default()).expect("light");
    scene.update_world();
    lights
        .attach(
            handle,
            at_child,
            [1.0, 0.0, 0.0],
            &scene.world(child).expect("child world"),
        )
        .expect("attach");
    set_local_checked(
        &mut scene,
        &instances,
        &parents,
        &mut lights,
        root,
        translation(Vec3::new(4.0, 0.0, 0.0)),
    )
    .expect("move");
    let last = position(&lights, handle);
    assert!(close(last, [5.0, 2.0, 0.0]));
    remove_instance_checked(&mut instances, &mut lights, at_child).expect("remove group");
    assert_eq!(lights.attached_count(), 0);
    assert!(close(position(&lights, handle), last));
    assert_eq!(
        instances
            .get_ref(at_child)
            .expect_err("handle never reused")
            .code(),
        "api.unknown_handle"
    );

    // Node removal drops a whole subtree of groups at their last valid world.
    let mut scene = Scene::new();
    let root = node(&mut scene, None, translation(Vec3::new(2.0, 0.0, 0.0)));
    let child = node(
        &mut scene,
        Some(root),
        translation(Vec3::new(0.0, 1.0, 0.0)),
    );
    let mut parents: NodeParents = vec![(root, None), (child, Some(root))];
    let mut instances = HandleMap::new("instance");
    let at_child = instances.insert(group(child, 1));
    let mut lights = LocalLights::new();
    let handle = lights.add(LocalLight::default()).expect("light");
    scene.update_world();
    lights
        .attach(
            handle,
            at_child,
            [0.0, 0.0, 3.0],
            &scene.world(child).expect("child world"),
        )
        .expect("attach");
    let last = position(&lights, handle);
    remove_node_checked(&mut scene, &mut instances, &mut parents, &mut lights, root)
        .expect("remove node");
    assert!(parents.is_empty(), "no links outlive their nodes");
    assert!(instances.get_ref(at_child).expect("stale group").stale);
    assert_eq!(lights.attached_count(), 0);
    assert!(close(position(&lights, handle), last));
}

#[test]
fn a_group_rebuild_keeps_the_attachment_keyed_by_the_instance_handle() {
    let mut scene = Scene::new();
    let node = node(&mut scene, None, Transform::IDENTITY);
    let parents: NodeParents = vec![(node, None)];
    let mut instances = HandleMap::new("instance");
    let handle = instances.insert(group(node, 1));
    let mut lights = LocalLights::new();
    let light = lights.add(LocalLight::default()).expect("light");
    scene.update_world();
    lights
        .attach(
            light,
            handle,
            [1.0, 0.0, 0.0],
            &scene.world(node).expect("world"),
        )
        .expect("attach");

    instances
        .get_mut(handle)
        .expect("group")
        .rebuild(&mut scene, [(MeshId(1), None), (MeshId(2), None)])
        .expect("rebuild");
    let group = instances.get_ref(handle).expect("group");
    assert_eq!(group.instances.len(), 2, "one scene instance per submesh");
    assert_eq!(group.tag, 1);
    assert_eq!(lights.attached_count(), 1, "the key is the JS handle");

    set_local_checked(
        &mut scene,
        &instances,
        &parents,
        &mut lights,
        node,
        translation(Vec3::new(0.0, 5.0, 0.0)),
    )
    .expect("move after rebuild");
    assert!(close(position(&lights, light), [1.0, 5.0, 0.0]));
}

/// The public renderer and pipeline wired as the Engine wires them, on a software GPU (SKIP
/// without an adapter). The renderer-owned registry keeps an attached light across a real
/// `prepare` and a real pipeline execution, and the attachment follows its instance-group
/// parent to a new world transform in the same frame.
#[test]
fn software_gpu_public_renderer_pipeline_carries_attached_lights() {
    use std::cell::RefCell;
    use std::rc::Rc;

    use raa_core::testing::{skip, software_gpu};
    use raa_core::{FrameGraph, GpuContext, OffscreenTarget, wgpu};
    use raa_geom::builtin::DebugCube;
    use raa_geom::{GeometryProvider, MeshData, ParamValue, Params, Quality};
    use raa_lit::{AttributedMesh, LitOptions, LitRenderer};
    use raa_pipeline::Pipeline;
    use raa_scene::RenderList;

    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    const SIZE: (u32, u32) = (64, 64);

    fn execute(pipeline: &Rc<RefCell<Pipeline>>, gpu: &GpuContext) {
        let target = OffscreenTarget::new(gpu, SIZE.0, SIZE.1, FORMAT);
        let mut graph = FrameGraph::new(FORMAT);
        Pipeline::install(pipeline, &mut graph);
        graph.compile().expect("compile");
        let commands = graph
            .execute(gpu, target.view(), target.size())
            .expect("execute");
        gpu.queue().submit([commands]);
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
    }

    let Some(gpu) = software_gpu() else {
        skip(
            "light_attachments_public_renderer_pipeline",
            "software_gpu returned None",
        );
        return;
    };
    let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, FORMAT, false)));
    let mut lit = LitRenderer::new(&gpu, &pipeline, LitOptions::default()).expect("lit renderer");
    let mut data = MeshData::new();
    DebugCube
        .generate(
            &Params::new().with("half", ParamValue::Scalar(0.5)),
            Quality::LOD0,
            &mut data,
        )
        .expect("cube");
    let mesh = {
        let id = lit
            .upload(&gpu, &AttributedMesh::from_mesh(data))
            .expect("upload");
        lit.submesh_ids(id).expect("submeshes")[0]
    };
    let mut scene = Scene::new();
    let node = node(&mut scene, None, translation(Vec3::new(2.0, 0.0, 0.0)));
    scene
        .add_instance(node, mesh, MaterialId(0))
        .expect("instance");
    scene.camera.set_viewport(SIZE.0, SIZE.1);
    scene.update_world();
    let handle = lit
        .local_lights_mut()
        .add(LocalLight::default())
        .expect("light");
    lit.local_lights_mut()
        .attach(
            handle,
            1,
            [0.0, 1.0, 0.0],
            &scene.world(node).expect("world"),
        )
        .expect("attach");
    assert_eq!(lit.local_lights().attached_count(), 1);

    let mut list = RenderList::with_capacity(1);
    let (mut tags, mut slots) = (Vec::with_capacity(1), Vec::with_capacity(1));
    let counts = lit
        .prepare(&gpu, &mut scene, &mut list, &mut tags, &mut slots)
        .expect("prepare with a local light");
    assert_eq!(counts.instances, 1);
    execute(&pipeline, &gpu);

    // Same-frame movement: the parent moves, the registry follows the group's new world and
    // the next prepared, submitted frame keeps the attachment.
    let moved = translation(Vec3::new(-4.0, 0.0, 3.0));
    scene.set_local(node, moved).expect("scene move");
    scene.update_world();
    lit.local_lights_mut()
        .update_parent(1, &scene.world(node).expect("moved world"))
        .expect("update parent");
    lit.prepare(&gpu, &mut scene, &mut list, &mut tags, &mut slots)
        .expect("prepare after movement");
    execute(&pipeline, &gpu);
    assert_eq!(lit.local_lights().attached_count(), 1);
    assert!(close(
        lit.local_lights().get(handle).expect("light").position,
        [-4.0, 1.0, 3.0]
    ));
}
