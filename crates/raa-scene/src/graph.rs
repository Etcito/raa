//! The scene: a node hierarchy with local/world transforms, plus the camera and lights.
//!
//! Nodes live in a generational slot arena, so a removed node's handle is refused as
//! stale. An `order` list keeps every parent before its children: `update_world` walks
//! it once, without allocating.
//!
//! Instances (a mesh and a material drawn at a node) live in a second generational
//! arena. A private `draw_order` list of instance slots is kept sorted by
//! (mesh, material, slot) at mutation time, so `build_render_list` only walks it: the
//! frame path never sorts and reuses preallocated render list buffers.

use std::cmp::Ordering;

use raa_core::{MaterialId, MeshId};
use raa_math::Mat4;

use crate::{
    Ambient, Batch, Camera, DirectionalLight, InstanceId, NodeId, RenderItem, RenderList,
    SceneError, Transform,
};

/// One node of the hierarchy.
struct Node {
    parent: Option<NodeId>,
    local: Transform,
    /// World matrix as of the last `update_world`.
    world: Mat4,
    /// The local transform changed since the last `update_world`.
    dirty: bool,
    /// The world matrix was recomputed in the current `update_world` pass.
    updated: bool,
}

/// A slot of the node arena. `generation` moves on when the occupant is removed.
struct NodeSlot {
    generation: u32,
    node: Option<Node>,
}

/// One instance: a mesh drawn with a material at a node's world matrix.
struct Instance {
    node: NodeId,
    mesh: MeshId,
    material: MaterialId,
    visible: bool,
}

/// A slot of the instance arena. `generation` moves on when the occupant is removed.
struct InstanceSlot {
    generation: u32,
    instance: Option<Instance>,
}

/// A scene: camera, lights and a hierarchy of transformed nodes.
pub struct Scene {
    /// The camera the scene is viewed through.
    pub camera: Camera,
    /// The single directional light.
    pub light: DirectionalLight,
    /// The ambient term.
    pub ambient: Ambient,
    /// Node arena, addressed by `NodeId::index`.
    nodes: Vec<NodeSlot>,
    /// Free node slots, reused last-in, first-out.
    free_nodes: Vec<u32>,
    /// Every live node, parents before children.
    order: Vec<NodeId>,
    node_count: usize,
    /// Instance arena, addressed by `InstanceId::index`.
    instances: Vec<InstanceSlot>,
    /// Free instance slots, reused last-in, first-out.
    free_instances: Vec<u32>,
    /// Every live instance slot, sorted by (mesh, material, slot). Maintained by the
    /// mutating methods, never sorted in the frame path.
    draw_order: Vec<u32>,
    instance_count: usize,
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

impl Scene {
    /// An empty scene with the default camera and lights.
    pub fn new() -> Self {
        Self::with_capacity(0, 0)
    }

    /// An empty scene with room for `nodes` nodes and `instances` instances before
    /// reallocating.
    pub fn with_capacity(nodes: usize, instances: usize) -> Self {
        Scene {
            camera: Camera::default(),
            light: DirectionalLight::default(),
            ambient: Ambient::default(),
            nodes: Vec::with_capacity(nodes),
            free_nodes: Vec::with_capacity(nodes),
            order: Vec::with_capacity(nodes),
            node_count: 0,
            instances: Vec::with_capacity(instances),
            free_instances: Vec::with_capacity(instances),
            draw_order: Vec::with_capacity(instances),
            instance_count: 0,
        }
    }

    /// Adds a node under `parent` (or as a root). Its world matrix is the identity until
    /// the next [`Scene::update_world`].
    ///
    /// A stale parent gives [`SceneError::StaleNode`].
    pub fn add_node(
        &mut self,
        parent: Option<NodeId>,
        local: Transform,
    ) -> Result<NodeId, SceneError> {
        if let Some(p) = parent {
            self.node(p)?;
        }
        let node = Node {
            parent,
            local,
            world: Mat4::IDENTITY,
            dirty: true,
            updated: false,
        };
        let id = match self.free_nodes.pop() {
            Some(index) => {
                // Invariant: free_nodes only holds indices of empty slots.
                let slot = &mut self.nodes[index as usize];
                slot.node = Some(node);
                NodeId::new(index, slot.generation)
            }
            None => {
                // Invariant: node ids are u32; more than u32::MAX slots cannot be addressed.
                let index = u32::try_from(self.nodes.len()).expect("Scene exceeds u32::MAX nodes");
                self.nodes.push(NodeSlot {
                    generation: 0,
                    node: Some(node),
                });
                NodeId::new(index, 0)
            }
        };
        self.order.push(id);
        self.node_count += 1;
        Ok(id)
    }

    /// Removes `id`, its whole subtree and every instance attached to it. A stale id
    /// gives [`SceneError::StaleNode`].
    pub fn remove_node(&mut self, id: NodeId) -> Result<(), SceneError> {
        self.node(id)?;
        self.free_node(id);
        // Parents come before children in `order`, so one scan frees every node whose
        // parent was freed earlier in this scan (or is the subtree root).
        for i in 0..self.order.len() {
            let Some(&child) = self.order.get(i) else {
                break;
            };
            let orphan = match self.node(child) {
                Ok(node) => node.parent.is_some_and(|p| self.node(p).is_err()),
                Err(_) => false,
            };
            if orphan {
                self.free_node(child);
            }
        }
        let nodes = &self.nodes;
        self.order.retain(|&n| Self::lookup(nodes, n).is_some());
        // Drop the instances whose node was just freed.
        let mut dropped = false;
        for index in 0..self.instances.len() {
            let Some(slot) = self.instances.get_mut(index) else {
                break;
            };
            let orphan = slot
                .instance
                .as_ref()
                .is_some_and(|inst| Self::lookup(&self.nodes, inst.node).is_none());
            if orphan {
                slot.instance = None;
                slot.generation = slot.generation.wrapping_add(1);
                // Invariant: the arena never exceeds u32::MAX slots (checked on insert).
                self.free_instances.push(index as u32);
                self.instance_count -= 1;
                dropped = true;
            }
        }
        if dropped {
            let instances = &self.instances;
            self.draw_order.retain(|&s| {
                instances
                    .get(s as usize)
                    .is_some_and(|slot| slot.instance.is_some())
            });
        }
        Ok(())
    }

    /// Replaces the local transform of `id` and marks it dirty, so the next
    /// [`Scene::update_world`] recomputes it and its subtree.
    pub fn set_local(&mut self, id: NodeId, t: Transform) -> Result<(), SceneError> {
        let node = self.node_mut(id)?;
        node.local = t;
        node.dirty = true;
        Ok(())
    }

    /// The local transform of `id`.
    pub fn local(&self, id: NodeId) -> Result<Transform, SceneError> {
        Ok(self.node(id)?.local)
    }

    /// The world matrix of `id` as of the last [`Scene::update_world`].
    pub fn world(&self, id: NodeId) -> Result<Mat4, SceneError> {
        Ok(self.node(id)?.world)
    }

    /// Number of live nodes.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Recomputes the world matrix of every dirty node and of every node below one:
    /// `world = parent.world * local`, or `local` for roots. Clears the dirty flags.
    ///
    /// Frame path: performs no heap allocation.
    pub fn update_world(&mut self) {
        for &id in &self.order {
            let (parent_world, parent_updated) =
                match Self::lookup(&self.nodes, id).and_then(|n| n.parent) {
                    Some(p) => match Self::lookup(&self.nodes, p) {
                        Some(parent) => (Some(parent.world), parent.updated),
                        None => (None, false),
                    },
                    None => (None, false),
                };
            let Some(node) = Self::lookup_mut(&mut self.nodes, id) else {
                continue;
            };
            let recompute = node.dirty || parent_updated;
            node.updated = recompute;
            if recompute {
                let local = node.local.to_mat4();
                node.world = match parent_world {
                    Some(pw) => pw * local,
                    None => local,
                };
            }
        }
        for &id in &self.order {
            if let Some(node) = Self::lookup_mut(&mut self.nodes, id) {
                node.dirty = false;
                node.updated = false;
            }
        }
    }

    /// Attaches an instance of `mesh` drawn with `material` to `node`. It is visible.
    ///
    /// A stale node gives [`SceneError::StaleNode`].
    pub fn add_instance(
        &mut self,
        node: NodeId,
        mesh: MeshId,
        material: MaterialId,
    ) -> Result<InstanceId, SceneError> {
        self.node(node)?;
        let instance = Instance {
            node,
            mesh,
            material,
            visible: true,
        };
        let id = match self.free_instances.pop() {
            Some(index) => {
                // Invariant: free_instances only holds indices of empty slots.
                let slot = &mut self.instances[index as usize];
                slot.instance = Some(instance);
                InstanceId::new(index, slot.generation)
            }
            None => {
                // Invariant: instance ids are u32; more slots cannot be addressed.
                let index =
                    u32::try_from(self.instances.len()).expect("Scene exceeds u32::MAX instances");
                self.instances.push(InstanceSlot {
                    generation: 0,
                    instance: Some(instance),
                });
                InstanceId::new(index, 0)
            }
        };
        self.insert_draw_order(id.index());
        self.instance_count += 1;
        Ok(id)
    }

    /// Removes an instance. A stale id gives [`SceneError::StaleInstance`].
    pub fn remove_instance(&mut self, id: InstanceId) -> Result<(), SceneError> {
        self.instance(id)?;
        self.remove_draw_order(id.index());
        if let Some(slot) = self.instances.get_mut(id.index() as usize) {
            slot.instance = None;
            slot.generation = slot.generation.wrapping_add(1);
            self.free_instances.push(id.index());
            self.instance_count -= 1;
        }
        Ok(())
    }

    /// Shows or hides an instance. Hidden instances stay in the scene but are left out
    /// of [`Scene::build_render_list`]. A stale id gives [`SceneError::StaleInstance`].
    pub fn set_instance_visible(
        &mut self,
        id: InstanceId,
        visible: bool,
    ) -> Result<(), SceneError> {
        self.instance_mut(id)?.visible = visible;
        Ok(())
    }

    /// Changes the material of an instance (and so its place in the draw order). A
    /// stale id gives [`SceneError::StaleInstance`].
    pub fn set_instance_material(
        &mut self,
        id: InstanceId,
        m: MaterialId,
    ) -> Result<(), SceneError> {
        if self.instance(id)?.material == m {
            return Ok(());
        }
        self.remove_draw_order(id.index());
        self.instance_mut(id)?.material = m;
        self.insert_draw_order(id.index());
        Ok(())
    }

    /// Number of live instances, visible or not.
    pub fn instance_count(&self) -> usize {
        self.instance_count
    }

    /// Fills `out` with the visible instances, ordered by (mesh, material, instance
    /// slot), and with one [`Batch`] per contiguous run of the same mesh. `out` is
    /// cleared first. The result is deterministic: it depends only on the scene state.
    ///
    /// World matrices are read as of the last [`Scene::update_world`]; call it first
    /// when transforms changed.
    ///
    /// Frame path: the order is maintained at mutation time, so nothing is sorted here,
    /// and nothing is allocated when both `out.items` and `out.batches` have room for
    /// the visible instances (for example `RenderList::with_capacity(instance_count)`).
    pub fn build_render_list(&self, out: &mut RenderList) {
        out.clear();
        for &slot in &self.draw_order {
            let Some(inst) = self.instance_at(slot) else {
                continue;
            };
            if !inst.visible {
                continue;
            }
            // Invariant: an instance's node is live (remove_node drops its instances).
            let Some(node) = Self::lookup(&self.nodes, inst.node) else {
                continue;
            };
            let first = out.items.len() as u32;
            out.items.push(RenderItem {
                mesh: inst.mesh,
                material: inst.material,
                world: node.world,
            });
            match out.batches.last_mut() {
                Some(batch) if batch.mesh == inst.mesh => batch.count += 1,
                _ => out.batches.push(Batch {
                    mesh: inst.mesh,
                    first,
                    count: 1,
                }),
            }
        }
    }

    fn instance_at(&self, slot: u32) -> Option<&Instance> {
        self.instances.get(slot as usize)?.instance.as_ref()
    }

    fn instance(&self, id: InstanceId) -> Result<&Instance, SceneError> {
        match self.instances.get(id.index() as usize) {
            Some(slot) if slot.generation == id.generation() => slot.instance.as_ref(),
            _ => None,
        }
        .ok_or(SceneError::StaleInstance)
    }

    fn instance_mut(&mut self, id: InstanceId) -> Result<&mut Instance, SceneError> {
        match self.instances.get_mut(id.index() as usize) {
            Some(slot) if slot.generation == id.generation() => slot.instance.as_mut(),
            _ => None,
        }
        .ok_or(SceneError::StaleInstance)
    }

    /// Compares two live instance slots by (mesh, material, slot).
    fn draw_cmp(&self, a: u32, b: u32) -> Ordering {
        let key = |s: u32| self.instance_at(s).map(|i| (i.mesh, i.material));
        key(a).cmp(&key(b)).then(a.cmp(&b))
    }

    /// Inserts a live slot at its sorted position in `draw_order`.
    fn insert_draw_order(&mut self, slot: u32) {
        let pos = match self
            .draw_order
            .binary_search_by(|&o| self.draw_cmp(o, slot))
        {
            Ok(p) | Err(p) => p,
        };
        self.draw_order.insert(pos, slot);
    }

    /// Removes a live slot from `draw_order` (its key must be unchanged since insert).
    fn remove_draw_order(&mut self, slot: u32) {
        if let Ok(pos) = self
            .draw_order
            .binary_search_by(|&o| self.draw_cmp(o, slot))
        {
            self.draw_order.remove(pos);
        }
    }

    fn lookup(nodes: &[NodeSlot], id: NodeId) -> Option<&Node> {
        match nodes.get(id.index() as usize) {
            Some(slot) if slot.generation == id.generation() => slot.node.as_ref(),
            _ => None,
        }
    }

    fn lookup_mut(nodes: &mut [NodeSlot], id: NodeId) -> Option<&mut Node> {
        match nodes.get_mut(id.index() as usize) {
            Some(slot) if slot.generation == id.generation() => slot.node.as_mut(),
            _ => None,
        }
    }

    fn node(&self, id: NodeId) -> Result<&Node, SceneError> {
        Self::lookup(&self.nodes, id).ok_or(SceneError::StaleNode)
    }

    fn node_mut(&mut self, id: NodeId) -> Result<&mut Node, SceneError> {
        Self::lookup_mut(&mut self.nodes, id).ok_or(SceneError::StaleNode)
    }

    /// Empties the slot of a live `id`, bumps its generation and frees it.
    fn free_node(&mut self, id: NodeId) {
        if let Some(slot) = self.nodes.get_mut(id.index() as usize)
            && slot.generation == id.generation()
            && slot.node.take().is_some()
        {
            slot.generation = slot.generation.wrapping_add(1);
            self.free_nodes.push(id.index());
            self.node_count -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_math::{Quat, Vec3, approx_eq};

    fn at(x: f32, y: f32, z: f32) -> Transform {
        Transform::from_trs(Vec3::new(x, y, z), Quat::IDENTITY, Vec3::ONE)
    }

    fn assert_mat_eq(a: Mat4, b: Mat4) {
        for (ca, cb) in a.cols.iter().zip(b.cols.iter()) {
            for (x, y) in ca.iter().zip(cb.iter()) {
                assert!(approx_eq(*x, *y, 1e-5), "{a:?} != {b:?}");
            }
        }
    }

    #[test]
    fn child_world_is_parent_times_child() {
        let mut scene = Scene::new();
        let parent_t = Transform::from_trs(
            Vec3::new(1.0, 2.0, 3.0),
            Quat::from_rotation_y(0.7),
            Vec3::splat(2.0),
        );
        let child_t = at(0.5, -1.0, 0.25);
        let parent = scene.add_node(None, parent_t).unwrap();
        let child = scene.add_node(Some(parent), child_t).unwrap();
        scene.update_world();
        assert_mat_eq(scene.world(parent).unwrap(), parent_t.to_mat4());
        assert_mat_eq(
            scene.world(child).unwrap(),
            parent_t.to_mat4() * child_t.to_mat4(),
        );
    }

    #[test]
    fn moving_the_parent_moves_the_child() {
        let mut scene = Scene::with_capacity(4, 4);
        let parent = scene.add_node(None, at(0.0, 0.0, 0.0)).unwrap();
        let child = scene.add_node(Some(parent), at(1.0, 0.0, 0.0)).unwrap();
        scene.update_world();
        let before = scene.world(child).unwrap();
        scene.set_local(parent, at(0.0, 5.0, 0.0)).unwrap();
        assert_eq!(scene.world(child).unwrap(), before);
        scene.update_world();
        let p = scene.world(child).unwrap().transform_point(Vec3::ZERO);
        assert_eq!(p, Vec3::new(1.0, 5.0, 0.0));
        assert_eq!(scene.local(parent).unwrap(), at(0.0, 5.0, 0.0));
    }

    #[test]
    fn removed_node_id_is_stale() {
        let mut scene = Scene::new();
        let node = scene.add_node(None, Transform::IDENTITY).unwrap();
        scene.remove_node(node).unwrap();
        let err = scene.world(node).unwrap_err();
        assert_eq!(err.code(), "scene.stale_node");
        assert_eq!(scene.remove_node(node), Err(SceneError::StaleNode));
        assert_eq!(
            scene.set_local(node, Transform::IDENTITY),
            Err(SceneError::StaleNode)
        );
        assert_eq!(
            scene.add_node(Some(node), Transform::IDENTITY),
            Err(SceneError::StaleNode)
        );
        // The reused slot does not revive the old id.
        let fresh = scene.add_node(None, Transform::IDENTITY).unwrap();
        assert_ne!(fresh, node);
        assert_eq!(scene.local(node), Err(SceneError::StaleNode));
    }

    #[test]
    fn removing_a_parent_removes_its_subtree() {
        let mut scene = Scene::new();
        let other = scene.add_node(None, at(9.0, 0.0, 0.0)).unwrap();
        let root = scene.add_node(None, Transform::IDENTITY).unwrap();
        let a = scene.add_node(Some(root), at(1.0, 0.0, 0.0)).unwrap();
        let b = scene.add_node(Some(root), at(2.0, 0.0, 0.0)).unwrap();
        let a1 = scene.add_node(Some(a), at(3.0, 0.0, 0.0)).unwrap();
        let sibling = scene.add_node(Some(other), at(0.0, 1.0, 0.0)).unwrap();
        assert_eq!(scene.node_count(), 6);
        scene.remove_node(root).unwrap();
        assert_eq!(scene.node_count(), 2);
        for id in [root, a, b, a1] {
            assert_eq!(scene.world(id), Err(SceneError::StaleNode));
        }
        assert_eq!(scene.order, vec![other, sibling]);
        scene.update_world();
        let p = scene.world(sibling).unwrap().transform_point(Vec3::ZERO);
        assert_eq!(p, Vec3::new(9.0, 1.0, 0.0));
    }

    #[test]
    fn untouched_subtree_keeps_its_world() {
        let mut scene = Scene::new();
        let moved = scene.add_node(None, at(1.0, 0.0, 0.0)).unwrap();
        let moved_child = scene.add_node(Some(moved), at(0.0, 1.0, 0.0)).unwrap();
        let still = scene.add_node(None, at(-1.0, 0.0, 0.0)).unwrap();
        let still_child = scene.add_node(Some(still), at(0.0, 0.0, 1.0)).unwrap();
        scene.update_world();
        let kept = scene.world(still_child).unwrap();

        scene.set_local(moved, at(4.0, 0.0, 0.0)).unwrap();
        scene.update_world();
        assert_eq!(scene.world(still_child).unwrap(), kept);
        assert_eq!(
            scene
                .world(moved_child)
                .unwrap()
                .transform_point(Vec3::ZERO),
            Vec3::new(4.0, 1.0, 0.0)
        );
        // Flags are cleared after the pass.
        for &id in &scene.order {
            let node = scene.node(id).unwrap();
            assert!(!node.dirty && !node.updated);
        }
    }

    fn keys(list: &RenderList) -> Vec<(u32, u32)> {
        list.items
            .iter()
            .map(|i| (i.mesh.0, i.material.0))
            .collect()
    }

    #[test]
    fn items_are_ordered_by_mesh_material_slot_whatever_the_insertion_order() {
        let mut scene = Scene::with_capacity(4, 8);
        let a = scene.add_node(None, at(1.0, 0.0, 0.0)).unwrap();
        let b = scene.add_node(None, at(2.0, 0.0, 0.0)).unwrap();
        // Slots 0..5 in insertion order.
        scene.add_instance(a, MeshId(2), MaterialId(1)).unwrap();
        scene.add_instance(b, MeshId(1), MaterialId(3)).unwrap();
        scene.add_instance(a, MeshId(1), MaterialId(0)).unwrap();
        scene.add_instance(b, MeshId(2), MaterialId(1)).unwrap();
        scene.add_instance(a, MeshId(1), MaterialId(3)).unwrap();
        scene.update_world();
        let mut list = RenderList::with_capacity(8);
        scene.build_render_list(&mut list);
        assert_eq!(keys(&list), vec![(1, 0), (1, 3), (1, 3), (2, 1), (2, 1)]);
        // Equal keys keep slot order: slot 1 (node b) before slot 4 (node a).
        let x = |i: usize| list.items[i].world.transform_point(Vec3::ZERO).x;
        assert_eq!((x(1), x(2)), (2.0, 1.0));
        assert_eq!((x(3), x(4)), (1.0, 2.0));
        assert_eq!(scene.instance_count(), 5);
    }

    #[test]
    fn batches_are_the_mesh_runs() {
        let mut scene = Scene::new();
        let n = scene.add_node(None, Transform::IDENTITY).unwrap();
        for (mesh, mat) in [(5, 0), (3, 1), (5, 2), (3, 0), (3, 1), (9, 0)] {
            scene
                .add_instance(n, MeshId(mesh), MaterialId(mat))
                .unwrap();
        }
        let mut list = RenderList::default();
        scene.build_render_list(&mut list);
        let batches: Vec<_> = list
            .batches
            .iter()
            .map(|b| (b.mesh.0, b.first, b.count))
            .collect();
        assert_eq!(batches, vec![(3, 0, 3), (5, 3, 2), (9, 5, 1)]);
    }

    #[test]
    fn hidden_instances_are_skipped() {
        let mut scene = Scene::new();
        let n = scene.add_node(None, Transform::IDENTITY).unwrap();
        let i0 = scene.add_instance(n, MeshId(1), MaterialId(0)).unwrap();
        let i1 = scene.add_instance(n, MeshId(1), MaterialId(1)).unwrap();
        scene.add_instance(n, MeshId(2), MaterialId(0)).unwrap();
        scene.set_instance_visible(i1, false).unwrap();
        let mut list = RenderList::default();
        scene.build_render_list(&mut list);
        assert_eq!(keys(&list), vec![(1, 0), (2, 0)]);
        assert_eq!(list.batches.len(), 2);
        assert_eq!(scene.instance_count(), 3);
        scene.set_instance_visible(i0, false).unwrap();
        scene.set_instance_visible(i1, true).unwrap();
        scene.build_render_list(&mut list);
        assert_eq!(keys(&list), vec![(1, 1), (2, 0)]);
    }

    #[test]
    fn set_instance_material_reorders() {
        let mut scene = Scene::new();
        let n = scene.add_node(None, Transform::IDENTITY).unwrap();
        let first = scene.add_instance(n, MeshId(1), MaterialId(0)).unwrap();
        scene.add_instance(n, MeshId(1), MaterialId(5)).unwrap();
        scene.set_instance_material(first, MaterialId(9)).unwrap();
        let mut list = RenderList::default();
        scene.build_render_list(&mut list);
        assert_eq!(keys(&list), vec![(1, 5), (1, 9)]);
        scene.set_instance_material(first, MaterialId(2)).unwrap();
        scene.build_render_list(&mut list);
        assert_eq!(keys(&list), vec![(1, 2), (1, 5)]);
    }

    #[test]
    fn remove_node_drops_its_instances() {
        let mut scene = Scene::new();
        let keep = scene.add_node(None, Transform::IDENTITY).unwrap();
        let root = scene.add_node(None, Transform::IDENTITY).unwrap();
        let child = scene.add_node(Some(root), Transform::IDENTITY).unwrap();
        let kept = scene.add_instance(keep, MeshId(1), MaterialId(0)).unwrap();
        let on_root = scene.add_instance(root, MeshId(0), MaterialId(0)).unwrap();
        let on_child = scene.add_instance(child, MeshId(2), MaterialId(0)).unwrap();
        scene.remove_node(root).unwrap();
        assert_eq!(scene.instance_count(), 1);
        assert_eq!(
            scene.remove_instance(on_root),
            Err(SceneError::StaleInstance)
        );
        assert_eq!(
            scene.set_instance_visible(on_child, true),
            Err(SceneError::StaleInstance)
        );
        let mut list = RenderList::default();
        scene.build_render_list(&mut list);
        assert_eq!(keys(&list), vec![(1, 0)]);
        assert_eq!(scene.draw_order, vec![kept.index()]);
        assert_eq!(
            scene.add_instance(root, MeshId(0), MaterialId(0)),
            Err(SceneError::StaleNode)
        );
    }

    #[test]
    fn stale_instance_is_refused() {
        let mut scene = Scene::new();
        let n = scene.add_node(None, Transform::IDENTITY).unwrap();
        let inst = scene.add_instance(n, MeshId(1), MaterialId(0)).unwrap();
        scene.remove_instance(inst).unwrap();
        assert_eq!(scene.instance_count(), 0);
        let err = scene.remove_instance(inst).unwrap_err();
        assert_eq!(err.code(), "scene.stale_instance");
        assert_eq!(
            scene.set_instance_visible(inst, false),
            Err(SceneError::StaleInstance)
        );
        assert_eq!(
            scene.set_instance_material(inst, MaterialId(3)),
            Err(SceneError::StaleInstance)
        );
        // The reused slot does not revive the old id.
        let fresh = scene.add_instance(n, MeshId(2), MaterialId(0)).unwrap();
        assert_ne!(fresh, inst);
        assert_eq!(fresh.index(), inst.index());
        assert_eq!(scene.remove_instance(inst), Err(SceneError::StaleInstance));
        assert_eq!(
            scene.set_instance_visible(inst, false),
            Err(SceneError::StaleInstance)
        );
        assert_eq!(
            scene.set_instance_material(inst, MaterialId(3)),
            Err(SceneError::StaleInstance)
        );
        assert_eq!(scene.instance_count(), 1);
        let mut list = RenderList::with_capacity(1);
        scene.build_render_list(&mut list);
        assert_eq!(keys(&list), vec![(2, 0)]);
    }

    #[test]
    fn reused_slots_and_material_changes_keep_equal_keys_in_slot_order() {
        let mut scene = Scene::with_capacity(4, 3);
        let nodes = [0.0, 1.0, 2.0, 3.0].map(|x| scene.add_node(None, at(x, 0.0, 0.0)).unwrap());
        let removed = scene
            .add_instance(nodes[0], MeshId(2), MaterialId(4))
            .unwrap();
        let kept = scene
            .add_instance(nodes[1], MeshId(2), MaterialId(4))
            .unwrap();
        let changed = scene
            .add_instance(nodes[2], MeshId(2), MaterialId(9))
            .unwrap();
        scene.remove_instance(removed).unwrap();
        let reused = scene
            .add_instance(nodes[3], MeshId(2), MaterialId(4))
            .unwrap();
        assert_eq!(reused.index(), removed.index());
        assert_ne!(reused, removed);
        scene.set_instance_material(changed, MaterialId(4)).unwrap();
        scene.set_instance_material(kept, MaterialId(4)).unwrap();
        scene.update_world();

        let mut list = RenderList::with_capacity(3);
        scene.build_render_list(&mut list);
        let positions: Vec<_> = list
            .items
            .iter()
            .map(|item| item.world.transform_point(Vec3::ZERO).x)
            .collect();
        assert_eq!(positions, vec![3.0, 1.0, 2.0]);
        assert_eq!(
            scene.draw_order,
            vec![reused.index(), kept.index(), changed.index()]
        );

        scene.remove_instance(reused).unwrap();
        scene.build_render_list(&mut list);
        assert_eq!(list.items.len(), 2);
        assert_eq!(list.batches[0].count, 2);
        assert_eq!(scene.instance_count(), 2);
    }

    #[test]
    fn two_builds_give_identical_items() {
        let mut scene = Scene::new();
        let root = scene.add_node(None, at(0.0, 1.0, 0.0)).unwrap();
        for i in 0..20u32 {
            let n = scene.add_node(Some(root), at(i as f32, 0.0, 0.0)).unwrap();
            scene
                .add_instance(n, MeshId(i % 3), MaterialId(i % 4))
                .unwrap();
        }
        scene.update_world();
        let mut a = RenderList::with_capacity(20);
        let mut b = RenderList::with_capacity(20);
        scene.build_render_list(&mut a);
        scene.build_render_list(&mut b);
        assert_eq!(a, b);
        // Rebuilding into a list with room keeps its buffers.
        let ptr = a.items.as_ptr();
        scene.build_render_list(&mut a);
        assert_eq!(a.items.as_ptr(), ptr);
        assert_eq!(a, b);
    }

    #[test]
    fn build_render_list_reuses_preallocated_item_and_batch_buffers() {
        let mut scene = Scene::with_capacity(1, 6);
        let node = scene.add_node(None, Transform::IDENTITY).unwrap();
        let ids = [5, 4, 3, 2, 1, 0].map(|mesh| {
            let id = scene
                .add_instance(node, MeshId(mesh), MaterialId(0))
                .unwrap();
            scene.set_instance_visible(id, mesh % 2 == 0).unwrap();
            id
        });
        // Capacity only covers the three visible instances, each with its own batch.
        let mut list = RenderList::with_capacity(3);
        let buffers = |list: &RenderList| {
            (
                list.items.capacity(),
                list.items.as_ptr(),
                list.batches.capacity(),
                list.batches.as_ptr(),
            )
        };
        let before = buffers(&list);
        for _ in 0..2 {
            scene.build_render_list(&mut list);
            assert_eq!(list.items.len(), 3);
            assert_eq!(list.batches.len(), 3);
            assert_eq!(buffers(&list), before);
        }

        for id in ids {
            scene.set_instance_visible(id, false).unwrap();
        }
        scene.build_render_list(&mut list);
        assert!(list.items.is_empty());
        assert!(list.batches.is_empty());
        assert_eq!(buffers(&list), before);
    }

    #[test]
    fn build_render_list_reads_worlds_as_of_the_last_update_world() {
        let mut scene = Scene::with_capacity(2, 1);
        let root = scene.add_node(None, at(1.0, 0.0, 0.0)).unwrap();
        let child = scene.add_node(Some(root), at(0.0, 2.0, 0.0)).unwrap();
        scene.add_instance(child, MeshId(0), MaterialId(0)).unwrap();
        let mut list = RenderList::with_capacity(1);
        scene.build_render_list(&mut list);
        assert_eq!(list.items[0].world, Mat4::IDENTITY);

        scene.update_world();
        scene.build_render_list(&mut list);
        assert_eq!(list.items[0].world, at(1.0, 2.0, 0.0).to_mat4());

        scene.set_local(root, at(3.0, 0.0, 0.0)).unwrap();
        scene.build_render_list(&mut list);
        assert_eq!(list.items[0].world, at(1.0, 2.0, 0.0).to_mat4());

        scene.update_world();
        scene.build_render_list(&mut list);
        assert_eq!(list.items[0].world, at(3.0, 2.0, 0.0).to_mat4());
    }
}
