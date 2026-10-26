//! The engine state behind the JS `Engine` (host and wasm32): the scene, the lit renderer,
//! the pass pipeline and the JS handle tables. JS instance handles map to instance groups:
//! one scene instance per sub-mesh of a lit mesh, each tagged with the JS handle so the
//! per-instance extras (S10) of every sub-mesh draw are shared.

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::{MaterialId, MeshId, RaaError};
use raa_csg::CsgResolver;
use raa_geom::{GeometryRegistry, MeshData};
use raa_lit::{LitMeshId, LitRenderer};
use raa_loader::LoaderRegistry;
use raa_pipeline::Pipeline;
use raa_scene::{InstanceId, NodeId, OrbitController, RenderList, Scene};
use raa_tonemap::TonemapControl;

use raa_scene::SceneError;

use crate::convert::{HandleMap, scene_error};
use crate::feature_map::FeatureMap;

/// The material drawn by a sub-mesh without one (the lit renderer's default material).
pub(crate) const DEFAULT_MATERIAL: MaterialId = MaterialId(0);

/// Which API created a mesh handle.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum MeshOrigin {
    /// `primitiveMesh` or `loadPrimsAsset` (S10 default attributes).
    V0,
    /// `addMesh` (m07).
    M07,
}

/// A JS mesh handle: the lit mesh and where it came from.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct MeshEntry {
    pub lit: LitMeshId,
    pub origin: MeshOrigin,
}

/// A JS instance handle: one scene instance per sub-mesh of `mesh` at `node`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct InstanceGroup {
    pub node: NodeId,
    pub mesh: LitMeshId,
    /// One per sub-mesh, in sub-mesh order; empty until built.
    pub instances: Vec<InstanceId>,
    pub visible: bool,
    /// The JS instance handle, set as the tag of every scene instance.
    pub tag: u32,
    /// v0 `addInstance(node, mesh, material)`: every sub-mesh draws this material.
    pub material: Option<MaterialId>,
    /// The node was removed (`removeNode`): the group is never rebuilt and using it
    /// throws `scene.stale_instance`, as a v0 instance of a removed node did.
    pub stale: bool,
}

impl InstanceGroup {
    pub(crate) fn new(
        node: NodeId,
        mesh: LitMeshId,
        tag: u32,
        material: Option<MaterialId>,
    ) -> Self {
        Self {
            node,
            mesh,
            instances: Vec::new(),
            visible: true,
            tag,
            material,
            stale: false,
        }
    }

    /// Replaces the scene instances of the group by one per `(sub-mesh id, sub-mesh
    /// material)` part, keeping node, visibility and tag. The override material wins,
    /// then the sub-mesh material, then [`DEFAULT_MATERIAL`]. Previous instances that are
    /// already stale (their node was removed) are skipped. A stale node refuses with
    /// `scene.stale_node` and marks the group stale; the group then lists only the
    /// instances that exist.
    pub(crate) fn rebuild(
        &mut self,
        scene: &mut Scene,
        parts: impl IntoIterator<Item = (MeshId, Option<MaterialId>)>,
    ) -> Result<(), RaaError> {
        for id in self.instances.drain(..) {
            // A stale instance is already gone from the scene: nothing to remove.
            let _ = scene.remove_instance(id);
        }
        for (mesh, sub_material) in parts {
            let material = self.material.or(sub_material).unwrap_or(DEFAULT_MATERIAL);
            let id = match scene.add_instance(self.node, mesh, material) {
                Ok(id) => id,
                Err(e) => {
                    self.stale |= e == SceneError::StaleNode;
                    return Err(scene_error(&e));
                }
            };
            self.instances.push(id);
            scene
                .set_instance_tag(id, self.tag)
                .map_err(|e| scene_error(&e))?;
            if !self.visible {
                scene
                    .set_instance_visible(id, false)
                    .map_err(|e| scene_error(&e))?;
            }
        }
        Ok(())
    }

    /// Removes every scene instance of the group. All are tried; the first error
    /// (`scene.stale_instance` when the node was removed, as in v0) is returned.
    pub(crate) fn remove_from(&mut self, scene: &mut Scene) -> Result<(), RaaError> {
        let mut first = self.check_live();
        for id in self.instances.drain(..) {
            if let Err(e) = scene.remove_instance(id)
                && first.is_ok()
            {
                first = Err(scene_error(&e));
            }
        }
        first
    }

    /// Shows or hides every scene instance of the group (kept across rebuilds).
    pub(crate) fn set_visible(&mut self, scene: &mut Scene, visible: bool) -> Result<(), RaaError> {
        self.check_live()?;
        self.visible = visible;
        for &id in &self.instances {
            scene
                .set_instance_visible(id, visible)
                .map_err(|e| scene_error(&e))?;
        }
        Ok(())
    }

    /// `scene.stale_instance` when the group's node was removed.
    pub(crate) fn check_live(&self) -> Result<(), RaaError> {
        if self.stale {
            Err(scene_error(&SceneError::StaleInstance))
        } else {
            Ok(())
        }
    }
}

/// Marks stale every live group whose node no longer exists in `scene`.
fn mark_stale_groups(scene: &Scene, groups: &mut HandleMap<InstanceGroup>) {
    for (_, group) in groups.iter_mut() {
        if !group.stale && scene.local(group.node).is_err() {
            group.stale = true;
        }
    }
}

/// Runs `rebuild` on every live (not stale) group whose mesh is in `dirty`, then clears
/// `dirty` (capacity kept). Every group is tried; the first error is returned.
fn rebuild_dirty(
    dirty: &mut Vec<LitMeshId>,
    groups: &mut HandleMap<InstanceGroup>,
    mut rebuild: impl FnMut(&mut InstanceGroup) -> Result<(), RaaError>,
) -> Result<(), RaaError> {
    if dirty.is_empty() {
        return Ok(());
    }
    let mut first = Ok(());
    for (_, group) in groups.iter_mut() {
        if group.stale || !dirty.contains(&group.mesh) {
            continue;
        }
        if let Err(e) = rebuild(group)
            && first.is_ok()
        {
            first = Err(e);
        }
    }
    dirty.clear();
    first
}

/// A loaded asset: one (mesh handle, material handle) per LoadedMesh, in mesh order.
pub(crate) struct AssetEntry {
    pub parts: Vec<(u32, u32)>,
}

/// Everything the engine owns once the GPU is up.
pub(crate) struct EngineState {
    pub scene: Scene,
    pub list: RenderList,
    /// `tags[i]` is the scene tag of `list.items[i]`.
    pub tags: Vec<u32>,
    pub orbit: OrbitController,
    /// The 8 S3 kinds; assets load with strict kinds.
    pub geometry: GeometryRegistry,
    pub loaders: LoaderRegistry,
    pub booleans: CsgResolver,
    /// Reused generation buffer for primitiveMesh.
    pub scratch: MeshData,
    pub lit: LitRenderer,
    pub pipeline: Rc<RefCell<Pipeline>>,
    pub tonemap: TonemapControl,
    pub features: FeatureMap,
    pub max_materials: u32,
    pub meshes: HandleMap<MeshEntry>,
    pub materials: HandleMap<MaterialId>,
    pub nodes: HandleMap<NodeId>,
    pub instances: HandleMap<InstanceGroup>,
    /// Lit meshes whose geometry was replaced since the last frame: their instance groups
    /// are rebuilt by [`EngineState::rebuild_dirty_groups`] in `update`, before
    /// `lit.prepare`.
    pub dirty_meshes: Vec<LitMeshId>,
    /// Asset handle to an index into `assets`.
    pub asset_handles: HandleMap<u32>,
    pub assets: Vec<AssetEntry>,
    /// setTransforms copies the typed arrays here; grown only for a larger input.
    pub scratch_nodes: Vec<u32>,
    pub scratch_trs: Vec<f32>,
}

impl EngineState {
    /// The scene node behind JS handle `h` (`api.unknown_handle`).
    pub(crate) fn node(&self, h: u32) -> Result<NodeId, RaaError> {
        self.nodes.get(h)
    }

    pub(crate) fn material(&self, h: u32) -> Result<MaterialId, RaaError> {
        self.materials.get(h)
    }

    pub(crate) fn mesh(&self, h: u32) -> Result<MeshEntry, RaaError> {
        self.meshes.get(h)
    }

    pub(crate) fn instance_group(&self, h: u32) -> Result<&InstanceGroup, RaaError> {
        self.instances.get_ref(h)
    }

    pub(crate) fn instance_group_mut(&mut self, h: u32) -> Result<&mut InstanceGroup, RaaError> {
        self.instances.get_mut(h)
    }

    /// Creates the instance group of JS handle `next_handle()` for `mesh` at `node` and
    /// returns that handle. `material` overrides every sub-mesh material (v0 addInstance).
    pub(crate) fn add_instance_group(
        &mut self,
        node: NodeId,
        mesh: LitMeshId,
        material: Option<MaterialId>,
    ) -> Result<u32, RaaError> {
        let tag = self.instances.next_handle();
        let mut group = InstanceGroup::new(node, mesh, tag, material);
        if let Err(e) = rebuild_group(&mut self.scene, &self.lit, &mut group) {
            // Leave no half-built group in the scene.
            let _ = group.remove_from(&mut self.scene);
            return Err(e);
        }
        Ok(self.instances.insert(group))
    }

    /// Rebuilds the group of JS handle `h` from its mesh's current sub-meshes.
    pub(crate) fn rebuild_instance_group(&mut self, h: u32) -> Result<(), RaaError> {
        let group = self.instances.get_mut(h)?;
        rebuild_group(&mut self.scene, &self.lit, group)
    }

    /// Removes the group of JS handle `h` (its handle is never reused) and its scene
    /// instances; the instance extras of its tag are cleared. A stale group (its node was
    /// removed) is unmapped too and throws `scene.stale_instance`, as in v0.
    pub(crate) fn remove_instance_group(&mut self, h: u32) -> Result<(), RaaError> {
        let mut group = self.instances.remove(h)?;
        self.lit.clear_instance_extra(group.tag);
        self.lit.handle().set_outline(group.tag, None);
        self.lit.handle().clear_wireframe(group.tag);
        group.remove_from(&mut self.scene)
    }

    /// Removes `node`, its subtree and their scene instances, and marks stale the groups
    /// whose node went away (later use throws `scene.stale_instance`). Their instance
    /// extras are cleared; their handles stay mapped, as in v0.
    pub(crate) fn remove_node(&mut self, node: NodeId) -> Result<(), RaaError> {
        self.scene.remove_node(node).map_err(|e| scene_error(&e))?;
        mark_stale_groups(&self.scene, &mut self.instances);
        for (_, group) in self.instances.iter() {
            if group.stale {
                self.lit.clear_instance_extra(group.tag);
                self.lit.handle().set_outline(group.tag, None);
                self.lit.handle().clear_wireframe(group.tag);
            }
        }
        Ok(())
    }

    /// Records that the geometry of `mesh` was replaced: its instance groups are rebuilt
    /// at the next frame, keeping node, visibility and tag.
    pub(crate) fn mark_mesh_dirty(&mut self, mesh: LitMeshId) {
        if !self.dirty_meshes.contains(&mesh) {
            self.dirty_meshes.push(mesh);
        }
    }

    /// Frees the mesh of JS handle `h` (any origin) and drops it from the dirty list.
    pub(crate) fn remove_mesh(&mut self, h: u32) -> Result<(), RaaError> {
        let mesh = self.meshes.remove(h)?;
        self.dirty_meshes.retain(|&m| m != mesh.lit);
        self.lit.remove(mesh.lit);
        Ok(())
    }

    /// Rebuilds the groups of the meshes marked dirty since the last frame (called by
    /// `update` before `lit.prepare`). Stale groups and groups whose mesh was removed
    /// since are skipped. Nothing is done, and nothing allocated, when the list is empty.
    pub(crate) fn rebuild_dirty_groups(&mut self) -> Result<(), RaaError> {
        let (scene, lit) = (&mut self.scene, &self.lit);
        rebuild_dirty(&mut self.dirty_meshes, &mut self.instances, |group| {
            if lit.submesh_ids(group.mesh).is_none() {
                return Ok(());
            }
            rebuild_group(scene, lit, group)
        })
    }
}

/// Rebuilds `group` from the sub-meshes of its lit mesh (`render.unknown_mesh` when the
/// mesh was removed).
fn rebuild_group(
    scene: &mut Scene,
    lit: &LitRenderer,
    group: &mut InstanceGroup,
) -> Result<(), RaaError> {
    let unknown = || {
        crate::convert::api_error(
            "render.unknown_mesh",
            format!("lit mesh {} does not exist", group.mesh.0),
        )
    };
    let info = lit.info(group.mesh).ok_or_else(unknown)?;
    let ids = lit.submesh_ids(group.mesh).ok_or_else(unknown)?;
    let parts = ids
        .iter()
        .zip(&info.submeshes)
        .map(|(&id, sub)| (id, sub.material));
    group.rebuild(scene, parts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_scene::Transform;

    fn scene_with_node() -> (Scene, NodeId) {
        let mut scene = Scene::new();
        let node = scene.add_node(None, Transform::IDENTITY).unwrap();
        (scene, node)
    }

    fn parts() -> [(MeshId, Option<MaterialId>); 2] {
        [(MeshId(3), Some(MaterialId(5))), (MeshId(4), None)]
    }

    #[test]
    fn rebuild_makes_one_tagged_instance_per_submesh() {
        let (mut scene, node) = scene_with_node();
        let mut g = InstanceGroup::new(node, LitMeshId(0), 7, None);
        g.rebuild(&mut scene, parts()).unwrap();
        assert_eq!(g.instances.len(), 2);
        assert_eq!(scene.instance_count(), 2);
        for &id in &g.instances {
            assert_eq!(scene.instance_tag(id).unwrap(), 7);
        }
        let mut list = RenderList::default();
        let mut tags = Vec::new();
        scene.update_world();
        scene.build_render_list_tagged(&mut list, &mut tags);
        let mut used: Vec<_> = list.items.iter().map(|i| (i.mesh, i.material)).collect();
        used.sort();
        assert_eq!(
            used,
            [(MeshId(3), MaterialId(5)), (MeshId(4), DEFAULT_MATERIAL)]
        );
        assert_eq!(tags, [7, 7]);
    }

    #[test]
    fn override_material_wins_and_rebuild_keeps_visibility() {
        let (mut scene, node) = scene_with_node();
        let mut g = InstanceGroup::new(node, LitMeshId(0), 2, Some(MaterialId(9)));
        g.rebuild(&mut scene, parts()).unwrap();
        g.set_visible(&mut scene, false).unwrap();
        g.rebuild(&mut scene, parts()[..1].iter().copied()).unwrap();
        assert_eq!(scene.instance_count(), 1);
        let mut list = RenderList::default();
        let mut tags = Vec::new();
        scene.update_world();
        scene.build_render_list_tagged(&mut list, &mut tags);
        assert!(list.items.is_empty());
        g.set_visible(&mut scene, true).unwrap();
        scene.build_render_list_tagged(&mut list, &mut tags);
        assert_eq!(list.items.len(), 1);
        assert_eq!(list.items[0].material, MaterialId(9));
    }

    #[test]
    fn removal_clears_the_group_and_reports_stale_instances() {
        let (mut scene, node) = scene_with_node();
        let mut g = InstanceGroup::new(node, LitMeshId(0), 1, None);
        g.rebuild(&mut scene, parts()).unwrap();
        g.remove_from(&mut scene).unwrap();
        assert!(g.instances.is_empty());
        assert_eq!(scene.instance_count(), 0);

        g.rebuild(&mut scene, parts()).unwrap();
        scene.remove_node(node).unwrap();
        let e = g.remove_from(&mut scene).unwrap_err();
        assert_eq!(e.code(), "scene.stale_instance");
        assert!(g.instances.is_empty());
        let e = g.rebuild(&mut scene, parts()).unwrap_err();
        assert_eq!(e.code(), "scene.stale_node");
    }

    #[test]
    fn removed_node_marks_groups_stale_and_use_throws_stale_instance() {
        let (mut scene, node) = scene_with_node();
        let other = scene.add_node(None, Transform::IDENTITY).unwrap();
        let mut groups: HandleMap<InstanceGroup> = HandleMap::new("instance");
        for n in [node, other] {
            let mut g = InstanceGroup::new(n, LitMeshId(0), groups.next_handle(), None);
            g.rebuild(&mut scene, parts()).unwrap();
            groups.insert(g);
        }
        scene.remove_node(node).unwrap();
        mark_stale_groups(&scene, &mut groups);
        assert!(groups.get_ref(1).unwrap().stale);
        assert!(!groups.get_ref(2).unwrap().stale);

        let g = groups.get_mut(1).unwrap();
        let e = g.set_visible(&mut scene, false).unwrap_err();
        assert_eq!(e.code(), "scene.stale_instance");
        let e = g.remove_from(&mut scene).unwrap_err();
        assert_eq!(e.code(), "scene.stale_instance");
        // Still stale with no instance left (e.g. after a failed rebuild).
        let e = g.remove_from(&mut scene).unwrap_err();
        assert_eq!(e.code(), "scene.stale_instance");
        assert_eq!(scene.instance_count(), 2);
    }

    #[test]
    fn dirty_meshes_rebuild_live_groups_keeping_node_visibility_and_tag() {
        let (mut scene, node) = scene_with_node();
        let gone = scene.add_node(None, Transform::IDENTITY).unwrap();
        let mut groups: HandleMap<InstanceGroup> = HandleMap::new("instance");
        for (n, mesh) in [(node, 0), (node, 1), (gone, 0)] {
            let mut g = InstanceGroup::new(n, LitMeshId(mesh), groups.next_handle(), None);
            g.rebuild(&mut scene, parts()).unwrap();
            groups.insert(g);
        }
        groups
            .get_mut(1)
            .unwrap()
            .set_visible(&mut scene, false)
            .unwrap();
        scene.remove_node(gone).unwrap();
        mark_stale_groups(&scene, &mut groups);

        let mut dirty = vec![LitMeshId(0)];
        let replaced = [(MeshId(8), None)];
        let mut rebuilt = Vec::new();
        rebuild_dirty(&mut dirty, &mut groups, |g| {
            rebuilt.push(g.tag);
            g.rebuild(&mut scene, replaced)
        })
        .unwrap();
        assert_eq!(rebuilt, [1]);
        assert!(dirty.is_empty());
        let g = groups.get_ref(1).unwrap();
        assert_eq!((g.node, g.tag, g.visible), (node, 1, false));
        assert_eq!(g.instances.len(), 1);
        assert_eq!(scene.instance_tag(g.instances[0]).unwrap(), 1);
        assert_eq!(groups.get_ref(2).unwrap().instances.len(), 2);
        assert_eq!(scene.instance_count(), 3);

        // An empty list does nothing; a failing rebuild is reported after all are tried.
        rebuild_dirty(&mut dirty, &mut groups, |_| unreachable!()).unwrap();
        let mut dirty = vec![LitMeshId(0), LitMeshId(1)];
        let mut tried = 0;
        let e = rebuild_dirty(&mut dirty, &mut groups, |_| {
            tried += 1;
            Err(scene_error(&SceneError::StaleNode))
        })
        .unwrap_err();
        assert_eq!((e.code(), tried), ("scene.stale_node", 2));
        assert!(dirty.is_empty());
    }

    #[test]
    fn rebuild_on_a_removed_node_marks_the_group_stale() {
        let (mut scene, node) = scene_with_node();
        let mut g = InstanceGroup::new(node, LitMeshId(0), 1, None);
        scene.remove_node(node).unwrap();
        assert!(g.rebuild(&mut scene, parts()).is_err());
        assert!(g.stale);
    }

    #[test]
    fn group_handles_refuse_unknown() {
        let mut groups: HandleMap<InstanceGroup> = HandleMap::new("instance");
        let (_, node) = scene_with_node();
        assert_eq!(groups.next_handle(), 1);
        let h = groups.insert(InstanceGroup::new(node, LitMeshId(0), 1, None));
        assert_eq!(groups.get_ref(h).unwrap().tag, h);
        groups.get_mut(h).unwrap().visible = false;
        assert!(!groups.remove(h).unwrap().visible);
        for h in [0, 1, 2] {
            assert_eq!(groups.get_ref(h).unwrap_err().code(), "api.unknown_handle");
        }
        assert_eq!(groups.next_handle(), 2);
    }
}
