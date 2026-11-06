//! Compacted visible id lists per mesh, material and LOD level.
//!
//! [`IdPrep`] is the id-list successor of `prep::DrawPrep`: it validates the same way,
//! keeps the same transparent sort and the same rest-then-posed grouping, but writes u32
//! instance slots instead of the 128-byte S1 records (those live in
//! [`InstanceStore`](crate::InstanceStore), addressed by the same slot). One camera
//! section plus one section per shadow cascade make one contiguous id list; each
//! [`LitDraw`] indexes it with its instance range, so `@builtin(instance_index)` reads a
//! slot id and every (level mesh, material) batch stays one draw.
//!
//! All buffers are preallocated in [`IdPrep::new`]; a frame allocates nothing.

use std::collections::HashMap;

use raa_core::{MaterialId, MeshId, RaaError};
use raa_math::{Mat4, Vec3};
use raa_scene::{RenderItem, RenderList};

use crate::material::DrawClass;
use crate::prep::{LitDraw, MaterialClass};
use crate::skin_views::SkinDraw;

/// Number of shadow cascade sections after the camera.
const CASCADES: usize = 4;

/// The inputs of one [`IdPrep`] camera or cascade preparation.
pub struct SelectInput<'a> {
    /// The full scene list in draw order (mesh, material, slot).
    pub list: &'a RenderList,
    /// `tags[i]` is the tag of `list.items[i]`.
    pub tags: &'a [u32],
    /// `slots[i]` is the stable scene instance slot of `list.items[i]`.
    pub slots: &'a [u32],
    /// Whether list item `i` (a list index) is visible in this call.
    pub selected: &'a dyn Fn(usize) -> bool,
    /// Indexed by `MaterialId`: one entry per material of the library.
    pub materials: &'a [MaterialClass],
    /// The camera view matrix (for transparent depths).
    pub view: Mat4,
    /// Whether a sub-mesh id is uploaded.
    pub known_mesh: &'a dyn Fn(MeshId) -> bool,
    /// Whether a class has a registered handler (Opaque is always handled).
    pub class_handled: &'a dyn Fn(DrawClass) -> bool,
    /// The level mesh of a rest instance; never called for a posed instance.
    pub level_mesh: &'a dyn Fn(u32, MeshId) -> MeshId,
}

/// Capacities of the preallocated id, draw and sort buffers.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SelectCapacity {
    pub ids: usize,
    pub draws: usize,
    pub sort: usize,
}

/// A transparent instance to sort: view depth, tag, index into the render list.
#[derive(Copy, Clone, Debug)]
struct SortKey {
    depth: f32,
    tag: u32,
    item: u32,
}

/// One selected item of a (mesh, material) run, grouped by level mesh or skin key.
#[derive(Copy, Clone, Debug)]
struct RunItem {
    posed: bool,
    key: u64,
    level: MeshId,
    item: u32,
}

/// Which id section a fill writes: the camera or one cascade.
#[derive(Copy, Clone, Debug)]
enum Section {
    Camera,
    Cascade(usize),
}

/// The CPU id-list builder; see the module docs.
#[derive(Debug)]
pub struct IdPrep {
    max_instances: u32,
    max_materials: u32,
    ids: Vec<u8>,
    sort: Vec<SortKey>,
    camera: [Vec<LitDraw>; 3],
    cascades: [[Vec<LitDraw>; 3]; CASCADES],
    camera_count: u32,
    prepared_cascade: [bool; CASCADES],
    bindings: HashMap<MeshId, (MeshId, u32)>,
    run: Vec<RunItem>,
}

impl IdPrep {
    /// Room for `max_instances` selected items per section and a library of at most
    /// `max_materials` materials.
    pub fn new(max_instances: u32, max_materials: u32) -> Self {
        let n = max_instances as usize;
        IdPrep {
            max_instances,
            max_materials,
            ids: Vec::with_capacity(n * 5 * 4),
            sort: Vec::with_capacity(n),
            camera: std::array::from_fn(|_| Vec::with_capacity(n)),
            cascades: std::array::from_fn(|_| std::array::from_fn(|_| Vec::with_capacity(n))),
            camera_count: 0,
            prepared_cascade: [false; CASCADES],
            bindings: HashMap::with_capacity(n),
            run: Vec::with_capacity(n),
        }
    }

    /// Empties every section and keeps the capacity.
    fn clear_sections(&mut self) {
        self.ids.clear();
        self.sort.clear();
        self.bindings.clear();
        for draws in &mut self.camera {
            draws.clear();
        }
        for section in &mut self.cascades {
            for draws in section {
                draws.clear();
            }
        }
        self.camera_count = 0;
        self.prepared_cascade = [false; CASCADES];
    }

    /// u32 LE instance slots of every section, camera first.
    pub fn id_bytes(&self) -> &[u8] {
        &self.ids
    }

    /// Number of ids written in every section.
    pub fn id_count(&self) -> u32 {
        (self.ids.len() / 4) as u32
    }

    /// Number of camera ids (the cascade sections start here).
    pub fn camera_count(&self) -> u32 {
        self.camera_count
    }

    /// The camera draws of `class` in submission order (transparent: back to front).
    pub fn draws(&self, class: DrawClass) -> &[LitDraw] {
        &self.camera[class_index(class)]
    }

    /// The cascade draws of `class`; empty for Transparent or a cascade not prepared.
    pub fn cascade_draws(&self, cascade: usize, class: DrawClass) -> &[LitDraw] {
        if cascade >= CASCADES || class == DrawClass::Transparent {
            return &[];
        }
        if !self.prepared_cascade[cascade] {
            return &[];
        }
        &self.cascades[cascade][class_index(class)]
    }

    /// Current capacities of the preallocated buffers.
    pub fn capacity(&self) -> SelectCapacity {
        let mut draws = 0;
        for list in self.camera.iter().chain(self.cascades.iter().flatten()) {
            draws = draws.max(list.capacity());
        }
        SelectCapacity {
            ids: self.ids.capacity() / 4,
            draws,
            sort: self.sort.capacity(),
        }
    }

    /// Clears every section and writes the camera section from `input`.
    ///
    /// Refusals (every section is left empty): `render.too_many_instances` (more selected
    /// items than `max_instances`), `render.too_many_materials`, `render.invalid_tags`
    /// (tags or slots differ in length from the items), `render.unknown_mesh` (an item
    /// mesh or a level mesh is not uploaded), `material.unknown_id` and
    /// `material.blend_unavailable`.
    pub fn prepare_camera(
        &mut self,
        input: &SelectInput<'_>,
        skinned: &dyn Fn(u32) -> Option<SkinDraw>,
    ) -> Result<(), RaaError> {
        self.clear_sections();
        self.validate(input)?;
        let result = (|| {
            self.fill_batched(input, skinned, DrawClass::Opaque, Section::Camera)?;
            self.fill_batched(input, skinned, DrawClass::Fur, Section::Camera)?;
            self.fill_transparent(input, skinned)?;
            Ok(())
        })();
        if result.is_err() {
            self.clear_sections();
            return result;
        }
        self.camera_count = self.id_count();
        Ok(())
    }

    /// Appends the Opaque and Fur caster draws of `cascade` after the existing sections;
    /// call order is camera, then cascades 0..3.
    pub fn prepare_cascade(
        &mut self,
        cascade: usize,
        input: &SelectInput<'_>,
        skinned: &dyn Fn(u32) -> Option<SkinDraw>,
    ) -> Result<(), RaaError> {
        if cascade >= CASCADES {
            return Ok(());
        }
        let first = self.id_count();
        for draws in &mut self.cascades[cascade] {
            draws.clear();
        }
        self.prepared_cascade[cascade] = false;
        self.validate(input)?;
        let result = (|| {
            self.fill_batched(input, skinned, DrawClass::Opaque, Section::Cascade(cascade))?;
            self.fill_batched(input, skinned, DrawClass::Fur, Section::Cascade(cascade))?;
            Ok(())
        })();
        if result.is_err() {
            self.ids.truncate(first as usize * 4);
            for draws in &mut self.cascades[cascade] {
                draws.clear();
            }
            return result;
        }
        self.prepared_cascade[cascade] = true;
        Ok(())
    }

    /// Resolves a frame-local draw id to its original submesh and optional instance tag.
    pub fn mesh_binding(&self, mesh: MeshId) -> (MeshId, Option<u32>) {
        self.bindings
            .get(&mesh)
            .map_or((mesh, None), |&(m, tag)| (m, Some(tag)))
    }

    fn validate(&self, input: &SelectInput<'_>) -> Result<(), RaaError> {
        let items = &input.list.items;
        let selected = items
            .iter()
            .enumerate()
            .filter(|(i, _)| (input.selected)(*i))
            .count();
        if selected > self.max_instances as usize {
            return Err(refusal(
                "render.too_many_instances",
                format!(
                    "{} selected instances exceed max_instances {}",
                    selected, self.max_instances
                ),
            ));
        }
        if input.materials.len() > self.max_materials as usize {
            return Err(refusal(
                "render.too_many_materials",
                format!(
                    "{} materials exceed max_materials {}",
                    input.materials.len(),
                    self.max_materials
                ),
            ));
        }
        if input.tags.len() != items.len() || input.slots.len() != items.len() {
            return Err(refusal(
                "render.invalid_tags",
                format!(
                    "{} tags and {} slots for {} render items",
                    input.tags.len(),
                    input.slots.len(),
                    items.len()
                ),
            ));
        }
        for item in items {
            if !(input.known_mesh)(item.mesh) {
                return Err(refusal(
                    "render.unknown_mesh",
                    format!("mesh {} is not uploaded", item.mesh.0),
                ));
            }
            let mc = material_class(input, item.material)?;
            if mc.class != DrawClass::Opaque && !(input.class_handled)(mc.class) {
                return Err(refusal(
                    "material.blend_unavailable",
                    format!(
                        "material {} is drawn as {:?} and no {:?} class handler is registered",
                        item.material.0, mc.class, mc.class
                    ),
                ));
            }
        }
        Ok(())
    }

    fn fill_batched(
        &mut self,
        input: &SelectInput<'_>,
        skinned: &dyn Fn(u32) -> Option<SkinDraw>,
        class: DrawClass,
        section: Section,
    ) -> Result<(), RaaError> {
        let items = &input.list.items;
        let mut i = 0;
        while let Some(item) = items.get(i) {
            let Ok(mc) = material_class(input, item.material) else {
                i += 1;
                continue;
            };
            if mc.class != class {
                i += 1;
                continue;
            }
            let mut end = i + 1;
            while items
                .get(end)
                .is_some_and(|o| o.mesh == item.mesh && o.material == item.material)
            {
                end += 1;
            }
            // Rest instances first, bucketed by level mesh, then posed ones by skin key.
            self.run.clear();
            for j in i..end {
                if !(input.selected)(j) {
                    continue;
                }
                let tag = input.tags[j];
                if let Some(draw) = skinned(tag) {
                    self.run.push(RunItem {
                        posed: true,
                        key: draw.skin,
                        level: item.mesh,
                        item: j as u32,
                    });
                } else {
                    let level = (input.level_mesh)(input.slots[j], item.mesh);
                    if !(input.known_mesh)(level) {
                        return Err(refusal(
                            "render.unknown_mesh",
                            format!("mesh {} is not uploaded", level.0),
                        ));
                    }
                    self.run.push(RunItem {
                        posed: false,
                        key: level.0 as u64,
                        level,
                        item: j as u32,
                    });
                }
            }
            self.run.sort_unstable_by_key(|r| (r.posed, r.key, r.item));
            let mut k = 0;
            while k < self.run.len() {
                let first = self.run[k];
                let first_instance = self.id_count();
                let mut depth = f32::INFINITY;
                while let Some(&run_item) = self.run.get(k) {
                    if run_item.posed != first.posed || run_item.key != first.key {
                        break;
                    }
                    if let Some(it) = items.get(run_item.item as usize) {
                        depth = depth.min(view_depth(&input.view, it));
                    }
                    self.push_id(input.slots[run_item.item as usize]);
                    k += 1;
                }
                let tag = input.tags[first.item as usize];
                let mesh = if first.posed {
                    self.draw_binding(input, item.mesh, tag, true)
                } else {
                    first.level
                };
                let draw = LitDraw {
                    mesh,
                    material: item.material,
                    material_fn: mc.material_fn,
                    class,
                    first_instance,
                    instance_count: self.id_count() - first_instance,
                    view_depth: depth,
                };
                self.draws_mut(section, class).push(draw);
            }
            i = end;
        }
        Ok(())
    }

    fn fill_transparent(
        &mut self,
        input: &SelectInput<'_>,
        skinned: &dyn Fn(u32) -> Option<SkinDraw>,
    ) -> Result<(), RaaError> {
        let items = &input.list.items;
        for (i, item) in items.iter().enumerate() {
            if !(input.selected)(i) {
                continue;
            }
            let transparent = material_class(input, item.material)
                .is_ok_and(|mc| mc.class == DrawClass::Transparent);
            if transparent {
                self.sort.push(SortKey {
                    depth: view_depth(&input.view, item),
                    tag: input.tags[i],
                    item: i as u32,
                });
            }
        }
        // In place (no scratch buffer): far first, then tag, then list order.
        self.sort.sort_unstable_by(|a, b| {
            b.depth
                .total_cmp(&a.depth)
                .then(a.tag.cmp(&b.tag))
                .then(a.item.cmp(&b.item))
        });
        let mut k = 0;
        while let Some(&key) = self.sort.get(k) {
            k += 1;
            let Some(item) = items.get(key.item as usize) else {
                continue;
            };
            let Ok(mc) = material_class(input, item.material) else {
                continue;
            };
            let posed = skinned(key.tag);
            let first_instance = self.id_count();
            self.push_id(input.slots[key.item as usize]);
            let mesh = if posed.is_some() {
                self.draw_binding(input, item.mesh, key.tag, true)
            } else {
                let level = (input.level_mesh)(input.slots[key.item as usize], item.mesh);
                if !(input.known_mesh)(level) {
                    return Err(refusal(
                        "render.unknown_mesh",
                        format!("mesh {} is not uploaded", level.0),
                    ));
                }
                level
            };
            self.camera[class_index(DrawClass::Transparent)].push(LitDraw {
                mesh,
                material: item.material,
                material_fn: mc.material_fn,
                class: DrawClass::Transparent,
                first_instance,
                instance_count: 1,
                view_depth: key.depth,
            });
        }
        Ok(())
    }

    fn push_id(&mut self, slot: u32) {
        self.ids.extend_from_slice(&slot.to_le_bytes());
    }

    fn draws_mut(&mut self, section: Section, class: DrawClass) -> &mut Vec<LitDraw> {
        match section {
            Section::Camera => &mut self.camera[class_index(class)],
            Section::Cascade(index) => &mut self.cascades[index][class_index(class)],
        }
    }

    fn draw_binding(
        &mut self,
        input: &SelectInput<'_>,
        mesh: MeshId,
        tag: u32,
        posed: bool,
    ) -> MeshId {
        if !posed {
            return mesh;
        }
        let mut id = MeshId(u32::MAX - self.bindings.len() as u32);
        // A real uploaded sub-mesh id is never near the top of the range; the attempt
        // bound keeps a permissive `known_mesh` from scanning the whole space.
        let mut attempts = 0;
        while ((input.known_mesh)(id) || self.bindings.contains_key(&id))
            && attempts <= self.max_instances as usize
        {
            id.0 = id.0.wrapping_sub(1);
            attempts += 1;
        }
        while self.bindings.contains_key(&id) {
            id.0 = id.0.wrapping_sub(1);
        }
        self.bindings.insert(id, (mesh, tag));
        id
    }
}

fn class_index(class: DrawClass) -> usize {
    match class {
        DrawClass::Opaque => 0,
        DrawClass::Fur => 1,
        DrawClass::Transparent => 2,
    }
}

fn refusal(code: &'static str, message: String) -> RaaError {
    RaaError::FrameGraph { code, message }
}

fn material_class(input: &SelectInput<'_>, id: MaterialId) -> Result<MaterialClass, RaaError> {
    input.materials.get(id.0 as usize).copied().ok_or_else(|| {
        refusal(
            "material.unknown_id",
            format!(
                "material {} is not in the library ({} materials)",
                id.0,
                input.materials.len()
            ),
        )
    })
}

/// Distance in front of the camera of the item's world translation.
fn view_depth(view: &Mat4, item: &RenderItem) -> f32 {
    let t = item.world.cols[3];
    -view.transform_point(Vec3::new(t[0], t[1], t[2])).z
}
