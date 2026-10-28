//! CPU draw-list preparation of the lit renderer: S1 instance bytes and S10 extras bytes
//! in draw order plus one [`LitDraw`] list per [`DrawClass`]. Everything is preallocated
//! from the renderer limits; [`DrawPrep::clear`] keeps the capacity and the transparent
//! sort is in place, so a frame allocates nothing.

use std::collections::HashMap;

use raa_core::{MaterialId, MeshId, RaaError};
use raa_math::{Mat4, Vec3};
use raa_scene::{RenderItem, RenderList};
use raa_shader::MaterialFnId;

use crate::extras::{INSTANCE_EXTRA_GPU_SIZE, InstanceExtra};
use crate::material::{DrawClass, SurfaceMaterial, SurfaceModel};

/// Size in bytes of one S1 instance record.
pub const INSTANCE_GPU_SIZE: usize = 128;

/// One instanced draw: `instance_count` instances from `first_instance` of the instance
/// and extras buffers, all with one sub-mesh and one material.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LitDraw {
    /// Prepared binding id; `DrawPrep::mesh_binding` recovers the source submesh.
    pub mesh: MeshId,
    pub material: MaterialId,
    pub material_fn: MaterialFnId,
    pub class: DrawClass,
    pub first_instance: u32,
    pub instance_count: u32,
    /// View-space distance in front of the camera of the instance's world translation;
    /// for a batch, the nearest of its instances.
    pub view_depth: f32,
}

/// What a material id draws as: its class and its material function.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MaterialClass {
    pub class: DrawClass,
    pub material_fn: MaterialFnId,
}

impl MaterialClass {
    /// The class and function of `m` (flat and standard use the standard function).
    pub fn of(m: &SurfaceMaterial) -> Self {
        let material_fn = match m.model {
            SurfaceModel::Function(id) => id,
            SurfaceModel::Flat | SurfaceModel::Standard => MaterialFnId::STANDARD,
        };
        MaterialClass {
            class: m.class(),
            material_fn,
        }
    }
}

/// The inputs of one [`DrawPrep::prepare`].
pub struct PrepInput<'a> {
    /// The render list (sorted by mesh, material, slot) and its tags (`tags[i]` is the
    /// tag of `list.items[i]`), as built by `Scene::build_render_list_tagged`.
    pub list: &'a RenderList,
    pub tags: &'a [u32],
    /// Indexed by `MaterialId`: one entry per material of the library.
    pub materials: &'a [MaterialClass],
    /// Instance extras by tag; a tag without an entry gets `InstanceExtra::default()`.
    pub extras: &'a HashMap<u32, InstanceExtra>,
    /// The camera view matrix (for transparent depths).
    pub view: Mat4,
    /// Whether a sub-mesh id is uploaded.
    pub known_mesh: &'a dyn Fn(MeshId) -> bool,
    /// Whether a class has a registered handler (Opaque is always handled).
    pub class_handled: &'a dyn Fn(DrawClass) -> bool,
}

/// Capacities of the preallocated buffers (to check that a frame does not grow them).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PrepCapacity {
    pub instances: usize,
    pub extras: usize,
    pub opaque: usize,
    pub fur: usize,
    pub transparent: usize,
    pub sort: usize,
}

/// A transparent instance to sort: view depth, tag, index into the render list.
#[derive(Copy, Clone, Debug)]
struct SortKey {
    depth: f32,
    tag: u32,
    item: u32,
}

/// The CPU draw-list builder; see the module docs.
#[derive(Debug)]
pub struct DrawPrep {
    max_instances: u32,
    max_materials: u32,
    instances: Vec<u8>,
    extras: Vec<u8>,
    opaque: Vec<LitDraw>,
    fur: Vec<LitDraw>,
    transparent: Vec<LitDraw>,
    sort: Vec<SortKey>,
    bindings: HashMap<MeshId, (MeshId, u32)>,
}

impl DrawPrep {
    /// Room for `max_instances` instances (and as many draws per class) and a library of
    /// at most `max_materials` materials.
    pub fn new(max_instances: u32, max_materials: u32) -> Self {
        let n = max_instances as usize;
        DrawPrep {
            max_instances,
            max_materials,
            instances: Vec::with_capacity(n * INSTANCE_GPU_SIZE),
            extras: Vec::with_capacity(n * INSTANCE_EXTRA_GPU_SIZE),
            opaque: Vec::with_capacity(n),
            fur: Vec::with_capacity(n),
            transparent: Vec::with_capacity(n),
            sort: Vec::with_capacity(n),
            bindings: HashMap::with_capacity(n),
        }
    }

    /// Empties every list and keeps the capacity.
    pub fn clear(&mut self) {
        self.instances.clear();
        self.extras.clear();
        self.opaque.clear();
        self.fur.clear();
        self.transparent.clear();
        self.sort.clear();
        self.bindings.clear();
    }

    /// S1 instance records (128 B each) in draw order: opaque, then fur, then transparent.
    pub fn instance_bytes(&self) -> &[u8] {
        &self.instances
    }

    /// S10 `InstanceExtraGpu` records (128 B each) in the same order as the instances.
    pub fn extra_bytes(&self) -> &[u8] {
        &self.extras
    }

    /// Number of instances written.
    pub fn instance_count(&self) -> u32 {
        (self.instances.len() / INSTANCE_GPU_SIZE) as u32
    }

    /// The draws of `class` in submission order (transparent: back to front).
    pub fn draws(&self, class: DrawClass) -> &[LitDraw] {
        match class {
            DrawClass::Opaque => &self.opaque,
            DrawClass::Fur => &self.fur,
            DrawClass::Transparent => &self.transparent,
        }
    }

    /// Current capacities of the preallocated buffers.
    pub fn capacity(&self) -> PrepCapacity {
        PrepCapacity {
            instances: self.instances.capacity(),
            extras: self.extras.capacity(),
            opaque: self.opaque.capacity(),
            fur: self.fur.capacity(),
            transparent: self.transparent.capacity(),
            sort: self.sort.capacity(),
        }
    }

    /// Fills the instance and extras bytes and the draw lists from `input`. Opaque and fur
    /// draws are batched by (mesh, material) runs of the list; transparent draws are one
    /// per instance, sorted back to front by the view depth of the world translation, ties
    /// by tag (then list order).
    ///
    /// Refusals (the lists are left empty): `render.too_many_instances` (more items than
    /// `max_instances`), `render.too_many_materials` (a library above `max_materials`),
    /// `render.invalid_tags` (tags and items differ in length), `render.unknown_mesh`,
    /// `material.unknown_id` (an item material outside the library),
    /// `material.blend_unavailable` (a material whose class has no handler).
    pub fn prepare(&mut self, input: &PrepInput<'_>) -> Result<(), RaaError> {
        self.prepare_skinned(input, &|_| false)
    }

    /// Splits posed instances and preserves their source submesh and scene tag.
    pub fn prepare_skinned(
        &mut self,
        input: &PrepInput<'_>,
        skinned: &dyn Fn(u32) -> bool,
    ) -> Result<(), RaaError> {
        self.clear();
        self.validate(input)?;
        self.fill(input, skinned);
        Ok(())
    }

    fn validate(&self, input: &PrepInput<'_>) -> Result<(), RaaError> {
        let items = &input.list.items;
        if items.len() > self.max_instances as usize {
            return Err(refusal(
                "render.too_many_instances",
                format!(
                    "{} instances exceed max_instances {}",
                    items.len(),
                    self.max_instances
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
        if input.tags.len() != items.len() {
            return Err(refusal(
                "render.invalid_tags",
                format!("{} tags for {} render items", input.tags.len(), items.len()),
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

    /// Writes everything; `input` is validated, so every material lookup succeeds.
    fn fill(&mut self, input: &PrepInput<'_>, skinned: &dyn Fn(u32) -> bool) {
        self.fill_batched(input, DrawClass::Opaque, skinned);
        self.fill_batched(input, DrawClass::Fur, skinned);
        self.fill_transparent(input, skinned);
    }

    fn fill_batched(
        &mut self,
        input: &PrepInput<'_>,
        class: DrawClass,
        skinned: &dyn Fn(u32) -> bool,
    ) {
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
            let first_instance = self.instance_count();
            let mut depth = f32::INFINITY;
            let mut j = i;
            while let Some(other) = items.get(j) {
                if other.mesh != item.mesh || other.material != item.material {
                    break;
                }
                if j > i && (skinned(input.tags[i]) || skinned(input.tags[j])) {
                    break;
                }
                depth = depth.min(view_depth(&input.view, other));
                self.push_instance(input, j);
                j += 1;
            }
            let draw = LitDraw {
                mesh: self.draw_binding(input, item.mesh, input.tags[i], skinned),
                material: item.material,
                material_fn: mc.material_fn,
                class,
                first_instance,
                instance_count: (j - i) as u32,
                view_depth: depth,
            };
            match class {
                DrawClass::Fur => self.fur.push(draw),
                _ => self.opaque.push(draw),
            }
            i = j;
        }
    }

    fn fill_transparent(&mut self, input: &PrepInput<'_>, skinned: &dyn Fn(u32) -> bool) {
        let items = &input.list.items;
        for (i, item) in items.iter().enumerate() {
            let transparent = material_class(input, item.material)
                .is_ok_and(|mc| mc.class == DrawClass::Transparent);
            if transparent {
                self.sort.push(SortKey {
                    depth: view_depth(&input.view, item),
                    tag: input.tags.get(i).copied().unwrap_or(0),
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
        for k in 0..self.sort.len() {
            // Invariant: k < len; push_instance below does not touch `sort`.
            let Some(&key) = self.sort.get(k) else {
                continue;
            };
            let Some(item) = items.get(key.item as usize) else {
                continue;
            };
            let Ok(mc) = material_class(input, item.material) else {
                continue;
            };
            let first_instance = self.instance_count();
            self.push_instance(input, key.item as usize);
            let mesh = self.draw_binding(input, item.mesh, key.tag, skinned);
            self.transparent.push(LitDraw {
                mesh,
                material: item.material,
                material_fn: mc.material_fn,
                class: DrawClass::Transparent,
                first_instance,
                instance_count: 1,
                view_depth: key.depth,
            });
        }
    }

    /// Resolves a frame-local draw id to its original submesh and optional instance tag.
    pub fn mesh_binding(&self, mesh: MeshId) -> (MeshId, Option<u32>) {
        self.bindings
            .get(&mesh)
            .map_or((mesh, None), |&(m, tag)| (m, Some(tag)))
    }

    fn draw_binding(
        &mut self,
        input: &PrepInput<'_>,
        mesh: MeshId,
        tag: u32,
        skinned: &dyn Fn(u32) -> bool,
    ) -> MeshId {
        if !skinned(tag) {
            return mesh;
        }
        let mut id = MeshId(u32::MAX - self.bindings.len() as u32);
        while (input.known_mesh)(id) || self.bindings.contains_key(&id) {
            id.0 -= 1;
        }
        self.bindings.insert(id, (mesh, tag));
        id
    }

    /// Appends the instance and extras records of list item `i`.
    fn push_instance(&mut self, input: &PrepInput<'_>, i: usize) {
        let Some(item) = input.list.items.get(i) else {
            return;
        };
        let mut record = [0u8; INSTANCE_GPU_SIZE];
        write_instance(&mut record, &item.world, item.material.0);
        let tag = input.tags.get(i).copied().unwrap_or(0);
        record[116..120].copy_from_slice(&tag.to_le_bytes());
        self.instances.extend_from_slice(&record);
        input
            .extras
            .get(&tag)
            .copied()
            .unwrap_or_default()
            .write_gpu(&mut self.extras);
    }
}

fn refusal(code: &'static str, message: String) -> RaaError {
    RaaError::FrameGraph { code, message }
}

fn material_class(input: &PrepInput<'_>, id: MaterialId) -> Result<MaterialClass, RaaError> {
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

/// Writes `values` as consecutive f32 LE into `out`.
fn put_f32s(out: &mut [u8], values: &[f32]) {
    for (chunk, v) in out.chunks_exact_mut(4).zip(values) {
        chunk.copy_from_slice(&v.to_le_bytes());
    }
}

/// S1 instance: model @0, normal matrix columns @64, material index @112, zero padding.
fn write_instance(out: &mut [u8; INSTANCE_GPU_SIZE], world: &Mat4, material: u32) {
    world.write_le_bytes(&mut out[0..64]);
    put_f32s(&mut out[64..112], world.normal_matrix().as_flattened());
    out[112..116].copy_from_slice(&material.to_le_bytes());
    out[116..128].fill(0);
}
