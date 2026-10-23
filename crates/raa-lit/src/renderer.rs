//! [`LitRenderer`]: the owner of the lit renderer's state. It uploads meshes, holds the
//! surface materials (default material id 0, grey standard), the material functions and
//! slot providers, the class handlers, the instance extras by tag and the forward
//! contributors, and prepares each frame: scene world update, tagged render list, draw
//! lists ([`DrawPrep`]), `FrameGpu` and the queue writes. Every frame buffer is sized at
//! [`LitRenderer::new`] from [`LitOptions`], so a warm frame allocates nothing.
//!
//! The renderer's state is shared with passes and contributors through [`LitHandle`]
//! (`Rc<RefCell<..>>`), so the accessors into it return `Ref`/`RefMut` guards.

use std::cell::{Cell, Ref, RefCell, RefMut};
use std::collections::HashMap;
use std::rc::Rc;

use raa_core::{GpuContext, MaterialId, MeshId, RaaError};
use raa_mesh::AttributedMesh;
use raa_pipeline::Pipeline;
use raa_scene::{RenderList, Scene};
use raa_shader::{Composer, MaterialFnId, MaterialFnRegistry, SlotProvider};

use crate::contributor::ForwardContributor;
use crate::extras::InstanceExtra;
use crate::frame::{FRAME_GPU_SIZE, FrameParams, write_frame_gpu};
use crate::handle::{LitHandle, LitShared};
use crate::library::SurfaceLibrary;
use crate::material::{DrawClass, SurfaceMaterial};
use crate::meshes::{LitMeshId, LitMeshInfo};
use crate::passes::install_passes;
use crate::prep::{MaterialClass, PrepInput};

/// Limits and sample count of a [`LitRenderer`]; the frame buffers are sized from them.
#[derive(Copy, Clone, Debug)]
pub struct LitOptions {
    /// Instances per frame (instance stream and extras array).
    pub max_instances: u32,
    /// Materials of the surface library (material array).
    pub max_materials: u32,
    /// MSAA samples of the forward pass: 1 or 4.
    pub msaa: u32,
}

impl Default for LitOptions {
    fn default() -> Self {
        LitOptions {
            max_instances: 16384,
            max_materials: 256,
            msaa: 4,
        }
    }
}

impl LitOptions {
    /// Refuses `render.invalid_msaa` unless `msaa` is 1 or 4.
    pub(crate) fn validate(&self) -> Result<(), RaaError> {
        if self.msaa == 1 || self.msaa == 4 {
            Ok(())
        } else {
            Err(refusal(
                "render.invalid_msaa",
                format!("msaa must be 1 or 4 samples, got {}", self.msaa),
            ))
        }
    }
}

/// What one [`LitRenderer::prepare`] submitted.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct FrameCounts {
    /// Instances written to the instance stream.
    pub instances: u32,
    /// Instanced draw calls of every class.
    pub draw_calls: u32,
    /// Triangles drawn (sub-mesh triangles x instances).
    pub triangles: u64,
}

/// One forward contributor with its pipeline entry flag (the forward pass records it only
/// while `enabled` is set) and its order.
pub(crate) struct ContributorSlot {
    pub(crate) order: u32,
    pub(crate) enabled: Rc<Cell<bool>>,
    pub(crate) contributor: Box<dyn ForwardContributor>,
}

/// The contributors in recording order (ascending order, then registration order), shared
/// with the forward pass.
pub(crate) type Contributors = Rc<RefCell<Vec<ContributorSlot>>>;

/// Index at which a contributor of `order` is inserted: after every slot of an order lower
/// than or equal to it.
fn insert_position(slots: &[ContributorSlot], order: u32) -> usize {
    slots.partition_point(|s| s.order <= order)
}

fn refusal(code: &'static str, message: String) -> RaaError {
    RaaError::FrameGraph { code, message }
}

pub(crate) fn pipeline_busy() -> RaaError {
    refusal(
        "pipeline.busy",
        "the pipeline is borrowed (called while a frame executes)".to_string(),
    )
}

/// The HDR lit renderer; see the module docs.
pub struct LitRenderer {
    opts: LitOptions,
    shared: Rc<RefCell<LitShared>>,
    pipeline: Rc<RefCell<Pipeline>>,
    /// Pass name per handled class (Opaque is always handled).
    class_handlers: HashMap<DrawClass, String>,
    /// Instance extras by scene instance tag.
    extras: HashMap<u32, InstanceExtra>,
    contributors: Contributors,
    /// Class and function per material id, rebuilt when the library changed; capacity
    /// `max_materials`.
    material_classes: Vec<MaterialClass>,
    /// The frame's `FrameGpu` bytes.
    frame_bytes: [u8; FRAME_GPU_SIZE],
    /// Frames prepared so far (`FrameGpu.clip.z`).
    frame_index: u64,
    /// A background that could not reach the pipeline (borrowed), applied at `prepare`.
    pending_background: Option<[f32; 3]>,
}

impl LitRenderer {
    /// A renderer drawing through `pipeline`: validates `opts` (`render.invalid_msaa`),
    /// creates the group-0 buffers sized from it, installs the default slot providers
    /// (priority 0), adds the default material id 0 (grey standard) and registers the
    /// required feature passes `prepass` (order 30) and `forward` (order 60) in `pipeline`
    /// (`pipeline.duplicate_pass` when it already has them, `pipeline.busy`).
    pub fn new(
        gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
        opts: LitOptions,
    ) -> Result<Self, RaaError> {
        opts.validate()?;
        let mut shared = {
            let p = pipeline.try_borrow().map_err(|_| pipeline_busy())?;
            LitShared::new(
                gpu,
                p.lighting().layout(),
                opts.max_instances,
                opts.max_materials,
                opts.msaa,
            )?
        };
        shared.materials.add(SurfaceMaterial::default())?;
        let renderer = LitRenderer {
            opts,
            shared: Rc::new(RefCell::new(shared)),
            pipeline: Rc::clone(pipeline),
            class_handlers: HashMap::new(),
            extras: HashMap::new(),
            contributors: Rc::new(RefCell::new(Vec::new())),
            material_classes: Vec::with_capacity(opts.max_materials as usize),
            frame_bytes: [0; FRAME_GPU_SIZE],
            frame_index: 0,
            pending_background: None,
        };
        install_passes(pipeline, renderer.handle(), renderer.contributors())?;
        Ok(renderer)
    }

    /// A shared view of the renderer's state for passes and contributors.
    pub fn handle(&self) -> LitHandle {
        LitHandle::new(Rc::clone(&self.shared))
    }

    /// The forward contributors in recording order (for the forward pass).
    pub(crate) fn contributors(&self) -> Contributors {
        Rc::clone(&self.contributors)
    }

    // ---- meshes ----

    /// Validates and uploads `m` (`render.invalid_mesh` carrying the `mesh.*` code); one
    /// internal [`MeshId`] is assigned per sub-mesh ([`submesh_ids`](Self::submesh_ids)).
    pub fn upload(&mut self, gpu: &GpuContext, m: &AttributedMesh) -> Result<LitMeshId, RaaError> {
        self.shared.borrow_mut().meshes.upload(gpu, m)
    }

    /// Replaces mesh `id` (`render.unknown_mesh`, `render.invalid_mesh`; a refused mesh
    /// changes nothing). Its sub-mesh ids are kept where the sub-mesh count allows.
    pub fn update(
        &mut self,
        gpu: &GpuContext,
        id: LitMeshId,
        m: &AttributedMesh,
    ) -> Result<(), RaaError> {
        self.shared.borrow_mut().meshes.update(gpu, id, m)
    }

    /// Frees mesh `id` and its sub-mesh ids (reused lowest first). Unknown ids are ignored.
    pub fn remove(&mut self, id: LitMeshId) {
        self.shared.borrow_mut().meshes.remove(id);
    }

    /// The internal [`MeshId`] of each sub-mesh of `id`, in sub-mesh order.
    pub fn submesh_ids(&self, id: LitMeshId) -> Option<Ref<'_, [MeshId]>> {
        Ref::filter_map(self.shared.borrow(), |s| s.meshes.submesh_ids(id)).ok()
    }

    /// Counts, effective sub-meshes and bounds of mesh `id`.
    pub fn info(&self, id: LitMeshId) -> Option<LitMeshInfo> {
        self.shared.borrow().meshes.info(id)
    }

    // ---- materials, functions, slots ----

    /// The surface library (id 0 is the default grey standard material).
    pub fn materials(&self) -> Ref<'_, SurfaceLibrary> {
        Ref::map(self.shared.borrow(), |s| &s.materials)
    }

    /// The surface library, to add or set materials (uploaded at the next prepare).
    pub fn materials_mut(&mut self) -> RefMut<'_, SurfaceLibrary> {
        RefMut::map(self.shared.borrow_mut(), |s| &mut s.materials)
    }

    /// The material function registry (`raa.standard` is id 0).
    pub fn fns(&self) -> Ref<'_, MaterialFnRegistry> {
        Ref::map(self.shared.borrow(), |s| &s.fns)
    }

    /// The material function registry, to register built-in functions.
    pub fn fns_mut(&mut self) -> RefMut<'_, MaterialFnRegistry> {
        RefMut::map(self.shared.borrow_mut(), |s| &mut s.fns)
    }

    /// Registers user WGSL as a material function, checked with the current slot providers
    /// (S11 rules; refusals keep their `material.fn_*` / `shader.*` code).
    pub fn register_user_fn(&mut self, name: &str, wgsl: &str) -> Result<MaterialFnId, RaaError> {
        self.shared.borrow_mut().register_user_fn(name, wgsl)
    }

    /// The shader composer with its slot providers.
    pub fn composer(&self) -> Ref<'_, Composer> {
        Ref::map(self.shared.borrow(), |s| s.composer())
    }

    /// Installs a slot provider (`shader.slot_conflict` on an equal priority) and clears
    /// the pipeline cache.
    pub fn add_slot(&mut self, p: SlotProvider) -> Result<(), RaaError> {
        self.shared.borrow_mut().add_slot(p)
    }

    // ---- classes, extras, contributors, background ----

    /// Declares that pass `pass_name` draws `class` (materials of an unhandled class are
    /// refused at prepare with `material.blend_unavailable`).
    pub fn register_class_handler(&mut self, class: DrawClass, pass_name: &str) {
        self.class_handlers.insert(class, pass_name.to_string());
    }

    /// Whether `class` has a handler; Opaque always has one (the forward pass).
    pub fn class_handled(&self, class: DrawClass) -> bool {
        class == DrawClass::Opaque || self.class_handlers.contains_key(&class)
    }

    /// Sets the extras of every instance with scene tag `tag`.
    pub fn set_instance_extra(&mut self, tag: u32, x: InstanceExtra) {
        self.extras.insert(tag, x);
    }

    /// The extras of tag `tag` (S10 defaults when none were set).
    pub fn instance_extra(&self, tag: u32) -> InstanceExtra {
        self.extras.get(&tag).copied().unwrap_or_default()
    }

    /// Returns tag `tag` to the default extras.
    pub fn clear_instance_extra(&mut self, tag: u32) {
        self.extras.remove(&tag);
    }

    /// Adds a forward contributor and its pipeline contributor entry (enabled; the forward
    /// pass records the contributor only while the entry is enabled). Refusals:
    /// `pipeline.duplicate_pass` (nothing added), `pipeline.busy`.
    pub fn add_contributor(&mut self, c: Box<dyn ForwardContributor>) -> Result<(), RaaError> {
        let order = c.order();
        let enabled = Rc::new(Cell::new(true));
        self.pipeline
            .try_borrow_mut()
            .map_err(|_| pipeline_busy())?
            .add_contributor_entry(c.name(), order, Rc::clone(&enabled))?;
        let mut slots = self
            .contributors
            .try_borrow_mut()
            .map_err(|_| pipeline_busy())?;
        let at = insert_position(&slots, order);
        slots.insert(
            at,
            ContributorSlot {
                order,
                enabled,
                contributor: c,
            },
        );
        Ok(())
    }

    /// Sets the display-referred background colour (S12: written unchanged where nothing
    /// was drawn while no sky contributor is enabled) in the pipeline's `EnvGpu`. A colour
    /// with a non-finite component is ignored.
    pub fn set_background(&mut self, rgb: [f32; 3]) {
        if !rgb.iter().all(|c| c.is_finite()) {
            return;
        }
        match self.pipeline.try_borrow_mut() {
            Ok(mut p) => {
                p.lighting_mut().env_mut().background = [rgb[0], rgb[1], rgb[2], 0.0];
                self.pending_background = None;
            }
            Err(_) => self.pending_background = Some(rgb),
        }
    }

    // ---- frame ----

    /// Prepares a frame: `scene.update_world`, `scene.build_render_list_tagged(list,
    /// tags)`, the draw lists (opaque and fur batched by (mesh, material), transparent one
    /// per instance back to front by view depth, ties by tag) with the instance and extras
    /// bytes in draw order, the `FrameGpu` from the scene camera, light and ambient, then
    /// the queue writes (frame, instances, extras and, when changed, every material).
    ///
    /// Refusals (nothing written): `render.too_many_instances`, `render.too_many_materials`,
    /// `render.unknown_mesh`, `material.unknown_id`, `material.blend_unavailable` (a
    /// material's class has no handler), `render.invalid_camera`, `pipeline.busy`.
    pub fn prepare(
        &mut self,
        gpu: &GpuContext,
        scene: &mut Scene,
        list: &mut RenderList,
        tags: &mut Vec<u32>,
    ) -> Result<FrameCounts, RaaError> {
        if let Some(rgb) = self.pending_background {
            self.set_background(rgb);
        }
        let (viewport, encode_srgb) = {
            let p = self.pipeline.try_borrow().map_err(|_| pipeline_busy())?;
            (
                p.targets().map_or((1, 1), |t| t.size()),
                !p.surface_is_srgb(),
            )
        };
        scene.update_world();
        scene.build_render_list_tagged(list, tags);

        let mut guard = self.shared.try_borrow_mut().map_err(|_| pipeline_busy())?;
        let s: &mut LitShared = &mut guard;
        let max_materials = self.opts.max_materials as usize;
        if s.materials.len() > max_materials {
            return Err(refusal(
                "render.too_many_materials",
                format!(
                    "{} materials exceed max_materials {max_materials}",
                    s.materials.len()
                ),
            ));
        }
        if s.materials.is_dirty() || self.material_classes.len() != s.materials.len() {
            self.material_classes.clear();
            for id in 0..s.materials.len() as u32 {
                if let Some(m) = s.materials.get(MaterialId(id)) {
                    self.material_classes.push(MaterialClass::of(m));
                }
            }
        }

        let meshes = &s.meshes;
        let handlers = &self.class_handlers;
        let known_mesh = |m: MeshId| meshes.submesh(m).is_some();
        let class_handled = |c: DrawClass| c == DrawClass::Opaque || handlers.contains_key(&c);
        let input = PrepInput {
            list,
            tags,
            materials: &self.material_classes,
            extras: &self.extras,
            view: scene.camera.view(),
            known_mesh: &known_mesh,
            class_handled: &class_handled,
        };
        s.prep.prepare(&input)?;

        let mut counts = FrameCounts {
            instances: s.prep.instance_count(),
            ..FrameCounts::default()
        };
        for class in [DrawClass::Opaque, DrawClass::Fur, DrawClass::Transparent] {
            for d in s.prep.draws(class) {
                let (_, index_count) = s.meshes.submesh_range(d.mesh).unwrap_or((0, 0));
                counts.draw_calls += 1;
                counts.triangles += u64::from(index_count / 3) * u64::from(d.instance_count);
            }
        }

        let params = FrameParams {
            viewport,
            encode_srgb,
            msaa: self.opts.msaa,
            frame_index: self.frame_index,
            ..FrameParams::default()
        };
        write_frame_gpu(&mut self.frame_bytes, scene, &params)?;
        s.upload(gpu, &self.frame_bytes);
        self.frame_index += 1;
        Ok(counts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake(&'static str, u32);

    impl ForwardContributor for Fake {
        fn name(&self) -> &str {
            self.0
        }

        fn order(&self) -> u32 {
            self.1
        }

        fn record(&self, _: &mut wgpu::RenderPass<'_>, _: &LitHandle) -> Result<(), RaaError> {
            Ok(())
        }
    }

    #[test]
    fn options_default_to_the_contract_limits() {
        let o = LitOptions::default();
        assert_eq!((o.max_instances, o.max_materials, o.msaa), (16384, 256, 4));
        assert!(o.validate().is_ok());
    }

    #[test]
    fn options_refuse_msaa_other_than_1_or_4() {
        for msaa in [1, 4] {
            let o = LitOptions {
                msaa,
                ..LitOptions::default()
            };
            assert!(o.validate().is_ok(), "msaa {msaa}");
        }
        for msaa in [0, 2, 3, 8, 16] {
            let o = LitOptions {
                msaa,
                ..LitOptions::default()
            };
            let err = o.validate().unwrap_err();
            assert_eq!(err.code(), "render.invalid_msaa");
            assert!(err.message().contains(&msaa.to_string()));
        }
    }

    #[test]
    fn contributors_insert_by_order_then_registration() {
        let mut slots: Vec<ContributorSlot> = Vec::new();
        for (name, order) in [("a", 900), ("b", 500), ("c", 900), ("d", 600), ("e", 500)] {
            let at = insert_position(&slots, order);
            slots.insert(
                at,
                ContributorSlot {
                    order,
                    enabled: Rc::new(Cell::new(true)),
                    contributor: Box::new(Fake(name, order)),
                },
            );
        }
        let names: Vec<&str> = slots.iter().map(|s| s.contributor.name()).collect();
        assert_eq!(names, ["b", "e", "d", "a", "c"]);
    }
}
