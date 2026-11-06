//! The state the lit renderer shares with its passes and contributors through
//! [`LitHandle`]: the mesh store, the draw lists, the group-0 buffers (S10 `FrameGpu`,
//! `SurfaceMaterialGpu` array, `InstanceExtraGpu` array, the persistent instance records,
//! the material texture atlas, its sampler and rects, the material data words) with their
//! layout and bind group, the compacted u32 instance id stream, the shader composer and
//! material functions, and the cache of render pipelines per shader variant and
//! [`PipelineSpec`].

use std::cell::{Cell, Ref, RefCell};
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::num::NonZeroU64;
use std::rc::Rc;

use raa_core::{GpuContext, MaterialId, MeshId, RaaError};
use raa_scene::{InstanceIndex, RenderList};
use raa_shader::{
    ComposeOptions, Composer, MaterialFnId, MaterialFnRegistry, PassKind, ShaderError,
    SlotProvider, TintMode, VariantKey,
};

use crate::extras::INSTANCE_EXTRA_GPU_SIZE;
use crate::frame::FRAME_GPU_SIZE;
use crate::instance_store::InstanceStore;
use crate::library::SurfaceLibrary;
use crate::lod::{LodParams, LodRegistry};
use crate::material::{DrawClass, SURFACE_MATERIAL_GPU_SIZE, SurfaceMaterial};
use crate::material_data::MaterialData;
use crate::meshes::{INSTANCE_SLOT, MeshStore, STREAM0_SLOT, STREAM1_SLOT};
use crate::prep::{INSTANCE_GPU_SIZE, LitDraw};
use crate::renderer::{InstanceFrameStats, NO_CAMERA_LEVEL};
use crate::select::IdPrep;
use crate::slots::default_slot_providers;
use crate::textures::{TEXTURE_RECT_SIZE, TexelData, TextureAtlas};

/// Stride of S10 vertex stream 0 (S1 Vertex: position, normal).
const STREAM0_STRIDE: u64 = 24;
/// Stride of S10 vertex stream 1 (uv, tangent, colour).
const STREAM1_STRIDE: u64 = 40;

const STREAM0_ATTRIBUTES: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

/// Stride of the compacted u32 instance id stream (S1 vertex slot 1).
const ID_STRIDE: u64 = 4;

/// The compacted instance id stream: one u32 slot per instance, read by
/// `@builtin(instance_index)` and resolved with `raa_instance(id)`.
const INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![2 => Uint32];

const STREAM1_ATTRIBUTES: [wgpu::VertexAttribute; 3] =
    wgpu::vertex_attr_array![10 => Float32x2, 11 => Float32x4, 12 => Float32x4];

/// Posed variants (instanced skinning) read the same id stream; the palette base comes
/// from the persistent instance record (group 0 binding 7).
const POSED_INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 1] = INSTANCE_ATTRIBUTES;

/// Posed variants read uv and colour from stream 1; the rest tangent comes from slot 3.
const POSED_STREAM1_ATTRIBUTES: [wgpu::VertexAttribute; 2] = [
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x2,
        offset: 0,
        shader_location: 10,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 24,
        shader_location: 12,
    },
];

/// Slot 3 of posed variants: the mesh's 64-byte rest skin records (tangent @24, joints
/// u16x4 @40, weights f32x4 @48).
const SKIN_RECORD_ATTRIBUTES: [wgpu::VertexAttribute; 3] = [
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 24,
        shader_location: 11,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Uint16x4,
        offset: 40,
        shader_location: 13,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 48,
        shader_location: 14,
    },
];

/// Vertex slot of the rest skin records in posed variants.
pub(crate) const SKIN_SLOT: u32 = 3;

/// Stages that read group 0 (the vertex stage reads materials and extras too).
const FRAME_VISIBILITY: wgpu::ShaderStages = wgpu::ShaderStages::VERTEX_FRAGMENT;

/// Fixed-function state of a lit render pipeline beside its shader variant. Together with
/// the [`VariantKey`] and the slots hash it keys the pipeline cache of [`LitHandle::pipeline_for`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct PipelineSpec {
    /// Colour targets in location order (`None` leaves a location without a target).
    pub color_targets: Vec<Option<wgpu::ColorTargetState>>,
    /// Depth state; `None` for a pass without a depth attachment.
    pub depth: Option<wgpu::DepthStencilState>,
    /// Culled face (`Some(Back)` for lit draws, `None` for DOUBLE_SIDED materials).
    pub cull: Option<wgpu::Face>,
    /// Alpha-to-coverage (fur shells).
    pub alpha_to_coverage: bool,
    /// Layout of group 2, private to the pass that binds it.
    pub extra_layout: Option<wgpu::BindGroupLayout>,
}

/// The vertex buffers of every lit pipeline, in slot order: S10 stream 0 at
/// [`STREAM0_SLOT`] (locations 0-1), the compacted u32 instance id stream at
/// [`INSTANCE_SLOT`] (location 2, step Instance) and S10 stream 1 at [`STREAM1_SLOT`]
/// (locations 10-12).
fn vertex_buffer_layouts() -> [Option<wgpu::VertexBufferLayout<'static>>; 3] {
    [
        Some(wgpu::VertexBufferLayout {
            array_stride: STREAM0_STRIDE,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &STREAM0_ATTRIBUTES,
        }),
        Some(wgpu::VertexBufferLayout {
            array_stride: ID_STRIDE,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &INSTANCE_ATTRIBUTES,
        }),
        Some(wgpu::VertexBufferLayout {
            array_stride: STREAM1_STRIDE,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &STREAM1_ATTRIBUTES,
        }),
    ]
}

/// The vertex buffers of posed variants: stream 0 (rest), the id stream, stream 1
/// without its tangent and the rest skin records in slot 3.
fn posed_vertex_buffer_layouts() -> [Option<wgpu::VertexBufferLayout<'static>>; 4] {
    [
        Some(wgpu::VertexBufferLayout {
            array_stride: STREAM0_STRIDE,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &STREAM0_ATTRIBUTES,
        }),
        Some(wgpu::VertexBufferLayout {
            array_stride: ID_STRIDE,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &POSED_INSTANCE_ATTRIBUTES,
        }),
        Some(wgpu::VertexBufferLayout {
            array_stride: STREAM1_STRIDE,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &POSED_STREAM1_ATTRIBUTES,
        }),
        Some(wgpu::VertexBufferLayout {
            array_stride: crate::skin_views::SKIN_RECORD_STRIDE,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &SKIN_RECORD_ATTRIBUTES,
        }),
    ]
}

/// Group 3 of posed variants: the joint palette, read-only storage (vertex stage).
fn palette_layout_entries() -> [wgpu::BindGroupLayoutEntry; 1] {
    [wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(crate::skin_views::PALETTE_MATRIX_BYTES),
        },
        count: None,
    }]
}

/// S10 group 0: binding 0 `FrameGpu` uniform (512 B), binding 1 `array<SurfaceMaterialGpu>`
/// and binding 2 `array<InstanceExtraGpu>`, both read-only storage; 3, 4, 5 the material
/// texture atlas, its sampler and rects; 6 the material data words (read-only storage);
/// 7 the persistent instance records (read-only storage).
fn frame_layout_entries() -> [wgpu::BindGroupLayoutEntry; 8] {
    let read_only = wgpu::BufferBindingType::Storage { read_only: true };
    [
        buffer_entry(0, wgpu::BufferBindingType::Uniform, FRAME_GPU_SIZE),
        buffer_entry(1, read_only, SURFACE_MATERIAL_GPU_SIZE),
        buffer_entry(2, read_only, INSTANCE_EXTRA_GPU_SIZE),
        wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: FRAME_VISIBILITY,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: FRAME_VISIBILITY,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
        buffer_entry(5, read_only, TEXTURE_RECT_SIZE),
        buffer_entry(6, read_only, 4),
        buffer_entry(7, read_only, INSTANCE_GPU_SIZE),
    ]
}

/// A buffer of `size` bytes with `usage | COPY_DST`.
fn buffer(
    gpu: &GpuContext,
    label: &'static str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn buffer_entry(
    binding: u32,
    ty: wgpu::BufferBindingType,
    min_size: usize,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: FRAME_VISIBILITY,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(min_size as u64),
        },
        count: None,
    }
}

/// A shader refusal as a [`RaaError`], keeping its code (`shader.*`, `material.fn_*`).
fn shader_error(e: ShaderError) -> RaaError {
    RaaError::FrameGraph {
        code: e.code,
        message: e.message,
    }
}

/// Refuses `render.invalid_msaa` unless `msaa` is 1 or 4 (S10 MSAA).
fn check_msaa(msaa: u32) -> Result<(), RaaError> {
    if msaa == 1 || msaa == 4 {
        Ok(())
    } else {
        Err(RaaError::FrameGraph {
            code: "render.invalid_msaa",
            message: format!("msaa must be 1 or 4 samples, got {msaa}"),
        })
    }
}

/// Hash of `spec` (std SipHash with fixed keys: equal specs hash equal within a process).
fn spec_hash(spec: &PipelineSpec) -> u64 {
    let mut h = DefaultHasher::new();
    spec.hash(&mut h);
    h.finish()
}

/// Key of one cached pipeline: variant, spec hash and the slots hash it was composed with.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct CacheKey {
    variant: VariantKey,
    spec: u64,
    slots: u64,
}

/// Built values by [`CacheKey`]. The spec is stored beside each value and compared on a
/// lookup, so two specs with one hash never share a value. A lookup with a slots hash other
/// than the current one empties the cache first: every variant composed with the old slot
/// winners is stale.
#[derive(Debug)]
pub(crate) struct PipelineCache<T> {
    slots: u64,
    entries: HashMap<CacheKey, (PipelineSpec, T)>,
}

impl<T: Clone> PipelineCache<T> {
    pub(crate) fn new() -> Self {
        PipelineCache {
            slots: 0,
            entries: HashMap::new(),
        }
    }

    /// Drops every value (the slot providers changed).
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
    }

    /// Number of cached values.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// The value cached for (`variant`, `spec`, `slots`), else `build()` stored and
    /// returned. A failed build stores nothing.
    pub(crate) fn get_or_build<E>(
        &mut self,
        variant: VariantKey,
        spec: &PipelineSpec,
        slots: u64,
        build: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, E> {
        if slots != self.slots {
            self.entries.clear();
            self.slots = slots;
        }
        let key = CacheKey {
            variant,
            spec: spec_hash(spec),
            slots,
        };
        if let Some((stored, value)) = self.entries.get(&key)
            && stored == spec
        {
            return Ok(value.clone());
        }
        let value = build()?;
        self.entries.insert(key, (spec.clone(), value.clone()));
        Ok(value)
    }
}

/// What the renderer, its passes and contributors share (`Rc<RefCell<LitShared>>`). The
/// renderer mutates it in its own methods (uploads, prepare); passes and contributors read
/// it through [`LitHandle`], which only ever borrows it immutably, so a held
/// [`LitHandle::draws`] never blocks [`LitHandle::pipeline_for`] (the cache has its own cell).
#[allow(dead_code, reason = "built and driven by LitRenderer")]
pub(crate) struct LitShared {
    msaa: u32,
    pub(crate) skin_source: crate::skin_views::SkinDrawSource,
    bound_skin: Cell<bool>,
    pub(crate) meshes: MeshStore,
    pub(crate) editor: crate::editor::EditorState,
    /// The camera and cascade id sections and their draw lists.
    pub(crate) select: IdPrep,
    /// The persistent instance and extras records by stable scene slot.
    pub(crate) store: InstanceStore,
    pub(crate) lods: LodRegistry,
    pub(crate) lod_options: LodParams,
    /// Camera hysteresis by stable slot; NO_CAMERA_LEVEL means no previous selection.
    pub(crate) levels: Vec<u8>,
    pub(crate) lod_bias: HashMap<u32, i8>,
    pub(crate) stats: InstanceFrameStats,
    pub(crate) materials: SurfaceLibrary,
    pub(crate) fns: MaterialFnRegistry,
    /// S23 tint mode per material; a material absent from the map is `multiply`.
    tint_modes: RefCell<HashMap<MaterialId, TintMode>>,
    composer: Composer,
    /// `composer.slots_hash()`, updated by [`LitShared::add_slot`].
    slots_hash: u64,
    /// Compacted u32 instance ids, 5 sections of `max_instances` (camera + cascades).
    pub(crate) id_buffer: wgpu::Buffer,
    /// `FrameGpu`, 512 B.
    frame_buffer: wgpu::Buffer,
    /// `SurfaceMaterialGpu` array, `max_materials` x 256 B.
    material_buffer: wgpu::Buffer,
    /// Persistent S1 instance records, group-0 binding 7.
    pub(crate) instance_storage: wgpu::Buffer,
    /// `InstanceExtraGpu` array, one record per slot.
    extras_buffer: wgpu::Buffer,
    /// The frame snapshot the cascade selection reads (index, list, tags, slots).
    pub(crate) index: InstanceIndex,
    pub(crate) items: RenderList,
    pub(crate) item_tags: Vec<u32>,
    pub(crate) item_slots: Vec<u32>,
    /// Scratch of `SurfaceLibrary::write_gpu_all`, preallocated for `max_materials`.
    material_bytes: Vec<u8>,
    /// Material textures, group-0 bindings 3 (atlas), 4 (sampler) and 5 (rects).
    textures: TextureAtlas,
    /// Material data words, group-0 binding 6.
    data: MaterialData,
    frame_layout: wgpu::BindGroupLayout,
    frame_bind_group: wgpu::BindGroup,
    /// The group-1 layout of `raa_pipeline::LightingResources`.
    lighting_layout: wgpu::BindGroupLayout,
    pipelines: RefCell<PipelineCache<wgpu::RenderPipeline>>,
    /// Group 3 of posed variants and its bind group over the palette (by generation).
    palette_layout: wgpu::BindGroupLayout,
    palette_group: Option<(u64, wgpu::BindGroup)>,
    modules: RefCell<HashMap<String, wgpu::ShaderModule>>,
    content_pipelines: RefCell<HashMap<(String, PipelineSpec), wgpu::RenderPipeline>>,
}

#[allow(dead_code, reason = "built and driven by LitRenderer")]
impl LitShared {
    /// Creates the group-0 buffers (sized for `max_instances` and `max_materials`, at least
    /// one record each), the group-0 layout and bind group, an empty mesh store and draw
    /// lists, the registry with `raa.standard` and a composer with the default slot
    /// providers (S11, priority 0). `lighting_layout` is the group-1 layout of the pipeline's
    /// `LightingResources`. Refuses `render.invalid_msaa` unless `msaa` is 1 or 4.
    pub(crate) fn new(
        gpu: &GpuContext,
        lighting_layout: &wgpu::BindGroupLayout,
        max_instances: u32,
        max_materials: u32,
        msaa: u32,
    ) -> Result<Self, RaaError> {
        check_msaa(msaa)?;
        let mut composer = Composer::new();
        for p in default_slot_providers() {
            composer.add_slot(p).map_err(shader_error)?;
        }
        let instances = u64::from(max_instances.max(1));
        let materials = u64::from(max_materials.max(1));
        let store = InstanceStore::new(max_instances.max(1));
        let instance_bytes = store.instance_bytes().len().max(INSTANCE_GPU_SIZE) as u64;
        let extra_bytes = store.extra_bytes().len().max(INSTANCE_EXTRA_GPU_SIZE) as u64;
        let storage = wgpu::BufferUsages::STORAGE;
        let id_buffer = buffer(
            gpu,
            "raa.lit.ids",
            instances * 5 * ID_STRIDE,
            wgpu::BufferUsages::VERTEX,
        );
        let frame_buffer = buffer(
            gpu,
            "raa.lit.frame",
            FRAME_GPU_SIZE as u64,
            wgpu::BufferUsages::UNIFORM,
        );
        let material_buffer = buffer(
            gpu,
            "raa.lit.materials",
            materials * SURFACE_MATERIAL_GPU_SIZE as u64,
            storage,
        );
        let instance_storage = buffer(gpu, "raa.lit.instance_records", instance_bytes, storage);
        let extras_buffer = buffer(gpu, "raa.lit.extras", extra_bytes, storage);
        let frame_layout =
            gpu.device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("raa.lit.frame_layout"),
                    entries: &frame_layout_entries(),
                });
        let palette_layout =
            gpu.device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("raa.lit.skin_palette_layout"),
                    entries: &palette_layout_entries(),
                });
        let textures = TextureAtlas::new(gpu);
        let data = MaterialData::new(gpu);
        let frame_bind_group = create_frame_bind_group(
            gpu,
            &frame_layout,
            [
                &frame_buffer,
                &material_buffer,
                &extras_buffer,
                &instance_storage,
            ],
            &textures,
            &data,
        );
        Ok(LitShared {
            msaa,
            skin_source: Default::default(),
            bound_skin: Cell::new(false),
            meshes: MeshStore::new(),
            editor: crate::editor::EditorState::default(),
            select: IdPrep::new(max_instances, max_materials),
            lods: LodRegistry::new(),
            lod_options: LodParams::default(),
            levels: vec![NO_CAMERA_LEVEL; store.capacity() as usize],
            lod_bias: HashMap::new(),
            stats: InstanceFrameStats::default(),
            store,
            materials: SurfaceLibrary::new(),
            fns: MaterialFnRegistry::new(),
            tint_modes: RefCell::new(HashMap::new()),
            slots_hash: composer.slots_hash(),
            composer,
            id_buffer,
            frame_buffer,
            material_buffer,
            instance_storage,
            extras_buffer,
            index: InstanceIndex::new(),
            items: RenderList::default(),
            item_tags: Vec::new(),
            item_slots: Vec::new(),
            material_bytes: Vec::with_capacity(materials as usize * SURFACE_MATERIAL_GPU_SIZE),
            textures,
            data,
            frame_layout,
            frame_bind_group,
            lighting_layout: lighting_layout.clone(),
            pipelines: RefCell::new(PipelineCache::new()),
            palette_layout,
            palette_group: None,
            modules: RefCell::new(HashMap::new()),
            content_pipelines: RefCell::new(HashMap::new()),
        })
    }

    /// MSAA sample count of the forward pass.
    pub(crate) fn msaa(&self) -> u32 {
        self.msaa
    }

    /// The slot providers and their winners.
    pub(crate) fn composer(&self) -> &Composer {
        &self.composer
    }

    /// Installs a slot provider (`shader.slot_conflict` on an equal priority, nothing
    /// changed) and clears the pipeline cache.
    pub(crate) fn add_slot(&mut self, p: SlotProvider) -> Result<(), RaaError> {
        self.composer.add_slot(p).map_err(shader_error)?;
        self.slots_hash = self.composer.slots_hash();
        self.pipelines.get_mut().clear();
        self.modules.get_mut().clear();
        self.content_pipelines.get_mut().clear();
        Ok(())
    }

    /// Registers a user material function (S11 rules, checked against the current slots);
    /// refusals keep their `material.fn_*` / `shader.*` code.
    pub(crate) fn register_user_fn(
        &mut self,
        name: &str,
        wgsl: &str,
    ) -> Result<MaterialFnId, RaaError> {
        self.fns
            .register_user(name, wgsl, &self.composer)
            .map_err(shader_error)
    }

    /// Adds a material texture to the atlas and returns its `raa_texture` id; group 0 is
    /// bound again when the atlas texture is created (first add). Refusals (nothing
    /// changes): `texture.invalid`, `texture.atlas_full`.
    pub(crate) fn add_texture(
        &mut self,
        gpu: &GpuContext,
        width: u32,
        height: u32,
        data: TexelData<'_>,
    ) -> Result<u32, RaaError> {
        let (id, created) = self.textures.add(gpu, width, height, data)?;
        if created {
            self.frame_bind_group = create_frame_bind_group(
                gpu,
                &self.frame_layout,
                [
                    &self.frame_buffer,
                    &self.material_buffer,
                    &self.extras_buffer,
                    &self.instance_storage,
                ],
                &self.textures,
                &self.data,
            );
        }
        Ok(id)
    }

    /// Adds a block of material data words and returns its base; group 0 is bound again
    /// when the buffer grows. Refusals (nothing changes): `material.data_invalid`,
    /// `material.data_full`.
    pub(crate) fn add_material_data(
        &mut self,
        gpu: &GpuContext,
        words: &[u32],
    ) -> Result<u32, RaaError> {
        let (base, grown) = self.data.add(gpu, words)?;
        if grown {
            self.frame_bind_group = create_frame_bind_group(
                gpu,
                &self.frame_layout,
                [
                    &self.frame_buffer,
                    &self.material_buffer,
                    &self.extras_buffer,
                    &self.instance_storage,
                ],
                &self.textures,
                &self.data,
            );
        }
        Ok(base)
    }

    /// Overwrites the block at `base` (same length); `material.data_invalid` otherwise.
    pub(crate) fn set_material_data(
        &mut self,
        gpu: &GpuContext,
        base: u32,
        words: &[u32],
    ) -> Result<(), RaaError> {
        self.data.set(gpu, base, words)
    }

    /// Frees the block at `base`; an unknown base does nothing. Returns whether it was live.
    pub(crate) fn remove_material_data(&mut self, gpu: &GpuContext, base: u32) -> bool {
        self.data.remove(gpu, base)
    }

    /// The material data store (placement, capacity, host copy of the blocks).
    pub(crate) fn material_data(&self) -> &MaterialData {
        &self.data
    }

    /// Removes texture `id`; an unknown or removed id does nothing. Returns whether it was
    /// live.
    pub(crate) fn remove_texture(&mut self, gpu: &GpuContext, id: u32) -> bool {
        self.textures.remove(gpu, id)
    }

    /// The atlas packing (rects of the live textures).
    pub(crate) fn texture_packer(&self) -> &crate::textures::ShelfPacker {
        self.textures.packer()
    }

    /// Writes the frame's `FrameGpu`, the changed instance and extras pages, the compacted
    /// id stream and, when the library changed, every material record. A grown store
    /// re-creates its GPU buffers and the frame bind group first.
    pub(crate) fn upload(&mut self, gpu: &GpuContext, frame: &[u8; FRAME_GPU_SIZE]) {
        let queue = gpu.queue();
        queue.write_buffer(&self.frame_buffer, 0, frame);
        self.ensure_store_buffers(gpu);
        self.store
            .upload(queue, &self.instance_storage, &self.extras_buffer);
        write_prefix(queue, &self.id_buffer, self.select.id_bytes());
        if self.materials.is_dirty() {
            self.materials.write_gpu_all(&mut self.material_bytes);
            write_prefix(queue, &self.material_buffer, &self.material_bytes);
        }
    }

    /// Re-creates the instance and extras buffers, and the frame bind group over them,
    /// when the store grew past their size.
    fn ensure_store_buffers(&mut self, gpu: &GpuContext) {
        let instances = self.store.instance_bytes().len().max(INSTANCE_GPU_SIZE) as u64;
        let extras = self.store.extra_bytes().len().max(INSTANCE_EXTRA_GPU_SIZE) as u64;
        if self.instance_storage.size() == instances && self.extras_buffer.size() == extras {
            return;
        }
        let storage = wgpu::BufferUsages::STORAGE;
        self.instance_storage = buffer(gpu, "raa.lit.instance_records", instances, storage);
        self.extras_buffer = buffer(gpu, "raa.lit.extras", extras, storage);
        self.frame_bind_group = create_frame_bind_group(
            gpu,
            &self.frame_layout,
            [
                &self.frame_buffer,
                &self.material_buffer,
                &self.extras_buffer,
                &self.instance_storage,
            ],
            &self.textures,
            &self.data,
        );
    }

    /// Re-creates the palette bind group when the skin palette buffer was (re)created.
    pub(crate) fn sync_palette_group(&mut self, gpu: &GpuContext) {
        let Some((buffer, generation)) = self.skin_source.palette() else {
            self.palette_group = None;
            return;
        };
        if self
            .palette_group
            .as_ref()
            .is_some_and(|(g, _)| *g == generation)
        {
            return;
        }
        let group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa.lit.skin_palette"),
            layout: &self.palette_layout,
            entries: &[whole(0, &buffer)],
        });
        self.palette_group = Some((generation, group));
    }

    /// Whether posed draws can bind the palette this frame.
    pub(crate) fn palette_ready(&self) -> bool {
        self.palette_group.is_some()
    }

    /// Mirrors the editor overlay state into the skin draw source (the skin pass builds
    /// overlay geometry only for tags whose outline or wireframe is on).
    pub(crate) fn sync_skin_overlays(&self) {
        use crate::editor::WireframeMode;
        let e = &self.editor;
        self.skin_source.set_overlay_requests(
            e.global != WireframeMode::Off,
            e.wire.iter().map(|(&t, &m)| (t, m != WireframeMode::Off)),
            e.outlines.keys().copied(),
        );
    }

    /// The rest skin records and palette bind group a draw of instance `tag` binds when it
    /// is drawn posed; `None` draws rest (no tag, no pose, skinning disabled or no palette).
    pub(crate) fn posed_binding(
        &self,
        tag: Option<u32>,
    ) -> Option<(wgpu::Buffer, wgpu::BindGroup)> {
        let (_, group) = self.palette_group.as_ref()?;
        let skin = self.skin_source.skin_buffer(tag?)?;
        Some((skin, group.clone()))
    }

    /// The cached pipeline of (`key`, `spec`, current slots), composed and built on a miss.
    fn pipeline_for(
        &self,
        gpu: &GpuContext,
        key: VariantKey,
        spec: &PipelineSpec,
    ) -> Result<wgpu::RenderPipeline, RaaError> {
        check_msaa(key.msaa)?;
        self.pipelines
            .borrow_mut()
            .get_or_build(key, spec, self.slots_hash, || {
                self.build_pipeline(gpu, &key, spec)
            })
    }

    /// Composes the WGSL of `key` (tint mode `graph` when the key holds
    /// [`raa_shader::VARIANT_FLAG_TINT_GRAPH`]), validates it with naga (so a broken variant is a typed
    /// refusal, not a device error) and creates its render pipeline with the layouts
    /// [group 0, group 1 (lighting), `spec.extra_layout`]. Shadow variants leave group 1
    /// empty: that pass renders the shadow map group 1 samples and reads groups 0 and 2 only.
    fn build_pipeline(
        &self,
        gpu: &GpuContext,
        key: &VariantKey,
        spec: &PipelineSpec,
    ) -> Result<wgpu::RenderPipeline, RaaError> {
        let tint_mode = TintMode::from_variant_flags(key.flags);
        let wgsl = if tint_mode == TintMode::Multiply {
            self.composer.compose(&self.fns, key)
        } else {
            self.composer
                .compose_with(&self.fns, key, &ComposeOptions { tint_mode })
        }
        .map_err(shader_error)?;
        // A posed variant rewrites the composed pass section itself, so composer rewrites
        // of the entry points (tint mode `graph`) keep their skinning. Matching the raw
        // pass source missed them and silently reused the rest pipeline: graph-tinted
        // skinned meshes drew in rest pose.
        let wgsl = if key.flags & crate::skin_draw::POSED != 0 {
            crate::skin_draw::posed_module(&wgsl).ok_or_else(|| RaaError::FrameGraph {
                code: "shader.validate",
                message: format!(
                    "pass {:?}: the composed entry points have no vs_main to skin",
                    key.pass
                ),
            })?
        } else {
            wgsl
        };
        // Full source equality guards the hash lookup. Registration ids are not shader state.
        let content_key = (wgsl.clone(), spec.clone());
        if let Some(pipeline) = self.content_pipelines.borrow().get(&content_key) {
            return Ok(pipeline.clone());
        }
        let device = gpu.device();
        let mut modules = self.modules.borrow_mut();
        let module = if let Some(module) = modules.get(&wgsl) {
            module.clone()
        } else {
            Composer::validate(&wgsl).map_err(shader_error)?;
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("raa.lit.variant"),
                source: wgpu::ShaderSource::Wgsl(wgsl.clone().into()),
            });
            modules.insert(wgsl, module.clone());
            module
        };
        drop(modules);
        let shadow = key.pass == PassKind::Shadow;
        let posed = key.flags & crate::skin_draw::POSED != 0;
        let mut groups = vec![
            Some(&self.frame_layout),
            (!shadow).then_some(&self.lighting_layout),
        ];
        if let Some(extra) = &spec.extra_layout {
            groups.push(Some(extra));
        } else if posed {
            groups.push(None);
        }
        if posed {
            groups.push(Some(&self.palette_layout));
        }
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raa.lit.pipeline_layout"),
            bind_group_layouts: &groups,
            immediate_size: 0,
        });
        let buffers = if posed {
            posed_vertex_buffer_layouts().to_vec()
        } else {
            vertex_buffer_layouts().to_vec()
        };
        let fragment = (!shadow).then(|| wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &spec.color_targets,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("raa.lit.pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &buffers,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: spec.cull,
                ..Default::default()
            },
            depth_stencil: spec.depth.clone(),
            multisample: wgpu::MultisampleState {
                count: key.msaa,
                mask: !0,
                alpha_to_coverage_enabled: spec.alpha_to_coverage,
            },
            fragment,
            multiview_mask: None,
            cache: None,
        });
        self.content_pipelines
            .borrow_mut()
            .insert(content_key, pipeline.clone());
        Ok(pipeline)
    }
}

/// The group-0 bind group: `buffers` (frame, materials, extras, instance records) at
/// bindings 0..3 and 7, the texture atlas entries at 3..6 and the material data words at 6.
fn create_frame_bind_group(
    gpu: &GpuContext,
    layout: &wgpu::BindGroupLayout,
    buffers: [&wgpu::Buffer; 4],
    textures: &TextureAtlas,
    data: &MaterialData,
) -> wgpu::BindGroup {
    let [frame, materials, extras, instances] = buffers;
    let [atlas, sampler, rects] = textures.bind_entries();
    gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("raa.lit.frame_bind_group"),
        layout,
        entries: &[
            whole(0, frame),
            whole(1, materials),
            whole(2, extras),
            whole(7, instances),
            atlas,
            sampler,
            rects,
            data.bind_entry(),
        ],
    })
}

fn whole(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// Writes the part of `bytes` that fits `buffer` at offset 0 (nothing when empty).
fn write_prefix(queue: &wgpu::Queue, buffer: &wgpu::Buffer, bytes: &[u8]) {
    let n = bytes.len().min(buffer.size() as usize);
    if let Some(head) = bytes.get(..n)
        && !head.is_empty()
    {
        queue.write_buffer(buffer, 0, head);
    }
}

/// Shared view of the lit renderer for passes and contributors (an `Rc` to the renderer's
/// state; clones share it). Every method borrows the state immutably.
#[derive(Clone)]
pub struct LitHandle {
    pub(crate) shared: Rc<RefCell<LitShared>>,
}

impl LitHandle {
    #[allow(dead_code, reason = "created by LitRenderer::handle")]
    pub(crate) fn new(shared: Rc<RefCell<LitShared>>) -> Self {
        LitHandle { shared }
    }

    /// The camera draws of `class` in submission order (transparent: back to front).
    /// Instances `first_instance..first_instance + instance_count` index the compacted id
    /// stream (`@builtin(instance_index)`), whose ids address the instance records.
    pub fn draws(&self, class: DrawClass) -> Ref<'_, [LitDraw]> {
        self.camera_draws(class)
    }

    /// The camera draws of `class`; shadows draw the camera list until the cascade
    /// selection lands.
    pub fn camera_draws(&self, class: DrawClass) -> Ref<'_, [LitDraw]> {
        Ref::map(self.shared.borrow(), |s| s.select.draws(class))
    }

    /// Sets the vertex buffers of sub-mesh `mesh` and the id stream (stream 0 at slot 0,
    /// ids at slot 1, stream 1 at slot 2) and binds the index buffer starting at
    /// the sub-mesh's first index ([`submesh_range`](Self::submesh_range)), so
    /// `draw_indexed(0..count, 0, first..first + n)` draws exactly the sub-mesh. Returns
    /// `count`, the sub-mesh's index count. Unknown ids give `render.unknown_mesh`.
    pub fn bind_mesh(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        mesh: MeshId,
    ) -> Result<u32, RaaError> {
        let s = self.shared.borrow();
        let (mesh, tag) = s.select.mesh_binding(mesh);
        let (g, first, count) = s.meshes.resolve(mesh)?;
        let posed = s.posed_binding(tag);
        s.bound_skin.set(posed.is_some());
        pass.set_vertex_buffer(STREAM0_SLOT, g.stream0.slice(..));
        if let Some((skin, palette)) = &posed {
            // Instanced skinning: rest skin records in slot 3, the palette in group 3.
            pass.set_vertex_buffer(SKIN_SLOT, skin.slice(..));
            pass.set_bind_group(crate::skin_draw::PALETTE_GROUP, palette, &[]);
        }
        pass.set_vertex_buffer(INSTANCE_SLOT, s.id_buffer.slice(..));
        pass.set_vertex_buffer(STREAM1_SLOT, g.stream1.slice(..));
        let start = u64::from(first) * 4;
        let end = start + u64::from(count) * 4;
        let indices = if count == 0 {
            g.index.slice(..)
        } else {
            g.index.slice(start..end)
        };
        pass.set_index_buffer(indices, wgpu::IndexFormat::Uint32);
        Ok(count)
    }

    /// `(first_index, index_count)` of sub-mesh `mesh` in its lit mesh's index buffer;
    /// `(0, 0)` when unknown.
    pub fn submesh_range(&self, mesh: MeshId) -> (u32, u32) {
        let s = self.shared.borrow();
        s.meshes
            .submesh_range(s.select.mesh_binding(mesh).0)
            .unwrap_or((0, 0))
    }

    /// The compacted u32 instance id stream (4 B per id, camera section first).
    pub fn id_buffer(&self) -> wgpu::Buffer {
        self.shared.borrow().id_buffer.clone()
    }

    /// The S10 group-0 layout: `FrameGpu` uniform (512 B), `array<SurfaceMaterialGpu>`,
    /// `array<InstanceExtraGpu>` and `array<RaaInstanceGpu>` read-only storage, then the
    /// material texture atlas (`texture_2d<f32>`), its filtering sampler and
    /// `array<vec4<f32>>` rects, and the material data words.
    pub fn frame_layout(&self) -> wgpu::BindGroupLayout {
        self.shared.borrow().frame_layout.clone()
    }

    /// The group-0 bind group over the renderer's frame, material and extras buffers and the
    /// texture atlas. It is replaced when the first texture creates the atlas: fetch it for
    /// each frame rather than keeping it.
    pub fn frame_bind_group(&self) -> wgpu::BindGroup {
        self.shared.borrow().frame_bind_group.clone()
    }

    /// The render pipeline of shader variant `key` with the fixed-function state `spec`:
    /// composed by raa-shader (material function, slot winners, pass entry points),
    /// validated with naga and built with the layouts [group 0, group 1 lighting,
    /// `spec.extra_layout`] (shadow variants leave group 1 empty) and the vertex buffers of
    /// [`bind_mesh`](Self::bind_mesh). Cached by (key, spec hash, slots hash); installing a
    /// slot provider clears the cache. Refusals: `render.invalid_msaa` (key msaa not 1 or
    /// 4) and the raa-shader codes (`shader.unknown_fn`, `shader.slot_missing`,
    /// `shader.parse`, `shader.validate`).
    pub fn pipeline_for(
        &self,
        gpu: &GpuContext,
        key: VariantKey,
        spec: &PipelineSpec,
    ) -> Result<wgpu::RenderPipeline, RaaError> {
        let s = self.shared.borrow();
        let mut key = key;
        if s.bound_skin.get() {
            key.flags |= crate::skin_draw::POSED;
        }
        s.pipeline_for(gpu, key, spec)
    }

    /// The pipeline of a draw of prepared binding `mesh`: the posed (instanced skinning)
    /// variant when [`bind_mesh`](Self::bind_mesh) binds that draw posed, else the rest
    /// variant. Unlike [`pipeline_for`](Self::pipeline_for) it does not depend on the last
    /// bound mesh, so it can be resolved before any draw is bound (fur plans in prepare).
    pub fn pipeline_for_draw(
        &self,
        gpu: &GpuContext,
        mesh: MeshId,
        key: VariantKey,
        spec: &PipelineSpec,
    ) -> Result<wgpu::RenderPipeline, RaaError> {
        let s = self.shared.borrow();
        let mut key = key;
        if s.posed_binding(s.select.mesh_binding(mesh).1).is_some() {
            key.flags |= crate::skin_draw::POSED;
        }
        s.pipeline_for(gpu, key, spec)
    }

    /// Whether a draw of prepared binding `mesh` is drawn posed (instanced skinning).
    pub fn draw_is_posed(&self, mesh: MeshId) -> bool {
        let s = self.shared.borrow();
        s.posed_binding(s.select.mesh_binding(mesh).1).is_some()
    }

    /// Stores the S23 tint mode of material `id` (`multiply` by default). Draws of a
    /// material in `graph` mode carry [`material_variant_flags`](Self::material_variant_flags)
    /// in their [`VariantKey`], so their pipelines compose without the post-material tint
    /// colour multiply; the variants of the other mode stay cached.
    pub fn set_material_tint_mode(&self, id: MaterialId, mode: TintMode) {
        let s = self.shared.borrow();
        let mut modes = s.tint_modes.borrow_mut();
        if mode == TintMode::Multiply {
            modes.remove(&id);
        } else {
            modes.insert(id, mode);
        }
    }

    /// The S23 tint mode of material `id` (`multiply` when never set).
    pub fn material_tint_mode(&self, id: MaterialId) -> TintMode {
        let s = self.shared.borrow();
        let mode = s.tint_modes.borrow().get(&id).copied();
        mode.unwrap_or_default()
    }

    /// The [`VariantKey::flags`] bits a draw of material `id` adds to its material flags:
    /// [`raa_shader::VARIANT_FLAG_TINT_GRAPH`] in tint mode `graph`, 0 otherwise.
    pub fn material_variant_flags(&self, id: MaterialId) -> u32 {
        self.material_tint_mode(id).variant_flag()
    }

    /// Connect the skin pass's live views before preparing draws. Bind each mesh before
    /// requesting its pipeline so the auxiliary tangent layout matches that draw.
    pub fn set_skin_views(&self, views: &crate::SkinViews) {
        let mut s = self.shared.borrow_mut();
        s.skin_source = views.draw_source();
        s.palette_group = None;
        s.sync_skin_overlays();
    }

    /// MSAA sample count of the forward pass (1 or 4).
    pub fn msaa(&self) -> u32 {
        self.shared.borrow().msaa()
    }

    /// A copy of material `id` of the library, if it exists.
    pub fn material_record(&self, id: MaterialId) -> Option<SurfaceMaterial> {
        self.shared.borrow().materials.get(id).cloned()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    fn key(pass: PassKind, msaa: u32) -> VariantKey {
        VariantKey {
            material_fn: MaterialFnId::STANDARD,
            pass,
            msaa,
            flags: 0,
        }
    }

    fn forward_spec() -> PipelineSpec {
        PipelineSpec {
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
            cull: Some(wgpu::Face::Back),
            alpha_to_coverage: false,
            extra_layout: None,
        }
    }

    /// Looks up (`k`, `spec`, `slots`) and counts the builds in `builds`; the built value
    /// is the build number.
    fn lookup(
        cache: &mut PipelineCache<u32>,
        builds: &Cell<u32>,
        k: VariantKey,
        spec: &PipelineSpec,
        slots: u64,
    ) -> u32 {
        let got: Result<u32, ()> = cache.get_or_build(k, spec, slots, || {
            builds.set(builds.get() + 1);
            Ok(builds.get())
        });
        got.expect("build succeeds")
    }

    #[test]
    fn same_key_and_spec_hit_the_cache() {
        let mut cache = PipelineCache::new();
        let builds = Cell::new(0);
        let k = key(PassKind::Forward, 4);
        let first = lookup(&mut cache, &builds, k, &forward_spec(), 7);
        let again = lookup(&mut cache, &builds, k, &forward_spec().clone(), 7);
        assert_eq!((first, again, builds.get(), cache.len()), (1, 1, 1, 1));
    }

    #[test]
    fn other_key_or_spec_misses() {
        let mut cache = PipelineCache::new();
        let builds = Cell::new(0);
        let spec = forward_spec();
        lookup(&mut cache, &builds, key(PassKind::Forward, 4), &spec, 7);
        lookup(&mut cache, &builds, key(PassKind::Forward, 1), &spec, 7);
        lookup(&mut cache, &builds, key(PassKind::Prepass, 4), &spec, 7);
        let flagged = VariantKey {
            flags: crate::FLAG_DOUBLE_SIDED,
            ..key(PassKind::Forward, 4)
        };
        lookup(&mut cache, &builds, flagged, &spec, 7);
        let double_sided = PipelineSpec {
            cull: None,
            ..forward_spec()
        };
        lookup(
            &mut cache,
            &builds,
            key(PassKind::Forward, 4),
            &double_sided,
            7,
        );
        let coverage = PipelineSpec {
            alpha_to_coverage: true,
            ..forward_spec()
        };
        lookup(&mut cache, &builds, key(PassKind::Forward, 4), &coverage, 7);
        assert_eq!((builds.get(), cache.len()), (6, 6));
        // Every one of them is now a hit.
        lookup(&mut cache, &builds, key(PassKind::Forward, 1), &spec, 7);
        lookup(&mut cache, &builds, key(PassKind::Forward, 4), &coverage, 7);
        assert_eq!(builds.get(), 6);
    }

    #[test]
    fn slots_change_misses_and_clears_the_cache() {
        let mut cache = PipelineCache::new();
        let builds = Cell::new(0);
        let k = key(PassKind::Forward, 4);
        let spec = forward_spec();
        lookup(&mut cache, &builds, k, &spec, 7);
        lookup(&mut cache, &builds, key(PassKind::Prepass, 1), &spec, 7);
        assert_eq!(cache.len(), 2);
        assert_eq!(lookup(&mut cache, &builds, k, &spec, 8), 3);
        assert_eq!(cache.len(), 1, "entries of the old slots are dropped");
        assert_eq!(lookup(&mut cache, &builds, k, &spec, 8), 3);
        // Back to the old slots: composed again, not served from a stale entry.
        assert_eq!(lookup(&mut cache, &builds, k, &spec, 7), 4);
        cache.clear();
        assert_eq!(lookup(&mut cache, &builds, k, &spec, 7), 5);
    }

    #[test]
    fn failed_build_is_not_cached() {
        let mut cache: PipelineCache<u32> = PipelineCache::new();
        let k = key(PassKind::Forward, 4);
        let spec = forward_spec();
        let err = cache.get_or_build(k, &spec, 7, || Err(check_msaa(2).unwrap_err()));
        assert_eq!(err.unwrap_err().code(), "render.invalid_msaa");
        assert_eq!(cache.len(), 0);
        let builds = Cell::new(0);
        assert_eq!(lookup(&mut cache, &builds, k, &spec, 7), 1);
    }

    #[test]
    fn equal_specs_hash_equal() {
        assert_eq!(
            spec_hash(&forward_spec()),
            spec_hash(&forward_spec().clone())
        );
        assert_ne!(
            spec_hash(&forward_spec()),
            spec_hash(&PipelineSpec::default())
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn identical_sources_share_modules_and_pipelines_but_state_does_not() {
        let Some(gpu) = raa_core::testing::software_gpu() else {
            raa_core::testing::skip("content_cache", "no adapter");
            return;
        };
        let lighting = raa_pipeline::LightingResources::new(&gpu);
        let mut shared = LitShared::new(&gpu, lighting.layout(), 1, 1, 4).unwrap();
        let source = shared.fns.get(MaterialFnId::STANDARD).unwrap().wgsl.clone();
        let a = shared.register_user_fn("first", &source).unwrap();
        let b = shared.register_user_fn("second", &source).unwrap();
        let spec = forward_spec();
        let variant = |material_fn| VariantKey {
            material_fn,
            ..key(PassKind::Forward, 4)
        };
        let first = shared.pipeline_for(&gpu, variant(a), &spec).unwrap();
        let second = shared.pipeline_for(&gpu, variant(b), &spec).unwrap();
        assert_eq!(first, second);
        assert_eq!(shared.modules.borrow().len(), 1);
        assert_eq!(shared.content_pipelines.borrow().len(), 1);
        let other = shared
            .pipeline_for(&gpu, variant(b), &PipelineSpec { cull: None, ..spec })
            .unwrap();
        assert_ne!(first, other);
        assert_eq!(shared.modules.borrow().len(), 1);
        assert_eq!(shared.content_pipelines.borrow().len(), 2);
    }

    /// Expected stride, step mode and (location, offset) of one vertex buffer.
    type ExpectedBuffer = (u64, wgpu::VertexStepMode, &'static [(u32, u64)]);

    #[test]
    fn vertex_buffers_follow_s10() {
        assert_eq!((STREAM0_SLOT, INSTANCE_SLOT, STREAM1_SLOT), (0, 1, 2));
        let layouts = vertex_buffer_layouts();
        let expect: [ExpectedBuffer; 3] = [
            (24, wgpu::VertexStepMode::Vertex, &[(0, 0), (1, 12)]),
            (4, wgpu::VertexStepMode::Instance, &[(2, 0)]),
            (
                40,
                wgpu::VertexStepMode::Vertex,
                &[(10, 0), (11, 8), (12, 24)],
            ),
        ];
        for (layout, (stride, step, attrs)) in layouts.iter().zip(expect) {
            let layout = layout.as_ref().expect("every slot has a layout");
            assert_eq!((layout.array_stride, layout.step_mode), (stride, step));
            let got: Vec<(u32, u64)> = layout
                .attributes
                .iter()
                .map(|a| (a.shader_location, a.offset))
                .collect();
            assert_eq!(got, attrs);
        }
        assert_eq!(INSTANCE_ATTRIBUTES[0].format, wgpu::VertexFormat::Uint32);
    }

    #[test]
    fn frame_layout_follows_s10_group0() {
        let entries = frame_layout_entries();
        let (buffers, rest) = entries.split_at(3);
        let (textures, data) = rest.split_at(3);
        let sizes: Vec<(u32, bool, Option<u64>)> = buffers
            .iter()
            .map(|e| match e.ty {
                wgpu::BindingType::Buffer {
                    ty,
                    min_binding_size,
                    ..
                } => (
                    e.binding,
                    ty == wgpu::BufferBindingType::Uniform,
                    min_binding_size.map(NonZeroU64::get),
                ),
                _ => panic!("group 0 bindings 0..3 are buffers"),
            })
            .collect();
        assert_eq!(
            sizes,
            [
                (0, true, Some(512)),
                (1, false, Some(256)),
                (2, false, Some(128))
            ]
        );
        for e in buffers.iter().skip(1) {
            assert!(matches!(
                e.ty,
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    ..
                }
            ));
        }
        // Material textures: atlas, filtering sampler, read-only rects of 16 B.
        assert_eq!(
            textures.iter().map(|e| e.binding).collect::<Vec<_>>(),
            [3, 4, 5]
        );
        assert!(matches!(
            textures[0].ty,
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            }
        ));
        assert!(matches!(
            textures[1].ty,
            wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)
        ));
        assert!(matches!(
            textures[2].ty,
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                min_binding_size: Some(n),
                ..
            } if n.get() == 16
        ));
        // Material data words (binding 6) and the instance records (binding 7): read-only
        // storage, both stages.
        assert_eq!(data.len(), 2);
        assert_eq!((data[0].binding, data[1].binding), (6, 7));
        for (entry, min) in data.iter().zip([4, 128]) {
            assert_eq!(entry.visibility, wgpu::ShaderStages::VERTEX_FRAGMENT);
            assert!(matches!(
                entry.ty,
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    min_binding_size: Some(n),
                    ..
                } if n.get() == min
            ));
        }
    }

    #[test]
    fn shader_refusals_keep_their_code() {
        let composer = Composer::new();
        let reg = MaterialFnRegistry::new();
        let missing = composer
            .compose(&reg, &key(PassKind::Forward, 4))
            .map_err(shader_error)
            .unwrap_err();
        assert_eq!(missing.code(), "shader.slot_missing");
        assert!(!missing.message().is_empty());
        let unknown = VariantKey {
            material_fn: MaterialFnId(99),
            ..key(PassKind::Forward, 4)
        };
        let err = composer.compose(&reg, &unknown).map_err(shader_error);
        assert_eq!(err.unwrap_err().code(), "shader.unknown_fn");
        let bad = Composer::validate("fn broken(").map_err(shader_error);
        assert_eq!(bad.unwrap_err().code(), "shader.parse");
    }

    #[test]
    fn msaa_must_be_1_or_4() {
        assert!(check_msaa(1).is_ok() && check_msaa(4).is_ok());
        for n in [0, 2, 8] {
            let err = check_msaa(n).unwrap_err();
            assert_eq!(err.code(), "render.invalid_msaa");
            assert!(err.message().contains(&n.to_string()));
        }
    }
}
