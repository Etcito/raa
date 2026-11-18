//! Driver B of the GI volume (S37): virtualized-geometry members are voxelized from the clusters
//! their per-level GI cull selects, one indirect dispatch of `raa_gi_voxelize_virtual` per rebuilt
//! level. Driver B is installed beside driver A only when the engine has a `VgHandle`.
//!
//! LIFECYCLE (contract 'vg-gi-frame' LIFECYCLE): `VgHandle::prepare_frame` runs from the m29 api
//! frame before `GiHandle::prepare_frame`, and `gi.volume` (order 25) records after `vg.resolve`
//! (order 23), so every `with_gi_frame` here sees this frame's VG resources. `GiPart::prepare_frame`
//! runs on every frame, active or not, and publishes the member tags; `VoxelDriver::prepare_frame`
//! runs on active frames with `work.voxelize`; `release` runs on reallocation and inactive frames.
//!
//! The contract 'gi-virtual-kernel' builders live here because `shaders.rs` belongs to
//! raa-gi-volume: `voxelize_virtual_module_wgsl` assembles the module sources and
//! `virtual_group2_entries` the ten group-2 entries.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::rc::Rc;

use raa_core::{GpuContext, Lazy, RaaError, lazy_compute_pipeline};
use raa_lit::{LitHandle, LitRenderer};
use raa_vg::GI_BASE;
use raa_vg::culling::{CullPhase, CullWork};
use raa_vg::feature::VgHandle;
use raa_vg::scene::SyncItem;

use crate::clipmap;
use crate::membership::{Driver, Member};
use crate::parts::{GiPart, LevelVoxelize, PartFrame, PartsCtx, VoxelDriver};
use crate::shaders;
use crate::vertex_colors::{GiVertexColors, POOL_COLOR_WIDTH, WHITE_TEXEL, pool_rows};

const VOXELIZE_VIRTUAL_WGSL: &str = include_str!("../wgsl/voxelize_virtual.wgsl");

/// Bind-group cache key: (`VgScene::generation`, `LitHandle::storage_generation`, level).
type CacheKey = (u64, u64, u32);

/// The state the pool colour texture was built from: (scene generation, colour generation, pool
/// vertex capacity).
type PoolKey = (u64, u64, u64);

/// The virtual-voxelize module (driver B): `raa_shader::DEPTH_WGSL` + `PASS_VIEW_WGSL` +
/// `raa_vg::RASTER_COMMON_WGSL` + `VOXELIZE_KERNEL_WGSL` + `voxelize_virtual.wgsl`, entry
/// `raa_gi_voxelize_virtual` at `@workgroup_size(128)`, one workgroup per selected cluster and
/// one lane per triangle.
///
/// Handoff: `voxelize_virtual_module_wgsl` and `virtual_group2_entries` are taken over by the
/// card 'raa-gi: tests/virtual_variants.rs for the driver B kernel, group and level view', which
/// calls both from crates/raa-gi/tests/virtual_variants.rs through the generated module
/// `raa_gi::parts::virtual_driver`. Today the builder returns exactly that concatenation and its
/// only entry point is `raa_gi_voxelize_virtual`, and `virtual_group2_entries` returns the ten
/// COMPUTE entries of contract 'gi-virtual-kernel' (bindings 1-8 storage, binding 9 the 32-B
/// `GiVoxelizeGpu`, binding 10 the optional vertex colour texture). The state they carry is the group-2 binding table and the entry-point name.
pub fn voxelize_virtual_module_wgsl() -> String {
    [
        raa_shader::DEPTH_WGSL,
        raa_vg::culling::PASS_VIEW_WGSL,
        raa_vg::RASTER_COMMON_WGSL,
        shaders::VOXELIZE_KERNEL_WGSL,
        VOXELIZE_VIRTUAL_WGSL,
    ]
    .concat()
}

/// Group 2 of `raa_gi_voxelize_virtual`: bindings 1-6 as `RASTER_COMMON_WGSL` declares them
/// (instances, clusters, vertices, triangles, the cull `visible` list and its `state`), 7 the
/// `RaaGiVgMaterial` array of `LitHandle::material_storage`, 8 the voxels read-write and 9 the
/// `GiVoxelizeGpu` uniform of the level, 10 the optional vertex colours as a non-filterable
/// `Rgba8Unorm` 2D texture parallel to the vertex pool (a texture, because the eight storage
/// buffers of the stage are all taken). Binding 0 (`view`) is declared but unused, so it has no
/// entry.
pub fn virtual_group2_entries() -> [wgpu::BindGroupLayoutEntry; 10] {
    [
        storage_entry(1, true),
        storage_entry(2, true),
        storage_entry(3, true),
        storage_entry(4, true),
        storage_entry(5, true),
        storage_entry(6, true),
        storage_entry(7, true),
        storage_entry(8, false),
        wgpu::BindGroupLayoutEntry {
            binding: 9,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(32),
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 10,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
    ]
}

/// A COMPUTE storage-buffer entry.
fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// The GI virtual-geometry voxel driver (driver B).
pub struct VirtualDriver {
    device: wgpu::Device,
    vg: VgHandle,
    lit: LitHandle,
    /// Built on the first frame with virtual geometry to voxelize (raa_core::Lazy says why).
    pipeline: Lazy<wgpu::ComputePipeline>,
    group2: wgpu::BindGroupLayout,
    work: RefCell<Option<CullWork>>,
    capacity: Cell<Option<u32>>,
    claimed: Cell<u32>,
    cache: RefCell<HashMap<CacheKey, wgpu::BindGroup>>,
    slots: RefCell<Vec<u32>>,
    tags: RefCell<Vec<u32>>,
    colors: GiVertexColors,
    /// The pool-parallel colour texture of the registered streams and the state it was built
    /// from: (scene generation, colour generation, pool vertex capacity).
    pool_colors: RefCell<Option<(wgpu::TextureView, PoolKey)>>,
    /// One white texel, bound while no stream is registered.
    white: wgpu::TextureView,
}

impl VirtualDriver {
    /// Creates the `raa_gi_voxelize_virtual` pipeline (layout `[None, None, group 2]`, contract
    /// 'gi-virtual-kernel'); the `CullWork` is created on the first prepared frame inside
    /// contract 'vg-gi-frame'.
    fn new(
        gpu: &GpuContext,
        lit: LitHandle,
        vg: VgHandle,
        colors: GiVertexColors,
    ) -> VirtualDriver {
        let device = gpu.device().clone();
        let group2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gi.virtual.group2"),
            entries: &virtual_group2_entries(),
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gi.virtual"),
            source: wgpu::ShaderSource::Wgsl(voxelize_virtual_module_wgsl().into()),
        });
        let pipeline = lazy_compute_pipeline(
            &device,
            "gi.virtual",
            &module,
            "raa_gi_voxelize_virtual",
            &[None, None, Some(&group2)],
            &[],
        );
        let white = color_texture(gpu, 1, 1, &[WHITE_TEXEL]).create_view(&Default::default());
        VirtualDriver {
            device,
            vg,
            lit,
            pipeline,
            group2,
            work: RefCell::new(None),
            capacity: Cell::new(None),
            claimed: Cell::new(0),
            cache: RefCell::new(HashMap::new()),
            slots: RefCell::new(Vec::new()),
            tags: RefCell::new(Vec::new()),
            colors,
            pool_colors: RefCell::new(None),
            white,
        }
    }

    /// Keeps the pool colour texture in step with the registered streams and the scene pool:
    /// rebuilt when the streams or the pool changed, dropped (white dummy) with no stream.
    fn sync_pool_colors(&self, gpu: &GpuContext, scene: &raa_vg::scene::VgScene) {
        let mut pool = self.pool_colors.borrow_mut();
        let capacity = scene.vertices().map_or(0, |b| b.size() / 16);
        let rows = pool_rows(capacity);
        let wanted = self.colors.has_clusters() && capacity > 0 && rows.is_some();
        if !wanted {
            if pool.take().is_some() {
                self.cache.borrow_mut().clear();
            }
            return;
        }
        let key = (scene.generation(), self.colors.cluster_generation(), capacity);
        if pool.as_ref().is_some_and(|(_, built)| *built == key) {
            return;
        }
        let rows = rows.unwrap_or(1);
        let mut texels = vec![WHITE_TEXEL; (POOL_COLOR_WIDTH * rows) as usize];
        self.colors.fill_pool(scene, &mut texels);
        let view = color_texture(gpu, POOL_COLOR_WIDTH, rows, &texels).create_view(&Default::default());
        *pool = Some((view, key));
        self.cache.borrow_mut().clear();
    }
}

/// An `Rgba8Unorm` texture of `width` x `height` packed colour words.
fn color_texture(gpu: &GpuContext, width: u32, height: u32, texels: &[u32]) -> wgpu::Texture {
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("gi.virtual.colors"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let bytes: Vec<u8> = texels.iter().flat_map(|t| t.to_le_bytes()).collect();
    gpu.queue().write_texture(
        texture.as_image_copy(),
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        size,
    );
    texture
}

impl VoxelDriver for VirtualDriver {
    fn name(&self) -> &'static str {
        "virtual"
    }

    fn claims(&self, item: &SyncItem, lit: &LitRenderer) -> bool {
        lit.virtual_instances().contains(item.instance)
    }

    fn prepare_frame(&self, gpu: &GpuContext, members: &[Member], levels: &[LevelVoxelize]) {
        let mut slots = self.slots.borrow_mut();
        slots.clear();
        slots.extend(
            members
                .iter()
                .filter(|member| member.driver == Driver::Virtual)
                .map(|member| member.item.slot),
        );
        slots.sort_unstable();
        slots.dedup();
        self.claimed.set(slots.len().min(u32::MAX as usize) as u32);
        let capacity = self.vg.get().max_visible_clusters;
        self.vg.with_gi_frame(|frame| {
            let Some(frame) = frame else {
                return;
            };
            let missing = self.work.borrow().is_none();
            if missing || self.capacity.get() != Some(capacity) {
                let size = u64::from(capacity);
                *self.work.borrow_mut() =
                    Some(CullWork::new(gpu, frame.cull, "raa-gi virtual", size, size));
                self.capacity.set(Some(capacity));
                self.cache.borrow_mut().clear();
            }
            self.sync_pool_colors(gpu, frame.scene);
            for level in levels {
                let view = clipmap::level_view(level.origin, level.voxel_size, level.resolution);
                frame.slots.write(gpu, GI_BASE + level.level, &view);
            }
        });
    }

    fn record_level(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        level: &LevelVoxelize,
        voxels: &wgpu::Buffer,
    ) {
        self.vg.with_gi_frame(|frame| {
            let Some(frame) = frame else {
                return;
            };
            let work = self.work.borrow();
            let Some(work) = work.as_ref() else {
                return;
            };
            let (Some(instances), Some(clusters), Some(vertices), Some(triangles)) = (
                frame.scene.instances(),
                frame.scene.clusters(),
                frame.scene.vertices(),
                frame.scene.triangles(),
            ) else {
                return;
            };
            if frame.scene.instance_count() == 0 {
                return;
            }
            frame.cull.record(
                encoder,
                frame.scene,
                frame.cull_scene,
                GI_BASE + level.level,
                work,
                CullPhase::First,
            );
            let key = (
                frame.scene.generation(),
                self.lit.storage_generation(),
                level.level,
            );
            let mut cache = self.cache.borrow_mut();
            let pool_colors = self.pool_colors.borrow();
            let group = cache.entry(key).or_insert_with(|| {
                let materials = self.lit.material_storage();
                let colors = pool_colors.as_ref().map_or(&self.white, |(view, _)| view);
                let entries = [
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: instances.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: clusters.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: vertices.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: triangles.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: work.visible().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: work.state().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: materials.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: voxels.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: level.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: wgpu::BindingResource::TextureView(colors),
                    },
                ];
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gi.virtual.group2"),
                    layout: &self.group2,
                    entries: &entries,
                })
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gi.virtual"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.pipeline.get());
            pass.set_bind_group(2, &*group, &[]);
            pass.dispatch_workgroups_indirect(work.args(), CullWork::software_args_offset());
        });
    }

    fn voxelized_instances(&self) -> u32 {
        self.claimed.get()
    }

    fn release(&self) {
        *self.work.borrow_mut() = None;
        self.capacity.set(None);
        self.claimed.set(0);
        *self.pool_colors.borrow_mut() = None;
        self.cache.borrow_mut().clear();
    }

    fn allocated_bytes(&self) -> u64 {
        self.work
            .borrow()
            .as_ref()
            .map_or(0, CullWork::allocated_bytes)
    }
}

impl GiPart for VirtualDriver {
    fn name(&self) -> &'static str {
        "virtual"
    }

    /// Publishes the sorted, deduplicated tags of every member, whichever driver voxelizes it,
    /// through `VgHandle::set_gi_members` (contract 'vg-gi-frame'), so `INSTANCE_GI` reaches the
    /// VG records before GI's first allocation frame and before an instance switches to driver B.
    fn prepare_frame(&self, _gpu: &GpuContext, frame: &PartFrame<'_>) {
        let mut tags = self.tags.borrow_mut();
        tags.clear();
        tags.extend(frame.members.iter().map(|member| member.item.tag));
        tags.sort_unstable();
        tags.dedup();
        self.vg.set_gi_members(&tags);
    }

    fn release(&self) {
        VoxelDriver::release(self);
    }

    fn allocated_bytes(&self) -> u64 {
        0
    }
}

/// Adds the part `virtual` and its driver when the engine has a `VgHandle`; without virtualized
/// geometry driver A keeps every member. The same `Rc` is registered both ways.
pub(crate) fn install(ctx: &mut PartsCtx<'_>) -> Result<(), RaaError> {
    let Some(vg) = ctx.vg.clone() else {
        return Ok(());
    };
    let driver = Rc::new(VirtualDriver::new(ctx.gpu, ctx.lit.handle(), vg, ctx.shared.vertex_colors()));
    ctx.add_driver(Rc::clone(&driver) as Rc<dyn VoxelDriver>);
    ctx.add_part(driver);
    Ok(())
}
