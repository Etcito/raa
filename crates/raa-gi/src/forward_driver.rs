//! Driver A of the GI volume (S37): voxelizes every member no other driver claims, one compute
//! dispatch of `raa_gi_voxelize_forward` per (member, rebuilt level) work item.
//!
//! LIFECYCLE (contract 'gi-frame-lifecycle'): `prepare_frame` runs inside
//! `GiHandle::prepare_frame` after `GiVolume::prepare`; `record_level` runs inside the
//! `gi.volume` record (`GiVolume::record`); `release` runs on reallocation and inactive frames.
//!
//! The lit mesh buffers come from contract 'lit-mesh-storage' (`LitHandle::mesh_storage`,
//! `LitHandle::lod_chain_into`); bind groups are keyed by `LitMeshStorage::key`, which follows
//! the `Handoff:` doc comment on `GpuLitMesh::key` (crates/raa-lit/src/meshes.rs:202).

use std::cell::RefCell;
use std::collections::HashMap;
use std::num::NonZeroU64;

use raa_core::GpuContext;
use raa_lit::{LitHandle, LitMeshStorage, LitRenderer, LodLevel};
use raa_vg::scene::SyncItem;

use crate::clipmap;
use crate::layouts::{GI_WORK_STRIDE, GiForwardWorkGpu};
use crate::membership::{Driver, Member};
use crate::parts::{LevelVoxelize, VoxelDriver};
use crate::shaders;

/// Work-uniform slots allocated on first use.
const INITIAL_SLOTS: usize = 64;
/// Threads per workgroup of `raa_gi_voxelize_forward`.
const WORKGROUP: u32 = 64;
/// Largest workgroup count of one dispatch dimension.
const MAX_GROUPS: u32 = 65535;

/// One dispatch: a member's chosen mesh range voxelized into one level.
struct WorkItem {
    level: u32,
    record: GiForwardWorkGpu,
    storage: LitMeshStorage,
}

/// Bind-group cache key: (`LitMeshStorage::key`, lit storage generation, level).
type CacheKey = (u64, u64, u32);

/// The built-in forward voxel driver (driver A).
pub struct ForwardDriver {
    device: wgpu::Device,
    lit: LitHandle,
    pipeline: wgpu::ComputePipeline,
    group2: wgpu::BindGroupLayout,
    work: RefCell<Vec<WorkItem>>,
    uniform: RefCell<Option<(wgpu::Buffer, usize)>>,
    scratch: RefCell<Vec<LodLevel>>,
    cache: RefCell<HashMap<CacheKey, wgpu::BindGroup>>,
}

impl ForwardDriver {
    /// Creates the `raa_gi_voxelize_forward` pipeline (layout [None, None, group 2], contract
    /// 'gi-shaders'); buffers are created on the first `prepare_frame`.
    pub fn new(gpu: &GpuContext, lit: LitHandle) -> ForwardDriver {
        let device = gpu.device().clone();
        let group2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gi.forward.group2"),
            entries: &shaders::forward_group2_entries(),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gi.forward"),
            bind_group_layouts: &[None, None, Some(&group2)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gi.forward"),
            source: wgpu::ShaderSource::Wgsl(shaders::voxelize_forward_module_wgsl().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gi.forward"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("raa_gi_voxelize_forward"),
            compilation_options: Default::default(),
            cache: None,
        });
        ForwardDriver {
            device,
            lit,
            pipeline,
            group2,
            work: RefCell::new(Vec::new()),
            uniform: RefCell::new(None),
            scratch: RefCell::new(Vec::new()),
            cache: RefCell::new(HashMap::new()),
        }
    }

    /// Ensures the work uniform holds `slots` slots, doubling it; growth clears the cache.
    fn ensure_slots(&self, slots: usize) {
        let mut uniform = self.uniform.borrow_mut();
        let current = uniform.as_ref().map_or(0, |(_, n)| *n);
        if current >= slots && current > 0 {
            return;
        }
        let mut count = current.max(INITIAL_SLOTS);
        while count < slots {
            count *= 2;
        }
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gi.forward.work"),
            size: (count * GI_WORK_STRIDE) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        *uniform = Some((buffer, count));
        self.cache.borrow_mut().clear();
    }
}

/// The largest column length of the 3x3 part of `world`.
fn world_scale(item: &SyncItem) -> f32 {
    item.world.cols[..3]
        .iter()
        .map(|c| (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt())
        .fold(0.0, f32::max)
}

impl VoxelDriver for ForwardDriver {
    fn name(&self) -> &'static str {
        "forward"
    }

    fn claims(&self, _item: &SyncItem, _lit: &LitRenderer) -> bool {
        false
    }

    fn prepare_frame(&self, gpu: &GpuContext, members: &[Member], levels: &[LevelVoxelize]) {
        let mut work = self.work.borrow_mut();
        let mut scratch = self.scratch.borrow_mut();
        work.clear();
        for level in levels {
            for member in members.iter().filter(|m| m.driver == Driver::Forward) {
                let mesh = if self.lit.lod_chain_into(member.item.mesh, &mut scratch) {
                    let index =
                        clipmap::forward_lod(&scratch, world_scale(&member.item), level.voxel_size);
                    scratch.get(index).map_or(member.item.mesh, |l| l.mesh)
                } else {
                    member.item.mesh
                };
                let Some(storage) = self.lit.mesh_storage(mesh) else {
                    continue;
                };
                work.push(WorkItem {
                    level: level.level,
                    record: GiForwardWorkGpu {
                        slot: member.item.slot,
                        first_index: storage.first_index,
                        triangle_count: storage.index_count / 3,
                        level: level.level,
                    },
                    storage,
                });
            }
        }
        if work.is_empty() {
            return;
        }
        self.ensure_slots(work.len());
        let uniform = self.uniform.borrow();
        let Some((buffer, _)) = uniform.as_ref() else {
            return;
        };
        for (k, item) in work.iter().enumerate() {
            let offset = (k * GI_WORK_STRIDE) as u64;
            gpu.queue()
                .write_buffer(buffer, offset, &item.record.to_bytes());
        }
    }

    fn record_level(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        level: &LevelVoxelize,
        voxels: &wgpu::Buffer,
    ) {
        let work = self.work.borrow();
        let uniform = self.uniform.borrow();
        let Some((buffer, _)) = uniform.as_ref() else {
            return;
        };
        if !work.iter().any(|item| item.level == level.level) {
            return;
        }
        let generation = self.lit.storage_generation();
        let mut cache = self.cache.borrow_mut();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gi.forward"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        for (k, item) in work.iter().enumerate() {
            if item.level != level.level {
                continue;
            }
            let key = (item.storage.key, generation, level.level);
            let group = cache.entry(key).or_insert_with(|| {
                let instances = self.lit.instance_storage();
                let extras = self.lit.extras_storage();
                let materials = self.lit.material_storage();
                let resources = [
                    level.params.as_entire_binding(),
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: 0,
                        size: NonZeroU64::new(GiForwardWorkGpu::SIZE as u64),
                    }),
                    instances.as_entire_binding(),
                    extras.as_entire_binding(),
                    materials.as_entire_binding(),
                    item.storage.stream0.as_entire_binding(),
                    item.storage.index.as_entire_binding(),
                    voxels.as_entire_binding(),
                ];
                let entries: Vec<wgpu::BindGroupEntry> = resources
                    .into_iter()
                    .zip(0u32..)
                    .map(|(resource, binding)| wgpu::BindGroupEntry { binding, resource })
                    .collect();
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("gi.forward.group2"),
                    layout: &self.group2,
                    entries: &entries,
                })
            });
            let offset = (k * GI_WORK_STRIDE) as u32;
            pass.set_bind_group(2, &*group, &[offset]);
            let groups = item
                .record
                .triangle_count
                .div_ceil(WORKGROUP)
                .min(MAX_GROUPS);
            if groups > 0 {
                pass.dispatch_workgroups(groups, 1, 1);
            }
        }
    }

    fn voxelized_instances(&self) -> u32 {
        let mut slots: Vec<u32> = self.work.borrow().iter().map(|i| i.record.slot).collect();
        slots.sort_unstable();
        slots.dedup();
        slots.len() as u32
    }

    fn release(&self) {
        self.work.borrow_mut().clear();
        self.cache.borrow_mut().clear();
        *self.uniform.borrow_mut() = None;
    }

    fn allocated_bytes(&self) -> u64 {
        self.uniform.borrow().as_ref().map_or(0, |(b, _)| b.size())
    }
}
