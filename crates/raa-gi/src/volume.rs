// Adapted from solstice crates/solstice-render/src/gi/volume.rs at 7e465b5: the GPU volume of the
// GI clipmap (its buffers, atlas, per-level uniforms, flood parameter slots and compute pipelines
// with their bind groups).
//
// MIT License
//
// Copyright (c) 2026 SamG-Coder
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//! The GPU volume of the GI clipmap (S37, S38), crate-private: share 1, allocation and pipelines
//! (contract 'gi-volume-internal'). Per-frame prepare and recording come in later shares.

use std::rc::Rc;

use raa_core::GpuContext;
use raa_math::Vec3;

use crate::clipmap::{self, RebuildRounds, RelightCursor};
use crate::layouts::{GiFloodGpu, GiLightGpu, GiVolumeParamsGpu, GiVoxelizeGpu, GiWriteGpu};
use crate::parts::{LevelVoxelize, VoxelDriver};
use crate::settings::{GiQuality, GiSettings};
use crate::shaders;

/// Byte stride of one flood parameter slot: the dynamic-offset alignment of the uniform.
const SLOT_BYTES: u64 = 256;

/// `RAA_GI_FLAG_VALID`: every level has been committed once (S38; `prelude.wgsl:164`).
const FLAG_VALID: u32 = 1;

/// `RAA_GI_FLAG_REFLECTIONS`: the volume is valid and reflections are on (S38;
/// `prelude.wgsl:165`).
const FLAG_REFLECTIONS: u32 = 2;

/// `RAA_GI_FLAG_LOCAL_SHADOWS`: local-light shadows are on (S38; `prelude.wgsl:166`).
const FLAG_LOCAL_SHADOWS: u32 = 4;

/// One slice of a pending rebuild: its level, begin and commit flags and flood passes.
#[derive(Clone, Debug)]
pub(crate) struct RebuildSlice {
    pub level: usize,
    pub begin: bool,
    pub passes: std::ops::Range<usize>,
    pub commit: bool,
}

/// The frame's rebuild work: the voxelize levels, the rebuild slice and the level to relight.
#[derive(Clone, Debug)]
pub(crate) struct VolumeWork {
    pub rebuild_all: bool,
    pub slice: Option<RebuildSlice>,
    pub relight: usize,
    pub voxelize: Vec<LevelVoxelize>,
}

/// The GPU volume of the GI clipmap (S37, S38): the six volume resources, the per-level
/// uniforms, the flood parameter slots, and the seed, flood, voxel-light and write pipelines.
///
/// Handoff: `GiVolume` (this file) today holds the allocation of `GiVolume::new` with `origins`,
/// `pending`, `light_updates` and `params` at their initial values, since no frame is prepared
/// yet. Card 'raa-gi: volume.rs share 2, prepare' adds `GiVolume::prepare` over these fields.
/// Handoff: `GiVolume` (this file) today holds the four pipelines and their bind groups of
/// `GiVolume::new` and records nothing. Card 'raa-gi: volume.rs share 3, record' adds
/// `GiVolume::record` over them.
///
/// Lifecycle: created and dropped only by `GiHandle::prepare_frame` (contract 'gi-frame-lifecycle').
pub(crate) struct GiVolume {
    quality: GiQuality,
    voxels: wgpu::Buffer,
    seeds: wgpu::Buffer,
    _radiance: wgpu::Buffer,
    flood_a: wgpu::Buffer,
    _flood_b: wgpu::Buffer,
    flood_params: wgpu::Buffer,
    voxelize_params: Vec<wgpu::Buffer>,
    light_params: Vec<wgpu::Buffer>,
    write_params: Vec<wgpu::Buffer>,
    atlas_view: wgpu::TextureView,
    seed_pipeline: wgpu::ComputePipeline,
    flood_pipeline: wgpu::ComputePipeline,
    light_pipeline: wgpu::ComputePipeline,
    write_pipeline: wgpu::ComputePipeline,
    seed_bind: wgpu::BindGroup,
    flood_binds: [wgpu::BindGroup; 2],
    light_binds: Vec<wgpu::BindGroup>,
    write_binds: Vec<wgpu::BindGroup>,
    origins: Vec<Option<[i32; 3]>>,
    pending: Option<(usize, [i32; 3], usize)>,
    light_updates: Vec<u32>,
    params: [u8; 96],
}

fn buffer(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn layout(
    device: &wgpu::Device,
    label: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries,
    })
}

/// A compute pipeline of `source`'s entry point over the bind group layouts `groups`.
fn compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    entry: &str,
    groups: &[Option<&wgpu::BindGroupLayout>],
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: groups,
        immediate_size: 0,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Group-2 bind group whose entries are `resources` in binding order.
fn bind_group(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    resources: Vec<wgpu::BindingResource>,
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = resources
        .into_iter()
        .zip(0u32..)
        .map(|(resource, binding)| wgpu::BindGroupEntry { binding, resource })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &entries,
    })
}

/// One flood parameter slot as a dynamic-offset binding of exactly one record, so a slot's
/// offset plus its size stays inside the buffer.
fn flood_slot(params: &wgpu::Buffer) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
        buffer: params,
        offset: 0,
        size: wgpu::BufferSize::new(GiFloodGpu::SIZE as u64),
    })
}

impl GiVolume {
    /// Allocates the volume for `quality` (`GI_VOLUME_BYTES`) and builds its pipelines. `frame`
    /// is the 432-B `GiFrameGpu` buffer; `group1_layout` is the compute group 1 layout, taken by
    /// the voxel-light pipeline. Every origin starts as `None`.
    pub(crate) fn new(
        gpu: &GpuContext,
        quality: GiQuality,
        frame: &wgpu::Buffer,
        group1_layout: &wgpu::BindGroupLayout,
    ) -> GiVolume {
        let device = gpu.device();
        let queue = gpu.queue();
        let resolution = quality.resolution;
        let levels = quality.levels as usize;
        let level_cells = resolution * resolution * resolution;
        let cells = u64::from(level_cells);
        let volume_cells = cells * u64::from(quality.levels);

        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let flood = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let uniform = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let voxels = buffer(device, "raa-gi.voxels", 12 * volume_cells, storage);
        let seeds = buffer(device, "raa-gi.seeds", 4 * volume_cells, storage);
        let radiance = buffer(
            device,
            "raa-gi.radiance",
            8 * volume_cells,
            wgpu::BufferUsages::STORAGE,
        );
        let flood_a = buffer(device, "raa-gi.flood-a", 4 * cells, flood);
        let flood_b = buffer(device, "raa-gi.flood-b", 4 * cells, flood);
        let atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("raa-gi.atlas"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: resolution * quality.levels,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let atlas_view = atlas.create_view(&wgpu::TextureViewDescriptor::default());

        let per_level = |label: &str, size: u64| -> Vec<wgpu::Buffer> {
            (0..levels)
                .map(|_| buffer(device, label, size, uniform))
                .collect()
        };
        let voxelize_params = per_level("raa-gi.voxelize", GiVoxelizeGpu::SIZE as u64);
        let light_params = per_level("raa-gi.light", GiLightGpu::SIZE as u64);
        let write_params = per_level("raa-gi.write", GiWriteGpu::SIZE as u64);
        let passes = clipmap::flood_pass_count();
        let flood_params = buffer(
            device,
            "raa-gi.flood-params",
            (passes + levels) as u64 * SLOT_BYTES,
            uniform,
        );

        let steps = clipmap::flood_steps(resolution);
        for pass in 0..passes {
            let mut step = [0i32; 3];
            if let Some(axis) = step.get_mut(pass % 3) {
                *axis = steps.get(pass / 3).copied().unwrap_or(0);
            }
            let record = GiFloodGpu {
                step,
                resolution,
                level_offset: 0,
            };
            queue.write_buffer(&flood_params, pass as u64 * SLOT_BYTES, &record.to_bytes());
        }
        for level in 0..levels {
            let record = GiFloodGpu {
                step: [0; 3],
                resolution,
                level_offset: level as u32 * level_cells,
            };
            let slot = (passes + level) as u64;
            queue.write_buffer(&flood_params, slot * SLOT_BYTES, &record.to_bytes());
        }

        let empty = layout(device, "raa-gi.empty", &[]);
        let seed_layout = layout(device, "raa-gi.seed", &shaders::seed_group2_entries());
        let flood_layout = layout(device, "raa-gi.flood", &shaders::flood_group2_entries());
        let light_layout = layout(device, "raa-gi.light", &shaders::light_group2_entries());
        let write_layout = layout(device, "raa-gi.write", &shaders::write_group2_entries());
        let seed_pipeline = compute_pipeline(
            device,
            "raa-gi.seed",
            &shaders::seed_module_wgsl(),
            "raa_gi_seed",
            &[Some(&empty), Some(&empty), Some(&seed_layout)],
        );
        let flood_pipeline = compute_pipeline(
            device,
            "raa-gi.flood",
            &shaders::flood_module_wgsl(),
            "raa_gi_flood",
            &[Some(&empty), Some(&empty), Some(&flood_layout)],
        );
        let light_pipeline = compute_pipeline(
            device,
            "raa-gi.light",
            &shaders::voxel_light_module_wgsl(),
            "raa_gi_voxel_light",
            &[Some(&empty), Some(group1_layout), Some(&light_layout)],
        );
        let write_pipeline = compute_pipeline(
            device,
            "raa-gi.write",
            &shaders::volume_write_module_wgsl(),
            "raa_gi_volume_write",
            &[Some(&empty), Some(&empty), Some(&write_layout)],
        );

        let seed_bind = bind_group(
            device,
            "raa-gi.seed",
            &seed_layout,
            vec![
                flood_slot(&flood_params),
                voxels.as_entire_binding(),
                flood_a.as_entire_binding(),
            ],
        );
        let flood_pass = |read: &wgpu::Buffer, write: &wgpu::Buffer| {
            bind_group(
                device,
                "raa-gi.flood",
                &flood_layout,
                vec![
                    flood_slot(&flood_params),
                    read.as_entire_binding(),
                    write.as_entire_binding(),
                ],
            )
        };
        let flood_binds = [
            flood_pass(&flood_a, &flood_b),
            flood_pass(&flood_b, &flood_a),
        ];
        let light_binds = light_params
            .iter()
            .map(|params| {
                bind_group(
                    device,
                    "raa-gi.light",
                    &light_layout,
                    vec![
                        params.as_entire_binding(),
                        voxels.as_entire_binding(),
                        radiance.as_entire_binding(),
                        frame.as_entire_binding(),
                    ],
                )
            })
            .collect();
        let write_binds = write_params
            .iter()
            .map(|params| {
                bind_group(
                    device,
                    "raa-gi.write",
                    &write_layout,
                    vec![
                        params.as_entire_binding(),
                        seeds.as_entire_binding(),
                        radiance.as_entire_binding(),
                        wgpu::BindingResource::TextureView(&atlas_view),
                    ],
                )
            })
            .collect();

        GiVolume {
            quality,
            voxels,
            seeds,
            _radiance: radiance,
            flood_a,
            _flood_b: flood_b,
            flood_params,
            voxelize_params,
            light_params,
            write_params,
            atlas_view,
            seed_pipeline,
            flood_pipeline,
            light_pipeline,
            write_pipeline,
            seed_bind,
            flood_binds,
            light_binds,
            write_binds,
            origins: vec![None; levels],
            pending: None,
            light_updates: vec![0; levels],
            params: [0; 96],
        }
    }

    /// Follows the camera and decides this frame's work (solstice `volume.rs:398-529`,
    /// `GiVolume::prepare`). A first build rebuilds every level at its snapped origin in one
    /// frame; afterwards one pending slice advances a rebuild by `FLOOD_PASSES_PER_FRAME` of the
    /// 18 flood passes, storing the origin and completing the round on its last pass. Writes the
    /// per-level `GiVoxelizeGpu`, `GiLightGpu` and `GiWriteGpu` uniforms, relights one committed
    /// level (a level being rebuilt is never relit, solstice `volume.rs:454-466`) and stores the
    /// `GiVolumeParamsGpu` bytes for [`GiVolume::params_bytes`].
    pub(crate) fn prepare(
        &mut self,
        gpu: &GpuContext,
        camera_position: Vec3,
        rounds: &mut RebuildRounds,
        cursor: &mut RelightCursor,
        settings: &GiSettings,
    ) -> VolumeWork {
        rounds.start_waiting();
        let resolution = self.quality.resolution;
        let levels = self.quality.levels as usize;
        let first_build = self.origins.first().copied().flatten().is_none();
        let mut slice = None;
        if first_build {
            for level in 0..levels {
                let voxel = clipmap::voxel_size(level as u32);
                let origin = clipmap::snapped_origin(camera_position, voxel, resolution);
                if let Some(slot) = self.origins.get_mut(level) {
                    *slot = Some(origin);
                }
                rounds.complete(level);
            }
        } else {
            if self.pending.is_none() {
                self.pending = clipmap::choose_rebuild(&self.origins, camera_position, resolution)
                    .map(|(level, origin)| (level, origin, 0))
                    .or_else(|| {
                        rounds.choose_outdated().and_then(|level| {
                            let origin = self.origins.get(level).copied().flatten()?;
                            Some((level, origin, 0))
                        })
                    });
            }
            if let Some((level, origin, next_pass)) = self.pending {
                let passes = clipmap::flood_pass_count();
                let end = (next_pass + clipmap::FLOOD_PASSES_PER_FRAME).min(passes);
                let commit = end == passes;
                slice = Some(RebuildSlice {
                    level,
                    begin: next_pass == 0,
                    passes: next_pass..end,
                    commit,
                });
                self.pending = Some((level, origin, end));
                if commit {
                    if let Some(slot) = self.origins.get_mut(level) {
                        *slot = Some(origin);
                    }
                    rounds.complete(level);
                }
            }
        }
        let committed = slice.as_ref().is_some_and(|slice| slice.commit);
        let being_rebuilt = if committed {
            None
        } else {
            self.pending.map(|(level, _, _)| level)
        };
        let relight = cursor.next(self.quality.levels, being_rebuilt);
        if let Some(updates) = self.light_updates.get_mut(relight) {
            *updates = updates.wrapping_add(1);
        }

        let mut uniform = GiVolumeParamsGpu {
            resolution: resolution as f32,
            intensity: settings.intensity,
            level_count: self.quality.levels,
            ..Default::default()
        };
        let mut voxelize = Vec::new();
        for level in 0..levels {
            let origin = self.origins.get(level).copied().flatten().unwrap_or([0; 3]);
            let voxel = clipmap::voxel_size(level as u32);
            let voxelize_origin = self
                .pending
                .filter(|pending| pending.0 == level)
                .map_or(origin, |pending| pending.1);
            let level_offset = level as u32 * resolution * resolution * resolution;
            let record = GiVoxelizeGpu {
                origin: voxelize_origin.map(|c| c as f32 * voxel),
                voxel_size: voxel,
                resolution,
                level_offset,
            };
            if let Some(params) = self.voxelize_params.get(level) {
                gpu.queue().write_buffer(params, 0, &record.to_bytes());
            }
            let record = GiLightGpu {
                wrap_origin: origin,
                level: level as u32,
                updates: self.light_updates.get(level).copied().unwrap_or(0),
                resolution,
            };
            if let Some(params) = self.light_params.get(level) {
                gpu.queue().write_buffer(params, 0, &record.to_bytes());
            }
            let record = GiWriteGpu {
                wrap_origin: origin,
                level: level as u32,
                resolution,
                voxel_size: voxel,
            };
            if let Some(params) = self.write_params.get(level) {
                gpu.queue().write_buffer(params, 0, &record.to_bytes());
            }
            if let Some(slot) = uniform.levels.get_mut(level) {
                *slot = [
                    origin[0] as f32 * voxel,
                    origin[1] as f32 * voxel,
                    origin[2] as f32 * voxel,
                    voxel,
                ];
            }
            let begin = first_build || slice.as_ref().is_some_and(|s| s.begin && s.level == level);
            if begin && let Some(params) = self.voxelize_params.get(level).cloned() {
                voxelize.push(LevelVoxelize {
                    level: level as u32,
                    origin: voxelize_origin,
                    voxel_size: voxel,
                    resolution,
                    level_offset,
                    params,
                });
            }
        }

        let valid = self.all_committed();
        let mut flags = 0;
        if valid {
            flags |= FLAG_VALID;
            if settings.reflections {
                flags |= FLAG_REFLECTIONS;
            }
        }
        if settings.local_light_shadows {
            flags |= FLAG_LOCAL_SHADOWS;
        }
        uniform.flags = flags;
        self.params = uniform.to_bytes();
        if committed {
            self.pending = None;
        }
        VolumeWork {
            rebuild_all: first_build,
            slice,
            relight,
            voxelize,
        }
    }

    /// Records this frame's rebuilds and lighting update, in the order of solstice
    /// `crates/solstice-render/src/gi/volume.rs:531-683` (`GiVolume::record`). A rebuilt level
    /// clears its voxels, runs every driver, seeds, advances its flood passes, then copies flood
    /// A into its seeds and writes the atlas on commit. The relit level is lit and written last.
    /// Drivers replace solstice's cull and software dispatch; the light pass binds the given
    /// compute group 1.
    pub(crate) fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        work: &VolumeWork,
        drivers: &[Rc<dyn VoxelDriver>],
        group1: &wgpu::BindGroup,
    ) {
        let passes = clipmap::flood_pass_count();
        if work.rebuild_all {
            for level in 0..self.quality.levels as usize {
                let slice = RebuildSlice {
                    level,
                    begin: true,
                    passes: 0..passes,
                    commit: true,
                };
                self.record_slice(encoder, work, &slice, drivers);
            }
        } else if let Some(slice) = &work.slice {
            self.record_slice(encoder, work, slice, drivers);
        }
        let resolution = self.quality.resolution;
        self.record_light(encoder, work.relight, group1, resolution);
        self.record_write(encoder, work.relight, resolution);
    }

    /// Records one level's rebuild slice: the clear, drivers, seed and flood passes on begin, and
    /// the flood-to-seed copy and atlas write on commit.
    fn record_slice(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        work: &VolumeWork,
        slice: &RebuildSlice,
        drivers: &[Rc<dyn VoxelDriver>],
    ) {
        let resolution = self.quality.resolution;
        let level_cells = u64::from(resolution).pow(3);
        let level_offset = slice.level as u64 * level_cells;
        if slice.begin {
            encoder.clear_buffer(&self.voxels, level_offset * 12, Some(level_cells * 12));
            if let Some(level) = work
                .voxelize
                .iter()
                .find(|level| level.level as usize == slice.level)
            {
                for driver in drivers {
                    driver.record_level(encoder, level, &self.voxels);
                }
            }
            self.record_seed(encoder, slice.level, resolution);
        }
        self.record_flood(encoder, slice, resolution);
        if slice.commit {
            encoder.copy_buffer_to_buffer(
                &self.flood_a,
                0,
                &self.seeds,
                level_offset * 4,
                level_cells * 4,
            );
            self.record_write(encoder, slice.level, resolution);
        }
    }

    /// The seed dispatch of one level, at the flood parameter slot `18 + level`.
    fn record_seed(&self, encoder: &mut wgpu::CommandEncoder, level: usize, resolution: u32) {
        let groups = resolution.div_ceil(4);
        let slot = clipmap::flood_pass_count() + level;
        let offset = (slot as u64 * SLOT_BYTES) as u32;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("raa.gi.seed"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.seed_pipeline);
        pass.set_bind_group(2, &self.seed_bind, &[offset]);
        pass.dispatch_workgroups(groups, groups, groups);
    }

    /// The flood passes of `slice`, alternating A to B on even pass indices and B to A on odd.
    fn record_flood(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        slice: &RebuildSlice,
        resolution: u32,
    ) {
        let groups = resolution.div_ceil(4);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("raa.gi.flood"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.flood_pipeline);
        for index in slice.passes.clone() {
            let offset = (index as u64 * SLOT_BYTES) as u32;
            pass.set_bind_group(2, &self.flood_binds[index % 2], &[offset]);
            pass.dispatch_workgroups(groups, groups, groups);
        }
    }

    /// The atlas write of one level.
    fn record_write(&self, encoder: &mut wgpu::CommandEncoder, level: usize, resolution: u32) {
        let Some(bind) = self.write_binds.get(level) else {
            return;
        };
        let groups = resolution.div_ceil(4);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("raa.gi.write"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.write_pipeline);
        pass.set_bind_group(2, bind, &[]);
        pass.dispatch_workgroups(groups, groups, groups);
    }

    /// The relight dispatch of one level: group 1 is the GI compute group 1 and group 2 the
    /// level's light bind group.
    fn record_light(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        level: usize,
        group1: &wgpu::BindGroup,
        resolution: u32,
    ) {
        let Some(bind) = self.light_binds.get(level) else {
            return;
        };
        let groups = resolution.div_ceil(4);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("raa.gi.light"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.light_pipeline);
        pass.set_bind_group(1, group1, &[]);
        pass.set_bind_group(2, bind, &[]);
        pass.dispatch_workgroups(groups, groups, groups);
    }

    /// The `GiVolumeParamsGpu` bytes of the latest [`GiVolume::prepare`], passed to
    /// `LightingResources::set_gi`.
    pub(crate) fn params_bytes(&self) -> [u8; 96] {
        self.params
    }

    /// Whether a rebuild slice is pending, so stats report `pendingRebuild`.
    pub(crate) fn pending_slice(&self) -> bool {
        self.pending.is_some()
    }

    /// Whether every level has been committed at least once; the `VALID` flag condition.
    pub(crate) fn all_committed(&self) -> bool {
        self.origins.iter().all(|origin| origin.is_some())
    }

    /// The sampled D3 view of the whole atlas, created once by `GiVolume::new`. A clone of the
    /// stored view, so repeated calls compare equal and `LightingResources::set_gi` marks no
    /// rebuild.
    pub(crate) fn atlas_view(&self) -> wgpu::TextureView {
        self.atlas_view.clone()
    }

    /// The voxels buffer (`array<atomic<u32>>`, three words per voxel over all levels), which
    /// the drivers voxelize into.
    pub(crate) fn voxels(&self) -> &wgpu::Buffer {
        &self.voxels
    }

    /// Every GPU byte this volume owns: the six volume resources of `clipmap::volume_bytes`
    /// plus the per-level voxelize, light and write uniforms and the flood parameter slots.
    pub(crate) fn allocated_bytes(&self) -> u64 {
        let uniforms: u64 = self
            .voxelize_params
            .iter()
            .chain(&self.light_params)
            .chain(&self.write_params)
            .map(wgpu::Buffer::size)
            .sum();
        clipmap::volume_bytes(self.quality) + uniforms + self.flood_params.size()
    }
}
