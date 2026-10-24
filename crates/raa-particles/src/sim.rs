//! The `particles.sim` feature pass (S12, order 50, not required): the compute simulation
//! of every emitter of the shared [`EmitterTable`].
//!
//! The pass owns the `cs_sim` pipeline (explicit layout of the private group 0 of
//! `sim.wgsl`), the particle states, the `EmitterGpu` table, the slot -> emitter table, the
//! spawn commands (`MAX_STEPS_PER_FRAME * MAX_EMITTERS` pairs), the claim counters, the
//! step uniform slots and their bind group.
//!
//! # Capacity
//!
//! The capacity is the sum of the emitters' `max_particles` (at least 1; it also covers the
//! end of the slot table, which a freed range in front of a live one can push past the
//! sum). The buffers are recreated only when it grows; a new particle buffer is zeroed
//! (age 0, life 0: every slot dead) and the tables are uploaded again. Slots past the end of
//! the slot table are written as `SLOT_FREE`.
//!
//! # Frame
//!
//! `prepare` advances the shared clock to `ctx.time_s`, uploads the dirty emitter and slot
//! tables, then writes that frame's spawn commands and step uniforms from preallocated
//! byte vectors (a warm frame allocates nothing). `record` clears the claim counters and
//! runs one compute pass with one dispatch of `ceil(capacity / 64)` workgroups per step,
//! each at its dynamic uniform offset; it records nothing without steps or emitters.
//!
//! The draw passes bind the particle and emitter buffers through the same
//! [`SharedSimBuffers`] (see [`SimPass::buffers`]); its generation changes whenever the
//! particle buffer is recreated. Disabling the pass freezes the simulation (the clock does
//! not advance while disabled); it sets no group-1 resource.

#![allow(dead_code)] // consumed by ParticleFeature (particles.opaque, particles.blend)

use std::cell::RefCell;
use std::num::NonZeroU64;
use std::rc::Rc;

use raa_core::wgpu;
use raa_core::{GpuContext, RaaError};
use raa_pipeline::{FeaturePass, LightingResources, PrepareCtx, RecordCtx};

use crate::gpu::{
    EMITTER_GPU_SIZE, MAX_EMITTERS, MAX_STEPS_PER_FRAME, SPAWN_COMMAND_SIZE, STEP_GPU_SIZE, STEP_S,
    StepGpu, step_slot_stride, write_emitter_table, write_slot_table, write_spawn_table,
    write_step_table,
};
use crate::shaders::sim_module_wgsl;
use crate::spawn::PARTICLE_STRIDE;
use crate::state::{EmitterTable, SLOT_FREE, SharedEmitters};

/// S12 name of the pass.
pub(crate) const NAME: &str = "particles.sim";
/// S12 order of the pass (after the shadow and AO passes, before `forward`).
pub(crate) const ORDER: u32 = 50;
/// The pass can be disabled (S12).
pub(crate) const REQUIRED: bool = false;
/// Entry point of `sim.wgsl`.
const ENTRY: &str = "cs_sim";
/// `@workgroup_size` of `cs_sim`.
const WORKGROUP_SIZE: u32 = 64;
/// Bytes of the emitter table (`MAX_EMITTERS` records).
const EMITTERS_SIZE: u64 = (MAX_EMITTERS * EMITTER_GPU_SIZE) as u64;
/// Bytes of the spawn commands of one frame (`MAX_STEPS_PER_FRAME * MAX_EMITTERS` pairs).
const SPAWNS_SIZE: u64 = (MAX_STEPS_PER_FRAME * MAX_EMITTERS * SPAWN_COMMAND_SIZE) as u64;
/// Bytes of the claim counters (one `u32` per spawn command).
const CLAIMS_SIZE: u64 = (MAX_STEPS_PER_FRAME * MAX_EMITTERS * 4) as u64;
/// Bytes of one slot -> emitter entry.
const SLOT_ENTRY_SIZE: u64 = 4;

/// The simulation buffers the draw passes bind.
pub(crate) struct SimBuffers {
    particles: wgpu::Buffer,
    emitters: wgpu::Buffer,
    capacity: u32,
    generation: u64,
}

impl SimBuffers {
    /// The 48 B particle states, `capacity` records (STORAGE).
    pub(crate) fn particles(&self) -> &wgpu::Buffer {
        &self.particles
    }

    /// The `EmitterGpu` table, `MAX_EMITTERS` records of 256 B (STORAGE).
    pub(crate) fn emitters(&self) -> &wgpu::Buffer {
        &self.emitters
    }

    /// Particle slots in the particle buffer.
    pub(crate) fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Changes every time the particle buffer is recreated (draw bind groups go stale).
    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }
}

/// The simulation buffers shared by `particles.sim` and the draw passes.
pub(crate) type SharedSimBuffers = Rc<RefCell<SimBuffers>>;

/// The capacity the table needs: the sum of `max_particles`, at least the slot table length,
/// at least 1.
fn required_capacity(table: &EmitterTable) -> u32 {
    let sum = table
        .entries()
        .iter()
        .flatten()
        .fold(0u32, |s, e| s.saturating_add(e.desc.max_particles));
    let slots = u32::try_from(table.slot_emitters().len()).unwrap_or(u32::MAX);
    sum.max(slots).max(1)
}

fn buffer(gpu: &GpuContext, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn storage_entry(binding: u32, read_only: bool, min_size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(min_size),
        },
        count: None,
    }
}

/// The explicit layout of `sim.wgsl` group 0.
fn bind_group_layout(gpu: &GpuContext) -> wgpu::BindGroupLayout {
    gpu.device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa-particles.sim"),
            entries: &[
                storage_entry(0, false, PARTICLE_STRIDE),
                storage_entry(1, true, EMITTER_GPU_SIZE as u64),
                storage_entry(2, true, SLOT_ENTRY_SIZE),
                storage_entry(3, true, SPAWN_COMMAND_SIZE as u64),
                storage_entry(4, false, 4),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: NonZeroU64::new(STEP_GPU_SIZE as u64),
                    },
                    count: None,
                },
            ],
        })
}

/// The `cs_sim` pipeline on `layout`.
fn compute_pipeline(gpu: &GpuContext, layout: &wgpu::BindGroupLayout) -> wgpu::ComputePipeline {
    let device = gpu.device();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("raa-particles.sim"),
        source: wgpu::ShaderSource::Wgsl(sim_module_wgsl().into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("raa-particles.sim"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("raa-particles.sim"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(ENTRY),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn state_busy() -> RaaError {
    RaaError::FrameGraph {
        code: "pipeline.busy",
        message: "the particle emitter table is borrowed (re-entrant call while a frame executes)"
            .to_string(),
    }
}

/// The `particles.sim` compute pass.
pub(crate) struct SimPass {
    emitters: SharedEmitters,
    buffers: SharedSimBuffers,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    slot_emitter: wgpu::Buffer,
    spawns: wgpu::Buffer,
    claims: wgpu::Buffer,
    step_uniforms: wgpu::Buffer,
    /// Byte distance between two step uniform slots.
    step_stride: u64,
    bind_group: wgpu::BindGroup,
    enabled: bool,
    /// Steps prepared for this frame.
    steps: u32,
    /// Emitter records the steps loop over (0 without emitters).
    emitter_count: u32,
    emitter_bytes: Vec<u8>,
    slot_bytes: Vec<u8>,
    spawn_bytes: Vec<u8>,
    step_bytes: Vec<u8>,
    step_records: Vec<StepGpu>,
}

impl SimPass {
    /// Builds the pipeline and the buffers for the current capacity of `emitters`.
    pub(crate) fn new(gpu: &GpuContext, emitters: SharedEmitters) -> Result<SimPass, RaaError> {
        let capacity = required_capacity(&*emitters.try_borrow().map_err(|_| state_busy())?);
        let layout = bind_group_layout(gpu);
        let pipeline = compute_pipeline(gpu, &layout);
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let step_stride = step_slot_stride(gpu.limits().min_uniform_buffer_offset_alignment);
        let particles = particle_buffer(gpu, capacity);
        let emitter_table = buffer(gpu, "raa-particles.emitters", EMITTERS_SIZE, storage);
        let slot_emitter = slot_buffer(gpu, capacity);
        let spawns = buffer(gpu, "raa-particles.spawns", SPAWNS_SIZE, storage);
        let claims = buffer(gpu, "raa-particles.claims", CLAIMS_SIZE, storage);
        let step_uniforms = buffer(
            gpu,
            "raa-particles.steps",
            MAX_STEPS_PER_FRAME as u64 * step_stride,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let buffers = SimBuffers {
            particles,
            emitters: emitter_table,
            capacity,
            generation: 0,
        };
        let bind_group = bind_group(
            gpu,
            &layout,
            &buffers,
            [&slot_emitter, &spawns, &claims, &step_uniforms],
        );
        Ok(SimPass {
            emitters,
            buffers: Rc::new(RefCell::new(buffers)),
            layout,
            pipeline,
            slot_emitter,
            spawns,
            claims,
            step_uniforms,
            step_stride,
            bind_group,
            enabled: true,
            steps: 0,
            emitter_count: 0,
            emitter_bytes: Vec::with_capacity(EMITTERS_SIZE as usize),
            slot_bytes: Vec::with_capacity(capacity as usize * SLOT_ENTRY_SIZE as usize),
            spawn_bytes: Vec::with_capacity(SPAWNS_SIZE as usize),
            step_bytes: Vec::with_capacity(MAX_STEPS_PER_FRAME * step_stride as usize),
            step_records: Vec::with_capacity(MAX_STEPS_PER_FRAME),
        })
    }

    /// The particle and emitter buffers, shared with the draw passes.
    pub(crate) fn buffers(&self) -> SharedSimBuffers {
        Rc::clone(&self.buffers)
    }

    /// The particle state buffer in use.
    pub(crate) fn particle_buffer(&self) -> wgpu::Buffer {
        self.buffers.borrow().particles.clone()
    }

    /// The emitter table buffer.
    pub(crate) fn emitter_buffer(&self) -> wgpu::Buffer {
        self.buffers.borrow().emitters.clone()
    }

    /// Recreates the capacity-sized buffers and the bind group for `capacity` slots.
    fn grow(&mut self, gpu: &GpuContext, capacity: u32) -> Result<(), RaaError> {
        let mut buffers = self.buffers.try_borrow_mut().map_err(|_| state_busy())?;
        buffers.particles = particle_buffer(gpu, capacity);
        buffers.capacity = capacity;
        buffers.generation = buffers.generation.wrapping_add(1);
        self.slot_emitter = slot_buffer(gpu, capacity);
        self.slot_bytes.reserve(
            (capacity as usize * SLOT_ENTRY_SIZE as usize).saturating_sub(self.slot_bytes.len()),
        );
        self.bind_group = bind_group(
            gpu,
            &self.layout,
            &buffers,
            [
                &self.slot_emitter,
                &self.spawns,
                &self.claims,
                &self.step_uniforms,
            ],
        );
        Ok(())
    }
}

/// A zeroed particle buffer of `capacity` records (age 0, life 0: dead).
fn particle_buffer(gpu: &GpuContext, capacity: u32) -> wgpu::Buffer {
    buffer(
        gpu,
        "raa-particles.particles",
        u64::from(capacity) * PARTICLE_STRIDE,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    )
}

/// The slot -> emitter table of `capacity` entries.
fn slot_buffer(gpu: &GpuContext, capacity: u32) -> wgpu::Buffer {
    buffer(
        gpu,
        "raa-particles.slot_emitter",
        u64::from(capacity) * SLOT_ENTRY_SIZE,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    )
}

/// The group 0 of `sim.wgsl`; `rest` = slot_emitter, spawns, claims, step uniforms.
fn bind_group(
    gpu: &GpuContext,
    layout: &wgpu::BindGroupLayout,
    buffers: &SimBuffers,
    rest: [&wgpu::Buffer; 4],
) -> wgpu::BindGroup {
    let [slot_emitter, spawns, claims, steps] = rest;
    gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("raa-particles.sim"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffers.particles.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffers.emitters.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: slot_emitter.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: spawns.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: claims.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: steps,
                    offset: 0,
                    size: NonZeroU64::new(STEP_GPU_SIZE as u64),
                }),
            },
        ],
    })
}

impl FeaturePass for SimPass {
    fn name(&self) -> &str {
        NAME
    }

    fn order(&self) -> u32 {
        ORDER
    }

    fn required(&self) -> bool {
        REQUIRED
    }

    /// Disabling freezes the simulation (its clock stops). No group-1 resource is touched.
    fn set_enabled(&mut self, enabled: bool, _lighting: &mut LightingResources) {
        self.enabled = enabled;
        if !enabled {
            self.steps = 0;
        }
    }

    /// Advances the clock to `ctx.time_s`, grows the buffers when the capacity grew, uploads
    /// the dirty tables, then writes this frame's spawn commands and step uniforms.
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        self.steps = 0;
        self.emitter_count = 0;
        if !self.enabled {
            return Ok(());
        }
        let shared = Rc::clone(&self.emitters);
        let mut table = shared.try_borrow_mut().map_err(|_| state_busy())?;
        let steps = table.advance(ctx.time_s).min(MAX_STEPS_PER_FRAME);

        let capacity = required_capacity(&table);
        let grown = capacity
            > self
                .buffers
                .try_borrow()
                .map_err(|_| state_busy())?
                .capacity;
        if grown {
            self.grow(ctx.gpu, capacity)?;
        }
        let queue = ctx.gpu.queue();
        if table.take_gpu_dirty() || grown {
            let buffers = self.buffers.try_borrow().map_err(|_| state_busy())?;
            write_emitter_table(table.emitter_records(), &mut self.emitter_bytes);
            queue.write_buffer(&buffers.emitters, 0, &self.emitter_bytes);
            write_slot_table(table.slot_emitters(), &mut self.slot_bytes);
            let slot_end = buffers.capacity as usize * SLOT_ENTRY_SIZE as usize;
            let free = SLOT_FREE.to_le_bytes();
            while self.slot_bytes.len() < slot_end {
                self.slot_bytes.extend_from_slice(&free);
            }
            self.slot_bytes.truncate(slot_end);
            queue.write_buffer(&self.slot_emitter, 0, &self.slot_bytes);
        }

        let emitter_count = table.emitter_records().len().min(MAX_EMITTERS) as u32;
        if steps == 0 || emitter_count == 0 {
            return Ok(());
        }
        write_spawn_table(table.spawn_commands(), &mut self.spawn_bytes);
        queue.write_buffer(&self.spawns, 0, &self.spawn_bytes);
        let base = table.step_base_s();
        self.step_records.clear();
        for i in 0..steps {
            let sim_time = base + (i + 1) as f64 * f64::from(STEP_S);
            self.step_records.push(StepGpu::new(
                STEP_S,
                sim_time as f32,
                i as u32,
                emitter_count,
            ));
        }
        write_step_table(&self.step_records, self.step_stride, &mut self.step_bytes);
        queue.write_buffer(&self.step_uniforms, 0, &self.step_bytes);
        self.steps = steps as u32;
        self.emitter_count = emitter_count;
        Ok(())
    }

    /// Clears the claim counters, then one compute pass with one dispatch per step (each at
    /// its uniform slot). Nothing without steps or emitters.
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if !self.enabled || self.steps == 0 || self.emitter_count == 0 {
            return Ok(());
        }
        let capacity = self
            .buffers
            .try_borrow()
            .map_err(|_| state_busy())?
            .capacity;
        let groups = capacity.div_ceil(WORKGROUP_SIZE);
        let claims = u64::from(self.steps) * MAX_EMITTERS as u64 * 4;
        ctx.encoder
            .clear_buffer(&self.claims, 0, Some(claims.min(CLAIMS_SIZE)));
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("raa-particles.sim"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&self.pipeline);
        for step in 0..u64::from(self.steps) {
            let offset = (step * self.step_stride) as u32;
            pass.set_bind_group(0, &self.bind_group, &[offset]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desc::{EmitterDesc, ParticleKind};
    use raa_core::testing::{block_on, skip, software_gpu};
    use raa_pipeline::Targets;

    #[test]
    fn sim_pass_is_named_ordered_and_optional() {
        assert_eq!(NAME, "particles.sim");
        assert_eq!(ORDER, 50);
        const { assert!(!REQUIRED) };
    }

    #[test]
    fn capacity_is_the_sum_of_max_particles_at_least_one() {
        let mut table = EmitterTable::new();
        assert_eq!(required_capacity(&table), 1);
        let mut a = EmitterDesc::for_kind(ParticleKind::Confetti);
        a.max_particles = 100;
        let mut b = EmitterDesc::for_kind(ParticleKind::Snow);
        b.max_particles = 50;
        let id_a = table.add(a).unwrap_or_else(|e| panic!("{e}"));
        table.add(b).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(required_capacity(&table), 150);
        // A freed range in front of a live one: the slot table still ends at 150.
        table.remove(id_a).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(required_capacity(&table), 150);
    }

    /// Runs one prepare + record of `pass` at `time_s` into a fresh encoder and submits it.
    fn frame(gpu: &GpuContext, pass: &mut SimPass, lighting: &mut LightingResources, time_s: f32) {
        let targets = Targets::new(gpu, (8, 8));
        let mut prepare = PrepareCtx {
            gpu,
            size: (8, 8),
            frame: 0,
            time_s,
            dt_s: 0.0,
            targets: &targets,
            lighting,
        };
        pass.prepare(&mut prepare).unwrap_or_else(|e| panic!("{e}"));
        let backbuffer = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("test.backbuffer"),
            size: wgpu::Extent3d {
                width: 8,
                height: 8,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = backbuffer.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let mut record = RecordCtx {
            gpu,
            encoder: &mut encoder,
            size: (8, 8),
            frame: 0,
            time_s,
            targets: &targets,
            lighting,
            backbuffer: &view,
            backbuffer_format: wgpu::TextureFormat::Rgba8Unorm,
            surface_is_srgb: false,
        };
        pass.record(&mut record).unwrap_or_else(|e| panic!("{e}"));
        gpu.queue().submit([encoder.finish()]);
        if let Err(e) = gpu.device().poll(wgpu::PollType::wait_indefinitely()) {
            panic!("device poll: {e:?}");
        }
    }

    #[test]
    fn confetti_simulates_to_one_second_without_validation_errors() {
        let test = "confetti_simulates_to_one_second_without_validation_errors";
        let Some(gpu) = software_gpu() else {
            skip(
                test,
                "software_gpu returned None; see the software_gpu line above",
            );
            return;
        };
        let scope = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
        let emitters = EmitterTable::shared();
        let mut pass = SimPass::new(&gpu, Rc::clone(&emitters)).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(pass.name(), "particles.sim");
        assert_eq!(pass.order(), 50);
        assert!(!pass.required());
        let shared = pass.buffers();
        assert_eq!(shared.borrow().capacity(), 1);

        let desc = EmitterDesc::for_kind(ParticleKind::Confetti);
        let max = desc.max_particles;
        emitters
            .borrow_mut()
            .add(desc)
            .unwrap_or_else(|e| panic!("{e}"));
        let mut lighting = LightingResources::new(&gpu);
        frame(&gpu, &mut pass, &mut lighting, 1.0);
        assert_eq!(pass.steps, 60, "the first frame simulates from 0 to 1 s");
        assert_eq!(pass.emitter_count, 1);
        assert_eq!(shared.borrow().capacity(), max);
        assert_eq!(shared.borrow().generation(), 1);
        assert_eq!(
            pass.particle_buffer().size(),
            u64::from(max) * PARTICLE_STRIDE
        );
        assert_eq!(pass.emitter_buffer().size(), EMITTERS_SIZE);
        assert!(emitters.borrow().count() > 0);

        // The same time again: no step, nothing recorded, no regrowth.
        frame(&gpu, &mut pass, &mut lighting, 1.0);
        assert_eq!(pass.steps, 0);
        assert_eq!(shared.borrow().generation(), 1);

        // Disabled: the clock is frozen.
        pass.set_enabled(false, &mut lighting);
        frame(&gpu, &mut pass, &mut lighting, 2.0);
        assert!((emitters.borrow().sim_time_s() - 1.0).abs() < 1e-6);
        pass.set_enabled(true, &mut lighting);
        frame(&gpu, &mut pass, &mut lighting, 1.5);
        assert_eq!(pass.steps, 30);

        let error = block_on(scope.pop());
        assert!(error.is_none(), "validation error: {error:?}");
        drop(pass);
        std::mem::forget(gpu);
    }
}
