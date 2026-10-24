//! The `particles.opaque` forward contributor (S12, order 600): the confetti and snow
//! particles as lit quads inside the forward MSAA pass, after the opaque draws.
//!
//! # Shader
//!
//! [`opaque_module_wgsl`] is `lit_module` (PRELUDE + LIB + the slot winners) followed by
//! `particles_opaque.wgsl`. The contributor composes it once, at construction, from the
//! composer of the `LitRenderer` it is installed into. The particle feature installs after
//! the sky, shadow, AO, material and contact features (m08..m11), so the Shadow, Ao,
//! Ambient and Direct winners are final by then; a slot provider added later is not seen
//! by these draws.
//!
//! # Pipelines and group 2
//!
//! One render pipeline per msaa count, built at the first prepare that needs it: groups
//! [frame layout, lighting layout, own group 2], no vertex buffers, HDR_FORMAT without
//! blending, Depth32Float Less with depth writes, no culling (confetti is double sided),
//! alpha-to-coverage when msaa > 1. Group 2 binds the particle states (binding 0) and the
//! emitter table (binding 1), both read-only storage; `prepare` rebuilds it whenever
//! `particles.sim` recreated its buffers (their generation changed).
//!
//! # Draws
//!
//! `prepare` lists every confetti and snow emitter (a list that keeps its capacity, so a
//! warm frame allocates nothing); `record` draws vertices `id*6 .. id*6+6` (the emitter id
//! is `vertex_index / 6` in the shader) over instances `slot_base .. slot_base +
//! max_particles` (instance = slot).

#![allow(dead_code)] // consumed by ParticleFeature (particles.opaque)

use std::num::NonZeroU64;
use std::ops::Range;

use raa_core::wgpu;
use raa_core::{GpuContext, RaaError};
use raa_lit::{ForwardContributor, LitHandle, LitRenderer};
use raa_pipeline::{HDR_FORMAT, PrepareCtx};
use raa_shader::Composer;

use crate::desc::ParticleKind;
use crate::gpu::EMITTER_GPU_SIZE;
use crate::shaders::lit_module;
use crate::sim::SharedSimBuffers;
use crate::spawn::PARTICLE_STRIDE;
use crate::state::SharedEmitters;

/// S12 name of the contributor.
pub(crate) const NAME: &str = "particles.opaque";
/// S12 order of the contributor inside the forward pass.
pub(crate) const ORDER: u32 = 600;
/// Vertices of one particle quad (two triangles).
const QUAD_VERTICES: u32 = 6;

/// The `particles.opaque` pass body appended to the lit module head.
pub(crate) const OPAQUE_WGSL: &str = include_str!("wgsl/particles_opaque.wgsl");

/// The `particles.opaque` draw module: PRELUDE + LIB + the winner of every slot of
/// `composer` + `particles_opaque.wgsl` (entries `vs_main`, `fs_main`). Refuses
/// `shader.slot_missing` when a slot has no provider. Not validated (the variant tests and
/// the GPU pipeline creation validate it).
pub fn opaque_module_wgsl(composer: &Composer) -> Result<String, RaaError> {
    lit_module(composer, OPAQUE_WGSL)
}

fn state_busy() -> RaaError {
    RaaError::FrameGraph {
        code: "pipeline.busy",
        message: format!("{NAME}: the particle state is borrowed (re-entrant call)"),
    }
}

fn storage_entry(binding: u32, min_size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(min_size),
        },
        count: None,
    }
}

/// The group-2 layout: particle states and emitter table, read-only storage.
fn group2_layout(gpu: &GpuContext) -> wgpu::BindGroupLayout {
    gpu.device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa-particles.opaque.layout"),
            entries: &[
                storage_entry(0, PARTICLE_STRIDE),
                storage_entry(1, EMITTER_GPU_SIZE as u64),
            ],
        })
}

/// The group-2 bind group over the current simulation buffers and their generation.
fn group2(
    gpu: &GpuContext,
    layout: &wgpu::BindGroupLayout,
    buffers: &SharedSimBuffers,
) -> Result<(u64, wgpu::BindGroup), RaaError> {
    let b = buffers.try_borrow().map_err(|_| state_busy())?;
    let group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("raa-particles.opaque"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: b.particles().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b.emitters().as_entire_binding(),
            },
        ],
    });
    Ok((b.generation(), group))
}

/// Whether `kind` is drawn by this contributor.
fn drawn_here(kind: ParticleKind) -> bool {
    matches!(kind, ParticleKind::Confetti | ParticleKind::Snow)
}

/// One emitter draw: vertices (emitter id) and instances (slot range).
struct EmitterDraw {
    vertices: Range<u32>,
    instances: Range<u32>,
}

/// Draws the confetti and snow particles; see the module docs.
pub(crate) struct OpaqueContributor {
    emitters: SharedEmitters,
    buffers: SharedSimBuffers,
    module: wgpu::ShaderModule,
    layout: wgpu::BindGroupLayout,
    /// The group-2 bind group and the buffer generation it was built for.
    group: (u64, wgpu::BindGroup),
    /// One pipeline per msaa count built so far.
    pipelines: Vec<(u32, wgpu::RenderPipeline)>,
    /// The msaa count of the current frame.
    msaa: u32,
    /// This frame's emitter draws.
    plan: Vec<EmitterDraw>,
}

impl OpaqueContributor {
    /// Composes the draw module with a snapshot of `lit`'s slot winners (see the module
    /// docs on install order) and creates the group-2 layout and bind group over `buffers`.
    /// Refusals: `shader.slot_missing`, `pipeline.busy`.
    pub(crate) fn new(
        gpu: &GpuContext,
        lit: &LitRenderer,
        emitters: SharedEmitters,
        buffers: SharedSimBuffers,
    ) -> Result<OpaqueContributor, RaaError> {
        let wgsl = opaque_module_wgsl(&lit.composer())?;
        let module = gpu
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("raa-particles.opaque"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
        let layout = group2_layout(gpu);
        let group = group2(gpu, &layout, &buffers)?;
        Ok(OpaqueContributor {
            emitters,
            buffers,
            module,
            layout,
            group,
            pipelines: Vec::with_capacity(2),
            msaa: lit.handle().msaa(),
            plan: Vec::new(),
        })
    }

    /// Builds the contributor and adds it to `lit` (order 600). Refusals (nothing added):
    /// those of [`OpaqueContributor::new`], `pipeline.duplicate_pass`, `pipeline.busy`.
    pub(crate) fn install(
        gpu: &GpuContext,
        lit: &mut LitRenderer,
        emitters: SharedEmitters,
        buffers: SharedSimBuffers,
    ) -> Result<(), RaaError> {
        let c = OpaqueContributor::new(gpu, lit, emitters, buffers)?;
        lit.add_contributor(Box::new(c))
    }

    /// The pipeline for `msaa`, built on first use.
    fn pipeline(&mut self, lit: &LitHandle, ctx: &PrepareCtx<'_>, msaa: u32) {
        if self.pipelines.iter().any(|(m, _)| *m == msaa) {
            return;
        }
        let pipeline = build_pipeline(lit, ctx, &self.module, &self.layout, msaa);
        self.pipelines.push((msaa, pipeline));
    }

    /// Fills `plan` with one draw per live confetti or snow emitter.
    fn plan_draws(&mut self) -> Result<(), RaaError> {
        let table = self.emitters.try_borrow().map_err(|_| state_busy())?;
        for (id, entry) in (0u32..).zip(table.entries()) {
            let Some(e) = entry else { continue };
            if !drawn_here(e.desc.kind) || e.desc.max_particles == 0 {
                continue;
            }
            let first = id.saturating_mul(QUAD_VERTICES);
            self.plan.push(EmitterDraw {
                vertices: first..first.saturating_add(QUAD_VERTICES),
                instances: e.slot_base..e.slot_base.saturating_add(e.desc.max_particles),
            });
        }
        Ok(())
    }
}

/// The fixed-function state of the particle quads: groups [frame, lighting, group 2], no
/// vertex buffers, one HDR_FORMAT target without blending, Depth32Float Less with depth
/// writes, no culling, alpha-to-coverage when `msaa > 1`.
fn build_pipeline(
    lit: &LitHandle,
    ctx: &PrepareCtx<'_>,
    module: &wgpu::ShaderModule,
    group2: &wgpu::BindGroupLayout,
    msaa: u32,
) -> wgpu::RenderPipeline {
    let device = ctx.gpu.device();
    let frame_layout = lit.frame_layout();
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("raa-particles.opaque.layout"),
        bind_group_layouts: &[
            Some(&frame_layout),
            Some(ctx.lighting.layout()),
            Some(group2),
        ],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("raa-particles.opaque"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: msaa,
            mask: !0,
            alpha_to_coverage_enabled: msaa > 1,
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

impl ForwardContributor for OpaqueContributor {
    fn name(&self) -> &str {
        NAME
    }

    fn order(&self) -> u32 {
        ORDER
    }

    /// Builds the pipeline of the current msaa count once, rebuilds group 2 when the
    /// simulation buffers were recreated, then lists the confetti and snow emitter draws.
    fn prepare(&mut self, lit: &LitHandle, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        self.plan.clear();
        let msaa = lit.msaa();
        self.msaa = msaa;
        self.pipeline(lit, ctx, msaa);
        let generation = self
            .buffers
            .try_borrow()
            .map_err(|_| state_busy())?
            .generation();
        if generation != self.group.0 {
            self.group = group2(ctx.gpu, &self.layout, &self.buffers)?;
        }
        let planned = self.plan_draws();
        if planned.is_err() {
            self.plan.clear();
        }
        planned
    }

    /// Draws every planned emitter's slot range; groups 0 and 1 are bound by the forward
    /// pass. Nothing without confetti or snow emitters.
    fn record(&self, pass: &mut wgpu::RenderPass<'_>, _lit: &LitHandle) -> Result<(), RaaError> {
        if self.plan.is_empty() {
            return Ok(());
        }
        let Some((_, pipeline)) = self.pipelines.iter().find(|(m, _)| *m == self.msaa) else {
            return Err(RaaError::Resource(format!(
                "{NAME}: record ran before prepare built its pipeline"
            )));
        };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(2, &self.group.1, &[]);
        for d in &self.plan {
            pass.draw(d.vertices.clone(), d.instances.clone());
        }
        Ok(())
    }
}

// raa-lit's default slot providers are crate-private (used by `LitRenderer::new`); the
// test compiles their source file (raa-lit is only read).
#[cfg(all(test, not(target_arch = "wasm32")))]
#[allow(dead_code, clippy::duplicate_mod)]
#[path = "../../raa-lit/src/slots.rs"]
mod lit_slots;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;
    use crate::desc::EmitterDesc;
    use crate::sim::SimPass;
    use crate::state::EmitterTable;
    use raa_core::testing::{block_on, skip, software_gpu};
    use raa_lit::LitOptions;
    use raa_pipeline::{FeaturePass, LightingResources, Pipeline, Targets};

    fn default_composer() -> Composer {
        let mut composer = Composer::new();
        for p in lit_slots::default_slot_providers() {
            if let Err(e) = composer.add_slot(p) {
                panic!("{e}");
            }
        }
        composer
    }

    #[test]
    fn name_and_order_follow_s12() {
        assert_eq!(NAME, "particles.opaque");
        assert_eq!(ORDER, 600);
        assert!(drawn_here(ParticleKind::Confetti));
        assert!(drawn_here(ParticleKind::Snow));
        assert!(!drawn_here(ParticleKind::Splash));
        assert!(!drawn_here(ParticleKind::Sparkles));
    }

    #[test]
    fn opaque_module_with_default_slots_validates() {
        let wgsl = opaque_module_wgsl(&default_composer()).unwrap_or_else(|e| panic!("{e}"));
        assert!(wgsl.contains("fn vs_main("));
        assert!(wgsl.contains("fn fs_main("));
        if let Err(e) = Composer::validate(&wgsl) {
            panic!("{e}");
        }
    }

    #[test]
    fn opaque_module_with_layered_brdf_validates() {
        let mut composer = default_composer();
        for p in raa_brdf::providers() {
            if let Err(e) = composer.add_slot(p) {
                panic!("{e}");
            }
        }
        let wgsl = opaque_module_wgsl(&composer).unwrap_or_else(|e| panic!("{e}"));
        if let Err(e) = Composer::validate(&wgsl) {
            panic!("{e}");
        }
    }

    fn texture(gpu: &GpuContext, format: wgpu::TextureFormat) -> wgpu::TextureView {
        gpu.device()
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("test.opaque"),
                size: wgpu::Extent3d {
                    width: 8,
                    height: 8,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 4,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    #[test]
    fn confetti_and_snow_draw_in_an_msaa_pass_without_validation_errors() {
        let test = "confetti_and_snow_draw_in_an_msaa_pass_without_validation_errors";
        let Some(gpu) = software_gpu() else {
            skip(
                test,
                "software_gpu returned None; see the software_gpu line above",
            );
            return;
        };
        let scope = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
        let pipeline = Rc::new(RefCell::new(Pipeline::new(
            &gpu,
            wgpu::TextureFormat::Rgba8Unorm,
            false,
        )));
        let lit = LitRenderer::new(&gpu, &pipeline, LitOptions::default())
            .unwrap_or_else(|e| panic!("{e}"));
        let emitters = EmitterTable::shared();
        for kind in [
            ParticleKind::Splash,
            ParticleKind::Confetti,
            ParticleKind::Snow,
        ] {
            emitters
                .borrow_mut()
                .add(EmitterDesc::for_kind(kind))
                .unwrap_or_else(|e| panic!("{e}"));
        }
        let mut sim = SimPass::new(&gpu, Rc::clone(&emitters)).unwrap_or_else(|e| panic!("{e}"));
        let mut c = OpaqueContributor::new(&gpu, &lit, Rc::clone(&emitters), sim.buffers())
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(c.name(), "particles.opaque");
        assert_eq!(c.order(), 600);

        let handle = lit.handle();
        let targets = Targets::new(&gpu, (8, 8));
        let mut lighting = LightingResources::new(&gpu);
        let mut ctx = PrepareCtx {
            gpu: &gpu,
            size: (8, 8),
            frame: 0,
            time_s: 1.0,
            dt_s: 0.0,
            targets: &targets,
            lighting: &mut lighting,
        };
        sim.prepare(&mut ctx).unwrap_or_else(|e| panic!("{e}"));
        c.prepare(&handle, &mut ctx)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(c.plan.len(), 2, "confetti and snow, not splash");
        assert_eq!(c.group.0, sim.buffers().borrow().generation());
        assert_eq!(c.pipelines.len(), 1);
        assert_eq!(c.plan[0].vertices, 6..12);
        let table = emitters.borrow();
        let confetti = table.emitter(1).unwrap_or_else(|| panic!("emitter 1"));
        assert_eq!(
            c.plan[0].instances,
            confetti.slot_base..confetti.slot_base + confetti.desc.max_particles
        );
        drop(table);

        let color = texture(&gpu, HDR_FORMAT);
        let depth = texture(&gpu, wgpu::TextureFormat::Depth32Float);
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("test.opaque"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &handle.frame_bind_group(), &[]);
            pass.set_bind_group(1, lighting.bind_group(), &[]);
            c.record(&mut pass, &handle)
                .unwrap_or_else(|e| panic!("{e}"));
        }
        gpu.queue().submit([encoder.finish()]);
        if let Err(e) = gpu.device().poll(wgpu::PollType::wait_indefinitely()) {
            panic!("device poll: {e:?}");
        }
        let error = block_on(scope.pop());
        assert!(error.is_none(), "validation error: {error:?}");
        drop(c);
        drop(sim);
        drop(lit);
        std::mem::forget(gpu);
    }

    #[test]
    fn opaque_module_refuses_a_missing_slot() {
        let Err(e) = opaque_module_wgsl(&Composer::new()) else {
            panic!("an empty composer must be refused");
        };
        assert_eq!(e.code(), "shader.slot_missing");
    }
}
