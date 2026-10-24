//! The `particles.blend` feature pass (S12, order 85, not required): splash droplets and
//! sparkles blended over the HDR colour after the transparent pass.
//!
//! # Shader
//!
//! [`blend_module_wgsl`] is `lit_module` (PRELUDE + LIB + the slot winners) followed by
//! `particles_blend.wgsl`. The pass composes it once, at construction, from the composer of
//! the `LitRenderer` it is installed into; as for `particles.opaque`, the particle feature
//! installs after the lighting features (m08..m11), so the Fog winner and the group-1
//! resources it reads are final by then.
//!
//! # Pipelines and group 2
//!
//! Two render pipelines on that module, groups [frame layout, lighting layout, own group
//! 2], no vertex buffers, one HDR_FORMAT target, single sample, no depth attachment, no
//! culling:
//! - splash (`vs_splash`, `fs_splash`): premultiplied, One / OneMinusSrcAlpha;
//! - sparkles (`vs_sparkles`, `fs_sparkles`): additive, One / One.
//!
//! Group 2 binds the particle states (0) and the emitter table (1), read-only storage, and
//! the single-sample prepass `depth` target (2, `texture_depth_2d`, textureLoad) for the
//! soft depth fade. It is rebuilt on resize (the depth view changed) and whenever
//! `particles.sim` recreated its buffers (their generation changed).
//!
//! # Frame
//!
//! `prepare` lists the splash and sparkles emitter draws (lists that keep their capacity).
//! `record`, only when such a draw exists: copies `hdr.src` into `hdr.dst`, one render pass
//! on `hdr.dst()` that loads it, draws the splash ranges then the sparkle ranges (vertices
//! `id*6 .. id*6+6`, instances `slot_base .. slot_base + max_particles`), then swaps the
//! HDR chain. Without such emitters it records nothing and leaves the chain untouched.

#![allow(dead_code)] // consumed by ParticleFeature (particles.blend)

use std::cell::RefCell;
use std::num::NonZeroU64;
use std::ops::Range;
use std::rc::Rc;

use raa_core::wgpu;
use raa_core::{GpuContext, RaaError};
use raa_lit::{LitHandle, LitRenderer};
use raa_pipeline::{
    FeaturePass, HDR_FORMAT, LightingResources, Pipeline, PrepareCtx, RecordCtx, Targets,
};
use raa_shader::Composer;

use crate::desc::ParticleKind;
use crate::gpu::EMITTER_GPU_SIZE;
use crate::shaders::lit_module;
use crate::sim::SharedSimBuffers;
use crate::spawn::PARTICLE_STRIDE;
use crate::state::SharedEmitters;

/// S12 name of the pass.
pub(crate) const NAME: &str = "particles.blend";
/// S12 order of the pass (after `transparent.oit`, before `dof` and `bloom`).
pub(crate) const ORDER: u32 = 85;
/// The pass can be disabled (S12).
pub(crate) const REQUIRED: bool = false;
/// Vertices of one particle quad (two triangles).
const QUAD_VERTICES: u32 = 6;

/// The `particles.blend` pass body appended to the lit module head.
pub(crate) const BLEND_WGSL: &str = include_str!("wgsl/particles_blend.wgsl");

/// The `particles.blend` draw module: PRELUDE + LIB + the winner of every slot of
/// `composer` + `particles_blend.wgsl` (entries `vs_splash`, `fs_splash`, `vs_sparkles`,
/// `fs_sparkles`). Refuses `shader.slot_missing` when a slot has no provider. Not
/// validated (the variant tests and the GPU pipeline creation validate it).
pub fn blend_module_wgsl(composer: &Composer) -> Result<String, RaaError> {
    lit_module(composer, BLEND_WGSL)
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

/// The group-2 layout: particle states, emitter table, single-sample depth.
fn group2_layout(gpu: &GpuContext) -> wgpu::BindGroupLayout {
    gpu.device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa-particles.blend.layout"),
            entries: &[
                storage_entry(0, PARTICLE_STRIDE),
                storage_entry(1, EMITTER_GPU_SIZE as u64),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        })
}

/// The group-2 bind group over the current simulation buffers and `depth`, with the
/// buffer generation it was built for.
fn group2(
    gpu: &GpuContext,
    layout: &wgpu::BindGroupLayout,
    buffers: &SharedSimBuffers,
    depth: &wgpu::TextureView,
) -> Result<(u64, wgpu::BindGroup), RaaError> {
    let b = buffers.try_borrow().map_err(|_| state_busy())?;
    let group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("raa-particles.blend"),
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
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(depth),
            },
        ],
    });
    Ok((b.generation(), group))
}

/// One emitter draw: vertices (emitter id) and instances (slot range).
struct EmitterDraw {
    vertices: Range<u32>,
    instances: Range<u32>,
}

/// Blends the splash and sparkles particles; see the module docs.
pub(crate) struct BlendPass {
    emitters: SharedEmitters,
    buffers: SharedSimBuffers,
    lit: LitHandle,
    layout: wgpu::BindGroupLayout,
    /// The group-2 bind group and the buffer generation it was built for; None after a
    /// resize until the next prepare (or before the first one).
    group: Option<(u64, wgpu::BindGroup)>,
    splash: wgpu::RenderPipeline,
    sparkles: wgpu::RenderPipeline,
    /// This frame's splash draws.
    splash_draws: Vec<EmitterDraw>,
    /// This frame's sparkles draws.
    sparkle_draws: Vec<EmitterDraw>,
}

impl BlendPass {
    /// Composes the draw module with a snapshot of `lit`'s slot winners (see the module
    /// docs on install order) and builds both pipelines on `lighting`'s group-1 layout.
    /// Refusals: `shader.slot_missing`.
    pub(crate) fn new(
        gpu: &GpuContext,
        lit: &LitRenderer,
        lighting: &LightingResources,
        emitters: SharedEmitters,
        buffers: SharedSimBuffers,
    ) -> Result<BlendPass, RaaError> {
        let wgsl = blend_module_wgsl(&lit.composer())?;
        let device = gpu.device();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa-particles.blend"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });
        let layout = group2_layout(gpu);
        let handle = lit.handle();
        let frame_layout = handle.frame_layout();
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raa-particles.blend.layout"),
            bind_group_layouts: &[Some(&frame_layout), Some(lighting.layout()), Some(&layout)],
            immediate_size: 0,
        });
        let splash = build_pipeline(
            gpu,
            &pipeline_layout,
            &module,
            "splash",
            wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING,
        );
        let additive = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let sparkles = build_pipeline(
            gpu,
            &pipeline_layout,
            &module,
            "sparkles",
            wgpu::BlendState {
                color: additive,
                alpha: additive,
            },
        );
        Ok(BlendPass {
            emitters,
            buffers,
            lit: handle,
            layout,
            group: None,
            splash,
            sparkles,
            splash_draws: Vec::new(),
            sparkle_draws: Vec::new(),
        })
    }

    /// Builds the pass and adds it to `pipeline` (order 85). Refusals (nothing added):
    /// `pipeline.busy` when the pipeline is borrowed, `pipeline.duplicate_pass` when
    /// `particles.blend` is already installed, those of [`BlendPass::new`].
    pub(crate) fn install(
        gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
        lit: &LitRenderer,
        emitters: SharedEmitters,
        buffers: SharedSimBuffers,
    ) -> Result<(), RaaError> {
        let mut p = pipeline
            .try_borrow_mut()
            .map_err(|_| RaaError::FrameGraph {
                code: "pipeline.busy",
                message: format!("{NAME}: the pipeline is borrowed (re-entrant call)"),
            })?;
        if p.passes().iter().any(|info| info.name == NAME) {
            return Err(RaaError::FrameGraph {
                code: "pipeline.duplicate_pass",
                message: format!("the pass `{NAME}` is already installed"),
            });
        }
        let pass = BlendPass::new(gpu, lit, p.lighting(), emitters, buffers)?;
        p.add(Box::new(pass))
    }

    /// Rebuilds group 2 over `depth` unless it is current for the buffer generation.
    fn ensure_group(
        &mut self,
        gpu: &GpuContext,
        depth: &wgpu::TextureView,
    ) -> Result<(), RaaError> {
        let generation = self
            .buffers
            .try_borrow()
            .map_err(|_| state_busy())?
            .generation();
        if self.group.as_ref().is_some_and(|(g, _)| *g == generation) {
            return Ok(());
        }
        self.group = Some(group2(gpu, &self.layout, &self.buffers, depth)?);
        Ok(())
    }

    /// Fills the draw lists with one draw per live splash or sparkles emitter.
    fn plan_draws(&mut self) -> Result<(), RaaError> {
        let table = self.emitters.try_borrow().map_err(|_| state_busy())?;
        for (id, entry) in (0u32..).zip(table.entries()) {
            let Some(e) = entry else { continue };
            if e.desc.max_particles == 0 {
                continue;
            }
            let list = match e.desc.kind {
                ParticleKind::Splash => &mut self.splash_draws,
                ParticleKind::Sparkles => &mut self.sparkle_draws,
                ParticleKind::Snow | ParticleKind::Confetti => continue,
            };
            let first = id.saturating_mul(QUAD_VERTICES);
            list.push(EmitterDraw {
                vertices: first..first.saturating_add(QUAD_VERTICES),
                instances: e.slot_base..e.slot_base.saturating_add(e.desc.max_particles),
            });
        }
        Ok(())
    }

    fn has_draws(&self) -> bool {
        !self.splash_draws.is_empty() || !self.sparkle_draws.is_empty()
    }
}

/// One blended particle pipeline (`vs_<entry>`, `fs_<entry>`): no vertex buffers, one
/// HDR_FORMAT target with `blend`, single sample, no depth attachment, no culling.
fn build_pipeline(
    gpu: &GpuContext,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    entry: &str,
    blend: wgpu::BlendState,
) -> wgpu::RenderPipeline {
    let vs = format!("vs_{entry}");
    let fs = format!("fs_{entry}");
    let label = format!("raa-particles.blend.{entry}");
    gpu.device()
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some(&vs),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some(&fs),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: Some(blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
}

impl FeaturePass for BlendPass {
    fn name(&self) -> &str {
        NAME
    }

    fn order(&self) -> u32 {
        ORDER
    }

    fn required(&self) -> bool {
        REQUIRED
    }

    /// The depth view changed: group 2 is rebuilt over the new one.
    fn resize(&mut self, gpu: &GpuContext, targets: &Targets) {
        self.group = None;
        if self.ensure_group(gpu, targets.depth()).is_err() {
            // the buffers are borrowed: the next prepare retries
            self.group = None;
        }
    }

    /// Rebuilds group 2 when the simulation buffers were recreated (or after a resize),
    /// then lists the splash and sparkles emitter draws.
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        self.splash_draws.clear();
        self.sparkle_draws.clear();
        self.ensure_group(ctx.gpu, ctx.targets.depth())?;
        let planned = self.plan_draws();
        if planned.is_err() {
            self.splash_draws.clear();
            self.sparkle_draws.clear();
        }
        planned
    }

    /// Copies hdr.src into hdr.dst, blends the splash then the sparkle draws over it and
    /// swaps. Nothing without splash or sparkles emitters.
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if !self.has_draws() {
            return Ok(());
        }
        let Some((_, group)) = self.group.as_ref() else {
            return Err(RaaError::Resource(format!(
                "{NAME}: record ran before prepare built its bind group"
            )));
        };
        let hdr = ctx.targets.hdr();
        let src = hdr.src_texture();
        ctx.encoder.copy_texture_to_texture(
            src.as_image_copy(),
            hdr.dst_texture().as_image_copy(),
            src.size(),
        );
        {
            let frame = self.lit.frame_bind_group();
            let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("raa-particles.blend"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: hdr.dst(),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &frame, &[]);
            pass.set_bind_group(1, ctx.lighting.bind_group(), &[]);
            pass.set_bind_group(2, group, &[]);
            if !self.splash_draws.is_empty() {
                pass.set_pipeline(&self.splash);
                for d in &self.splash_draws {
                    pass.draw(d.vertices.clone(), d.instances.clone());
                }
            }
            if !self.sparkle_draws.is_empty() {
                pass.set_pipeline(&self.sparkles);
                for d in &self.sparkle_draws {
                    pass.draw(d.vertices.clone(), d.instances.clone());
                }
            }
        }
        hdr.swap();
        Ok(())
    }
}

// raa-lit's default slot providers are crate-private (used by `LitRenderer::new`); the
// test compiles their source file (raa-lit is only read).
#[cfg(test)]
#[allow(dead_code, clippy::duplicate_mod)]
#[path = "../../raa-lit/src/slots.rs"]
mod lit_slots;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desc::EmitterDesc;
    use crate::sim::SimPass;
    use crate::state::EmitterTable;
    use raa_core::testing::{block_on, skip, software_gpu};
    use raa_lit::LitOptions;

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
    fn blend_pass_is_named_ordered_and_optional() {
        assert_eq!(NAME, "particles.blend");
        assert_eq!(ORDER, 85);
        const { assert!(!REQUIRED) };
    }

    #[test]
    fn blend_module_with_default_slots_validates() {
        let wgsl = blend_module_wgsl(&default_composer()).unwrap_or_else(|e| panic!("{e}"));
        for entry in ["vs_splash", "fs_splash", "vs_sparkles", "fs_sparkles"] {
            assert!(wgsl.contains(&format!("fn {entry}(")), "{entry}");
        }
        if let Err(e) = Composer::validate(&wgsl) {
            panic!("{e}");
        }
    }

    #[test]
    fn blend_module_refuses_a_missing_slot() {
        let Err(e) = blend_module_wgsl(&Composer::new()) else {
            panic!("an empty composer must be refused");
        };
        assert_eq!(e.code(), "shader.slot_missing");
    }

    /// One prepare (sim, then blend) and record (sim, then blend) at 1 s; returns whether
    /// the HDR chain was swapped.
    fn frame(
        gpu: &GpuContext,
        sim: &mut SimPass,
        blend: &mut BlendPass,
        targets: &Targets,
        lighting: &mut LightingResources,
    ) -> bool {
        targets.hdr().reset();
        let mut prepare = PrepareCtx {
            gpu,
            size: (8, 8),
            frame: 0,
            time_s: 1.0,
            dt_s: 0.0,
            targets,
            lighting,
        };
        sim.prepare(&mut prepare).unwrap_or_else(|e| panic!("{e}"));
        blend
            .prepare(&mut prepare)
            .unwrap_or_else(|e| panic!("{e}"));
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
        let before = targets.hdr().src_texture().clone();
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let mut record = RecordCtx {
            gpu,
            encoder: &mut encoder,
            size: (8, 8),
            frame: 0,
            time_s: 1.0,
            targets,
            lighting,
            backbuffer: &view,
            backbuffer_format: wgpu::TextureFormat::Rgba8Unorm,
            surface_is_srgb: false,
        };
        sim.record(&mut record).unwrap_or_else(|e| panic!("{e}"));
        blend.record(&mut record).unwrap_or_else(|e| panic!("{e}"));
        gpu.queue().submit([encoder.finish()]);
        if let Err(e) = gpu.device().poll(wgpu::PollType::wait_indefinitely()) {
            panic!("device poll: {e:?}");
        }
        *targets.hdr().src_texture() != before
    }

    #[test]
    fn splash_and_sparkles_blend_and_swap_without_validation_errors() {
        let test = "splash_and_sparkles_blend_and_swap_without_validation_errors";
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
        let snow = emitters
            .borrow_mut()
            .add(EmitterDesc::for_kind(ParticleKind::Snow))
            .unwrap_or_else(|e| panic!("{e}"));
        let mut sim = SimPass::new(&gpu, Rc::clone(&emitters)).unwrap_or_else(|e| panic!("{e}"));
        let mut lighting = LightingResources::new(&gpu);
        let mut blend = BlendPass::new(&gpu, &lit, &lighting, Rc::clone(&emitters), sim.buffers())
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(blend.name(), "particles.blend");
        assert_eq!(blend.order(), 85);
        assert!(!blend.required());
        let targets = Targets::new(&gpu, (8, 8));
        blend.resize(&gpu, &targets);
        assert!(blend.group.is_some());

        // Snow only: nothing recorded, the HDR chain untouched.
        let swapped = frame(&gpu, &mut sim, &mut blend, &targets, &mut lighting);
        assert!(!blend.has_draws());
        assert!(!swapped, "no splash or sparkles emitter: no swap");

        for kind in [ParticleKind::Splash, ParticleKind::Sparkles] {
            emitters
                .borrow_mut()
                .add(EmitterDesc::for_kind(kind))
                .unwrap_or_else(|e| panic!("{e}"));
        }
        let swapped = frame(&gpu, &mut sim, &mut blend, &targets, &mut lighting);
        assert_eq!(blend.splash_draws.len(), 1);
        assert_eq!(blend.sparkle_draws.len(), 1);
        assert_eq!(blend.splash_draws[0].vertices, 6..12);
        assert_eq!(blend.sparkle_draws[0].vertices, 12..18);
        let table = emitters.borrow();
        let splash = table.emitter(1).unwrap_or_else(|| panic!("emitter 1"));
        assert_eq!(
            blend.splash_draws[0].instances,
            splash.slot_base..splash.slot_base + splash.desc.max_particles
        );
        drop(table);
        let generation = sim.buffers().borrow().generation();
        assert_eq!(blend.group.as_ref().map(|(g, _)| *g), Some(generation));
        assert!(swapped, "the blended colour is the new hdr.src");

        // Without splash or sparkles again: nothing.
        emitters
            .borrow_mut()
            .remove(1)
            .unwrap_or_else(|e| panic!("{e}"));
        emitters
            .borrow_mut()
            .remove(2)
            .unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(snow, 0);
        assert!(!frame(&gpu, &mut sim, &mut blend, &targets, &mut lighting));

        let error = block_on(scope.pop());
        assert!(error.is_none(), "validation error: {error:?}");
        drop(blend);
        drop(sim);
        drop(lit);
        std::mem::forget(gpu);
    }

    #[test]
    fn install_refuses_a_duplicate_pass() {
        let test = "install_refuses_a_duplicate_pass";
        let Some(gpu) = software_gpu() else {
            skip(
                test,
                "software_gpu returned None; see the software_gpu line above",
            );
            return;
        };
        let pipeline = Rc::new(RefCell::new(Pipeline::new(
            &gpu,
            wgpu::TextureFormat::Rgba8Unorm,
            false,
        )));
        let lit = LitRenderer::new(&gpu, &pipeline, LitOptions::default())
            .unwrap_or_else(|e| panic!("{e}"));
        let emitters = EmitterTable::shared();
        let sim = SimPass::new(&gpu, Rc::clone(&emitters)).unwrap_or_else(|e| panic!("{e}"));
        BlendPass::install(&gpu, &pipeline, &lit, Rc::clone(&emitters), sim.buffers())
            .unwrap_or_else(|e| panic!("{e}"));
        let info = pipeline
            .borrow()
            .passes()
            .into_iter()
            .find(|p| p.name == NAME)
            .unwrap_or_else(|| panic!("{NAME} listed"));
        assert_eq!(info.order, ORDER);
        assert!(!info.required);
        let Err(e) = BlendPass::install(&gpu, &pipeline, &lit, emitters, sim.buffers()) else {
            panic!("a second install must be refused");
        };
        assert_eq!(e.code(), "pipeline.duplicate_pass");
        drop(sim);
        drop(lit);
        drop(pipeline);
        std::mem::forget(gpu);
    }
}
