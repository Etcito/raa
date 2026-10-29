//! `OitFeature::install` and the `OitHandle`: the `transparent.oit` render pass (S12, order
//! 80).
//!
//! Install builds the head clear compute and the resolve pipelines on the layouts of a 1x1
//! [`OitBuffers`]; each resize recreates the buffers for the surface size (keeping the
//! layouts, the factor in use and the overflow count). Each prepare polls the counter
//! readback and rebuilds the buffers when the pool factor setting changed or the overflow
//! rule asks for it (`needs_rebuild`); a new `max_fragments` only rewrites the resolve
//! params. Each frame with Transparent draws records (1) the head clear, (2) the
//! Transparent draws of raa-lit appending to the lists (no colour target, the prepass depth
//! read-only), (3) the counter copy of the readback and (4) the fullscreen resolve of hdr.src
//! into hdr.dst, then swaps. Without a Transparent draw it records nothing and leaves the
//! HDR chain untouched.
//!
//! The pass and every handle clone share one state (settings and stats). The pass sets no
//! group-1 resource.

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::wgpu;
use raa_core::{GpuContext, RaaError};
use raa_lit::{DrawClass, FLAG_DOUBLE_SIDED, LitHandle, LitRenderer, PipelineSpec};
use raa_pipeline::{FeaturePass, HDR_FORMAT, HdrChain, Pipeline, PrepareCtx, RecordCtx, Targets};
use raa_shader::{PassKind, VariantKey};

use crate::buffers::{OitBuffers, OitLayouts};
use crate::shaders::{CLEAR_ENTRY, CLEAR_WGSL, RESOLVE_FS_ENTRY, RESOLVE_VS_ENTRY, RESOLVE_WGSL};

/// S12 name of the pass.
const NAME: &str = "transparent.oit";
/// S12 order of the pass (after `opaque.copy`, before `particles.blend`).
const ORDER: u32 = 80;
/// Material flags that select a pipeline (cull mode), kept in the variant key (as forward).
const VARIANT_FLAGS: u32 = FLAG_DOUBLE_SIDED;
/// Accepted `max_fragments` (OIT value: 16 by default).
const MAX_FRAGMENTS_RANGE: std::ops::RangeInclusive<u32> = 4..=32;
/// Accepted `pool_factor` (OIT value: 8 x pixels by default).
const POOL_FACTOR_RANGE: std::ops::RangeInclusive<u32> = 2..=32;

/// Settings of the linked-list OIT.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct OitSettings {
    /// Nearest fragments the resolve keeps per pixel (4..=32, default 16).
    pub max_fragments: u32,
    /// Initial node pool in nodes per pixel (2..=32, default 8); doubled after an overflow
    /// frame, bounded by the adapter limit.
    pub pool_factor: u32,
}

impl Default for OitSettings {
    fn default() -> Self {
        OitSettings {
            max_fragments: 16,
            pool_factor: 8,
        }
    }
}

impl OitSettings {
    /// Refuses with `oit.invalid`, naming the field outside its range.
    fn validate(&self) -> Result<(), RaaError> {
        in_range("max_fragments", self.max_fragments, &MAX_FRAGMENTS_RANGE)?;
        in_range("pool_factor", self.pool_factor, &POOL_FACTOR_RANGE)
    }
}

fn in_range(field: &str, v: u32, range: &std::ops::RangeInclusive<u32>) -> Result<(), RaaError> {
    if range.contains(&v) {
        return Ok(());
    }
    Err(refusal(
        "oit.invalid",
        &format!(
            "OitSettings.{field} must be in {}..={} (got {v})",
            range.start(),
            range.end()
        ),
    ))
}

/// What the pass reports (information only).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct OitStats {
    /// Nodes in the pool in use (0 before the first resize).
    pub pool_nodes: u32,
    /// Frames whose fragment count exceeded the pool.
    pub overflow_frames: u32,
}

fn refusal(code: &'static str, message: &str) -> RaaError {
    RaaError::FrameGraph {
        code,
        message: message.to_string(),
    }
}

/// Same refusal as raa-lit's for a pipeline borrowed while a frame executes.
fn pipeline_busy() -> RaaError {
    refusal(
        "pipeline.busy",
        "the pipeline is borrowed (called while a frame executes)",
    )
}

/// `pipeline.busy` for the shared OIT state (only reachable by a re-entrant call).
fn state_busy() -> RaaError {
    refusal(
        "pipeline.busy",
        "the OIT state is borrowed (re-entrant call while a frame executes)",
    )
}

/// State shared by the pass and every clone of an `OitHandle`.
#[derive(Default)]
struct OitState {
    settings: OitSettings,
    stats: OitStats,
}

type SharedOit = Rc<RefCell<OitState>>;

/// Depth32Float, compare Less, no writes: tested against the prepass depth.
fn depth_state() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: Some(false),
        depth_compare: Some(wgpu::CompareFunction::Less),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

/// The pipelines and specs built on one set of [`OitLayouts`].
struct Pipelines {
    clear: wgpu::ComputePipeline,
    resolve: wgpu::RenderPipeline,
    /// Transparent draws, back faces culled.
    culled: PipelineSpec,
    /// Transparent draws of DOUBLE_SIDED materials (no culling).
    double_sided: PipelineSpec,
}

impl Pipelines {
    fn new(gpu: &GpuContext, layouts: &OitLayouts) -> Pipelines {
        let device = gpu.device();
        let clear_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa-oit.clear"),
            source: wgpu::ShaderSource::Wgsl(CLEAR_WGSL.into()),
        });
        let clear_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raa-oit.clear"),
            bind_group_layouts: &[Some(&layouts.clear)],
            immediate_size: 0,
        });
        let clear = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("raa-oit.clear"),
            layout: Some(&clear_layout),
            module: &clear_module,
            entry_point: Some(CLEAR_ENTRY),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let resolve_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa-oit.resolve"),
            source: wgpu::ShaderSource::Wgsl(RESOLVE_WGSL.into()),
        });
        let resolve_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raa-oit.resolve"),
            bind_group_layouts: &[Some(&layouts.resolve)],
            immediate_size: 0,
        });
        let resolve = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("raa-oit.resolve"),
            layout: Some(&resolve_layout),
            vertex: wgpu::VertexState {
                module: &resolve_module,
                entry_point: Some(RESOLVE_VS_ENTRY),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &resolve_module,
                entry_point: Some(RESOLVE_FS_ENTRY),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let culled = PipelineSpec {
            color_targets: Vec::new(),
            depth: Some(depth_state()),
            cull: Some(wgpu::Face::Back),
            alpha_to_coverage: false,
            extra_layout: Some(layouts.transparent.clone()),
        };
        let double_sided = PipelineSpec {
            cull: None,
            ..culled.clone()
        };
        Pipelines {
            clear,
            resolve,
            culled,
            double_sided,
        }
    }

    /// The spec for a draw whose material has `flags`.
    fn spec(&self, flags: u32) -> &PipelineSpec {
        if flags & FLAG_DOUBLE_SIDED != 0 {
            &self.double_sided
        } else {
            &self.culled
        }
    }
}

/// The resolve group reading one texture of the HDR chain.
struct ResolveGroup {
    texture: wgpu::Texture,
    group: wgpu::BindGroup,
}

/// Resolve groups of both textures of the HDR chain over `buffers`.
fn resolve_groups(gpu: &GpuContext, buffers: &OitBuffers, hdr: &HdrChain) -> [ResolveGroup; 2] {
    [
        (hdr.src_texture(), hdr.src()),
        (hdr.dst_texture(), hdr.dst()),
    ]
    .map(|(texture, view)| ResolveGroup {
        texture: texture.clone(),
        group: buffers.resolve_bind_group(gpu, view),
    })
}

/// The buffers in use and what is built on them.
struct Sized {
    buffers: OitBuffers,
    /// Built on `buffers.layouts()`.
    pipelines: Pipelines,
    /// None until a resize gave the HDR chain.
    resolve: Option<[ResolveGroup; 2]>,
}

/// The `transparent.oit` render pass.
struct OitPass {
    state: SharedOit,
    lit: LitHandle,
    /// Settings the buffers were last built or updated with.
    applied: OitSettings,
    /// Overflow frames counted by buffers that a new pool factor setting replaced.
    overflow_base: u32,
    sized: Sized,
}

impl OitPass {
    /// Builds 1x1 buffers with the default settings and the pipelines on their layouts.
    fn new(gpu: &GpuContext, state: SharedOit, lit: LitHandle) -> OitPass {
        let applied = OitSettings::default();
        let mut buffers = OitBuffers::create(gpu, 1, 1, applied.pool_factor);
        buffers.set_max_fragments(gpu, applied.max_fragments);
        let pipelines = Pipelines::new(gpu, buffers.layouts());
        OitPass {
            state,
            lit,
            applied,
            overflow_base: 0,
            sized: Sized {
                buffers,
                pipelines,
                resolve: None,
            },
        }
    }

    /// The handle's settings (the applied ones while the state is borrowed).
    fn settings(&self) -> OitSettings {
        self.state
            .try_borrow()
            .map(|st| st.settings)
            .unwrap_or(self.applied)
    }

    /// Recreates the buffers for the target size with `settings`: a new pool factor starts
    /// a new growth (new layouts, pipelines rebuilt), otherwise the layouts, the factor in
    /// use and the overflow count carry over.
    fn rebuild(&mut self, gpu: &GpuContext, targets: &Targets, settings: OitSettings) {
        let (w, h) = targets.size();
        let mut buffers = if settings.pool_factor == self.applied.pool_factor {
            self.sized.buffers.rebuild(gpu, w, h)
        } else {
            self.overflow_base = self.overflow_frames();
            let buffers = OitBuffers::create(gpu, w, h, settings.pool_factor);
            self.sized.pipelines = Pipelines::new(gpu, buffers.layouts());
            buffers
        };
        buffers.set_max_fragments(gpu, settings.max_fragments);
        self.sized.resolve = Some(resolve_groups(gpu, &buffers, targets.hdr()));
        self.sized.buffers = buffers;
        self.applied = settings;
    }

    fn overflow_frames(&self) -> u32 {
        self.overflow_base
            .saturating_add(self.sized.buffers.overflow_frames())
    }

    /// Writes the stats into the shared state (skipped while it is borrowed).
    fn publish_stats(&self) {
        if let Ok(mut st) = self.state.try_borrow_mut() {
            st.stats = OitStats {
                pool_nodes: self.sized.buffers.pool_nodes(),
                overflow_frames: self.overflow_frames(),
            };
        }
    }

    /// Records every Transparent draw into `pass` (groups 0-2 bound by the caller).
    fn draw_transparent(
        &self,
        gpu: &GpuContext,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> Result<(), RaaError> {
        let lit = &self.lit;
        let draws = lit.draws(DrawClass::Transparent);
        for d in draws.iter() {
            if d.instance_count == 0 {
                continue;
            }
            let count = lit.bind_mesh(pass, d.mesh)?;
            if count == 0 {
                continue;
            }
            let flags = lit.material_record(d.material).map_or(0, |m| m.flags) & VARIANT_FLAGS;
            let key = VariantKey {
                material_fn: d.material_fn,
                pass: PassKind::Transparent,
                msaa: 1,
                flags: flags | lit.material_variant_flags(d.material),
            };
            let pipeline = lit.pipeline_for(gpu, key, self.sized.pipelines.spec(flags))?;
            pass.set_pipeline(&pipeline);
            let instances = d.first_instance..d.first_instance.saturating_add(d.instance_count);
            lit.draw_surface_instances(pass, 0..count, instances);
        }
        Ok(())
    }
}

impl FeaturePass for OitPass {
    fn name(&self) -> &str {
        NAME
    }

    fn order(&self) -> u32 {
        ORDER
    }

    /// Recreates the buffers for the new size and the resolve groups of the HDR chain.
    fn resize(&mut self, gpu: &GpuContext, targets: &Targets) {
        let settings = self.settings();
        self.rebuild(gpu, targets, settings);
        self.publish_stats();
    }

    /// Reads the counter readback, then rebuilds the buffers when the pool factor setting
    /// changed or an overflow grew the factor; a new `max_fragments` alone rewrites the
    /// resolve params.
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let settings = self.state.try_borrow().map_err(|_| state_busy())?.settings;
        self.sized.buffers.poll();
        if settings.pool_factor != self.applied.pool_factor || self.sized.buffers.needs_rebuild() {
            self.rebuild(ctx.gpu, ctx.targets, settings);
        } else if settings.max_fragments != self.applied.max_fragments {
            self.sized
                .buffers
                .set_max_fragments(ctx.gpu, settings.max_fragments);
            self.applied = settings;
        }
        self.publish_stats();
        Ok(())
    }

    /// Clears the heads, appends the Transparent draws, copies the counter for the
    /// readback, resolves hdr.src() into hdr.dst() and swaps. Without a Transparent draw it
    /// records nothing.
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let any = self
            .lit
            .draws(DrawClass::Transparent)
            .iter()
            .any(|d| d.instance_count > 0);
        if !any {
            return Ok(());
        }
        let Some(resolve) = self.sized.resolve.as_ref() else {
            return Err(RaaError::Resource(format!(
                "{NAME}: record ran before a resize created the buffers"
            )));
        };
        let hdr = ctx.targets.hdr();
        let Some(resolve_group) = resolve.iter().find(|r| r.texture == *hdr.src_texture()) else {
            return Err(RaaError::Resource(format!(
                "{NAME}: the HDR source is not a texture the pass was resized for"
            )));
        };
        let buffers = &self.sized.buffers;
        let pipelines = &self.sized.pipelines;
        // (1) heads to END, counter to 0
        {
            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("raa-oit.clear"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(&pipelines.clear);
            pass.set_bind_group(0, buffers.clear_bind_group(), &[]);
            let (x, y) = buffers.clear_dispatch();
            pass.dispatch_workgroups(x, y, 1);
        }
        // (2) the Transparent draws append to the lists, depth-tested against the prepass
        // depth (read-only: loaded and kept)
        {
            let frame = self.lit.frame_bind_group();
            let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("raa-oit.transparent"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: ctx.targets.depth(),
                    depth_ops: None,
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &frame, &[]);
            pass.set_bind_group(1, ctx.lighting.bind_group(), &[]);
            pass.set_bind_group(2, buffers.transparent_bind_group(), &[]);
            self.draw_transparent(ctx.gpu, &mut pass)?;
        }
        // (3) counter readback
        buffers.copy_counter(ctx.encoder);
        // (4) resolve over hdr.src() into hdr.dst()
        {
            let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("raa-oit.resolve"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: hdr.dst(),
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipelines.resolve);
            pass.set_bind_group(0, &resolve_group.group, &[]);
            pass.draw(0..3, 0..1);
        }
        hdr.swap();
        Ok(())
    }
}

/// Installs the linked-list OIT on a pipeline and a lit renderer.
pub struct OitFeature;

impl OitFeature {
    /// Builds the head clear and resolve pipelines, adds the FeaturePass `transparent.oit`
    /// (order 80) to `pipeline` with the default settings and registers it as the handler
    /// of `DrawClass::Transparent` in `lit`.
    ///
    /// Refusals (nothing installed): `pipeline.busy` when the pipeline is borrowed,
    /// `pipeline.duplicate_pass` when `transparent.oit` is already installed.
    pub fn install(
        gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
        lit: &mut LitRenderer,
    ) -> Result<OitHandle, RaaError> {
        let mut p = pipeline.try_borrow_mut().map_err(|_| pipeline_busy())?;
        if p.passes().iter().any(|info| info.name == NAME) {
            return Err(refusal(
                "pipeline.duplicate_pass",
                "the pass `transparent.oit` is already installed",
            ));
        }
        let state = Rc::new(RefCell::new(OitState::default()));
        p.add(Box::new(OitPass::new(gpu, state.clone(), lit.handle())))?;
        lit.register_class_handler(DrawClass::Transparent, NAME);
        Ok(OitHandle { state })
    }
}

/// Access to the installed OIT (clones share one state).
#[derive(Clone)]
pub struct OitHandle {
    state: SharedOit,
}

impl OitHandle {
    /// Replaces the settings, applied at the next prepare. Refuses `oit.invalid` naming the
    /// field when `max_fragments` is outside 4..=32 or `pool_factor` outside 2..=32 (the
    /// stored settings are unchanged on refusal); `pipeline.busy` on a re-entrant call.
    pub fn set(&self, s: OitSettings) -> Result<(), RaaError> {
        s.validate()?;
        self.state
            .try_borrow_mut()
            .map_err(|_| state_busy())?
            .settings = s;
        Ok(())
    }

    /// The current settings (defaults until set).
    pub fn get(&self) -> OitSettings {
        self.state
            .try_borrow()
            .map(|st| st.settings)
            .unwrap_or_default()
    }

    /// Pool size in use and overflow frames counted so far.
    pub fn stats(&self) -> OitStats {
        self.state
            .try_borrow()
            .map(|st| st.stats)
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn handle() -> OitHandle {
        OitHandle {
            state: Rc::new(RefCell::new(OitState::default())),
        }
    }

    #[test]
    fn settings_default_to_the_oit_values() {
        let s = OitSettings::default();
        assert_eq!(s.max_fragments, 16);
        assert_eq!(s.pool_factor, 8);
        assert_eq!(s.max_fragments as usize, crate::DEFAULT_MAX_FRAGMENTS);
        assert_eq!(handle().get(), s);
        assert_eq!(handle().stats(), OitStats::default());
    }

    #[test]
    fn set_refuses_max_fragments_out_of_range() {
        for max_fragments in [0, 3, 33, u32::MAX] {
            let h = handle();
            let err = h
                .set(OitSettings {
                    max_fragments,
                    ..OitSettings::default()
                })
                .unwrap_err();
            assert_eq!(err.code(), "oit.invalid");
            let msg = err.to_string();
            assert!(msg.contains("max_fragments"), "{msg}");
            assert!(msg.contains("4..=32"), "{msg}");
            assert!(msg.contains(&max_fragments.to_string()), "{msg}");
        }
    }

    #[test]
    fn set_refuses_pool_factor_out_of_range() {
        for pool_factor in [0, 1, 33, u32::MAX] {
            let h = handle();
            let err = h
                .set(OitSettings {
                    pool_factor,
                    ..OitSettings::default()
                })
                .unwrap_err();
            assert_eq!(err.code(), "oit.invalid");
            let msg = err.to_string();
            assert!(msg.contains("pool_factor"), "{msg}");
            assert!(msg.contains("2..=32"), "{msg}");
            assert!(!msg.contains("max_fragments"), "{msg}");
        }
    }

    #[test]
    fn set_accepts_the_range_bounds() {
        let h = handle();
        for (max_fragments, pool_factor) in [(4, 2), (32, 32), (16, 8)] {
            let s = OitSettings {
                max_fragments,
                pool_factor,
            };
            h.set(s).unwrap();
            assert_eq!(h.get(), s);
        }
    }

    #[test]
    fn get_keeps_the_old_settings_after_a_refusal() {
        let h = handle();
        let good = OitSettings {
            max_fragments: 8,
            pool_factor: 4,
        };
        h.set(good).unwrap();
        let bad = OitSettings {
            max_fragments: 64,
            pool_factor: 4,
        };
        assert!(h.set(bad).is_err());
        assert_eq!(h.get(), good);
        let bad = OitSettings {
            max_fragments: 8,
            pool_factor: 1,
        };
        assert!(h.set(bad).is_err());
        assert_eq!(h.clone().get(), good);
    }

    #[test]
    fn set_refuses_while_the_state_is_borrowed() {
        let h = handle();
        let _guard = h.state.borrow();
        let err = h.set(OitSettings::default()).unwrap_err();
        assert_eq!(err.code(), "pipeline.busy");
    }

    #[test]
    fn pass_name_and_order_follow_s12() {
        assert_eq!(NAME, "transparent.oit");
        assert_eq!(ORDER, 80);
    }

    #[test]
    fn transparent_depth_is_tested_less_without_writes() {
        let depth = depth_state();
        assert_eq!(depth.depth_write_enabled, Some(false));
        assert_eq!(depth.depth_compare, Some(wgpu::CompareFunction::Less));
        assert_eq!(depth.format, wgpu::TextureFormat::Depth32Float);
    }
}
