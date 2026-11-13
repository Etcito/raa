//! The `vg.gbuffer` feature pass of RAA's virtualized geometry (S12 order 32, S32): one
//! fullscreen `VgGbuffer` draw per material function merges the resolved surface into the S12
//! depth and G-buffer targets, so the forward pass, the shadows and post-processing see the
//! virtualized pixels like any other opaque surface.
//!
//! The pass owns its own [`VgShadeSlots`] (contract 'vg-shade-slots'), created lazily while the
//! path is active and dropped when the frame state has no resources, so no vg buffer exists
//! while the path is off (P3-Q8).

// The feature share that installs the pass lands after this card; until then the pass is
// reached only through its trait and this module's tests.
#![allow(dead_code)]

use raa_core::{RaaError, wgpu};
use raa_lit::{MaterialFnId, PipelineSpec};
use raa_pipeline::{FeaturePass, PrepareCtx, RecordCtx};
use raa_shader::{PassKind, VariantKey};

use crate::frame_passes::SharedVg;
use crate::resolve::{surface_bind_group, surface_layout};
use crate::shade::VgShadeSlots;

/// `vg.gbuffer` (S12 order 32): merges the resolved virtualized surface into the depth,
/// normal, material and motion targets with one fullscreen draw per material function.
pub(crate) struct VgGbufferPass {
    state: SharedVg,
    layout: Option<wgpu::BindGroupLayout>,
    pipelines: Vec<wgpu::RenderPipeline>,
    pipeline_fns: Vec<MaterialFnId>,
    entries: Vec<(MaterialFnId, u32)>,
    slots: Option<VgShadeSlots>,
    group: Option<wgpu::BindGroup>,
    group_generation: u64,
    surface: Option<wgpu::Texture>,
}

impl VgGbufferPass {
    pub(crate) const NAME: &'static str = "vg.gbuffer";
    pub(crate) const ORDER: u32 = 32;

    /// Handoff: `raa-vg: VgFeature::install and VgHandle (settings, instances, active state,
    /// prepare_frame)` takes `VgGbufferPass::new(state: SharedVg) -> VgGbufferPass` exported by
    /// crates/raa-vg/src/gbuffer.rs; it attaches by registering the struct as the order-32
    /// `FeaturePass` beside `VgCullPass`, `VgRasterPass` and `VgResolvePass`, disabled until
    /// `VgFrameState::active`. State/keys: the `SharedVg` (`Rc<RefCell<VgFrameState>>`) with
    /// `VgFrameState::material_fns` (one slot and pipeline per function) and
    /// `VgFrameState::settings.debug_view` (the slot's debug code); the pass owns its
    /// `VgShadeSlots` and its group-2 bind group.
    pub(crate) fn new(state: SharedVg) -> VgGbufferPass {
        VgGbufferPass {
            state,
            layout: None,
            pipelines: Vec::new(),
            pipeline_fns: Vec::new(),
            entries: Vec::new(),
            slots: None,
            group: None,
            group_generation: 0,
            surface: None,
        }
    }
}

impl FeaturePass for VgGbufferPass {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn order(&self) -> u32 {
        Self::ORDER
    }

    /// While the path is active with resources, writes the slots for `material_fns` (the debug
    /// view code comes from the settings) and fetches one `VgGbuffer` pipeline per function with
    /// the group-2 surface layout, reusing the vector and the bind group until the surface or
    /// the slots generation changes. Inactive or without resources it drops its slots and bind
    /// group, so no vg buffer remains while the path is off.
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let state = self.state.borrow();
        if !state.active || state.resources.is_none() {
            self.pipelines.clear();
            self.pipeline_fns.clear();
            self.entries.clear();
            self.slots = None;
            self.group = None;
            self.surface = None;
            return Ok(());
        }
        let debug = state.settings.debug_view.code();
        self.entries.clear();
        self.entries
            .extend(state.material_fns.iter().map(|&f| (f, debug)));
        let slots = self.slots.get_or_insert_with(|| VgShadeSlots::new(ctx.gpu));
        slots.write(ctx.gpu, &self.entries);

        if self.pipeline_fns.as_slice() != state.material_fns.as_slice() {
            let layout = self.layout.get_or_insert_with(|| surface_layout(ctx.gpu));
            let spec = PipelineSpec {
                color_targets: vec![
                    Some(color_target(wgpu::TextureFormat::Rgba16Float)),
                    Some(color_target(wgpu::TextureFormat::Rgba8Unorm)),
                    Some(color_target(wgpu::TextureFormat::Rg16Float)),
                ],
                depth: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                cull: None,
                alpha_to_coverage: false,
                extra_layout: Some(layout.clone()),
            };
            self.pipelines.clear();
            for &material_fn in &state.material_fns {
                let key = VariantKey {
                    material_fn,
                    pass: PassKind::VgGbuffer,
                    msaa: 1,
                    flags: 0,
                };
                self.pipelines
                    .push(state.lit.pipeline_for(ctx.gpu, key, &spec)?);
            }
            self.pipeline_fns.clear();
            self.pipeline_fns.extend_from_slice(&state.material_fns);
        }

        let Some(resources) = state.resources.as_ref() else {
            return Ok(());
        };
        let surface = resources.surface.ids().texture();
        let generation = self.slots.as_ref().map_or(0, VgShadeSlots::generation);
        let stale = self.group.is_none()
            || self.group_generation != generation
            || self.surface.as_ref() != Some(surface);
        if stale {
            let layout = self.layout.get_or_insert_with(|| surface_layout(ctx.gpu));
            let Some(slots) = self.slots.as_ref() else {
                return Ok(());
            };
            self.group = Some(surface_bind_group(
                ctx.gpu,
                layout,
                &resources.surface,
                slots.buffer(),
            ));
            self.group_generation = generation;
            self.surface = Some(surface.clone());
        }
        Ok(())
    }

    /// One render pass loading the normal, material and motion targets and the depth, binding
    /// the lit frame group (0), the lighting group (1) and the surface group (2), and drawing
    /// the fullscreen triangle of each function in ascending id order. After those draws it
    /// seeds `VgResources::history` from `ctx.targets.depth()` through
    /// `Hzb::record_from_depth` and sets `VgFrameState::occlusion_history`, so the next frame's
    /// first culling phase tests the merged depth. Records nothing while inactive or without
    /// resources.
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let frame = {
            let state = self.state.borrow();
            if !state.active || state.resources.is_none() {
                return Ok(());
            }
            state.lit.frame_bind_group()
        };
        let (Some(group), Some(slots)) = (self.group.as_ref(), self.slots.as_ref()) else {
            return Ok(());
        };
        if self.pipelines.is_empty() {
            return Ok(());
        }
        {
            let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("raa-vg.gbuffer"),
                color_attachments: &[
                    loaded(ctx.targets.normal()),
                    loaded(ctx.targets.material()),
                    loaded(ctx.targets.motion()),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: ctx.targets.depth(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &frame, &[]);
            pass.set_bind_group(1, ctx.lighting.bind_group(), &[]);
            for (i, pipeline) in self.pipelines.iter().enumerate() {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(2, group, &[slots.offset(i)]);
                pass.draw(0..3, 0..1);
            }
        }
        let mut state = self.state.borrow_mut();
        if let Some(resources) = state.resources.as_ref() {
            resources
                .history
                .record_from_depth(ctx.encoder, ctx.targets.depth());
        }
        state.occlusion_history = true;
        state.recorded.push(Self::NAME);
        Ok(())
    }
}

/// A colour target written whole, without blending.
fn color_target(format: wgpu::TextureFormat) -> wgpu::ColorTargetState {
    wgpu::ColorTargetState {
        format,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    }
}

/// A loaded, stored colour attachment of `view`.
fn loaded(view: &wgpu::TextureView) -> Option<wgpu::RenderPassColorAttachment<'_>> {
    Some(wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
        },
    })
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use raa_core::testing::{skip, software_gpu};
    use raa_lit::{LitOptions, LitRenderer};
    use raa_pipeline::Pipeline;

    use super::*;
    use crate::frame_passes::VgFrameState;

    /// The pass is `vg.gbuffer` at S12 order 32 and cannot be made required.
    #[test]
    fn pass_name_order_and_not_required() {
        let test = "pass_name_order_and_not_required";
        assert_eq!(VgGbufferPass::NAME, "vg.gbuffer");
        assert_eq!(VgGbufferPass::ORDER, 32);
        let Some(gpu) = software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let pipeline = Rc::new(RefCell::new(Pipeline::new(
            &gpu,
            wgpu::TextureFormat::Rgba8Unorm,
            false,
        )));
        let renderer = LitRenderer::new(&gpu, &pipeline, LitOptions::default()).expect("lit");
        let state: SharedVg = Rc::new(RefCell::new(VgFrameState::new(renderer.handle())));
        let pass = VgGbufferPass::new(state);
        assert!(!FeaturePass::required(&pass));
    }
}
