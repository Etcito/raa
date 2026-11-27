//! The two feature passes of the lit renderer (S12), registered by [`LitRenderer::new`]
//! through [`install_passes`]:
//!
//! - `prepass` (order 30, required): the Opaque and Fur draws, single sample, with the
//!   `Prepass` shader variants into the pipeline targets `depth` (Depth32Float, Less,
//!   cleared to 1.0), `normal`, `material` and `motion` (Rgba16Float, cleared to zero).
//! - `forward` (order 60, required): the Opaque and Fur draws with the `Forward` variants
//!   into an MSAA HDR_FORMAT colour target and an MSAA Depth32Float target (Less, cleared to
//!   1.0) of `LitOptions::msaa` samples, created on resize, then the enabled forward
//!   contributors by order; the colour is cleared to the background (as HDR) and resolved
//!   into `hdr.dst()`, then the chain swaps. With 1 sample the pass renders straight into
//!   `hdr.dst()` (no MSAA colour, no resolve) with a single-sample depth of its own.
//!
//! Both passes bind group 0 (the renderer's frame bind group) and group 1 (the pipeline's
//! lighting resources) once, then draw each prepared draw with the variant of its material
//! function, culling back faces unless the material is DOUBLE_SIDED. The pipelines come from
//! the [`LitHandle`] cache and the two [`PipelineSpec`]s of a pass are built once, so a warm
//! frame allocates nothing.
//!
//! [`LitRenderer::new`]: crate::LitRenderer::new

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::{GpuContext, RaaError};
use raa_pipeline::{
    FeaturePass, HDR_FORMAT, ImageDesc, ImageDim, Pipeline, PrepareCtx, RecordCtx, Targets,
};
use raa_shader::{
    LIGHT_GPU_SIZE, MAX_LOCAL_LIGHTS, PassKind, VARIANT_FLAG_NO_AREA_LIGHTS, VariantKey,
};

use crate::handle::{DrawMemo, LitHandle, PipelineSpec};
use crate::material::{DrawClass, FLAG_DIFFUSE_TRANSMISSION, FLAG_DOUBLE_SIDED};
use crate::planar_reflection::{PlanarTargets, PlanarView};
use crate::renderer::{Contributors, pipeline_busy};

/// Name of the prepass in the pipeline (S12).
pub(crate) const PREPASS_NAME: &str = "prepass";
/// Default order of the prepass (S12).
pub(crate) const PREPASS_ORDER: u32 = 30;
/// Name of the forward pass in the pipeline (S12).
pub(crate) const FORWARD_NAME: &str = "forward";
/// Default order of the forward pass (S12).
pub(crate) const FORWARD_ORDER: u32 = 60;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// S12 `normal` target: view-space normal xyz, roughness.
const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// S12 `material` target: F0 rgb sRGB-encoded, metallic.
const MATERIAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// S12 `motion` target: signed UV displacement xy, history reject z, Rgba16Float.
const MOTION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The classes both passes draw, in order: opaque, then fur (drawn as opaque base).
const DRAWN_CLASSES: [DrawClass; 2] = [DrawClass::Opaque, DrawClass::Fur];

/// Material flags that select a pipeline (cull mode), kept in the variant key.
const VARIANT_FLAGS: u32 = FLAG_DOUBLE_SIDED;

/// Material flags that select a forward-only shader variant: diffuse transmission compiles the
/// back-lighting term of `opaque_lighting.wgsl` (tested there as `RAA_VARIANT_FLAGS`), so every
/// other draw keeps the uber-shader without it. The prepass never carries them.
const FORWARD_VARIANT_FLAGS: u32 = FLAG_DIFFUSE_TRANSMISSION;

/// Registers the `prepass` and `forward` feature passes of `lit` in `pipeline`; `forward`
/// records the enabled `contributors`. Refusals (nothing added): `pipeline.duplicate_pass`
/// when either name is taken, `pipeline.busy`.
pub(crate) fn install_passes(
    pipeline: &Rc<RefCell<Pipeline>>,
    lit: LitHandle,
    contributors: Contributors,
) -> Result<(), RaaError> {
    let mut p = pipeline.try_borrow_mut().map_err(|_| pipeline_busy())?;
    if let Some(taken) = p.passes().into_iter().find(|info| {
        info.name == PREPASS_NAME || info.name == FORWARD_NAME || info.name == "editor"
    }) {
        return Err(RaaError::FrameGraph {
            code: "pipeline.duplicate_pass",
            message: format!(
                "the pipeline already has an entry `{}`: one lit renderer per pipeline",
                taken.name
            ),
        });
    }
    let msaa = lit.msaa();
    p.add(Box::new(Prepass::new(lit.clone())))?;
    p.add(Box::new(ForwardPass::new(lit.clone(), contributors, msaa)))?;
    p.add(Box::new(crate::editor::EditorPass::new(lit)))?;
    Ok(())
}

/// Depth32Float, written, compare Less (S12).
fn depth_state() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(true),
        depth_compare: Some(wgpu::CompareFunction::Less),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

/// An opaque colour target of `format` (no blending).
fn color_target(format: wgpu::TextureFormat) -> Option<wgpu::ColorTargetState> {
    Some(wgpu::ColorTargetState {
        format,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    })
}

/// The two pipeline specs of a pass: back faces culled, and DOUBLE_SIDED (no culling).
struct Specs {
    culled: PipelineSpec,
    double_sided: PipelineSpec,
}

impl Specs {
    fn new(color_targets: Vec<Option<wgpu::ColorTargetState>>) -> Specs {
        let culled = PipelineSpec {
            color_targets,
            depth: Some(depth_state()),
            cull: Some(wgpu::Face::Back),
            alpha_to_coverage: false,
            extra_layout: None,
        };
        let double_sided = PipelineSpec {
            cull: None,
            ..culled.clone()
        };
        Specs {
            culled,
            double_sided,
        }
    }

    /// The spec for a draw whose material has `flags`.
    fn get(&self, flags: u32) -> &PipelineSpec {
        if flags & FLAG_DOUBLE_SIDED != 0 {
            &self.double_sided
        } else {
            &self.culled
        }
    }
}

#[derive(Clone, Copy)]
enum DrawSource<'a> {
    Camera(u32),
    Reflection(&'a PlanarView),
}

/// Draws every Opaque and Fur draw of `lit` into `pass` (groups 0 and 1 bound by the
/// caller): per draw the `kind` variant of its material function at `msaa` samples, its
/// sub-mesh ([`LitHandle::bind_mesh`]) and its instances
/// `first_instance..first_instance + instance_count`.
fn draw_lit(
    gpu: &GpuContext,
    pass: &mut wgpu::RenderPass<'_>,
    lit: &LitHandle,
    kind: PassKind,
    msaa: u32,
    specs: &Specs,
    source: DrawSource<'_>,
) -> Result<(), RaaError> {
    let (capture, extra_flags) = match source {
        DrawSource::Camera(flags) => (None, flags),
        DrawSource::Reflection(view) => (Some(view), 0),
    };
    // One memo per pass recording: repeated sub-meshes of a lit mesh, pipelines and the id
    // stream are bound once, and the hashed pipeline cache is consulted once per variant.
    let mut memo = DrawMemo::new();
    let edges_only = kind == PassKind::Forward && capture.is_none() && lit.surface_edges_only();
    for class in DRAWN_CLASSES {
        let camera_draws;
        let draws = if let Some(view) = capture {
            view.select.draws(class)
        } else {
            camera_draws = lit.camera_draws(class);
            &camera_draws
        };
        for d in draws {
            if d.instance_count == 0 {
                continue;
            }
            let indices = lit.bind_selected_mesh_memo(pass, d.mesh, capture, &mut memo)?;
            if indices.is_empty() {
                continue;
            }
            let gpu_flags = lit.material_record(d.material).map_or(0, |m| m.gpu_flags());
            let flags = gpu_flags & VARIANT_FLAGS;
            let forward_flags = if kind == PassKind::Forward {
                gpu_flags & FORWARD_VARIANT_FLAGS
            } else {
                0
            };
            let key = VariantKey {
                material_fn: d.material_fn,
                pass: kind,
                msaa,
                flags: flags | forward_flags | lit.material_variant_flags(d.material) | extra_flags,
            };
            let pipeline = lit.pipeline_memo(gpu, key, specs.get(flags), &mut memo)?;
            memo.set_pipeline(pass, &pipeline);
            let instances = d.first_instance..d.first_instance.saturating_add(d.instance_count);
            if kind == PassKind::Forward && capture.is_none() {
                lit.draw_surface_instances_with(pass, indices, instances, edges_only);
            } else {
                pass.draw_indexed(indices, 0, instances);
            }
        }
    }
    Ok(())
}

/// [`VARIANT_FLAG_NO_AREA_LIGHTS`] when no light record of this frame is an area light (kind 3
/// at byte 52 of `LightGpu`), else 0. Records past the active count are also scanned, so a
/// stale area record only keeps the full variant. Switching between the two variants compiles
/// the other pipeline once per material function; it is cached afterwards.
fn area_free_flag(lighting: &raa_pipeline::LightingResources) -> u32 {
    let has_area = (0..MAX_LOCAL_LIGHTS as u32).any(|i| {
        lighting
            .local_light_record(i)
            .is_some_and(|record: &[u8; LIGHT_GPU_SIZE]| {
                u32::from_le_bytes([record[52], record[53], record[54], record[55]]) == 3
            })
    });
    if has_area {
        0
    } else {
        VARIANT_FLAG_NO_AREA_LIGHTS
    }
}

/// The `prepass` feature pass; see the module docs.
struct Prepass {
    lit: LitHandle,
    specs: Specs,
}

impl Prepass {
    fn new(lit: LitHandle) -> Prepass {
        Prepass {
            lit,
            specs: Specs::new(vec![
                color_target(NORMAL_FORMAT),
                color_target(MATERIAL_FORMAT),
                color_target(MOTION_FORMAT),
            ]),
        }
    }
}

/// A cleared, stored colour attachment of `view`.
fn cleared(
    view: &wgpu::TextureView,
    color: wgpu::Color,
) -> Option<wgpu::RenderPassColorAttachment<'_>> {
    Some(wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(color),
            store: wgpu::StoreOp::Store,
        },
    })
}

/// A depth attachment of `view` cleared to 1.0.
fn cleared_depth(
    view: &wgpu::TextureView,
    store: wgpu::StoreOp,
) -> wgpu::RenderPassDepthStencilAttachment<'_> {
    wgpu::RenderPassDepthStencilAttachment {
        view,
        depth_ops: Some(wgpu::Operations {
            load: wgpu::LoadOp::Clear(1.0),
            store,
        }),
        stencil_ops: None,
    }
}

impl FeaturePass for Prepass {
    fn name(&self) -> &str {
        PREPASS_NAME
    }

    fn order(&self) -> u32 {
        PREPASS_ORDER
    }

    fn required(&self) -> bool {
        true
    }

    /// Publishes the four G-buffer targets under their S55 ports (`depth`, `normal`,
    /// `material`, `motion`) at the frame's size; the prepass writes them in `record`.
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let targets = ctx.targets;
        let (width, height) = targets.size();
        let rows = [
            ("depth", targets.depth(), DEPTH_FORMAT),
            ("normal", targets.normal(), NORMAL_FORMAT),
            ("material", targets.material(), MATERIAL_FORMAT),
            ("motion", targets.motion(), MOTION_FORMAT),
        ];
        for (port, view, format) in rows {
            let desc = ImageDesc::new(ImageDim::D2, format, width, height, 1, 1);
            ctx.resources.publish(PREPASS_NAME, port, view, desc);
        }
        Ok(())
    }

    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let targets = ctx.targets;
        let frame = self.lit.frame_bind_group();
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("raa.lit.prepass"),
            color_attachments: &[
                cleared(targets.normal(), wgpu::Color::TRANSPARENT),
                cleared(targets.material(), wgpu::Color::TRANSPARENT),
                cleared(targets.motion(), wgpu::Color::TRANSPARENT),
            ],
            depth_stencil_attachment: Some(cleared_depth(targets.depth(), wgpu::StoreOp::Store)),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_bind_group(0, &frame, &[]);
        pass.set_bind_group(1, ctx.lighting.bind_group(), &[]);
        let extra_flags = if self.lit.temporal_consumer() {
            raa_shader::VARIANT_FLAG_MOTION_DEFORM
        } else {
            0
        };
        draw_lit(
            ctx.gpu,
            &mut pass,
            &self.lit,
            PassKind::Prepass,
            1,
            &self.specs,
            DrawSource::Camera(extra_flags),
        )
    }
}

/// The forward pass's own surface-sized attachments.
struct ForwardTargets {
    /// MSAA HDR_FORMAT colour, resolved into `hdr.dst()`; `None` with 1 sample.
    color: Option<wgpu::TextureView>,
    /// Depth32Float with the pass's sample count.
    depth: wgpu::TextureView,
}

/// The targets are destroyed explicitly when replaced (resize) or dropped: wgpu's WebGPU
/// backend does not destroy a dropped texture, so without this every resize kept the
/// previous MSAA colour and depth (about 240 MB at 3440 x 1440) until a JS garbage
/// collection. Work already submitted still completes (GPUTexture.destroy semantics).
impl Drop for ForwardTargets {
    fn drop(&mut self) {
        if let Some(color) = &self.color {
            color.texture().destroy();
        }
        self.depth.texture().destroy();
    }
}

/// A render attachment texture of `size` and `samples`, returned as its default view.
fn attachment(
    gpu: &GpuContext,
    label: &str,
    size: (u32, u32),
    format: wgpu::TextureFormat,
    samples: u32,
) -> wgpu::TextureView {
    gpu.device()
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size.0.max(1),
                height: size.1.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: samples,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

/// The `forward` feature pass; see the module docs.
struct ForwardPass {
    lit: LitHandle,
    contributors: Contributors,
    msaa: u32,
    specs: Specs,
    /// Created by `resize` (before the first prepare and on every size change).
    targets: Option<ForwardTargets>,
    reflection: Option<PlanarTargets>,
    combined_lighting: Option<wgpu::TextureView>,
}

impl ForwardPass {
    fn new(lit: LitHandle, contributors: Contributors, msaa: u32) -> ForwardPass {
        ForwardPass {
            lit,
            contributors,
            msaa,
            specs: Specs::new(vec![color_target(HDR_FORMAT)]),
            targets: None,
            reflection: None,
            combined_lighting: None,
        }
    }

    fn record_reflection(
        &mut self,
        ctx: &mut RecordCtx<'_>,
        view: &PlanarView,
    ) -> Result<(), RaaError> {
        if self
            .reflection
            .as_ref()
            .is_none_or(|targets| targets.size != ctx.size)
        {
            self.reflection = Some(PlanarTargets::new(ctx.gpu, ctx.size, self.msaa));
        }
        let targets = self.reflection.as_ref().expect("reflection targets");
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("raa.lit.planar.capture"),
            color_attachments: &[cleared(&targets.color, wgpu::Color::TRANSPARENT)],
            depth_stencil_attachment: Some(cleared_depth(&targets.depth, wgpu::StoreOp::Discard)),
            ..Default::default()
        });
        pass.set_bind_group(0, view.frame_group(), &[]);
        pass.set_bind_group(1, ctx.lighting.bind_group(), &[]);
        draw_lit(
            ctx.gpu,
            &mut pass,
            &self.lit,
            PassKind::Forward,
            1,
            &self.specs,
            DrawSource::Reflection(view),
        )
    }

    fn record_camera(
        &self,
        ctx: &mut RecordCtx<'_>,
        own: &ForwardTargets,
        destination: &wgpu::TextureView,
        frame: &wgpu::BindGroup,
        reflection: Option<&PlanarView>,
    ) -> Result<(), RaaError> {
        let clear = wgpu::LoadOp::Clear(background_clear(ctx.lighting.env().background));
        let color = match &own.color {
            Some(samples) => wgpu::RenderPassColorAttachment {
                view: samples,
                depth_slice: None,
                resolve_target: Some(destination),
                ops: wgpu::Operations {
                    load: clear,
                    store: wgpu::StoreOp::Discard,
                },
            },
            None => wgpu::RenderPassColorAttachment {
                view: destination,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: clear,
                    store: wgpu::StoreOp::Store,
                },
            },
        };
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("raa.lit.forward"),
            color_attachments: &[Some(color)],
            depth_stencil_attachment: Some(cleared_depth(&own.depth, wgpu::StoreOp::Discard)),
            ..Default::default()
        });
        pass.set_bind_group(0, frame, &[]);
        pass.set_bind_group(1, ctx.lighting.bind_group(), &[]);
        draw_lit(
            ctx.gpu,
            &mut pass,
            &self.lit,
            PassKind::Forward,
            self.msaa,
            &self.specs,
            DrawSource::Camera(area_free_flag(ctx.lighting)),
        )?;
        if let (Some(view), Some(targets)) = (reflection, self.reflection.as_ref()) {
            targets.composite(ctx.gpu, &mut pass, &self.lit, view);
            pass.set_bind_group(0, frame, &[]);
            pass.set_bind_group(1, ctx.lighting.bind_group(), &[]);
        }
        let slots = self
            .contributors
            .try_borrow()
            .map_err(|_| pipeline_busy())?;
        for slot in slots.iter().filter(|s| s.enabled.get()) {
            slot.contributor.record(&mut pass, &self.lit)?;
            // Later contributors see the full lighting group even after sky bound its own.
            pass.set_bind_group(1, ctx.lighting.bind_group(), &[]);
        }
        Ok(())
    }
}

/// The display-referred background rgb of `EnvGpu` as the HDR clear colour (alpha 1).
fn background_clear(background: [f32; 4]) -> wgpu::Color {
    wgpu::Color {
        r: f64::from(background[0]),
        g: f64::from(background[1]),
        b: f64::from(background[2]),
        a: 1.0,
    }
}

impl FeaturePass for ForwardPass {
    fn name(&self) -> &str {
        FORWARD_NAME
    }

    fn order(&self) -> u32 {
        FORWARD_ORDER
    }

    fn required(&self) -> bool {
        true
    }

    fn resize(&mut self, gpu: &GpuContext, targets: &Targets) {
        let size = targets.size();
        let color = (self.msaa > 1)
            .then(|| attachment(gpu, "raa.lit.forward.color", size, HDR_FORMAT, self.msaa));
        let depth = attachment(gpu, "raa.lit.forward.depth", size, DEPTH_FORMAT, self.msaa);
        self.targets = Some(ForwardTargets { color, depth });
        if let Some(combined) = self.combined_lighting.take() {
            combined.texture().destroy();
        }
    }

    /// Prepares the enabled contributors in order.
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let mut slots = self
            .contributors
            .try_borrow_mut()
            .map_err(|_| pipeline_busy())?;
        for slot in slots.iter_mut().filter(|s| s.enabled.get()) {
            slot.contributor.prepare(&self.lit, ctx)?;
        }
        Ok(())
    }

    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let lit = self.lit.clone();
        let reflection = lit.planar_view();
        if let Some(view) = reflection.as_deref() {
            self.record_reflection(ctx, view)?;
        }
        let Some(own) = self.targets.as_ref() else {
            return Err(RaaError::Resource(format!(
                "{FORWARD_NAME}: record ran before resize created its targets"
            )));
        };
        let hdr = ctx.targets.hdr();
        let (width, height) = ctx.targets.size();
        let desc = ImageDesc::new(ImageDim::D2, HDR_FORMAT, width, height, 1, 1);
        let frame = self.lit.frame_bind_group();
        if let Some(preview) = self.lit.lighting_view_group(ctx.gpu) {
            if self.combined_lighting.is_none() {
                let combined = ctx.gpu.device().create_texture(&wgpu::TextureDescriptor {
                    label: Some("raa.lit.forward.combined-lighting"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: HDR_FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                self.combined_lighting = Some(combined.create_view(&Default::default()));
            }
            let combined = self
                .combined_lighting
                .as_ref()
                .expect("combined lighting target");
            self.record_camera(ctx, own, combined, &frame, reflection.as_deref())?;
            ctx.resources
                .publish(FORWARD_NAME, "combined-lighting", combined, desc);
            self.record_camera(ctx, own, hdr.dst(), &preview, None)?;
        } else {
            if let Some(combined) = self.combined_lighting.take() {
                combined.texture().destroy();
            }
            self.record_camera(ctx, own, hdr.dst(), &frame, reflection.as_deref())?;
        }
        hdr.swap();
        ctx.resources
            .publish(FORWARD_NAME, "color", hdr.src(), desc);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_cull_back_faces_unless_double_sided() {
        let specs = Specs::new(vec![color_target(HDR_FORMAT)]);
        assert_eq!(specs.get(0).cull, Some(wgpu::Face::Back));
        assert_eq!(
            specs.get(crate::FLAG_TRANSPARENT).cull,
            Some(wgpu::Face::Back)
        );
        assert_eq!(specs.get(FLAG_DOUBLE_SIDED).cull, None);
        assert_eq!(specs.get(FLAG_DOUBLE_SIDED | crate::FLAG_FUR).cull, None);
        for spec in [&specs.culled, &specs.double_sided] {
            let depth = spec.depth.as_ref().expect("depth state");
            assert_eq!(depth.format, wgpu::TextureFormat::Depth32Float);
            assert_eq!(depth.depth_compare, Some(wgpu::CompareFunction::Less));
            assert_eq!(depth.depth_write_enabled, Some(true));
            assert!(!spec.alpha_to_coverage && spec.extra_layout.is_none());
        }
    }

    #[test]
    fn prepass_writes_the_s12_normal_material_and_motion_targets() {
        let pass = Specs::new(vec![
            color_target(NORMAL_FORMAT),
            color_target(MATERIAL_FORMAT),
            color_target(MOTION_FORMAT),
        ]);
        let formats: Vec<_> = pass
            .culled
            .color_targets
            .iter()
            .map(|t| t.as_ref().map(|t| t.format))
            .collect();
        assert_eq!(
            formats,
            [
                Some(wgpu::TextureFormat::Rgba16Float),
                Some(wgpu::TextureFormat::Rgba8Unorm),
                Some(wgpu::TextureFormat::Rgba16Float)
            ]
        );
    }

    #[test]
    fn pass_names_and_orders_follow_s12() {
        assert_eq!((PREPASS_NAME, PREPASS_ORDER), ("prepass", 30));
        assert_eq!((FORWARD_NAME, FORWARD_ORDER), ("forward", 60));
    }

    #[test]
    fn background_clears_as_hdr_with_opaque_alpha() {
        let c = background_clear([0.05, 0.06, 0.08, 0.0]);
        assert_eq!(
            [c.r as f32, c.g as f32, c.b as f32, c.a as f32],
            [0.05, 0.06, 0.08, 1.0]
        );
    }
}
