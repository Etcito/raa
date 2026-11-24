//! `SsrFeature::install` and the `SsrHandle`: the `ssr` render pass (S12, order 65).
//!
//! Each frame the pass builds the Hi-Z min pyramid of `depth` (`cs_hiz_copy` then one
//! `cs_hiz_reduce` per level), copies hdr.src() into mip 0 of its HDR mip chain and
//! downsamples the other mips, traces the reflections into the trace target, composites
//! into hdr.dst() (group 1 = the lighting bind group) and swaps.
//!
//! The pass and every handle clone share one state: the description and the camera given
//! by `set_camera`. Until a camera arrives the composite runs with intensity 0, so the HDR
//! chain advances unchanged. The pass sets no group-1 resource.

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::wgpu;
use raa_core::{GpuContext, RaaError};
use raa_math::Mat4;
use raa_pipeline::{
    FeaturePass, HDR_FORMAT, ImageDesc, ImageDim, Pipeline, PrepareCtx, RecordCtx, Targets,
};

use crate::desc::SsrDesc;
use crate::resources::{SsrPipelines, SsrTargets, SsrViewKey, TRACE_FORMAT};
use crate::shaders::SSR_PARAMS_SIZE;
use crate::tunables::{self, tunables};

/// S12 name of the pass.
const NAME: &str = "ssr";
/// S12 order of the pass (after `forward`, before `fog`).
const ORDER: u32 = 65;
pub(crate) const PARAMS_LEN: usize = SSR_PARAMS_SIZE as usize;

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

/// `pipeline.busy` for the shared SSR state (only reachable by a re-entrant call).
fn state_busy() -> RaaError {
    refusal(
        "pipeline.busy",
        "the ssr state is borrowed (re-entrant call while a frame executes)",
    )
}

/// Scene camera of the pass, with the inverses computed by `set_camera`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct SsrCamera {
    view: Mat4,
    proj: Mat4,
    inv_view: Mat4,
    inv_proj: Mat4,
}

impl SsrCamera {
    /// None when either matrix is singular.
    pub(crate) fn new(view: &Mat4, proj: &Mat4) -> Option<SsrCamera> {
        Some(SsrCamera {
            view: *view,
            proj: *proj,
            inv_view: view.inverse()?,
            inv_proj: proj.inverse()?,
        })
    }

    pub(crate) const IDENTITY: SsrCamera = SsrCamera {
        view: Mat4::IDENTITY,
        proj: Mat4::IDENTITY,
        inv_view: Mat4::IDENTITY,
        inv_proj: Mat4::IDENTITY,
    };
}

/// State shared by the pass and every clone of an `SsrHandle`.
struct SsrState {
    desc: SsrDesc,
    camera: Option<SsrCamera>,
    /// Current tunable values, in `tunables()` order; `generation` counts changes.
    tunables: [f64; 3],
    generation: u64,
}

impl SsrState {
    fn new() -> SsrState {
        SsrState {
            desc: SsrDesc::default(),
            camera: None,
            tunables: tunables::defaults(),
            generation: 0,
        }
    }
}

/// The trace module's pipeline constants for `values`.
pub(crate) fn constants(values: &[f64; 3]) -> [(&'static str, f64); 3] {
    std::array::from_fn(|i| (tunables::OVERRIDE_NAMES[i], values[i]))
}

type SharedSsr = Rc<RefCell<SsrState>>;

fn put_f32s(out: &mut [u8], at: usize, values: [f32; 4]) {
    for (i, v) in values.into_iter().enumerate() {
        out[at + 4 * i..at + 4 * i + 4].copy_from_slice(&v.to_le_bytes());
    }
}

/// Writes `SsrParams` (layout of `SSR_PARAMS_SIZE`): proj @0, inv_proj @64, view @128,
/// inv_view @192, viewport @256, params @272 (intensity, max_roughness, thickness_m, Hi-Z
/// mips), params2 @288 (HDR mips, 0, 0, 0). Without a camera: identity matrices and
/// intensity 0.
pub(crate) fn pack_params(
    out: &mut [u8; PARAMS_LEN],
    size: (u32, u32),
    mips: u32,
    desc: &SsrDesc,
    camera: Option<&SsrCamera>,
) {
    let (cam, intensity) = match camera {
        Some(c) => (*c, desc.intensity),
        None => (SsrCamera::IDENTITY, 0.0),
    };
    cam.proj.write_le_bytes(&mut out[0..64]);
    cam.inv_proj.write_le_bytes(&mut out[64..128]);
    cam.view.write_le_bytes(&mut out[128..192]);
    cam.inv_view.write_le_bytes(&mut out[192..256]);
    let (w, h) = (size.0.max(1) as f32, size.1.max(1) as f32);
    put_f32s(out, 256, [w, h, 1.0 / w, 1.0 / h]);
    let m = mips as f32;
    put_f32s(
        out,
        272,
        [intensity, desc.max_roughness, desc.thickness_m, m],
    );
    put_f32s(out, 288, [m, 0.0, 0.0, 0.0]);
}

/// One fullscreen draw into `target` (cleared) with group 0 and an optional group 1.
pub(crate) fn draw(
    encoder: &mut wgpu::CommandEncoder,
    target: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    group0: &wgpu::BindGroup,
    group1: Option<&wgpu::BindGroup>,
) {
    draw_targets(encoder, [target], pipeline, group0, group1);
}

pub(crate) fn draw_targets<const N: usize>(
    encoder: &mut wgpu::CommandEncoder,
    targets: [&wgpu::TextureView; N],
    pipeline: &wgpu::RenderPipeline,
    group0: &wgpu::BindGroup,
    group1: Option<&wgpu::BindGroup>,
) {
    let attachments = targets.map(|view| {
        Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })
    });
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("raa-ssr"),
        color_attachments: &attachments,
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, group0, &[]);
    if let Some(g) = group1 {
        pass.set_bind_group(1, g, &[]);
    }
    pass.draw(0..3, 0..1);
}

struct SsrPass {
    name: String,
    state: SharedSsr,
    pipelines: SsrPipelines,
    targets: Option<SsrTargets>,
    bytes: [u8; PARAMS_LEN],
    /// `SsrState::generation` the trace pipelines were built with.
    built_generation: u64,
}

impl FeaturePass for SsrPass {
    fn name(&self) -> &str {
        &self.name
    }

    fn order(&self) -> u32 {
        ORDER
    }

    /// Recreates the Hi-Z pyramid, the HDR mip chain, the trace target and their groups.
    fn resize(&mut self, gpu: &GpuContext, targets: &Targets) {
        self.targets = Some(SsrTargets::new(gpu, &self.pipelines, targets));
    }

    /// Writes `SsrParams` from the handle's description and camera (in place). The bound
    /// `depth`, `normal` and `material` inputs key the bind groups, which are rebuilt only
    /// when one of them differs from the views they were built from.
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let (desc, camera, values, generation) = {
            let st = self.state.try_borrow().map_err(|_| state_busy())?;
            (st.desc, st.camera, st.tunables, st.generation)
        };
        if generation != self.built_generation {
            self.pipelines
                .rebuild_trace(ctx.gpu.device(), &constants(&values));
            self.built_generation = generation;
        }
        let Some(t) = self.targets.as_mut() else {
            return Ok(());
        };
        let views = SsrViewKey {
            depth: ctx.targets.depth_input(ctx.resources, &self.name),
            normal: ctx.targets.normal_input(ctx.resources, &self.name),
            material: ctx.targets.material_input(ctx.resources, &self.name),
        };
        t.set_views(ctx.gpu.device(), &self.pipelines, views);
        pack_params(
            &mut self.bytes,
            t.size,
            t.mip_count(),
            &desc,
            camera.as_ref(),
        );
        ctx.gpu
            .queue()
            .write_buffer(&self.pipelines.params, 0, &self.bytes);
        let desc = ImageDesc::new(ImageDim::D2, TRACE_FORMAT, t.size.0, t.size.1, 1, 1);
        ctx.resources
            .publish(&self.name, "trace", &t.trace.view, desc);
        let (w, h) = t.size;
        let hiz = ImageDesc::new(
            ImageDim::D2,
            wgpu::TextureFormat::R32Float,
            w.max(1),
            h.max(1),
            1,
            t.mip_count(),
        );
        ctx.resources
            .publish(&self.name, "hiz", &t.hiz.full_view, hiz);
        Ok(())
    }

    /// Hi-Z copy and reduce chain, copy of hdr.src() into the HDR mip 0 and downsample,
    /// trace, composite into hdr.dst(), swap.
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let Some(t) = self.targets.as_ref() else {
            return Err(RaaError::Resource(
                "ssr: record ran before a resize created the targets".to_owned(),
            ));
        };
        let p = &self.pipelines;
        // All sampled HDR views belong to the private mip chain, populated from the
        // current source below; no bind group embeds an A/B or bound chain view.
        let hdr = ctx.targets.hdr();
        let encoder = &mut *ctx.encoder;
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("raa-ssr.hiz"),
                timestamp_writes: None,
            });
            for (k, (level, group)) in t.hiz.levels.iter().zip(&t.hiz.groups).enumerate() {
                pass.set_pipeline(if k == 0 { &p.hiz_copy } else { &p.hiz_reduce });
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(level.groups.0, level.groups.1, 1);
            }
        }
        encoder.copy_texture_to_texture(
            hdr.src_texture().as_image_copy(),
            t.hdr_mips.as_image_copy(),
            wgpu::Extent3d {
                width: t.size.0,
                height: t.size.1,
                depth_or_array_layers: 1,
            },
        );
        for (target, group) in t.hdr_views.iter().skip(1).zip(&t.down_groups) {
            draw(encoder, target, &p.downsample, group, None);
        }
        let lighting = ctx.lighting.opaque_bind_group();
        draw(
            encoder,
            &t.trace.view,
            &p.trace,
            &t.trace.group,
            Some(lighting),
        );
        draw(
            encoder,
            hdr.dst(),
            &p.composite,
            &t.composite_group,
            Some(lighting),
        );
        hdr.swap();
        let (w, h) = t.size;
        let desc = ImageDesc::new(ImageDim::D2, HDR_FORMAT, w, h, 1, 1);
        ctx.resources.publish(&self.name, "color", hdr.src(), desc);
        Ok(())
    }
}

/// Installs screen-space reflections on a pipeline.
pub struct SsrFeature;

impl SsrFeature {
    /// Builds the Hi-Z, downsample, trace and composite pipelines (with the pipeline's
    /// lighting layout at group 1) and adds the FeaturePass `ssr` (order 65) to `pipeline`
    /// with the SSR defaults and no camera.
    ///
    /// Refusals (nothing installed): `pipeline.busy` when the pipeline is borrowed,
    /// `pipeline.duplicate_pass` when `ssr` is already installed.
    pub fn install(
        gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
    ) -> Result<SsrHandle, RaaError> {
        Self::install_named(gpu, pipeline, NAME.to_owned())
    }

    /// Handoff: `SsrFeature::install_instance` is called by card 'Wiring 4/5: multi-instance post passes'; it adds `ssr#<instance>` through `Pipeline::add` and returns an independent `SsrHandle` whose clones control that pass's descriptor and tunables.
    pub fn install_instance(
        gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
        instance: &str,
    ) -> Result<SsrHandle, RaaError> {
        Self::install_named(gpu, pipeline, format!("{NAME}#{instance}"))
    }

    fn install_named(
        gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
        name: String,
    ) -> Result<SsrHandle, RaaError> {
        let mut p = pipeline.try_borrow_mut().map_err(|_| pipeline_busy())?;
        if p.passes().iter().any(|info| info.name == name) {
            return Err(refusal(
                "pipeline.duplicate_pass",
                &format!("the pass `{name}` is already installed"),
            ));
        }
        let state = Rc::new(RefCell::new(SsrState::new()));
        let defaults = constants(&tunables::defaults());
        let pipelines = SsrPipelines::new(gpu, p.lighting().opaque_layout(), &defaults);
        p.add(Box::new(SsrPass {
            name,
            state: state.clone(),
            pipelines,
            targets: None,
            bytes: [0; PARAMS_LEN],
            built_generation: 0,
        }))?;
        Ok(SsrHandle { state })
    }
}

/// Access to the installed SSR (clones share one state).
#[derive(Clone)]
pub struct SsrHandle {
    state: SharedSsr,
}

impl SsrHandle {
    pub(crate) fn camera(&self) -> Option<SsrCamera> {
        self.state.try_borrow().ok().and_then(|state| state.camera)
    }
    /// Replaces the description after `SsrDesc::validate` (`post.invalid`; the stored
    /// description is unchanged on refusal). `pipeline.busy` on a re-entrant call.
    pub fn set(&self, d: SsrDesc) -> Result<(), RaaError> {
        d.validate()?;
        self.state.try_borrow_mut().map_err(|_| state_busy())?.desc = d;
        Ok(())
    }

    /// The current description (SSR defaults until set).
    pub fn get(&self) -> SsrDesc {
        self.state
            .try_borrow()
            .map(|st| st.desc)
            .unwrap_or_default()
    }

    /// Sets one S56 tunable of [`tunables`] by key.
    ///
    /// Handoff: card 'Executor extension: apply Tunable bindings' calls
    /// `SsrHandle::set_tunable` through HandleSet.ssr (crates/raa-graph/src/exec.rs) with a
    /// catalog key and its value as f64. Today it validates with
    /// `raa_pipeline::validate_tunable`, refuses with `post.invalid` (the code of
    /// crates/raa-ssr/src/desc.rs) without changing anything, and otherwise stores the value
    /// (shared by every clone and the pass) so the trace pipelines are rebuilt at the pass's
    /// next prepare. `pipeline.busy` on a re-entrant call.
    pub fn set_tunable(&self, key: &str, value: f64) -> Result<(), RaaError> {
        let specs = tunables();
        let spec = raa_pipeline::validate_tunable(specs, key, value)
            .map_err(|m| refusal("post.invalid", &m))?;
        let index = specs.iter().position(|s| s.key == spec.key).unwrap_or(0);
        let mut st = self.state.try_borrow_mut().map_err(|_| state_busy())?;
        if st.tunables[index] != value {
            st.tunables[index] = value;
            st.generation += 1;
        }
        Ok(())
    }

    /// The current value of a tunable, `None` for an unknown key.
    pub fn tunable(&self, key: &str) -> Option<f64> {
        let index = tunables().iter().position(|s| s.key == key)?;
        self.state.try_borrow().ok().map(|st| st.tunables[index])
    }

    /// Sets the scene camera (world -> view, view -> clip) used from the next frame. A
    /// singular matrix keeps the previous camera.
    pub fn set_camera(&self, view: &Mat4, proj: &Mat4) {
        if let (Some(cam), Ok(mut st)) = (SsrCamera::new(view, proj), self.state.try_borrow_mut()) {
            st.camera = Some(cam);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_at(bytes: &[u8], at: usize) -> f32 {
        f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
    }

    fn mat_at(bytes: &[u8], at: usize) -> [f32; 16] {
        std::array::from_fn(|i| f32_at(bytes, at + 4 * i))
    }

    fn handle() -> SsrHandle {
        SsrHandle {
            state: Rc::new(RefCell::new(SsrState::new())),
        }
    }

    #[test]
    fn set_tunable_refuses_with_post_invalid_and_keeps_state() {
        let h = handle();
        for (key, value) in [
            ("nope", 1.0),
            ("maxSteps", f64::NAN),
            ("maxSteps", 32.5),
            ("edgeFade", f64::INFINITY),
        ] {
            let err = h.set_tunable(key, value).unwrap_err();
            assert_eq!(err.code(), "post.invalid", "{key} = {value}");
        }
        assert_eq!(h.state.borrow().generation, 0);
        assert_eq!(h.tunable("maxSteps"), Some(64.0));
        h.set_tunable("maxSteps", 64.0).unwrap();
        assert_eq!(h.state.borrow().generation, 0);
        h.clone().set_tunable("maxSteps", 32.0).unwrap();
        assert_eq!(h.tunable("maxSteps"), Some(32.0));
        assert_eq!(h.state.borrow().generation, 1);
    }

    #[test]
    fn pack_params_places_every_field_at_its_offset() {
        let desc = SsrDesc {
            intensity: 0.75,
            max_roughness: 0.4,
            thickness_m: 0.25,
        };
        let cam = SsrCamera::new(&Mat4::IDENTITY, &Mat4::IDENTITY).unwrap();
        let mut out = [0xffu8; PARAMS_LEN];
        pack_params(&mut out, (1280, 720), 11, &desc, Some(&cam));
        let identity = Mat4::IDENTITY.to_cols_array();
        for at in [0, 64, 128, 192] {
            assert_eq!(mat_at(&out, at), identity, "matrix at {at}");
        }
        assert_eq!(f32_at(&out, 256), 1280.0);
        assert_eq!(f32_at(&out, 260), 720.0);
        assert_eq!(f32_at(&out, 264), 1.0 / 1280.0);
        assert_eq!(f32_at(&out, 268), 1.0 / 720.0);
        assert_eq!(f32_at(&out, 272), 0.75);
        assert_eq!(f32_at(&out, 276), 0.4);
        assert_eq!(f32_at(&out, 280), 0.25);
        assert_eq!(f32_at(&out, 284), 11.0);
        assert_eq!(f32_at(&out, 288), 11.0);
        for at in [292, 296, 300] {
            assert_eq!(f32_at(&out, at), 0.0);
        }
    }

    #[test]
    fn pack_params_without_camera_composites_with_intensity_zero() {
        let mut out = [0u8; PARAMS_LEN];
        pack_params(&mut out, (64, 32), 7, &SsrDesc::default(), None);
        assert_eq!(f32_at(&out, 272), 0.0);
        assert_eq!(f32_at(&out, 276), SsrDesc::default().max_roughness);
        assert_eq!(mat_at(&out, 64), Mat4::IDENTITY.to_cols_array());
    }

    #[test]
    fn set_camera_computes_inverses_and_keeps_previous_on_singular() {
        let h = handle();
        let view = Mat4::from_translation(raa_math::Vec3::new(1.0, 2.0, -3.0));
        let proj = Mat4::perspective_rh_zo(1.0, 1.5, 0.1, 100.0);
        h.set_camera(&view, &proj);
        let cam = h.state.borrow().camera.unwrap();
        assert_eq!(Some(cam.inv_view), view.inverse());
        assert_eq!(Some(cam.inv_proj), proj.inverse());
        h.set_camera(&Mat4::from_scale(raa_math::Vec3::new(0.0, 1.0, 1.0)), &proj);
        assert_eq!(h.state.borrow().camera, Some(cam));
        h.set_camera(&view, &Mat4::from_scale(raa_math::Vec3::new(1.0, 0.0, 1.0)));
        assert_eq!(h.state.borrow().camera, Some(cam));
    }

    #[test]
    fn handle_set_refuses_invalid_and_keeps_desc() {
        let h = handle();
        let bad = SsrDesc {
            max_roughness: 1.5,
            ..SsrDesc::default()
        };
        let err = h.set(bad).unwrap_err();
        assert_eq!(err.code(), "post.invalid");
        assert_eq!(h.get(), SsrDesc::default());
        let good = SsrDesc {
            thickness_m: 0.5,
            ..SsrDesc::default()
        };
        h.set(good).unwrap();
        assert_eq!(h.clone().get(), good);
    }

    #[cfg(not(target_arch = "wasm32"))]
    mod gpu {
        use std::cell::RefCell;
        use std::rc::Rc;

        use raa_core::testing::{block_on, skip, software_gpu};
        use raa_core::{FrameGraph, OffscreenTarget, wgpu};
        use raa_math::{Mat4, Vec3};
        use raa_pipeline::{FeaturePass, Pipeline, RecordCtx};

        use super::super::*;

        const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
        const SIZE: u32 = 32;
        /// One Rgba16Float row of 32 px is exactly COPY_BYTES_PER_ROW_ALIGNMENT.
        const ROW: u32 = SIZE * 8;

        /// Clears hdr.dst() to a constant colour and depth to 1, then swaps (no geometry).
        struct Setup(f64);

        impl FeaturePass for Setup {
            fn name(&self) -> &str {
                "test.setup"
            }
            fn order(&self) -> u32 {
                30
            }
            fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
                let hdr = ctx.targets.hdr();
                let c = wgpu::Color {
                    r: self.0,
                    g: self.0,
                    b: self.0,
                    a: 1.0,
                };
                let depth = ctx
                    .targets
                    .depth_attachment_input(ctx.resources, self.name());
                let pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("test.setup"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: hdr.dst(),
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(c),
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
                drop(pass);
                hdr.swap();
                Ok(())
            }
        }

        /// Copies hdr.src() into a readback buffer after ssr.
        struct Probe(Rc<wgpu::Buffer>);

        impl FeaturePass for Probe {
            fn name(&self) -> &str {
                "test.probe"
            }
            fn order(&self) -> u32 {
                150
            }
            fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
                ctx.encoder.copy_texture_to_buffer(
                    ctx.targets.hdr().src_texture().as_image_copy(),
                    wgpu::TexelCopyBufferInfo {
                        buffer: &self.0,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(ROW),
                            rows_per_image: None,
                        },
                    },
                    wgpu::Extent3d {
                        width: SIZE,
                        height: SIZE,
                        depth_or_array_layers: 1,
                    },
                );
                Ok(())
            }
        }

        fn f16_to_f32(bits: u16) -> f32 {
            let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
            let exp = i32::from((bits >> 10) & 0x1f);
            let frac = f32::from(bits & 0x3ff);
            match exp {
                0 => sign * frac * 2f32.powi(-24),
                _ => sign * (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
            }
        }

        #[test]
        fn constant_hdr_without_geometry_is_unchanged() {
            let Some(gpu) = software_gpu() else {
                skip(
                    "constant_hdr_without_geometry_is_unchanged",
                    "software_gpu returned None; see the software_gpu line above",
                );
                return;
            };
            let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
            let target = OffscreenTarget::new(&gpu, SIZE, SIZE, FORMAT);
            let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, FORMAT, false)));
            let buffer = Rc::new(gpu.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("test.readback"),
                size: u64::from(ROW * SIZE),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            {
                let mut p = pipeline.borrow_mut();
                p.add(Box::new(Setup(2.0))).unwrap();
                p.add(Box::new(Probe(buffer.clone()))).unwrap();
            }
            let handle = SsrFeature::install(&gpu, &pipeline).unwrap();
            let err = SsrFeature::install(&gpu, &pipeline).err().unwrap();
            assert_eq!(err.code(), "pipeline.duplicate_pass");
            let busy = pipeline.borrow_mut();
            let err = SsrFeature::install(&gpu, &pipeline).err().unwrap();
            assert_eq!(err.code(), "pipeline.busy");
            drop(busy);
            let view = Mat4::look_at_rh(
                Vec3::new(0.0, 1.0, 4.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            handle.set_camera(&view, &Mat4::perspective_rh_zo(1.0, 1.0, 0.1, 100.0));
            let mut graph = FrameGraph::new(FORMAT);
            Pipeline::install(&pipeline, &mut graph);
            graph.compile().unwrap();
            let commands = graph.execute(&gpu, target.view(), (SIZE, SIZE)).unwrap();
            gpu.queue().submit([commands]);
            let slice = buffer.slice(..);
            slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
            gpu.device()
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            let error = block_on(errors.pop());
            assert!(error.is_none(), "validation error: {error:?}");
            let mapped = slice.get_mapped_range().unwrap();
            for (i, px) in mapped.chunks(2).enumerate() {
                let v = f16_to_f32(u16::from_le_bytes([px[0], px[1]]));
                let want = if i % 4 == 3 { 1.0 } else { 2.0 };
                assert!((v - want).abs() < 1e-2, "channel {i}: {v}, want {want}");
            }
        }
    }
}
