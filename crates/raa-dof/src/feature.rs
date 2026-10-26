//! `DofFeature::install` and the `DofHandle`: the `dof` pass (S12, order 110).
//!
//! On resize the pass rebuilds its [`DofTargets`]. Each frame it packs `DofParams` from the
//! handle's description and camera, then records: autofocus (only when enabled), the
//! full-res CoC, 16x16 tile min/max, tile dilation, the half-res prefilter of both layers,
//! the near and far spiral gathers and the full-res composite into hdr.dst(), and swaps.
//! Until `DofHandle::set_camera` is called the CoC pass writes 0 everywhere (identity
//! inverse projection, aperture term 0), the gathered layers are cleared and the composite
//! copies hdr.src() into hdr.dst(), so the HDR chain stays valid.
//!
//! The pass and every handle clone share one state (description and camera). The pass sets
//! no group-1 resource, so disabling it leaves the HDR chain untouched.

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::wgpu;
use raa_core::{GpuContext, RaaError};
use raa_math::{Mat4, Vec4};
use raa_pipeline::{FeaturePass, Pipeline, PrepareCtx, RecordCtx, Targets};

use crate::desc::{DofDesc, REFERENCE_VIEWPORT_H};
use crate::resources::{DofPipelines, DofTargets};
use crate::shaders::{COC_WORKGROUP, PARAMS_LENS2_OFFSET, PARAMS_SIZE, aperture_term_px};

/// S12 name of the pass.
const NAME: &str = "dof";
/// S12 order of the pass (after `particles.blend`, before `bloom`).
const ORDER: u32 = 110;
/// Autofocus blend factor per frame after the first (snapping) frame.
const AUTOFOCUS_BLEND: f32 = 0.25;
const PARAMS_LEN: usize = PARAMS_SIZE as usize;

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

/// `pipeline.busy` for the shared dof state (only reachable by a re-entrant call).
fn state_busy() -> RaaError {
    refusal(
        "pipeline.busy",
        "the dof state is borrowed (re-entrant call while a frame executes)",
    )
}

/// The camera of the frame as the pass needs it.
#[derive(Copy, Clone, Debug, PartialEq)]
struct DofCamera {
    /// Kept with its inverse (only the inverse is packed today).
    #[allow(dead_code)]
    proj: Mat4,
    inv_proj: Mat4,
}

/// State shared by the pass and every clone of a `DofHandle`.
struct DofState {
    desc: DofDesc,
    /// None until the first `set_camera`.
    camera: Option<DofCamera>,
}

type SharedDof = Rc<RefCell<DofState>>;

fn write_vec4(out: &mut [u8], v: [f32; 4]) {
    for (chunk, x) in out.chunks_exact_mut(4).zip(v) {
        chunk.copy_from_slice(&x.to_le_bytes());
    }
}

/// Camera near distance in metres: the linear depth at clip depth 0 (0 if not finite).
fn near_of(inv_proj: &Mat4) -> f32 {
    let v = inv_proj.mul_vec4(Vec4 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    });
    let near = -v.z / v.w;
    if near.is_finite() && near > 0.0 {
        near
    } else {
        0.0
    }
}

/// Packs `DofParams` (layout in `shaders.rs`) for a `size` viewport. Without a camera the
/// inverse projection is the identity and the aperture term and CoC clamp are 0, so every
/// CoC is 0; autofocus is then off as well.
fn pack_params(
    out: &mut [u8; PARAMS_LEN],
    desc: &DofDesc,
    camera: Option<&DofCamera>,
    size: (u32, u32),
    blend: f32,
) {
    let (w, h) = (size.0.max(1) as f32, size.1.max(1) as f32);
    let (inv_proj, k, max_coc, near) = match camera {
        Some(c) => (
            c.inv_proj,
            aperture_term_px(desc, h),
            desc.max_coc_px * h / REFERENCE_VIEWPORT_H,
            near_of(&c.inv_proj),
        ),
        None => (Mat4::IDENTITY, 0.0, 0.0, 0.0),
    };
    inv_proj.write_le_bytes(&mut out[0..64]);
    write_vec4(&mut out[64..80], [desc.focus_distance_m, k, max_coc, near]);
    let lens2 = PARAMS_LENS2_OFFSET as usize;
    write_vec4(
        &mut out[lens2..lens2 + 16],
        [desc.focal_length_mm / 1000.0, 0.0, 0.0, 0.0],
    );
    write_vec4(&mut out[96..112], [w, h, 1.0 / w, 1.0 / h]);
    let autofocus = match (desc.autofocus, camera) {
        (Some([u, v]), Some(_)) => [u, v, 1.0, blend],
        _ => [0.0, 0.0, 0.0, blend],
    };
    write_vec4(&mut out[112..128], autofocus);
}

struct DofPass {
    state: SharedDof,
    pipelines: DofPipelines,
    targets: Option<DofTargets>,
    bytes: [u8; PARAMS_LEN],
    /// The next autofocus frame snaps (blend 1): set at creation, on resize and while
    /// autofocus is off.
    snap: bool,
    /// Autofocus runs this frame (enabled and a camera is set).
    autofocus: bool,
    /// A camera is set this frame.
    has_camera: bool,
}

/// A render pass of one draw of the fullscreen triangle into `targets` (cleared first).
fn draw(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    targets: &[&wgpu::TextureView],
    pipeline: &wgpu::RenderPipeline,
    group: &wgpu::BindGroup,
) {
    let attachments: Vec<_> = targets.iter().map(|v| Some(attachment(v))).collect();
    let mut pass = begin(encoder, label, &attachments);
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, group, &[]);
    pass.draw(0..3, 0..1);
}

fn attachment(view: &wgpu::TextureView) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment {
        view,
        resolve_target: None,
        depth_slice: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            store: wgpu::StoreOp::Store,
        },
    }
}

fn begin<'e>(
    encoder: &'e mut wgpu::CommandEncoder,
    label: &str,
    attachments: &[Option<wgpu::RenderPassColorAttachment<'_>>],
) -> wgpu::RenderPass<'e> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: attachments,
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

fn dispatch(
    pass: &mut wgpu::ComputePass<'_>,
    pipeline: &wgpu::ComputePipeline,
    group: &wgpu::BindGroup,
    groups: (u32, u32),
) {
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, group, &[]);
    pass.dispatch_workgroups(groups.0, groups.1, 1);
}

impl FeaturePass for DofPass {
    fn name(&self) -> &str {
        NAME
    }

    fn order(&self) -> u32 {
        ORDER
    }

    /// Rebuilds the surface-sized targets and bind groups; the next autofocus frame snaps.
    fn resize(&mut self, gpu: &GpuContext, targets: &Targets) {
        self.targets = Some(DofTargets::new(gpu, &self.pipelines, targets));
        self.snap = true;
    }

    /// Writes `DofParams` from the handle's description and camera (in place).
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let (desc, camera) = {
            let st = self.state.try_borrow().map_err(|_| state_busy())?;
            (st.desc, st.camera)
        };
        let Some(targets) = self.targets.as_ref() else {
            return Ok(());
        };
        self.has_camera = camera.is_some();
        self.autofocus = desc.autofocus.is_some() && self.has_camera;
        let blend = if self.autofocus {
            let b = if self.snap { 1.0 } else { AUTOFOCUS_BLEND };
            self.snap = false;
            b
        } else {
            self.snap = true;
            1.0
        };
        pack_params(
            &mut self.bytes,
            &desc,
            camera.as_ref(),
            targets.sizes.full,
            blend,
        );
        ctx.gpu
            .queue()
            .write_buffer(&self.pipelines.params, 0, &self.bytes);
        Ok(())
    }

    /// Autofocus (when enabled), CoC, tiles, dilation, prefilter, both gathers, composite
    /// into hdr.dst(), then swap. Without a camera only the CoC (all 0) and the composite
    /// run, over cleared layers.
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let Some(t) = self.targets.as_ref() else {
            return Err(RaaError::Resource(
                "dof: record ran before a resize created the targets".to_owned(),
            ));
        };
        let hdr = ctx.targets.hdr();
        let Some(groups) = t.hdr_groups(hdr.src_texture()) else {
            return Err(RaaError::Resource(
                "dof: the HDR source is not a texture the pass was resized for".to_owned(),
            ));
        };
        let p = &self.pipelines;
        let encoder = &mut *ctx.encoder;
        let full = t.sizes.full;
        let tiles = t.sizes.tiles;
        let per_group = |s: (u32, u32)| (s.0.div_ceil(COC_WORKGROUP), s.1.div_ceil(COC_WORKGROUP));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("raa-dof.coc"),
                timestamp_writes: None,
            });
            if self.autofocus {
                dispatch(&mut pass, &p.autofocus, &t.autofocus_group, (1, 1));
            }
            dispatch(&mut pass, &p.coc, &t.coc_group, per_group(full));
            if self.has_camera {
                dispatch(&mut pass, &p.tiles, &t.tiles_group, tiles);
                dispatch(&mut pass, &p.dilate, &t.dilate_group, per_group(tiles));
            }
        }
        if self.has_camera {
            draw(
                encoder,
                "raa-dof.prefilter",
                &[&t.near_prefilter_view, &t.far_prefilter_view],
                &p.prefilter,
                &groups.prefilter,
            );
            let near = [&t.near_gather_view];
            draw(
                encoder,
                "raa-dof.gather_near",
                &near,
                &p.gather_near,
                &t.gather_group,
            );
            let far = [&t.far_gather_view];
            draw(
                encoder,
                "raa-dof.gather_far",
                &far,
                &p.gather_far,
                &t.gather_group,
            );
        } else {
            let cleared = [
                Some(attachment(&t.near_gather_view)),
                Some(attachment(&t.far_gather_view)),
            ];
            drop(begin(encoder, "raa-dof.clear_layers", &cleared));
        }
        draw(
            encoder,
            "raa-dof.composite",
            &[hdr.dst()],
            &p.composite,
            &groups.composite,
        );
        hdr.swap();
        Ok(())
    }
}

/// Installs depth of field on a pipeline.
pub struct DofFeature;

impl DofFeature {
    /// Builds the pipelines, buffers and sampler of the pass and adds the FeaturePass `dof`
    /// (order 110) to `pipeline` with the DOF defaults and no camera.
    ///
    /// Refusals (nothing installed): `pipeline.busy` when the pipeline is borrowed,
    /// `pipeline.duplicate_pass` when `dof` is already installed.
    pub fn install(
        gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
    ) -> Result<DofHandle, RaaError> {
        let mut p = pipeline.try_borrow_mut().map_err(|_| pipeline_busy())?;
        if p.passes().iter().any(|info| info.name == NAME) {
            return Err(refusal(
                "pipeline.duplicate_pass",
                "the pass `dof` is already installed",
            ));
        }
        let state = Rc::new(RefCell::new(DofState {
            desc: DofDesc::default(),
            camera: None,
        }));
        p.add(Box::new(DofPass {
            state: state.clone(),
            pipelines: DofPipelines::new(gpu),
            targets: None,
            bytes: [0; PARAMS_LEN],
            snap: true,
            autofocus: false,
            has_camera: false,
        }))?;
        Ok(DofHandle { state })
    }
}

/// Access to the installed depth of field (clones share one state).
#[derive(Clone)]
pub struct DofHandle {
    state: SharedDof,
}

impl DofHandle {
    /// Replaces the description after `DofDesc::validate` (`post.invalid`; the stored
    /// description is unchanged on refusal). `pipeline.busy` on a re-entrant call.
    pub fn set(&self, d: DofDesc) -> Result<(), RaaError> {
        d.validate()?;
        self.state.try_borrow_mut().map_err(|_| state_busy())?.desc = d;
        Ok(())
    }

    /// A copy of the current description (DOF defaults until set).
    pub fn get(&self) -> DofDesc {
        self.state
            .try_borrow()
            .map(|st| st.desc)
            .unwrap_or_default()
    }

    /// The frame's camera (called every frame by api-m14). The inverse projection is
    /// computed with `Mat4::inverse`; a singular `proj` keeps the previous camera. `view`
    /// is not used by the pass. Ignored on a re-entrant call.
    pub fn set_camera(&self, view: &Mat4, proj: &Mat4) {
        let _ = view;
        let Some(inv_proj) = proj.inverse() else {
            return;
        };
        if let Ok(mut st) = self.state.try_borrow_mut() {
            st.camera = Some(DofCamera {
                proj: *proj,
                inv_proj,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec4_at(bytes: &[u8], offset: usize) -> [f32; 4] {
        [0, 4, 8, 12].map(|i| {
            let o = offset + i;
            f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
        })
    }

    fn camera() -> DofCamera {
        let proj = Mat4::perspective_rh_zo(1.0, 16.0 / 9.0, 0.1, 100.0);
        DofCamera {
            proj,
            inv_proj: proj.inverse().unwrap(),
        }
    }

    fn handle() -> DofHandle {
        DofHandle {
            state: Rc::new(RefCell::new(DofState {
                desc: DofDesc::default(),
                camera: None,
            })),
        }
    }

    #[test]
    fn params_scale_max_coc_with_the_viewport() {
        let desc = DofDesc::default();
        let mut out = [0u8; PARAMS_LEN];
        pack_params(&mut out, &desc, Some(&camera()), (2560, 1440), 0.25);
        let lens = vec4_at(&out, 64);
        assert_eq!(lens[0], 10.0);
        assert!((lens[1] - aperture_term_px(&desc, 1440.0)).abs() < 1e-6);
        assert_eq!(lens[2], 48.0);
        assert!((lens[3] - 0.1).abs() < 1e-4, "near {}", lens[3]);
        assert_eq!(vec4_at(&out, 80), [0.05, 0.0, 0.0, 0.0]);
        assert_eq!(
            vec4_at(&out, 96),
            [2560.0, 1440.0, 1.0 / 2560.0, 1.0 / 1440.0]
        );
        assert_eq!(vec4_at(&out, 112), [0.0, 0.0, 0.0, 0.25]);
        let mut inv = [0u8; 64];
        camera().inv_proj.write_le_bytes(&mut inv);
        assert_eq!(&out[0..64], &inv[..]);
    }

    #[test]
    fn params_carry_the_autofocus_flag_and_uv() {
        let desc = DofDesc {
            autofocus: Some([0.25, 0.75]),
            ..DofDesc::default()
        };
        let mut out = [0u8; PARAMS_LEN];
        pack_params(&mut out, &desc, Some(&camera()), (1280, 720), 1.0);
        assert_eq!(vec4_at(&out, 112), [0.25, 0.75, 1.0, 1.0]);
        assert_eq!(vec4_at(&out, 64)[2], 24.0);
        // without a camera: identity, CoC 0 and autofocus off
        pack_params(&mut out, &desc, None, (1280, 720), 1.0);
        assert_eq!(vec4_at(&out, 112)[2], 0.0);
        assert_eq!(vec4_at(&out, 64)[1..3], [0.0, 0.0]);
        let mut id = [0u8; 64];
        Mat4::IDENTITY.write_le_bytes(&mut id);
        assert_eq!(&out[0..64], &id[..]);
    }

    #[test]
    fn handle_set_refuses_invalid_and_keeps_desc() {
        let h = handle();
        let bad = DofDesc {
            f_stop: 0.1,
            ..DofDesc::default()
        };
        let err = h.set(bad).unwrap_err();
        assert_eq!(err.code(), "post.invalid");
        assert_eq!(h.get(), DofDesc::default());
        let good = DofDesc {
            focus_distance_m: 3.0,
            autofocus: Some([0.5, 0.5]),
            ..DofDesc::default()
        };
        h.set(good).unwrap();
        assert_eq!(h.clone().get(), good);
    }

    #[test]
    fn singular_projection_keeps_the_previous_camera() {
        let h = handle();
        let c = camera();
        h.set_camera(&Mat4::IDENTITY, &c.proj);
        let zero = Mat4::from_cols(
            Vec4::default(),
            Vec4::default(),
            Vec4::default(),
            Vec4::default(),
        );
        h.set_camera(&Mat4::IDENTITY, &zero);
        assert_eq!(h.state.borrow().camera, Some(c));
    }

    #[cfg(not(target_arch = "wasm32"))]
    mod gpu {
        use std::cell::RefCell;
        use std::rc::Rc;

        use raa_core::testing::{skip, software_gpu};
        use raa_core::{FrameGraph, GpuContext, OffscreenTarget, wgpu};
        use raa_math::Mat4;
        use raa_pipeline::{FeaturePass, Pipeline, RecordCtx};

        use super::super::*;

        const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
        const SIZE: u32 = 32;
        /// One Rgba16Float row of 32 px is exactly COPY_BYTES_PER_ROW_ALIGNMENT.
        const ROW: u32 = SIZE * 8;
        const COLOUR: [f64; 3] = [0.8, 0.4, 0.2];

        /// Clears hdr.dst() to COLOUR and depth to 1, then swaps (stands in for forward).
        struct Setup;

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
                    r: COLOUR[0],
                    g: COLOUR[1],
                    b: COLOUR[2],
                    a: 1.0,
                };
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
                        view: ctx.targets.depth(),
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

        /// Copies hdr.src() into a readback buffer after dof.
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

        /// Largest |rgb - COLOUR| over every pixel of the HDR chain after `frames` frames.
        fn run(gpu: &GpuContext, desc: Option<DofDesc>, camera: bool, frames: u32) -> f32 {
            let target = OffscreenTarget::new(gpu, SIZE, SIZE, FORMAT);
            let pipeline = Rc::new(RefCell::new(Pipeline::new(gpu, FORMAT, false)));
            let buffer = Rc::new(gpu.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("test.readback"),
                size: u64::from(ROW * SIZE),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            {
                let mut p = pipeline.borrow_mut();
                p.add(Box::new(Setup)).unwrap();
                p.add(Box::new(Probe(buffer.clone()))).unwrap();
            }
            let handle = DofFeature::install(gpu, &pipeline).unwrap();
            if let Some(d) = desc {
                handle.set(d).unwrap();
            }
            let err = DofFeature::install(gpu, &pipeline).err().unwrap();
            assert_eq!(err.code(), "pipeline.duplicate_pass");
            let mut graph = FrameGraph::new(FORMAT);
            Pipeline::install(&pipeline, &mut graph);
            graph.compile().unwrap();
            let proj = Mat4::perspective_rh_zo(1.0, 1.0, 0.1, 100.0);
            for _ in 0..frames {
                if camera {
                    handle.set_camera(&Mat4::IDENTITY, &proj);
                }
                let commands = graph.execute(gpu, target.view(), (SIZE, SIZE)).unwrap();
                gpu.queue().submit([commands]);
            }
            let slice = buffer.slice(..);
            slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
            gpu.device()
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            let mapped = slice.get_mapped_range().unwrap();
            let mut worst = 0.0f32;
            for px in mapped.chunks_exact(8) {
                for (c, want) in px.chunks_exact(2).zip(COLOUR) {
                    let v = f16_to_f32(u16::from_le_bytes([c[0], c[1]]));
                    worst =
                        worst
                            .max((v - want as f32).abs())
                            .max(if v.is_nan() { 1.0 } else { 0.0 });
                }
            }
            worst
        }

        #[test]
        fn constant_hdr_keeps_its_colour_through_dof() {
            let Some(gpu) = software_gpu() else {
                skip(
                    "constant_hdr_keeps_its_colour_through_dof",
                    "software_gpu returned None; see the software_gpu line above",
                );
                return;
            };
            // depth 1 = the far plane (100 m), well behind the 10 m focus: maximum far CoC
            let e = run(&gpu, None, true, 1);
            assert!(e < 1e-2, "dof shifted a constant colour by {e}");
            let autofocus = DofDesc {
                autofocus: Some([0.5, 0.5]),
                focus_distance_m: 2.0,
                ..DofDesc::default()
            };
            let e = run(&gpu, Some(autofocus), true, 2);
            assert!(
                e < 1e-2,
                "dof with autofocus shifted a constant colour by {e}"
            );
            // no camera yet: a copy through the composite with CoC 0
            let e = run(&gpu, None, false, 1);
            assert!(
                e < 1e-3,
                "dof without a camera shifted a constant colour by {e}"
            );
        }
    }
}
