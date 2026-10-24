//! `GradingFeature::install` and the `GradingHandle`: the `grading` render pass (S12, order 130).
//!
//! Each frame the pass draws one fullscreen triangle reading hdr.src() into hdr.dst() with
//! `fs_grading` (white balance, exposure, log contrast, CDL, split toning, saturation and
//! vibrance, vignette), then swaps. The two bind groups (one per HDR chain texture) are
//! built on resize.
//!
//! The pass and every handle clone share one state (the description). The pass sets no
//! group-1 resource, so disabling it leaves the HDR chain untouched.

use std::cell::RefCell;
use std::num::NonZeroU64;
use std::rc::Rc;

use raa_core::wgpu;
use raa_core::{GpuContext, RaaError};
use raa_pipeline::{FeaturePass, HDR_FORMAT, Pipeline, PrepareCtx, RecordCtx, Targets};

use crate::desc::GradingDesc;
use crate::shaders::{FS_GRADING, GRADING_WGSL, PARAMS_SIZE, VS_MAIN};
use crate::wb::white_balance_matrix;

/// S12 name of the pass.
const NAME: &str = "grading";
/// S12 order of the pass (after `bloom`, before `tonemap`).
const ORDER: u32 = 130;
/// Byte size of the uniform buffer (one `GradingParams`).
const UNIFORM_LEN: usize = PARAMS_SIZE as usize;

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

/// `pipeline.busy` for the shared grading state (only reachable by a re-entrant call).
fn state_busy() -> RaaError {
    refusal(
        "pipeline.busy",
        "the grading state is borrowed (re-entrant call while a frame executes)",
    )
}

/// State shared by the pass and every clone of a `GradingHandle`.
struct GradingState {
    desc: GradingDesc,
}

type SharedGrading = Rc<RefCell<GradingState>>;

/// Packs `GradingParams` in place: wb columns @0/@16/@32, exposure_contrast @48, lift @64,
/// gamma @80, gain @96, shadows_tint @112, highlights_tint @128, misc @144 (split_balance,
/// vignette, aspect w / h of `size`, 0).
fn pack_params(out: &mut [u8; UNIFORM_LEN], size: (u32, u32), d: &GradingDesc) {
    let m = white_balance_matrix(d.temperature_k, d.tint);
    let col = |j: usize| [m[0][j], m[1][j], m[2][j], 0.0];
    let rgb = |v: [f32; 3]| [v[0], v[1], v[2], 0.0];
    let aspect = size.0.max(1) as f32 / size.1.max(1) as f32;
    let vecs: [[f32; 4]; 10] = [
        col(0),
        col(1),
        col(2),
        [d.exposure_ev.exp2(), d.contrast, d.saturation, d.vibrance],
        rgb(d.lift),
        rgb(d.gamma),
        rgb(d.gain),
        rgb(d.shadows_tint),
        rgb(d.highlights_tint),
        [d.split_balance, d.vignette, aspect, 0.0],
    ];
    for (chunk, x) in out
        .chunks_exact_mut(4)
        .zip(vecs.iter().flat_map(|v| v.iter()))
    {
        chunk.copy_from_slice(&x.to_le_bytes());
    }
}

fn layout_entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty,
        count: None,
    }
}

/// Group-0 layout: params, source, sampler.
fn create_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("raa-grading.layout"),
        entries: &[
            layout_entry(
                0,
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(PARAMS_SIZE),
                },
            ),
            layout_entry(
                1,
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
            ),
            layout_entry(
                2,
                wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            ),
        ],
    })
}

fn create_pipeline(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> wgpu::RenderPipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("raa-grading.grading"),
        source: wgpu::ShaderSource::Wgsl(GRADING_WGSL.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(FS_GRADING),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(FS_GRADING),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some(VS_MAIN),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some(FS_GRADING),
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

/// The `grading` render pass.
struct GradingPass {
    state: SharedGrading,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    bytes: [u8; UNIFORM_LEN],
    /// One (texture, bind group reading it) per texture of the HDR chain (A and B).
    groups: Vec<(wgpu::Texture, wgpu::BindGroup)>,
}

impl GradingPass {
    fn new(gpu: &GpuContext, state: SharedGrading) -> Self {
        let device = gpu.device();
        let layout = create_layout(device);
        let pipeline = create_pipeline(device, &layout);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("raa-grading.linear_clamp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-grading.params"),
            size: PARAMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        GradingPass {
            state,
            layout,
            pipeline,
            sampler,
            uniform,
            bytes: [0; UNIFORM_LEN],
            groups: Vec::new(),
        }
    }

    fn group(&self, device: &wgpu::Device, src: &wgpu::TextureView) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-grading.group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(src),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }
}

impl FeaturePass for GradingPass {
    fn name(&self) -> &str {
        NAME
    }

    fn order(&self) -> u32 {
        ORDER
    }

    /// Rebuilds the bind group of each texture of the HDR chain.
    fn resize(&mut self, gpu: &GpuContext, targets: &Targets) {
        let device = gpu.device();
        let hdr = targets.hdr();
        self.groups = vec![
            (hdr.src_texture().clone(), self.group(device, hdr.src())),
            (hdr.dst_texture().clone(), self.group(device, hdr.dst())),
        ];
    }

    /// Writes the params from the handle's description and the viewport aspect (in place).
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let desc = self.state.try_borrow().map_err(|_| state_busy())?.desc;
        pack_params(&mut self.bytes, ctx.size, &desc);
        ctx.gpu.queue().write_buffer(&self.uniform, 0, &self.bytes);
        Ok(())
    }

    /// Grades hdr.src() into hdr.dst() (one fullscreen draw) and swaps.
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let hdr = ctx.targets.hdr();
        let src = hdr.src_texture();
        let Some((_, group)) = self.groups.iter().find(|(t, _)| t == src) else {
            return Err(RaaError::Resource(
                "grading: the HDR source is not a texture the pass was resized for".to_owned(),
            ));
        };
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("raa-grading"),
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
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.draw(0..3, 0..1);
        drop(pass);
        hdr.swap();
        Ok(())
    }
}

/// Installs HDR colour grading on a pipeline.
pub struct GradingFeature;

impl GradingFeature {
    /// Builds the render pipeline (`fs_grading` into HDR_FORMAT), the linear clamp sampler
    /// and the params buffer, and adds the FeaturePass `grading` (order 130) to `pipeline`
    /// with the `neutral` description.
    ///
    /// Refusals (nothing installed): `pipeline.busy` when the pipeline is borrowed,
    /// `pipeline.duplicate_pass` when `grading` is already installed.
    pub fn install(
        gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
    ) -> Result<GradingHandle, RaaError> {
        let mut p = pipeline.try_borrow_mut().map_err(|_| pipeline_busy())?;
        if p.passes().iter().any(|info| info.name == NAME) {
            return Err(refusal(
                "pipeline.duplicate_pass",
                "the pass `grading` is already installed",
            ));
        }
        let state = Rc::new(RefCell::new(GradingState {
            desc: GradingDesc::default(),
        }));
        p.add(Box::new(GradingPass::new(gpu, state.clone())))?;
        Ok(GradingHandle { state })
    }
}

/// Access to the installed grading (clones share one state).
#[derive(Clone)]
pub struct GradingHandle {
    state: SharedGrading,
}

impl GradingHandle {
    /// Replaces the description after `GradingDesc::validate` (`post.invalid`; the stored
    /// description is unchanged on refusal). `pipeline.busy` on a re-entrant call.
    pub fn set(&self, d: GradingDesc) -> Result<(), RaaError> {
        d.validate()?;
        self.state.try_borrow_mut().map_err(|_| state_busy())?.desc = d;
        Ok(())
    }

    /// The current description (`neutral` until set).
    pub fn get(&self) -> GradingDesc {
        self.state
            .try_borrow()
            .map(|st| st.desc)
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(bytes: &[u8; UNIFORM_LEN], vec: usize) -> [f32; 4] {
        let o = vec * 16;
        [0, 4, 8, 12].map(|i| {
            f32::from_le_bytes([
                bytes[o + i],
                bytes[o + i + 1],
                bytes[o + i + 2],
                bytes[o + i + 3],
            ])
        })
    }

    #[test]
    fn neutral_params_pack_identity() {
        let mut out = [0u8; UNIFORM_LEN];
        pack_params(&mut out, (200, 100), &GradingDesc::default());
        for j in 0..3 {
            let c = read(&out, j);
            for (i, v) in c.iter().enumerate() {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((v - want).abs() < 1e-3, "wb column {j} = {c:?}");
            }
        }
        assert_eq!(read(&out, 3), [1.0, 1.0, 1.0, 0.0]);
        assert_eq!(read(&out, 4), [0.0; 4]);
        assert_eq!(read(&out, 5), [1.0, 1.0, 1.0, 0.0]);
        assert_eq!(read(&out, 6), [1.0, 1.0, 1.0, 0.0]);
        assert_eq!(read(&out, 7), [0.5, 0.5, 0.5, 0.0]);
        assert_eq!(read(&out, 8), [0.5, 0.5, 0.5, 0.0]);
        assert_eq!(read(&out, 9), [0.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn params_pack_columns_and_exposure() {
        let d = GradingDesc {
            temperature_k: 9000.0,
            exposure_ev: 2.0,
            ..GradingDesc::default()
        };
        let m = white_balance_matrix(9000.0, 0.0);
        let mut out = [0u8; UNIFORM_LEN];
        pack_params(&mut out, (1, 1), &d);
        assert_eq!(read(&out, 0), [m[0][0], m[1][0], m[2][0], 0.0]);
        assert_eq!(read(&out, 2), [m[0][2], m[1][2], m[2][2], 0.0]);
        assert_eq!(read(&out, 3)[0], 4.0);
    }

    #[test]
    fn handle_set_refuses_invalid_and_keeps_desc() {
        let handle = GradingHandle {
            state: Rc::new(RefCell::new(GradingState {
                desc: GradingDesc::default(),
            })),
        };
        let good = GradingDesc::preset("candy").unwrap();
        handle.set(good).unwrap();
        let bad = GradingDesc {
            contrast: 0.0,
            ..GradingDesc::default()
        };
        let err = handle.set(bad).unwrap_err();
        assert_eq!(err.code(), "post.invalid");
        assert_eq!(handle.get(), good);
        assert_eq!(handle.clone().get(), good);
    }

    #[cfg(not(target_arch = "wasm32"))]
    mod gpu {
        use std::cell::RefCell;
        use std::rc::Rc;

        use raa_core::testing::{skip, software_gpu};
        use raa_core::{FrameGraph, OffscreenTarget, wgpu};
        use raa_pipeline::{FeaturePass, Pipeline, RecordCtx};

        use super::super::*;

        const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
        const SIZE: u32 = 32;
        /// One Rgba16Float row of 32 px is exactly COPY_BYTES_PER_ROW_ALIGNMENT.
        const ROW: u32 = 32 * 8;

        /// Clears hdr.dst() to a constant colour and swaps (stands in for forward).
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
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                drop(pass);
                hdr.swap();
                Ok(())
            }
        }

        /// Copies the centre row of hdr.src() into a readback buffer after grading.
        struct Probe(Rc<wgpu::Buffer>);

        impl FeaturePass for Probe {
            fn name(&self) -> &str {
                "test.probe"
            }
            fn order(&self) -> u32 {
                150
            }
            fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
                let mut copy = ctx.targets.hdr().src_texture().as_image_copy();
                copy.origin.y = SIZE / 2;
                ctx.encoder.copy_texture_to_buffer(
                    copy,
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
                        height: 1,
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
        fn neutral_grade_keeps_mid_grey_at_centre() {
            let Some(gpu) = software_gpu() else {
                skip(
                    "neutral_grade_keeps_mid_grey_at_centre",
                    "software_gpu returned None; see the software_gpu line above",
                );
                return;
            };
            let target = OffscreenTarget::new(&gpu, SIZE, SIZE, FORMAT);
            let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, FORMAT, false)));
            let buffer = Rc::new(gpu.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("test.readback"),
                size: u64::from(ROW),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            {
                let mut p = pipeline.borrow_mut();
                p.add(Box::new(Setup(0.18))).unwrap();
                p.add(Box::new(Probe(buffer.clone()))).unwrap();
            }
            let handle = GradingFeature::install(&gpu, &pipeline).unwrap();
            handle.set(GradingDesc::preset("neutral").unwrap()).unwrap();
            let err = GradingFeature::install(&gpu, &pipeline).err().unwrap();
            assert_eq!(err.code(), "pipeline.duplicate_pass");
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
            let mapped = slice.get_mapped_range().unwrap();
            let o = (SIZE / 2) as usize * 8;
            for ch in 0..3 {
                let v = f16_to_f32(u16::from_le_bytes([
                    mapped[o + 2 * ch],
                    mapped[o + 2 * ch + 1],
                ]));
                assert!(
                    (v - 0.18).abs() < 1e-2,
                    "neutral grade of 0.18 gave {v} (channel {ch})"
                );
            }
        }
    }
}
