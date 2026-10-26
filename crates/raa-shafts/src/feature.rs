//! `ShaftsFeature::install` and the `ShaftsHandle`: the `sunshafts` render pass (S12, order 69).
//!
//! On resize the pass creates the half-resolution Rgba16Float march target (at least 1x1)
//! and the composite bind group of each texture of the HDR chain. Each frame it marches the
//! view ray into the half-res target (`fs_march`, group 1 = the lighting bind group), then
//! adds the bilateral upsample of the march to hdr.src() into hdr.dst() (`fs_composite`)
//! and swaps. With shadows disabled the march writes 0, so the pass is a valid no-op.
//!
//! The pass and every handle clone share one state: the description, the camera from
//! `set_camera` and the sun from `set_sun` (both called every frame by api-m14; the pass
//! has no scene access). Until a camera is set the in-scatter intensity is 0.

use std::cell::RefCell;
use std::num::NonZeroU64;
use std::rc::Rc;

use raa_core::wgpu;
use raa_core::{GpuContext, RaaError};
use raa_math::{Mat4, Vec3};
use raa_pipeline::{FeaturePass, HDR_FORMAT, Pipeline, PrepareCtx, RecordCtx, Targets};

use crate::desc::ShaftsDesc;
use crate::shaders::{FS_COMPOSITE, FS_MARCH, PARAMS_SIZE, SHAFTS_WGSL, VS_MAIN};

/// S12 name of the pass.
const NAME: &str = "sunshafts";
/// S12 order of the pass (after `fog`, before `opaque.copy`).
const ORDER: u32 = 69;
/// Format of the half-res march target.
const MARCH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
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

/// `pipeline.busy` for the shared shafts state (only reachable by a re-entrant call).
fn state_busy() -> RaaError {
    refusal(
        "pipeline.busy",
        "the sun shafts state is borrowed (re-entrant call while a frame executes)",
    )
}

/// State shared by the pass and every clone of a `ShaftsHandle`.
struct ShaftsState {
    desc: ShaftsDesc,
    /// False until `set_camera` receives an invertible camera (intensity 0 until then).
    has_camera: bool,
    view: Mat4,
    inv_view_proj: Mat4,
    camera_pos: Vec3,
    /// Normalised sun travel direction.
    sun_dir: Vec3,
    /// Linear colour x intensity.
    sun_color: [f32; 3],
}

impl Default for ShaftsState {
    fn default() -> Self {
        ShaftsState {
            desc: ShaftsDesc::default(),
            has_camera: false,
            view: Mat4::IDENTITY,
            inv_view_proj: Mat4::IDENTITY,
            camera_pos: Vec3::new(0.0, 0.0, 0.0),
            sun_dir: Vec3::new(0.0, -1.0, 0.0),
            sun_color: [1.0; 3],
        }
    }
}

type SharedShafts = Rc<RefCell<ShaftsState>>;

fn write_vec4(out: &mut [u8; PARAMS_LEN], offset: usize, v: [f32; 4]) {
    for (chunk, x) in out[offset..offset + 16].chunks_exact_mut(4).zip(v) {
        chunk.copy_from_slice(&x.to_le_bytes());
    }
}

/// Packs `ShaftsParams` (layout in [`PARAMS_SIZE`]) for a full-res `size` and its half-res
/// march target `half`.
fn pack_params(
    out: &mut [u8; PARAMS_LEN],
    st: &ShaftsState,
    size: (u32, u32),
    half: (u32, u32),
    frame: u64,
) {
    st.inv_view_proj.write_le_bytes(&mut out[0..64]);
    st.view.write_le_bytes(&mut out[64..128]);
    let c = st.camera_pos;
    write_vec4(out, 128, [c.x, c.y, c.z, 0.0]);
    let intensity = if st.has_camera {
        st.desc.intensity
    } else {
        0.0
    };
    let d = st.sun_dir;
    write_vec4(out, 144, [d.x, d.y, d.z, intensity]);
    let [r, g, b] = st.sun_color;
    write_vec4(out, 160, [r, g, b, 0.0]);
    let viewport = [half.0, half.1, size.0, size.1].map(|v| v.max(1) as f32);
    write_vec4(out, 176, viewport);
    // the shader takes the frame index modulo 64; keep it exact in f32
    let frame = (frame % 64) as f32;
    write_vec4(
        out,
        192,
        [st.desc.density, st.desc.max_distance_m, frame, 0.0],
    );
}

/// Half resolution of `size`, at least 1x1.
fn half_size(size: (u32, u32)) -> (u32, u32) {
    (size.0.div_ceil(2).max(1), size.1.div_ceil(2).max(1))
}

fn layout_entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty,
        count: None,
    }
}

fn texture_type(sample_type: wgpu::TextureSampleType) -> wgpu::BindingType {
    wgpu::BindingType::Texture {
        sample_type,
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    }
}

/// Group-0 layout shared by both entries: params, depth, march result, hdr src, sampler.
fn create_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let float = wgpu::TextureSampleType::Float { filterable: true };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("raa-shafts.layout"),
        entries: &[
            layout_entry(
                0,
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(PARAMS_SIZE),
                },
            ),
            layout_entry(1, texture_type(wgpu::TextureSampleType::Depth)),
            layout_entry(2, texture_type(float)),
            layout_entry(3, texture_type(float)),
            layout_entry(
                4,
                wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            ),
        ],
    })
}

/// A fullscreen pipeline with layout [private group 0, lighting layout].
fn create_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    layouts: [&wgpu::BindGroupLayout; 2],
    entry: &str,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(entry),
        bind_group_layouts: &[Some(layouts[0]), Some(layouts[1])],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(entry),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(VS_MAIN),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn texture_2d(
    device: &wgpu::Device,
    label: &str,
    size: (u32, u32),
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: MARCH_FORMAT,
        usage,
        view_formats: &[],
    })
}

/// The composite bind group reading one texture of the HDR chain.
struct HdrGroup {
    texture: wgpu::Texture,
    group: wgpu::BindGroup,
}

/// The surface-sized resources: the march target and the bind groups over it.
struct Chain {
    size: (u32, u32),
    half: (u32, u32),
    _march: wgpu::Texture,
    march_view: wgpu::TextureView,
    /// `fs_march`: depth, with the 1x1 dummy at bindings 2-3.
    march_group: wgpu::BindGroup,
    /// One entry per texture of the HDR chain (A and B).
    hdr: [HdrGroup; 2],
}

/// The `sunshafts` render pass.
struct ShaftsPass {
    state: SharedShafts,
    layout: wgpu::BindGroupLayout,
    march: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    bytes: [u8; PARAMS_LEN],
    _dummy: wgpu::Texture,
    dummy_view: wgpu::TextureView,
    chain: Option<Chain>,
}

impl ShaftsPass {
    fn new(gpu: &GpuContext, lighting_layout: &wgpu::BindGroupLayout, state: SharedShafts) -> Self {
        let device = gpu.device();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa-shafts.shafts"),
            source: wgpu::ShaderSource::Wgsl(SHAFTS_WGSL.into()),
        });
        let layout = create_layout(device);
        let layouts = [&layout, lighting_layout];
        let march = create_pipeline(device, &module, layouts, FS_MARCH, MARCH_FORMAT);
        let composite = create_pipeline(device, &module, layouts, FS_COMPOSITE, HDR_FORMAT);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("raa-shafts.linear_clamp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-shafts.params"),
            size: PARAMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dummy = texture_2d(
            device,
            "raa-shafts.dummy",
            (1, 1),
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let dummy_view = dummy.create_view(&wgpu::TextureViewDescriptor::default());
        ShaftsPass {
            state,
            layout,
            march,
            composite,
            sampler,
            uniform,
            bytes: [0; PARAMS_LEN],
            _dummy: dummy,
            dummy_view,
            chain: None,
        }
    }

    fn group(
        &self,
        device: &wgpu::Device,
        depth: &wgpu::TextureView,
        march: &wgpu::TextureView,
        hdr: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let view = |binding, view| wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::TextureView(view),
        };
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-shafts.group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform.as_entire_binding(),
                },
                view(1, depth),
                view(2, march),
                view(3, hdr),
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }
}

/// One fullscreen draw into `target` (cleared) with group 0 and the lighting group 1.
fn draw(
    encoder: &mut wgpu::CommandEncoder,
    target: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    group: &wgpu::BindGroup,
    lighting: &wgpu::BindGroup,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("raa-shafts"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
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
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, group, &[]);
    pass.set_bind_group(1, lighting, &[]);
    pass.draw(0..3, 0..1);
}

impl FeaturePass for ShaftsPass {
    fn name(&self) -> &str {
        NAME
    }

    fn order(&self) -> u32 {
        ORDER
    }

    /// Recreates the half-res march target (RENDER_ATTACHMENT | TEXTURE_BINDING), the march
    /// bind group and the composite bind group of both textures of the HDR chain.
    fn resize(&mut self, gpu: &GpuContext, targets: &Targets) {
        let device = gpu.device();
        let size = targets.size();
        let half = half_size(size);
        let march = texture_2d(
            device,
            "raa-shafts.march",
            half,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let march_view = march.create_view(&wgpu::TextureViewDescriptor::default());
        let depth = targets.depth();
        let march_group = self.group(device, depth, &self.dummy_view, &self.dummy_view);
        let hdr = targets.hdr();
        let hdr_group = |texture: &wgpu::Texture, view| HdrGroup {
            texture: texture.clone(),
            group: self.group(device, depth, &march_view, view),
        };
        let groups = [
            hdr_group(hdr.src_texture(), hdr.src()),
            hdr_group(hdr.dst_texture(), hdr.dst()),
        ];
        self.chain = Some(Chain {
            size,
            half,
            _march: march,
            march_view,
            march_group,
            hdr: groups,
        });
    }

    /// Packs `ShaftsParams` from the shared state (in place) and uploads it.
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let st = self.state.try_borrow().map_err(|_| state_busy())?;
        let Some(chain) = self.chain.as_ref() else {
            return Ok(());
        };
        pack_params(&mut self.bytes, &st, chain.size, chain.half, ctx.frame);
        ctx.gpu.queue().write_buffer(&self.uniform, 0, &self.bytes);
        Ok(())
    }

    /// Marches into the half-res target, composites into hdr.dst() and swaps.
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let Some(chain) = self.chain.as_ref() else {
            return Err(RaaError::Resource(
                "sunshafts: record ran before a resize created the march target".to_owned(),
            ));
        };
        let hdr = ctx.targets.hdr();
        let src = hdr.src_texture();
        let Some(groups) = chain.hdr.iter().find(|g| g.texture == *src) else {
            return Err(RaaError::Resource(
                "sunshafts: the HDR source is not a texture the pass was resized for".to_owned(),
            ));
        };
        let lighting = ctx.lighting.bind_group();
        let encoder = &mut *ctx.encoder;
        draw(
            encoder,
            &chain.march_view,
            &self.march,
            &chain.march_group,
            lighting,
        );
        draw(encoder, hdr.dst(), &self.composite, &groups.group, lighting);
        hdr.swap();
        Ok(())
    }
}

/// Installs volumetric sun shafts on a pipeline.
pub struct ShaftsFeature;

impl ShaftsFeature {
    /// Builds the march and composite pipelines (layout [private, lighting layout]), the
    /// sampler and the params buffer, and adds the FeaturePass `sunshafts` (order 69) to
    /// `pipeline` with the default description.
    ///
    /// Refusals (nothing installed): `pipeline.busy` when the pipeline is borrowed,
    /// `pipeline.duplicate_pass` when `sunshafts` is already installed.
    pub fn install(
        gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
    ) -> Result<ShaftsHandle, RaaError> {
        let mut p = pipeline.try_borrow_mut().map_err(|_| pipeline_busy())?;
        if p.passes().iter().any(|info| info.name == NAME) {
            return Err(refusal(
                "pipeline.duplicate_pass",
                "the pass `sunshafts` is already installed",
            ));
        }
        let state = Rc::new(RefCell::new(ShaftsState::default()));
        let pass = ShaftsPass::new(gpu, p.lighting().layout(), state.clone());
        p.add(Box::new(pass))?;
        Ok(ShaftsHandle { state })
    }
}

/// Access to the installed sun shafts (clones share one state).
#[derive(Clone)]
pub struct ShaftsHandle {
    state: SharedShafts,
}

impl ShaftsHandle {
    /// Replaces the description after `ShaftsDesc::validate` (`post.invalid`; the stored
    /// description is unchanged on refusal). `pipeline.busy` on a re-entrant call.
    pub fn set(&self, d: ShaftsDesc) -> Result<(), RaaError> {
        d.validate()?;
        self.state.try_borrow_mut().map_err(|_| state_busy())?.desc = d;
        Ok(())
    }

    /// The current description (defaults until set).
    pub fn get(&self) -> ShaftsDesc {
        self.state
            .try_borrow()
            .map(|st| st.desc)
            .unwrap_or_default()
    }

    /// The scene camera of the frame. A singular view or view-projection keeps the
    /// previous camera.
    pub fn set_camera(&self, view: &Mat4, proj: &Mat4) {
        let (Some(inv_view), Some(inv_view_proj)) = (view.inverse(), proj.mul(view).inverse())
        else {
            return;
        };
        let Ok(mut st) = self.state.try_borrow_mut() else {
            return;
        };
        let t = inv_view.cols[3];
        st.view = *view;
        st.inv_view_proj = inv_view_proj;
        st.camera_pos = Vec3::new(t[0], t[1], t[2]);
        st.has_camera = true;
    }

    /// The sun of the frame: travel direction (normalised here; a zero or non-finite
    /// direction is ignored) and linear colour x intensity.
    pub fn set_sun(&self, dir: Vec3, color: [f32; 3]) {
        let Ok(mut st) = self.state.try_borrow_mut() else {
            return;
        };
        let dir = dir.normalize_or_zero();
        if dir.is_finite() && dir.length_squared() > 0.0 {
            st.sun_dir = dir;
        }
        if color.iter().all(|c| c.is_finite()) {
            st.sun_color = color;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(bytes: &[u8; PARAMS_LEN], offset: usize) -> f32 {
        f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    }

    #[test]
    fn pack_params_places_density_max_distance_and_intensity() {
        let st = ShaftsState {
            desc: ShaftsDesc {
                intensity: 2.0,
                density: 0.05,
                max_distance_m: 40.0,
            },
            ..ShaftsState::default()
        };
        let mut out = [0u8; PARAMS_LEN];
        pack_params(&mut out, &st, (101, 50), half_size((101, 50)), 67);
        assert_eq!(read(&out, 192), 0.05);
        assert_eq!(read(&out, 196), 40.0);
        assert_eq!(read(&out, 200), 3.0);
        assert_eq!(
            [176, 180, 184, 188].map(|o| read(&out, o)),
            [51.0, 25.0, 101.0, 50.0]
        );
        // no camera yet: intensity 0
        assert_eq!(read(&out, 156), 0.0);
        let handle = ShaftsHandle {
            state: Rc::new(RefCell::new(st)),
        };
        let view = Mat4::from_translation(Vec3::new(-1.0, -2.0, -3.0));
        let proj = Mat4::perspective_rh_zo(1.0, 2.0, 0.1, 100.0);
        handle.set_camera(&view, &proj);
        handle.set_camera(&Mat4::default(), &proj);
        handle.set_sun(Vec3::new(0.0, -2.0, 0.0), [3.0, 2.0, 1.0]);
        pack_params(&mut out, &handle.state.borrow(), (8, 8), (4, 4), 0);
        assert_eq!(read(&out, 156), 2.0);
        assert_eq!([128, 132, 136].map(|o| read(&out, o)), [1.0, 2.0, 3.0]);
        assert_eq!([144, 148, 152].map(|o| read(&out, o)), [0.0, -1.0, 0.0]);
        assert_eq!([160, 164, 168].map(|o| read(&out, o)), [3.0, 2.0, 1.0]);
        assert_eq!(read(&out, 64 + 48), -1.0);
    }

    #[test]
    fn half_size_is_at_least_one_texel() {
        assert_eq!(half_size((1, 1)), (1, 1));
        assert_eq!(half_size((0, 0)), (1, 1));
        assert_eq!(half_size((1280, 721)), (640, 361));
    }

    #[test]
    fn handle_set_refuses_invalid_and_keeps_desc() {
        let handle = ShaftsHandle {
            state: Rc::new(RefCell::new(ShaftsState::default())),
        };
        let bad = ShaftsDesc {
            density: -1.0,
            ..ShaftsDesc::default()
        };
        let err = handle.set(bad).unwrap_err();
        assert_eq!(err.code(), "post.invalid");
        assert_eq!(handle.get(), ShaftsDesc::default());
        let good = ShaftsDesc {
            max_distance_m: 30.0,
            ..ShaftsDesc::default()
        };
        handle.set(good).unwrap();
        assert_eq!(handle.clone().get(), good);
    }

    #[cfg(not(target_arch = "wasm32"))]
    mod gpu {
        use std::future::Future;
        use std::task::{Context, Poll, Waker};

        use raa_core::testing::{skip, software_gpu};
        use raa_core::{FrameGraph, OffscreenTarget};

        use super::super::*;

        const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
        /// One Rgba16Float row of 32 px is exactly COPY_BYTES_PER_ROW_ALIGNMENT.
        const ROW: u32 = 32 * 8;
        const SIZE: u32 = 32;

        /// Clears hdr.dst() to a constant colour and swaps (stands in for forward).
        struct Setup(f64);

        impl FeaturePass for Setup {
            fn name(&self) -> &str {
                "test.setup"
            }
            fn order(&self) -> u32 {
                60
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

        /// Copies the first row of hdr.src() into a readback buffer after the shafts.
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

        /// Polls `fut` to completion, driving the device between polls.
        fn block_on<F: Future>(gpu: &GpuContext, fut: F) -> F::Output {
            let mut fut = std::pin::pin!(fut);
            let mut cx = Context::from_waker(Waker::noop());
            loop {
                if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
                    return v;
                }
                let _ = gpu.device().poll(wgpu::PollType::wait_indefinitely());
            }
        }

        #[test]
        fn shadows_disabled_leaves_constant_hdr_unchanged() {
            let Some(gpu) = software_gpu() else {
                skip(
                    "shadows_disabled_leaves_constant_hdr_unchanged",
                    "software_gpu returned None; see the software_gpu line above",
                );
                return;
            };
            let scope = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
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
                p.add(Box::new(Setup(2.0))).unwrap();
                p.add(Box::new(Probe(buffer.clone()))).unwrap();
            }
            let handle = ShaftsFeature::install(&gpu, &pipeline).unwrap();
            let err = ShaftsFeature::install(&gpu, &pipeline).err().unwrap();
            assert_eq!(err.code(), "pipeline.duplicate_pass");
            let view = Mat4::look_at_rh(
                Vec3::new(0.0, 2.0, 5.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            );
            handle.set_camera(&view, &Mat4::perspective_rh_zo(1.0, 1.0, 0.1, 100.0));
            handle.set_sun(Vec3::new(0.3, -1.0, 0.2), [5.0, 5.0, 5.0]);
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
            for px in mapped.chunks_exact(8).take(SIZE as usize) {
                let r = f16_to_f32(u16::from_le_bytes([px[0], px[1]]));
                assert!((r - 2.0).abs() < 1e-2, "shafts without shadows gave {r}");
            }
            drop(mapped);
            let error = block_on(&gpu, scope.pop());
            assert!(error.is_none(), "validation error: {error:?}");
        }
    }
}
