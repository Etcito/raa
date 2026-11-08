use std::cell::RefCell;
use std::num::NonZeroU64;
use std::rc::Rc;

use raa_core::wgpu;
use raa_core::{GpuContext, RaaError};
use raa_pipeline::{FeaturePass, HDR_FORMAT, Pipeline, PrepareCtx, RecordCtx, Targets};

use crate::desc::CelSettings;
use crate::shaders::{CEL_WGSL, FS_CEL, PARAMS_SIZE, VS_MAIN};
use raa_math::Mat4;

/// S12 name of the pass.
const NAME: &str = "cel.resolve";
/// S12 order of the pass (after `bloom`, before `tonemap`).
const ORDER: u32 = 190;
/// Byte size of the uniform buffer (one `CelPost`).
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

/// `pipeline.busy` for the shared cel state (only reachable by a re-entrant call).
fn state_busy() -> RaaError {
    refusal(
        "pipeline.busy",
        "the cel state is borrowed (re-entrant call while a frame executes)",
    )
}

/// State shared by the pass and every clone of a `CelHandle`.
struct CelState {
    desc: CelSettings,
    camera: [Mat4; 2],
}

type SharedCel = Rc<RefCell<CelState>>;

fn layout_entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty,
        count: None,
    }
}

/// Group-0 layout: params, HDR source, depth, normal, camera.
fn create_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("raa-cel.layout"),
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
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
            ),
            layout_entry(
                2,
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
            ),
            layout_entry(
                3,
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
            ),
            layout_entry(
                4,
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(128),
                },
            ),
        ],
    })
}

fn create_pipeline(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> wgpu::RenderPipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("raa-cel.cel"),
        source: wgpu::ShaderSource::Wgsl(CEL_WGSL.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(FS_CEL),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(FS_CEL),
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
            entry_point: Some(FS_CEL),
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

/// The `cel` render pass.
struct CelPass {
    state: SharedCel,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    camera: wgpu::Buffer,
    uniform: wgpu::Buffer,
    bytes: [u8; UNIFORM_LEN],
    /// One (texture, bind group reading it) per texture of the HDR chain (A and B).
    groups: Vec<(wgpu::Texture, wgpu::BindGroup)>,
}

impl CelPass {
    fn new(gpu: &GpuContext, state: SharedCel) -> Self {
        let device = gpu.device();
        let layout = create_layout(device);
        let pipeline = create_pipeline(device, &layout);
        let camera = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-cel.camera"),
            size: 128,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-cel.params"),
            size: PARAMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        CelPass {
            state,
            layout,
            pipeline,
            camera,
            uniform,
            bytes: [0; UNIFORM_LEN],
            groups: Vec::new(),
        }
    }

    fn group(
        &self,
        device: &wgpu::Device,
        src: &wgpu::TextureView,
        targets: &Targets,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-cel.group"),
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
                    resource: wgpu::BindingResource::TextureView(targets.depth()),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(targets.normal()),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.camera.as_entire_binding(),
                },
            ],
        })
    }
}

impl FeaturePass for CelPass {
    fn name(&self) -> &str {
        NAME
    }

    fn set_enabled(&mut self, enabled: bool, lighting: &mut raa_pipeline::LightingResources) {
        if !enabled {
            lighting.env_mut().cel = [0.0; 4];
        }
    }

    fn order(&self) -> u32 {
        ORDER
    }

    /// Rebuilds the bind group of each texture of the HDR chain.
    fn resize(&mut self, gpu: &GpuContext, targets: &Targets) {
        let device = gpu.device();
        let hdr = targets.hdr();
        self.groups = vec![
            (
                hdr.src_texture().clone(),
                self.group(device, hdr.src(), targets),
            ),
            (
                hdr.dst_texture().clone(),
                self.group(device, hdr.dst(), targets),
            ),
        ];
    }

    /// Writes the params from the handle's description and the viewport size (in place).
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let (desc, camera) = {
            let st = self.state.try_borrow().map_err(|_| state_busy())?;
            (st.desc, st.camera)
        };
        self.bytes = desc.to_bytes(ctx.size);
        let mut bytes = [0; 128];
        for (out, m) in bytes.chunks_exact_mut(64).zip(camera) {
            m.write_le_bytes(out);
        }
        ctx.gpu.queue().write_buffer(&self.camera, 0, &bytes);
        ctx.lighting.env_mut().cel = [
            u32::from(desc.toon) as f32,
            desc.softness,
            desc.specular,
            0.0,
        ];
        ctx.gpu.queue().write_buffer(&self.uniform, 0, &self.bytes);
        Ok(())
    }

    /// Resolves hdr.src() into hdr.dst() (one fullscreen draw) and swaps.
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let hdr = ctx.targets.hdr();
        let src = hdr.src_texture();
        let Some((_, group)) = self.groups.iter().find(|(t, _)| t == src) else {
            return Err(RaaError::Resource(
                "cel: the HDR source is not a texture the pass was resized for".to_owned(),
            ));
        };
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("raa-cel"),
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

/// Installs cel palette resolve on a pipeline.
pub struct CelFeature;

impl CelFeature {
    /// Adds `cel.resolve` at order 190, enabled with default controls.
    /// Install the Direct provider separately with `raa_cel::install` and select
    /// neutral linear tone mapping in the host to preserve the resolved palette.
    ///
    /// Refusals (nothing installed): `pipeline.busy` when the pipeline is borrowed,
    /// `pipeline.duplicate_pass` when `cel.resolve` is already installed.
    pub fn install(
        gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
    ) -> Result<CelHandle, RaaError> {
        let mut p = pipeline.try_borrow_mut().map_err(|_| pipeline_busy())?;
        if p.passes().iter().any(|info| info.name == NAME) {
            return Err(refusal(
                "pipeline.duplicate_pass",
                "the pass `cel.resolve` is already installed",
            ));
        }
        let state = Rc::new(RefCell::new(CelState {
            desc: CelSettings::default(),
            camera: [Mat4::IDENTITY; 2],
        }));
        p.add(Box::new(CelPass::new(gpu, state.clone())))?;
        Ok(CelHandle { state })
    }
}

/// Access to the installed cel (clones share one state).
#[derive(Clone)]
pub struct CelHandle {
    state: SharedCel,
}

impl CelHandle {
    /// Replaces the description after `CelSettings::validate` (`cel.invalid`; the stored
    /// description is unchanged on refusal). `pipeline.busy` on a re-entrant call.
    pub fn set(&self, d: CelSettings) -> Result<(), RaaError> {
        d.validate()?;
        self.state.try_borrow_mut().map_err(|_| state_busy())?.desc = d;
        Ok(())
    }

    /// The current description (defaults until set).
    pub fn get(&self) -> CelSettings {
        self.state
            .try_borrow()
            .map(|st| st.desc)
            .unwrap_or_default()
    }

    /// Sets inverse projection/view for world anchoring and geometric outlines.
    /// A singular matrix keeps the previous camera.
    pub fn set_camera(&self, view: &Mat4, proj: &Mat4) {
        if let (Some(iv), Some(ip)) = (view.inverse(), proj.inverse())
            && let Ok(mut state) = self.state.try_borrow_mut()
        {
            state.camera = [ip, iv];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn handle_refusal_keeps_settings_and_singular_camera_keeps_inverses() {
        let handle = CelHandle {
            state: Rc::new(RefCell::new(CelState {
                desc: CelSettings::default(),
                camera: [Mat4::IDENTITY; 2],
            })),
        };
        let good = CelSettings {
            bands: 7,
            world_scale: 64.0,
            ..Default::default()
        };
        handle.set(good).unwrap();
        assert_eq!(handle.clone().get(), good);
        assert_eq!(
            handle
                .set(CelSettings { bands: 1, ..good })
                .unwrap_err()
                .code(),
            "cel.invalid"
        );
        assert_eq!(handle.get(), good);
        let view = Mat4::from_translation(raa_math::Vec3::new(1.0, 2.0, 3.0));
        let proj = Mat4::perspective_rh_zo(1.0, 1.5, 0.1, 100.0);
        handle.set_camera(&view, &proj);
        let before = handle.state.borrow().camera;
        assert_eq!(before, [proj.inverse().unwrap(), view.inverse().unwrap()]);
        handle.set_camera(&Mat4::from_scale(raa_math::Vec3::new(0.0, 1.0, 1.0)), &proj);
        assert_eq!(handle.state.borrow().camera, before);
    }
}
