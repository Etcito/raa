//! Optional display-space visualization of signed UV motion.
//!
//! Zero is dark grey (0.1). Positive X adds red, negative X adds cyan;
//! positive Y (down) adds green, negative Y adds magenta. Each component
//! reaches full strength at 0.05 UV units per frame and is clamped there.

use raa_core::{GpuContext, RaaError};
use raa_pipeline::{FeaturePass, Pipeline, RecordCtx, Targets};
use std::{cell::RefCell, collections::HashMap, rc::Rc};

const SHADER: &str = r#"
@group(0) @binding(0) var motion: texture_2d<f32>;
@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(p[i], 0.0, 1.0);
}
@fragment fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let v = clamp(textureLoad(motion, vec2<i32>(pos.xy), 0).xy * 20.0, vec2<f32>(-1.0), vec2<f32>(1.0));
    let positive = max(v, vec2<f32>(0.0));
    let negative = max(-v, vec2<f32>(0.0));
    let rgb = vec3<f32>(positive.x + negative.y, positive.y + negative.x, negative.x + negative.y);
    return vec4<f32>(min(vec3<f32>(0.1) + 0.9 * rgb, vec3<f32>(1.0)), 1.0);
}
"#;

/// Installs the disabled `motion.display` overlay. Use the ordinary pass controls to enable it.
pub fn install_motion_display(
    gpu: &GpuContext,
    pipeline: &Rc<RefCell<Pipeline>>,
) -> Result<(), RaaError> {
    let mut pipeline = pipeline
        .try_borrow_mut()
        .map_err(|_| RaaError::FrameGraph {
            code: "pipeline.busy",
            message: "the pipeline is borrowed during install".into(),
        })?;
    pipeline.add(Box::new(MotionDisplay::new(gpu)))?;
    pipeline.set_enabled("motion.display", false)
}

struct MotionDisplay {
    shader: wgpu::ShaderModule,
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    bound: Vec<(wgpu::Texture, wgpu::BindGroup)>,
}

impl MotionDisplay {
    fn new(gpu: &GpuContext) -> Self {
        let device = gpu.device();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa.motion.display"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa.motion.display"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raa.motion.display"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        Self {
            shader,
            layout,
            pipeline_layout,
            pipelines: HashMap::new(),
            bound: Vec::new(),
        }
    }
}

impl FeaturePass for MotionDisplay {
    fn name(&self) -> &str {
        "motion.display"
    }
    fn order(&self) -> u32 {
        201
    }
    fn output_overlay(&self) -> bool {
        true
    }
    fn resize(&mut self, _: &GpuContext, _: &Targets) {
        self.bound.clear();
    }
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let device = ctx.gpu.device();
        let pipeline = self
            .pipelines
            .entry(ctx.backbuffer_format)
            .or_insert_with(|| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("raa.motion.display"),
                    layout: Some(&self.pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &self.shader,
                        entry_point: Some("vs_main"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    primitive: Default::default(),
                    depth_stencil: None,
                    multisample: Default::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &self.shader,
                        entry_point: Some("fs_main"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: ctx.backbuffer_format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
            });
        let texture = ctx.targets.motion_texture();
        let index = match self.bound.iter().position(|(t, _)| t == texture) {
            Some(i) => i,
            None => {
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("raa.motion.display"),
                    layout: &self.layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(ctx.targets.motion()),
                    }],
                });
                self.bound.push((texture.clone(), group));
                self.bound.len() - 1
            }
        };
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("raa.motion.display"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.backbuffer,
                resolve_target: None,
                depth_slice: None,
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
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.bound[index].1, &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}
