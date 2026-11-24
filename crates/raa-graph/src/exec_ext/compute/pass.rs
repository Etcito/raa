//! Compute pass lifecycle: the pipeline clears the resource table and begins graph frames;
//! for each enabled pass it resizes when targets change, runs before_prepare, then prepare;
//! before_record, record and after_record run per pass, followed by graph frame end.

use super::{GraphFormat, PlanNode, PortType, SizeSpec, plan};
use crate::format::{ComputeExtent, ExtentBasis};
use raa_core::{GpuContext, RaaError};
use raa_pipeline::{
    FeaturePass, ImageDesc, ImageDim, PrepareCtx, RecordCtx, ResourceTable, Targets,
};
use std::{cell::RefCell, rc::Rc};

const FRAME_PACK: &str = r#"
@group(0) @binding(0) var<uniform> lit: array<vec4<u32>, 48>;
@group(0) @binding(1) var<storage, read_write> frame: array<vec4<u32>, 23>;
@compute @workgroup_size(1) fn main() {
  for (var i = 0u; i < 4u; i++) {
    frame[2u+i] = lit[4u+i];
    frame[6u+i] = lit[8u+i];
    frame[10u+i] = lit[12u+i];
    frame[14u+i] = lit[16u+i];
    frame[18u+i] = lit[32u+i];
  }
  frame[22] = vec4<u32>(lit[42].z * frame[22].y, 0u, 0u, 0u);
}
"#;

struct Output {
    key: String,
    format: GraphFormat,
    size: Option<SizeSpec>,
    extent: Option<ComputeExtent>,
    image: Option<(wgpu::Texture, wgpu::TextureView, ImageDesc)>,
}

pub(super) struct ComputeNodeState {
    pub module: String,
    pub node: String,
    pub ran: bool,
    pub history_pairs_valid: bool,
    inputs: Vec<plan::ComputeInput>,
    outputs: Vec<Output>,
    pipeline: wgpu::ComputePipeline,
    input_layout: wgpu::BindGroupLayout,
    output_layout: wgpu::BindGroupLayout,
    common: wgpu::BindGroup,
    frame: wgpu::Buffer,
    params: wgpu::Buffer,
    pack_pipeline: wgpu::ComputePipeline,
    pack_group: wgpu::BindGroup,
    input_cache: Option<(Vec<Binding>, wgpu::BindGroup)>,
    output_group: Option<wgpu::BindGroup>,
    surface: wgpu::TextureFormat,
}

#[derive(Clone, PartialEq)]
enum Binding {
    Image(wgpu::TextureView),
    Buffer(wgpu::Buffer, u64),
}

/// Per-node compute work. The `rg-compute-exec` LIFECYCLE runs resize on target
/// generation changes, then graph-frame before_prepare and prepare; recording runs
/// before_record, record and after_record for each enabled pass, followed by end.
pub(super) struct ComputeNodePass {
    entry: String,
    pub state: Rc<RefCell<ComputeNodeState>>,
}

fn layout(device: &wgpu::Device, types: &[wgpu::BindingType]) -> wgpu::BindGroupLayout {
    let entries: Vec<_> = types
        .iter()
        .enumerate()
        .map(|(i, ty)| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: *ty,
            count: None,
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &entries,
    })
}
fn buffer_type(ty: wgpu::BufferBindingType) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty,
        has_dynamic_offset: false,
        min_binding_size: None,
    }
}
fn build_pipeline(
    device: &wgpu::Device,
    source: &str,
    layouts: &[&wgpu::BindGroupLayout],
    entry: &str,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layouts: Vec<_> = layouts.iter().map(|l| Some(*l)).collect();
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &layouts,
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}
fn buffer(device: &wgpu::Device, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage,
        mapped_at_creation: false,
    })
}
fn group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    resources: &[wgpu::BindingResource<'_>],
) -> wgpu::BindGroup {
    let entries: Vec<_> = resources
        .iter()
        .enumerate()
        .map(|(i, resource)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: resource.clone(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &entries,
    })
}
impl ComputeNodePass {
    pub fn new(
        gpu: &GpuContext,
        node: &PlanNode,
        module: String,
        inputs: Vec<plan::ComputeInput>,
        lit: &wgpu::Buffer,
        surface: wgpu::TextureFormat,
    ) -> Self {
        let device = gpu.device();
        let uniform = buffer_type(wgpu::BufferBindingType::Uniform);
        let storage = buffer_type(wgpu::BufferBindingType::Storage { read_only: true });
        let common_layout = layout(
            device,
            &[
                uniform,
                uniform,
                wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            ],
        );
        let input_types: Vec<_> = inputs
            .iter()
            .map(|input| match input.ty {
                PortType::Buffer => storage,
                _ => wgpu::BindingType::Texture {
                    sample_type: if input.ty == PortType::Depth2d {
                        wgpu::TextureSampleType::Depth
                    } else {
                        wgpu::TextureSampleType::Float {
                            filterable: input.filterable,
                        }
                    },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
            })
            .collect();
        let input_layout = layout(device, &input_types);
        let outputs: Vec<_> = node
            .compute
            .as_ref()
            .into_iter()
            .flat_map(|c| &c.outputs)
            .filter_map(|o| {
                Some(Output {
                    key: o.key.clone(),
                    format: o.format.clone()?,
                    size: o.size.clone(),
                    extent: o.extent.clone(),
                    image: None,
                })
            })
            .collect();
        let output_types: Vec<_> = outputs
            .iter()
            .map(|o| wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: o.format.to_wgpu(surface),
                view_dimension: wgpu::TextureViewDimension::D2,
            })
            .collect();
        let output_layout = layout(device, &output_types);
        let pipeline = build_pipeline(
            device,
            &module,
            &[&common_layout, &input_layout, &output_layout],
            "rg_entry",
        );
        let params_data = plan::pack_params(&node.params);
        let params = buffer(
            device,
            params_data.len() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        gpu.queue().write_buffer(&params, 0, &params_data);
        let frame = buffer(
            device,
            plan::RG_FRAME_BYTES,
            wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let common = group(
            device,
            &common_layout,
            &[
                params.as_entire_binding(),
                frame.as_entire_binding(),
                wgpu::BindingResource::Sampler(&sampler),
            ],
        );
        let pack_layout = layout(
            device,
            &[
                uniform,
                buffer_type(wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        );
        let pack_pipeline = build_pipeline(device, FRAME_PACK, &[&pack_layout], "main");
        let pack_group = group(
            device,
            &pack_layout,
            &[lit.as_entire_binding(), frame.as_entire_binding()],
        );
        let state = ComputeNodeState {
            module,
            node: node.id.clone(),
            ran: false,
            history_pairs_valid: false,
            inputs,
            outputs,
            pipeline,
            input_layout,
            output_layout,
            common,
            frame,
            params,
            pack_pipeline,
            pack_group,
            input_cache: None,
            output_group: None,
            surface,
        };
        Self {
            entry: plan::compute_entry(&node.id),
            state: Rc::new(RefCell::new(state)),
        }
    }
}
impl ComputeNodeState {
    pub fn update(&mut self, gpu: &GpuContext, node: &PlanNode, inputs: Vec<plan::ComputeInput>) {
        gpu.queue()
            .write_buffer(&self.params, 0, &plan::pack_params(&node.params));
        self.inputs = inputs;
        if let Some(compute) = &node.compute {
            for (output, declaration) in self.outputs.iter_mut().zip(&compute.outputs) {
                output.size = declaration.size.clone();
                output.extent = declaration.extent.clone();
            }
        }
    }
    fn has_parent_extent(&self) -> bool {
        self.outputs.iter().any(|output| {
            matches!(
                output.extent.as_ref().map(|e| &e.basis),
                Some(ExtentBasis::Parent(_))
            )
        })
    }

    fn prepare_outputs(
        &mut self,
        gpu: &GpuContext,
        resources: &ResourceTable,
        entry: &str,
        scene: (u32, u32),
        display: (u32, u32),
    ) -> Result<(), RaaError> {
        let surface = self.surface;
        let limit = gpu.device().limits().max_texture_dimension_2d;
        for output in &mut self.outputs {
            let size = plan::resolve_output_size(
                output.size.as_ref(),
                output.extent.as_ref(),
                scene,
                display,
                |key| {
                    let source = self
                        .inputs
                        .iter()
                        .find(|input| input.key == key)?
                        .source
                        .as_ref()?;
                    let image = resources
                        .input(entry, key)
                        .or_else(|| resources.get(&source.0, &source.1))?;
                    Some((image.desc.width, image.desc.height))
                },
            );
            let Some(size) = size.filter(|&(w, h)| w != 0 && h != 0) else {
                if output.image.take().is_some() {
                    self.output_group = None;
                }
                continue;
            };
            if size.0 > limit || size.1 > limit {
                return Err(RaaError::Resource(format!(
                    "{}.{} extent {}×{} exceeds the device limit {limit}",
                    entry, output.key, size.0, size.1
                )));
            }
            if output
                .image
                .as_ref()
                .is_none_or(|(_, _, d)| (d.width, d.height) != size)
            {
                let format = output.format.to_wgpu(surface);
                let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width: size.0,
                        height: size.1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                let view = texture.create_view(&Default::default());
                output.image = Some((
                    texture,
                    view,
                    ImageDesc::new(ImageDim::D2, format, size.0, size.1, 1, 1),
                ));
                self.output_group = None;
            }
            if let Some((_, view, desc)) = &output.image {
                resources.publish(entry, &output.key, view, *desc);
            }
        }
        Ok(())
    }

    fn dispatch_size(&self) -> Option<(u32, u32)> {
        plan::dispatch_size(self.outputs.iter().map(|output| {
            let (texture, _, _) = output.image.as_ref()?;
            Some((texture.width(), texture.height()))
        }))
    }
    pub fn stats(&self) -> super::NodeStats {
        let resources = self
            .outputs
            .iter()
            .filter_map(|output| {
                let (_, _, desc) = output.image.as_ref()?;
                Some(super::ResourceStat {
                    port: output.key.clone(),
                    width: desc.width,
                    height: desc.height,
                    layers: 1,
                    mips: 1,
                    format: output.format.name().to_owned(),
                    bytes: u64::from(desc.width)
                        * u64::from(desc.height)
                        * u64::from(desc.format.block_copy_size(None).unwrap_or(0)),
                })
            })
            .collect();
        super::NodeStats {
            node: self.node.clone(),
            type_id: "compute.wgsl".into(),
            ran: self.ran,
            resources,
        }
    }
}
impl FeaturePass for ComputeNodePass {
    fn name(&self) -> &str {
        &self.entry
    }
    fn order(&self) -> u32 {
        plan::COMPUTE_ORDER
    }
    fn resize(&mut self, _: &GpuContext, _: &Targets) {
        let mut state = self.state.borrow_mut();
        for output in &mut state.outputs {
            output.image = None;
        }
        state.output_group = None;
    }
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let mut state = self.state.borrow_mut();
        state.ran = false;
        if state.has_parent_extent() {
            // Some producers publish only after recording. Resolve the live input then.
            return Ok(());
        }
        state.prepare_outputs(
            ctx.gpu,
            ctx.resources,
            &self.entry,
            ctx.size,
            ctx.targets.display_rect().size,
        )
    }
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let mut state = self.state.borrow_mut();
        state.ran = false;
        if state.has_parent_extent() {
            state.prepare_outputs(
                ctx.gpu,
                ctx.resources,
                &self.entry,
                ctx.size,
                ctx.targets.display_rect().size,
            )?;
        }
        let Some(size) = state.dispatch_size() else {
            return Ok(());
        };
        let bindings: Option<Vec<_>> = state
            .inputs
            .iter()
            .map(|input| {
                if let Some(image) = ctx.resources.input(&self.entry, &input.key) {
                    return Some(Binding::Image(image.view));
                }
                let (entry, port) = input.source.as_ref()?;
                if input.ty == PortType::Buffer {
                    ctx.resources
                        .buffer(entry, port)
                        .map(|(b, s)| Binding::Buffer(b, s))
                } else {
                    ctx.resources
                        .get(entry, port)
                        .map(|image| Binding::Image(image.view))
                }
            })
            .collect();
        let Some(bindings) = bindings else {
            return Ok(());
        };
        if state
            .input_cache
            .as_ref()
            .is_none_or(|(old, _)| *old != bindings)
        {
            let resources: Vec<_> = bindings
                .iter()
                .map(|b| match b {
                    Binding::Image(view) => wgpu::BindingResource::TextureView(view),
                    Binding::Buffer(buffer, _) => buffer.as_entire_binding(),
                })
                .collect();
            let group = group(ctx.gpu.device(), &state.input_layout, &resources);
            state.input_cache = Some((bindings, group));
        }
        if state.output_group.is_none() {
            let resources: Option<Vec<_>> = state
                .outputs
                .iter()
                .map(|o| {
                    o.image
                        .as_ref()
                        .map(|(_, v, _)| wgpu::BindingResource::TextureView(v))
                })
                .collect();
            let Some(resources) = resources else {
                return Ok(());
            };
            state.output_group = Some(group(ctx.gpu.device(), &state.output_layout, &resources));
        }
        ctx.gpu.queue().write_buffer(
            &state.frame,
            0,
            &plan::frame_rows(size, ctx.frame, ctx.time_s),
        );
        let valid = u32::from(!state.inputs.iter().any(|i| i.history) || state.history_pairs_valid);
        let mut flags = [0u8; 16];
        flags[4..8].copy_from_slice(&valid.to_le_bytes());
        ctx.gpu.queue().write_buffer(&state.frame, 352, &flags);
        {
            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("compute frame pack"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(&state.pack_pipeline);
            pass.set_bind_group(0, &state.pack_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = ctx
                .encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&self.entry),
                    timestamp_writes: None,
                });
            pass.set_pipeline(&state.pipeline);
            pass.set_bind_group(0, &state.common, &[]);
            if let Some((_, group)) = &state.input_cache {
                pass.set_bind_group(1, group, &[]);
            }
            if let Some(group) = &state.output_group {
                pass.set_bind_group(2, group, &[]);
            }
            pass.dispatch_workgroups(size.0.div_ceil(8), size.1.div_ceil(8), 1);
        }
        state.ran = true;
        Ok(())
    }
}
