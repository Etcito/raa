//! GPU resources of the `taa` pass: [`TaaPipelines`] (created once at install) and
//! [`TaaTargets`] (one target generation, recreated on resize).
//!
//! Group 0 follows contract `taa-bindings`: the frame and params uniforms, the scene, depth,
//! motion and history textures, a filtering sampler and the two write-only history/output
//! storage textures. The resolve reads hdr.src() and the history, so there is one group per
//! HDR orientation (A and B) and per history parity; record picks the one matching
//! `hdr.src_texture()` and the write index.

use std::num::NonZeroU64;

use raa_core::GpuContext;
use raa_core::wgpu;
use raa_pipeline::{HDR_FORMAT, Targets};

use crate::history::HistoryPair;
use crate::shaders::{
    TAA_ENTRY, TAA_PARAMS_SIZE, TAA_WORKGROUP, TEMPORAL_FRAME_SIZE, taa_module_source,
};
use crate::tunables::taa_constants;

fn entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::BindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty,
        count: None,
    }
}

const fn uniform(size: u64) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: NonZeroU64::new(size),
    }
}

fn texture_type(sample_type: wgpu::TextureSampleType) -> wgpu::BindingType {
    wgpu::BindingType::Texture {
        sample_type,
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    }
}

const fn storage_type(format: wgpu::TextureFormat) -> wgpu::BindingType {
    wgpu::BindingType::StorageTexture {
        access: wgpu::StorageTextureAccess::WriteOnly,
        format,
        view_dimension: wgpu::TextureViewDimension::D2,
    }
}

const FLOAT: wgpu::TextureSampleType = wgpu::TextureSampleType::Float { filterable: true };
const UNFILTERABLE: wgpu::TextureSampleType = wgpu::TextureSampleType::Float { filterable: false };
const DEPTH: wgpu::TextureSampleType = wgpu::TextureSampleType::Depth;
const FILTERING: wgpu::BindingType =
    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering);

fn layout(
    device: &wgpu::Device,
    label: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries,
    })
}

fn view_entry(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

/// The nine group 0 entries of `cs_taa`, in binding order: 0 and 1 uniforms, 2/4/5 textures,
/// 3 depth, 6 a filtering sampler and 7/8 write-only `Rgba16Float` storage textures.
pub(crate) fn taa_layout_entries() -> [wgpu::BindGroupLayoutEntry; 9] {
    let cs = wgpu::ShaderStages::COMPUTE;
    [
        entry(0, cs, uniform(TEMPORAL_FRAME_SIZE)),
        entry(1, cs, uniform(TAA_PARAMS_SIZE)),
        entry(2, cs, texture_type(UNFILTERABLE)),
        entry(3, cs, texture_type(DEPTH)),
        entry(4, cs, texture_type(UNFILTERABLE)),
        entry(5, cs, texture_type(FLOAT)),
        entry(6, cs, FILTERING),
        entry(7, cs, storage_type(HDR_FORMAT)),
        entry(8, cs, storage_type(HDR_FORMAT)),
    ]
}

/// The one group 0 layout, the resolve compute pipeline, the linear clamp sampler and the
/// `RaaTaaParams` uniform, created once at install.
pub(crate) struct TaaPipelines {
    pub(crate) layout: wgpu::BindGroupLayout,
    pub(crate) pipeline: wgpu::ComputePipeline,
    module: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,
    pub(crate) sampler: wgpu::Sampler,
    /// `RaaTaaParams` (TAA_PARAMS_SIZE bytes), written each frame by the pass.
    pub(crate) params: wgpu::Buffer,
}

fn build_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    values: &[f64; 6],
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("raa-taa"),
        layout: Some(layout),
        module,
        entry_point: Some(TAA_ENTRY),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &taa_constants(values),
            ..Default::default()
        },
        cache: None,
    })
}

impl TaaPipelines {
    /// Rebuilds `pipeline` alone with the override constants of `values`.
    pub(crate) fn rebuild(&mut self, gpu: &GpuContext, values: &[f64; 6]) {
        self.pipeline = build_pipeline(gpu.device(), &self.pipeline_layout, &self.module, values);
    }

    pub(crate) fn new(gpu: &GpuContext, values: &[f64; 6]) -> TaaPipelines {
        let device = gpu.device();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa-taa"),
            source: wgpu::ShaderSource::Wgsl(taa_module_source().into()),
        });
        let layout = layout(device, "raa-taa", &taa_layout_entries());
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raa-taa"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = build_pipeline(device, &pipeline_layout, &module, values);
        TaaPipelines {
            layout,
            pipeline,
            module,
            pipeline_layout,
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("raa-taa.sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                ..Default::default()
            }),
            params: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("raa-taa.params"),
                size: TAA_PARAMS_SIZE,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        }
    }
}

/// The four bind groups of one target generation: two HDR orientations by two history
/// parities, with the size they were built at.
///
/// Handoff: `raa-taa: taa feature pass, install and handle` calls `TaaTargets::new` with the
/// frame buffer and the `HistoryPair`, `TaaTargets::group` with `hdr.src_texture()` and the
/// write index, `TaaTargets::size` and `dispatch_size` for the dispatch, and
/// `TaaPipelines::new` at install. Today `TaaTargets::new` builds the four groups from the
/// targets' depth and motion views, the history pair's two parities and the whole frame and
/// params buffers; `TaaTargets::group` returns the group whose read HDR texture equals `src`
/// at parity `write_index & 1`, or None after the chain was recreated.
pub(crate) struct TaaTargets {
    size: (u32, u32),
    /// The read HDR texture of each orientation (0 = source, 1 = destination).
    read: [wgpu::Texture; 2],
    /// `[orientation][parity]`.
    groups: [[wgpu::BindGroup; 2]; 2],
}

impl TaaTargets {
    pub(crate) fn new(
        gpu: &GpuContext,
        pipelines: &TaaPipelines,
        targets: &Targets,
        frame: &wgpu::Buffer,
        history: &HistoryPair,
    ) -> TaaTargets {
        let device = gpu.device();
        let hdr = targets.hdr();
        let group = |label: &str,
                     read: &wgpu::TextureView,
                     written: &wgpu::TextureView,
                     read_history: usize,
                     write_history: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &pipelines.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: frame.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: pipelines.params.as_entire_binding(),
                    },
                    view_entry(2, read),
                    view_entry(3, targets.depth()),
                    view_entry(4, targets.motion()),
                    view_entry(5, history.view(read_history)),
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(&pipelines.sampler),
                    },
                    view_entry(7, written),
                    view_entry(8, history.view(write_history)),
                ],
            })
        };
        let groups = [
            [
                group("raa-taa.src.0", hdr.src(), hdr.dst(), 1, 0),
                group("raa-taa.src.1", hdr.src(), hdr.dst(), 0, 1),
            ],
            [
                group("raa-taa.dst.0", hdr.dst(), hdr.src(), 1, 0),
                group("raa-taa.dst.1", hdr.dst(), hdr.src(), 0, 1),
            ],
        ];
        TaaTargets {
            size: targets.size(),
            read: [hdr.src_texture().clone(), hdr.dst_texture().clone()],
            groups,
        }
    }

    /// The group whose read HDR texture equals `src` at parity `write_index & 1`, or None
    /// when the chain was recreated since the last resize.
    pub(crate) fn group(
        &self,
        src: &wgpu::Texture,
        write_index: usize,
    ) -> Option<&wgpu::BindGroup> {
        let orientation = self.read.iter().position(|texture| texture == src)?;
        Some(&self.groups[orientation][write_index & 1])
    }

    /// The surface size the groups were built at.
    pub(crate) fn size(&self) -> (u32, u32) {
        self.size
    }
}

/// Workgroups along each axis: `ceil(size / TAA_WORKGROUP)`, at least 1.
pub(crate) fn dispatch_size(size: (u32, u32)) -> (u32, u32) {
    (
        size.0.div_ceil(TAA_WORKGROUP).max(1),
        size.1.div_ceil(TAA_WORKGROUP).max(1),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn float(filterable: bool) -> wgpu::BindingType {
        texture_type(wgpu::TextureSampleType::Float { filterable })
    }

    #[test]
    fn layout_entries_are_the_taa_bindings() {
        let entries = taa_layout_entries();
        assert_eq!(entries.len(), 9);
        for (binding, entry) in entries.iter().enumerate() {
            assert_eq!(entry.binding, binding as u32);
            assert_eq!(entry.visibility, wgpu::ShaderStages::COMPUTE);
            assert_eq!(entry.count, None);
        }
        assert_eq!(TEMPORAL_FRAME_SIZE, 768);
        assert_eq!(TAA_PARAMS_SIZE, 16);
        assert_eq!(entries[0].ty, uniform(768));
        assert_eq!(entries[1].ty, uniform(16));
        assert_eq!(entries[2].ty, float(false));
        assert_eq!(entries[3].ty, texture_type(DEPTH));
        assert_eq!(entries[4].ty, float(false));
        assert_eq!(entries[5].ty, float(true));
        assert_eq!(entries[6].ty, FILTERING);
        assert_eq!(entries[7].ty, storage_type(HDR_FORMAT));
        assert_eq!(entries[8].ty, storage_type(HDR_FORMAT));
    }

    #[test]
    fn dispatch_rounds_up_to_at_least_one_workgroup() {
        assert_eq!(dispatch_size((1280, 720)), (160, 90));
        assert_eq!(dispatch_size((17, 9)), (3, 2));
        assert_eq!(dispatch_size((0, 0)), (1, 1));
    }
}
