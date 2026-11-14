//! The `gi.history` debug-view recording (S38, S39): one `raa_gi_debug_view` compute dispatch
//! from `hdr.src` into `hdr.dst`, then the HDR chain swap. Crate-private; the shader module and
//! its group-2 entries come from `shaders`.

use raa_core::GpuContext;
use raa_pipeline::RecordCtx;

use crate::shaders;

/// The debug-view pipeline of `gi.history`.
///
/// Lifecycle: as contract 'gi-frame-lifecycle' states, `DebugRenderer::record` runs only inside
/// the gi.history record (order 64, `Pipeline::record_single`), after every pass's prepare, and
/// creates only a per-frame bind group.
pub(crate) struct DebugRenderer {
    pipeline: wgpu::ComputePipeline,
    group2_layout: wgpu::BindGroupLayout,
}

impl DebugRenderer {
    /// Builds the `raa_gi_debug_view` compute pipeline (8x8) with the layout [None,
    /// `group1_layout`, the debug group 2 of `debug_group2_entries()`].
    ///
    /// Handoff: `DebugRenderer::new` builds the pipeline today and `DebugRenderer::record`
    /// binds a per-frame group 2 and dispatches it. Card 'raa-gi: feature.rs share 2, passes,
    /// install and prepare_frame' calls `DebugRenderer::new` at install and
    /// `DebugRenderer::record` from the gi.history record with the `GiFrameGpu` buffer, the
    /// `GiVolume` voxels buffer and the compute group 1 bind group. The shader builders it
    /// reads are the upstream Handoff on crates/raa-gi/src/shaders.rs:225
    /// `debug_view_module_wgsl`.
    pub(crate) fn new(gpu: &GpuContext, group1_layout: &wgpu::BindGroupLayout) -> DebugRenderer {
        let device = gpu.device();
        let group2_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa-gi.debug"),
            entries: &shaders::debug_group2_entries(),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raa-gi.debug"),
            bind_group_layouts: &[None, Some(group1_layout), Some(&group2_layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa-gi.debug"),
            source: wgpu::ShaderSource::Wgsl(shaders::debug_view_module_wgsl().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("raa-gi.debug"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("raa_gi_debug_view"),
            compilation_options: Default::default(),
            cache: None,
        });
        DebugRenderer {
            pipeline,
            group2_layout,
        }
    }

    /// Records the debug view of this frame: group 2 is 0 `frame`, 1 `ctx.targets.depth()`,
    /// 2 `ctx.targets.hdr().src()`, 3 `voxels`, 4 `ctx.targets.hdr().dst()`, since the HDR chain
    /// swaps every pass. One 8x8 thread per pixel of `ctx.size`; then the chain swaps.
    pub(crate) fn record(
        &self,
        ctx: &mut RecordCtx<'_>,
        frame: &wgpu::Buffer,
        voxels: &wgpu::Buffer,
        group1: &wgpu::BindGroup,
    ) {
        let targets = ctx.targets;
        let bind = ctx
            .gpu
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("raa-gi.debug"),
                layout: &self.group2_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: frame.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(targets.depth()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(targets.hdr().src()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: voxels.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(targets.hdr().dst()),
                    },
                ],
            });
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("raa.gi.debug_view"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(1, group1, &[]);
        pass.set_bind_group(2, &bind, &[]);
        pass.dispatch_workgroups(ctx.size.0.div_ceil(8), ctx.size.1.div_ceil(8), 1);
        drop(pass);
        targets.hdr().swap();
    }
}
