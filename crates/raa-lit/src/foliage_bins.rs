//! Counted 16-by-16 tile bins for generation-owned optimizer work.
//! The m7-foliage-resources LIFECYCLE retains submitted resources until completion;
//! this stage does not publish or release a foliage generation.

use raa_core::{GpuContext, RaaError};
use wgpu::util::DeviceExt;

use crate::foliage_resources::FoliageCompute;

const SHADER: &str = include_str!("wgsl/foliage_bins.wgsl");

fn invalid(message: &str) -> RaaError {
    RaaError::FrameGraph {
        code: "foliage.invalid",
        message: message.into(),
    }
}

/// Number of 64-thread workgroups needed for `items`.
pub fn dispatch_size(items: u32) -> u32 {
    items.div_ceil(64)
}

/// Number of tiles across a square reference image.
pub fn tiles_per_axis(resolution: u32) -> u32 {
    resolution.div_ceil(16)
}

/// Inclusive tile bounds for a clipped pixel-space rectangle.
pub fn rectangle_tile_bounds(
    resolution: u32,
    min: [f32; 2],
    max: [f32; 2],
) -> Option<([u32; 2], [u32; 2])> {
    if resolution == 0
        || min.iter().chain(max.iter()).any(|v| !v.is_finite())
        || min[0] > max[0]
        || min[1] > max[1]
        || max[0] < 0.0
        || max[1] < 0.0
        || min[0] > resolution as f32
        || min[1] > resolution as f32
    {
        return None;
    }
    let last = tiles_per_axis(resolution) - 1;
    let tile = |v: f32| ((v.clamp(0.0, resolution as f32) / 16.0).floor() as u32).min(last);
    Some(([tile(min[0]), tile(min[1])], [tile(max[0]), tile(max[1])]))
}

/// Count words above capacity signal overflow; consumers must wait for a retry.
pub fn needs_bin_retry(counts: &[u32], capacity: u32) -> bool {
    counts.iter().any(|&count| count > capacity)
}

/// Handoff: 'Rasterize differentiable foliage coverage and normals' calls `encode_bins`;
/// 'Run differentiable foliage optimization' calls `encode_bins`. It resets tile counts
/// and records conservative projected-box binning; counts above capacity signal retry.
pub fn encode_bins(
    gpu: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    work: &FoliageCompute,
) -> Result<(), RaaError> {
    gpu.ensure_live()?;
    let side = tiles_per_axis(work.resolution);
    let tiles = side
        .checked_mul(side)
        .and_then(|n| n.checked_mul(work.views))
        .ok_or_else(|| invalid("bin tile count overflow"))?;
    let items = work
        .count
        .checked_mul(work.views)
        .ok_or_else(|| invalid("bin dispatch overflow"))?;
    let limit = gpu.device().limits().max_compute_workgroups_per_dimension;
    if work.resolution == 0
        || work.views == 0
        || work.views > 128
        || work.references.views.len() != work.views as usize
        || dispatch_size(tiles) > limit
        || dispatch_size(items) > limit
    {
        return Err(invalid("invalid bin dimensions or dispatch"));
    }
    raa_shader::Composer::validate(SHADER).map_err(|e| RaaError::FrameGraph {
        code: e.code,
        message: e.message,
    })?;
    let device = gpu.device();
    let mut table = vec![0u8; 128 * 80];
    for (i, (matrix, near_far)) in work.references.views.iter().enumerate() {
        for (j, value) in matrix.iter().chain(near_far.iter()).enumerate() {
            if !value.is_finite() {
                return Err(invalid("non-finite view table"));
            }
            table[i * 80 + j * 4..i * 80 + j * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    let views = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("foliage.bin.views"),
        contents: &table,
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("foliage.bins"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let pipeline = |entry| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: None,
            module: &shader,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let reset = pipeline("reset_bins");
    let fill = pipeline("fill_bins");
    let reset_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("foliage.bins.reset"),
        layout: &reset.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: work.params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: work.bins.as_entire_binding(),
            },
        ],
    });
    let fill_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("foliage.bins.fill"),
        layout: &fill.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: work.params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: work.triangles.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: work.bins.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: views.as_entire_binding(),
            },
        ],
    });
    for (pipeline, group, groups) in [
        (&reset, &reset_group, dispatch_size(tiles)),
        (&fill, &fill_group, dispatch_size(items)),
    ] {
        if groups == 0 {
            continue;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("foliage.bins"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    Ok(())
}
