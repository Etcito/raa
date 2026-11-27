//! Differentiable foliage gradients (M7, contracts `m7-compute` and `m7-soft-coverage`).
//!
//! [`encode_gradients`] reduces one triangle-owned gradient record per micro-triangle from
//! the counted 16-by-16 tile bins, the raster stage's published pixel records and the
//! captured reference views. The objective is exactly the weighted silhouette, the
//! 1-cosine normal and the squared-edge regularization; the analytic derivatives flow
//! through the soft coverage, the projection, the depth visibility and the normal
//! normalization to all three positions, the independent triangle normal and the triangle
//! opacity that the optimizer anneals toward `opacityTarget`. The triangle owns its
//! reduction in registers, so no unsupported float atomic is ever needed.
//!
//! The submitted WGSL is [`gradients_wgsl`]: the one shared `foliage_soft_coverage`
//! section of `foliage_raster.wgsl` followed by [`FOLIAGE_GRADIENTS_WGSL`]. The gradient
//! formula therefore evaluates the same sigmoid as the raster instead of transcribing it.

use raa_core::{GpuContext, RaaError};
use wgpu::util::DeviceExt;

use crate::foliage_bins::dispatch_size;
use crate::foliage_resources::{
    FOLIAGE_PARAMS_SIZE, FOLIAGE_VIEW_TABLE_SIZE, FoliageCompute, FoliageParams, FoliageReferences,
};

/// The gradient-stage WGSL section; [`gradients_wgsl`] prepends the shared coverage helper.
pub const FOLIAGE_GRADIENTS_WGSL: &str = include_str!("wgsl/foliage_gradients.wgsl");
/// Bytes of one gradient record: three position lanes, normal/opacity, and one zero lane.
pub const GRADIENT_RECORD_STRIDE: usize = 80;

fn invalid(message: impl Into<String>) -> RaaError {
    RaaError::FrameGraph {
        code: "foliage.invalid",
        message: message.into(),
    }
}

fn bake_failed(message: impl Into<String>) -> RaaError {
    RaaError::FrameGraph {
        code: "foliage.bake_failed",
        message: message.into(),
    }
}

/// The weighted silhouette term `0.5 * (coverage - reference_coverage)^2`.
pub fn silhouette_loss(coverage: f32, reference_coverage: f32) -> f32 {
    let error = coverage - reference_coverage;
    0.5 * error * error
}

/// `d(silhouette_loss)/d(coverage)`, the factor the gradient stage chains into the
/// clamped coverage sum.
pub fn silhouette_gradient(coverage: f32, reference_coverage: f32) -> f32 {
    coverage - reference_coverage
}

fn normalize3(v: [f32; 3]) -> Option<[f32; 3]> {
    let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    (length > 1e-8).then(|| v.map(|value| value / length))
}

/// The 1-cosine normal term `1 - dot(normalize(predicted), normalize(reference))`.
/// Identical directions give 0 and orthogonal directions give 1; a zero direction has no
/// cosine and contributes no loss, like a reference background pixel.
pub fn normal_loss(predicted: [f32; 3], reference: [f32; 3]) -> f32 {
    match (normalize3(predicted), normalize3(reference)) {
        (Some(p), Some(r)) => {
            1.0 - p
                .iter()
                .zip(r)
                .map(|(predicted, reference)| predicted * reference)
                .sum::<f32>()
        }
        _ => 0.0,
    }
}

/// `d(foliage_soft_coverage)/d(edge_distance)`: `s * (1 - s) / temperature`. At the edge
/// (`distance` 0) with temperature 1 it is exactly `0.25`.
pub fn soft_coverage_derivative(edge_distance: f32, temperature: f32) -> f32 {
    let coverage = crate::foliage_raster::soft_coverage(edge_distance, temperature);
    coverage * (1.0 - coverage) / temperature
}

/// The squared edge length `|a - b|^2` of the regularization term.
pub fn edge_loss(a: [f32; 3], b: [f32; 3]) -> f32 {
    (0..3)
        .map(|axis| (a[axis] - b[axis]) * (a[axis] - b[axis]))
        .sum()
}

/// The analytic gradients of [`edge_loss`] with respect to its endpoints:
/// `2 * (a - b)` for `a` and `2 * (b - a)` for `b`.
pub fn edge_gradient(a: [f32; 3], b: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let mut gradient_a = [0.0f32; 3];
    let mut gradient_b = [0.0f32; 3];
    for axis in 0..3 {
        gradient_a[axis] = 2.0 * (a[axis] - b[axis]);
        gradient_b[axis] = 2.0 * (b[axis] - a[axis]);
    }
    (gradient_a, gradient_b)
}

/// One optimizer micro-triangle's gradient record, the CPU mirror of the WGSL `Gradient`.
///
/// Handoff: 'Update micro-triangle vertices normals and opacity' consumes `GradientRecord`
/// (`to_bytes`, `GRADIENT_RECORD_STRIDE`); today it holds the analytic derivatives of the
/// weighted silhouette, 1-cosine normal and squared-edge objective for the three positions
/// (`d_position`), the independent normal (`d_normal`) and the annealed opacity
/// (`d_opacity`) in the five-lane 80-byte layout.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GradientRecord {
    pub d_position: [[f32; 3]; 3],
    pub d_normal: [f32; 3],
    pub d_opacity: f32,
}

impl GradientRecord {
    /// A record with no derivative.
    pub const ZERO: Self = Self {
        d_position: [[0.0; 3]; 3],
        d_normal: [0.0; 3],
        d_opacity: 0.0,
    };

    /// The 80-byte little-endian record: lanes 0..2 are `dP0..dP2` with a zero fourth
    /// component, lane 3 is `dNormal.xyz, dOpacity` and lane 4 is zero.
    pub fn to_bytes(self) -> [u8; GRADIENT_RECORD_STRIDE] {
        let mut lanes = [[0.0f32; 4]; 5];
        for (lane, position) in self.d_position.iter().enumerate() {
            lanes[lane][..3].copy_from_slice(position);
        }
        lanes[3] = [
            self.d_normal[0],
            self.d_normal[1],
            self.d_normal[2],
            self.d_opacity,
        ];
        let mut out = [0u8; GRADIENT_RECORD_STRIDE];
        for (lane, values) in lanes.iter().enumerate() {
            for (component, value) in values.iter().enumerate() {
                let offset = 4 * (4 * lane + component);
                out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
        out
    }
}

/// The submitted gradient WGSL: the shared `foliage_soft_coverage` section of the raster
/// stage followed by [`FOLIAGE_GRADIENTS_WGSL`].
pub fn gradients_wgsl() -> String {
    let mut source = String::from(crate::foliage_raster::soft_coverage_wgsl());
    source.push('\n');
    source.push_str(FOLIAGE_GRADIENTS_WGSL);
    source
}

fn gradient_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let compute = wgpu::ShaderStages::COMPUTE;
    let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: compute,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let uniform = |binding: u32, size: usize| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: compute,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: std::num::NonZeroU64::new(size as u64),
        },
        count: None,
    };
    let texture = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: compute,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2Array,
            multisampled: false,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("raa.lit.foliage.gradient.layout"),
        entries: &[
            uniform(0, FOLIAGE_PARAMS_SIZE),
            storage(1, true),
            storage(2, false),
            storage(3, false),
            storage(4, false),
            texture(5),
            texture(6),
            uniform(7, FOLIAGE_VIEW_TABLE_SIZE),
        ],
    })
}

fn validate_reference(
    name: &str,
    texture: &wgpu::Texture,
    format: wgpu::TextureFormat,
    resolution: u32,
    views: u32,
) -> Result<(), RaaError> {
    if texture.format() != format
        || texture.dimension() != wgpu::TextureDimension::D2
        || texture.mip_level_count() != 1
        || texture.width() != resolution
        || texture.height() != resolution
        || texture.depth_or_array_layers() != views
    {
        return Err(invalid(format!(
            "foliage gradient {name} reference must be {resolution}x{resolution} with {views} layer(s)"
        )));
    }
    Ok(())
}

/// Reduces the analytic gradient records of every micro-triangle of `work`.
///
/// One invocation owns one triangle: it scans the pixels of its counted bins in every
/// reference view, evaluates the published predicted pixel, and accumulates the weighted
/// silhouette, 1-cosine normal and squared-edge derivatives in registers before writing
/// its own 80-byte record. No stage-owned buffer is created or dropped apart from the
/// staged view-table upload; invalid dimensions or references return `foliage.invalid`
/// before any GPU work and a rejected shader returns `foliage.bake_failed`.
///
/// Handoff: 'Run differentiable foliage optimization' calls `encode_gradients`; today it
/// runs the triangle-owned gradient kernel after the raster stage and leaves the published
/// pixel records untouched. The earlier handoff for the same downstream card is
/// `crates/raa-lit/src/foliage_raster.rs:263` `encode_raster`.
pub fn encode_gradients(
    gpu: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    work: &FoliageCompute,
) -> Result<(), RaaError> {
    gpu.ensure_live()?;
    let resolution = work.resolution;
    let views = work.views;
    let groups = dispatch_size(work.count);
    let limit = gpu.device().limits().max_compute_workgroups_per_dimension;
    if resolution == 0 || views == 0 || views > 128 || groups > limit {
        return Err(invalid("invalid gradient dimensions or dispatch"));
    }
    let references: &FoliageReferences = &work.references;
    if references.views.len() != views as usize {
        return Err(invalid(
            "gradient view table count does not match the views",
        ));
    }
    validate_reference(
        "normal_coverage",
        &references.normal_coverage,
        wgpu::TextureFormat::Rgba16Float,
        resolution,
        views,
    )?;
    validate_reference(
        "depth",
        &references.depth,
        wgpu::TextureFormat::Rgba32Float,
        resolution,
        views,
    )?;
    let mut table = vec![0u8; FOLIAGE_VIEW_TABLE_SIZE];
    for (index, (matrix, near_far)) in references.views.iter().enumerate() {
        for value in matrix.iter().chain(near_far.iter()) {
            if !value.is_finite() {
                return Err(invalid("non-finite gradient view table"));
            }
        }
        let record = FoliageParams::view_record(*matrix, near_far[0], near_far[1]);
        table[index * 80..(index + 1) * 80].copy_from_slice(&record);
    }
    if groups == 0 {
        return Ok(());
    }
    let source = gradients_wgsl();
    raa_shader::Composer::validate(&source).map_err(|e| bake_failed(e.message))?;
    let device = gpu.device();
    let views_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("foliage.gradient.views"),
        contents: &table,
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("foliage.gradients"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout = gradient_layout(device);
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("foliage.gradients.layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("foliage.gradients"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("gradient_triangles"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("foliage.gradients"),
        layout: &layout,
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
                binding: 2,
                resource: work.gradients.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: work.bins.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: work.pixels.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(
                    &references.normal_coverage.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(
                    &references.depth.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: views_buffer.as_entire_binding(),
            },
        ],
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("foliage.gradients"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(groups, 1, 1);
    Ok(())
}
