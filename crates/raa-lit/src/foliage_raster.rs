//! Differentiable foliage raster (M7, contracts `m7-compute` and `m7-soft-coverage`).
//!
//! [`encode_raster`] resolves every pixel of every reference view of one
//! [`FoliageCompute`] generation from the counted 16-by-16 tile bins: each triangle of the
//! pixel's bin contributes `foliage_soft_coverage` on its three edges times its opacity to
//! the silhouette, and its world normal is weighted by that coverage and by the soft
//! ordering of the captured reference depth. The output pixel record holds the normalized
//! predicted world normal and the coverage in lane 0 and the predicted depth, zero loss
//! and two zeros in lane 1 (stride 32). A pixel without a positive predicted coverage is
//! the background record: zero coverage, a defined zero normal and far depth.
//!
//! The analytic helper `foliage_soft_coverage` lives once in `foliage_raster.wgsl`;
//! [`soft_coverage_wgsl`] returns exactly that section so the gradient stage evaluates the
//! same formula, including the convention that its derivative with respect to distance is
//! `1 / (4 * temperature)` at the edge.

use raa_core::{GpuContext, RaaError};
use wgpu::util::DeviceExt;

use crate::foliage_bins::dispatch_size;
use crate::foliage_resources::{
    FOLIAGE_PARAMS_SIZE, FOLIAGE_VIEW_TABLE_SIZE, FoliageCompute, FoliageParams,
};

/// The full WGSL module of the raster stage.
pub const FOLIAGE_RASTER_WGSL: &str = include_str!("wgsl/foliage_raster.wgsl");
/// Bytes of one raster pixel record: lane 0 normal XYZ and coverage, lane 1 depth, loss, 0, 0.
pub const PIXEL_RECORD_STRIDE: usize = 32;
/// Marker that starts the reusable soft-coverage section of [`FOLIAGE_RASTER_WGSL`].
const SOFT_COVERAGE_MARKER: &str = "// raa foliage-raster part: soft coverage\n";
/// Marker that ends it.
const RASTER_MARKER: &str = "// raa foliage-raster part: raster\n";

fn invalid(message: impl Into<String>) -> RaaError {
    RaaError::FrameGraph {
        code: "foliage.invalid",
        message: message.into(),
    }
}

/// The analytic soft coverage of one triangle edge, the CPU twin of the WGSL helper:
/// `sigmoid(edge_distance / temperature)` with a positive distance inside the triangle.
///
/// At distance 0 the value is 0.5 and its derivative with respect to distance is
/// `1 / (4 * temperature)`, the temperature convention the gradient stage relies on.
pub fn soft_coverage(edge_distance: f32, temperature: f32) -> f32 {
    1.0 / (1.0 + (-edge_distance / temperature).exp())
}

/// The signed pixel-space distance from `point` to the line through `a` and `b`, positive
/// on the left of `a -> b` and in pixels. A zero-length edge has distance 0.
pub fn signed_edge_distance(a: [f32; 2], b: [f32; 2], point: [f32; 2]) -> f32 {
    let edge = [b[0] - a[0], b[1] - a[1]];
    let length = (edge[0] * edge[0] + edge[1] * edge[1]).sqrt();
    if length <= f32::EPSILON {
        return 0.0;
    }
    (edge[0] * (point[1] - a[1]) - edge[1] * (point[0] - a[0])) / length
}

fn cross2(a: [f32; 2], b: [f32; 2]) -> f32 {
    a[0] * b[1] - a[1] * b[0]
}

/// The soft coverage of one triangle at `point` before its opacity: the product of the
/// three inward-signed edge values, whatever the winding. At a point where all three edge
/// distances are zero the product is `0.5^3 = 0.125`.
pub fn triangle_soft_coverage(vertices: [[f32; 2]; 3], point: [f32; 2], temperature: f32) -> f32 {
    let [a, b, c] = vertices;
    let area = cross2([b[0] - a[0], b[1] - a[1]], [c[0] - a[0], c[1] - a[1]]);
    if area.abs() <= f32::EPSILON {
        return 0.0;
    }
    let orient = area.signum();
    let d0 = signed_edge_distance(b, c, point);
    let d1 = signed_edge_distance(c, a, point);
    let d2 = signed_edge_distance(a, b, point);
    soft_coverage(orient * d0, temperature)
        * soft_coverage(orient * d1, temperature)
        * soft_coverage(orient * d2, temperature)
}

/// The soft depth visibility of one predicted surface against the captured reference: the
/// same sigmoid family, with a depth tolerance of `temperature / resolution` so the band
/// spans the pixel temperature along the view depth. A background reference (zero coverage
/// or zero depth) has no surface and no normal target, so its visibility is zero.
pub fn depth_visibility(
    reference_coverage: f32,
    reference_depth: f32,
    predicted_depth: f32,
    temperature: f32,
    resolution: u32,
) -> f32 {
    if reference_coverage <= 0.0 || reference_depth <= 0.0 || resolution == 0 {
        return 0.0;
    }
    let scale = (temperature / resolution as f32).max(1e-6);
    soft_coverage((reference_depth - predicted_depth) / scale, 1.0)
}

/// One predicted triangle contribution at a pixel: its soft triangle coverage (opacity
/// included), its depth visibility, its world normal and its standard-Z depth.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RasterSample {
    pub coverage: f32,
    pub visibility: f32,
    pub normal: [f32; 3],
    pub depth: f32,
}

/// One raster pixel record, the CPU mirror of the WGSL `Pixel` struct.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RasterPixel {
    pub normal: [f32; 3],
    pub coverage: f32,
    pub depth: f32,
    pub loss: f32,
}

impl RasterPixel {
    /// What a pixel without predicted coverage keeps: zero coverage, the defined zero
    /// normal and the far standard-Z depth.
    pub const BACKGROUND: RasterPixel = RasterPixel {
        normal: [0.0; 3],
        coverage: 0.0,
        depth: 1.0,
        loss: 0.0,
    };

    /// The 32-byte little-endian record: lane 0 `normal.xyz, coverage`, lane 1
    /// `depth, loss, 0, 0`.
    pub fn to_bytes(self) -> [u8; PIXEL_RECORD_STRIDE] {
        let mut out = [0u8; PIXEL_RECORD_STRIDE];
        let lanes = [
            self.normal[0],
            self.normal[1],
            self.normal[2],
            self.coverage,
            self.depth,
            self.loss,
        ];
        for (index, value) in lanes.iter().enumerate() {
            out[4 * index..4 * index + 4].copy_from_slice(&value.to_le_bytes());
        }
        out
    }
}

/// Resolves the predicted normal, coverage and depth of one pixel from its visible
/// triangle samples: coverage is their sum clamped to 1, the normal is the normalized
/// coverage/visibility-weighted sum and the depth is that same weighted mean. No positive
/// coverage gives [`RasterPixel::BACKGROUND`]; a silhouette without a normal vote (the
/// reference background) keeps its coverage, a zero normal and the far depth.
pub fn resolve_pixel(samples: &[RasterSample]) -> RasterPixel {
    let mut coverage = 0.0f32;
    let mut weight = 0.0f32;
    let mut normal = [0.0f32; 3];
    let mut depth = 0.0f32;
    for sample in samples {
        coverage += sample.coverage;
        let w = sample.coverage * sample.visibility;
        weight += w;
        for (sum, value) in normal.iter_mut().zip(sample.normal) {
            *sum += w * value;
        }
        depth += w * sample.depth;
    }
    if coverage <= 0.0 {
        return RasterPixel::BACKGROUND;
    }
    let mut out = RasterPixel {
        coverage: coverage.min(1.0),
        ..RasterPixel::BACKGROUND
    };
    let length = normal.iter().map(|v| v * v).sum::<f32>().sqrt();
    if weight > 1e-8 && length > 1e-8 {
        out.normal = normal.map(|v| v / length);
        out.depth = depth / weight;
    }
    out
}

/// The reusable WGSL of `foliage_soft_coverage`: the `// raa foliage-raster part: soft
/// coverage` section of [`FOLIAGE_RASTER_WGSL`], including its temperature derivative
/// convention and no binding or entry point.
///
/// Handoff: 'Differentiate silhouette normal and edge losses' includes `soft_coverage_wgsl`
/// in its gradient module and reads the 32-byte `RasterPixel` record (`resolve_pixel`,
/// `PIXEL_RECORD_STRIDE`); today this returns the `foliage_soft_coverage` section of
/// `foliage_raster.wgsl` so the gradient source shares the one analytic formula.
pub fn soft_coverage_wgsl() -> &'static str {
    let start = FOLIAGE_RASTER_WGSL
        .find(SOFT_COVERAGE_MARKER)
        .expect("foliage_raster.wgsl declares the soft-coverage part");
    let rest = &FOLIAGE_RASTER_WGSL[start + SOFT_COVERAGE_MARKER.len()..];
    let end = rest.find(RASTER_MARKER).unwrap_or(rest.len());
    &FOLIAGE_RASTER_WGSL[start..start + SOFT_COVERAGE_MARKER.len() + end]
}

/// The dedicated compute group 0 of the optimizer kernels (`m7-compute`), verbatim for
/// every stage: params, triangles, gradients, bins, pixels, the normal/coverage and depth
/// reference arrays and the 128-record view table.
fn raster_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
        label: Some("raa.lit.foliage.raster.layout"),
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

/// Rasterizes the predicted world normals and coverage of every reference view into the
/// pixel records of `work`.
///
/// The counted bins, triangle records, reference normal/coverage and depth arrays and the
/// view table come from the generation; the stage allocates only its pipeline and the
/// staged view-table upload and never mutates another stage's buffers. Invalid dimensions,
/// a mismatched view table or reference arrays, non-finite matrices and a dispatch above
/// the device limit return `foliage.invalid` before any GPU work.
///
/// Handoff: 'Run differentiable foliage optimization' calls `encode_raster`; today it
/// scans each pixel's counted bin (filled by `crates/raa-lit/src/foliage_bins.rs:59`
/// `encode_bins`), accumulates the soft edge coverage and the reference depth-weighted
/// world normals, and writes one 32-byte pixel record per view pixel.
pub fn encode_raster(
    gpu: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    work: &FoliageCompute,
) -> Result<(), RaaError> {
    gpu.ensure_live()?;
    let resolution = work.resolution;
    let views = work.views;
    let pixels = resolution
        .checked_mul(resolution)
        .ok_or_else(|| invalid("raster pixel count overflow"))?;
    let groups = dispatch_size(pixels);
    let limit = gpu.device().limits().max_compute_workgroups_per_dimension;
    if resolution == 0
        || views == 0
        || views > 128
        || work.references.views.len() != views as usize
        || groups > limit
        || views > limit
    {
        return Err(invalid("invalid raster dimensions or dispatch"));
    }
    for (name, texture, format) in [
        (
            "normal_coverage",
            &work.references.normal_coverage,
            wgpu::TextureFormat::Rgba16Float,
        ),
        (
            "depth",
            &work.references.depth,
            wgpu::TextureFormat::Rgba32Float,
        ),
    ] {
        if texture.format() != format
            || texture.dimension() != wgpu::TextureDimension::D2
            || texture.mip_level_count() != 1
            || texture.width() != resolution
            || texture.height() != resolution
            || texture.depth_or_array_layers() != views
        {
            return Err(invalid(format!(
                "foliage raster {name} reference must be {resolution}x{resolution} with {views} layer(s)"
            )));
        }
    }
    raa_shader::Composer::validate(FOLIAGE_RASTER_WGSL).map_err(|e| RaaError::FrameGraph {
        code: e.code,
        message: e.message,
    })?;
    let device = gpu.device();
    let mut table = vec![0u8; FOLIAGE_VIEW_TABLE_SIZE];
    for (index, (matrix, near_far)) in work.references.views.iter().enumerate() {
        for value in matrix.iter().chain(near_far.iter()) {
            if !value.is_finite() {
                return Err(invalid("non-finite raster view table"));
            }
        }
        let record = FoliageParams::view_record(*matrix, near_far[0], near_far[1]);
        table[index * 80..(index + 1) * 80].copy_from_slice(&record);
    }
    let views_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("foliage.raster.views"),
        contents: &table,
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("foliage.raster"),
        source: wgpu::ShaderSource::Wgsl(FOLIAGE_RASTER_WGSL.into()),
    });
    let layout = raster_layout(device);
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("foliage.raster.layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("foliage.raster"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("raster_pixels"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("foliage.raster"),
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
                    &work
                        .references
                        .normal_coverage
                        .create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(
                    &work.references.depth.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: views_buffer.as_entire_binding(),
            },
        ],
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("foliage.raster"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(groups, views, 1);
    Ok(())
}
