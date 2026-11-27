//! Bounded differentiable foliage update (M7, contract `m7-compute`).
//!
//! [`encode_update`] applies one optimizer step to every micro-triangle of a
//! [`FoliageCompute`] generation: it reads the 80-byte gradient record the previous stage
//! wrote, moves the three vertices by a bounded finite learning-rate step, normalizes the
//! independent normal separately (with a defined fallback when it is degenerate) and
//! anneals the opacity toward the params target. The CPU twins [`update_triangle`],
//! [`resolve_normal`], [`anneal_opacity`] and [`bounded_step`] evaluate exactly the
//! formulas of [`FOLIAGE_UPDATE_WGSL`].
//!
//! [`freeze_terminal`] is the terminal readback: it parses the five-lane 80-byte triangle
//! records, refuses a non-finite value with `foliage.bake_failed` instead of publishing
//! NaNs, normalizes every retained normal and writes the exact opaque opacity `1` for
//! every retained triangle. The final output is therefore opaque triangle geometry, never
//! an alpha-blended splat.

use raa_core::{GpuContext, RaaError};

use crate::foliage_bins::dispatch_size;
use crate::foliage_gradients::GradientRecord;
use crate::foliage_resources::{FOLIAGE_PARAMS_SIZE, FoliageCompute, MicroTriangle};

/// The update-stage WGSL module.
pub const FOLIAGE_UPDATE_WGSL: &str = include_str!("wgsl/foliage_update.wgsl");
/// Bytes of one GPU triangle record: three position lanes, normal/opacity and one zero lane.
pub const TRIANGLE_RECORD_STRIDE: usize = 80;
/// Longest position step one iteration may take, in source-root metres.
pub const MAX_POSITION_STEP: f32 = 0.05;
/// Longest independent-normal correction one iteration may apply.
pub const MAX_NORMAL_STEP: f32 = 1.0;
/// Longest opacity correction one iteration may apply.
pub const MAX_OPACITY_STEP: f32 = 0.25;
/// Distance within which the opacity anneal snaps exactly onto its target.
pub const OPACITY_ANNEAL_STEP: f32 = 0.1;
/// Shortest length a normal may have before it counts as degenerate.
pub const NORMAL_EPSILON: f32 = 1e-8;

/// The defined fallback normal when neither the candidate, the previous nor the geometric
/// normal is usable.
const FALLBACK_NORMAL: [f32; 3] = [0.0, 1.0, 0.0];

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

fn finite3(v: [f32; 3]) -> bool {
    v.iter().all(|value| value.is_finite())
}

fn length3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn scaled(v: [f32; 3], scale: f32) -> [f32; 3] {
    v.map(|value| value * scale)
}

fn difference(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// One bounded, finite gradient-descent step `-learning_rate * gradient`, scaled back to
/// `limit` when it is longer. A non-finite rate, gradient or limit takes no step.
pub fn bounded_step(gradient: [f32; 3], learning_rate: f32, limit: f32) -> [f32; 3] {
    if !learning_rate.is_finite() || !finite3(gradient) || !limit.is_finite() || limit <= 0.0 {
        return [0.0; 3];
    }
    let step = scaled(gradient, -learning_rate);
    let length = length3(step);
    if !length.is_finite() {
        return [0.0; 3];
    }
    if length > limit {
        return scaled(step, limit / length);
    }
    step
}

/// One bounded anneal of `current` toward `target`: a non-finite current restarts from
/// the opaque `1`, a non-finite target anneals toward it, and a remaining distance within
/// [`OPACITY_ANNEAL_STEP`] snaps exactly onto the target, so repeated updates terminate on
/// it instead of approaching forever.
pub fn anneal_opacity(current: f32, target: f32) -> f32 {
    let goal = if target.is_finite() {
        target.clamp(0.0, 1.0)
    } else {
        1.0
    };
    let start = if current.is_finite() {
        current.clamp(0.0, 1.0)
    } else {
        1.0
    };
    let delta = goal - start;
    if delta.abs() <= OPACITY_ANNEAL_STEP {
        return goal;
    }
    start + OPACITY_ANNEAL_STEP * delta.signum()
}

/// The updated independent normal: `candidate` normalized on its own. A zero or non-finite
/// candidate keeps the normalized previous normal; when that is degenerate too the
/// triangle's geometric normal is used, and the defined `+Y` fallback is last.
pub fn resolve_normal(
    candidate: [f32; 3],
    previous: [f32; 3],
    positions: [[f32; 3]; 3],
) -> [f32; 3] {
    for value in [candidate, previous] {
        let length = length3(value);
        if length.is_finite() && length > NORMAL_EPSILON {
            return scaled(value, 1.0 / length);
        }
    }
    let geometric = cross(
        difference(positions[1], positions[0]),
        difference(positions[2], positions[0]),
    );
    let length = length3(geometric);
    if length.is_finite() && length > NORMAL_EPSILON {
        return scaled(geometric, 1.0 / length);
    }
    FALLBACK_NORMAL
}

fn stepped_position(position: [f32; 3], gradient: [f32; 3], learning_rate: f32) -> [f32; 3] {
    let step = bounded_step(gradient, learning_rate, MAX_POSITION_STEP);
    let next = [
        position[0] + step[0],
        position[1] + step[1],
        position[2] + step[2],
    ];
    if finite3(next) {
        next
    } else if finite3(position) {
        position
    } else {
        [0.0; 3]
    }
}

/// One CPU update step, the exact twin of the `update_triangles` kernel: the three
/// vertices take their bounded finite gradient step, the independent normal is normalized
/// on its own, and the opacity takes its bounded gradient step and anneal.
pub fn update_triangle(
    triangle: MicroTriangle,
    gradient: GradientRecord,
    learning_rate: f32,
    opacity_target: f32,
) -> MicroTriangle {
    let positions = [triangle.p0, triangle.p1, triangle.p2];
    let stepped = [
        stepped_position(positions[0], gradient.d_position[0], learning_rate),
        stepped_position(positions[1], gradient.d_position[1], learning_rate),
        stepped_position(positions[2], gradient.d_position[2], learning_rate),
    ];
    let normal_step = bounded_step(gradient.d_normal, learning_rate, MAX_NORMAL_STEP);
    let candidate = [
        triangle.normal[0] + normal_step[0],
        triangle.normal[1] + normal_step[1],
        triangle.normal[2] + normal_step[2],
    ];
    let opacity_step = bounded_step(
        [gradient.d_opacity, 0.0, 0.0],
        learning_rate,
        MAX_OPACITY_STEP,
    )[0];
    MicroTriangle {
        p0: stepped[0],
        p1: stepped[1],
        p2: stepped[2],
        normal: resolve_normal(candidate, triangle.normal, stepped),
        opacity: anneal_opacity(
            (triangle.opacity + opacity_step).clamp(0.0, 1.0),
            opacity_target,
        ),
    }
}

/// The 80-byte little-endian GPU record: `p0.xyz/0@0`, `p1.xyz/0@16`, `p2.xyz/0@32`,
/// `normal.xyz, opacity@48` and a zero lane at 64.
pub fn triangle_to_record(triangle: MicroTriangle) -> [u8; TRIANGLE_RECORD_STRIDE] {
    let lanes = [
        [triangle.p0[0], triangle.p0[1], triangle.p0[2], 0.0],
        [triangle.p1[0], triangle.p1[1], triangle.p1[2], 0.0],
        [triangle.p2[0], triangle.p2[1], triangle.p2[2], 0.0],
        [
            triangle.normal[0],
            triangle.normal[1],
            triangle.normal[2],
            triangle.opacity,
        ],
        [0.0; 4],
    ];
    let mut out = [0u8; TRIANGLE_RECORD_STRIDE];
    for (lane, values) in lanes.iter().enumerate() {
        for (component, value) in values.iter().enumerate() {
            let offset = 4 * (4 * lane + component);
            out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    out
}

fn read_f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

/// The frozen terminal generation of a readback: parses `count` five-lane records, refuses
/// any non-finite value with `foliage.bake_failed` instead of publishing NaNs or
/// infinities, normalizes every retained normal and writes the exact opaque opacity `1`.
/// A readback shorter than `count` records returns `foliage.invalid` before any value is
/// published.
pub fn freeze_terminal(bytes: &[u8], count: u32) -> Result<Vec<MicroTriangle>, RaaError> {
    let wanted = (count as usize)
        .checked_mul(TRIANGLE_RECORD_STRIDE)
        .ok_or_else(|| invalid("terminal foliage triangle count overflow"))?;
    if bytes.len() < wanted {
        return Err(invalid(format!(
            "terminal foliage readback has {} bytes, expected {wanted}",
            bytes.len()
        )));
    }
    let mut triangles = Vec::with_capacity(count as usize);
    let (records, _) = bytes[..wanted].as_chunks::<TRIANGLE_RECORD_STRIDE>();
    for (index, record) in records.iter().enumerate() {
        let positions = [
            [
                read_f32(record, 0),
                read_f32(record, 4),
                read_f32(record, 8),
            ],
            [
                read_f32(record, 16),
                read_f32(record, 20),
                read_f32(record, 24),
            ],
            [
                read_f32(record, 32),
                read_f32(record, 36),
                read_f32(record, 40),
            ],
        ];
        let normal = [
            read_f32(record, 48),
            read_f32(record, 52),
            read_f32(record, 56),
        ];
        let opacity = read_f32(record, 60);
        if !positions.iter().all(|position| finite3(*position))
            || !finite3(normal)
            || !opacity.is_finite()
        {
            return Err(bake_failed(format!(
                "terminal foliage micro-triangle {index} is not finite"
            )));
        }
        triangles.push(MicroTriangle {
            p0: positions[0],
            p1: positions[1],
            p2: positions[2],
            normal: resolve_normal(normal, normal, positions),
            opacity: 1.0,
        });
    }
    Ok(triangles)
}

fn update_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
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
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("raa.lit.foliage.update.layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: compute,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(FOLIAGE_PARAMS_SIZE as u64),
                },
                count: None,
            },
            storage(1, false),
            storage(2, true),
        ],
    })
}

/// Applies one bounded, finite optimizer step to every micro-triangle of `work`.
///
/// The stage reads the params uniform, the writable triangle records and the read-only
/// gradient records of the generation; it creates only its own pipeline and bind group and
/// never owns or drops another stage's buffers. Invalid dimensions, under-sized buffers or
/// a dispatch above the device limit return `foliage.invalid` before any GPU work, and a
/// rejected shader returns `foliage.bake_failed`.
///
/// Handoff: 'Run differentiable foliage optimization' calls `encode_update` and
/// `freeze_terminal`; today this stage reads the gradient records of the previous stage,
/// applies the bounded learning-rate step to the vertices, normalizes the independent
/// normal with its defined fallback and anneals the opacity toward the params target
/// without writing a non-finite value into a triangle record. The earlier handoffs for the
/// same downstream card are `crates/raa-lit/src/foliage_raster.rs:259` `encode_raster` and
/// `crates/raa-lit/src/foliage_gradients.rs:234` `encode_gradients`.
pub fn encode_update(
    gpu: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    work: &FoliageCompute,
) -> Result<(), RaaError> {
    gpu.ensure_live()?;
    let groups = dispatch_size(work.count);
    let limit = gpu.device().limits().max_compute_workgroups_per_dimension;
    let needed = u64::from(work.count) * TRIANGLE_RECORD_STRIDE as u64;
    if work.resolution == 0
        || groups > limit
        || work.triangles.size() < needed
        || work.gradients.size() < needed
    {
        return Err(invalid("invalid update dimensions, buffers or dispatch"));
    }
    if groups == 0 {
        return Ok(());
    }
    raa_shader::Composer::validate(FOLIAGE_UPDATE_WGSL).map_err(|e| bake_failed(e.message))?;
    let device = gpu.device();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("foliage.update"),
        source: wgpu::ShaderSource::Wgsl(FOLIAGE_UPDATE_WGSL.into()),
    });
    let layout = update_layout(device);
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("foliage.update.layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("foliage.update"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("update_triangles"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("foliage.update"),
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
        ],
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("foliage.update"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(groups, 1, 1);
    Ok(())
}
