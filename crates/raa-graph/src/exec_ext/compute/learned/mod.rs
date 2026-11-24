//! Opt-in FP32 area inference on the graph device.

use super::{buffer, buffer_type, group, layout, storage_binding};
use raa_core::{GpuContext, RaaError};
use raa_pipeline::ResourceTable;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{borrow::Cow, future::Future};
use wgpu::{BufferBindingType as BindingType, BufferUsages as Usage};

pub const MANIFEST: &str = include_str!("manifest.json");
pub const WEIGHTS: &[u8] = include_bytes!("weights.fp32le");
pub const CHANNELS: usize = 12;
const HIDDEN: u64 = 16;
const WORKGROUP: u32 = 8;
const UNIFORM_BYTES: u64 = 24;
const ENTRIES: [&str; 6] = [
    "stem",
    "depthwise1",
    "pointwise1",
    "depthwise2",
    "pointwise2",
    "head",
];

pub struct AreaPilotPackage(Cow<'static, [u8]>);

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

impl AreaPilotPackage {
    pub fn embedded() -> Self {
        Self(Cow::Borrowed(WEIGHTS))
    }

    /// Resolve the manifest and FP32 blob before allocating GPU resources.
    pub async fn load(
        fetch: impl Future<Output = Result<(String, Vec<u8>), RaaError>>,
    ) -> Result<Self, RaaError> {
        let (manifest, weights) = fetch.await?;
        let mut actual: Value =
            serde_json::from_str(&manifest).map_err(|e| refuse(&e.to_string()))?;
        let baseline: Value = serde_json::from_str(MANIFEST).expect("embedded ABI manifest");
        let golden = include_bytes!("golden.json");
        if weights.len() != WEIGHTS.len() || actual["weightsSha256"] != hash(&weights) {
            return Err(refuse("weight blob length or SHA-256 mismatch"));
        }

        let package_hash = actual
            .as_object_mut()
            .ok_or_else(|| refuse("manifest must be an object"))?
            .remove("packageSha256");
        // Reconstruct the one legal ABI with hashes derived from the supplied blob.
        let mut expected = baseline;
        expected.as_object_mut().unwrap().remove("packageSha256");
        expected["weightsSha256"] = Value::String(hash(&weights));
        let mut previous_end = 0;
        for tensor in expected["tensors"].as_array_mut().unwrap() {
            let offset = tensor["offset"].as_u64().unwrap() as usize;
            let length = tensor["byteLength"].as_u64().unwrap() as usize;
            let bytes = &weights[offset..offset + length];
            if weights[previous_end..offset].iter().any(|&byte| byte != 0)
                || bytes
                    .chunks_exact(4)
                    .any(|word| !f32::from_le_bytes(word.try_into().unwrap()).is_finite())
            {
                return Err(refuse("invalid tensor padding or non-finite weight"));
            }
            previous_end = offset + length;
            tensor["sha256"] = Value::String(hash(bytes));
        }
        if actual != expected {
            return Err(refuse(
                "manifest differs from the FP32 operator ABI or tensor hashes",
            ));
        }
        // The v1 manifest uses Python's exponent spelling for the two tolerances.
        let canonical = serde_json::to_string(&expected)
            .map_err(|e| refuse(&e.to_string()))?
            .replace("0.00001", "1e-05")
            .replace("1e-5", "1e-05");
        let mut digest = Sha256::new();
        digest.update(canonical.as_bytes());
        digest.update(&weights);
        digest.update(golden);
        if package_hash != Some(Value::String(format!("{:x}", digest.finalize()))) {
            return Err(refuse("package SHA-256 mismatch"));
        }

        Ok(Self(Cow::Owned(weights)))
    }
}

#[derive(Clone, Copy)]
pub struct AreaPilotRect {
    pub size: [u32; 2],
    pub origin: [u32; 2],
    pub extent: [u32; 2],
}

impl AreaPilotRect {
    pub fn full(width: u32, height: u32) -> Self {
        Self {
            size: [width, height],
            origin: [0, 0],
            extent: [width, height],
        }
    }

    fn workgroups(self) -> [u32; 2] {
        self.extent.map(|n| n.div_ceil(WORKGROUP))
    }

    pub fn check_limits(self, limits: &wgpu::Limits) -> Result<AreaPilotMemory, RaaError> {
        for axis in 0..2 {
            if self.extent[axis] == 0
                || self.origin[axis]
                    .checked_add(self.extent[axis])
                    .is_none_or(|end| end > self.size[axis])
                || self.size[axis] > limits.max_texture_dimension_2d
            {
                return Err(refuse("active rect or scene extent exceeds bounds"));
            }
        }
        if self
            .workgroups()
            .iter()
            .any(|&n| n > limits.max_compute_workgroups_per_dimension)
            || limits.max_compute_workgroup_size_x < WORKGROUP
            || limits.max_compute_workgroup_size_y < WORKGROUP
            || limits.max_compute_workgroup_size_z < 1
            || limits.max_compute_invocations_per_workgroup < WORKGROUP * WORKGROUP
            || limits.max_bind_groups < 1
            || limits.max_storage_buffers_per_shader_stage < 4
            || limits.max_uniform_buffers_per_shader_stage < 1
            || limits.max_bindings_per_bind_group < 5
            || limits.max_uniform_buffer_binding_size < UNIFORM_BYTES
            || limits.max_buffer_size < UNIFORM_BYTES
        {
            return Err(refuse(
                "compute dispatch or binding limits are insufficient",
            ));
        }
        let pixels = u64::from(self.size[0]) * u64::from(self.size[1]);
        let hidden = pixels
            .checked_mul(HIDDEN * 4)
            .ok_or_else(|| refuse("tensor byte count overflows"))?;
        let io = pixels * (CHANNELS as u64 * 4);
        let weights = WEIGHTS.len() as u64;
        let storage_limit = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        if hidden > storage_limit
            || weights > storage_limit
            || pixels > u64::from(u32::MAX) / HIDDEN
        {
            return Err(refuse("NHWC storage buffer exceeds device limits"));
        }
        Ok(AreaPilotMemory {
            io,
            hidden,
            weights,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AreaPilotMemory {
    /// Bytes in each of input and output.
    pub io: u64,
    /// Bytes in each of two ping-pong activation buffers.
    pub hidden: u64,
    pub weights: u64,
}

impl AreaPilotMemory {
    pub fn resident_bytes(self) -> u64 {
        2 * self.io + 2 * self.hidden + self.weights + UNIFORM_BYTES
    }
}

fn refuse(message: &str) -> RaaError {
    RaaError::Resource(format!("area inference pilot: {message}"))
}

struct Stage {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
}

pub struct AreaPilot {
    rect: AreaPilotRect,
    memory: AreaPilotMemory,
    input: wgpu::Buffer,
    output: wgpu::Buffer,
    weights: wgpu::Buffer,
    queue: wgpu::Queue,
    stages: [Stage; 6],
}

impl AreaPilot {
    /// Build a complete candidate on the supplied graph device. The caller can
    /// adopt the candidate for its next real frame after this succeeds.
    pub async fn prepare(
        gpu: &GpuContext,
        rect: AreaPilotRect,
        package: AreaPilotPackage,
    ) -> Result<Self, RaaError> {
        gpu.ensure_live()?;
        gpu.check_compute_limits(|limits| rect.check_limits(limits).map(|_| ()))?;
        let memory = rect.check_limits(&gpu.limits())?;
        let device = gpu.device();
        let queue = gpu.queue();
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let weights = buffer(device, memory.weights, Usage::STORAGE | Usage::COPY_DST);
        queue.write_buffer(&weights, 0, &package.0);
        let input = buffer(device, memory.io, Usage::STORAGE | Usage::COPY_DST);
        let output = buffer(device, memory.io, Usage::STORAGE | Usage::COPY_SRC);
        let a = buffer(device, memory.hidden, Usage::STORAGE);
        let b = buffer(device, memory.hidden, Usage::STORAGE);
        let frame = buffer(device, UNIFORM_BYTES, Usage::UNIFORM | Usage::COPY_DST);
        let frame_bytes: Vec<_> = rect
            .size
            .into_iter()
            .chain(rect.origin)
            .chain(rect.extent)
            .flat_map(u32::to_le_bytes)
            .collect();
        queue.write_buffer(&frame, 0, &frame_bytes);

        let read = buffer_type(BindingType::Storage { read_only: true });
        let write = buffer_type(BindingType::Storage { read_only: false });
        let bindings = layout(
            device,
            &[read, read, write, read, buffer_type(BindingType::Uniform)],
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("area inference pilot"),
            bind_group_layouts: &[Some(&bindings)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("area inference FP32"),
            source: wgpu::ShaderSource::Wgsl(include_str!("pilot.wgsl").into()),
        });
        let stages = ENTRIES
            .into_iter()
            .zip([
                (&input, &a),
                (&a, &b),
                (&b, &a),
                (&a, &b),
                (&b, &a),
                (&a, &output),
            ])
            .map(|(entry, (src, dst))| Stage {
                pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                }),
                group: group(
                    device,
                    &bindings,
                    &[
                        storage_binding(&weights, memory.weights),
                        src.as_entire_binding(),
                        dst.as_entire_binding(),
                        input.as_entire_binding(),
                        frame.as_entire_binding(),
                    ],
                ),
            })
            .collect::<Vec<_>>();
        if let Some(error) = scope.pop().await {
            return Err(refuse(&error.to_string()));
        }
        Ok(Self {
            rect,
            memory,
            input,
            output,
            weights,
            queue: queue.clone(),
            stages: stages.try_into().ok().unwrap(),
        })
    }

    pub fn memory(&self) -> AreaPilotMemory {
        self.memory
    }

    /// Publish the same resident storage used by the dispatches to graph buffer inputs.
    pub fn publish(&self, table: &ResourceTable, entry: &str) {
        table.publish_buffer(entry, "weights", &self.weights, self.memory.weights);
        table.publish_buffer(entry, "output", &self.output, self.memory.io);
    }

    pub fn output(&self) -> &wgpu::Buffer {
        &self.output
    }

    /// Raw NHWC samples on the scene grid. Upload once before the frame submission.
    pub fn upload(&self, samples: &[f32]) -> Result<(), RaaError> {
        if samples.len() as u64 != self.memory.io / 4 {
            return Err(refuse("input sample count does not match the scene extent"));
        }
        let mut packed = Vec::with_capacity(samples.len() * 4);
        for value in samples {
            packed.extend_from_slice(&value.to_le_bytes());
        }
        self.queue.write_buffer(&self.input, 0, &packed);
        Ok(())
    }

    /// Record only as part of a real frame. Timestamp indices, when supplied,
    /// must be 0 and 1 in a timestamp query set on the same device.
    pub fn record_real_frame(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        query: Option<&wgpu::QuerySet>,
    ) {
        self.record(encoder, 1, false, query);
    }

    /// Synthetic four-light compute control. Shared mode uses identical trunk
    /// input for every head. The head is evaluated four times into the same output.
    pub fn record_four_light_benchmark(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        shared: bool,
        query: Option<&wgpu::QuerySet>,
    ) {
        self.record(encoder, 4, shared, query);
    }

    fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        lights: usize,
        shared: bool,
        query: Option<&wgpu::QuerySet>,
    ) {
        let [x, y] = self.rect.workgroups();
        for light in 0..lights {
            let start = if shared && light > 0 {
                self.stages.len() - 1
            } else {
                0
            };
            for (i, stage) in self.stages.iter().enumerate().skip(start) {
                let begin = light == 0 && i == 0;
                let end = light + 1 == lights && i + 1 == self.stages.len();
                let timestamp_writes = query.filter(|_| begin || end).map(|query_set| {
                    wgpu::ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: begin.then_some(0),
                        end_of_pass_write_index: end.then_some(1),
                    }
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(ENTRIES[i]),
                    timestamp_writes,
                });
                pass.set_pipeline(&stage.pipeline);
                pass.set_bind_group(0, &stage.group, &[]);
                pass.dispatch_workgroups(x, y, 1);
            }
        }
    }
}

/// Per-view activation slot. A staged replacement becomes visible only when
/// the owner starts a real frame. Dropping a failed candidate preserves the active one.
#[derive(Default)]
pub struct AreaPilotSlot {
    active: Option<AreaPilot>,
    pending: Option<AreaPilot>,
}

impl AreaPilotSlot {
    pub fn stage(&mut self, candidate: AreaPilot) {
        self.pending = Some(candidate);
    }

    pub fn active(&self) -> Option<&AreaPilot> {
        self.active.as_ref()
    }

    pub fn begin_real_frame(&mut self) -> Option<&AreaPilot> {
        if let Some(ready) = self.pending.take() {
            self.active = Some(ready);
        }
        self.active()
    }
}
