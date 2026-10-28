//! Vertex-only compute. Pose ownership, edges and FeaturePass scheduling live above this API.
use crate::SkinMesh;
use raa_core::{GpuContext, wgpu};
use wgpu::util::DeviceExt;

pub const SKIN_WGSL: &str = include_str!("skin.wgsl");

/// Immutable packed rest inputs, shared by every instance of a mesh.
pub struct SkinVertexInputs {
    pub buffer: wgpu::Buffer,
    pub vertices: u32,
}

impl SkinVertexInputs {
    /// Packs validated S18 inputs, resolving generated/default tangents as skin_cpu does.
    /// The rest streams, UVs, colours and topology are never modified.
    pub fn new(gpu: &GpuContext, mesh: &SkinMesh) -> Self {
        let bytes = Self::bytes(mesh);
        Self {
            buffer: gpu
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("skin.rest"),
                    contents: &bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                }),
            vertices: mesh.rest.vertex_count() as u32,
        }
    }

    /// Little-endian 64-byte records matching binding 0 of [`SKIN_WGSL`].
    pub fn bytes(mesh: &SkinMesh) -> Vec<u8> {
        let mut rest = mesh.rest.clone();
        rest.ensure_tangents();
        let (mut s0, mut s1, mut joints, mut weights) = (vec![], vec![], vec![], vec![]);
        rest.write_stream0_bytes(&mut s0);
        rest.write_stream1_bytes(&mut s1);
        mesh.write_joint_bytes(&mut joints);
        mesh.write_weight_bytes(&mut weights);
        let mut bytes = Vec::with_capacity(rest.vertex_count() * 64);
        for i in 0..rest.vertex_count() {
            bytes.extend_from_slice(&s0[i * 24..i * 24 + 24]);
            bytes.extend_from_slice(&s1[i * 40 + 8..i * 40 + 24]);
            bytes.extend_from_slice(&joints[i * 8..i * 8 + 8]);
            bytes.extend_from_slice(&weights[i * 16..i * 16 + 16]);
        }
        bytes
    }
}

/// Reusable S18 vertex pipeline. Records into the caller's encoder before draw passes.
pub struct SkinVertexKernel {
    pipeline: wgpu::ComputePipeline,
}

impl SkinVertexKernel {
    pub fn new(gpu: &GpuContext) -> Self {
        let module = gpu
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("skin.vertices"),
                source: wgpu::ShaderSource::Wgsl(SKIN_WGSL.into()),
            });
        let pipeline = gpu
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("skin.vertices"),
                layout: None,
                module: &module,
                entry_point: Some("skin_vertices"),
                compilation_options: Default::default(),
                cache: None,
            });
        Self { pipeline }
    }

    /// Bind a validated instance pose (column-major f32x16 per joint, STORAGE) and
    /// the raa-lit view's private stream0/tangents (24/16 bytes per vertex, STORAGE).
    /// Caller validates pose length, finiteness and joint range before binding.
    /// Shared stream 1 and indices are deliberately not writable or bound here.
    pub fn bind(
        &self,
        gpu: &GpuContext,
        inputs: &SkinVertexInputs,
        pose: &wgpu::Buffer,
        stream0: &wgpu::Buffer,
        tangents: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        assert_eq!(stream0.size(), u64::from(inputs.vertices) * 24);
        assert_eq!(tangents.size(), u64::from(inputs.vertices) * 16);
        let buffers = [&inputs.buffer, pose, stream0, tangents];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skin.instance"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        })
    }

    /// Dispatch once per posed instance. For absent/disabled skinning select rest resources.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        binding: &wgpu::BindGroup,
        vertices: u32,
    ) {
        if vertices == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("skin.vertices"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, binding, &[]);
        pass.dispatch_workgroups(vertices.div_ceil(64), 1, 1);
    }
}
