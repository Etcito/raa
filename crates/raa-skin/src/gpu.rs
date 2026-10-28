//! Vertex and edge compute. Pose ownership and FeaturePass scheduling live above this API.
use std::collections::BTreeSet;

use crate::SkinMesh;
use raa_core::{GpuContext, wgpu};
use wgpu::util::DeviceExt;

pub const SKIN_WGSL: &str = include_str!("skin.wgsl");
pub const SKIN_EDGES_WGSL: &str = include_str!("skin_edges.wgsl");

/// Immutable packed rest inputs, shared by every instance of a mesh.
pub struct SkinVertexInputs {
    pub buffer: wgpu::Buffer,
    pub edge_pairs: wgpu::Buffer,
    pub vertices: u32,
    pub edges: u32,
}

impl SkinVertexInputs {
    /// Packs validated S18 inputs, resolving generated/default tangents as skin_cpu does.
    /// The rest streams, UVs, colours and topology are never modified.
    pub fn new(gpu: &GpuContext, mesh: &SkinMesh) -> Self {
        let bytes = Self::bytes(mesh);
        let mut set = BTreeSet::new();
        for tri in mesh.rest.base.indices.chunks_exact(3) {
            for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                set.insert((a.min(b), a.max(b)));
            }
        }
        let pairs: Vec<u32> = set.into_iter().flat_map(|(a, b)| [a, b]).collect();
        let pair_bytes: Vec<_> = pairs.iter().flat_map(|i| i.to_le_bytes()).collect();
        let buffer = |label, contents| {
            gpu.device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                })
        };
        Self {
            buffer: buffer("skin.rest", &bytes),
            edge_pairs: buffer("skin.edges", &pair_bytes),
            vertices: mesh.rest.vertex_count() as u32,
            edges: (pairs.len() / 2) as u32,
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

/// Reusable S18 vertex and edge pipelines. Records into the caller's encoder before draws.
pub struct SkinVertexKernel {
    pipeline: wgpu::ComputePipeline,
    edges: wgpu::ComputePipeline,
}

impl SkinVertexKernel {
    pub fn new(gpu: &GpuContext) -> Self {
        let pipeline = |label: &str, source: &str, entry_point: &str| {
            let module = gpu
                .device()
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
            gpu.device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: None,
                    module: &module,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };
        Self {
            pipeline: pipeline("skin.vertices", SKIN_WGSL, "skin_vertices"),
            edges: pipeline("skin.edges", SKIN_EDGES_WGSL, "skin_edges"),
        }
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

    /// Bind the rest edge topology and the posed stream0 (48-byte segment records).
    pub fn bind_edges(
        &self,
        gpu: &GpuContext,
        inputs: &SkinVertexInputs,
        stream0: &wgpu::Buffer,
        segments: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        assert_eq!(stream0.size(), u64::from(inputs.vertices) * 24);
        assert_eq!(segments.size(), u64::from(inputs.edges) * 48);
        let buffers = [&inputs.edge_pairs, stream0, segments];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skin.edges"),
            layout: &self.edges.get_bind_group_layout(0),
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

    /// Dispatch after [`Self::encode`] so endpoints come from that instance's posed stream.
    pub fn encode_edges(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        binding: &wgpu::BindGroup,
        edges: u32,
    ) {
        if edges == 0 {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("skin.edges"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.edges);
        pass.set_bind_group(0, binding, &[]);
        pass.dispatch_workgroups(edges.div_ceil(64), 1, 1);
    }
}
