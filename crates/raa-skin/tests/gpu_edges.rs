//! S18 posed edge segments: the rest topology, posed endpoints, duplicates and instances.
#![cfg(not(target_arch = "wasm32"))]

use std::collections::BTreeSet;

use raa_core::wgpu::util::DeviceExt;
use raa_core::{GpuContext, testing::software_gpu, wgpu};
use raa_skin::{CpuSkin, SKIN_PARITY_EPS, SkinMesh, SkinVertexInputs, SkinVertexKernel, skin_cpu};

const ID: [f32; 16] = [
    1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
];
// T(1,-1,0) * Rz(90 degrees): a rigid bend with a non-zero translation.
const BEND: [f32; 16] = [
    0., 1., 0., 0., -1., 0., 0., 0., 0., 0., 1., 0., 1., -1., 0., 1.,
];

fn fixture() -> SkinMesh {
    // Vertices 0 and 1 share the rest position (0,0,0) but bind to different joints.
    let rest = raa_mesh::from_js_arrays(
        &[0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.],
        &[1., 0., 0.].repeat(4),
        None,
        None,
        None,
        &[0, 1, 2, 1, 3, 2],
        &[],
    )
    .unwrap();
    let joints = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0];
    SkinMesh::new(rest, &joints, &[1., 0., 0., 0.].repeat(4)).unwrap()
}

fn pose(translate: [f32; 3], second: [f32; 16]) -> Vec<f32> {
    let mut first = ID;
    first[12..15].copy_from_slice(&translate);
    [first, second].concat()
}

fn close(a: f32, b: f32) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= SKIN_PARITY_EPS
}

fn upload(gpu: &GpuContext, bytes: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    gpu.device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytes,
            usage: usage | wgpu::BufferUsages::COPY_SRC,
        })
}

fn read(gpu: &GpuContext, buffer: &wgpu::Buffer) -> Vec<f32> {
    let staging = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    encoder.map_buffer_on_submit(&staging, wgpu::MapMode::Read, .., |_| {});
    gpu.queue().submit([encoder.finish()]);
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let bytes = staging.get_mapped_range(..).unwrap().to_vec();
    staging.unmap();
    bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect()
}

fn pairs(mesh: &SkinMesh) -> Vec<u32> {
    let mut edges = BTreeSet::new();
    for tri in mesh.rest.base.indices.chunks_exact(3) {
        for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            edges.insert((a.min(b), a.max(b)));
        }
    }
    edges.into_iter().flat_map(|(a, b)| [a, b]).collect()
}

fn records(cpu: &CpuSkin, pairs: &[u32]) -> Vec<f32> {
    assert_eq!(pairs.len(), cpu.edge_segments.len() * 2);
    pairs
        .chunks_exact(2)
        .flat_map(|p| {
            p.iter().flat_map(|&v| {
                let v = v as usize;
                cpu.positions[v].into_iter().chain(cpu.normals[v])
            })
        })
        .collect()
}

fn instance(
    gpu: &GpuContext,
    kernel: &SkinVertexKernel,
    inputs: &SkinVertexInputs,
    encoder: &mut wgpu::CommandEncoder,
    pose: &[f32],
) -> wgpu::Buffer {
    let (n, e) = (inputs.vertices as usize, inputs.edges as usize);
    let bytes: Vec<_> = pose.iter().flat_map(|v| v.to_le_bytes()).collect();
    let pose = upload(gpu, &bytes, wgpu::BufferUsages::STORAGE);
    let stream0 = upload(gpu, &vec![0; n * 24], wgpu::BufferUsages::STORAGE);
    let tangents = upload(gpu, &vec![0; n * 16], wgpu::BufferUsages::STORAGE);
    let segments = upload(gpu, &vec![0; e * 48], wgpu::BufferUsages::STORAGE);
    let vertex = kernel.bind(gpu, inputs, &pose, &stream0, &tangents);
    let edge = kernel.bind_edges(gpu, inputs, &stream0, &segments);
    kernel.encode(encoder, &vertex, inputs.vertices);
    kernel.encode_edges(encoder, &edge, inputs.edges);
    segments
}

#[test]
fn posed_edges_follow_bent_duplicates_and_isolate_instances() {
    let Some(gpu) = software_gpu() else {
        panic!("posed edge parity requires an adapter; no comparison executed");
    };
    let mesh = fixture();
    let inputs = SkinVertexInputs::new(&gpu, &mesh);
    let kernel = SkinVertexKernel::new(&gpu);
    let pairs = pairs(&mesh);
    let mut scaled = ID;
    for axis in [0, 5, 10] {
        scaled[axis] = 2.;
    }
    scaled[13] = -1.;
    let poses = [pose([2., 0., 0.], BEND), pose([0., 3., 0.], scaled)];
    let mut encoder = gpu.device().create_command_encoder(&Default::default());
    let out: Vec<_> = poses
        .iter()
        .map(|p| instance(&gpu, &kernel, &inputs, &mut encoder, p))
        .collect();
    gpu.queue().submit([encoder.finish()]);
    let cpu: Vec<_> = poses
        .iter()
        .map(|p| skin_cpu(&mesh, Some(p)).unwrap())
        .collect();
    let (a, b) = (read(&gpu, &out[0]), read(&gpu, &out[1]));
    for (actual, expected) in [&a, &b]
        .into_iter()
        .zip([records(&cpu[0], &pairs), records(&cpu[1], &pairs)])
    {
        assert_eq!(actual.len(), expected.len());
        for (i, (x, y)) in actual.iter().zip(&expected).enumerate() {
            assert!(close(*x, *y), "component {i}: {x} vs {y}");
        }
    }
    // Duplicated rest vertices keep their own posed endpoint in the (0,1) edge.
    assert_ne!(cpu[0].positions[0], cpu[0].positions[1]);
    assert_ne!(a, b, "one instance's edges must not overwrite another's");
    std::mem::forget(gpu);
}
