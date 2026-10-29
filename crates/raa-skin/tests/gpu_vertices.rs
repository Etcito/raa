#![cfg(not(target_arch = "wasm32"))]
use raa_core::wgpu::util::DeviceExt;
use raa_core::{GpuContext, testing::software_gpu, wgpu};
use raa_skin::{SKIN_PARITY_EPS, SkinMesh, SkinVertexInputs, SkinVertexKernel, skin_cpu};

const ID: [f32; 16] = [
    1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
];
const BEND: [f32; 16] = [
    0., 1., 0., 0., -1., 0., 0., 0., 0., 0., 1., 0., 1., -1., 0., 1.,
];

fn fixture() -> SkinMesh {
    // 65 vertices exercise both the workgroup boundary and the tail guard.
    let rest = raa_mesh::from_js_arrays(
        &[2., 0., 1.].repeat(65),
        &[1., 0., 0.].repeat(65),
        Some(&[0.25, 0.75].repeat(65)),
        Some(&[0., 1., 0., -1.].repeat(65)),
        None,
        &[0, 1, 2],
        &[],
    )
    .unwrap();
    let mut mesh = SkinMesh::new(
        rest,
        &[0, 1, 2, 3].repeat(65),
        &[0.1, 0.2, 0.3, 0.4].repeat(65),
    )
    .unwrap();
    mesh.weights[..4].copy_from_slice(&[1., 0., 0., 0.]);
    mesh.weights[4..8].copy_from_slice(&[0., 1., 0., 0.]);
    mesh.weights[8..12].copy_from_slice(&[0.25, 0.75, 0., 0.]);
    mesh
}
fn close(a: f32, b: f32) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= SKIN_PARITY_EPS
}
#[test]
fn parity_rejects_nonfinite_and_excess_absolute_error() {
    assert!(close(0., SKIN_PARITY_EPS));
    assert!(!close(0., SKIN_PARITY_EPS * 1.01));
    for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(!close(x, x));
        assert!(!close(0., x));
        assert!(!close(x, 0.));
    }
    assert!(!close(1000., 1000.001));
}
fn upload(gpu: &GpuContext, bytes: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    gpu.device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytes,
            usage: usage | wgpu::BufferUsages::COPY_SRC,
        })
}
fn read(gpu: &GpuContext, buffer: &wgpu::Buffer) -> Vec<u8> {
    let staging = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    gpu.queue().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    rx.recv().unwrap().unwrap();
    let bytes = staging.slice(..).get_mapped_range().unwrap().to_vec();
    staging.unmap();
    bytes
}
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect()
}
#[test]
fn gpu_vertices_match_cpu_and_preserve_shared_attributes() {
    let Some(gpu) = software_gpu() else {
        panic!("GPU parity requires an adapter; no comparison executed");
    };
    let mesh = fixture();
    let inputs = SkinVertexInputs::new(&gpu, &mesh);
    let kernel = SkinVertexKernel::new(&gpu);
    let mut stream1 = vec![];
    mesh.rest.write_stream1_bytes(&mut stream1);
    let mut indices = vec![];
    mesh.rest.write_index_bytes(&mut indices);
    let shared1 = upload(&gpu, &stream1, wgpu::BufferUsages::VERTEX);
    let shared_indices = upload(&gpu, &indices, wgpu::BufferUsages::INDEX);
    let mut scale = ID;
    scale[0] = 3.;
    scale[5] = 2.;
    scale[12] = 4.;
    let mut translated = ID;
    translated[13] = -2.;
    translated[14] = 3.;
    let poses = [
        ID.repeat(4),
        [ID, BEND, scale, translated].concat(),
        [0.; 64].to_vec(),
    ];
    let mut outputs = vec![];
    let mut encoder = gpu.device().create_command_encoder(&Default::default());
    for pose in &poses {
        let bytes: Vec<_> = pose.iter().flat_map(|x| x.to_le_bytes()).collect();
        let pose_buffer = upload(&gpu, &bytes, wgpu::BufferUsages::STORAGE);
        let s0 = upload(&gpu, &vec![0; 65 * 24], wgpu::BufferUsages::STORAGE);
        let tangents = upload(&gpu, &vec![0; 65 * 16], wgpu::BufferUsages::STORAGE);
        let binding = kernel.bind(&gpu, &inputs, &pose_buffer, &s0, &tangents);
        kernel.encode(&mut encoder, &binding, inputs.vertices);
        outputs.push((s0, tangents));
    }
    gpu.queue().submit([encoder.finish()]);
    for (pose, (s0, t)) in poses.iter().zip(&outputs) {
        let cpu = skin_cpu(&mesh, Some(pose)).unwrap();
        let actual = floats(&read(&gpu, s0));
        let tangent_bytes = read(&gpu, t);
        let tangents = floats(&tangent_bytes);
        for i in 0..65 {
            for c in 0..3 {
                assert!(
                    close(actual[i * 6 + c], cpu.positions[i][c]),
                    "position {i}/{c}"
                );
                assert!(
                    close(actual[i * 6 + 3 + c], cpu.normals[i][c]),
                    "normal {i}/{c}"
                );
                assert!(
                    close(tangents[i * 4 + c], cpu.tangents[i][c]),
                    "tangent {i}/{c}"
                );
            }
            assert_eq!(
                &tangent_bytes[i * 16 + 12..i * 16 + 16],
                &cpu.tangents[i][3].to_le_bytes()
            );
        }
    }
    assert_eq!(read(&gpu, &inputs.buffer), SkinVertexInputs::bytes(&mesh));
    assert_eq!(read(&gpu, &shared1), stream1);
    assert_eq!(read(&gpu, &shared_indices), indices);
    std::mem::forget(gpu);
}

/// Run with a supplied Node WebGPU module:
/// SKIN_WEBGPU_MODULE=/path/to/webgpu/index.js cargo test -p raa-skin --test gpu_vertices node_compatibility -- --ignored
#[test]
#[ignore = "requires the Node webgpu package; native parity is not a substitute"]
fn node_compatibility() {
    let mesh = fixture();
    let mut scale = ID;
    scale[0] = 3.;
    scale[5] = 2.;
    scale[12] = 4.;
    let mut translated = ID;
    translated[13] = -2.;
    translated[14] = 3.;
    let poses = [
        ID.repeat(4),
        [ID, BEND, scale, translated].concat(),
        [0.; 64].to_vec(),
    ];
    let expected: Vec<Vec<f32>> = poses
        .iter()
        .map(|pose| {
            let cpu = skin_cpu(&mesh, Some(pose)).unwrap();
            (0..65)
                .flat_map(|i| {
                    cpu.positions[i]
                        .into_iter()
                        .chain(cpu.normals[i])
                        .chain(cpu.tangents[i])
                })
                .collect()
        })
        .collect();
    let status = std::process::Command::new("node")
        .args(["--input-type=module", "--eval", NODE_PROOF])
        .env("SKIN_SHADER", raa_skin::SKIN_WGSL)
        .env(
            "SKIN_INPUT",
            format!("{:?}", SkinVertexInputs::bytes(&mesh)),
        )
        .env("SKIN_POSES", format!("{poses:?}"))
        .env("SKIN_EXPECTED", format!("{expected:?}"))
        .env("SKIN_EPS", SKIN_PARITY_EPS.to_string())
        .status()
        .expect("launch Node WebGPU proof");
    assert!(status.success(), "Node compatibility parity failed");
}

const NODE_PROOF: &str = r#"
import assert from 'node:assert/strict';
const {create, globals} = await import(process.env.SKIN_WEBGPU_MODULE || 'webgpu');
Object.assign(globalThis, globals);
const gpu = create([]);
const adapter = await gpu.requestAdapter({featureLevel: 'compatibility'});
assert(adapter, 'compatibility adapter required');
const device = await adapter.requestDevice();
device.addEventListener('uncapturederror', e => { throw e.error; });
const module = device.createShaderModule({code: process.env.SKIN_SHADER});
const pipeline = await device.createComputePipelineAsync({layout:'auto', compute:{module, entryPoint:'skin_vertices'}});
const input = new Uint8Array(JSON.parse(process.env.SKIN_INPUT));
function buffer(bytes, usage) {
  const b = device.createBuffer({size:bytes.byteLength, usage, mappedAtCreation:true});
  new Uint8Array(b.getMappedRange()).set(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength));
  b.unmap(); return b;
}
const rest = buffer(input, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
const expected = JSON.parse(process.env.SKIN_EXPECTED);
const poses = JSON.parse(process.env.SKIN_POSES);
const eps = Number(process.env.SKIN_EPS);
const encoder = device.createCommandEncoder();
const outputs = poses.map(pose => {
  const p = buffer(new Float32Array(pose), GPUBufferUsage.STORAGE);
  const s = buffer(new Uint8Array(65*24), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const t = buffer(new Uint8Array(65*16), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const bind = device.createBindGroup({layout:pipeline.getBindGroupLayout(0), entries:
    [rest,p,s,t].map((buffer,binding)=>({binding,resource:{buffer}}))});
  const pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0,bind);
  pass.dispatchWorkgroups(2); pass.end();
  return [s,t];
});
device.queue.submit([encoder.finish()]);
async function read(b) {
  const staging = device.createBuffer({size:b.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  const e = device.createCommandEncoder(); e.copyBufferToBuffer(b,0,staging,0,b.size);
  device.queue.submit([e.finish()]); await staging.mapAsync(GPUMapMode.READ);
  const bytes = staging.getMappedRange().slice(0); staging.unmap(); return bytes;
}
for (let j=0;j<outputs.length;j++) {
  const s = new Float32Array(await read(outputs[j][0]));
  const tb = await read(outputs[j][1]); const t = new Float32Array(tb);
  for (let i=0;i<65;i++) {
    const actual = [...s.slice(i*6,i*6+6),...t.slice(i*4,i*4+4)];
    for (let c=0;c<9;c++) {
      const a=actual[c], b=expected[j][i*10+c];
      assert(Number.isFinite(a) && Number.isFinite(b) && Math.abs(a-b)<=eps, `pose ${j}, vertex ${i}, component ${c}: ${a} vs ${b}`);
    }
    assert.equal(new DataView(tb).getUint32(i*16+12,true), new DataView(input.buffer).getUint32(i*64+36,true));
  }
}
assert.deepEqual(new Uint8Array(await read(rest)),input);
console.log('Node compatibility skin.wgsl / skin_cpu parity passed', adapter.info);
device.destroy();
"#;
