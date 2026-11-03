//! S18 `skin` pass resources: pose validation, the shared joint palette (one block per
//! instance, one upload per frame), overlay-only compute (posed stream 0 parity per
//! instance) and the rest/posed selection across enable, disable and re-enable.
#![cfg(not(target_arch = "wasm32"))]

use std::rc::Rc;

use raa_core::{GpuContext, testing::software_gpu, wgpu};
use raa_lit::{PALETTE_BLOCK, RestMeshView};
use raa_pipeline::{FeaturePass, LightingResources};
use raa_skin::{
    SKIN_FEATURE_BIT, SKIN_ORDER, SKIN_PARITY_EPS, SkinFrameStats, SkinMesh, SkinPass,
    SkinVertexInputs, skin_cpu, validate_pose,
};

const ID: [f32; 16] = [
    1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
];

fn fixture() -> SkinMesh {
    let rest = raa_mesh::from_js_arrays(
        &[0., 0., 0., 1., 0., 0., 0., 1., 0.],
        &[0., 0., 1.].repeat(3),
        None,
        None,
        None,
        &[0, 1, 2],
        &[],
    )
    .unwrap();
    SkinMesh::new(rest, &[0; 12], &[1., 0., 0., 0.].repeat(3)).unwrap()
}

fn pose(t: [f32; 3]) -> Vec<f32> {
    let mut m = ID;
    m[12..15].copy_from_slice(&t);
    m.to_vec()
}

fn buffer(gpu: &GpuContext, size: u64) -> wgpu::Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::INDEX
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn rest(gpu: &GpuContext, mesh: &SkinMesh) -> (RestMeshView, wgpu::Buffer) {
    let vertices = mesh.rest.vertex_count() as u64;
    let stream0 = buffer(gpu, vertices * 24);
    let view = RestMeshView::new(
        stream0.clone(),
        buffer(gpu, vertices * 40),
        buffer(gpu, 12),
        Some(buffer(gpu, 3 * 24)),
    )
    .unwrap();
    (view, stream0)
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

fn check(gpu: &GpuContext, mesh: &SkinMesh, matrices: &[f32], stream0: &wgpu::Buffer) {
    let cpu = skin_cpu(mesh, Some(matrices)).unwrap();
    let actual = read(gpu, stream0);
    for (i, expected) in cpu.positions.iter().enumerate() {
        for (c, e) in expected.iter().enumerate() {
            let a = actual[i * 6 + c];
            assert!(
                a.is_finite() && (a - e).abs() <= SKIN_PARITY_EPS,
                "vertex {i} component {c}"
            );
        }
    }
}

fn encode(pass: &mut SkinPass, gpu: &GpuContext) -> SkinFrameStats {
    let mut encoder = gpu.device().create_command_encoder(&Default::default());
    let stats = pass.encode(gpu, &mut encoder);
    gpu.queue().submit([encoder.finish()]);
    stats
}

fn pass(gpu: &GpuContext, mesh: &SkinMesh, instances: &[u32]) -> SkinPass {
    let mut pass = SkinPass::new(gpu);
    let inputs = Rc::new(SkinVertexInputs::new(gpu, mesh).expect("skin inputs"));
    for &instance in instances {
        let (view, _) = rest(gpu, mesh);
        pass.add_instance(instance, inputs.clone(), view).unwrap();
    }
    pass
}

/// Asks for editor overlay geometry of `tags` (what the lit editor does on outline).
fn overlay(pass: &SkinPass, tags: &[u32]) {
    pass.views()
        .draw_source()
        .set_overlay_requests(false, [], tags.iter().copied());
}

#[test]
fn pose_lengths_are_validated() {
    assert_eq!(validate_pose(&ID.repeat(2)).unwrap(), 2);
    for bad in [vec![], ID[..15].to_vec(), vec![f32::NAN; 16]] {
        assert_eq!(validate_pose(&bad).unwrap_err().code(), "skin.pose_length");
    }
    assert_eq!(
        validate_pose(&ID.repeat(257)).unwrap_err().code(),
        "skin.cap_joints"
    );
}

#[test]
fn poses_share_one_palette_and_one_upload_per_frame_without_compute() {
    let Some(gpu) = software_gpu() else {
        panic!("palette upload requires an adapter; no comparison executed");
    };
    let mesh = fixture();
    let tags: Vec<u32> = (0..60).collect();
    let mut pass = pass(&gpu, &mesh, &tags);
    for &tag in &tags {
        pass.set_pose(&gpu, tag, &pose([tag as f32, 0., 0.]))
            .unwrap();
    }
    let views = pass.views();
    assert_eq!(views.palette_matrices(), 60 * PALETTE_BLOCK);
    let palette = views.palette_buffer().expect("one palette buffer");
    for &tag in &tags {
        let (base, capacity) = views.palette_block(tag).expect("block");
        assert_eq!((base % PALETTE_BLOCK, capacity), (0, PALETTE_BLOCK));
        let cpu = views.palette_cpu();
        let at = base as usize * 16;
        assert_eq!(&cpu[at..at + 16], &pose([tag as f32, 0., 0.])[..]);
    }
    let first = encode(&mut pass, &gpu);
    assert_eq!(
        first,
        SkinFrameStats {
            posed: 60,
            palette_writes: 1,
            dispatches: 0,
            overlays: 0
        },
        "the game path: one palette write, zero compute"
    );
    let gpu_palette = read(&gpu, &palette);
    assert_eq!(&gpu_palette[..60 * 4 * 16], &pass.views().palette_cpu()[..]);
    let idle = encode(&mut pass, &gpu);
    assert_eq!(
        (idle.palette_writes, idle.dispatches),
        (0, 0),
        "no pose change"
    );
    for &tag in &tags {
        pass.set_pose(&gpu, tag, &pose([0., tag as f32, 0.]))
            .unwrap();
    }
    let again = encode(&mut pass, &gpu);
    assert_eq!(
        (again.palette_writes, again.dispatches),
        (1, 0),
        "60 poses, one write"
    );
    assert_eq!(pass.views().palette_buffer(), Some(palette), "no regrowth");
    pass.remove_instance(5);
    pass.add_instance(
        99,
        Rc::new(SkinVertexInputs::new(&gpu, &mesh).unwrap()),
        rest(&gpu, &mesh).0,
    )
    .unwrap();
    pass.set_pose(&gpu, 99, &pose([1., 1., 1.])).unwrap();
    assert_eq!(
        pass.views().palette_block(99),
        Some((5 * PALETTE_BLOCK, PALETTE_BLOCK))
    );
    std::mem::forget(gpu);
}

#[test]
fn instances_keep_independent_overlay_poses() {
    let Some(gpu) = software_gpu() else {
        panic!("pose parity requires an adapter; no comparison executed");
    };
    let mesh = fixture();
    let mut pass = pass(&gpu, &mesh, &[1, 2, 3]);
    let poses = [pose([2., 0., 0.]), pose([0., 3., 0.]), pose([0., 0., 5.])];
    for (instance, p) in [1, 2, 3].into_iter().zip(&poses) {
        pass.set_pose(&gpu, instance, p).unwrap();
    }
    overlay(&pass, &[1, 2]);
    let stats = encode(&mut pass, &gpu);
    // Vertices + edges for each overlaid instance; instance 3 is drawn but not overlaid.
    assert_eq!((stats.dispatches, stats.overlays), (4, 2));
    assert!(pass.views().skinned_stream0(3).is_none());
    for (instance, p) in [1, 2].into_iter().zip(&poses) {
        let stream0 = pass
            .views()
            .skinned_stream0(instance)
            .expect("overlay view");
        check(&gpu, &mesh, p, stream0);
    }
    overlay(&pass, &[]);
    let stats = encode(&mut pass, &gpu);
    assert_eq!(
        (stats.dispatches, stats.overlays),
        (0, 0),
        "overlay off: no compute"
    );
    std::mem::forget(gpu);
}

#[test]
fn missing_or_toggled_skinning_selects_rest_or_fresh_poses() {
    let Some(gpu) = software_gpu() else {
        panic!("enable-state parity requires an adapter; no comparison executed");
    };
    let mesh = fixture();
    let (view, rest0) = rest(&gpu, &mesh);
    let mut pass = SkinPass::new(&gpu);
    assert_eq!((pass.name(), pass.order()), ("skin", 15));
    assert_eq!((SKIN_ORDER, SKIN_FEATURE_BIT), (15, 20));
    pass.add_instance(
        7,
        Rc::new(SkinVertexInputs::new(&gpu, &mesh).expect("skin inputs")),
        view,
    )
    .unwrap();
    overlay(&pass, &[7]);
    let source = pass.views().draw_source();
    assert!(!source.posed(7));
    assert!(pass.views().skinned_stream0(7).is_none());
    assert!(std::ptr::eq(pass.views().stream0_for(7, &rest0), &rest0));

    pass.set_pose(&gpu, 7, &pose([1., 0., 0.])).unwrap();
    assert!(source.posed(7));
    let mut lighting = LightingResources::new(&gpu);
    pass.set_enabled(false, &mut lighting);
    assert!(!source.posed(7), "disabled skinning draws rest");
    assert!(pass.views().skinned_stream0(7).is_none());
    assert_eq!(encode(&mut pass, &gpu).dispatches, 0);
    let second = pose([0., 0., 4.]);
    pass.set_pose(&gpu, 7, &second).unwrap();
    pass.set_enabled(true, &mut lighting);
    assert!(source.posed(7));
    assert_eq!(encode(&mut pass, &gpu).dispatches, 2);
    let stream0 = pass.views().skinned_stream0(7).expect("re-enabled view");
    check(&gpu, &mesh, &second, stream0);
    pass.clear_pose(7);
    assert!(!source.posed(7));
    assert!(pass.views().skinned_stream0(7).is_none());
    pass.remove_instance(7);
    assert!(pass.views().skinned_stream0(7).is_none());
    assert_eq!(pass.views().palette_matrices(), 0, "every block freed");
    std::mem::forget(gpu);
}
