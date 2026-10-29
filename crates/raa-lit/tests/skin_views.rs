//! S18 skinned views: rest fallback and shared rest resources, and instance isolation.

#![cfg(not(target_arch = "wasm32"))]

use raa_core::testing::{skip, software_gpu};
use raa_core::{GpuContext, wgpu};
use raa_lit::{RestMeshView, SkinViews, SKINNED_EDGE_STRIDE, SKINNED_STREAM0_STRIDE, SKINNED_TANGENT_STRIDE};

fn buffer(gpu: &GpuContext, label: &str, size: u64) -> wgpu::Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn rest(gpu: &GpuContext, vertices: u32, indices: u32, edges: u32) -> (RestMeshView, [wgpu::Buffer; 4]) {
    let s0 = buffer(gpu, "rest.s0", u64::from(vertices) * 24);
    let s1 = buffer(gpu, "rest.s1", u64::from(vertices) * 40);
    let ix = buffer(gpu, "rest.ix", u64::from(indices) * 4);
    let ed = buffer(gpu, "rest.ed", u64::from(edges) * 24);
    let view =
        RestMeshView::new(s0.clone(), s1.clone(), ix.clone(), Some(ed.clone())).expect("rest");
    (view, [s0, s1, ix, ed])
}

fn read(gpu: &GpuContext, buffer: &wgpu::Buffer, len: u64) -> Vec<u8> {
    let readback = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device().create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &readback, 0, len);
    encoder.map_buffer_on_submit(&readback, wgpu::MapMode::Read, .., |_| {});
    gpu.queue().submit([encoder.finish()]);
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let bytes = readback.get_mapped_range(..).expect("range").to_vec();
    readback.unmap();
    bytes
}

#[test]
fn absent_or_disabled_skinning_selects_rest_resources() {
    let test = "absent_or_disabled_skinning_selects_rest_resources";
    let Some(gpu) = software_gpu() else { skip(test, "software_gpu returned None"); return; };
    let (rest, [rest0, rest1, rest_index, rest_edges]) = rest(&gpu, 4, 6, 4);
    let mut skins = SkinViews::default();
    assert!(skins.skinned_stream0(7).is_none());
    assert!(std::ptr::eq(skins.stream0_for(7, &rest0), &rest0));
    assert!(std::ptr::eq(skins.edges_for(7, Some(&rest_edges)).expect("rest"), &rest_edges));
    skins.add_skinned_view(&gpu, 7, rest.clone()).expect("add");
    {
        let view = skins.skinned_view(7).expect("active view");
        gpu.queue().write_buffer(&rest1, 0, &[0x33; 160]);
        gpu.queue().write_buffer(&rest_index, 0, &[0x44; 24]);
        assert!(read(&gpu, &view.rest.stream1, 160).iter().all(|v| *v == 0x33));
        assert!(read(&gpu, &view.rest.index, 24).iter().all(|v| *v == 0x44));
        let sizes = (
            view.rest.stream1.size(),
            view.rest.index.size(),
            view.stream0.size(),
            view.tangents.size(),
            view.edges.as_ref().expect("posed").size(),
        );
        assert_eq!(
            sizes,
            (
                rest1.size(),
                rest_index.size(),
                4 * SKINNED_STREAM0_STRIDE,
                4 * SKINNED_TANGENT_STRIDE,
                4 * SKINNED_EDGE_STRIDE
            )
        );
        assert!(std::ptr::eq(skins.stream0_for(7, &rest0), &view.stream0));
        assert!(std::ptr::eq(
            skins.skinned_tangents(7).expect("tangents"),
            &view.tangents
        ));
        assert!(std::ptr::eq(
            skins.edges_for(7, Some(&rest_edges)).expect("posed"),
            view.edges.as_ref().expect("edges")
        ));
    }
    skins.set_skinning_enabled(false);
    assert!(skins.skinned_view(7).is_none());
    assert!(std::ptr::eq(skins.stream0_for(7, &rest0), &rest0));
    assert!(std::ptr::eq(skins.edges_for(7, Some(&rest_edges)).expect("rest"), &rest_edges));
    skins.set_skinning_enabled(true);
    assert!(skins.skinned_stream0(7).is_some());
}

#[test]
fn views_are_isolated_per_instance() {
    let test = "views_are_isolated_per_instance";
    let Some(gpu) = software_gpu() else { skip(test, "software_gpu returned None"); return; };
    let (quad, _) = rest(&gpu, 4, 6, 4);
    let (triangle, _) = rest(&gpu, 3, 3, 3);
    let mut skins = SkinViews::default();
    skins.add_skinned_view(&gpu, 1, quad.clone()).expect("quad");
    skins.add_skinned_view(&gpu, 2, triangle).expect("triangle");
    let a = skins.skinned_stream0(1).expect("1").clone();
    let b = skins.skinned_stream0(2).expect("2").clone();
    assert_eq!((a.size(), b.size()), (96, 3 * SKINNED_STREAM0_STRIDE));
    gpu.queue().write_buffer(&a, 0, &[0x11; 96]);
    gpu.queue().write_buffer(&b, 0, &[0x22; 72]);
    assert!(read(&gpu, &a, 96).iter().all(|v| *v == 0x11));
    assert!(read(&gpu, &b, 72).iter().all(|v| *v == 0x22));
    skins.remove_skinned_view(1);
    assert!(skins.skinned_stream0(1).is_none());
    assert!(read(&gpu, &b, 72).iter().all(|v| *v == 0x22));
    skins.add_skinned_view(&gpu, 1, quad).expect("re-add");
    assert_eq!(skins.skinned_stream0(1).expect("1").size(), 96);
    assert!(read(&gpu, &b, 72).iter().all(|v| *v == 0x22));
}
