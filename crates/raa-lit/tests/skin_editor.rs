//! S18 editor overlays: wireframe edges and selection draws follow the selected
//! instance's posed resources, and disabled skinning returns both to rest.

#![cfg(not(target_arch = "wasm32"))]

use raa_core::testing::{skip, software_gpu};
use raa_core::{GpuContext, wgpu};
use raa_lit::skin_editor::{OverlayGeometry, REST_EDGE_STRIDE};
use raa_lit::{RestMeshView, SKINNED_EDGE_STRIDE, SkinViews};

fn buffer(gpu: &GpuContext, label: &str, size: u64) -> wgpu::Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Four vertices, six indices and four edges: a quad shared by every instance.
fn rest(gpu: &GpuContext) -> (RestMeshView, [wgpu::Buffer; 4]) {
    let s0 = buffer(gpu, "rest.s0", 4 * 24);
    let s1 = buffer(gpu, "rest.s1", 4 * 40);
    let ix = buffer(gpu, "rest.ix", 6 * 4);
    let ed = buffer(gpu, "rest.ed", 4 * REST_EDGE_STRIDE);
    let view = RestMeshView::new(s0.clone(), s1.clone(), ix.clone(), Some(ed.clone())).expect("rest");
    (view, [s0, s1, ix, ed])
}

fn select(skins: &SkinViews, tag: u32, rest0: &wgpu::Buffer, edges: &wgpu::Buffer) -> OverlayGeometry {
    OverlayGeometry::select(&skins.draw_source(), tag, rest0, Some(edges))
}

#[test]
fn posed_instances_route_both_overlays_to_their_own_views() {
    let test = "posed_instances_route_both_overlays_to_their_own_views";
    let Some(gpu) = software_gpu() else { skip(test, "software_gpu returned None"); return; };
    let (rest, [rest0, _, _, rest_edges]) = rest(&gpu);
    let mut skins = SkinViews::default();
    skins.add_skinned_view(&gpu, 1, rest.clone()).expect("one");
    skins.add_skinned_view(&gpu, 2, rest).expect("two");
    let a = select(&skins, 1, &rest0, &rest_edges);
    let b = select(&skins, 2, &rest0, &rest_edges);
    assert!(a.posed && b.posed);
    assert_eq!(a.edge_stride, SKINNED_EDGE_STRIDE);
    assert_eq!(a.edge_count(), 4);
    assert_eq!(a.edges.as_ref().expect("posed edges").size(), 4 * SKINNED_EDGE_STRIDE);
    assert_eq!(a.stream0, *skins.skinned_stream0(1).expect("posed stream"));
    assert_eq!(b.stream0, *skins.skinned_stream0(2).expect("posed stream"));
    assert_ne!(a.stream0, b.stream0, "instances keep separate posed streams");
    assert_ne!(a.edges, b.edges, "instances keep separate posed segments");
}

#[test]
fn absent_and_removed_views_select_rest() {
    let test = "absent_and_removed_views_select_rest";
    let Some(gpu) = software_gpu() else { skip(test, "software_gpu returned None"); return; };
    let (rest, [rest0, _, _, rest_edges]) = rest(&gpu);
    let mut skins = SkinViews::default();
    skins.add_skinned_view(&gpu, 3, rest).expect("view");
    skins.remove_skinned_view(3);
    let gone = select(&skins, 3, &rest0, &rest_edges);
    assert!(!gone.posed);
    assert_eq!(gone.stream0, rest0);
    assert_eq!(gone.edges.as_ref(), Some(&rest_edges));
    assert_eq!(gone.edge_stride, REST_EDGE_STRIDE);
    assert_eq!(gone.edge_count(), 4);
    let absent = select(&skins, 99, &rest0, &rest_edges);
    assert!(!absent.posed);
    assert_eq!(absent.stream0, rest0);
    assert_eq!(absent.edges.as_ref(), Some(&rest_edges));
}

#[test]
fn disabled_skinning_returns_both_overlays_to_rest_in_the_same_frame() {
    let test = "disabled_skinning_returns_both_overlays_to_rest_in_the_same_frame";
    let Some(gpu) = software_gpu() else { skip(test, "software_gpu returned None"); return; };
    let (rest, [rest0, _, _, rest_edges]) = rest(&gpu);
    let mut skins = SkinViews::default();
    skins.add_skinned_view(&gpu, 5, rest).expect("view");
    let source = skins.draw_source();
    let posed = OverlayGeometry::select(&source, 5, &rest0, Some(&rest_edges));
    assert!(posed.posed);
    skins.set_skinning_enabled(false);
    // Same frame and same connected source: both overlays read rest now.
    let rest_pose = OverlayGeometry::select(&source, 5, &rest0, Some(&rest_edges));
    assert!(!rest_pose.posed);
    assert_eq!(rest_pose.stream0, rest0);
    assert_eq!(rest_pose.edges.as_ref(), Some(&rest_edges));
    assert_eq!(rest_pose.edge_stride, REST_EDGE_STRIDE);
    skins.set_skinning_enabled(true);
    let again = OverlayGeometry::select(&source, 5, &rest0, Some(&rest_edges));
    assert!(again.posed);
    assert_eq!(again.stream0, posed.stream0);
    assert_eq!(again.edges, posed.edges);
}
