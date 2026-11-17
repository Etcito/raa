//! ResourceTable publish/get/clear, input binding, buffer entries, insertion order, replace
//! in place, and sample_kind_for (S55, contract resource-table-additions).

use raa_core::testing::{skip, software_gpu};
use raa_core::{GpuContext, wgpu};
use raa_pipeline::{ImageDesc, ImageDim, ResourceTable, SampleKind, sample_kind_for};

fn view(gpu: &GpuContext, format: wgpu::TextureFormat) -> wgpu::TextureView {
    let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("resource_table test"),
        size: wgpu::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    texture.create_view(&Default::default())
}

fn desc(format: wgpu::TextureFormat) -> ImageDesc {
    ImageDesc::new(ImageDim::D2, format, 4, 4, 1, 1)
}

fn key(pass: &str, port: &str) -> (String, String) {
    (pass.to_string(), port.to_string())
}

#[test]
fn resource_table_on_the_software_gpu() {
    let Some(gpu) = software_gpu() else {
        skip(
            "resource_table_on_the_software_gpu",
            "software_gpu returned None; see the software_gpu line above",
        );
        return;
    };
    let hdr = wgpu::TextureFormat::Rgba16Float;
    let depth = wgpu::TextureFormat::Depth32Float;
    let a_view = view(&gpu, hdr);
    let b_view = view(&gpu, depth);
    let table = ResourceTable::new();

    // (1) publish then get; unknown key.
    table.publish("forward", "color", &a_view, desc(hdr));
    assert_eq!(
        table.get("forward", "color").map(|i| i.desc),
        Some(desc(hdr))
    );
    assert!(table.get("forward", "depth").is_none());
    assert!(table.get("nope", "color").is_none());

    // (2) A, B, A again: replaced in place.
    table.publish("prepass", "depth", &b_view, desc(depth));
    table.publish("forward", "color", &b_view, desc(depth));
    assert_eq!(
        table.get("forward", "color").map(|i| i.desc),
        Some(desc(depth))
    );
    let entries = table.entries();
    let expected = vec![
        ("forward".to_string(), "color".to_string(), desc(depth)),
        ("prepass".to_string(), "depth".to_string(), desc(depth)),
    ];
    assert_eq!(entries, expected);

    // (3) buffer under C.
    let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("resource_table buffer"),
        size: 256,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    table.publish_buffer("vg.cull", "cullList", &buffer, 256);
    assert_eq!(table.buffer("vg.cull", "cullList").map(|b| b.1), Some(256));
    assert!(table.get("vg.cull", "cullList").is_none());
    assert!(table.buffer("forward", "color").is_none());
    assert_eq!(table.entries().len(), 2);
    let buffers = table.buffer_entries();
    assert_eq!(
        buffers,
        vec![("vg.cull".to_string(), "cullList".to_string(), 256)]
    );

    // (4) inputs round trip, never in entries.
    let image = table.get("prepass", "depth").expect("published depth");
    table.bind_input("ao.gtao", "depth", image);
    assert_eq!(
        table.input("ao.gtao", "depth").map(|i| i.desc),
        Some(desc(depth))
    );
    assert!(table.input("prepass", "depth").is_none());
    let keys: Vec<_> = table
        .entries()
        .into_iter()
        .map(|e| key(&e.0, &e.1))
        .collect();
    assert!(!keys.contains(&key("ao.gtao", "depth")));

    // (5) clear empties everything; publishing afterwards works.
    table.clear();
    assert!(table.entries().is_empty());
    assert!(table.buffer_entries().is_empty());
    assert!(table.input("ao.gtao", "depth").is_none());
    assert!(table.get("forward", "color").is_none());
    table.publish("tonemap", "color", &a_view, desc(hdr));
    let entries = table.entries();
    assert_eq!(
        entries,
        vec![("tonemap".to_string(), "color".to_string(), desc(hdr))]
    );
    std::mem::forget(gpu);
}

#[test]
fn sample_kind_for_maps_formats() {
    use wgpu::TextureFormat as F;
    assert_eq!(sample_kind_for(F::Depth32Float), SampleKind::Depth);
    assert_eq!(sample_kind_for(F::Rgba32Uint), SampleKind::Uint);
    assert_eq!(sample_kind_for(F::R32Float), SampleKind::UnfilterableFloat);
    assert_eq!(sample_kind_for(F::Rgba16Float), SampleKind::Float);
    assert_eq!(sample_kind_for(F::R8Unorm), SampleKind::Float);
    assert_eq!(sample_kind_for(F::R32Sint), SampleKind::Sint);
    let desc = ImageDesc::new(ImageDim::D2, F::Rgba16Float, 4, 4, 1, 1);
    assert_eq!(desc.sample, SampleKind::Float);
}
