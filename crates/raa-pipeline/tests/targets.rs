//! Native GPU test of the pipeline targets: S12 sizes, formats and usages, HDR chain
//! swap/reset and `ensure`. Skips with the SKIP line when no software adapter is available.

#![cfg(not(target_arch = "wasm32"))]

use raa_core::testing::{skip, software_gpu};
use raa_core::{GpuContext, wgpu};
use raa_pipeline::{HDR_FORMAT, Targets};

const GBUFFER: wgpu::TextureUsages =
    wgpu::TextureUsages::RENDER_ATTACHMENT.union(wgpu::TextureUsages::TEXTURE_BINDING);

fn check(texture: &wgpu::Texture, size: (u32, u32), format: wgpu::TextureFormat) {
    assert_eq!((texture.width(), texture.height()), size);
    assert_eq!(texture.depth_or_array_layers(), 1);
    assert_eq!(texture.format(), format);
    assert_eq!(texture.sample_count(), 1);
    assert_eq!(texture.mip_level_count(), 1);
}

fn targets_have_s12_sizes_formats_and_usages(gpu: &GpuContext) {
    let targets = Targets::new(gpu, (64, 48));
    assert_eq!(targets.size(), (64, 48));

    check(
        targets.depth_texture(),
        (64, 48),
        wgpu::TextureFormat::Depth32Float,
    );
    assert_eq!(targets.depth_texture().usage(), GBUFFER);
    check(
        targets.normal_texture(),
        (64, 48),
        wgpu::TextureFormat::Rgba16Float,
    );
    assert_eq!(targets.normal_texture().usage(), GBUFFER);
    check(
        targets.material_texture(),
        (64, 48),
        wgpu::TextureFormat::Rgba8Unorm,
    );
    assert_eq!(targets.material_texture().usage(), GBUFFER);

    assert_eq!(HDR_FORMAT, wgpu::TextureFormat::Rgba16Float);
    let hdr_usage = GBUFFER
        | wgpu::TextureUsages::STORAGE_BINDING
        | wgpu::TextureUsages::COPY_SRC
        | wgpu::TextureUsages::COPY_DST;
    for texture in [targets.hdr().src_texture(), targets.hdr().dst_texture()] {
        check(texture, (64, 48), HDR_FORMAT);
        assert_eq!(texture.usage(), hdr_usage);
    }
}

fn hdr_chain_swaps_and_resets_on_the_gpu_textures(gpu: &GpuContext) {
    let targets = Targets::new(gpu, (16, 16));
    let hdr = targets.hdr();
    let a = hdr.dst_texture().clone();
    let b = hdr.src_texture().clone();
    assert_ne!(a, b);

    hdr.swap();
    assert_eq!(hdr.src_texture(), &a);
    assert_eq!(hdr.dst_texture(), &b);
    hdr.swap();
    hdr.swap();
    assert_eq!(hdr.src_texture(), &a);

    hdr.reset();
    assert_eq!(hdr.dst_texture(), &a);
    assert_eq!(hdr.src_texture(), &b);
}

fn ensure_recreates_only_when_the_size_changes(gpu: &GpuContext) {
    let mut targets = Targets::new(gpu, (32, 32));
    let depth = targets.depth_texture().clone();
    assert!(!targets.ensure(gpu, (32, 32)));
    assert_eq!(targets.depth_texture(), &depth);

    assert!(targets.ensure(gpu, (40, 24)));
    assert_eq!(targets.size(), (40, 24));
    assert_ne!(targets.depth_texture(), &depth);
    check(targets.hdr().dst_texture(), (40, 24), HDR_FORMAT);

    assert!(targets.ensure(gpu, (0, 0)));
    assert_eq!(targets.size(), (1, 1));
    assert!(!targets.ensure(gpu, (0, 1)));
}

/// Clears `hdr.dst()` (RENDER_ATTACHMENT), copies it out (COPY_SRC) and reads the f16 bits
/// back. 64 px x 8 B makes one 512 B row, a multiple of COPY_BYTES_PER_ROW_ALIGNMENT.
fn hdr_dst_clears_and_reads_back(gpu: &GpuContext) {
    let targets = Targets::new(gpu, (64, 2));
    let hdr = targets.hdr();
    let mut encoder = gpu.device().create_command_encoder(&Default::default());
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("clear hdr dst"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: hdr.dst(),
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color {
                    r: 1.0,
                    g: 0.5,
                    b: 0.25,
                    a: 2.0,
                }),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    let row_bytes = 64 * 8;
    let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("hdr readback"),
        size: u64::from(row_bytes) * 2,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        hdr.dst_texture().as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(2),
            },
        },
        hdr.dst_texture().size(),
    );
    gpu.queue().submit([encoder.finish()]);

    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    gpu.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll");
    rx.recv().expect("map callback").expect("buffer maps");
    let mapped = slice.get_mapped_range().expect("mapped range");
    // f16 bits of 1.0, 0.5, 0.25 and 2.0
    let expected = [0x3C00u16, 0x3800, 0x3400, 0x4000];
    for (i, px) in mapped.chunks_exact(8).enumerate() {
        let got: Vec<u16> = px
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(got, expected, "pixel {i}");
    }
    drop(mapped);
    buffer.unmap();
}

/// One GPU context for every check. It is leaked on purpose: on the GL backend of the
/// WSL D3D12 driver, a process that dropped a device after creating these Rgba16Float
/// targets crashed at exit (SIGSEGV after every assertion passed); never dropping the
/// device avoids that driver teardown path. Test-only, the process ends right after.
#[test]
fn targets_on_the_software_gpu() {
    let Some(gpu) = software_gpu() else {
        skip(
            "targets_on_the_software_gpu",
            "software_gpu returned None; see the software_gpu line above",
        );
        return;
    };
    targets_have_s12_sizes_formats_and_usages(&gpu);
    hdr_chain_swaps_and_resets_on_the_gpu_textures(&gpu);
    ensure_recreates_only_when_the_size_changes(&gpu);
    hdr_dst_clears_and_reads_back(&gpu);
    std::mem::forget(gpu);
}
