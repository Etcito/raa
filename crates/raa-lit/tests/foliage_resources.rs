//! Native foliage resources (M7, contract m7-foliage-resources): the declared group-2
//! descriptors, the params byte offsets, complete-layout counting and the CPU records,
//! plus software-GPU tests of the native D3 allocation at both supported sizes and of the
//! payload validation before the initialize submission.

#![cfg(not(target_arch = "wasm32"))]

use raa_core::testing::{skip, software_gpu};
use raa_core::wgpu;
use raa_core::{GpuContext, RaaError};
use raa_lit::foliage_resources::{
    FOLIAGE_MODE_DIFFERENTIABLE, FOLIAGE_MODE_POM, FOLIAGE_MODE_SPOM, FOLIAGE_PARAMS_SIZE,
    FOLIAGE_VIEW_RECORD_SIZE, FOLIAGE_VIEW_TABLE_SIZE, FoliageParams, FoliageResources,
    FoliageSource, MicroTriangle, PackedFoliageLayers, SourceTriangle, foliage_layout_entries,
};

fn invalid_code() -> &'static str {
    RaaError::FrameGraph {
        code: "foliage.invalid",
        message: String::new(),
    }
    .code()
}

fn params(resolution: u32, mode: u32) -> FoliageParams {
    FoliageParams {
        min: [-2.0, 0.0, -2.0],
        max: [2.0, 9.0, 2.0],
        resolution,
        mode,
        translucency: 0.35,
        generation: 7,
        cutoff: 0.25,
        view_count: 2,
        texture_resolution: 64,
    }
}

fn f32_at(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

fn assert_invalid(params: &FoliageParams) {
    let err = params.validate().expect_err("params must be refused");
    assert_eq!(err.code(), invalid_code(), "{}", err.message());
}

#[test]
fn params_byte_offsets_match_the_contract() {
    let p = FoliageParams {
        min: [-1.5, -2.5, -3.5],
        max: [4.5, 5.5, 6.5],
        resolution: 32,
        mode: FOLIAGE_MODE_POM,
        translucency: 0.25,
        generation: 0x0102_0304,
        cutoff: 0.75,
        view_count: 96,
        texture_resolution: 256,
    };
    let bytes = p.to_bytes();
    assert_eq!(bytes.len(), FOLIAGE_PARAMS_SIZE);
    assert_eq!(f32_at(&bytes, 0), -1.5);
    assert_eq!(f32_at(&bytes, 4), -2.5);
    assert_eq!(f32_at(&bytes, 8), -3.5);
    assert_eq!(u32_at(&bytes, 12), 32);
    assert_eq!(f32_at(&bytes, 16), 4.5);
    assert_eq!(f32_at(&bytes, 20), 5.5);
    assert_eq!(f32_at(&bytes, 24), 6.5);
    assert_eq!(u32_at(&bytes, 28), FOLIAGE_MODE_POM);
    assert_eq!(f32_at(&bytes, 32), 0.25);
    assert_eq!(u32_at(&bytes, 36), 0x0102_0304);
    assert_eq!(f32_at(&bytes, 40), 0.75);
    assert_eq!(u32_at(&bytes, 44), 96);
    assert_eq!(u32_at(&bytes, 48), 256);
    for offset in [52, 56, 60] {
        assert_eq!(u32_at(&bytes, offset), 0, "offset {offset} is zero");
    }
    let record = FoliageParams::view_record(std::array::from_fn(|i| i as f32), 0.1, 500.0);
    assert_eq!(record.len(), FOLIAGE_VIEW_RECORD_SIZE);
    assert_eq!(f32_at(&record, 0), 0.0);
    assert_eq!(f32_at(&record, 60), 15.0);
    assert_eq!(f32_at(&record, 64), 0.1);
    assert_eq!(f32_at(&record, 68), 500.0);
    assert_eq!(f32_at(&record, 72), 0.0);
    assert_eq!(f32_at(&record, 76), 0.0);
}

#[test]
fn params_accept_both_native_sizes_and_refuse_bad_descriptors() {
    for resolution in [16, 32] {
        for mode in [FOLIAGE_MODE_SPOM, FOLIAGE_MODE_POM, 2, 3, 4, 5] {
            for texture_resolution in [64, 128, 256, 512] {
                let mut p = params(resolution, mode);
                p.texture_resolution = texture_resolution;
                p.view_count = 1;
                p.translucency = 0.0;
                p.cutoff = 1.0;
                p.validate().expect("valid descriptor");
                p.view_count = 128;
                p.translucency = 1.0;
                p.cutoff = 0.0;
                p.validate().expect("boundary descriptor");
            }
        }
    }
    let mut p = params(16, FOLIAGE_MODE_SPOM);
    p.resolution = 24;
    assert_invalid(&p);
    p.resolution = 0;
    assert_invalid(&p);
    p = params(16, 6);
    assert_invalid(&p);
    p = params(16, FOLIAGE_MODE_SPOM);
    p.translucency = -0.01;
    assert_invalid(&p);
    p.translucency = 1.01;
    assert_invalid(&p);
    p = params(16, FOLIAGE_MODE_SPOM);
    p.cutoff = f32::NAN;
    assert_invalid(&p);
    p = params(16, FOLIAGE_MODE_SPOM);
    p.view_count = 0;
    assert_invalid(&p);
    p.view_count = 129;
    assert_invalid(&p);
    p = params(16, FOLIAGE_MODE_SPOM);
    p.texture_resolution = 32;
    assert_invalid(&p);
    p.texture_resolution = 1024;
    assert_invalid(&p);
    p = params(16, FOLIAGE_MODE_SPOM);
    p.min[0] = f32::INFINITY;
    assert_invalid(&p);
    p = params(16, FOLIAGE_MODE_SPOM);
    p.max[1] = f32::NEG_INFINITY;
    assert_invalid(&p);
    p = params(16, FOLIAGE_MODE_SPOM);
    p.min = [1.0, 0.0, 0.0];
    p.max = [1.0, 1.0, 1.0];
    assert_invalid(&p);
    p = params(16, FOLIAGE_MODE_SPOM);
    p.max[2] = p.min[2] - 1.0;
    assert_invalid(&p);
}

#[test]
fn layout_entries_declare_the_group_two_resources() {
    let entries = foliage_layout_entries();
    assert_eq!(entries.len(), 6);
    let bindings: Vec<u32> = entries.iter().map(|e| e.binding).collect();
    assert_eq!(bindings, [0, 1, 2, 3, 4, 5]);
    for entry in &entries {
        assert_eq!(entry.visibility, wgpu::ShaderStages::FRAGMENT);
    }
    for (index, size) in [(0, FOLIAGE_PARAMS_SIZE), (5, FOLIAGE_VIEW_TABLE_SIZE)] {
        assert!(matches!(
            entries[index].ty,
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(n),
            } if n.get() == size as u64
        ));
    }
    for (index, dimension) in [
        (1, wgpu::TextureViewDimension::D3),
        (3, wgpu::TextureViewDimension::D2Array),
        (4, wgpu::TextureViewDimension::D2Array),
    ] {
        assert!(matches!(
            entries[index].ty,
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension,
                multisampled: false,
            } if view_dimension == dimension
        ));
    }
    assert!(matches!(
        entries[2].ty,
        wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)
    ));
    assert_eq!(FOLIAGE_VIEW_TABLE_SIZE, 128 * FOLIAGE_VIEW_RECORD_SIZE);
}

#[test]
fn complete_groups_stay_within_the_fragment_limits() {
    let entries = foliage_layout_entries();
    let sampled = entries
        .iter()
        .filter(|e| {
            matches!(e.ty, wgpu::BindingType::Texture { .. })
                && e.visibility.contains(wgpu::ShaderStages::FRAGMENT)
        })
        .count();
    let storage = entries
        .iter()
        .filter(|e| {
            matches!(
                e.ty,
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { .. },
                    ..
                }
            )
        })
        .count();
    assert_eq!(
        sampled, 3,
        "the foliage group declares three sampled textures"
    );
    assert_eq!(storage, 0, "the foliage group declares no storage buffer");
    // Declared accounting inputs: group 0 (handle.rs frame_layout_entries) adds one
    // sampled texture and the opaque lighting group 1 uses ten; P6 already used 7 of the
    // 8 fragment storage buffers.
    let frame_textures = 1;
    let opaque_lighting_textures = 10;
    let existing_storage = 7;
    let textures = frame_textures + opaque_lighting_textures + sampled;
    assert_eq!(textures, 14, "groups 0+1+2 sample 14 fragment textures");
    assert!(textures <= 16);
    assert!(existing_storage + storage <= 8);
}

#[test]
fn source_and_micro_triangle_records_are_declared() {
    let triangle = SourceTriangle {
        positions: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        normals: [[0.0, 0.0, 1.0]; 3],
        uv: [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        material: 3,
        source_tag: 11,
        foliage: true,
    };
    let source = FoliageSource {
        triangles: vec![triangle],
        min: [0.0, 0.0, 0.0],
        max: [1.0, 1.0, 0.0],
        root_world: raa_math::Mat4::IDENTITY,
        generation: 7,
    };
    assert_eq!(source.triangles[0].material, 3);
    assert_eq!(source.triangles[0].source_tag, 11);
    assert!(source.triangles[0].foliage);
    assert_eq!(source.generation, 7);
    let micro = MicroTriangle {
        p0: [0.0, 0.0, 0.0],
        p1: [1.0, 0.0, 0.0],
        p2: [0.0, 0.0, 1.0],
        normal: [0.0, 1.0, 0.0],
        opacity: 1.0,
    };
    assert_eq!(micro.normal, [0.0, 1.0, 0.0]);
    assert_eq!(micro.opacity, 1.0);
}

fn layers_texture(
    gpu: &GpuContext,
    label: &str,
    size: (u32, u32),
    count: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: count,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn packed_layers(
    gpu: &GpuContext,
    size: u32,
    count: u32,
    view_records: usize,
) -> PackedFoliageLayers {
    let views = vec![
        (
            [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
            [0.1, 100.0],
        );
        view_records
    ];
    PackedFoliageLayers {
        albedo_coverage: layers_texture(
            gpu,
            "foliage.test.albedo",
            (size, size),
            count,
            wgpu::TextureFormat::Rgba8Unorm,
        ),
        normal_depth: layers_texture(
            gpu,
            "foliage.test.normal",
            (size, size),
            count,
            wgpu::TextureFormat::Rgba16Float,
        ),
        views,
        submission: gpu.queue().submit(Vec::<wgpu::CommandBuffer>::new()),
    }
}

fn volume_bytes(resolution: u32) -> Vec<u8> {
    vec![0u8; 4 * (resolution * resolution * resolution) as usize]
}

#[test]
fn native_volume_allocation_has_both_supported_sizes_and_no_atlas_fallback() {
    let test = "native_volume_allocation_has_both_supported_sizes_and_no_atlas_fallback";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    for resolution in [16, 32] {
        let p = params(resolution, FOLIAGE_MODE_DIFFERENTIABLE);
        let mut resources = FoliageResources::new(&gpu, p).expect("allocate foliage generation");
        assert_eq!(resources.params(), p);
        let volume = resources.volume();
        assert_eq!(volume.dimension(), wgpu::TextureDimension::D3);
        assert_eq!(volume.format(), wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(volume.mip_level_count(), 1);
        assert_eq!(volume.width(), resolution);
        assert_eq!(volume.height(), resolution);
        assert_eq!(volume.depth_or_array_layers(), resolution);
        let _ = resources
            .initialize(&gpu, None, &volume_bytes(resolution))
            .expect("initialize distilled generation");
        let _ = resources.bind_group();
        let err = resources
            .initialize(&gpu, None, &volume_bytes(resolution))
            .expect_err("a second initialization is refused");
        assert_eq!(err.code(), invalid_code());
    }
}

#[test]
fn initialize_validates_payload_before_submission() {
    let test = "initialize_validates_payload_before_submission";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let p = params(16, FOLIAGE_MODE_DIFFERENTIABLE);
    let mut resources = FoliageResources::new(&gpu, p).expect("allocate foliage generation");
    let err = resources
        .initialize(&gpu, None, &[0u8; 12])
        .expect_err("a short volume payload is refused");
    assert_eq!(err.code(), invalid_code());
    let mut impostor = FoliageResources::new(&gpu, params(16, FOLIAGE_MODE_SPOM))
        .expect("allocate impostor generation");
    let err = impostor
        .initialize(&gpu, None, &volume_bytes(16))
        .expect_err("impostor mode needs packed layers");
    assert_eq!(err.code(), invalid_code());
    let wrong = packed_layers(&gpu, 32, p.view_count, p.view_count as usize);
    let err = impostor
        .initialize(&gpu, Some(&wrong), &volume_bytes(16))
        .expect_err("wrong layer dimensions are refused");
    assert_eq!(err.code(), invalid_code());
    let short_table = packed_layers(&gpu, 64, p.view_count, 1);
    let err = impostor
        .initialize(&gpu, Some(&short_table), &volume_bytes(16))
        .expect_err("a short view table is refused");
    assert_eq!(err.code(), invalid_code());
    let good = packed_layers(&gpu, 64, p.view_count, p.view_count as usize);
    impostor
        .initialize(&gpu, Some(&good), &volume_bytes(16))
        .expect("packed layers initialize");
}
