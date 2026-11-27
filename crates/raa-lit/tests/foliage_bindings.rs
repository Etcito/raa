//! Foliage bindings (M7, contract `m7-lit-binding`): the mode-and-layout keyed pipeline
//! cache, the complete group 0+1+2 fragment accounting of the additive API, and the
//! `LitHandle` association and binding entry points.
//!
//! Upstream anchors: `foliage_layout_entries()`
//! (`crates/raa-lit/src/foliage_resources.rs:196`) declares the exact fragment-only group 2,
//! `FoliageResources::bind_group` (`crates/raa-lit/src/foliage_resources.rs:392`) is the
//! immutable generation group bound per draw, and `Composer::compose_foliage`
//! (`crates/raa-shader/src/composer.rs:179`) composes the prepass and forward modules.

#![cfg(not(target_arch = "wasm32"))]

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

use raa_core::testing::{skip, software_gpu};
use raa_core::wgpu;
use raa_core::{MaterialId, MeshId, RaaError};
use raa_lit::foliage_bindings::{
    FoliageLayoutCounts, FoliagePipelineCache, FoliagePipelineKey,
    MAX_FOLIAGE_FRAGMENT_STORAGE_BUFFERS, MAX_FOLIAGE_FRAGMENT_TEXTURES,
    OPAQUE_GROUP1_FRAGMENT_TEXTURES, layout_identity, validate_foliage_layouts,
};
use raa_lit::foliage_resources::{
    FOLIAGE_MODE_POM, FOLIAGE_MODE_SPOM, FoliageParams, FoliageResources, foliage_layout_entries,
};
use raa_lit::{LitOptions, LitRenderer, PipelineSpec};
use raa_pipeline::{HDR_FORMAT, Pipeline};
use raa_shader::{MaterialFnId, PassKind, VariantKey};

fn invalid_code() -> &'static str {
    RaaError::FrameGraph {
        code: "foliage.invalid",
        message: String::new(),
    }
    .code()
}

fn buffer(
    binding: u32,
    visibility: wgpu::ShaderStages,
    uniform: bool,
) -> wgpu::BindGroupLayoutEntry {
    let ty = if uniform {
        wgpu::BufferBindingType::Uniform
    } else {
        wgpu::BufferBindingType::Storage { read_only: true }
    };
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn variant(pass: PassKind) -> VariantKey {
    VariantKey {
        material_fn: MaterialFnId::STANDARD,
        pass,
        msaa: 1,
        flags: 0,
    }
}

fn params(mode: u32) -> FoliageParams {
    FoliageParams {
        min: [-2.0, 0.0, -2.0],
        max: [2.0, 9.0, 2.0],
        resolution: 16,
        mode,
        translucency: 0.35,
        generation: 3,
        cutoff: 0.25,
        view_count: 2,
        texture_resolution: 64,
    }
}

fn forward_spec() -> PipelineSpec {
    PipelineSpec {
        color_targets: vec![Some(wgpu::ColorTargetState {
            format: HDR_FORMAT,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })],
        depth: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        cull: Some(wgpu::Face::Back),
        alpha_to_coverage: false,
        extra_layout: None,
    }
}

#[test]
fn layout_counts_ignore_entries_inactive_in_the_fragment_stage() {
    let vertex_only_storage = buffer(0, wgpu::ShaderStages::VERTEX, false);
    let compute_storage = buffer(1, wgpu::ShaderStages::COMPUTE, false);
    let entries = [
        vertex_only_storage,
        compute_storage,
        texture(2, wgpu::ShaderStages::FRAGMENT),
        buffer(3, wgpu::ShaderStages::FRAGMENT, false),
        buffer(4, wgpu::ShaderStages::FRAGMENT, true),
        sampler(5, wgpu::ShaderStages::FRAGMENT),
    ];
    let counts = FoliageLayoutCounts::of(&entries);
    assert_eq!(counts.textures, 1, "only the fragment texture counts");
    assert_eq!(
        counts.storage_buffers, 1,
        "vertex and compute storage and fragment uniforms are not fragment storage"
    );
}

#[test]
fn complete_layout_uses_fourteen_textures_and_five_storage_buffers() {
    let foliage = FoliageLayoutCounts::of(&foliage_layout_entries());
    assert_eq!(foliage.textures, 3, "the D3 volume and the two D2 arrays");
    assert_eq!(foliage.storage_buffers, 0);

    let complete = FoliageLayoutCounts::complete();
    assert_eq!(
        complete.textures,
        1 + OPAQUE_GROUP1_FRAGMENT_TEXTURES + 3,
        "group 0 atlas + opaque group 1 + foliage group 2"
    );
    assert_eq!(
        complete.storage_buffers, 5,
        "the frame buffers 1, 2, 5, 6 and 7"
    );
    assert!(complete.textures <= MAX_FOLIAGE_FRAGMENT_TEXTURES);
    assert_eq!(validate_foliage_layouts(), Ok(complete));
}

#[test]
fn layout_limits_refuse_one_entry_over_budget() {
    let ok = FoliageLayoutCounts {
        textures: MAX_FOLIAGE_FRAGMENT_TEXTURES,
        storage_buffers: MAX_FOLIAGE_FRAGMENT_STORAGE_BUFFERS,
    };
    assert_eq!(ok.validate(), Ok(()));

    let too_many_textures = FoliageLayoutCounts {
        textures: MAX_FOLIAGE_FRAGMENT_TEXTURES + 1,
        storage_buffers: 0,
    };
    assert_eq!(
        too_many_textures
            .validate()
            .expect_err("over the texture limit")
            .code(),
        invalid_code()
    );

    let too_many_buffers = FoliageLayoutCounts {
        textures: 0,
        storage_buffers: MAX_FOLIAGE_FRAGMENT_STORAGE_BUFFERS + 1,
    };
    assert_eq!(
        too_many_buffers
            .validate()
            .expect_err("over the storage limit")
            .code(),
        invalid_code()
    );
}

#[test]
fn pipeline_keys_separate_modes_and_layout_identities() {
    let entries = foliage_layout_entries();
    let identity = layout_identity(&entries);
    assert_eq!(identity, layout_identity(&foliage_layout_entries()));
    let mut other_entries = entries.clone();
    other_entries[0].binding = 9;
    assert_ne!(
        identity,
        layout_identity(&other_entries),
        "a different actual layout has a different identity"
    );

    let base = FoliagePipelineKey::new(FOLIAGE_MODE_SPOM, &entries, variant(PassKind::Prepass), 11);
    assert_eq!(base.layout, identity);
    let same = FoliagePipelineKey::new(FOLIAGE_MODE_SPOM, &entries, variant(PassKind::Prepass), 11);
    let other_mode =
        FoliagePipelineKey::new(FOLIAGE_MODE_POM, &entries, variant(PassKind::Prepass), 11);
    let other_layout = FoliagePipelineKey::new(
        FOLIAGE_MODE_SPOM,
        &other_entries,
        variant(PassKind::Prepass),
        11,
    );
    let other_pass =
        FoliagePipelineKey::new(FOLIAGE_MODE_SPOM, &entries, variant(PassKind::Forward), 11);
    assert_eq!(base, same);
    assert_ne!(base, other_mode);
    assert_ne!(base, other_layout);
    assert_ne!(base, other_pass);
}

#[test]
fn foliage_cache_reuses_a_key_and_separates_modes_layouts_and_slots() {
    let spec = forward_spec();
    let mut cache: FoliagePipelineCache<u32> = FoliagePipelineCache::new();
    let builds = Cell::new(0u32);
    let build = |value: u32| {
        builds.set(builds.get() + 1);
        Ok::<u32, ()>(value)
    };
    let base = FoliagePipelineKey::from_parts(FOLIAGE_MODE_SPOM, 7, variant(PassKind::Forward), 11);
    assert_eq!(cache.get_or_build(base, &spec, 3, || build(1)), Ok(1));
    assert_eq!(cache.get_or_build(base, &spec, 3, || build(2)), Ok(1));
    assert_eq!(builds.get(), 1, "an equal key reuses the cached value");

    let other_mode =
        FoliagePipelineKey::from_parts(FOLIAGE_MODE_POM, 7, variant(PassKind::Forward), 11);
    let other_layout =
        FoliagePipelineKey::from_parts(FOLIAGE_MODE_SPOM, 8, variant(PassKind::Forward), 11);
    assert_eq!(cache.get_or_build(other_mode, &spec, 3, || build(3)), Ok(3));
    assert_eq!(
        cache.get_or_build(other_layout, &spec, 3, || build(4)),
        Ok(4)
    );
    assert_eq!(
        builds.get(),
        3,
        "mode and layout identity separate the cache"
    );

    assert_eq!(cache.get_or_build(base, &spec, 4, || build(5)), Ok(5));
    assert_eq!(
        builds.get(),
        4,
        "new slot winners drop the cached pipelines"
    );
}

#[test]
fn associations_drive_the_pipeline_lookup_and_clear_before_release() {
    let test = "associations_drive_the_pipeline_lookup_and_clear_before_release";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, HDR_FORMAT, false)));
    let lit = LitRenderer::new(
        &gpu,
        &pipeline,
        LitOptions {
            msaa: 1,
            ..LitOptions::default()
        },
    )
    .expect("lit renderer");
    let handle = lit.handle();
    let resources =
        Rc::new(FoliageResources::new(&gpu, params(FOLIAGE_MODE_POM)).expect("resources"));
    let material = MaterialId(0);

    handle
        .set_foliage(material, Some(resources))
        .expect("associate");
    assert_eq!(handle.foliage_mode(material), Some(FOLIAGE_MODE_POM));

    let key = variant(PassKind::Forward);
    let spec = forward_spec();
    let first = handle
        .pipeline_for_foliage(&gpu, MeshId(0), key, &spec, material)
        .expect("foliage pipeline");
    let second = handle
        .pipeline_for_foliage(&gpu, MeshId(0), key, &spec, material)
        .expect("cached foliage pipeline");
    assert!(
        Arc::ptr_eq(&first, &second),
        "one pipeline per mode and layout"
    );

    let other_mode = handle
        .pipeline_for_foliage(&gpu, MeshId(0), key, &spec, MaterialId(1))
        .expect_err("no association");
    assert_eq!(other_mode.code(), invalid_code());

    handle.set_foliage(material, None).expect("detach");
    assert_eq!(handle.foliage_mode(material), None);
    let detached = handle
        .pipeline_for_foliage(&gpu, MeshId(0), key, &spec, material)
        .expect_err("detached association");
    assert_eq!(detached.code(), invalid_code());
}
