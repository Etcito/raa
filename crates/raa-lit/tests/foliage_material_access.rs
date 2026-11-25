//! LitHandle material access for foliage compute (M7, contract `m7-work`): the two public
//! methods added to `crates/raa-lit/src/handle.rs` by 'Evaluate unlit source appearance for
//! foliage baking', and the existing fragment frame layout and `pipeline_for` they must not
//! disturb.
//!
//! Cited handoff anchors:
//! - `LitHandle::material_fn_source` (`crates/raa-lit/src/handle.rs:1938`): the source WGSL
//!   of a material function after the pipeline cache's canonical-id mapping
//!   (`LitShared::canonical_fn`, `crates/raa-lit/src/handle.rs:1033`), or `None` for an
//!   unknown id.
//! - `LitHandle::material_compute_group` (`crates/raa-lit/src/handle.rs:1955`): a
//!   `COMPUTE`-visible layout of bindings 0-7 of `frame_layout_entries`
//!   (`crates/raa-lit/src/handle.rs:215`) and a bind group over the renderer's live frame,
//!   material, extras and instance buffers, its texture atlas and its material data words.
//! - `LitHandle::material_fn_of` (`crates/raa-lit/src/handle.rs:1812`): the function id of
//!   a `SurfaceModel::Function` material, as the foliage appearance pass reads it.

#![cfg(not(target_arch = "wasm32"))]

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::GpuContext;
use raa_core::testing::{block_on, skip, software_gpu};
use raa_lit::{
    FLAG_UNLIT, LitHandle, LitOptions, LitRenderer, MaterialFnId, PipelineSpec, SurfaceMaterial,
    SurfaceModel,
};
use raa_pipeline::Pipeline;
use raa_shader::{PassKind, VariantKey};

const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// The M6 leaf-style material function: an unlit green whose main vein runs along the
/// interpolated source UV, so the foliage pass evaluates a real `material_main`.
const LEAF_WGSL: &str = "\
fn material_main(in: MaterialInput) -> MaterialOutput {
    var o = raa_default_output(in);
    let vein = 1.0 - abs(in.uv.x * 2.0 - 1.0);
    let leaf = mix(vec3<f32>(0.11, 0.32, 0.08), vec3<f32>(0.24, 0.58, 0.14), vein);
    o.base_color = leaf;
    o.emissive = vec3<f32>(0.0);
    return o;
}
";

/// A minimal compute shader over the returned group 0: every binding 0-7 is declared with
/// the type of its `frame_layout_entries` slot (0 `FrameGpu` uniform, 1 materials, 2 extras,
/// 3 `texture_2d<f32>` filterable atlas, 4 filtering sampler, 5 rects, 6 material data
/// words, 7 instance records) and binding 8 is not declared.
const COMPUTE_WGSL: &str = "\
struct FrameProbe {
    lanes: array<vec4<f32>, 48>,
};

@group(0) @binding(0) var<uniform> frame: FrameProbe;
@group(0) @binding(1) var<storage, read> materials: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> extras: array<vec4<f32>>;
@group(0) @binding(3) var atlas: texture_2d<f32>;
@group(0) @binding(4) var atlas_sampler: sampler;
@group(0) @binding(5) var<storage, read> rects: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> material_words: array<u32>;
@group(0) @binding(7) var<storage, read> instances: array<vec4<f32>>;

@compute @workgroup_size(1)
fn cs_main() {
    let texel = textureSampleLevel(atlas, atlas_sampler, vec2<f32>(0.0), 0.0);
    let probe = frame.lanes[0] + materials[0] + extras[0] + rects[0]
        + vec4<f32>(f32(material_words[0])) + instances[0] + texel;
    _ = probe;
}
";

/// The same shape but reading binding 8, which the compute layout leaves out: creating this
/// pipeline against the returned layout must be refused.
const BINDING_EIGHT_WGSL: &str = "\
@group(0) @binding(8) var<storage, read> previous: array<vec4<f32>>;

@compute @workgroup_size(1)
fn cs_main() {
    _ = previous[0];
}
";

/// A fragment shader over the existing group-0 frame layout: only binding 0 (the `FrameGpu`
/// uniform) is read, so the layout's `VERTEX_FRAGMENT` visibility is what the pipeline uses.
const FRAME_WGSL: &str = "\
struct FrameProbe {
    lanes: array<vec4<f32>, 48>,
};

@group(0) @binding(0) var<uniform> frame: FrameProbe;

@vertex
fn vs_main() -> @builtin(position) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return frame.lanes[0];
}
";

/// A software-GPU renderer with the leaf function registered twice, as the pipeline cache
/// sees a canonical id and an alias of equal source.
fn leaf_renderer(gpu: &GpuContext) -> (LitRenderer, MaterialFnId, MaterialFnId) {
    let pipeline = Rc::new(RefCell::new(Pipeline::new(gpu, SURFACE, false)));
    let mut lit = LitRenderer::new(
        gpu,
        &pipeline,
        LitOptions {
            msaa: 1,
            ..LitOptions::default()
        },
    )
    .expect("lit renderer");
    let primary = lit
        .register_user_fn("leaf", LEAF_WGSL)
        .expect("register the leaf function");
    let alias = lit
        .register_user_fn("leaf_alias", LEAF_WGSL)
        .expect("register the equal-source alias");
    (lit, primary, alias)
}

/// The fixed-function state of the ordinary forward material pipeline.
fn forward_spec() -> PipelineSpec {
    PipelineSpec {
        color_targets: vec![Some(wgpu::ColorTargetState {
            format: raa_pipeline::HDR_FORMAT,
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

/// The leaf material record: an unlit `SurfaceModel::Function` material of `fid`.
fn leaf_material(fid: MaterialFnId) -> SurfaceMaterial {
    SurfaceMaterial {
        model: SurfaceModel::Function(fid),
        flags: FLAG_UNLIT,
        ..SurfaceMaterial::default()
    }
}

fn compute_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    wgsl: &str,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("foliage_material_access.compute"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("foliage_material_access.compute"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("foliage_material_access.compute"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("cs_main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

#[test]
fn material_fn_source_answers_the_material_and_its_canonical_alias() {
    let test = "material_fn_source_answers_the_material_and_its_canonical_alias";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let (mut lit, primary, alias) = leaf_renderer(&gpu);
    assert_ne!(primary, alias, "each registration keeps its own id");
    let handle: LitHandle = lit.handle();
    assert_eq!(
        handle.material_fn_source(primary).as_deref(),
        Some(LEAF_WGSL),
        "the registered function is returned verbatim"
    );
    assert_eq!(
        handle.material_fn_source(alias).as_deref(),
        Some(LEAF_WGSL),
        "the canonical alias maps to the first registration's source"
    );

    let material = lit
        .materials_mut()
        .add(leaf_material(alias))
        .expect("leaf material");
    assert_eq!(handle.material_fn_of(material), alias);
    assert_eq!(
        handle
            .material_fn_source(handle.material_fn_of(material))
            .as_deref(),
        Some(LEAF_WGSL),
        "the foliage path reaches the source through the material's function id"
    );

    let unknown = MaterialFnId(9_999);
    assert_eq!(handle.material_fn_source(unknown), None);
    assert_eq!(
        handle.material_fn_source(MaterialFnId::STANDARD).as_deref(),
        Some(lit.fns().get(MaterialFnId::STANDARD).unwrap().wgsl.as_str())
    );
}

#[test]
fn material_compute_group_runs_a_compute_dispatch_without_error() {
    let test = "material_compute_group_runs_a_compute_dispatch_without_error";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let (lit, _, _) = leaf_renderer(&gpu);
    let handle = lit.handle();
    let (layout, group) = handle.material_compute_group(&gpu);
    let device = gpu.device();
    let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let pipeline = compute_pipeline(device, &layout, COMPUTE_WGSL);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("foliage_material_access.compute"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("foliage_material_access.compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    gpu.queue().submit([encoder.finish()]);
    let error = block_on(errors.pop());
    assert!(error.is_none(), "GPU validation error: {error:?}");

    // Binding 8 (the vertex-only previous records) is absent from the compute layout: a
    // pipeline reading it against that layout is refused, while the dispatch above passed.
    let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let _binding_eight = compute_pipeline(device, &layout, BINDING_EIGHT_WGSL);
    let error = block_on(errors.pop());
    assert!(
        error.is_some(),
        "the compute group must leave binding 8 out"
    );
}

#[test]
fn fragment_frame_layout_and_pipeline_for_still_create() {
    let test = "fragment_frame_layout_and_pipeline_for_still_create";
    let Some(gpu) = software_gpu() else {
        skip(test, "software_gpu returned None");
        return;
    };
    let (lit, _, _) = leaf_renderer(&gpu);
    let handle = lit.handle();
    let frame_layout = handle.frame_layout();
    let device = gpu.device();
    let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("foliage_material_access.frame"),
        source: wgpu::ShaderSource::Wgsl(FRAME_WGSL.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("foliage_material_access.frame"),
        bind_group_layouts: &[Some(&frame_layout)],
        immediate_size: 0,
    });
    let _frame_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("foliage_material_access.frame"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: SURFACE,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    let key = VariantKey {
        material_fn: MaterialFnId::STANDARD,
        pass: PassKind::Forward,
        msaa: 1,
        flags: 0,
    };
    let _standard = handle
        .pipeline_for(&gpu, key, &forward_spec())
        .expect("an ordinary standard forward material pipeline still creates");
    let error = block_on(errors.pop());
    assert!(error.is_none(), "GPU validation error: {error:?}");
}
