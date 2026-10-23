//! Group-1 lighting resources (S10). CPU part: `EnvGpu` bytes, offsets and defaults. GPU
//! part (SKIP without a software adapter): the dummies bind in a render pipeline whose WGSL
//! declares group 1 exactly as the shader prelude, render with no validation error and read
//! back the dummy values. The dirty-flag logic behind the setters is device-free and tested
//! next to it (src/lighting.rs, `State` tests), plus a GPU `upload` test there.

#![cfg(not(target_arch = "wasm32"))]

use raa_core::testing::{block_on, skip, software_gpu};
use raa_core::{CLEAR_COLOR, GpuContext, wgpu};
use raa_pipeline::{EnvGpu, LightingResources};

fn f32_at(bytes: &[u8], offset: usize) -> f32 {
    let mut word = [0u8; 4];
    word.copy_from_slice(&bytes[offset..offset + 4]);
    f32::from_le_bytes(word)
}

fn vec4_at(bytes: &[u8], offset: usize) -> [f32; 4] {
    [0, 4, 8, 12].map(|o| f32_at(bytes, offset + o))
}

#[test]
fn env_gpu_defaults_are_the_clear_colour_and_constant_background() {
    let env = EnvGpu::default();
    let clear = [CLEAR_COLOR.r, CLEAR_COLOR.g, CLEAR_COLOR.b].map(|c| c as f32);
    assert_eq!(clear, [0.05, 0.06, 0.08]);
    assert_eq!(env.background, [0.05, 0.06, 0.08, 0.0]);
    assert_eq!(env.sky_params[2], 0.0, "background mode 0 (constant)");

    let bytes = env.to_bytes();
    assert_eq!(bytes.len(), 256);
    assert_eq!(EnvGpu::SIZE, 256);
    assert_eq!(vec4_at(&bytes, 80), [0.05, 0.06, 0.08, 0.0]);
    for offset in (0..256).step_by(16).filter(|o| *o != 80) {
        assert_eq!(vec4_at(&bytes, offset), [0.0; 4], "offset {offset}");
    }
}

#[test]
fn env_gpu_fields_are_written_at_their_s10_offsets() {
    let mut env = EnvGpu::default();
    let fields: [(&mut [f32; 4], usize); 12] = [
        (&mut env.sun_disc, 0),
        (&mut env.zenith, 16),
        (&mut env.horizon, 32),
        (&mut env.ground, 48),
        (&mut env.sky_params, 64),
        (&mut env.background, 80),
        (&mut env.clouds, 96),
        (&mut env.clouds2, 112),
        (&mut env.cloud_color, 128),
        (&mut env.fog, 144),
        (&mut env.fog_color, 160),
        (&mut env.aerial, 176),
    ];
    let mut expected = Vec::new();
    for (field, offset) in fields {
        let base = offset as f32;
        *field = [base + 1.0, base + 2.0, base + 3.0, base + 4.0];
        expected.push((offset, *field));
    }
    let bytes = env.to_bytes();
    for (offset, value) in expected {
        assert_eq!(vec4_at(&bytes, offset), value, "offset {offset}");
        assert_eq!(
            &bytes[offset..offset + 4],
            &(offset as f32 + 1.0).to_le_bytes()
        );
    }
    assert!(bytes[192..].iter().all(|b| *b == 0), "reserved @192..256");
}

/// Group 1 as the shader prelude declares it (S10), and a fragment shader that reads every
/// binding: target 0 = (shadow compare, brdf scale, brdf bias, ao), target 1 = (contact,
/// opaque depth, env cube red, background red + every zero-default uniform and black).
const SHADER: &str = r#"
struct ShadowGpu {
    cascade_view_proj: array<mat4x4<f32>, 4>,
    splits: vec4<f32>,
    params: vec4<f32>,
    params2: vec4<f32>,
    reserved: vec4<f32>,
}
struct EnvSh {
    sh: array<vec4<f32>, 9>,
    info: vec4<f32>,
}
struct EnvGpu {
    sun_disc: vec4<f32>,
    zenith: vec4<f32>,
    horizon: vec4<f32>,
    ground: vec4<f32>,
    sky_params: vec4<f32>,
    background: vec4<f32>,
    clouds: vec4<f32>,
    clouds2: vec4<f32>,
    cloud_color: vec4<f32>,
    fog: vec4<f32>,
    fog_color: vec4<f32>,
    aerial: vec4<f32>,
    reserved: array<vec4<f32>, 4>,
}

@group(1) @binding(0) var shadow_map: texture_depth_2d_array;
@group(1) @binding(1) var shadow_cmp: sampler_comparison;
@group(1) @binding(2) var<uniform> shadow: ShadowGpu;
@group(1) @binding(3) var env_specular: texture_cube<f32>;
@group(1) @binding(4) var<uniform> env_sh: EnvSh;
@group(1) @binding(5) var brdf_lut: texture_2d<f32>;
@group(1) @binding(6) var linear_clamp: sampler;
@group(1) @binding(7) var ao: texture_2d<f32>;
@group(1) @binding(8) var contact: texture_2d<f32>;
@group(1) @binding(9) var opaque_color: texture_2d<f32>;
@group(1) @binding(10) var opaque_depth: texture_2d<f32>;
@group(1) @binding(11) var<uniform> env: EnvGpu;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let xy = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
    return vec4<f32>(xy, 0.0, 1.0);
}

struct Out {
    @location(0) a: vec4<f32>,
    @location(1) b: vec4<f32>,
}

@fragment
fn fs() -> Out {
    let uv = vec2<f32>(0.5, 0.5);
    let cmp = textureSampleCompareLevel(shadow_map, shadow_cmp, uv, 0, 0.5);
    let lut = textureSampleLevel(brdf_lut, linear_clamp, uv, 0.0).xy;
    let occlusion = textureSampleLevel(ao, linear_clamp, uv, 0.0).r;
    let contact_shadow = textureSampleLevel(contact, linear_clamp, uv, 0.0).r;
    let depth = textureLoad(opaque_depth, vec2<i32>(0, 0), 0).r;
    let cube = textureSampleLevel(env_specular, linear_clamp, vec3<f32>(0.3, 1.0, -0.2), 0.0);
    let zeros = shadow.params.w + shadow.cascade_view_proj[0][0][0] + env_sh.info.w
        + env_sh.sh[0].r + env.sky_params.z
        + textureSampleLevel(opaque_color, linear_clamp, uv, 0.0).r;
    var out: Out;
    out.a = vec4<f32>(cmp, lut.x, lut.y, occlusion);
    out.b = vec4<f32>(contact_shadow, depth, cube.r, env.background.r + zeros);
    return out;
}
"#;

/// Rgba16Float: renderable on every adapter (the GL software one has no Rgba32Float target).
const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// f32 value of f16 bits (normal numbers and zero only, enough for the test values).
fn f16_value(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    if exponent == 0 {
        return 0.0;
    }
    sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15)
}

fn target(gpu: &GpuContext) -> wgpu::Texture {
    gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("resources test target"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TARGET_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// Copies the 1x1 Rgba16Float `texture` out and returns its texel.
fn read_texel(gpu: &GpuContext, texture: &wgpu::Texture) -> [f32; 4] {
    let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("resources readback"),
        size: 256,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device().create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(1),
            },
        },
        texture.size(),
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
    let texel = [0, 2, 4, 6].map(|o| f16_value(u16::from_le_bytes([mapped[o], mapped[o + 1]])));
    drop(mapped);
    buffer.unmap();
    texel
}

fn assert_near(got: [f32; 4], want: [f32; 4], what: &str) {
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() <= 1e-3, "{what}: got {got:?}, want {want:?}");
    }
}

/// Renders one pixel with `lighting.bind_group()` at group 1 and returns both targets.
fn render_with_group_1(gpu: &GpuContext, lighting: &LightingResources) -> [[f32; 4]; 2] {
    let device = gpu.device();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("resources test shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let empty = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("resources test group 0"),
        entries: &[],
    });
    let empty_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("resources test group 0"),
        layout: &empty,
        entries: &[],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("resources test layout"),
        bind_group_layouts: &[Some(&empty), Some(lighting.layout())],
        immediate_size: 0,
    });
    let color = Some(wgpu::ColorTargetState {
        format: TARGET_FORMAT,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("resources test pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[color.clone(), color],
        }),
        multiview_mask: None,
        cache: None,
    });

    let targets = [target(gpu), target(gpu)];
    let views = targets
        .each_ref()
        .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()));
    let attachments = views.each_ref().map(|view| {
        Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("resources test pass"),
            color_attachments: &attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &empty_group, &[]);
        pass.set_bind_group(1, lighting.bind_group(), &[]);
        pass.draw(0..3, 0..1);
    }
    gpu.queue().submit([encoder.finish()]);
    targets.each_ref().map(|t| read_texel(gpu, t))
}

fn dummies_bind_and_read_their_values(gpu: &GpuContext) {
    let scope = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let lighting = LightingResources::new(gpu);
    let [a, b] = render_with_group_1(gpu, &lighting);
    let error = block_on(scope.pop());
    assert!(error.is_none(), "validation error: {error:?}");

    // compare 0.5 <= depth 1 passes; LUT (1, 0); ao 1
    assert_near(a, [1.0, 1.0, 0.0, 1.0], "shadow, brdf lut, ao");
    // contact 1; opaque depth 1; cube = ambient red 0.6 x 0.3; background red 0.05
    assert_near(
        b,
        [1.0, 1.0, 0.18, 0.05],
        "contact, depth, env cube, background",
    );
}

fn cleared_features_still_bind(gpu: &GpuContext) {
    let scope = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
    let mut lighting = LightingResources::new(gpu);
    lighting.clear_shadow();
    lighting.clear_env();
    lighting.clear_ao();
    lighting.clear_contact();
    lighting.clear_opaque_copy();
    lighting.env_mut().background = [0.25, 0.5, 1.0, 0.0];
    assert_eq!(lighting.env().background, [0.25, 0.5, 1.0, 0.0]);
    render_with_group_1(gpu, &lighting);
    let error = block_on(scope.pop());
    assert!(error.is_none(), "validation error: {error:?}");
}

/// One GPU context for every check, leaked like in tests/targets.rs (the GL backend of the
/// WSL driver crashed at exit after dropping a device that created Rgba16Float textures).
#[test]
fn lighting_resources_on_the_software_gpu() {
    let Some(gpu) = software_gpu() else {
        skip(
            "lighting_resources_on_the_software_gpu",
            "software_gpu returned None; see the software_gpu line above",
        );
        return;
    };
    dummies_bind_and_read_their_values(&gpu);
    cleared_features_still_bind(&gpu);
    std::mem::forget(gpu);
}
