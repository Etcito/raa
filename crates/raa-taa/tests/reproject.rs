//! Contract `raa-temporal-wgsl`: the shared reprojection WGSL validates in a compute module,
//! its `RaaTemporalFrame` matches the raa-shader prelude `FrameGpu` layout, and each of its
//! four functions is declared exactly once. The resolve `cs_taa` keeps the current colour of
//! a pixel whose motion carries the S43 history reject flag.

use raa_shader::Composer;
use raa_taa::{REPROJECT_WGSL, TEMPORAL_FRAME_SIZE};

/// A compute module that binds the twin as a uniform and a storage buffer and calls every
/// helper: the shape the downstream temporal passes compose.
const PROBE_WGSL: &str = "
@group(0) @binding(0) var<uniform> f: RaaTemporalFrame;
@group(0) @binding(1) var<storage, read_write> probe_out: array<vec4<f32>>;

@compute @workgroup_size(1)
fn cs_reproject_probe(@builtin(global_invocation_id) id: vec3<u32>) {
    let uv = vec2<f32>(0.5, 0.5);
    let surface = raa_temporal_surface_uv(uv, vec4<f32>(0.0));
    let sky = raa_temporal_sky_uv(
        uv,
        f.inv_view,
        f.inv_proj,
        f.previous_view_proj,
        f.camera_jitter,
    );
    let on_screen = raa_temporal_on_screen(surface.xy);
    let usable = raa_temporal_history_usable(f.temporal);
    probe_out[id.x] = vec4<f32>(
        surface.z,
        sky.z,
        select(0.0, 1.0, on_screen),
        select(0.0, 1.0, usable),
    );
}
";

/// The `(name, byte offset)` members and span of struct `name` in `module`.
fn struct_layout(module: &naga::Module, name: &str) -> (u32, Vec<(String, u32)>) {
    let (_, ty) = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("{name} is declared"));
    let naga::TypeInner::Struct { members, span } = &ty.inner else {
        panic!("{name} is a struct");
    };
    let fields = members
        .iter()
        .map(|m| (m.name.clone().unwrap_or_default(), m.offset))
        .collect();
    (*span, fields)
}

#[test]
fn reproject_wgsl_validates_in_a_compute_module() {
    let source = format!("{REPROJECT_WGSL}\n{PROBE_WGSL}");
    assert_eq!(Composer::validate(&source), Ok(()));
}

#[test]
fn raa_temporal_frame_matches_the_prelude_frame_gpu() {
    let prelude = naga::front::wgsl::parse_str(raa_shader::PRELUDE_WGSL)
        .or_else(|_| {
            // The composer emits PRELUDE, LIB and CLOUD_SHADOW first; parse that prefix when
            // the prelude needs the later declarations.
            naga::front::wgsl::parse_str(&format!(
                "{}\n{}\n{}",
                raa_shader::PRELUDE_WGSL,
                raa_shader::LIB_WGSL,
                raa_shader::CLOUD_SHADOW_WGSL
            ))
        })
        .expect("the raa-shader prelude parses");
    let reproject = naga::front::wgsl::parse_str(REPROJECT_WGSL).expect("REPROJECT_WGSL parses");

    let (prelude_span, prelude_members) = struct_layout(&prelude, "FrameGpu");
    let (span, members) = struct_layout(&reproject, "RaaTemporalFrame");
    assert_eq!(members, prelude_members, "RaaTemporalFrame members");
    assert_eq!(span, prelude_span, "RaaTemporalFrame span");
    assert_eq!(span, 768);
    assert_eq!(u64::from(span), TEMPORAL_FRAME_SIZE);
}

#[test]
fn each_reprojection_function_is_declared_once() {
    for name in [
        "raa_temporal_on_screen",
        "raa_temporal_surface_uv",
        "raa_temporal_sky_uv",
        "raa_temporal_history_usable",
    ] {
        let declaration = format!("fn {name}(");
        assert_eq!(
            REPROJECT_WGSL.matches(&declaration).count(),
            1,
            "{declaration}"
        );
    }
}

/// The resolve `cs_taa` on the software GPU over an 8 x 8 frame whose motion target carries the
/// S43 reject flag (z = 1): a pixel with that flag keeps its current colour, exactly as with no
/// usable history, while the same frame without the flag blends its history. This is the rule
/// that keeps a moving object from leaving a trail, a doubled copy or a dark outline.
#[cfg(not(target_arch = "wasm32"))]
mod reject_flag {
    use raa_core::testing::{block_on, skip, software_gpu};
    use raa_core::{GpuContext, wgpu};
    use raa_taa::{TAA_ENTRY, TEMPORAL_FRAME_SIZE, taa_module_source};

    const SIZE: u32 = 8;
    /// The probed output pixel: its 3 x 3 neighbourhood holds both scene colours.
    const PROBE: usize = (4 * SIZE + 4) as usize;
    const HDR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

    /// IEEE half bits of `v` for the exact values these fixtures use (0, 0.5, 1).
    fn f16_bits(v: f32) -> u16 {
        match v {
            1.0 => 0x3C00,
            0.5 => 0x3800,
            _ => 0,
        }
    }

    fn f16_to_f32(h: u16) -> f32 {
        let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exponent = i32::from((h >> 10) & 0x1F);
        let mantissa = f32::from(h & 0x3FF);
        match exponent {
            0 => sign * mantissa * 2f32.powi(-24),
            31 => f32::NAN,
            e => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(e - 15),
        }
    }

    fn texture(
        gpu: &GpuContext,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> wgpu::Texture {
        gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("reject_flag"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    }

    /// An Rgba16Float texture filled by `texel(x, y)`.
    fn hdr_input(gpu: &GpuContext, texel: impl Fn(u32, u32) -> [f32; 4]) -> wgpu::Texture {
        let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        let tex = texture(gpu, HDR, usage);
        let mut bytes = Vec::with_capacity((SIZE * SIZE * 8) as usize);
        for y in 0..SIZE {
            for x in 0..SIZE {
                for c in texel(x, y) {
                    bytes.extend_from_slice(&f16_bits(c).to_le_bytes());
                }
            }
        }
        gpu.queue().write_texture(
            tex.as_image_copy(),
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SIZE * 8),
                rows_per_image: Some(SIZE),
            },
            wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
        );
        tex
    }

    /// A Depth32Float texture cleared to 0.5: every pixel is geometry, none is sky.
    fn depth_input(gpu: &GpuContext) -> wgpu::Texture {
        let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT;
        let tex = texture(gpu, wgpu::TextureFormat::Depth32Float, usage);
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = gpu.device().create_command_encoder(&Default::default());
        drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("reject_flag.depth"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.5),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        }));
        gpu.queue().submit([encoder.finish()]);
        tex
    }

    /// The 768-byte frame: viewport 8 x 8, no jitter, temporal = (0, accumulation 1, camera
    /// history 1, 0). Matrices stay zero: no pixel takes the sky path.
    fn frame_bytes() -> Vec<u8> {
        let mut bytes = vec![0u8; TEMPORAL_FRAME_SIZE as usize];
        let viewport = [
            SIZE as f32,
            SIZE as f32,
            1.0 / SIZE as f32,
            1.0 / SIZE as f32,
        ];
        for (i, v) in viewport.iter().enumerate() {
            bytes[384 + 4 * i..388 + 4 * i].copy_from_slice(&v.to_le_bytes());
        }
        for (i, v) in [0u32, 1, 1, 0].iter().enumerate() {
            bytes[672 + 4 * i..676 + 4 * i].copy_from_slice(&v.to_le_bytes());
        }
        bytes
    }

    fn uniform(gpu: &GpuContext, bytes: &[u8]) -> wgpu::Buffer {
        let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("reject_flag.uniform"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue().write_buffer(&buffer, 0, bytes);
        buffer
    }

    /// Reads an 8 x 8 Rgba16Float texture back as f32 texels.
    fn read(gpu: &GpuContext, tex: &wgpu::Texture) -> Vec<[f32; 4]> {
        let row = (SIZE * 8) as usize;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
        let padded = row.div_ceil(align) * align;
        let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("reject_flag.readback"),
            size: (padded * SIZE as usize) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = gpu.device().create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            tex.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded as u32),
                    rows_per_image: Some(SIZE),
                },
            },
            wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue().submit([encoder.finish()]);
        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
        rx.recv().expect("map callback").expect("map");
        let mapped = slice.get_mapped_range().expect("mapped range");
        mapped
            .chunks_exact(padded)
            .flat_map(|r| r[..row].as_chunks::<8>().0.iter().map(|c| c.as_slice()))
            .map(|px| {
                std::array::from_fn(|c| f16_to_f32(u16::from_le_bytes([px[2 * c], px[2 * c + 1]])))
            })
            .collect()
    }

    /// One `cs_taa` dispatch: a scene of red and blue columns, a uniform green history (far
    /// outside the neighbourhood, so a blend visibly pulls the result), motion xy 0 with
    /// reject flag `reject`, and `RaaTaaParams.history_usable` = `history_usable`. Returns the
    /// output texels.
    fn resolve(gpu: &GpuContext, reject: f32, history_usable: u32) -> Vec<[f32; 4]> {
        let device = gpu.device();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reject_flag"),
            source: wgpu::ShaderSource::Wgsl(taa_module_source().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("reject_flag"),
            layout: None,
            module: &module,
            entry_point: Some(TAA_ENTRY),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let scene = hdr_input(gpu, |x, _| {
            if x % 2 == 0 {
                [1.0, 0.0, 0.0, 1.0]
            } else {
                [0.0, 0.0, 1.0, 1.0]
            }
        });
        let motion = hdr_input(gpu, |_, _| [0.0, 0.0, reject, 0.0]);
        let history = hdr_input(gpu, |_, _| [0.0, 1.0, 0.0, 1.0]);
        let depth = depth_input(gpu);
        let storage = wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC;
        let output = texture(gpu, HDR, storage);
        let history_out = texture(gpu, HDR, storage);
        let frame = uniform(gpu, &frame_bytes());
        let mut params = [0u8; 16];
        params[..4].copy_from_slice(&history_usable.to_le_bytes());
        let params = uniform(gpu, &params);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let views: Vec<wgpu::TextureView> =
            [&scene, &depth, &motion, &history, &output, &history_out]
                .iter()
                .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()))
                .collect();
        let tex = |binding: u32, index: usize| wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::TextureView(&views[index]),
        };
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reject_flag"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params.as_entire_binding(),
                },
                tex(2, 0),
                tex(3, 1),
                tex(4, 2),
                tex(5, 3),
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                tex(7, 4),
                tex(8, 5),
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        gpu.queue().submit([encoder.finish()]);
        read(gpu, &output)
    }

    #[test]
    fn a_pixel_whose_motion_carries_the_reject_flag_does_not_blend_its_history() {
        let test = "a_pixel_whose_motion_carries_the_reject_flag_does_not_blend_its_history";
        let Some(gpu) = software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
        let current_only = resolve(&gpu, 0.0, 0)[PROBE];
        let rejected = resolve(&gpu, 1.0, 1)[PROBE];
        let blended = resolve(&gpu, 0.0, 1)[PROBE];
        let error = block_on(errors.pop());
        assert!(error.is_none(), "validation error: {error:?}");
        assert_eq!(
            rejected, current_only,
            "a rejected pixel keeps its current colour, as without history"
        );
        assert!(
            (blended[1] - current_only[1]).abs() > 1.0e-3,
            "without the flag the green history is blended in: blended {blended:?}, current {current_only:?}"
        );
    }
}
