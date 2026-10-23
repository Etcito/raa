//! BRDF LUT (IBL value): 128x128 Rg16Float split-sum scale A and bias B of GGX with
//! height-correlated Smith visibility, computed once on the GPU by `wgsl/brdf_lut.wgsl`.
//! Core WebGPU has no Rg16Float storage texture, so the compute pass writes
//! `pack2x16float(A, B)` into a u32 storage buffer that is copied into the texture.

use raa_core::GpuContext;

/// Width and height of the LUT in texels (x = n_dot_v, y = roughness, texel centres).
pub(crate) const LUT_SIZE: u32 = 128;

/// Texel format of the LUT: (A, B).
pub(crate) const LUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

/// Workgroup edge of `cs_main` (`@workgroup_size(8, 8)`).
const WORKGROUP: u32 = 8;

/// Bytes of one packed texel (pack2x16float) and of one LUT row (512, 256-aligned).
const TEXEL_BYTES: u32 = 4;
const ROW_BYTES: u32 = LUT_SIZE * TEXEL_BYTES;

/// Samples of the CPU reference integral.
pub(crate) const CPU_SAMPLES: u32 = 256;

/// The compute shader filling the packed LUT buffer.
pub(crate) const BRDF_LUT_WGSL: &str = include_str!("wgsl/brdf_lut.wgsl");

/// The LUT texture (TEXTURE_BINDING | COPY_DST) and its D2 view.
#[allow(dead_code, reason = "used by IblFeature::install")]
pub(crate) struct BrdfLut {
    pub(crate) texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
}

/// Creates the storage buffer, texture, pipeline and bind group, records the dispatch and
/// the buffer-to-texture copy into its own encoder and submits once.
#[allow(dead_code, reason = "called once by IblFeature::install")]
pub(crate) fn create_brdf_lut(gpu: &GpuContext) -> BrdfLut {
    let device = gpu.device();
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("raa-ibl.brdf_lut.packed"),
        size: u64::from(ROW_BYTES) * u64::from(LUT_SIZE),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let size = wgpu::Extent3d {
        width: LUT_SIZE,
        height: LUT_SIZE,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("raa-ibl.brdf_lut"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: LUT_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("raa-ibl.brdf_lut"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(buffer.size()),
            },
            count: None,
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("raa-ibl.brdf_lut"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("raa-ibl.brdf_lut"),
        source: wgpu::ShaderSource::Wgsl(BRDF_LUT_WGSL.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("raa-ibl.brdf_lut"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("cs_main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("raa-ibl.brdf_lut"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("raa-ibl.brdf_lut"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("raa-ibl.brdf_lut"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let groups = LUT_SIZE.div_ceil(WORKGROUP);
        pass.dispatch_workgroups(groups, groups, 1);
    }
    encoder.copy_buffer_to_texture(
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(ROW_BYTES),
                rows_per_image: Some(LUT_SIZE),
            },
        },
        texture.as_image_copy(),
        size,
    );
    gpu.queue().submit([encoder.finish()]);

    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2),
        ..Default::default()
    });
    BrdfLut { texture, view }
}

/// CPU port of the `brdf_lut.wgsl` integral in f64 with CPU_SAMPLES Hammersley points:
/// the split-sum (A, B) at (n_dot_v, roughness), alpha = roughness^2.
#[allow(dead_code, reason = "CPU reference of the GPU LUT, used by tests")]
pub(crate) fn cpu_dfg(n_dot_v: f64, roughness: f64) -> [f64; 2] {
    let alpha = roughness * roughness;
    let a2 = alpha * alpha;
    let v = [(1.0 - n_dot_v * n_dot_v).max(0.0).sqrt(), 0.0, n_dot_v];
    let n = f64::from(CPU_SAMPLES);
    let (mut a, mut b) = (0.0, 0.0);
    for i in 0..CPU_SAMPLES {
        let xi = [
            f64::from(i) / n,
            f64::from(i.reverse_bits()) * 2.328_306_436_538_696_3e-10,
        ];
        let phi = 2.0 * std::f64::consts::PI * xi[0];
        let cos_theta = ((1.0 - xi[1]) / (1.0 + (a2 - 1.0) * xi[1])).sqrt();
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let h = [sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta];
        let v_dot_h = v[0] * h[0] + v[1] * h[1] + v[2] * h[2];
        let n_dot_l = 2.0 * v_dot_h * h[2] - v[2];
        let n_dot_h = h[2];
        if n_dot_l > 0.0 && v_dot_h > 0.0 && n_dot_h > 0.0 {
            let ggx_v = n_dot_l * (n_dot_v * n_dot_v * (1.0 - a2) + a2).sqrt();
            let ggx_l = n_dot_v * (n_dot_l * n_dot_l * (1.0 - a2) + a2).sqrt();
            let visibility = 0.5 / (ggx_v + ggx_l).max(1e-12);
            let weight = visibility * 4.0 * n_dot_l * v_dot_h / n_dot_h;
            let fc = (1.0 - v_dot_h).powi(5);
            a += (1.0 - fc) * weight;
            b += fc * weight;
        }
    }
    [a / n, b / n]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brdf_lut_shader_validates() {
        raa_shader::Composer::validate(BRDF_LUT_WGSL).expect("brdf_lut.wgsl validates");
    }

    #[test]
    fn cpu_lut_matches_known_values() {
        let [a, b] = cpu_dfg(1.0, 0.03);
        assert!(
            (a + b - 1.0).abs() <= 0.05,
            "smooth, normal view: A + B = {} (A {a}, B {b}), want 1 +- 0.05",
            a + b
        );
        let [a, b] = cpu_dfg(0.1, 1.0);
        assert!(
            a + b < 0.8,
            "rough, grazing view: A + B = {} (A {a}, B {b}), want < 0.8 (multiscatter loss)",
            a + b
        );
        // G2 <= 1, so A + B <= 1 up to the sampling noise of CPU_SAMPLES points.
        for &(n_dot_v, roughness) in &[(0.5, 0.5), (0.1, 0.03), (1.0, 1.0)] {
            let [a, b] = cpu_dfg(n_dot_v, roughness);
            assert!(
                a.is_finite() && b.is_finite() && a >= 0.0 && b >= 0.0 && a + b <= 1.01,
                "({n_dot_v}, {roughness}): A {a}, B {b} outside [0, 1]"
            );
        }
    }

    /// The GPU context is leaked like in raa-pipeline's tests (the GL backend of the WSL
    /// driver crashed at exit after dropping a device that created half-float textures).
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn brdf_lut_builds() {
        let Some(gpu) = raa_core::testing::software_gpu() else {
            raa_core::testing::skip(
                "brdf_lut_builds",
                "software_gpu returned None; see the software_gpu line above",
            );
            return;
        };
        let scope = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
        let lut = create_brdf_lut(&gpu);
        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
        let error = raa_core::testing::block_on(scope.pop());
        assert!(error.is_none(), "validation error: {error:?}");
        assert_eq!(lut.texture.format(), LUT_FORMAT);
        assert_eq!(
            (lut.texture.width(), lut.texture.height()),
            (LUT_SIZE, LUT_SIZE)
        );
        drop(lut);
        std::mem::forget(gpu);
    }
}
