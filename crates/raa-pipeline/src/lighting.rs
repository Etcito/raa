//! Group-1 lighting resources (S10): the shared layout, 1x1 dummies and the bind group the
//! forward, transparent, fur and particle passes bind. Features set their resources; a
//! disabled feature clears them back to the dummies (S12) so the frame stays valid.

use std::num::NonZeroU64;

use raa_core::GpuContext;

use crate::HDR_FORMAT;
use crate::env::EnvGpu;

/// Size in bytes of the `ShadowGpu` uniform (binding 2).
const SHADOW_GPU_SIZE: usize = 320;
/// Size in bytes of the `EnvSh` uniform (binding 4).
const ENV_SH_SIZE: usize = 160;

/// Linear colour of the dummy env cube: the default ambient term (colour (0.6, 0.65, 0.75)
/// x intensity 0.3), so a frame without an env reads the flat ambient.
const DUMMY_ENV_COLOR: [f32; 3] = [0.6 * 0.3, 0.65 * 0.3, 0.75 * 0.3];

/// The seven texture bindings of group 1.
#[derive(Clone, Debug, PartialEq)]
struct Views<V> {
    shadow_map: V,
    env_specular: V,
    brdf_lut: V,
    ao: V,
    contact: V,
    opaque_color: V,
    opaque_depth: V,
}

/// What `upload` has to do: rebuild the bind group and/or write each uniform.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Pending {
    bind_group: bool,
    shadow: bool,
    env_sh: bool,
    env: bool,
}

/// Device-free part of `LightingResources`: current views, uniform contents and the dirty
/// flags. Generic over the view type so the flag logic is tested without a GPU.
struct State<V> {
    dummies: Views<V>,
    views: Views<V>,
    shadow: [u8; SHADOW_GPU_SIZE],
    sh: [[f32; 4]; 9],
    sh_info: [f32; 4],
    env: EnvGpu,
    pending: Pending,
}

impl<V: Clone> State<V> {
    fn new(dummies: Views<V>) -> State<V> {
        State {
            views: dummies.clone(),
            dummies,
            shadow: [0; SHADOW_GPU_SIZE],
            sh: [[0.0; 4]; 9],
            sh_info: [0.0; 4],
            env: EnvGpu::default(),
            pending: Pending::default(),
        }
    }

    fn set_shadow(&mut self, view: V, uniform: &[u8; SHADOW_GPU_SIZE]) {
        self.views.shadow_map = view;
        self.shadow = *uniform;
        self.pending.bind_group = true;
        self.pending.shadow = true;
    }

    fn clear_shadow(&mut self) {
        self.views.shadow_map = self.dummies.shadow_map.clone();
        self.shadow = [0; SHADOW_GPU_SIZE];
        self.pending.bind_group = true;
        self.pending.shadow = true;
    }

    fn set_env(&mut self, cube: V, sh: [[f32; 4]; 9], info: [f32; 4]) {
        self.views.env_specular = cube;
        self.sh = sh;
        self.sh_info = info;
        self.pending.bind_group = true;
        self.pending.env_sh = true;
    }

    fn clear_env(&mut self) {
        self.views.env_specular = self.dummies.env_specular.clone();
        self.sh = [[0.0; 4]; 9];
        self.sh_info = [0.0; 4];
        self.pending.bind_group = true;
        self.pending.env_sh = true;
    }

    fn set_brdf_lut(&mut self, view: V) {
        self.views.brdf_lut = view;
        self.pending.bind_group = true;
    }

    fn set_ao(&mut self, view: V) {
        self.views.ao = view;
        self.pending.bind_group = true;
    }

    fn clear_ao(&mut self) {
        self.views.ao = self.dummies.ao.clone();
        self.pending.bind_group = true;
    }

    fn set_contact(&mut self, view: V) {
        self.views.contact = view;
        self.pending.bind_group = true;
    }

    fn clear_contact(&mut self) {
        self.views.contact = self.dummies.contact.clone();
        self.pending.bind_group = true;
    }

    fn set_opaque_copy(&mut self, color: V, depth: V) {
        self.views.opaque_color = color;
        self.views.opaque_depth = depth;
        self.pending.bind_group = true;
    }

    fn clear_opaque_copy(&mut self) {
        self.views.opaque_color = self.dummies.opaque_color.clone();
        self.views.opaque_depth = self.dummies.opaque_depth.clone();
        self.pending.bind_group = true;
    }

    fn env_mut(&mut self) -> &mut EnvGpu {
        self.pending.env = true;
        &mut self.env
    }

    /// Returns the pending work and marks everything clean.
    fn take_pending(&mut self) -> Pending {
        std::mem::take(&mut self.pending)
    }

    /// Little-endian `EnvSh` bytes: 9 SH coefficients @0, `info` @144.
    fn sh_bytes(&self) -> [u8; ENV_SH_SIZE] {
        let mut out = [0u8; ENV_SH_SIZE];
        let values = self.sh.iter().chain([&self.sh_info]).flat_map(|v| v.iter());
        for (chunk, value) in out.chunks_exact_mut(4).zip(values) {
            chunk.copy_from_slice(&value.to_le_bytes());
        }
        out
    }
}

/// f16 bits of a finite f32 (round to nearest even; values below the f16 normal range flush
/// to zero, values above it become infinity). Used only to fill the dummy textures.
fn f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    if exponent <= 0 {
        return sign;
    }
    if exponent >= 31 {
        return sign | 0x7c00;
    }
    let mut half = ((exponent as u32) << 10) | (mantissa >> 13);
    let rest = mantissa & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && half & 1 == 1) {
        // a carry into the exponent is the correct rounding
        half += 1;
    }
    sign | half as u16
}

/// Little-endian f16 texel bytes.
fn f16_texel(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| f16_bits(*v).to_le_bytes())
        .collect()
}

const VISIBILITY: wgpu::ShaderStages = wgpu::ShaderStages::VERTEX_FRAGMENT;

fn texture_entry(
    binding: u32,
    sample_type: wgpu::TextureSampleType,
    view_dimension: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: VISIBILITY,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32, kind: wgpu::SamplerBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: VISIBILITY,
        ty: wgpu::BindingType::Sampler(kind),
        count: None,
    }
}

fn uniform_entry(binding: u32, size: usize) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: VISIBILITY,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: NonZeroU64::new(size as u64),
        },
        count: None,
    }
}

/// The 12 S10 group-1 bindings with their exact types.
fn layout_entries() -> [wgpu::BindGroupLayoutEntry; 12] {
    use wgpu::TextureSampleType as S;
    use wgpu::TextureViewDimension as D;
    let filterable = S::Float { filterable: true };
    [
        texture_entry(0, S::Depth, D::D2Array),
        sampler_entry(1, wgpu::SamplerBindingType::Comparison),
        uniform_entry(2, SHADOW_GPU_SIZE),
        texture_entry(3, filterable, D::Cube),
        uniform_entry(4, ENV_SH_SIZE),
        texture_entry(5, filterable, D::D2),
        sampler_entry(6, wgpu::SamplerBindingType::Filtering),
        texture_entry(7, filterable, D::D2),
        texture_entry(8, filterable, D::D2),
        texture_entry(9, filterable, D::D2),
        texture_entry(10, S::Float { filterable: false }, D::D2),
        uniform_entry(11, EnvGpu::SIZE),
    ]
}

/// A 1x1 texture (`layers` array layers) filled with `texel` in every layer.
fn dummy_texture(
    gpu: &GpuContext,
    label: &str,
    format: wgpu::TextureFormat,
    layers: u32,
    texel: &[u8],
) -> wgpu::Texture {
    let size = wgpu::Extent3d {
        width: 1,
        height: 1,
        depth_or_array_layers: layers,
    };
    let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let data = texel.repeat(layers as usize);
    gpu.queue().write_texture(
        texture.as_image_copy(),
        &data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(texel.len() as u32),
            rows_per_image: Some(1),
        },
        size,
    );
    texture
}

fn view_of(texture: &wgpu::Texture, dimension: wgpu::TextureViewDimension) -> wgpu::TextureView {
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(dimension),
        ..Default::default()
    })
}

/// Layers of the dummy shadow map. Two, not one: the GL backend of wgpu creates a 1-layer
/// D2 texture as TEXTURE_2D, and a `texture_depth_2d_array` view of it does not read back.
const DUMMY_SHADOW_LAYERS: u32 = 2;

/// Creates the dummy views: depth 1 (2d array), env cube of DUMMY_ENV_COLOR, BRDF LUT
/// (1, 0), ao/contact 1, opaque colour black, opaque depth 1. Depth32Float cannot be
/// written by a copy, so each shadow dummy layer is cleared to 1.0 by a render pass.
fn create_dummies(gpu: &GpuContext) -> Views<wgpu::TextureView> {
    use wgpu::TextureViewDimension as D;
    let shadow = gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("raa-pipeline.dummy.shadow_map"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: DUMMY_SHADOW_LAYERS,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let mut encoder = gpu
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("raa-pipeline.dummy.clear"),
        });
    for layer in 0..DUMMY_SHADOW_LAYERS {
        let attachment = shadow.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(D::D2),
            base_array_layer: layer,
            array_layer_count: Some(1),
            ..Default::default()
        });
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("raa-pipeline.dummy.shadow_clear"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &attachment,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }

    let [r, g, b] = DUMMY_ENV_COLOR;
    let env = dummy_texture(
        gpu,
        "raa-pipeline.dummy.env_specular",
        HDR_FORMAT,
        6,
        &f16_texel(&[r, g, b, 1.0]),
    );
    let lut = dummy_texture(
        gpu,
        "raa-pipeline.dummy.brdf_lut",
        wgpu::TextureFormat::Rg16Float,
        1,
        &f16_texel(&[1.0, 0.0]),
    );
    let one = dummy_texture(
        gpu,
        "raa-pipeline.dummy.one",
        wgpu::TextureFormat::R16Float,
        1,
        &f16_texel(&[1.0]),
    );
    let black = dummy_texture(
        gpu,
        "raa-pipeline.dummy.opaque_color",
        HDR_FORMAT,
        1,
        &f16_texel(&[0.0; 4]),
    );
    let depth = dummy_texture(
        gpu,
        "raa-pipeline.dummy.opaque_depth",
        wgpu::TextureFormat::R32Float,
        1,
        &1.0f32.to_le_bytes(),
    );
    gpu.queue().submit([encoder.finish()]);

    let one_view = view_of(&one, D::D2);
    Views {
        shadow_map: view_of(&shadow, D::D2Array),
        env_specular: view_of(&env, D::Cube),
        brdf_lut: view_of(&lut, D::D2),
        ao: one_view.clone(),
        contact: one_view,
        opaque_color: view_of(&black, D::D2),
        opaque_depth: view_of(&depth, D::D2),
    }
}

fn uniform_buffer(gpu: &GpuContext, label: &str, contents: &[u8]) -> wgpu::Buffer {
    let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: contents.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue().write_buffer(&buffer, 0, contents);
    buffer
}

/// Group-1 lighting resources (S10). Created with 1x1 dummies; feature passes set their
/// resources, and `clear_*` returns them to the dummies. Every setter marks the bind group
/// dirty and/or a uniform changed; the pipeline's prepare uploads them (`upload`).
pub struct LightingResources {
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    shadow_cmp: wgpu::Sampler,
    linear_clamp: wgpu::Sampler,
    shadow_buffer: wgpu::Buffer,
    sh_buffer: wgpu::Buffer,
    env_buffer: wgpu::Buffer,
    state: State<wgpu::TextureView>,
}

impl LightingResources {
    /// Creates the layout, samplers, uniforms and dummies, and a bind group over them.
    pub fn new(gpu: &GpuContext) -> LightingResources {
        let device = gpu.device();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa-pipeline.lighting"),
            entries: &layout_entries(),
        });
        let shadow_cmp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("raa-pipeline.shadow_cmp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let linear_clamp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("raa-pipeline.linear_clamp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let state = State::new(create_dummies(gpu));
        let shadow_buffer = uniform_buffer(gpu, "raa-pipeline.shadow_gpu", &state.shadow);
        let sh_buffer = uniform_buffer(gpu, "raa-pipeline.env_sh", &state.sh_bytes());
        let env_buffer = uniform_buffer(gpu, "raa-pipeline.env_gpu", &state.env.to_bytes());
        let bind_group = build_bind_group(
            gpu,
            &layout,
            &state.views,
            [&shadow_cmp, &linear_clamp],
            [&shadow_buffer, &sh_buffer, &env_buffer],
        );
        LightingResources {
            layout,
            bind_group,
            shadow_cmp,
            linear_clamp,
            shadow_buffer,
            sh_buffer,
            env_buffer,
            state,
        }
    }

    /// The S10 group-1 layout every lit pass uses at `@group(1)`.
    pub fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    /// The current group-1 bind group (rebuilt by the pipeline's prepare when dirty).
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// Sets the cascaded shadow map (a `texture_depth_2d_array` view) and its 320 B
    /// `ShadowGpu` uniform bytes.
    pub fn set_shadow(&mut self, view: wgpu::TextureView, uniform: &[u8; 320]) {
        self.state.set_shadow(view, uniform);
    }

    /// Returns to the dummy shadow map (depth 1) and a zero `ShadowGpu` (enabled 0).
    pub fn clear_shadow(&mut self) {
        self.state.clear_shadow();
    }

    /// Sets the prefiltered env cube, the SH9 irradiance coefficients and `EnvSh.info`
    /// (specular intensity, diffuse intensity, max mip, enabled).
    pub fn set_env(&mut self, cube: wgpu::TextureView, sh: [[f32; 4]; 9], info: [f32; 4]) {
        self.state.set_env(cube, sh, info);
    }

    /// Returns to the dummy ambient-colour cube and a zero `EnvSh` (enabled 0).
    pub fn clear_env(&mut self) {
        self.state.clear_env();
    }

    /// Sets the Rg16Float BRDF LUT (scale, bias).
    pub fn set_brdf_lut(&mut self, view: wgpu::TextureView) {
        self.state.set_brdf_lut(view);
    }

    /// Sets the R16Float screen-space AO texture.
    pub fn set_ao(&mut self, view: wgpu::TextureView) {
        self.state.set_ao(view);
    }

    /// Returns to the dummy AO (1).
    pub fn clear_ao(&mut self) {
        self.state.clear_ao();
    }

    /// Sets the R16Float contact-shadow texture.
    pub fn set_contact(&mut self, view: wgpu::TextureView) {
        self.state.set_contact(view);
    }

    /// Returns to the dummy contact shadow (1).
    pub fn clear_contact(&mut self) {
        self.state.clear_contact();
    }

    /// Sets the opaque HDR colour copy (HDR_FORMAT, with mips) and the R32Float opaque depth.
    pub fn set_opaque_copy(&mut self, color: wgpu::TextureView, depth: wgpu::TextureView) {
        self.state.set_opaque_copy(color, depth);
    }

    /// Returns to the dummy opaque copy (black colour, depth 1).
    pub fn clear_opaque_copy(&mut self) {
        self.state.clear_opaque_copy();
    }

    /// The `EnvGpu` uniform contents.
    pub fn env(&self) -> &EnvGpu {
        &self.state.env
    }

    /// Mutable `EnvGpu`; marks it changed so the next upload writes it.
    pub fn env_mut(&mut self) -> &mut EnvGpu {
        self.state.env_mut()
    }

    /// Writes the changed uniforms and rebuilds the bind group when a view changed.
    // called by the pipeline's prepare (Pipeline::install)
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn upload(&mut self, gpu: &GpuContext) {
        let pending = self.state.take_pending();
        if pending.shadow {
            gpu.queue()
                .write_buffer(&self.shadow_buffer, 0, &self.state.shadow);
        }
        if pending.env_sh {
            gpu.queue()
                .write_buffer(&self.sh_buffer, 0, &self.state.sh_bytes());
        }
        if pending.env {
            gpu.queue()
                .write_buffer(&self.env_buffer, 0, &self.state.env.to_bytes());
        }
        if pending.bind_group {
            self.bind_group = build_bind_group(
                gpu,
                &self.layout,
                &self.state.views,
                [&self.shadow_cmp, &self.linear_clamp],
                [&self.shadow_buffer, &self.sh_buffer, &self.env_buffer],
            );
        }
    }
}

/// Bind group over `views`, the samplers (comparison, linear clamp) and the uniforms
/// (ShadowGpu, EnvSh, EnvGpu) in S10 binding order.
fn build_bind_group(
    gpu: &GpuContext,
    layout: &wgpu::BindGroupLayout,
    views: &Views<wgpu::TextureView>,
    [shadow_cmp, linear_clamp]: [&wgpu::Sampler; 2],
    [shadow, sh, env]: [&wgpu::Buffer; 3],
) -> wgpu::BindGroup {
    use wgpu::BindingResource as R;
    let resources = [
        R::TextureView(&views.shadow_map),
        R::Sampler(shadow_cmp),
        shadow.as_entire_binding(),
        R::TextureView(&views.env_specular),
        sh.as_entire_binding(),
        R::TextureView(&views.brdf_lut),
        R::Sampler(linear_clamp),
        R::TextureView(&views.ao),
        R::TextureView(&views.contact),
        R::TextureView(&views.opaque_color),
        R::TextureView(&views.opaque_depth),
        env.as_entire_binding(),
    ];
    let entries: Vec<wgpu::BindGroupEntry> = resources
        .into_iter()
        .zip(0u32..)
        .map(|(resource, binding)| wgpu::BindGroupEntry { binding, resource })
        .collect();
    gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("raa-pipeline.lighting"),
        layout,
        entries: &entries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummies() -> Views<u32> {
        Views {
            shadow_map: 0,
            env_specular: 1,
            brdf_lut: 2,
            ao: 3,
            contact: 3,
            opaque_color: 4,
            opaque_depth: 5,
        }
    }

    const ONLY_BIND: Pending = Pending {
        bind_group: true,
        shadow: false,
        env_sh: false,
        env: false,
    };

    #[test]
    fn a_new_state_is_clean_and_bound_to_the_dummies() {
        let mut state = State::new(dummies());
        assert_eq!(state.views, dummies());
        assert_eq!(state.take_pending(), Pending::default());
        assert_eq!(state.env, EnvGpu::default());
    }

    #[test]
    fn take_pending_marks_everything_clean() {
        let mut state = State::new(dummies());
        state.set_shadow(10, &[7; SHADOW_GPU_SIZE]);
        state.env_mut().fog[0] = 0.1;
        assert_ne!(state.take_pending(), Pending::default());
        assert_eq!(state.take_pending(), Pending::default());
    }

    #[test]
    fn shadow_setters_mark_the_bind_group_and_the_shadow_uniform() {
        let mut state = State::new(dummies());
        state.set_shadow(10, &[7; SHADOW_GPU_SIZE]);
        assert_eq!(state.views.shadow_map, 10);
        assert_eq!(state.shadow, [7; SHADOW_GPU_SIZE]);
        let expected = Pending {
            shadow: true,
            ..ONLY_BIND
        };
        assert_eq!(state.take_pending(), expected);
        state.clear_shadow();
        assert_eq!(state.views.shadow_map, 0);
        assert_eq!(state.shadow, [0; SHADOW_GPU_SIZE]);
        assert_eq!(state.take_pending(), expected);
    }

    #[test]
    fn env_setters_mark_the_bind_group_and_the_sh_uniform() {
        let mut state = State::new(dummies());
        state.set_env(11, [[0.5; 4]; 9], [1.0, 1.0, 5.0, 1.0]);
        assert_eq!(state.views.env_specular, 11);
        let expected = Pending {
            env_sh: true,
            ..ONLY_BIND
        };
        assert_eq!(state.take_pending(), expected);
        state.clear_env();
        assert_eq!(state.views.env_specular, 1);
        assert_eq!(state.sh_info, [0.0; 4]);
        assert_eq!(state.take_pending(), expected);
    }

    #[test]
    fn texture_setters_mark_only_the_bind_group() {
        let mut state = State::new(dummies());
        type Op = fn(&mut State<u32>);
        type Check = fn(&Views<u32>) -> (u32, u32);
        let ops: [(Op, Check); 7] = [
            (|s| s.set_brdf_lut(12), |v| (v.brdf_lut, 12)),
            (|s| s.set_ao(13), |v| (v.ao, 13)),
            (|s| s.clear_ao(), |v| (v.ao, 3)),
            (|s| s.set_contact(14), |v| (v.contact, 14)),
            (|s| s.clear_contact(), |v| (v.contact, 3)),
            (
                |s| s.set_opaque_copy(15, 16),
                |v| (v.opaque_color * 100 + v.opaque_depth, 1516),
            ),
            (
                |s| s.clear_opaque_copy(),
                |v| (v.opaque_color * 100 + v.opaque_depth, 405),
            ),
        ];
        for (i, (op, check)) in ops.iter().enumerate() {
            op(&mut state);
            let (got, want) = check(&state.views);
            assert_eq!(got, want, "op {i}");
            assert_eq!(state.take_pending(), ONLY_BIND, "op {i}");
        }
    }

    #[test]
    fn env_mut_marks_only_the_env_uniform() {
        let mut state = State::new(dummies());
        state.env_mut().sky_params[2] = 1.0;
        assert_eq!(
            state.take_pending(),
            Pending {
                env: true,
                ..Pending::default()
            }
        );
        assert_eq!(state.env.sky_params, [0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn sh_bytes_put_the_coefficients_at_0_and_info_at_144() {
        let mut state = State::new(dummies());
        let mut sh = [[0.0; 4]; 9];
        for (i, c) in sh.iter_mut().enumerate() {
            c[0] = i as f32;
        }
        state.set_env(11, sh, [2.0, 3.0, 5.0, 1.0]);
        let bytes = state.sh_bytes();
        let at =
            |o: usize| f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
        for i in 0..9 {
            assert_eq!(at(i * 16), i as f32);
        }
        assert_eq!([at(144), at(148), at(152), at(156)], [2.0, 3.0, 5.0, 1.0]);
    }

    #[test]
    fn f16_bits_match_known_values() {
        assert_eq!(f16_bits(0.0), 0x0000);
        assert_eq!(f16_bits(1.0), 0x3C00);
        assert_eq!(f16_bits(0.5), 0x3800);
        assert_eq!(f16_bits(2.0), 0x4000);
        assert_eq!(f16_bits(-1.0), 0xBC00);
        assert_eq!(f16_bits(65504.0), 0x7BFF);
        assert_eq!(f16_bits(1.0e6), 0x7C00);
        // 0.18 lies between 0x31C2 (0.17993) and 0x31C3 (0.18005), nearer the second
        assert_eq!(f16_bits(0.18), 0x31C3);
    }

    /// Real setters and `upload` on the software GPU: the bind group is rebuilt only when a
    /// view changed. The context is leaked like in tests/targets.rs (GL driver teardown).
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn upload_rebuilds_the_bind_group_only_when_dirty() {
        let Some(gpu) = raa_core::testing::software_gpu() else {
            raa_core::testing::skip(
                "upload_rebuilds_the_bind_group_only_when_dirty",
                "software_gpu returned None; see the software_gpu line above",
            );
            return;
        };
        let mut lighting = LightingResources::new(&gpu);
        let first = lighting.bind_group().clone();
        lighting.env_mut().fog = [0.1, 0.2, 0.0, 5.0];
        lighting.upload(&gpu);
        assert_eq!(lighting.bind_group(), &first);

        let ao = dummy_texture(&gpu, "test.ao", wgpu::TextureFormat::R16Float, 1, &[0, 0]);
        lighting.set_ao(ao.create_view(&Default::default()));
        assert!(lighting.state.pending.bind_group);
        lighting.upload(&gpu);
        assert_eq!(lighting.state.pending, Pending::default());
        let second = lighting.bind_group().clone();
        assert_ne!(second, first);

        lighting.clear_ao();
        lighting.upload(&gpu);
        assert_ne!(lighting.bind_group(), &second);
        assert_eq!(lighting.state.views, lighting.state.dummies);
        std::mem::forget(lighting);
        std::mem::forget(gpu);
    }
}
