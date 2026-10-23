//! Shadow casters: the Opaque and Fur draws of a [`LitHandle`] rendered into the 4-layer
//! Depth32Float cascade array (CSM) with the `Shadow` shader variants.
//!
//! Each cascade is one depth-only render pass into its own layer view. Group 0 is the lit
//! renderer's frame bind group; group 1 is never bound (it holds this map and the Shadow
//! variants leave it empty); group 2 is the caster uniform (`RaaShadowCasterGpu` of
//! `PASS_SHADOW_WGSL`, one cascade view-projection per 256 B slot) selected with a dynamic
//! offset. The views, labels, bind group and specs are built once, so a frame allocates
//! nothing beyond what wgpu does.

use std::num::NonZeroU64;

use raa_core::{GpuContext, RaaError};
use raa_lit::{DrawClass, FLAG_DOUBLE_SIDED, FLAG_NO_SHADOW, LitHandle, PipelineSpec};
use raa_math::Mat4;
use raa_shader::{PassKind, VariantKey};

/// Number of cascades (CSM).
const CASCADES: usize = 4;
/// Bytes of one caster slot in the uniform buffer (the dynamic offset alignment).
const SLOT_SIZE: u64 = 256;
/// Bytes of `RaaShadowCasterGpu` (one mat4x4<f32>).
const CASTER_GPU_SIZE: u64 = 64;
// Each 256 B slot (the dynamic offset alignment) holds one caster; 4 slots = 1024 B.
const _: () = assert!(CASTER_GPU_SIZE <= SLOT_SIZE && SLOT_SIZE * CASCADES as u64 == 1024);
/// Depth format of the cascade array (CSM).
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Constant depth bias of the caster pipelines (CSM: constant bias 2).
const CONSTANT_BIAS: i32 = 2;
/// Slope-scaled depth bias of the caster pipelines (CSM: slope bias 2.0).
const SLOPE_BIAS: f32 = 2.0;

/// Render pass label of each cascade layer.
const LABELS: [&str; CASCADES] = [
    "raa.shadow.csm.0",
    "raa.shadow.csm.1",
    "raa.shadow.csm.2",
    "raa.shadow.csm.3",
];

/// The classes that cast shadows, in order: opaque, then fur (drawn as opaque base).
const CAST_CLASSES: [DrawClass; 2] = [DrawClass::Opaque, DrawClass::Fur];

/// Depth32Float, written, compare Less, with the CSM depth biases.
fn depth_state() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(true),
        depth_compare: Some(wgpu::CompareFunction::Less),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState {
            constant: CONSTANT_BIAS,
            slope_scale: SLOPE_BIAS,
            clamp: 0.0,
        },
    }
}

/// The two caster pipeline specs: back faces culled, and DOUBLE_SIDED (no culling).
struct Specs {
    culled: PipelineSpec,
    double_sided: PipelineSpec,
}

impl Specs {
    fn new(extra_layout: Option<wgpu::BindGroupLayout>) -> Specs {
        let culled = PipelineSpec {
            color_targets: Vec::new(),
            depth: Some(depth_state()),
            cull: Some(wgpu::Face::Back),
            alpha_to_coverage: false,
            extra_layout,
        };
        let double_sided = PipelineSpec {
            cull: None,
            ..culled.clone()
        };
        Specs {
            culled,
            double_sided,
        }
    }

    /// The spec for a draw whose material has `flags`.
    fn get(&self, flags: u32) -> &PipelineSpec {
        if flags & FLAG_DOUBLE_SIDED != 0 {
            &self.double_sided
        } else {
            &self.culled
        }
    }
}

/// The cascade array, its views, the caster uniform and the caster pipeline specs.
pub(crate) struct Casters {
    resolution: u32,
    /// Keeps the array alive beside its views.
    _texture: wgpu::Texture,
    /// One D2 view per layer, rendered by the cascade passes.
    layers: [wgpu::TextureView; CASCADES],
    /// D2Array view over the 4 layers, for `LightingResources::set_shadow`.
    array: wgpu::TextureView,
    /// 4 slots of [`SLOT_SIZE`] bytes, each one `RaaShadowCasterGpu`.
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    specs: Specs,
}

impl Casters {
    /// Creates the `resolution` x `resolution` x 4 Depth32Float array, its views, the
    /// group-2 layout (binding 0 uniform, vertex, dynamic offset, 64 B), the 1024 B caster
    /// uniform with its bind group and the two caster specs.
    pub(crate) fn new(gpu: &GpuContext, resolution: u32) -> Casters {
        let device = gpu.device();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("raa.shadow.csm.map"),
            size: wgpu::Extent3d {
                width: resolution.max(1),
                height: resolution.max(1),
                depth_or_array_layers: CASCADES as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let layers = std::array::from_fn(|i| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some(LABELS[i]),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: i as u32,
                array_layer_count: Some(1),
                ..Default::default()
            })
        });
        let array = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("raa.shadow.csm.array"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            base_array_layer: 0,
            array_layer_count: Some(CASCADES as u32),
            ..Default::default()
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa.shadow.caster_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: NonZeroU64::new(CASTER_GPU_SIZE),
                },
                count: None,
            }],
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa.shadow.caster_uniform"),
            size: SLOT_SIZE * CASCADES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa.shadow.caster_group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform,
                    offset: 0,
                    size: NonZeroU64::new(CASTER_GPU_SIZE),
                }),
            }],
        });
        Casters {
            resolution,
            _texture: texture,
            layers,
            array,
            uniform,
            bind_group,
            specs: Specs::new(Some(layout)),
        }
    }

    /// Width and height of each cascade layer in texels.
    pub(crate) fn resolution(&self) -> u32 {
        self.resolution
    }

    /// The D2Array view over the 4 cascades (group 1 binding 0 of the lit passes).
    pub(crate) fn array_view(&self) -> &wgpu::TextureView {
        &self.array
    }

    /// Writes cascade `i`'s view-projection into slot `i` (offset 256 * i).
    pub(crate) fn write(&self, gpu: &GpuContext, matrices: &[Mat4; CASCADES]) {
        let mut bytes = [0u8; CASTER_GPU_SIZE as usize];
        for (i, m) in matrices.iter().enumerate() {
            m.write_le_bytes(&mut bytes);
            gpu.queue()
                .write_buffer(&self.uniform, SLOT_SIZE * i as u64, &bytes);
        }
    }

    /// Renders the shadow casters of `lit` into each cascade layer: the layer is cleared to
    /// 1.0, then every Opaque and Fur draw whose material lacks NO_SHADOW is drawn with the
    /// Shadow variant of its material function (culling back faces unless DOUBLE_SIDED).
    pub(crate) fn record(
        &self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        lit: &LitHandle,
    ) -> Result<(), RaaError> {
        let frame = lit.frame_bind_group();
        for (i, (view, label)) in self.layers.iter().zip(LABELS).enumerate() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view,
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
            pass.set_bind_group(0, &frame, &[]);
            // Group 1 is never bound: it holds this map (Shadow variants leave it empty).
            pass.set_bind_group(2, &self.bind_group, &[(SLOT_SIZE * i as u64) as u32]);
            self.draw_casters(gpu, &mut pass, lit)?;
        }
        Ok(())
    }

    /// Draws the casting Opaque and Fur draws of `lit` into `pass` (groups 0 and 2 bound).
    fn draw_casters(
        &self,
        gpu: &GpuContext,
        pass: &mut wgpu::RenderPass<'_>,
        lit: &LitHandle,
    ) -> Result<(), RaaError> {
        for class in CAST_CLASSES {
            let draws = lit.draws(class);
            for d in draws.iter() {
                if d.instance_count == 0 {
                    continue;
                }
                let flags = lit.material_record(d.material).map_or(0, |m| m.flags);
                if flags & FLAG_NO_SHADOW != 0 {
                    continue;
                }
                let count = lit.bind_mesh(pass, d.mesh)?;
                if count == 0 {
                    continue;
                }
                let flags = flags & FLAG_DOUBLE_SIDED;
                let key = VariantKey {
                    material_fn: d.material_fn,
                    pass: PassKind::Shadow,
                    msaa: 1,
                    flags,
                };
                let pipeline = lit.pipeline_for(gpu, key, self.specs.get(flags))?;
                pass.set_pipeline(&pipeline);
                let instances = d.first_instance..d.first_instance.saturating_add(d.instance_count);
                pass.draw_indexed(0..count, 0, instances);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caster_specs_use_csm_bias_depth_format_and_cull_modes() {
        let specs = Specs::new(None);
        assert_eq!(specs.get(0).cull, Some(wgpu::Face::Back));
        assert_eq!(specs.get(FLAG_NO_SHADOW).cull, Some(wgpu::Face::Back));
        assert_eq!(specs.get(FLAG_DOUBLE_SIDED).cull, None);
        for spec in [&specs.culled, &specs.double_sided] {
            assert!(spec.color_targets.is_empty());
            assert!(!spec.alpha_to_coverage);
            let depth = spec.depth.as_ref().expect("depth state");
            assert_eq!(depth.format, wgpu::TextureFormat::Depth32Float);
            assert_eq!(depth.depth_write_enabled, Some(true));
            assert_eq!(depth.depth_compare, Some(wgpu::CompareFunction::Less));
            assert_eq!(depth.bias.constant, 2);
            assert_eq!(depth.bias.slope_scale, 2.0);
        }
    }
}
