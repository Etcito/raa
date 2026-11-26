//! Shadow casters: the Opaque and Fur draws of a [`LitHandle`] rendered into the 4-layer
//! Depth32Float cascade array (CSM), plus an optional hero layer per area-light shadow slot,
//! with the `Shadow` shader variants.
//!
//! Each cascade and each hero layer is one depth-only render pass into its own layer view.
//! Group 0 is the lit renderer's frame bind group; group 1 is never bound (it holds this map
//! and the Shadow variants leave it empty); group 2 is the caster uniform
//! (`RaaShadowCasterGpu` of `PASS_SHADOW_WGSL`, one caster view-projection per 256 B slot,
//! slot `i` = caster list `i`) selected with a dynamic offset. The views, labels, bind group
//! and specs are built once, so a frame allocates nothing beyond what wgpu does.

use std::num::NonZeroU64;

use raa_core::{GpuContext, RaaError};
use raa_lit::{
    DrawClass, FLAG_DOUBLE_SIDED, FLAG_NO_SHADOW, LitHandle, MAX_HERO_LIGHTS, PipelineSpec,
};
use raa_math::Mat4;
use raa_shader::{PassKind, VariantKey};

/// Number of cascades (CSM).
const CASCADES: usize = 4;
/// Caster lists per frame: 4 cascades followed by up to [`MAX_HERO_LIGHTS`] hero layers.
const SLOTS: usize = CASCADES + MAX_HERO_LIGHTS;
/// Bytes of one caster slot in the uniform buffer (the dynamic offset alignment).
const SLOT_SIZE: u64 = 256;
/// Bytes of `RaaShadowCasterGpu` (one mat4x4<f32>).
const CASTER_GPU_SIZE: u64 = 64;
// Each 256 B slot (the dynamic offset alignment) holds one caster; 8 slots = 2048 B.
const _: () = assert!(CASTER_GPU_SIZE <= SLOT_SIZE && SLOT_SIZE * SLOTS as u64 == 2048);
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

/// Render pass label of each hero layer (array layer `HERO_LAYER_BASE + h`).
const HERO_LABELS: [&str; MAX_HERO_LIGHTS] = [
    "raa.shadow.hero.0",
    "raa.shadow.hero.1",
    "raa.shadow.hero.2",
    "raa.shadow.hero.3",
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
    /// Number of hero layers the array holds (0..=[`MAX_HERO_LIGHTS`]).
    hero_layers: u32,
    /// Keeps the array alive beside its views.
    _texture: wgpu::Texture,
    /// One D2 view per cascade layer, rendered by the cascade passes.
    layers: [wgpu::TextureView; CASCADES],
    /// One D2 view per allocated hero layer, rendered by the hero passes.
    hero_views: [Option<wgpu::TextureView>; MAX_HERO_LIGHTS],
    /// D2Array view over all cascades and hero layers, for `LightingResources::set_shadow`.
    array: wgpu::TextureView,
    /// [`SLOTS`] slots of [`SLOT_SIZE`] bytes, each one `RaaShadowCasterGpu`.
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    specs: Specs,
}

impl Casters {
    /// Creates the `resolution` x `resolution` x `(4 + hero_layers)` Depth32Float array, its
    /// views, the group-2 layout (binding 0 uniform, vertex, dynamic offset, 64 B), the
    /// [`SLOTS`]-slot caster uniform with its bind group and the two caster specs.
    /// `hero_layers` is clamped to `0..=MAX_HERO_LIGHTS`.
    pub(crate) fn new(gpu: &GpuContext, resolution: u32, hero_layers: u32) -> Casters {
        let device = gpu.device();
        let hero_layers = hero_layers.min(MAX_HERO_LIGHTS as u32);
        let depth_or_array_layers = CASCADES as u32 + hero_layers;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("raa.shadow.csm.map"),
            size: wgpu::Extent3d {
                width: resolution.max(1),
                height: resolution.max(1),
                depth_or_array_layers,
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
        let hero_views = std::array::from_fn(|h| {
            ((h as u32) < hero_layers).then(|| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(HERO_LABELS[h]),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: raa_lit::HERO_LAYER_BASE + h as u32,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
        });
        let array = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("raa.shadow.csm.array"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            base_array_layer: 0,
            array_layer_count: Some(depth_or_array_layers),
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
            size: SLOT_SIZE * SLOTS as u64,
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
            hero_layers,
            _texture: texture,
            layers,
            hero_views,
            array,
            uniform,
            bind_group,
            specs: Specs::new(Some(layout)),
        }
    }

    /// Width and height of each layer in texels.
    pub(crate) fn resolution(&self) -> u32 {
        self.resolution
    }

    /// Number of hero layers the array holds.
    pub(crate) fn hero_layers(&self) -> u32 {
        self.hero_layers
    }

    /// The D2Array view over all cascades and hero layers (group 1 binding 0 of the lit passes).
    pub(crate) fn array_view(&self) -> &wgpu::TextureView {
        &self.array
    }

    /// The four single-layer D2 views of the cascade array, in cascade order.
    pub(crate) fn layer_views(&self) -> &[wgpu::TextureView; CASCADES] {
        &self.layers
    }

    /// Writes each caster list's view-projection into its slot (offset 256 * i); slot `i` is
    /// caster list `i` (cascades 0..3, then hero slots 4..7).
    pub(crate) fn write(&self, gpu: &GpuContext, matrices: &[Mat4; SLOTS]) {
        let mut bytes = [0u8; CASTER_GPU_SIZE as usize];
        for (i, m) in matrices.iter().enumerate() {
            m.write_le_bytes(&mut bytes);
            gpu.queue()
                .write_buffer(&self.uniform, SLOT_SIZE * i as u64, &bytes);
        }
    }

    /// Renders the shadow casters of `lit` into each cascade layer, then into hero layer `h`
    /// for every `h < hero_layers` whose `heroes[h]` is `Some`: the layer is cleared to 1.0,
    /// then every Opaque and Fur draw whose material lacks NO_SHADOW is drawn with the Shadow
    /// variant of its material function (culling back faces unless DOUBLE_SIDED). Without
    /// `cascades` (a G frame of forward "layers-interlaced", nothing reads the cascades) only
    /// the hero layers are rendered and the cascade layers keep their previous contents.
    pub(crate) fn record(
        &self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        lit: &LitHandle,
        heroes: &[Option<Mat4>; MAX_HERO_LIGHTS],
        cascades: bool,
    ) -> Result<(), RaaError> {
        if cascades {
            for (i, (view, label)) in self.layers.iter().zip(LABELS).enumerate() {
                self.record_layer(gpu, encoder, lit, view, label, i)?;
            }
        }
        for (h, label) in HERO_LABELS
            .iter()
            .enumerate()
            .take(self.hero_layers as usize)
        {
            if !heroes.get(h).is_some_and(Option::is_some) {
                continue;
            }
            let Some(view) = self.hero_views.get(h).and_then(Option::as_ref) else {
                continue;
            };
            self.record_layer(gpu, encoder, lit, view, label, CASCADES + h)?;
        }
        Ok(())
    }

    /// Renders caster list `list` into `view` in one depth-only pass (groups 0 and 2 bound).
    fn record_layer(
        &self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        lit: &LitHandle,
        view: &wgpu::TextureView,
        label: &str,
        list: usize,
    ) -> Result<(), RaaError> {
        let frame = lit.frame_bind_group();
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
        pass.set_bind_group(2, &self.bind_group, &[(SLOT_SIZE * list as u64) as u32]);
        self.draw_casters(gpu, &mut pass, lit, list)?;
        Ok(())
    }

    /// Draws caster list `list`'s casting Opaque and Fur draws of `lit` into `pass` (groups 0
    /// and 2 bound).
    fn draw_casters(
        &self,
        gpu: &GpuContext,
        pass: &mut wgpu::RenderPass<'_>,
        lit: &LitHandle,
        list: usize,
    ) -> Result<(), RaaError> {
        // One memo per caster pass: shared sub-mesh streams, the id stream and pipelines are
        // bound once, and the hashed pipeline cache is consulted once per variant.
        let mut memo = raa_lit::DrawMemo::new();
        for class in CAST_CLASSES {
            let draws = lit.cascade_draws(list, class);
            for d in draws.iter() {
                if d.instance_count == 0 {
                    continue;
                }
                let flags = lit.material_record(d.material).map_or(0, |m| m.flags);
                if flags & FLAG_NO_SHADOW != 0 {
                    continue;
                }
                let indices = lit.bind_mesh_memo(pass, d.mesh, &mut memo)?;
                if indices.is_empty() {
                    continue;
                }
                let flags = flags & FLAG_DOUBLE_SIDED;
                let key = VariantKey {
                    material_fn: d.material_fn,
                    pass: PassKind::Shadow,
                    msaa: 1,
                    flags,
                };
                let pipeline =
                    lit.pipeline_for_draw_memo(gpu, d.mesh, key, self.specs.get(flags), &mut memo)?;
                memo.set_pipeline(pass, &pipeline);
                let instances = d.first_instance..d.first_instance.saturating_add(d.instance_count);
                pass.draw_indexed(indices, 0, instances);
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
        assert_eq!(SLOTS, 8);
        assert_eq!(SLOT_SIZE * SLOTS as u64, 2048);
        assert_eq!(
            HERO_LABELS,
            [
                "raa.shadow.hero.0",
                "raa.shadow.hero.1",
                "raa.shadow.hero.2",
                "raa.shadow.hero.3",
            ]
        );
        for (h, label) in HERO_LABELS.iter().enumerate() {
            assert_eq!(*label, format!("raa.shadow.hero.{h}"));
        }
    }
}
