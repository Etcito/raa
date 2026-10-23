//! `BloomFeature::install` and the `BloomHandle`: the `bloom` render pass (S12, order 120).
//!
//! On resize the pass creates one HDR_FORMAT texture with the [`mip_sizes`] levels (level 0
//! = the HDR size halved once), a view and the bind groups of every draw. Each frame it
//! downsamples hdr.src() into level 0 (`fs_down_first`, Karis average), each level into the
//! next (`fs_down`), adds the tent upsample of each level into the next larger one
//! (`fs_up`, blend One, One) back to level 0, then composites `mix(hdr, bloom, strength)`
//! into hdr.dst() and swaps. With 0 levels (target smaller than 16 px) it composites a
//! dummy with strength 0, so the HDR chain advances unchanged.
//!
//! The pass and every handle clone share one state (the description). The pass sets no
//! group-1 resource, so disabling it leaves the HDR chain untouched.

use std::cell::RefCell;
use std::num::NonZeroU64;
use std::rc::Rc;

use raa_core::wgpu;
use raa_core::{GpuContext, RaaError};
use raa_pipeline::{FeaturePass, HDR_FORMAT, Pipeline, PrepareCtx, RecordCtx, Targets};

use crate::desc::{BloomDesc, MAX_MIPS, mip_sizes};
use crate::shaders::{
    BLOOM_WGSL, FS_COMPOSITE, FS_DOWN, FS_DOWN_FIRST, FS_UP, PARAMS_SIZE, VS_MAIN,
};

/// S12 name of the pass.
const NAME: &str = "bloom";
/// S12 order of the pass (after `dof`, before `grading`).
const ORDER: u32 = 120;
/// Byte stride of one draw's `BloomParams` in the uniform buffer (the largest
/// min_uniform_buffer_offset_alignment WebGPU allows, so a multiple of every device's).
const SLOT_STRIDE: usize = 256;
/// Draw slots: downsample i at i, upsample into level i at MAX_MIPS + i (i < MAX_MIPS - 1),
/// composite at the last slot.
const SLOTS: usize = 2 * MAX_MIPS;
const COMPOSITE_SLOT: usize = SLOTS - 1;
/// Byte size of the uniform buffer.
const UNIFORM_LEN: usize = SLOT_STRIDE * SLOTS;

fn refusal(code: &'static str, message: &str) -> RaaError {
    RaaError::FrameGraph {
        code,
        message: message.to_string(),
    }
}

/// Same refusal as raa-lit's for a pipeline borrowed while a frame executes.
fn pipeline_busy() -> RaaError {
    refusal(
        "pipeline.busy",
        "the pipeline is borrowed (called while a frame executes)",
    )
}

/// `pipeline.busy` for the shared bloom state (only reachable by a re-entrant call).
fn state_busy() -> RaaError {
    refusal(
        "pipeline.busy",
        "the bloom state is borrowed (re-entrant call while a frame executes)",
    )
}

/// State shared by the pass and every clone of a `BloomHandle`.
struct BloomState {
    desc: BloomDesc,
}

type SharedBloom = Rc<RefCell<BloomState>>;

/// Writes one `BloomParams` (texel_size @0, radius @8, strength @12) at `slot`.
fn write_slot(out: &mut [u8; UNIFORM_LEN], slot: usize, src: (u32, u32), radius: f32, s: f32) {
    let (w, h) = (src.0.max(1) as f32, src.1.max(1) as f32);
    let start = slot * SLOT_STRIDE;
    let Some(dst) = out.get_mut(start..start + PARAMS_SIZE as usize) else {
        return;
    };
    for (chunk, x) in dst.chunks_exact_mut(4).zip([1.0 / w, 1.0 / h, radius, s]) {
        chunk.copy_from_slice(&x.to_le_bytes());
    }
}

/// Packs the params of every draw for an HDR target of `hdr` size whose first `levels`
/// entries of `sizes` are the bloom levels. The composite strength is 0 without levels.
fn pack_params(
    out: &mut [u8; UNIFORM_LEN],
    hdr: (u32, u32),
    sizes: &[(u32, u32); MAX_MIPS],
    levels: usize,
    desc: &BloomDesc,
) {
    let (r, s) = (desc.radius, desc.strength);
    let levels = levels.min(MAX_MIPS);
    // downsample i reads hdr (i = 0) or level i - 1
    let mut src = hdr;
    for (i, &size) in sizes.iter().take(levels).enumerate() {
        write_slot(out, i, src, r, s);
        src = size;
    }
    // upsample into level i reads level i + 1
    for (i, &size) in sizes.iter().take(levels).enumerate().skip(1) {
        write_slot(out, MAX_MIPS + i - 1, size, r, s);
    }
    let strength = if levels == 0 { 0.0 } else { s };
    write_slot(out, COMPOSITE_SLOT, hdr, r, strength);
}

fn layout_entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty,
        count: None,
    }
}

fn texture_type() -> wgpu::BindingType {
    wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float { filterable: true },
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    }
}

/// Group-0 layout: params, source, sampler and, for the composite, the bloom level 0.
fn create_layout(device: &wgpu::Device, label: &str, composite: bool) -> wgpu::BindGroupLayout {
    let mut entries = vec![
        layout_entry(
            0,
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: NonZeroU64::new(PARAMS_SIZE),
            },
        ),
        layout_entry(1, texture_type()),
        layout_entry(
            2,
            wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        ),
    ];
    if composite {
        entries.push(layout_entry(3, texture_type()));
    }
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

fn create_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    layout: &wgpu::BindGroupLayout,
    entry: &str,
    blend: Option<wgpu::BlendState>,
) -> wgpu::RenderPipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(entry),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(entry),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(VS_MAIN),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// The bind groups that read one texture of the HDR chain.
struct HdrGroups {
    texture: wgpu::Texture,
    /// `fs_down_first` from this texture into level 0 (None without levels).
    down_first: Option<wgpu::BindGroup>,
    /// `fs_composite` of this texture and level 0 (or the dummy).
    composite: wgpu::BindGroup,
}

/// The surface-sized bloom levels and the bind group of every draw over them.
struct Chain {
    hdr_size: (u32, u32),
    sizes: [(u32, u32); MAX_MIPS],
    /// Kept alive for the views below (None without levels).
    _texture: Option<wgpu::Texture>,
    /// One single-mip view per level.
    views: Vec<wgpu::TextureView>,
    /// Group k reads level k (downsample into level k + 1).
    down_groups: Vec<wgpu::BindGroup>,
    /// Group k reads level k + 1 (upsample into level k).
    up_groups: Vec<wgpu::BindGroup>,
    /// One entry per texture of the HDR chain (A and B).
    hdr: [HdrGroups; 2],
}

/// The four render pipelines of the pass.
struct Pipelines {
    down_first: wgpu::RenderPipeline,
    down: wgpu::RenderPipeline,
    up: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
}

/// The `bloom` render pass.
struct BloomPass {
    state: SharedBloom,
    sample_layout: wgpu::BindGroupLayout,
    composite_layout: wgpu::BindGroupLayout,
    pipelines: Pipelines,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    bytes: [u8; UNIFORM_LEN],
    /// 1x1 level bound by the composite when the target has no bloom level.
    _dummy: wgpu::Texture,
    dummy_view: wgpu::TextureView,
    chain: Option<Chain>,
}

fn view_entry(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

impl BloomPass {
    fn new(gpu: &GpuContext, state: SharedBloom) -> Self {
        let device = gpu.device();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa-bloom.bloom"),
            source: wgpu::ShaderSource::Wgsl(BLOOM_WGSL.into()),
        });
        let sample_layout = create_layout(device, "raa-bloom.sample_layout", false);
        let composite_layout = create_layout(device, "raa-bloom.composite_layout", true);
        let additive = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let pipelines = Pipelines {
            down_first: create_pipeline(device, &module, &sample_layout, FS_DOWN_FIRST, None),
            down: create_pipeline(device, &module, &sample_layout, FS_DOWN, None),
            up: create_pipeline(
                device,
                &module,
                &sample_layout,
                FS_UP,
                Some(wgpu::BlendState {
                    color: additive,
                    alpha: additive,
                }),
            ),
            composite: create_pipeline(device, &module, &composite_layout, FS_COMPOSITE, None),
        };
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("raa-bloom.linear_clamp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-bloom.params"),
            size: UNIFORM_LEN as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dummy = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("raa-bloom.dummy"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_view = dummy.create_view(&wgpu::TextureViewDescriptor::default());
        BloomPass {
            state,
            sample_layout,
            composite_layout,
            pipelines,
            sampler,
            uniform,
            bytes: [0; UNIFORM_LEN],
            _dummy: dummy,
            dummy_view,
            chain: None,
        }
    }

    fn uniform_entry(&self, slot: usize) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &self.uniform,
                offset: (slot * SLOT_STRIDE) as u64,
                size: NonZeroU64::new(PARAMS_SIZE),
            }),
        }
    }

    /// A group of the sample layout: params at `slot`, `src` and the sampler.
    fn sample_group(
        &self,
        device: &wgpu::Device,
        slot: usize,
        src: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-bloom.sample"),
            layout: &self.sample_layout,
            entries: &[
                self.uniform_entry(slot),
                view_entry(1, src),
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }

    fn hdr_groups(
        &self,
        device: &wgpu::Device,
        texture: &wgpu::Texture,
        view: &wgpu::TextureView,
        views: &[wgpu::TextureView],
    ) -> HdrGroups {
        let level0 = views.first();
        let composite = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-bloom.composite"),
            layout: &self.composite_layout,
            entries: &[
                self.uniform_entry(COMPOSITE_SLOT),
                view_entry(1, view),
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                view_entry(3, level0.unwrap_or(&self.dummy_view)),
            ],
        });
        HdrGroups {
            texture: texture.clone(),
            down_first: level0.map(|_| self.sample_group(device, 0, view)),
            composite,
        }
    }
}

/// One fullscreen draw into `target`, cleared first unless `load`.
fn draw(
    encoder: &mut wgpu::CommandEncoder,
    target: &wgpu::TextureView,
    load: bool,
    pipeline: &wgpu::RenderPipeline,
    group: &wgpu::BindGroup,
) {
    let load = if load {
        wgpu::LoadOp::Load
    } else {
        wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
    };
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("raa-bloom"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, group, &[]);
    pass.draw(0..3, 0..1);
}

impl FeaturePass for BloomPass {
    fn name(&self) -> &str {
        NAME
    }

    fn order(&self) -> u32 {
        ORDER
    }

    /// Recreates the level texture (`mip_sizes` of the target, RENDER_ATTACHMENT |
    /// TEXTURE_BINDING), one view per level and the bind groups of every draw, with the
    /// first downsample and the composite built for both textures of the HDR chain.
    fn resize(&mut self, gpu: &GpuContext, targets: &Targets) {
        let device = gpu.device();
        let hdr_size = targets.size();
        let (sizes, levels) = mip_sizes(hdr_size.0, hdr_size.1);
        let texture = sizes.first().filter(|_| levels > 0).map(|&(w, h)| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("raa-bloom.levels"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: levels as u32,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HDR_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        });
        let views: Vec<wgpu::TextureView> = texture
            .iter()
            .flat_map(|t| {
                (0..levels as u32).map(move |level| {
                    t.create_view(&wgpu::TextureViewDescriptor {
                        label: Some("raa-bloom.level"),
                        base_mip_level: level,
                        mip_level_count: Some(1),
                        ..Default::default()
                    })
                })
            })
            .collect();
        let down_groups = views
            .iter()
            .take(levels.saturating_sub(1))
            .enumerate()
            .map(|(k, src)| self.sample_group(device, k + 1, src))
            .collect();
        let up_groups = views
            .iter()
            .enumerate()
            .skip(1)
            .map(|(k, src)| self.sample_group(device, MAX_MIPS + k - 1, src))
            .collect();
        let hdr = targets.hdr();
        let groups = [
            self.hdr_groups(device, hdr.src_texture(), hdr.src(), &views),
            self.hdr_groups(device, hdr.dst_texture(), hdr.dst(), &views),
        ];
        self.chain = Some(Chain {
            hdr_size,
            sizes,
            _texture: texture,
            views,
            down_groups,
            up_groups,
            hdr: groups,
        });
    }

    /// Writes the params of every draw from the handle's description (in place).
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let desc = self.state.try_borrow().map_err(|_| state_busy())?.desc;
        let Some(chain) = self.chain.as_ref() else {
            return Ok(());
        };
        let levels = chain.views.len();
        pack_params(&mut self.bytes, chain.hdr_size, &chain.sizes, levels, &desc);
        ctx.gpu.queue().write_buffer(&self.uniform, 0, &self.bytes);
        Ok(())
    }

    /// Downsamples hdr.src() through every level, upsamples back to level 0 additively,
    /// composites into hdr.dst() and swaps (one render pass per draw).
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let Some(chain) = self.chain.as_ref() else {
            return Err(RaaError::Resource(
                "bloom: record ran before a resize created the levels".to_owned(),
            ));
        };
        let hdr = ctx.targets.hdr();
        let src = hdr.src_texture();
        let Some(groups) = chain.hdr.iter().find(|g| g.texture == *src) else {
            return Err(RaaError::Resource(
                "bloom: the HDR source is not a texture the pass was resized for".to_owned(),
            ));
        };
        let p = &self.pipelines;
        let encoder = &mut *ctx.encoder;
        if let (Some(level0), Some(group)) = (chain.views.first(), groups.down_first.as_ref()) {
            draw(encoder, level0, false, &p.down_first, group);
        }
        for (target, group) in chain.views.iter().skip(1).zip(&chain.down_groups) {
            draw(encoder, target, false, &p.down, group);
        }
        for (target, group) in chain.views.iter().zip(&chain.up_groups).rev() {
            draw(encoder, target, true, &p.up, group);
        }
        draw(encoder, hdr.dst(), false, &p.composite, &groups.composite);
        hdr.swap();
        Ok(())
    }
}

/// Installs physically based bloom on a pipeline.
pub struct BloomFeature;

impl BloomFeature {
    /// Builds the four render pipelines (`fs_down_first`, `fs_down`, `fs_up` additive,
    /// `fs_composite`), the linear clamp sampler and the params buffer, and adds the
    /// FeaturePass `bloom` (order 120) to `pipeline` with the BLOOM defaults.
    ///
    /// Refusals (nothing installed): `pipeline.busy` when the pipeline is borrowed,
    /// `pipeline.duplicate_pass` when `bloom` is already installed.
    pub fn install(
        gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
    ) -> Result<BloomHandle, RaaError> {
        let mut p = pipeline.try_borrow_mut().map_err(|_| pipeline_busy())?;
        if p.passes().iter().any(|info| info.name == NAME) {
            return Err(refusal(
                "pipeline.duplicate_pass",
                "the pass `bloom` is already installed",
            ));
        }
        let state = Rc::new(RefCell::new(BloomState {
            desc: BloomDesc::default(),
        }));
        p.add(Box::new(BloomPass::new(gpu, state.clone())))?;
        Ok(BloomHandle { state })
    }
}

/// Access to the installed bloom (clones share one state).
#[derive(Clone)]
pub struct BloomHandle {
    state: SharedBloom,
}

impl BloomHandle {
    /// Replaces the description after `BloomDesc::validate` (`bloom.invalid`; the stored
    /// description is unchanged on refusal). `pipeline.busy` on a re-entrant call.
    pub fn set(&self, d: BloomDesc) -> Result<(), RaaError> {
        d.validate()?;
        self.state.try_borrow_mut().map_err(|_| state_busy())?.desc = d;
        Ok(())
    }

    /// The current description (BLOOM defaults until set).
    pub fn get(&self) -> BloomDesc {
        self.state
            .try_borrow()
            .map(|st| st.desc)
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(bytes: &[u8; UNIFORM_LEN], slot: usize) -> [f32; 4] {
        let o = slot * SLOT_STRIDE;
        [0, 4, 8, 12].map(|i| {
            f32::from_le_bytes([
                bytes[o + i],
                bytes[o + i + 1],
                bytes[o + i + 2],
                bytes[o + i + 3],
            ])
        })
    }

    #[test]
    fn pack_params_gives_each_draw_its_source_texel_size() {
        let (sizes, levels) = mip_sizes(1280, 720);
        assert_eq!(levels, 6);
        let desc = BloomDesc {
            strength: 0.5,
            radius: 0.25,
        };
        let mut out = [0u8; UNIFORM_LEN];
        pack_params(&mut out, (1280, 720), &sizes, levels, &desc);
        assert_eq!(slot(&out, 0), [1.0 / 1280.0, 1.0 / 720.0, 0.25, 0.5]);
        assert_eq!(slot(&out, 1), [1.0 / 640.0, 1.0 / 360.0, 0.25, 0.5]);
        assert_eq!(slot(&out, 5), [1.0 / 40.0, 1.0 / 22.0, 0.25, 0.5]);
        // upsample into level 0 reads level 1 (320x180); into level 4 reads level 5 (20x11)
        assert_eq!(slot(&out, MAX_MIPS), [1.0 / 320.0, 1.0 / 180.0, 0.25, 0.5]);
        assert_eq!(
            slot(&out, MAX_MIPS + 4),
            [1.0 / 20.0, 1.0 / 11.0, 0.25, 0.5]
        );
        assert_eq!(slot(&out, MAX_MIPS + 5), [0.0; 4]);
        assert_eq!(
            slot(&out, COMPOSITE_SLOT),
            [1.0 / 1280.0, 1.0 / 720.0, 0.25, 0.5]
        );
    }

    #[test]
    fn pack_params_without_levels_composites_with_strength_zero() {
        let (sizes, levels) = mip_sizes(12, 12);
        assert_eq!(levels, 0);
        let mut out = [0u8; UNIFORM_LEN];
        pack_params(&mut out, (12, 12), &sizes, levels, &BloomDesc::default());
        assert_eq!(slot(&out, 0), [0.0; 4]);
        assert_eq!(
            slot(&out, COMPOSITE_SLOT),
            [1.0 / 12.0, 1.0 / 12.0, 0.85, 0.0]
        );
    }

    #[test]
    fn slots_fit_every_draw_of_the_longest_chain() {
        // MAX_MIPS downsamples, MAX_MIPS - 1 upsamples, one composite
        assert_eq!(MAX_MIPS + (MAX_MIPS - 1) + 1, SLOTS);
        assert!(PARAMS_SIZE as usize <= SLOT_STRIDE);
    }

    #[test]
    fn handle_set_refuses_invalid_and_keeps_desc() {
        let handle = BloomHandle {
            state: Rc::new(RefCell::new(BloomState {
                desc: BloomDesc::default(),
            })),
        };
        let bad = BloomDesc {
            strength: 1.5,
            ..BloomDesc::default()
        };
        let err = handle.set(bad).unwrap_err();
        assert_eq!(err.code(), "bloom.invalid");
        assert_eq!(handle.get(), BloomDesc::default());
        let good = BloomDesc {
            radius: 0.5,
            ..BloomDesc::default()
        };
        handle.set(good).unwrap();
        assert_eq!(handle.clone().get(), good);
    }

    #[cfg(not(target_arch = "wasm32"))]
    mod gpu {
        use std::cell::RefCell;
        use std::rc::Rc;

        use raa_core::testing::{skip, software_gpu};
        use raa_core::{FrameGraph, GpuContext, OffscreenTarget, wgpu};
        use raa_pipeline::{FeaturePass, Pipeline, RecordCtx};

        use super::super::*;

        const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
        /// One Rgba16Float row of 32 px is exactly COPY_BYTES_PER_ROW_ALIGNMENT.
        const ROW: u32 = 32 * 8;

        /// Clears hdr.dst() to a constant colour and swaps (stands in for forward).
        struct Setup(f64);

        impl FeaturePass for Setup {
            fn name(&self) -> &str {
                "test.setup"
            }
            fn order(&self) -> u32 {
                30
            }
            fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
                let hdr = ctx.targets.hdr();
                let c = wgpu::Color {
                    r: self.0,
                    g: self.0,
                    b: self.0,
                    a: 1.0,
                };
                let _pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("test.setup"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: hdr.dst(),
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(c),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                drop(_pass);
                hdr.swap();
                Ok(())
            }
        }

        /// Copies the first row of hdr.src() into a readback buffer after bloom.
        struct Probe(Rc<wgpu::Buffer>);

        impl FeaturePass for Probe {
            fn name(&self) -> &str {
                "test.probe"
            }
            fn order(&self) -> u32 {
                150
            }
            fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
                let size = ctx.targets.size();
                ctx.encoder.copy_texture_to_buffer(
                    ctx.targets.hdr().src_texture().as_image_copy(),
                    wgpu::TexelCopyBufferInfo {
                        buffer: &self.0,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(ROW),
                            rows_per_image: None,
                        },
                    },
                    wgpu::Extent3d {
                        width: size.0.min(32),
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                );
                Ok(())
            }
        }

        fn f16_to_f32(bits: u16) -> f32 {
            let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
            let exp = i32::from((bits >> 10) & 0x1f);
            let frac = f32::from(bits & 0x3ff);
            match exp {
                0 => sign * frac * 2f32.powi(-24),
                _ => sign * (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
            }
        }

        /// Red channel of the first pixel of the HDR chain after one frame at `size`.
        fn run(gpu: &GpuContext, size: u32, desc: Option<BloomDesc>, enabled: bool) -> f32 {
            let target = OffscreenTarget::new(gpu, size, size, FORMAT);
            let pipeline = Rc::new(RefCell::new(Pipeline::new(gpu, FORMAT, false)));
            let buffer = Rc::new(gpu.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("test.readback"),
                size: u64::from(ROW),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            {
                let mut p = pipeline.borrow_mut();
                p.add(Box::new(Setup(2.0))).unwrap();
                p.add(Box::new(Probe(buffer.clone()))).unwrap();
            }
            let handle = BloomFeature::install(gpu, &pipeline).unwrap();
            if let Some(d) = desc {
                handle.set(d).unwrap();
            }
            pipeline.borrow_mut().set_enabled(NAME, enabled).unwrap();
            let err = BloomFeature::install(gpu, &pipeline).err().unwrap();
            assert_eq!(err.code(), "pipeline.duplicate_pass");
            let mut graph = FrameGraph::new(FORMAT);
            Pipeline::install(&pipeline, &mut graph);
            graph.compile().unwrap();
            let commands = graph.execute(gpu, target.view(), (size, size)).unwrap();
            gpu.queue().submit([commands]);
            let slice = buffer.slice(..);
            slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
            gpu.device()
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            let mapped = slice.get_mapped_range().unwrap();
            f16_to_f32(u16::from_le_bytes([mapped[0], mapped[1]]))
        }

        #[test]
        fn constant_hdr_gains_strength_times_levels_minus_one() {
            let Some(gpu) = software_gpu() else {
                skip(
                    "constant_hdr_gains_strength_times_levels_minus_one",
                    "software_gpu returned None; see the software_gpu line above",
                );
                return;
            };
            // 32 px: levels 16 and 8; a constant c accumulates to 2c in level 0, so the
            // composite is mix(c, 2c, s) = c (1 + s)
            let desc = BloomDesc {
                strength: 0.5,
                radius: 0.85,
            };
            let v = run(&gpu, 32, Some(desc), true);
            assert!((v - 3.0).abs() < 1e-2, "bloom of 2.0 gave {v}");
            // disabled: hdr untouched
            let v = run(&gpu, 32, Some(desc), false);
            assert!((v - 2.0).abs() < 1e-3, "disabled bloom gave {v}");
            // 12 px: no level, pass-through composite
            let v = run(&gpu, 12, None, true);
            assert!((v - 2.0).abs() < 1e-3, "bloom without levels gave {v}");
        }
    }
}
