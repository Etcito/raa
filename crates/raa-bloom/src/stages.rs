//! Repeatable bloom downsample and upsample stages.

use std::{cell::RefCell, num::NonZeroU64, rc::Rc};

use raa_core::{GpuContext, RaaError, wgpu};
use raa_pipeline::{FeaturePass, ImageDesc, ImageDim, Pipeline, PrepareCtx, RecordCtx};

use crate::feature::{self, Pipelines, SLOT_STRIDE, UNIFORM_LEN, write_slot};
use crate::{BloomDesc, filters::half_size, shaders::PARAMS_SIZE};

/// A repeatable bloom stage.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BloomStageKind {
    Down,
    Up,
}

impl BloomStageKind {
    pub fn type_id(self) -> &'static str {
        match self {
            Self::Down => "bloom.down",
            Self::Up => "bloom.up",
        }
    }
    pub fn from_type_id(id: &str) -> Option<Self> {
        match id {
            "bloom.down" => Some(Self::Down),
            "bloom.up" => Some(Self::Up),
            _ => None,
        }
    }
}

pub fn stage_entry(kind: BloomStageKind, instance: &str) -> String {
    format!("{}#{instance}", kind.type_id())
}

#[derive(Clone)]
struct State {
    desc: BloomDesc,
    karis: bool,
    composite: bool,
    level: Option<(wgpu::Texture, wgpu::TextureView, ImageDesc)>,
    ran: bool,
    published: Vec<(&'static str, ImageDesc)>,
}

/// Handle to one independently configured stage.
#[derive(Clone)]
pub struct BloomStageHandle {
    kind: BloomStageKind,
    entry: String,
    state: Rc<RefCell<State>>,
}

impl BloomStageHandle {
    pub fn kind(&self) -> BloomStageKind {
        self.kind
    }
    pub fn entry(&self) -> String {
        self.entry.clone()
    }
    pub fn set_values(&self, values: &[(String, f64)]) -> Result<(), RaaError> {
        let mut s = self
            .state
            .try_borrow_mut()
            .map_err(|_| feature::state_busy())?;
        let mut desc = s.desc;
        let mut karis = s.karis;
        for (key, value) in values {
            match (self.kind, key.as_str()) {
                (_, "radius") => {
                    desc.radius = *value as f32;
                    if desc.validate().is_err() {
                        return invalid(key);
                    }
                }
                (BloomStageKind::Down, "karis") if *value == 0.0 || *value == 1.0 => {
                    karis = *value == 1.0
                }
                (BloomStageKind::Down, "karis") => return invalid(key),
                (BloomStageKind::Up, "strength") => {
                    desc.strength = *value as f32;
                    if desc.validate().is_err() {
                        return invalid(key);
                    }
                }
                _ => return invalid(key),
            }
        }
        if desc.validate().is_err() {
            return invalid(
                if desc.radius.is_finite() && (0.0..=1.0).contains(&desc.radius) {
                    "strength"
                } else {
                    "radius"
                },
            );
        }
        s.desc = desc;
        s.karis = karis;
        Ok(())
    }
    /// Handoff: card 'post_stages extension 1/2: install, values, cameras and stats' calls `BloomStageHandle::set_composite` to enable Up compositing; the upstream handoff is stages.rs:131 `install_stage`.
    pub fn set_composite(&self, on: bool) {
        if self.kind == BloomStageKind::Up
            && let Ok(mut s) = self.state.try_borrow_mut()
        {
            s.composite = on;
        }
    }
    pub fn ran(&self) -> bool {
        self.state.try_borrow().map(|s| s.ran).unwrap_or(false)
    }
    pub fn published(&self) -> Vec<(&'static str, ImageDesc)> {
        self.state
            .try_borrow()
            .map(|s| s.published.clone())
            .unwrap_or_default()
    }
}

fn invalid(key: &str) -> Result<(), RaaError> {
    Err(RaaError::FrameGraph {
        code: "post.invalid",
        message: format!("invalid value for `{key}`"),
    })
}

/// Handoff: card 'raa-bloom stage passes 2/2: bloom.up composite' calls `composite_group` from `feature.rs`; this installer stores `set_composite` state for the Up pass, whose composite draw is added by that card.
pub fn install_stage(
    gpu: &GpuContext,
    pipeline: &Rc<RefCell<Pipeline>>,
    kind: BloomStageKind,
    instance: &str,
) -> Result<BloomStageHandle, RaaError> {
    let entry = stage_entry(kind, instance);
    let mut p = pipeline
        .try_borrow_mut()
        .map_err(|_| RaaError::FrameGraph {
            code: "pipeline.busy",
            message: "pipeline is borrowed".into(),
        })?;
    if p.passes().iter().any(|v| v.name == entry) {
        return Err(RaaError::FrameGraph {
            code: "pipeline.duplicate_pass",
            message: format!("the pass `{entry}` is already installed"),
        });
    }
    let state = Rc::new(RefCell::new(State {
        desc: BloomDesc::default(),
        karis: false,
        composite: false,
        level: None,
        ran: false,
        published: Vec::new(),
    }));
    p.add(Box::new(StagePass::new(
        gpu,
        kind,
        entry.clone(),
        state.clone(),
    )))?;
    Ok(BloomStageHandle { kind, entry, state })
}

struct StagePass {
    kind: BloomStageKind,
    name: String,
    state: Rc<RefCell<State>>,
    pipelines: Pipelines,
    layout: wgpu::BindGroupLayout,
    composite_layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    sampler: wgpu::Sampler,
    slot: usize,
    /// Composite bind group, keyed by the HDR src, HDR dst and bloom level textures it binds:
    /// `prepare` allocates a new level texture each frame, so a key without it would keep
    /// sampling the first frame's level (a frozen bloom that ignores the chain length).
    composite_bound: Option<(wgpu::Texture, wgpu::Texture, wgpu::Texture, wgpu::BindGroup)>,
}

impl StagePass {
    fn new(
        gpu: &GpuContext,
        kind: BloomStageKind,
        name: String,
        state: Rc<RefCell<State>>,
    ) -> Self {
        let device = gpu.device();
        let (pipelines, layout, _) = Pipelines::new(device);
        let composite_layout = feature::create_layout(device, "raa-bloom.stage.composite", true);
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-bloom.stage.params"),
            size: UNIFORM_LEN as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("raa-bloom.linear_clamp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            kind,
            name,
            state,
            pipelines,
            layout,
            composite_layout,
            uniform,
            sampler,
            slot: 0,
            composite_bound: None,
        }
    }
    fn group(&self, device: &wgpu::Device, view: &wgpu::TextureView) -> wgpu::BindGroup {
        let uniform = wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &self.uniform,
                offset: (self.slot * SLOT_STRIDE) as u64,
                size: NonZeroU64::new(PARAMS_SIZE),
            }),
        };
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-bloom.stage.sample"),
            layout: &self.layout,
            entries: &[
                uniform,
                feature::view_entry(1, view),
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }
}

impl FeaturePass for StagePass {
    fn name(&self) -> &str {
        &self.name
    }
    fn order(&self) -> u32 {
        120
    }

    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let input = match self.kind {
            BloomStageKind::Down => ctx.resources.input(&self.name, "src"),
            BloomStageKind::Up => ctx.resources.input(&self.name, "base"),
        };
        let size = match (self.kind, input.as_ref()) {
            (BloomStageKind::Down, Some(img)) => half_size((img.desc.width, img.desc.height)),
            (BloomStageKind::Down, None) => half_size(ctx.size),
            (BloomStageKind::Up, Some(img)) => (img.desc.width, img.desc.height),
            (BloomStageKind::Up, None) => {
                let mut s = self
                    .state
                    .try_borrow_mut()
                    .map_err(|_| feature::state_busy())?;
                s.level = None;
                s.published.clear();
                s.ran = false;
                return Ok(());
            }
        };
        let device = ctx.gpu.device();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("raa-bloom.stage.level"),
            size: wgpu::Extent3d {
                width: size.0.max(1),
                height: size.1.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: raa_pipeline::HDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let desc = ImageDesc::new(ImageDim::D2, raa_pipeline::HDR_FORMAT, size.0, size.1, 1, 1);
        let bloom = self
            .state
            .try_borrow()
            .map_err(|_| feature::state_busy())?
            .desc;
        self.slot = 0;
        let mut bytes = [0; UNIFORM_LEN];
        let sampled = if self.kind == BloomStageKind::Up {
            ctx.resources.input(&self.name, "src")
        } else {
            input.clone()
        };
        write_slot(
            &mut bytes,
            self.slot,
            sampled
                .as_ref()
                .map(|i| (i.desc.width, i.desc.height))
                .unwrap_or(ctx.size),
            bloom.radius,
            bloom.strength,
        );
        if self.kind == BloomStageKind::Up {
            write_slot(
                &mut bytes,
                crate::feature::COMPOSITE_SLOT,
                ctx.size,
                bloom.radius,
                bloom.strength,
            );
        }
        ctx.gpu.queue().write_buffer(&self.uniform, 0, &bytes);
        ctx.resources.publish(&self.name, "level", &view, desc);
        let mut s = self
            .state
            .try_borrow_mut()
            .map_err(|_| feature::state_busy())?;
        s.level = Some((texture, view, desc));
        s.published = vec![("level", desc)];
        s.ran = false;
        Ok(())
    }

    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if self.kind == BloomStageKind::Up {
            let composite = self
                .state
                .try_borrow()
                .map_err(|_| feature::state_busy())?
                .composite;
            let (Some(base), Some(src)) = (
                ctx.resources.input(&self.name, "base"),
                ctx.resources.input(&self.name, "src"),
            ) else {
                if composite {
                    let hdr = ctx.targets.hdr();
                    let extent = wgpu::Extent3d {
                        width: ctx.size.0,
                        height: ctx.size.1,
                        depth_or_array_layers: 1,
                    };
                    ctx.encoder.copy_texture_to_texture(
                        hdr.src_texture().as_image_copy(),
                        hdr.dst_texture().as_image_copy(),
                        extent,
                    );
                    hdr.swap();
                }
                return Ok(());
            };
            let Some((texture, target, desc)) = self
                .state
                .try_borrow()
                .map_err(|_| feature::state_busy())?
                .level
                .as_ref()
                .map(|(t, v, d)| (t.clone(), v.clone(), *d))
            else {
                return Ok(());
            };
            if base.desc.format != raa_pipeline::HDR_FORMAT
                || !base
                    .view
                    .texture()
                    .usage()
                    .contains(wgpu::TextureUsages::COPY_SRC)
                || (base.desc.width, base.desc.height) != (desc.width, desc.height)
            {
                return Ok(());
            }
            ctx.encoder.copy_texture_to_texture(
                base.view.texture().as_image_copy(),
                texture.as_image_copy(),
                wgpu::Extent3d {
                    width: desc.width,
                    height: desc.height,
                    depth_or_array_layers: 1,
                },
            );
            let group = self.group(ctx.gpu.device(), &src.view);
            feature::draw(ctx.encoder, &target, true, &self.pipelines.up, &group);
            if composite {
                let hdr = ctx.targets.hdr();
                let src_texture = hdr.src_texture().clone();
                let dst_texture = hdr.dst_texture().clone();
                let group = match &self.composite_bound {
                    Some((src, dst, level, group))
                        if src == &src_texture && dst == &dst_texture && level == &texture =>
                    {
                        group.clone()
                    }
                    _ => {
                        let group = feature::composite_group(
                            ctx.gpu.device(),
                            &self.composite_layout,
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &self.uniform,
                                    offset: (crate::feature::COMPOSITE_SLOT * SLOT_STRIDE) as u64,
                                    size: NonZeroU64::new(PARAMS_SIZE),
                                }),
                            },
                            &self.sampler,
                            hdr.src(),
                            &target,
                        );
                        self.composite_bound =
                            Some((src_texture, dst_texture, texture.clone(), group.clone()));
                        group
                    }
                };
                feature::draw(
                    ctx.encoder,
                    hdr.dst(),
                    false,
                    &self.pipelines.composite,
                    &group,
                );
                hdr.swap();
            }
            self.state
                .try_borrow_mut()
                .map_err(|_| feature::state_busy())?
                .ran = true;
            return Ok(());
        }
        let Some(input) = ctx.resources.input(&self.name, "src") else {
            return Ok(());
        };
        let Some((_, target, _)) = self
            .state
            .try_borrow()
            .map_err(|_| feature::state_busy())?
            .level
            .as_ref()
            .map(|(t, v, d)| (t.clone(), v.clone(), *d))
        else {
            return Ok(());
        };
        let group = self.group(ctx.gpu.device(), &input.view);
        let karis = self
            .state
            .try_borrow()
            .map_err(|_| feature::state_busy())?
            .karis;
        let pipeline = match karis {
            true => &self.pipelines.down_first,
            false => &self.pipelines.down,
        };
        feature::draw(ctx.encoder, &target, false, pipeline, &group);
        self.state
            .try_borrow_mut()
            .map_err(|_| feature::state_busy())?
            .ran = true;
        Ok(())
    }
}
