//! Independently installable SSR stages. Connect Hi-Z to trace, trace and current HDR to
//! incident, and incident plus the same HDR to replacement. The legacy composite shares the
//! shader's incident sampling and replacement functions.
//!
//! Diagnostic layout v1, in texture channel order:
//! - `trace`: hit UV, screen confidence, cone mip (RGBA16F).
//! - `incident`: unweighted scene-linear screen RGB and screen confidence (RGBA32F).
//! - `fallback`: scene-linear volume RGB or the legacy split-sum sky term, and its fraction
//!   of the reflection blend before global intensity (RGBA16F).
//! - `hitMeta` and `fallbackMeta`: source class, geometric/source validity, distance in
//!   metres from the receiver, age in real frames (RGBA32F). Classes are 0 (none), 1 (screen),
//!   2 (volume) and 3 (sky). Zero distance denotes an invalid result or sky. Screen uses
//!   current HDR (age 0), sky is frame-independent (age 0), and volume age is unavailable
//!   (-1). A geometric hit can have zero confidence at a screen edge or on a backface.
//!
//! Screen coverage is limited to on-screen geometry. Offscreen geometry is reachable only
//! when the existing volume fallback is enabled and represents that geometry.

use std::{cell::RefCell, rc::Rc};

use raa_core::{GpuContext, RaaError, wgpu};
use raa_pipeline::{FeaturePass, ImageDesc, ImageDim, Pipeline, PrepareCtx, RecordCtx, Targets};

use crate::{
    SsrDesc, SsrHandle,
    feature::{PARAMS_LEN, SsrCamera, constants, draw, draw_targets, pack_params},
    resources::{
        CompositeTargets, HizTargets, MultiRayIncident, SsrPipelines, SsrViewKey, TRACE_FORMAT, TraceTarget,
        diagnostic_view, replace_group,
    },
    tunables,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SsrStageKind {
    Hiz,
    Trace,
    Composite,
    Incident,
    Replace,
}

impl SsrStageKind {
    pub fn type_id(self) -> &'static str {
        match self {
            Self::Hiz => "ssr.hiz",
            Self::Trace => "ssr.trace",
            Self::Composite => "ssr.composite",
            Self::Incident => "ssr.incident",
            Self::Replace => "ssr.replace",
        }
    }
    pub fn from_type_id(id: &str) -> Option<Self> {
        match id {
            "ssr.hiz" => Some(Self::Hiz),
            "ssr.trace" => Some(Self::Trace),
            "ssr.composite" => Some(Self::Composite),
            "ssr.incident" => Some(Self::Incident),
            "ssr.replace" => Some(Self::Replace),
            _ => None,
        }
    }
}

pub fn stage_entry(kind: SsrStageKind, instance: &str) -> String {
    format!("{}#{instance}", kind.type_id())
}

pub const MAX_EXTRA_RAYS: u32 = 24;

struct State {
    desc: SsrDesc,
    extra_rays: u32,
    trace_source: Option<Rc<RefCell<State>>>,
    tunables: [f64; 3],
    generation: u64,
    camera: Option<SsrCamera>,
    ran: bool,
    published: Vec<(&'static str, ImageDesc)>,
}

/// Settings and publication status for one SSR stage instance.
#[derive(Clone)]
pub struct SsrStageHandle {
    kind: SsrStageKind,
    entry: String,
    state: Rc<RefCell<State>>,
}

impl SsrStageHandle {
    pub fn kind(&self) -> SsrStageKind {
        self.kind
    }
    pub fn entry(&self) -> String {
        self.entry.clone()
    }
    pub fn ran(&self) -> bool {
        self.state.borrow().ran
    }
    pub fn published(&self) -> Vec<(&'static str, ImageDesc)> {
        self.state.borrow().published.clone()
    }
    pub fn copy_camera_from(&self, source: &SsrHandle) {
        self.state.borrow_mut().camera = source.camera();
    }
    /// Use the same traversal limits and depth tolerance as the linked first-ray trace.
    pub fn set_trace_source(&self, source: Option<&Self>) {
        self.state.borrow_mut().trace_source = source.map(|source| source.state.clone());
    }
    pub fn set_values(&self, values: &[(String, f64)]) -> Result<(), RaaError> {
        let mut s = self.state.borrow_mut();
        let mut desc = s.desc;
        let mut tun = s.tunables;
        let mut extra_rays = s.extra_rays;
        for (key, value) in values {
            match (self.kind, key.as_str()) {
                (SsrStageKind::Composite | SsrStageKind::Replace, "intensity") => {
                    desc.intensity = *value as f32;
                    if !value.is_finite() || !(0.0..=2.0).contains(&desc.intensity) {
                        return Err(invalid(&format!("invalid value for `{key}`")));
                    }
                }
                (SsrStageKind::Composite | SsrStageKind::Replace, "maxRoughness") => {
                    desc.max_roughness = *value as f32;
                    if !value.is_finite() || !(0.0..=1.0).contains(&desc.max_roughness) {
                        return Err(invalid(&format!("invalid value for `{key}`")));
                    }
                }
                (SsrStageKind::Incident, "extraRays") => {
                    if value.fract() != 0.0 || !(0.0..=f64::from(MAX_EXTRA_RAYS)).contains(value) {
                        return Err(invalid(&format!("invalid value for `{key}`")));
                    }
                    extra_rays = *value as u32;
                }
                (SsrStageKind::Trace, "maxRoughness") => {
                    desc.max_roughness = *value as f32;
                    if !value.is_finite() || !(0.0..=1.0).contains(&desc.max_roughness) {
                        return Err(invalid(&format!("invalid value for `{key}`")));
                    }
                }
                (SsrStageKind::Trace, "thickness") => {
                    desc.thickness_m = *value as f32;
                    if !value.is_finite()
                        || !(0.0..=10.0).contains(&desc.thickness_m)
                        || desc.thickness_m == 0.0
                    {
                        return Err(invalid(&format!("invalid value for `{key}`")));
                    }
                }
                (SsrStageKind::Trace, _) => {
                    let spec = raa_pipeline::validate_tunable(tunables(), key, *value)
                        .map_err(|m| invalid(&m))?;
                    let i = tunables()
                        .iter()
                        .position(|t| t.key == spec.key)
                        .unwrap_or(0);
                    tun[i] = *value;
                }
                _ => return Err(invalid(&format!("unknown key `{key}`"))),
            }
        }
        desc.validate()?;
        let changed = s.tunables != tun;
        s.desc = desc;
        s.extra_rays = extra_rays;
        s.tunables = tun;
        if changed {
            s.generation += 1;
        }
        Ok(())
    }
}

fn invalid(message: &str) -> RaaError {
    RaaError::FrameGraph {
        code: "post.invalid",
        message: message.to_owned(),
    }
}

struct StagePass {
    entry: String,
    kind: SsrStageKind,
    state: Rc<RefCell<State>>,
    pipelines: SsrPipelines,
    targets: Option<StageTargets>,
    _dummy_hiz: Option<wgpu::Texture>,
    _dummy_trace: Option<wgpu::Texture>,
    bytes: [u8; PARAMS_LEN],
    built: u64,
    multiray: Option<([f64; 3], MultiRayIncident)>,
}

enum StageTargets {
    Hiz(HizTargets),
    Trace(TraceTarget, wgpu::TextureView),
    Composite(CompositeTargets),
    Incident(CompositeTargets, wgpu::TextureView),
    Replace {
        views: SsrViewKey,
        fallback: wgpu::TextureView,
        metadata: wgpu::TextureView,
    },
}

impl FeaturePass for StagePass {
    fn name(&self) -> &str {
        &self.entry
    }
    fn order(&self) -> u32 {
        65
    }
    fn resize(&mut self, gpu: &GpuContext, targets: &Targets) {
        let size = targets.size();
        let size = (size.0.max(1), size.1.max(1));
        let views = SsrViewKey {
            depth: targets.depth().clone(),
            normal: targets.normal().clone(),
            material: targets.material().clone(),
        };
        self.targets = Some(match self.kind {
            SsrStageKind::Hiz => {
                StageTargets::Hiz(HizTargets::new(gpu, &self.pipelines, size, &views.depth))
            }
            SsrStageKind::Trace => {
                let dummy = gpu.device().create_texture(&wgpu::TextureDescriptor {
                    label: Some("raa-ssr.trace_dummy_hiz"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R32Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::STORAGE_BINDING,
                    view_formats: &[],
                });
                let dummy_view = dummy.create_view(&wgpu::TextureViewDescriptor::default());
                let target =
                    TraceTarget::new(gpu.device(), &self.pipelines, size, &dummy_view, views);
                self._dummy_hiz = Some(dummy);
                let metadata = diagnostic_view(
                    gpu.device(),
                    "raa-ssr.hit_meta",
                    size,
                    wgpu::TextureFormat::Rgba32Float,
                );
                StageTargets::Trace(target, metadata)
            }
            SsrStageKind::Composite | SsrStageKind::Incident => {
                let dummy = gpu.device().create_texture(&wgpu::TextureDescriptor {
                    label: Some("raa-ssr.composite_dummy_trace"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: TRACE_FORMAT,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                let trace = dummy.create_view(&wgpu::TextureViewDescriptor::default());
                let target = CompositeTargets::new(
                    gpu.device(),
                    &self.pipelines,
                    size,
                    crate::hiz::mip_count(size.0, size.1),
                    &trace,
                    views,
                );
                self._dummy_trace = Some(dummy);
                if self.kind == SsrStageKind::Incident {
                    StageTargets::Incident(
                        target,
                        diagnostic_view(
                            gpu.device(),
                            "raa-ssr.incident",
                            size,
                            wgpu::TextureFormat::Rgba32Float,
                        ),
                    )
                } else {
                    StageTargets::Composite(target)
                }
            }
            SsrStageKind::Replace => StageTargets::Replace {
                views,
                fallback: diagnostic_view(gpu.device(), "raa-ssr.fallback", size, TRACE_FORMAT),
                metadata: diagnostic_view(
                    gpu.device(),
                    "raa-ssr.fallback_meta",
                    size,
                    wgpu::TextureFormat::Rgba32Float,
                ),
            },
        });
    }
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let (desc, camera, generation, tunables, extra_rays) = {
            let state = self.state.borrow();
            let source = state.trace_source.as_ref().map(|source| source.borrow());
            let (desc, tunables) = source.as_ref().map_or((state.desc, state.tunables),
                |trace| (trace.desc, trace.tunables));
            (desc, state.camera, state.generation, tunables, state.extra_rays)
        };
        if self.kind == SsrStageKind::Trace && self.built != generation {
            self.pipelines
                .rebuild_trace(ctx.gpu.device(), &constants(&tunables));
            self.built = generation;
        }
        let Some(t) = self.targets.as_mut() else {
            return Ok(());
        };
        let views = SsrViewKey {
            depth: ctx.targets.depth_input(ctx.resources, &self.entry),
            normal: ctx.targets.normal_input(ctx.resources, &self.entry),
            material: ctx.targets.material_input(ctx.resources, &self.entry),
        };
        let size = match t {
            StageTargets::Hiz(x) => x.size,
            StageTargets::Trace(..) | StageTargets::Replace { .. } => ctx.targets.size(),
            StageTargets::Composite(x) | StageTargets::Incident(x, _) => x.size,
        };
        let mips = match t {
            StageTargets::Hiz(x) => x.levels.len() as u32,
            StageTargets::Trace(..) => ctx
                .resources
                .input(&self.entry, "hiz")
                .map_or(0, |x| x.desc.mips),
            StageTargets::Composite(x) | StageTargets::Incident(x, _) => x.hdr_views.len() as u32,
            StageTargets::Replace { .. } => 1,
        };
        if let StageTargets::Hiz(h) = t {
            h.set_depth(ctx.gpu.device(), &self.pipelines, &views.depth);
        }
        if let StageTargets::Trace(trace, _) = t {
            let hiz = ctx
                .resources
                .input(&self.entry, "hiz")
                .map(|x| x.view)
                .or_else(|| {
                    self._dummy_hiz
                        .as_ref()
                        .map(|x| x.create_view(&wgpu::TextureViewDescriptor::default()))
                });
            if let Some(hiz) = hiz {
                trace.set_views(ctx.gpu.device(), &self.pipelines, &hiz, views.clone());
            }
        }
        if let StageTargets::Composite(composite) | StageTargets::Incident(composite, _) = t {
            let trace = ctx.resources.input(&self.entry, "trace").map(|x| x.view);
            if let Some(trace) = trace {
                composite.set_views(ctx.gpu.device(), &self.pipelines, &trace, views.clone());
            }
        }
        if let StageTargets::Replace { views: current, .. } = t {
            *current = views;
        }
        pack_params(&mut self.bytes, size, mips, &desc, camera.as_ref());
        if extra_rays > 0 {
            if self.multiray.as_ref().is_none_or(|(built, _)| *built != tunables) {
                self.multiray = Some((tunables, MultiRayIncident::new(
                    ctx.gpu.device(), &self.pipelines, &constants(&tunables),
                )));
            }
            let hiz_mips = ctx.resources.input(&self.entry, "hiz").map_or(0, |image| image.desc.mips);
            self.bytes[284..288].copy_from_slice(&(hiz_mips as f32).to_le_bytes());
            self.bytes[292..296].copy_from_slice(&(extra_rays as f32).to_le_bytes());
        }

        ctx.gpu
            .queue()
            .write_buffer(&self.pipelines.params, 0, &self.bytes);
        let mut s = self.state.borrow_mut();
        s.published.clear();
        let mut publish = |port, view, format, mips| {
            let desc = ImageDesc::new(ImageDim::D2, format, size.0, size.1, 1, mips);
            ctx.resources.publish(&self.entry, port, view, desc);
            s.published.push((port, desc));
        };
        match t {
            StageTargets::Hiz(x) => {
                publish("hiz", &x.full_view, wgpu::TextureFormat::R32Float, mips)
            }
            StageTargets::Trace(x, metadata) => {
                publish("trace", &x.view, TRACE_FORMAT, 1);
                publish("hitMeta", metadata, wgpu::TextureFormat::Rgba32Float, 1);
            }
            StageTargets::Incident(_, x) => {
                publish("incident", x, wgpu::TextureFormat::Rgba32Float, 1);
            }
            StageTargets::Replace {
                fallback, metadata, ..
            } => {
                publish("fallback", fallback, TRACE_FORMAT, 1);
                publish(
                    "fallbackMeta",
                    metadata,
                    wgpu::TextureFormat::Rgba32Float,
                    1,
                );
            }
            StageTargets::Composite(_) => {}
        }
        s.ran = false;
        Ok(())
    }
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let Some(t) = self.targets.as_ref() else {
            return Ok(());
        };
        match t {
            StageTargets::Hiz(t) => {
                let mut pass = ctx
                    .encoder
                    .begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("raa-ssr.hiz"),
                        timestamp_writes: None,
                    });
                for (i, (level, group)) in t.levels.iter().zip(&t.groups).enumerate() {
                    pass.set_pipeline(if i == 0 {
                        &self.pipelines.hiz_copy
                    } else {
                        &self.pipelines.hiz_reduce
                    });
                    pass.set_bind_group(0, group, &[]);
                    pass.dispatch_workgroups(level.groups.0, level.groups.1, 1);
                }
                self.state.borrow_mut().ran = true;
            }
            StageTargets::Trace(t, metadata) => {
                if ctx.resources.input(&self.entry, "hiz").is_some() {
                    draw_targets(
                        ctx.encoder,
                        [&t.view, metadata],
                        &self.pipelines.hit,
                        &t.group,
                        Some(ctx.lighting.opaque_bind_group()),
                    );
                    self.state.borrow_mut().ran = true;
                } else {
                    for view in [&t.view, metadata] {
                        clear(ctx.encoder, view);
                    }
                }
            }
            StageTargets::Incident(t, incident) => {
                let color = ctx.resources.input(&self.entry, "color");
                if let Some(color) = color
                    .filter(|image| {
                        (image.desc.width, image.desc.height) == t.size
                            && image
                                .view
                                .texture()
                                .usage()
                                .contains(wgpu::TextureUsages::COPY_SRC)
                    })
                    .filter(|_| ctx.resources.input(&self.entry, "trace").is_some())
                {
                    build_hdr_mips(ctx.encoder, color.view.texture(), t, &self.pipelines);
                    if self.state.borrow().extra_rays > 0 {
                        let Some(hiz) = ctx.resources.input(&self.entry, "hiz")
                            .filter(|image| (image.desc.width, image.desc.height) == t.size) else {
                            return Err(invalid("additional SSR rays require the scene-sized Hi-Z pyramid"));
                        };
                        let (_, multiray) = self.multiray.as_ref().expect("prepared multiray pass");
                        let group = multiray.group(ctx.gpu.device(), &self.pipelines, t, &hiz.view);
                        draw(ctx.encoder, incident, &multiray.pipeline, &group, None);
                    } else {
                        draw(ctx.encoder, incident, &self.pipelines.incident, &t.composite_group, None);
                    }
                    self.state.borrow_mut().ran = true;
                } else {
                    clear(ctx.encoder, incident);
                }
            }
            StageTargets::Replace {
                views,
                fallback,
                metadata,
            } => {
                let hdr = ctx.targets.hdr();
                if let Some(incident) = ctx.resources.input(&self.entry, "incident") {
                    let group = replace_group(
                        ctx.gpu.device(),
                        &self.pipelines,
                        hdr.src(),
                        &incident.view,
                        views,
                    );
                    draw_targets(
                        ctx.encoder,
                        [hdr.dst(), fallback, metadata],
                        &self.pipelines.replace,
                        &group,
                        Some(ctx.lighting.opaque_bind_group()),
                    );
                } else {
                    ctx.encoder.copy_texture_to_texture(
                        hdr.src_texture().as_image_copy(),
                        hdr.dst_texture().as_image_copy(),
                        hdr.dst_texture().size(),
                    );
                    clear(ctx.encoder, fallback);
                    clear(ctx.encoder, metadata);
                }
                hdr.swap();
                self.state.borrow_mut().ran = true;
            }
            StageTargets::Composite(t) => {
                let hdr = ctx.targets.hdr();
                let extent = wgpu::Extent3d {
                    width: t.size.0,
                    height: t.size.1,
                    depth_or_array_layers: 1,
                };
                if ctx.resources.input(&self.entry, "trace").is_some() {
                    build_hdr_mips(ctx.encoder, hdr.src_texture(), t, &self.pipelines);
                    draw(
                        ctx.encoder,
                        hdr.dst(),
                        &self.pipelines.composite,
                        &t.composite_group,
                        Some(ctx.lighting.opaque_bind_group()),
                    );
                } else {
                    ctx.encoder.copy_texture_to_texture(
                        hdr.src_texture().as_image_copy(),
                        hdr.dst_texture().as_image_copy(),
                        extent,
                    );
                }
                hdr.swap();
                self.state.borrow_mut().ran = true;
            }
        }
        Ok(())
    }
}

fn clear(encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("raa-ssr.clear"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })],
        ..Default::default()
    });
}

fn build_hdr_mips(
    encoder: &mut wgpu::CommandEncoder,
    color: &wgpu::Texture,
    targets: &CompositeTargets,
    pipelines: &SsrPipelines,
) {
    encoder.copy_texture_to_texture(
        color.as_image_copy(),
        targets.hdr_mips.as_image_copy(),
        targets.hdr_mips.size(),
    );
    for (target, group) in targets.hdr_views.iter().skip(1).zip(&targets.down_groups) {
        draw(encoder, target, &pipelines.downsample, group, None);
    }
}

/// Install an independently wired SSR stage with its own settings and render targets.
pub fn install_stage(
    gpu: &GpuContext,
    pipeline: &Rc<RefCell<Pipeline>>,
    kind: SsrStageKind,
    instance: &str,
) -> Result<SsrStageHandle, RaaError> {
    let entry = stage_entry(kind, instance);
    let mut p = pipeline
        .try_borrow_mut()
        .map_err(|_| RaaError::FrameGraph {
            code: "pipeline.busy",
            message: "the pipeline is borrowed".into(),
        })?;
    if p.passes().iter().any(|pass| pass.name == entry) {
        return Err(RaaError::FrameGraph {
            code: "pipeline.duplicate_pass",
            message: format!("the pass `{entry}` is already installed"),
        });
    }
    let state = Rc::new(RefCell::new(State {
        desc: SsrDesc::default(),
        extra_rays: 0,
        trace_source: None,
        tunables: tunables::defaults(),
        generation: 0,
        camera: None,
        ran: false,
        published: Vec::new(),
    }));
    let values = constants(&tunables::defaults());
    let pipelines = SsrPipelines::new(gpu, p.lighting().opaque_layout(), &values);
    p.add(Box::new(StagePass {
        entry: entry.clone(),
        kind,
        state: state.clone(),
        pipelines,
        targets: None,
        _dummy_hiz: None,
        _dummy_trace: None,
        bytes: [0; PARAMS_LEN],
        built: 0,
        multiray: None,
    }))?;
    Ok(SsrStageHandle { kind, entry, state })
}
