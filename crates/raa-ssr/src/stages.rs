//! Independently installable SSR Hi-Z and trace stages.

use std::{cell::RefCell, rc::Rc};

use raa_core::{GpuContext, RaaError, wgpu};
use raa_pipeline::{FeaturePass, ImageDesc, ImageDim, Pipeline, PrepareCtx, RecordCtx, Targets};

use crate::{
    SsrDesc, SsrHandle,
    feature::{PARAMS_LEN, SsrCamera, constants, draw, pack_params},
    resources::{HizTargets, SsrPipelines, SsrViewKey, TRACE_FORMAT, TraceTarget},
    tunables,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SsrStageKind {
    Hiz,
    Trace,
}

impl SsrStageKind {
    pub fn type_id(self) -> &'static str {
        match self {
            Self::Hiz => "ssr.hiz",
            Self::Trace => "ssr.trace",
        }
    }
    pub fn from_type_id(id: &str) -> Option<Self> {
        match id {
            "ssr.hiz" => Some(Self::Hiz),
            "ssr.trace" => Some(Self::Trace),
            _ => None,
        }
    }
}

pub fn stage_entry(kind: SsrStageKind, instance: &str) -> String {
    format!("{}#{instance}", kind.type_id())
}

struct State {
    desc: SsrDesc,
    tunables: [f64; 3],
    generation: u64,
    camera: Option<SsrCamera>,
    ran: bool,
    published: Vec<(&'static str, ImageDesc)>,
}

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
    pub fn set_values(&self, values: &[(String, f64)]) -> Result<(), RaaError> {
        let mut s = self.state.borrow_mut();
        let mut desc = s.desc;
        let mut tun = s.tunables;
        for (key, value) in values {
            match (self.kind, key.as_str()) {
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
    bytes: [u8; PARAMS_LEN],
    built: u64,
}

enum StageTargets {
    Hiz(HizTargets),
    Trace(TraceTarget),
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
                StageTargets::Trace(target)
            }
        });
    }
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let (desc, camera, generation, tunables) = {
            let state = self.state.borrow();
            (state.desc, state.camera, state.generation, state.tunables)
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
            StageTargets::Trace(_) => ctx.targets.size(),
        };
        let mut mips = match t {
            StageTargets::Hiz(x) => x.levels.len() as u32,
            StageTargets::Trace(_) => ctx
                .resources
                .input(&self.entry, "hiz")
                .map_or(0, |x| x.desc.mips),
        };
        if let StageTargets::Hiz(h) = t {
            h.set_depth(ctx.gpu.device(), &self.pipelines, &views.depth);
        }
        if let StageTargets::Trace(trace) = t {
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
                trace.set_views(ctx.gpu.device(), &self.pipelines, &hiz, views);
            }
        }
        pack_params(&mut self.bytes, size, mips, &desc, camera.as_ref());
        ctx.gpu
            .queue()
            .write_buffer(&self.pipelines.params, 0, &self.bytes);
        let (w, h) = size;
        let port = if self.kind == SsrStageKind::Hiz {
            "hiz"
        } else {
            "trace"
        };
        let format = if self.kind == SsrStageKind::Hiz {
            wgpu::TextureFormat::R32Float
        } else {
            TRACE_FORMAT
        };
        if self.kind == SsrStageKind::Trace {
            mips = 1;
        }
        let desc = ImageDesc::new(ImageDim::D2, format, w, h, 1, mips);
        let view = match t {
            StageTargets::Hiz(x) => &x.full_view,
            StageTargets::Trace(x) => &x.view,
        };
        ctx.resources.publish(&self.entry, port, view, desc);
        let mut s = self.state.borrow_mut();
        s.published = vec![(port, desc)];
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
            StageTargets::Trace(t) => {
                if ctx.resources.input(&self.entry, "hiz").is_some() {
                    draw(
                        ctx.encoder,
                        &t.view,
                        &self.pipelines.trace,
                        &t.group,
                        Some(ctx.lighting.opaque_bind_group()),
                    );
                    self.state.borrow_mut().ran = true;
                }
            }
        }
        Ok(())
    }
}

/// Handoff: card 'raa-ssr stage passes 2/2: ssr.composite' extends `install_stage` with the Composite kind; today this installs Hiz or Trace through `Pipeline::add` at order 65.
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
        bytes: [0; PARAMS_LEN],
        built: 0,
    }))?;
    Ok(SsrStageHandle { kind, entry, state })
}
