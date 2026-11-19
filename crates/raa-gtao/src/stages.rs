//! Independently installable GTAO trace and bilateral blur passes.
use std::{cell::RefCell, rc::Rc};

use raa_core::wgpu;
use raa_core::{GpuContext, RaaError};
use raa_pipeline::{FeaturePass, ImageDesc, ImageDim, Pipeline, PrepareCtx, RecordCtx, Targets};

use crate::{
    AoDesc, AoQuality, GtaoHandle,
    feature::{
        GtaoCamera, Kernel, PARAMS_LEN, WORKGROUP, buffer_entry, create_shader_modules,
        create_texture, pack_params, view_entry,
    },
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AoStageKind {
    Trace,
    Blur,
}

impl AoStageKind {
    pub fn type_id(self) -> &'static str {
        match self {
            Self::Trace => "ao.trace",
            Self::Blur => "ao.blur",
        }
    }
    pub fn from_type_id(id: &str) -> Option<Self> {
        match id {
            "ao.trace" => Some(Self::Trace),
            "ao.blur" => Some(Self::Blur),
            _ => None,
        }
    }
}

pub fn stage_entry(kind: AoStageKind, instance: &str) -> String {
    format!("{}#{instance}", kind.type_id())
}

#[derive(Default)]
struct State {
    desc: AoDesc,
    quality: AoQuality,
    camera: Option<GtaoCamera>,
    ran: bool,
    published: Vec<(&'static str, ImageDesc)>,
}
type Shared = Rc<RefCell<State>>;

#[derive(Clone)]
pub struct AoStageHandle {
    kind: AoStageKind,
    entry: String,
    state: Shared,
}

impl AoStageHandle {
    pub fn kind(&self) -> AoStageKind {
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
    pub fn copy_camera_from(&self, source: &GtaoHandle) {
        self.state.borrow_mut().camera = source.camera();
    }
    pub fn set_values(&self, values: &[(String, f64)]) -> Result<(), RaaError> {
        let mut s = self
            .state
            .try_borrow_mut()
            .map_err(|_| invalid("state is busy"))?;
        let (mut d, mut q) = (s.desc, s.quality);
        for (key, value) in values {
            if self.kind == AoStageKind::Blur {
                return Err(invalid(&format!("unknown key `{key}`")));
            }
            if !value.is_finite() {
                return Err(invalid(&format!("`{key}` must be finite")));
            }
            match key.as_str() {
                "radius" => d.radius_m = *value as f32,
                "intensity" => d.intensity = *value as f32,
                "power" => d.power = *value as f32,
                "slices" if value.fract() == 0.0 && *value >= 0.0 => q.slices = *value as u32,
                "steps" if value.fract() == 0.0 && *value >= 0.0 => q.steps = *value as u32,
                _ => return Err(invalid(&format!("invalid value for `{key}`"))),
            }
        }
        if !(d.radius_m.is_finite() && 0.0 < d.radius_m && d.radius_m <= 10.0) {
            return Err(invalid("`radius` must be finite and in (0, 10]"));
        }
        if !d.intensity.is_finite() || !(0.0..=1.0).contains(&d.intensity) {
            return Err(invalid("`intensity` must be finite and in [0, 1]"));
        }
        if !(d.power.is_finite() && 0.0 < d.power && d.power <= 8.0) {
            return Err(invalid("`power` must be finite and in (0, 8]"));
        }
        q.validate()?;
        s.desc = d;
        s.quality = q;
        Ok(())
    }
}

fn invalid(message: &str) -> RaaError {
    RaaError::FrameGraph {
        code: "ao.invalid",
        message: message.to_owned(),
    }
}

/// Handoff: card 'post_stages extension 1/2: install, values, cameras and stats' calls this
/// installer and the returned handle's value, camera and publication methods. Today it adds
/// one order-40 pass with private defaults, no camera, no group-1 writes, and per-prepare ports.
pub fn install_stage(
    gpu: &GpuContext,
    pipeline: &Rc<RefCell<Pipeline>>,
    kind: AoStageKind,
    instance: &str,
) -> Result<AoStageHandle, RaaError> {
    let entry = stage_entry(kind, instance);
    let mut p = pipeline
        .try_borrow_mut()
        .map_err(|_| RaaError::FrameGraph {
            code: "pipeline.busy",
            message: "pipeline is borrowed".into(),
        })?;
    if p.passes().iter().any(|pass| pass.name == entry) {
        return Err(RaaError::FrameGraph {
            code: "pipeline.duplicate_pass",
            message: format!("pass `{entry}` already exists"),
        });
    }
    let state = Rc::new(RefCell::new(State {
        desc: AoDesc::default(),
        quality: AoQuality::default(),
        ..State::default()
    }));
    p.add(Box::new(StagePass::new(
        gpu,
        kind,
        entry.clone(),
        state.clone(),
    )))?;
    Ok(AoStageHandle { kind, entry, state })
}

struct Surface {
    _raw: wgpu::Texture,
    raw: wgpu::TextureView,
    _blur: wgpu::Texture,
    blur: wgpu::TextureView,
    _ao: wgpu::Texture,
    ao: wgpu::TextureView,
    size: (u32, u32),
    src: Option<wgpu::TextureView>,
    depth: wgpu::TextureView,
    normal: wgpu::TextureView,
    groups: Vec<wgpu::BindGroup>,
}
struct StagePass {
    kind: AoStageKind,
    name: String,
    state: Shared,
    gtao: Kernel,
    h: Kernel,
    v: Kernel,
    uniform: wgpu::Buffer,
    bytes: [u8; PARAMS_LEN],
    surface: Option<Surface>,
}

impl StagePass {
    fn new(gpu: &GpuContext, kind: AoStageKind, name: String, state: Shared) -> Self {
        let (gm, dm) = create_shader_modules(gpu.device());
        let (gtao, h, v) = (
            Kernel::new(gpu.device(), &gm, "gtao"),
            Kernel::new(gpu.device(), &dm, "denoise_h"),
            Kernel::new(gpu.device(), &dm, "denoise_v"),
        );
        let uniform = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-gtao.stage.params"),
            size: crate::shaders::PARAMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            kind,
            name,
            state,
            gtao,
            h,
            v,
            uniform,
            bytes: [0; PARAMS_LEN],
            surface: None,
        }
    }
    fn groups(
        kernels: (&Kernel, &Kernel, &Kernel),
        uniform: &wgpu::Buffer,
        device: &wgpu::Device,
        targets: (&wgpu::TextureView, &wgpu::TextureView, &wgpu::TextureView),
        inputs: (&wgpu::TextureView, &wgpu::TextureView, &wgpu::TextureView),
    ) -> Vec<wgpu::BindGroup> {
        let (gtao, h_kernel, v_kernel) = kernels;
        let (raw, blur, ao) = targets;
        let (src, depth, normal) = inputs;
        let trace = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ao.trace"),
            layout: &gtao.layout,
            entries: &[
                view_entry(0, depth),
                view_entry(1, normal),
                buffer_entry(2, uniform),
                view_entry(3, raw),
            ],
        });
        let h = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ao.blur.h"),
            layout: &h_kernel.layout,
            entries: &[
                view_entry(0, src),
                view_entry(1, depth),
                view_entry(2, blur),
                buffer_entry(4, uniform),
            ],
        });
        let v = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ao.blur.v"),
            layout: &v_kernel.layout,
            entries: &[
                view_entry(0, blur),
                view_entry(1, depth),
                view_entry(3, ao),
                buffer_entry(4, uniform),
            ],
        });
        vec![trace, h, v]
    }
}

impl FeaturePass for StagePass {
    fn name(&self) -> &str {
        &self.name
    }
    fn order(&self) -> u32 {
        40
    }
    fn resize(&mut self, gpu: &GpuContext, targets: &Targets) {
        let (w, h) = targets.size();
        let size = (w.max(1), h.max(1));
        let dev = gpu.device();
        let (raw, rv) = create_texture(dev, "ao.stage.raw", size, wgpu::TextureFormat::R32Float);
        let (blur, bv) = create_texture(dev, "ao.stage.blur", size, wgpu::TextureFormat::R32Float);
        let (ao, av) = create_texture(dev, "ao.stage.ao", size, wgpu::TextureFormat::Rgba16Float);
        let src = rv.clone();
        let depth = targets.depth().clone();
        let normal = targets.normal().clone();
        let groups = Self::groups(
            (&self.gtao, &self.h, &self.v),
            &self.uniform,
            dev,
            (&rv, &bv, &av),
            (&src, &depth, &normal),
        );
        self.surface = Some(Surface {
            _raw: raw,
            raw: rv,
            _blur: blur,
            blur: bv,
            _ao: ao,
            ao: av,
            size,
            src: None,
            depth,
            normal,
            groups,
        });
    }
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let camera = self.state.borrow().camera;
        let mut pubs = Vec::new();
        if let Some(s) = self.surface.as_mut() {
            let depth = ctx.targets.depth_input(ctx.resources, &self.name);
            let normal = ctx.targets.normal_input(ctx.resources, &self.name);
            let src = (self.kind == AoStageKind::Blur)
                .then(|| ctx.resources.input(&self.name, "src"))
                .flatten()
                .map(|p| p.view);
            if s.depth != depth
                || (self.kind == AoStageKind::Trace && s.normal != normal)
                || (self.kind == AoStageKind::Blur && s.src != src)
            {
                let src_view = src.as_ref().unwrap_or(&s.raw);
                s.groups = Self::groups(
                    (&self.gtao, &self.h, &self.v),
                    &self.uniform,
                    ctx.gpu.device(),
                    (&s.raw, &s.blur, &s.ao),
                    (src_view, &depth, &normal),
                );
                s.depth = depth;
                s.normal = normal;
                s.src = src;
            }
            let desc = |fmt| ImageDesc::new(ImageDim::D2, fmt, s.size.0, s.size.1, 1, 1);
            if self.kind == AoStageKind::Trace {
                ctx.resources.publish(
                    &self.name,
                    "raw",
                    &s.raw,
                    desc(wgpu::TextureFormat::R32Float),
                );
                pubs.push(("raw", desc(wgpu::TextureFormat::R32Float)));
            } else {
                ctx.resources.publish(
                    &self.name,
                    "ao",
                    &s.ao,
                    desc(wgpu::TextureFormat::Rgba16Float),
                );
                ctx.resources.publish(
                    &self.name,
                    "blur",
                    &s.blur,
                    desc(wgpu::TextureFormat::R32Float),
                );
                pubs.extend([
                    ("ao", desc(wgpu::TextureFormat::Rgba16Float)),
                    ("blur", desc(wgpu::TextureFormat::R32Float)),
                ]);
            }
            if let Some(cam) = camera {
                let st = self.state.borrow();
                pack_params(
                    &mut self.bytes,
                    &cam,
                    ctx.size,
                    &st.desc,
                    &st.quality,
                    ctx.frame,
                );
                ctx.gpu.queue().write_buffer(&self.uniform, 0, &self.bytes);
            }
        }
        let mut st = self.state.borrow_mut();
        st.published = pubs;
        st.ran = false;
        Ok(())
    }
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let cam = self.state.borrow().camera;
        let Some(s) = self.surface.as_ref() else {
            return Ok(());
        };
        let src = ctx.resources.input(&self.name, "src");
        if cam.is_none() || (self.kind == AoStageKind::Blur && src.is_none()) {
            return Ok(());
        }
        let (x, y) = (s.size.0.div_ceil(WORKGROUP), s.size.1.div_ceil(WORKGROUP));
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ao.stage"),
                timestamp_writes: None,
            });
        if self.kind == AoStageKind::Trace {
            pass.set_pipeline(&self.gtao.pipeline);
            pass.set_bind_group(0, &s.groups[0], &[]);
            pass.dispatch_workgroups(x, y, 1);
        } else {
            for (kernel, group) in [(&self.h, &s.groups[1]), (&self.v, &s.groups[2])] {
                pass.set_pipeline(&kernel.pipeline);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(x, y, 1);
            }
        }
        self.state.borrow_mut().ran = true;
        Ok(())
    }
}
