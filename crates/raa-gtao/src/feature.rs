//! `GtaoFeature::install` and the `GtaoHandle`: the `ao.gtao` compute pass (S12, order 40)
//! and the `gtao` provider of the `Ao` slot. The pass runs GTAO into a full-resolution
//! R32Float texture, then the horizontal denoise into a second R32Float texture and the
//! vertical denoise into the final Rgba16Float texture (AO in .r, 1 = unoccluded) that the
//! lighting reads at S10 group 1 binding 7 (`set_ao`).
//!
//! The pass and every handle clone share one state (description, camera). Until
//! `GtaoHandle::set_camera` is called, or while the pass is disabled, the lighting keeps the
//! dummy (`clear_ao`, AO 1).

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::{GpuContext, RaaError};
use raa_lit::LitRenderer;
use raa_math::Mat4;
use raa_pipeline::{
    FeaturePass, ImageDesc, ImageDim, LightingResources, Pipeline, PrepareCtx, RecordCtx, Targets,
};

use crate::desc::{AoDesc, AoQuality};
use crate::shaders::{DENOISE_WGSL, GTAO_WGSL, PARAMS_SIZE, ao_provider};

/// S12 name of the pass.
const NAME: &str = "ao.gtao";
/// S12 order of the pass (after `prepass`, before `shadow.contact`).
const ORDER: u32 = 40;
/// Workgroup edge of the three entries (8x8).
pub(crate) const WORKGROUP: u32 = 8;
/// `PARAMS_SIZE` as an array length.
pub(crate) const PARAMS_LEN: usize = PARAMS_SIZE as usize;
/// Byte offsets of the `GtaoParams` fields.
const OFF_PROJ: usize = 0;
const OFF_INV_PROJ: usize = 64;
const OFF_VIEWPORT: usize = 128;
const OFF_PARAMS: usize = 144;
const OFF_QUALITY: usize = 160;
const OFF_PHASE: usize = 176;
/// The frame index given to the noise wraps here so it stays exact as an f32.
const FRAME_WRAP: u64 = 1024;

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

/// `pipeline.busy` for the shared GTAO state (only reachable by a re-entrant call).
fn state_busy() -> RaaError {
    refusal(
        "pipeline.busy",
        "the GTAO state is borrowed (re-entrant call while a frame executes)",
    )
}

/// The camera of the frame as the pass needs it.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct GtaoCamera {
    pub(crate) proj: Mat4,
    pub(crate) inv_proj: Mat4,
}

/// Resolution of the AO (pass-scale study): GTAO and its denoise run on one texel per
/// `factor` x `factor` pixel block (sampled at the block's centre pixel), then one dispatch
/// reconstructs full resolution, bilinear or depth-aware bilateral (4 taps, the denoise's
/// `exp(-|dz| / (0.02 z))` weight). Factor 1 is the original full-resolution pass. With forward
/// "layers-interlaced" and factor 2 the primary pass samples each block at the R frame's phase
/// pixel instead of its centre, and the reconstruction copies the texel there unchanged.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GtaoScale {
    pub factor: u32,
    pub bilateral: bool,
}

impl Default for GtaoScale {
    fn default() -> Self {
        GtaoScale {
            factor: 1,
            bilateral: true,
        }
    }
}

/// State shared by the pass and every clone of a `GtaoHandle`.
struct GtaoState {
    desc: AoDesc,
    quality: AoQuality,
    scale: GtaoScale,
    /// None until the first `set_camera`.
    camera: Option<GtaoCamera>,
}

type SharedGtao = Rc<RefCell<GtaoState>>;

fn write_vec4(out: &mut [u8], v: [f32; 4]) {
    for (chunk, x) in out.as_chunks_mut::<4>().0.iter_mut().zip(v) {
        chunk.copy_from_slice(&x.to_le_bytes());
    }
}

/// Packs `GtaoParams`: proj @0, inv_proj @64, viewport (w, h, 1/w, 1/h) @128, params
/// (radius_m, intensity, power, frame) @144, quality (slices, steps, 0, 0) @160.
pub(crate) fn pack_params(
    out: &mut [u8; PARAMS_LEN],
    camera: &GtaoCamera,
    size: (u32, u32),
    desc: &AoDesc,
    quality: &AoQuality,
    frame: u64,
) {
    let (w, h) = (size.0.max(1) as f32, size.1.max(1) as f32);
    camera.proj.write_le_bytes(&mut out[OFF_PROJ..OFF_INV_PROJ]);
    camera
        .inv_proj
        .write_le_bytes(&mut out[OFF_INV_PROJ..OFF_VIEWPORT]);
    write_vec4(&mut out[OFF_VIEWPORT..OFF_PARAMS], [w, h, 1.0 / w, 1.0 / h]);
    write_vec4(
        &mut out[OFF_PARAMS..OFF_QUALITY],
        [
            desc.radius_m,
            desc.intensity,
            desc.power,
            (frame % FRAME_WRAP) as f32,
        ],
    );
    write_vec4(
        &mut out[OFF_QUALITY..OFF_PHASE],
        [quality.slices as f32, quality.steps as f32, 0.0, 0.0],
    );
    write_vec4(&mut out[OFF_PHASE..PARAMS_LEN], [0.0; 4]);
}

/// The surface-sized textures and the bind group of each dispatch over them.
struct Textures {
    size: (u32, u32),
    /// Kept alive for the views bound below.
    _raw: wgpu::Texture,
    raw_view: wgpu::TextureView,
    _blur: wgpu::Texture,
    blur_view: wgpu::TextureView,
    _out: wgpu::Texture,
    out_view: wgpu::TextureView,
    /// Factor > 1: the low-resolution denoised AO and the reconstruction dispatch over it.
    low: Option<(wgpu::Texture, wgpu::TextureView, wgpu::BindGroup)>,
    /// Pixels per AO texel side and the low-resolution size.
    scale: GtaoScale,
    low_size: (u32, u32),
    gtao_group: wgpu::BindGroup,
    denoise_h_group: wgpu::BindGroup,
    denoise_v_group: wgpu::BindGroup,
    /// The depth and normal views the three groups were built from.
    depth: wgpu::TextureView,
    normal: wgpu::TextureView,
}

/// One compute pipeline and its derived group-0 layout.
pub(crate) struct Kernel {
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) layout: wgpu::BindGroupLayout,
}

impl Kernel {
    pub(crate) fn new(device: &wgpu::Device, module: &wgpu::ShaderModule, entry: &str) -> Self {
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: None,
            module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        let layout = pipeline.get_bind_group_layout(0);
        Kernel { pipeline, layout }
    }
}

pub(crate) fn create_texture(
    device: &wgpu::Device,
    label: &str,
    size: (u32, u32),
    format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

pub(crate) fn view_entry(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

pub(crate) fn buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// The `ao.gtao` compute pass (and its `ao.gtao#<instance>` clones).
struct GtaoPass {
    /// Pipeline entry name: `ao.gtao` or `ao.gtao#<instance>`.
    name: String,
    /// An instance pass never touches group 1.
    instance: bool,
    state: SharedGtao,
    gtao: Kernel,
    denoise_h: Kernel,
    denoise_v: Kernel,
    upsample: Kernel,
    uniform: wgpu::Buffer,
    bytes: [u8; PARAMS_LEN],
    textures: Option<Textures>,
    /// The output view must be (re)given to the lighting at the next prepare with a camera.
    needs_bind: bool,
    /// The lighting holds the output view (false: it holds the dummy).
    bound: bool,
}

impl GtaoPass {
    fn new(gpu: &GpuContext, state: SharedGtao) -> Self {
        Self::named(gpu, state, NAME.to_owned(), false)
    }

    fn named(gpu: &GpuContext, state: SharedGtao, name: String, instance: bool) -> Self {
        let device = gpu.device();
        let (gtao_module, denoise_module) = create_shader_modules(device);
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-gtao.params"),
            size: PARAMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        GtaoPass {
            name,
            instance,
            state,
            gtao: Kernel::new(device, &gtao_module, "gtao"),
            denoise_h: Kernel::new(device, &denoise_module, "denoise_h"),
            denoise_v: Kernel::new(device, &denoise_module, "denoise_v"),
            upsample: Kernel::new(device, &denoise_module, "upsample"),
            uniform,
            bytes: [0; PARAMS_LEN],
            textures: None,
            needs_bind: true,
            bound: false,
        }
    }

    /// (Re)creates the textures at `size` (full) and `scale`: raw and blur R32Float and, for a
    /// factor above 1, the low-resolution Rgba16Float result, all at ceil(size / factor); out
    /// Rgba16Float at full size. The lighting gets the new view at the next prepare.
    fn build(
        &mut self,
        device: &wgpu::Device,
        size: (u32, u32),
        scale: GtaoScale,
        depth: wgpu::TextureView,
        normal: wgpu::TextureView,
    ) {
        let factor = scale.factor.clamp(1, 8);
        let low_size = (size.0.div_ceil(factor), size.1.div_ceil(factor));
        let (raw, raw_view) =
            create_texture(device, "raa-gtao.raw", low_size, wgpu::TextureFormat::R32Float);
        let (blur, blur_view) =
            create_texture(device, "raa-gtao.blur", low_size, wgpu::TextureFormat::R32Float);
        let (out, out_view) = create_texture(
            device,
            "raa-gtao.out",
            size,
            wgpu::TextureFormat::Rgba16Float,
        );
        let low = (factor > 1).then(|| {
            let (texture, view) = create_texture(
                device,
                "raa-gtao.low",
                low_size,
                wgpu::TextureFormat::Rgba16Float,
            );
            let group = self.upsample_group(device, &view, &out_view, &depth);
            (texture, view, group)
        });
        let denoised = low.as_ref().map_or(&out_view, |(_, view, _)| view);
        let (gtao_group, denoise_h_group, denoise_v_group) =
            self.groups(device, &raw_view, &blur_view, denoised, &depth, &normal);
        self.textures = Some(Textures {
            size,
            _raw: raw,
            raw_view,
            _blur: blur,
            blur_view,
            _out: out,
            out_view,
            low,
            scale: GtaoScale { factor, ..scale },
            low_size,
            gtao_group,
            denoise_h_group,
            denoise_v_group,
            depth,
            normal,
        });
        self.needs_bind = true;
    }

    fn upsample_group(
        &self,
        device: &wgpu::Device,
        low_view: &wgpu::TextureView,
        out_view: &wgpu::TextureView,
        depth: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-gtao.upsample"),
            layout: &self.upsample.layout,
            entries: &[
                view_entry(0, low_view),
                view_entry(1, depth),
                view_entry(3, out_view),
                buffer_entry(4, &self.uniform),
            ],
        })
    }

    /// The three groups over `depth` and `normal` for the textures' raw, blur and out views.
    fn groups(
        &self,
        device: &wgpu::Device,
        raw_view: &wgpu::TextureView,
        blur_view: &wgpu::TextureView,
        out_view: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        normal: &wgpu::TextureView,
    ) -> (wgpu::BindGroup, wgpu::BindGroup, wgpu::BindGroup) {
        let gtao_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-gtao.gtao"),
            layout: &self.gtao.layout,
            entries: &[
                view_entry(0, depth),
                view_entry(1, normal),
                buffer_entry(2, &self.uniform),
                view_entry(3, raw_view),
            ],
        });
        let denoise_h_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-gtao.denoise_h"),
            layout: &self.denoise_h.layout,
            entries: &[
                view_entry(0, raw_view),
                view_entry(1, depth),
                view_entry(2, blur_view),
                buffer_entry(4, &self.uniform),
            ],
        });
        let denoise_v_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-gtao.denoise_v"),
            layout: &self.denoise_v.layout,
            entries: &[
                view_entry(0, blur_view),
                view_entry(1, depth),
                view_entry(3, out_view),
                buffer_entry(4, &self.uniform),
            ],
        });
        (gtao_group, denoise_h_group, denoise_v_group)
    }
}

pub(crate) fn create_shader_modules(
    device: &wgpu::Device,
) -> (wgpu::ShaderModule, wgpu::ShaderModule) {
    let gtao = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("raa-gtao.gtao"),
        source: wgpu::ShaderSource::Wgsl(GTAO_WGSL.into()),
    });
    let denoise = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("raa-gtao.denoise"),
        source: wgpu::ShaderSource::Wgsl(DENOISE_WGSL.into()),
    });
    (gtao, denoise)
}

impl FeaturePass for GtaoPass {
    fn name(&self) -> &str {
        &self.name
    }

    fn order(&self) -> u32 {
        ORDER
    }

    /// Drops the full-resolution textures and their groups; `resize` recreates them.
    fn release(&mut self, _gpu: &GpuContext) {
        self.textures = None;
    }

    /// Recreates the full-resolution textures (raw and blur R32Float, out Rgba16Float since
    /// R16Float is not a core storage format) and the bind groups of the three dispatches
    /// over the new depth and normal targets; the lighting gets the new view at the next
    /// prepare. Each bind group lists only the bindings its entry uses (derived layouts).
    fn resize(&mut self, gpu: &GpuContext, targets: &Targets) {
        let (w, h) = targets.size();
        let scale = self.state.try_borrow().map(|st| st.scale).unwrap_or_default();
        self.build(
            gpu.device(),
            (w.max(1), h.max(1)),
            scale,
            targets.depth().clone(),
            targets.normal().clone(),
        );
    }

    /// Disabling resets binding 7 to the dummy (AO 1); enabling gives the output back to
    /// the lighting at the next prepare.
    fn set_enabled(&mut self, enabled: bool, lighting: &mut LightingResources) {
        if self.instance {
            return;
        }
        if enabled {
            self.needs_bind = true;
        } else {
            lighting.clear_ao();
            self.bound = false;
        }
    }

    /// Without a camera or textures: `clear_ao` (once, so the lighting bind group is not
    /// rebuilt every frame while it already holds the dummy). Otherwise writes `GtaoParams`
    /// and, when needed, sets the output view on the lighting.
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let scale = self.state.try_borrow().map_err(|_| state_busy())?.scale;
        let depth = ctx.targets.depth_input(ctx.resources, &self.name);
        let normal = ctx.targets.normal_input(ctx.resources, &self.name);
        let wanted = GtaoScale {
            factor: scale.factor.clamp(1, 8),
            ..scale
        };
        if let Some((size, stale)) = self.textures.as_ref().map(|t| {
            (
                t.size,
                t.depth != depth || t.normal != normal || t.scale != wanted,
            )
        }) && stale
        {
            self.build(ctx.gpu.device(), size, wanted, depth, normal);
        }
        let st = self.state.try_borrow().map_err(|_| state_busy())?;
        if let Some(t) = self.textures.as_ref() {
            let (w, h) = t.size;
            let (lw, lh) = t.low_size;
            for (port, view, format, (w, h)) in [
                ("ao", &t.out_view, wgpu::TextureFormat::Rgba16Float, (w, h)),
                ("raw", &t.raw_view, wgpu::TextureFormat::R32Float, (lw, lh)),
                ("blur", &t.blur_view, wgpu::TextureFormat::R32Float, (lw, lh)),
            ] {
                let desc = ImageDesc::new(ImageDim::D2, format, w, h, 1, 1);
                ctx.resources.publish(&self.name, port, view, desc);
            }
        }
        let (Some(camera), Some(textures)) = (st.camera, self.textures.as_ref()) else {
            if !self.instance {
                if self.bound {
                    ctx.lighting.clear_ao();
                    self.bound = false;
                }
                self.needs_bind = true;
            }
            return Ok(());
        };
        pack_params(
            &mut self.bytes,
            &camera,
            ctx.size,
            &st.desc,
            &st.quality,
            ctx.frame,
        );
        // quality.zw: pixels per AO texel side, bilateral reconstruction (pass-scale study).
        write_vec4(
            &mut self.bytes[OFF_QUALITY..OFF_PHASE],
            [
                st.quality.slices as f32,
                st.quality.steps as f32,
                textures.scale.factor as f32,
                f32::from(u8::from(textures.scale.bilateral)),
            ],
        );
        // Phase mode (primary pass, forward "layers-interlaced", factor 2): each AO texel is
        // computed at the phase pixel of its block and the upsample returns it unchanged there,
        // so the forward's raa_slot_ao at the sampled pixel reads exactly that texel.
        let interlace = ctx.lighting.interlace();
        let phase = (!self.instance && interlace.enabled && textures.scale.factor == 2)
            .then(|| interlace.phase());
        write_vec4(
            &mut self.bytes[OFF_PHASE..PARAMS_LEN],
            match phase {
                Some([px, py]) => [px as f32, py as f32, 1.0, 0.0],
                None => [0.0; 4],
            },
        );
        ctx.gpu.queue().write_buffer(&self.uniform, 0, &self.bytes);
        if !self.instance && self.needs_bind {
            ctx.lighting.set_ao(textures.out_view.clone());
            self.needs_bind = false;
            self.bound = true;
        }
        Ok(())
    }

    /// One compute pass: gtao -> raw, denoise_h raw -> blur, denoise_v blur -> out, each
    /// ceil(w / 8) x ceil(h / 8) workgroups (nothing without a camera).
    ///
    /// The primary pass (the `Ao` slot's texture) records nothing on a G frame of forward
    /// "layers-interlaced" (`LightingInterlace::skip_ao`): the texture stays bound and published
    /// with the previous R frame's AO, which transparent OIT, fur shells and opaque particles
    /// read on that frame (one frame old in screen space).
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if !self.instance && ctx.lighting.interlace().skip_ao {
            return Ok(());
        }
        let has_camera = self
            .state
            .try_borrow()
            .map_err(|_| state_busy())?
            .camera
            .is_some();
        let Some(t) = self.textures.as_ref().filter(|_| has_camera) else {
            return Ok(());
        };
        let groups = (t.low_size.0.div_ceil(WORKGROUP), t.low_size.1.div_ceil(WORKGROUP));
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("raa-gtao.ao"),
                timestamp_writes: None,
            });
        for (kernel, group) in [
            (&self.gtao, &t.gtao_group),
            (&self.denoise_h, &t.denoise_h_group),
            (&self.denoise_v, &t.denoise_v_group),
        ] {
            pass.set_pipeline(&kernel.pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, 1);
        }
        if let Some((_, _, group)) = t.low.as_ref() {
            pass.set_pipeline(&self.upsample.pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(t.size.0.div_ceil(WORKGROUP), t.size.1.div_ceil(WORKGROUP), 1);
        }
        Ok(())
    }
}

/// Installs ground-truth ambient occlusion on a pipeline and its lit renderer.
pub struct GtaoFeature;

impl GtaoFeature {
    /// Adds the `Ao` slot provider `gtao` (priority 10) to `lit`, builds the `gtao`,
    /// `denoise_h` and `denoise_v` compute pipelines (layouts derived from the modules) and
    /// their 160 B uniform, and adds the FeaturePass `ao.gtao` (order 40) to `pipeline`. Its
    /// output reaches the lighting at binding 7 once a camera is set.
    ///
    /// Refusals (nothing installed): `pipeline.busy` when the pipeline is borrowed,
    /// `pipeline.duplicate_pass` when `ao.gtao` is already installed (checked before the slot
    /// is added), the `add_slot` refusal with its code (`shader.slot_conflict` when another
    /// `Ao` provider holds priority 10).
    pub fn install(
        gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
        lit: &mut LitRenderer,
    ) -> Result<GtaoHandle, RaaError> {
        let mut p = pipeline.try_borrow_mut().map_err(|_| pipeline_busy())?;
        if p.passes().iter().any(|info| info.name == NAME) {
            return Err(refusal(
                "pipeline.duplicate_pass",
                "the pass `ao.gtao` is already installed",
            ));
        }
        lit.add_slot(ao_provider())?;
        let state = Rc::new(RefCell::new(GtaoState {
            desc: AoDesc::default(),
            quality: AoQuality::default(),
            scale: GtaoScale::default(),
            camera: None,
        }));
        p.add(Box::new(GtaoPass::new(gpu, state.clone())))?;
        Ok(GtaoHandle { state })
    }

    /// Handoff: card 'Slots 3/3: gtao and contact instance passes in the slots extension'
    /// calls `GtaoFeature::install_instance` and `GtaoHandle::copy_camera_from`. Today it
    /// adds the pass `ao.gtao#<instance>` (order 40) through `GtaoPass::named` with its own
    /// `GtaoState` (default description and quality, no camera) and adds no `Ao` slot
    /// provider; the pass publishes `ao`, `raw` and `blur` under that name, reads depth and
    /// normal through `Targets::depth_input` and `Targets::normal_input` with its own name,
    /// and never calls `set_ao` or `clear_ao`. `copy_camera_from` copies the source handle's
    /// camera into this handle's state, ignored when both share one state or on a
    /// re-entrant borrow.
    ///
    /// Refusals (nothing installed): `pipeline.busy` when the pipeline is borrowed,
    /// `pipeline.duplicate_pass` when `ao.gtao#<instance>` is already installed.
    pub fn install_instance(
        gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
        instance: &str,
    ) -> Result<GtaoHandle, RaaError> {
        let name = format!("{NAME}#{instance}");
        let mut p = pipeline.try_borrow_mut().map_err(|_| pipeline_busy())?;
        if p.passes().iter().any(|info| info.name == name) {
            return Err(refusal(
                "pipeline.duplicate_pass",
                &format!("the pass `{name}` is already installed"),
            ));
        }
        let state = Rc::new(RefCell::new(GtaoState {
            desc: AoDesc::default(),
            quality: AoQuality::default(),
            scale: GtaoScale::default(),
            camera: None,
        }));
        p.add(Box::new(GtaoPass::named(gpu, state.clone(), name, true)))?;
        Ok(GtaoHandle { state })
    }
}

/// Access to the installed GTAO (clones share one state).
#[derive(Clone)]
pub struct GtaoHandle {
    state: SharedGtao,
}

impl GtaoHandle {
    /// Replaces the description after `AoDesc::validate` (`ao.invalid`; the stored
    /// description is unchanged on refusal). `pipeline.busy` on a re-entrant call.
    pub fn set(&self, d: AoDesc) -> Result<(), RaaError> {
        d.validate()?;
        self.state.try_borrow_mut().map_err(|_| state_busy())?.desc = d;
        Ok(())
    }

    /// The current description (GTAO defaults until set).
    pub fn get(&self) -> AoDesc {
        self.state
            .try_borrow()
            .map(|st| st.desc)
            .unwrap_or_default()
    }

    /// Replaces the sampling effort after `AoQuality::validate` (`ao.invalid`; unchanged on
    /// refusal). `pipeline.busy` on a re-entrant call.
    pub fn set_quality(&self, q: AoQuality) -> Result<(), RaaError> {
        q.validate()?;
        self.state
            .try_borrow_mut()
            .map_err(|_| state_busy())?
            .quality = q;
        Ok(())
    }

    /// Sets one S56 tunable. Today it validates through `raa_pipeline::validate_tunable`
    /// against [`crate::tunables`], refuses with `ao.invalid` without changing anything, and
    /// writes `slices`/`steps` through [`GtaoHandle::set_quality`] (a uniform written each
    /// prepare; raa-gtao has no WGSL row to store for the next prepare).
    ///
    /// Handoff: card 'Executor extension: apply Tunable bindings' calls it through
    /// `HandleSet.gtao` (crates/raa-graph/src/exec.rs) with a catalog key and its value as f64.
    pub fn set_tunable(&self, key: &str, value: f64) -> Result<(), RaaError> {
        let spec = raa_pipeline::validate_tunable(crate::tunables(), key, value)
            .map_err(|m| refusal("ao.invalid", &m))?;
        match spec.key {
            "scale" | "upsample" => {
                let mut st = self.state.try_borrow_mut().map_err(|_| state_busy())?;
                if spec.key == "scale" {
                    st.scale.factor = value as u32;
                } else {
                    st.scale.bilateral = value != 0.0;
                }
                return Ok(());
            }
            _ => {}
        }
        let mut q = self.quality();
        match spec.key {
            "slices" => q.slices = value as u32,
            _ => q.steps = value as u32,
        }
        self.set_quality(q)
    }

    /// The current sampling effort (4 x 8 until set).
    pub fn quality(&self) -> AoQuality {
        self.state
            .try_borrow()
            .map(|st| st.quality)
            .unwrap_or_default()
    }

    /// Copies `source`'s camera (None stays None) into this handle's state. Does nothing when
    /// both handles share one state or on a re-entrant borrow.
    pub fn copy_camera_from(&self, source: &GtaoHandle) {
        if Rc::ptr_eq(&self.state, &source.state) {
            return;
        }
        let camera = match source.state.try_borrow() {
            Ok(st) => st.camera,
            Err(_) => return,
        };
        if let Ok(mut st) = self.state.try_borrow_mut() {
            st.camera = camera;
        }
    }

    pub(crate) fn camera(&self) -> Option<GtaoCamera> {
        self.state.try_borrow().ok().and_then(|state| state.camera)
    }

    /// The frame's camera (called every frame): `proj` of the scene camera and `inv_proj`
    /// its inverse. Ignored on a re-entrant call.
    pub fn set_camera(&self, proj: &Mat4, inv_proj: &Mat4) {
        if let Ok(mut st) = self.state.try_borrow_mut() {
            st.camera = Some(GtaoCamera {
                proj: *proj,
                inv_proj: *inv_proj,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use raa_math::Vec3;

    #[test]
    fn set_tunable_writes_quality_and_refusals_change_nothing() {
        let h = GtaoHandle {
            state: Rc::new(RefCell::new(GtaoState {
                desc: AoDesc::default(),
                quality: AoQuality::default(),
                scale: GtaoScale::default(),
                camera: None,
            })),
        };
        h.set_tunable("slices", 6.0).unwrap();
        h.set_tunable("steps", 12.0).unwrap();
        let q = h.quality();
        for (k, v) in [
            ("slices", 9.0),
            ("steps", 0.0),
            ("steps", 2.5),
            ("slices", f64::NAN),
            ("x", 1.0),
        ] {
            let e = h.set_tunable(k, v).unwrap_err();
            assert_eq!(e.code(), "ao.invalid");
            assert_eq!(h.quality(), q);
        }
        assert_eq!((q.slices, q.steps), (6, 12));
    }

    fn f32_at(bytes: &[u8], offset: usize) -> f32 {
        f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    }

    #[test]
    fn pack_params_puts_viewport_at_128_and_params_at_144() {
        let camera = GtaoCamera {
            proj: Mat4::from_scale(Vec3::new(2.0, 3.0, 4.0)),
            inv_proj: Mat4::from_scale(Vec3::new(0.5, 0.25, 0.125)),
        };
        let desc = AoDesc {
            radius_m: 0.5,
            intensity: 0.75,
            power: 2.0,
        };
        let mut out = [0u8; PARAMS_LEN];
        let quality = AoQuality {
            slices: 2,
            steps: 5,
        };
        pack_params(&mut out, &camera, (1280, 720), &desc, &quality, 7);
        assert_eq!(out.len() as u64, PARAMS_SIZE);
        assert_eq!(f32_at(&out, 0), 2.0);
        assert_eq!(f32_at(&out, 64), 0.5);
        let at = |o: usize| [0, 4, 8, 12].map(|i| f32_at(&out, o + i));
        assert_eq!(at(128), [1280.0, 720.0, 1.0 / 1280.0, 1.0 / 720.0]);
        assert_eq!(at(144), [0.5, 0.75, 2.0, 7.0]);
        assert_eq!(at(160), [2.0, 5.0, 0.0, 0.0]);
        assert_eq!(at(176), [0.0; 4]);
    }

    #[test]
    fn handle_set_refuses_invalid_and_keeps_desc() {
        let handle = GtaoHandle {
            state: Rc::new(RefCell::new(GtaoState {
                desc: AoDesc::default(),
                quality: AoQuality::default(),
                scale: GtaoScale::default(),
                camera: None,
            })),
        };
        let bad = AoDesc {
            radius_m: 0.0,
            ..AoDesc::default()
        };
        let err = handle.set(bad).unwrap_err();
        assert_eq!(err.code(), "ao.invalid");
        assert_eq!(handle.get(), AoDesc::default());
        let good = AoDesc {
            power: 2.0,
            ..AoDesc::default()
        };
        handle.set(good).unwrap();
        assert_eq!(handle.clone().get(), good);
        assert_eq!(handle.quality(), AoQuality::default());
        let bad_quality = AoQuality {
            slices: 0,
            steps: 4,
        };
        assert!(handle.set_quality(bad_quality).is_err());
        let low = AoQuality {
            slices: 2,
            steps: 4,
        };
        handle.set_quality(low).unwrap();
        assert_eq!(handle.clone().quality(), low);
        assert_eq!(handle.get(), good);
    }

    #[test]
    fn prepare_publishes_ao_raw_and_blur() {
        let Some(gpu) = raa_core::testing::software_gpu() else {
            raa_core::testing::skip("prepare_publishes_ao_raw_and_blur", "software_gpu is None");
            return;
        };
        let state = Rc::new(RefCell::new(GtaoState {
            desc: AoDesc::default(),
            quality: AoQuality::default(),
            scale: GtaoScale::default(),
            camera: None,
        }));
        let mut pass = GtaoPass::new(&gpu, state);
        let targets = Targets::new(&gpu, (8, 8));
        pass.resize(&gpu, &targets);
        let table = raa_pipeline::ResourceTable::new();
        let mut lighting = LightingResources::new(&gpu);
        let mut ctx = PrepareCtx {
            gpu: &gpu,
            size: (8, 8),
            frame: 0,
            time_s: 0.0,
            dt_s: 0.0,
            targets: &targets,
            lighting: &mut lighting,
            resources: &table,
        };
        pass.prepare(&mut ctx).unwrap();
        let entries = table.entries();
        let format_of = |port: &str| {
            let found = entries.iter().find(|e| e.0 == "ao.gtao" && e.1 == port);
            found.map(|e| e.2.format)
        };
        assert_eq!(format_of("ao"), Some(wgpu::TextureFormat::Rgba16Float));
        assert_eq!(format_of("raw"), Some(wgpu::TextureFormat::R32Float));
        assert_eq!(format_of("blur"), Some(wgpu::TextureFormat::R32Float));
    }
}
