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
use raa_pipeline::{FeaturePass, LightingResources, Pipeline, PrepareCtx, RecordCtx, Targets};

use crate::desc::AoDesc;
use crate::shaders::{DENOISE_WGSL, GTAO_WGSL, PARAMS_SIZE, ao_provider};

/// S12 name of the pass.
const NAME: &str = "ao.gtao";
/// S12 order of the pass (after `prepass`, before `shadow.contact`).
const ORDER: u32 = 40;
/// Workgroup edge of the three entries (8x8).
const WORKGROUP: u32 = 8;
/// `PARAMS_SIZE` as an array length.
const PARAMS_LEN: usize = PARAMS_SIZE as usize;
/// Byte offsets of the `GtaoParams` fields.
const OFF_PROJ: usize = 0;
const OFF_INV_PROJ: usize = 64;
const OFF_VIEWPORT: usize = 128;
const OFF_PARAMS: usize = 144;
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
struct GtaoCamera {
    proj: Mat4,
    inv_proj: Mat4,
}

/// State shared by the pass and every clone of a `GtaoHandle`.
struct GtaoState {
    desc: AoDesc,
    /// None until the first `set_camera`.
    camera: Option<GtaoCamera>,
}

type SharedGtao = Rc<RefCell<GtaoState>>;

fn write_vec4(out: &mut [u8], v: [f32; 4]) {
    for (chunk, x) in out.chunks_exact_mut(4).zip(v) {
        chunk.copy_from_slice(&x.to_le_bytes());
    }
}

/// Packs `GtaoParams`: proj @0, inv_proj @64, viewport (w, h, 1/w, 1/h) @128, params
/// (radius_m, intensity, power, frame) @144.
fn pack_params(
    out: &mut [u8; PARAMS_LEN],
    camera: &GtaoCamera,
    size: (u32, u32),
    desc: &AoDesc,
    frame: u64,
) {
    let (w, h) = (size.0.max(1) as f32, size.1.max(1) as f32);
    camera.proj.write_le_bytes(&mut out[OFF_PROJ..OFF_INV_PROJ]);
    camera
        .inv_proj
        .write_le_bytes(&mut out[OFF_INV_PROJ..OFF_VIEWPORT]);
    write_vec4(&mut out[OFF_VIEWPORT..OFF_PARAMS], [w, h, 1.0 / w, 1.0 / h]);
    write_vec4(
        &mut out[OFF_PARAMS..PARAMS_LEN],
        [
            desc.radius_m,
            desc.intensity,
            desc.power,
            (frame % FRAME_WRAP) as f32,
        ],
    );
}

/// The surface-sized textures and the bind group of each dispatch over them.
struct Textures {
    size: (u32, u32),
    /// Kept alive for the views bound below.
    _raw: wgpu::Texture,
    _blur: wgpu::Texture,
    _out: wgpu::Texture,
    out_view: wgpu::TextureView,
    gtao_group: wgpu::BindGroup,
    denoise_h_group: wgpu::BindGroup,
    denoise_v_group: wgpu::BindGroup,
}

/// One compute pipeline and its derived group-0 layout.
struct Kernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

impl Kernel {
    fn new(device: &wgpu::Device, module: &wgpu::ShaderModule, entry: &str) -> Self {
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

fn create_texture(
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
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn view_entry(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

fn buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// The `ao.gtao` compute pass.
struct GtaoPass {
    state: SharedGtao,
    gtao: Kernel,
    denoise_h: Kernel,
    denoise_v: Kernel,
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
        let device = gpu.device();
        let gtao_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa-gtao.gtao"),
            source: wgpu::ShaderSource::Wgsl(GTAO_WGSL.into()),
        });
        let denoise_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa-gtao.denoise"),
            source: wgpu::ShaderSource::Wgsl(DENOISE_WGSL.into()),
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-gtao.params"),
            size: PARAMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        GtaoPass {
            state,
            gtao: Kernel::new(device, &gtao_module, "gtao"),
            denoise_h: Kernel::new(device, &denoise_module, "denoise_h"),
            denoise_v: Kernel::new(device, &denoise_module, "denoise_v"),
            uniform,
            bytes: [0; PARAMS_LEN],
            textures: None,
            needs_bind: true,
            bound: false,
        }
    }
}

impl FeaturePass for GtaoPass {
    fn name(&self) -> &str {
        NAME
    }

    fn order(&self) -> u32 {
        ORDER
    }

    /// Recreates the full-resolution textures (raw and blur R32Float, out Rgba16Float since
    /// R16Float is not a core storage format) and the bind groups of the three dispatches
    /// over the new depth and normal targets; the lighting gets the new view at the next
    /// prepare. Each bind group lists only the bindings its entry uses (derived layouts).
    fn resize(&mut self, gpu: &GpuContext, targets: &Targets) {
        let device = gpu.device();
        let (w, h) = targets.size();
        let size = (w.max(1), h.max(1));
        let (raw, raw_view) =
            create_texture(device, "raa-gtao.raw", size, wgpu::TextureFormat::R32Float);
        let (blur, blur_view) =
            create_texture(device, "raa-gtao.blur", size, wgpu::TextureFormat::R32Float);
        let (out, out_view) = create_texture(
            device,
            "raa-gtao.out",
            size,
            wgpu::TextureFormat::Rgba16Float,
        );
        let gtao_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-gtao.gtao"),
            layout: &self.gtao.layout,
            entries: &[
                view_entry(0, targets.depth()),
                view_entry(1, targets.normal()),
                buffer_entry(2, &self.uniform),
                view_entry(3, &raw_view),
            ],
        });
        let denoise_h_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-gtao.denoise_h"),
            layout: &self.denoise_h.layout,
            entries: &[
                view_entry(0, &raw_view),
                view_entry(1, targets.depth()),
                view_entry(2, &blur_view),
                buffer_entry(4, &self.uniform),
            ],
        });
        let denoise_v_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-gtao.denoise_v"),
            layout: &self.denoise_v.layout,
            entries: &[
                view_entry(0, &blur_view),
                view_entry(1, targets.depth()),
                view_entry(3, &out_view),
                buffer_entry(4, &self.uniform),
            ],
        });
        self.textures = Some(Textures {
            size,
            _raw: raw,
            _blur: blur,
            _out: out,
            out_view,
            gtao_group,
            denoise_h_group,
            denoise_v_group,
        });
        self.needs_bind = true;
    }

    /// Disabling resets binding 7 to the dummy (AO 1); enabling gives the output back to
    /// the lighting at the next prepare.
    fn set_enabled(&mut self, enabled: bool, lighting: &mut LightingResources) {
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
        let st = self.state.try_borrow().map_err(|_| state_busy())?;
        let (Some(camera), Some(textures)) = (st.camera, self.textures.as_ref()) else {
            if self.bound {
                ctx.lighting.clear_ao();
                self.bound = false;
            }
            self.needs_bind = true;
            return Ok(());
        };
        pack_params(&mut self.bytes, &camera, ctx.size, &st.desc, ctx.frame);
        ctx.gpu.queue().write_buffer(&self.uniform, 0, &self.bytes);
        if self.needs_bind {
            ctx.lighting.set_ao(textures.out_view.clone());
            self.needs_bind = false;
            self.bound = true;
        }
        Ok(())
    }

    /// One compute pass: gtao -> raw, denoise_h raw -> blur, denoise_v blur -> out, each
    /// ceil(w / 8) x ceil(h / 8) workgroups (nothing without a camera).
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let has_camera = self
            .state
            .try_borrow()
            .map_err(|_| state_busy())?
            .camera
            .is_some();
        let Some(t) = self.textures.as_ref().filter(|_| has_camera) else {
            return Ok(());
        };
        let groups = (t.size.0.div_ceil(WORKGROUP), t.size.1.div_ceil(WORKGROUP));
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
            camera: None,
        }));
        p.add(Box::new(GtaoPass::new(gpu, state.clone())))?;
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
        pack_params(&mut out, &camera, (1280, 720), &desc, 7);
        assert_eq!(out.len() as u64, PARAMS_SIZE);
        assert_eq!(f32_at(&out, 0), 2.0);
        assert_eq!(f32_at(&out, 64), 0.5);
        let at = |o: usize| [0, 4, 8, 12].map(|i| f32_at(&out, o + i));
        assert_eq!(at(128), [1280.0, 720.0, 1.0 / 1280.0, 1.0 / 720.0]);
        assert_eq!(at(144), [0.5, 0.75, 2.0, 7.0]);
    }

    #[test]
    fn handle_set_refuses_invalid_and_keeps_desc() {
        let handle = GtaoHandle {
            state: Rc::new(RefCell::new(GtaoState {
                desc: AoDesc::default(),
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
    }
}
