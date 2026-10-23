//! `ContactFeature::install` and the `ContactHandle`: the `shadow.contact` compute pass
//! (S12, order 45) writing a full-resolution Rgba16Float visibility texture (value in .r,
//! 1 = lit) that the lighting reads at S10 group 1 binding 8 (`set_contact`). No slot
//! provider is added: the csm `Shadow` provider of raa-shadow multiplies binding 8.
//!
//! The pass and every handle clone share one state (description, camera, frame counter).
//! Until `ContactHandle::set_camera` is called, or while the pass is disabled, the lighting
//! keeps the dummy (`clear_contact`, visibility 1).

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::{GpuContext, RaaError};
use raa_lit::LitRenderer;
use raa_math::{Mat4, Vec3, Vec4};
use raa_pipeline::{FeaturePass, LightingResources, Pipeline, PrepareCtx, RecordCtx, Targets};

use crate::desc::ContactDesc;
use crate::shaders::{CONTACT_WGSL, PARAMS_SIZE};

/// S12 name of the pass.
const NAME: &str = "shadow.contact";
/// S12 order of the pass (after `ao.gtao`, before `forward`).
const ORDER: u32 = 45;
/// Workgroup edge of the `contact` entry (8x8).
const WORKGROUP: u32 = 8;
/// `PARAMS_SIZE` as an array length.
const PARAMS_LEN: usize = PARAMS_SIZE as usize;
/// Byte offsets of the `ContactParams` fields.
const OFF_PROJ: usize = 0;
const OFF_INV_PROJ: usize = 64;
const OFF_VIEWPORT: usize = 128;
const OFF_SUN_VIEW: usize = 144;
const OFF_PARAMS: usize = 160;
/// The frame index given to the jitter wraps here so it stays exact as an f32.
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

/// `pipeline.busy` for the shared contact state (only reachable by a re-entrant call).
fn state_busy() -> RaaError {
    refusal(
        "pipeline.busy",
        "the contact shadow state is borrowed (re-entrant call while a frame executes)",
    )
}

/// The camera of the frame as the pass needs it.
#[derive(Copy, Clone, Debug, PartialEq)]
struct ContactCamera {
    proj: Mat4,
    inv_proj: Mat4,
    /// Unit view-space direction toward the sun.
    sun_view: Vec3,
}

/// State shared by the pass and every clone of a `ContactHandle`.
struct ContactState {
    desc: ContactDesc,
    /// None until the first `set_camera`.
    camera: Option<ContactCamera>,
    /// Frames prepared with a camera (jitter seed).
    frame: u64,
}

type SharedContact = Rc<RefCell<ContactState>>;

/// Unit view-space direction toward the sun: normalize(-(view * (sun_dir, 0)).xyz), where
/// `sun_dir` is the world travel direction of the light. Zero for a zero direction.
fn sun_view_dir(view: &Mat4, sun_dir: Vec3) -> Vec3 {
    (-view.mul_vec4(Vec4::from_vec3(sun_dir, 0.0)).xyz()).normalize_or_zero()
}

fn write_vec4(out: &mut [u8], v: [f32; 4]) {
    for (chunk, x) in out.chunks_exact_mut(4).zip(v) {
        chunk.copy_from_slice(&x.to_le_bytes());
    }
}

/// Packs `ContactParams`: proj @0, inv_proj @64, viewport (w, h, 1/w, 1/h) @128, sun_view
/// (xyz, 0) @144, params (distance_m, thickness_m, intensity, frame) @160.
fn pack_params(
    out: &mut [u8; PARAMS_LEN],
    camera: &ContactCamera,
    size: (u32, u32),
    desc: &ContactDesc,
    frame: u64,
) {
    let (w, h) = (size.0.max(1) as f32, size.1.max(1) as f32);
    camera.proj.write_le_bytes(&mut out[OFF_PROJ..OFF_INV_PROJ]);
    camera
        .inv_proj
        .write_le_bytes(&mut out[OFF_INV_PROJ..OFF_VIEWPORT]);
    write_vec4(
        &mut out[OFF_VIEWPORT..OFF_SUN_VIEW],
        [w, h, 1.0 / w, 1.0 / h],
    );
    write_vec4(
        &mut out[OFF_SUN_VIEW..OFF_PARAMS],
        Vec4::from_vec3(camera.sun_view, 0.0).to_array(),
    );
    write_vec4(
        &mut out[OFF_PARAMS..PARAMS_LEN],
        [
            desc.distance_m,
            desc.thickness_m,
            desc.intensity,
            (frame % FRAME_WRAP) as f32,
        ],
    );
}

/// The surface-sized output and the bind group over it.
struct Output {
    size: (u32, u32),
    /// Kept alive for the views below.
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    group: wgpu::BindGroup,
}

/// The `shadow.contact` compute pass.
struct ContactPass {
    state: SharedContact,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    bytes: [u8; PARAMS_LEN],
    output: Option<Output>,
    /// The output view must be (re)given to the lighting at the next prepare with a camera.
    needs_bind: bool,
    /// The lighting holds the output view (false: it holds the dummy).
    bound: bool,
}

impl ContactPass {
    fn new(gpu: &GpuContext, state: SharedContact) -> Self {
        let device = gpu.device();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raa-contact.contact"),
            source: wgpu::ShaderSource::Wgsl(CONTACT_WGSL.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("raa-contact.contact"),
            layout: None,
            module: &module,
            entry_point: Some("contact"),
            compilation_options: Default::default(),
            cache: None,
        });
        let layout = pipeline.get_bind_group_layout(0);
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-contact.params"),
            size: PARAMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ContactPass {
            state,
            pipeline,
            layout,
            uniform,
            bytes: [0; PARAMS_LEN],
            output: None,
            needs_bind: true,
            bound: false,
        }
    }
}

impl FeaturePass for ContactPass {
    fn name(&self) -> &str {
        NAME
    }

    fn order(&self) -> u32 {
        ORDER
    }

    /// Recreates the full-resolution Rgba16Float output (R16Float is not a core storage
    /// format) and its bind group over the new depth target; the lighting gets the new view
    /// at the next prepare.
    fn resize(&mut self, gpu: &GpuContext, targets: &Targets) {
        let device = gpu.device();
        let (w, h) = targets.size();
        let size = (w.max(1), h.max(1));
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("raa-contact.output"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-contact.bind_group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(targets.depth()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        });
        self.output = Some(Output {
            size,
            _texture: texture,
            view,
            group,
        });
        self.needs_bind = true;
    }

    /// Disabling resets binding 8 to the dummy (visibility 1); enabling gives the output
    /// back to the lighting at the next prepare.
    fn set_enabled(&mut self, enabled: bool, lighting: &mut LightingResources) {
        if enabled {
            self.needs_bind = true;
        } else {
            lighting.clear_contact();
            self.bound = false;
        }
    }

    /// Without a camera or an output: `clear_contact` (once, so the lighting bind group is
    /// not rebuilt every frame while it already holds the dummy). Otherwise writes
    /// `ContactParams` and, when needed, sets the output view on the lighting.
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let mut st = self.state.try_borrow_mut().map_err(|_| state_busy())?;
        let (Some(camera), Some(output)) = (st.camera, self.output.as_ref()) else {
            if self.bound {
                ctx.lighting.clear_contact();
                self.bound = false;
            }
            self.needs_bind = true;
            return Ok(());
        };
        pack_params(&mut self.bytes, &camera, output.size, &st.desc, st.frame);
        st.frame = st.frame.wrapping_add(1);
        ctx.gpu.queue().write_buffer(&self.uniform, 0, &self.bytes);
        if self.needs_bind {
            ctx.lighting.set_contact(output.view.clone());
            self.needs_bind = false;
            self.bound = true;
        }
        Ok(())
    }

    /// One compute pass of ceil(w / 8) x ceil(h / 8) workgroups (nothing without a camera).
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let has_camera = self
            .state
            .try_borrow()
            .map_err(|_| state_busy())?
            .camera
            .is_some();
        let Some(output) = self.output.as_ref().filter(|_| has_camera) else {
            return Ok(());
        };
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("raa-contact.contact"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &output.group, &[]);
        pass.dispatch_workgroups(
            output.size.0.div_ceil(WORKGROUP),
            output.size.1.div_ceil(WORKGROUP),
            1,
        );
        Ok(())
    }
}

/// Installs screen-space contact shadows on a pipeline.
pub struct ContactFeature;

impl ContactFeature {
    /// Builds the `contact` compute pipeline (layout derived from the module), its 176 B
    /// uniform, and adds the FeaturePass `shadow.contact` (order 45) to `pipeline`. Its output
    /// reaches the lighting at binding 8 once a camera is set; `lit` gets no slot provider
    /// (raa-shadow's `Shadow` provider multiplies binding 8).
    ///
    /// Refusals (nothing installed): `pipeline.busy` when the pipeline is borrowed, the
    /// `Pipeline::add` refusal with its code (`pipeline.duplicate_pass` when installed twice).
    pub fn install(
        gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
        _lit: &mut LitRenderer,
    ) -> Result<ContactHandle, RaaError> {
        let mut p = pipeline.try_borrow_mut().map_err(|_| pipeline_busy())?;
        let state = Rc::new(RefCell::new(ContactState {
            desc: ContactDesc::default(),
            camera: None,
            frame: 0,
        }));
        p.add(Box::new(ContactPass::new(gpu, state.clone())))?;
        Ok(ContactHandle { state })
    }
}

/// Access to the installed contact shadows (clones share one state).
#[derive(Clone)]
pub struct ContactHandle {
    state: SharedContact,
}

impl ContactHandle {
    /// Replaces the description after `ContactDesc::validate` (`shadow.invalid`; the stored
    /// description is unchanged on refusal). `pipeline.busy` on a re-entrant call.
    pub fn set(&self, d: ContactDesc) -> Result<(), RaaError> {
        d.validate()?;
        self.state.try_borrow_mut().map_err(|_| state_busy())?.desc = d;
        Ok(())
    }

    /// The current description (CONTACT defaults until set).
    pub fn get(&self) -> ContactDesc {
        self.state
            .try_borrow()
            .map(|st| st.desc)
            .unwrap_or_default()
    }

    /// The frame's camera (called every frame): `view` and `proj` of the scene camera,
    /// `inv_proj` the inverse of `proj`, and `sun_dir` the world travel direction of the sun.
    /// The pass marches toward normalize(-(view * (sun_dir, 0)).xyz). Ignored on a re-entrant
    /// call.
    pub fn set_camera(&self, view: &Mat4, proj: &Mat4, inv_proj: &Mat4, sun_dir: Vec3) {
        if let Ok(mut st) = self.state.try_borrow_mut() {
            st.camera = Some(ContactCamera {
                proj: *proj,
                inv_proj: *inv_proj,
                sun_view: sun_view_dir(view, sun_dir),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_at(bytes: &[u8], offset: usize) -> f32 {
        f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    }

    #[test]
    fn straight_down_sun_with_identity_view_points_up() {
        let s = sun_view_dir(&Mat4::IDENTITY, Vec3::new(0.0, -1.0, 0.0));
        assert_eq!(s, Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn pack_params_puts_sun_view_at_144_and_params_at_160() {
        let camera = ContactCamera {
            proj: Mat4::from_scale(Vec3::new(2.0, 3.0, 4.0)),
            inv_proj: Mat4::from_scale(Vec3::new(0.5, 0.25, 0.125)),
            sun_view: sun_view_dir(&Mat4::IDENTITY, Vec3::new(0.0, -1.0, 0.0)),
        };
        let desc = ContactDesc {
            distance_m: 0.5,
            thickness_m: 0.1,
            intensity: 0.75,
        };
        let mut out = [0u8; PARAMS_LEN];
        pack_params(&mut out, &camera, (1280, 720), &desc, 7);
        assert_eq!(out.len() as u64, PARAMS_SIZE);
        assert_eq!(f32_at(&out, 0), 2.0);
        assert_eq!(f32_at(&out, 64), 0.5);
        let at = |o: usize| [0, 4, 8, 12].map(|i| f32_at(&out, o + i));
        assert_eq!(at(128), [1280.0, 720.0, 1.0 / 1280.0, 1.0 / 720.0]);
        assert_eq!(at(144), [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(at(160), [0.5, 0.1, 0.75, 7.0]);
    }
}
