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
use raa_pipeline::{
    FeaturePass, ImageDesc, ImageDim, LightingResources, Pipeline, PrepareCtx, RecordCtx, Targets,
};

use crate::desc::ContactDesc;
use crate::shaders::{CONTACT_WGSL, PARAMS_SIZE};
use crate::tunables::TUNABLES;

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

/// `shadow.invalid` (the code of `ContactDesc::validate`) for a refused tunable.
fn invalid(message: String) -> RaaError {
    refusal("shadow.invalid", &message)
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
    /// Tunable `steps` (override STEPS of contact.wgsl).
    steps: u32,
    /// Incremented by `set_tunable` when a value changed.
    tunable_generation: u64,
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
    /// The depth view `group` was built from (the bound input or the target).
    depth: wgpu::TextureView,
}

/// The bind group of the pass: bound depth, params uniform and the storage output.
fn output_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform: &wgpu::Buffer,
    depth: &wgpu::TextureView,
    view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("raa-contact.bind_group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(depth),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(view),
            },
        ],
    })
}

/// The `shadow.contact` compute pass, or one of its instances `shadow.contact#<id>`.
struct ContactPass {
    /// Entry name the pass publishes and resolves its depth input under.
    name: String,
    /// An instance never writes or clears group-1 binding 8.
    instance: bool,
    state: SharedContact,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    /// `ContactState::tunable_generation` the pipeline was built with.
    built_generation: u64,
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
        Self::named(gpu, state, NAME.to_owned(), false)
    }

    fn named(gpu: &GpuContext, state: SharedContact, name: String, instance: bool) -> Self {
        let device = gpu.device();
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa-contact.layout"),
            entries: &[
                entry(
                    0,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                entry(
                    1,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(PARAMS_SIZE),
                    },
                ),
                entry(
                    2,
                    wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                ),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raa-contact.pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let (steps, built_generation) = state
            .try_borrow()
            .map(|st| (st.steps, st.tunable_generation))
            .unwrap_or((DEFAULT_STEPS, 0));
        let pipeline = build_pipeline(gpu, &pipeline_layout, steps);
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-contact.params"),
            size: PARAMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ContactPass {
            name,
            instance,
            state,
            pipeline,
            layout,
            pipeline_layout,
            built_generation,
            uniform,
            bytes: [0; PARAMS_LEN],
            output: None,
            needs_bind: true,
            bound: false,
        }
    }
}

/// Default of the `steps` tunable (contact.wgsl `override STEPS`).
const DEFAULT_STEPS: u32 = 24;

/// The `contact` compute pipeline with the override `STEPS` = `steps`.
fn build_pipeline(
    gpu: &GpuContext,
    layout: &wgpu::PipelineLayout,
    steps: u32,
) -> wgpu::ComputePipeline {
    let device = gpu.device();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("raa-contact.contact"),
        source: wgpu::ShaderSource::Wgsl(CONTACT_WGSL.into()),
    });
    let constants = [("STEPS", f64::from(steps))];
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("raa-contact.contact"),
        layout: Some(layout),
        module: &module,
        entry_point: Some("contact"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &constants,
            ..Default::default()
        },
        cache: None,
    })
}

impl FeaturePass for ContactPass {
    fn name(&self) -> &str {
        &self.name
    }

    fn order(&self) -> u32 {
        ORDER
    }

    /// Drops the visibility texture and its group; `resize` recreates them.
    fn release(&mut self, _gpu: &GpuContext) {
        self.output = None;
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
        let depth = targets.depth().clone();
        let group = output_group(device, &self.layout, &self.uniform, &depth, &view);
        self.output = Some(Output {
            size,
            _texture: texture,
            view,
            group,
            depth,
        });
        self.needs_bind = true;
    }

    /// Disabling resets binding 8 to the dummy (visibility 1); enabling gives the output
    /// back to the lighting at the next prepare. An instance never touches binding 8.
    fn set_enabled(&mut self, enabled: bool, lighting: &mut LightingResources) {
        if enabled {
            self.needs_bind = true;
        } else if !self.instance {
            lighting.clear_contact();
            self.bound = false;
        }
    }

    /// Without a camera or an output: `clear_contact` (once, so the lighting bind group is
    /// not rebuilt every frame while it already holds the dummy). Otherwise writes
    /// `ContactParams` and, when needed, sets the output view on the lighting.
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let mut st = self.state.try_borrow_mut().map_err(|_| state_busy())?;
        if st.tunable_generation != self.built_generation {
            self.pipeline = build_pipeline(ctx.gpu, &self.pipeline_layout, st.steps);
            self.built_generation = st.tunable_generation;
        }
        let depth = ctx.targets.depth_input(ctx.resources, &self.name);
        if let Some(o) = self.output.as_mut() {
            if o.depth != depth {
                o.group = output_group(
                    ctx.gpu.device(),
                    &self.layout,
                    &self.uniform,
                    &depth,
                    &o.view,
                );
                o.depth = depth;
            }
            let desc = ImageDesc::new(
                ImageDim::D2,
                wgpu::TextureFormat::Rgba16Float,
                o.size.0,
                o.size.1,
                1,
                1,
            );
            ctx.resources.publish(&self.name, "contact", &o.view, desc);
            ctx.resources.publish(&self.name, "output", &o.view, desc);
        }
        let (Some(camera), Some(output)) = (st.camera, self.output.as_ref()) else {
            if !self.instance && self.bound {
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
            if !self.instance {
                ctx.lighting.set_contact(output.view.clone());
                self.bound = true;
            }
            self.needs_bind = false;
        }
        Ok(())
    }

    /// One compute pass of ceil(w / 8) x ceil(h / 8) workgroups (nothing without a camera).
    /// The primary pass (binding 8) records nothing on a G frame of forward
    /// "layers-interlaced" (`LightingInterlace::skip_contact`): its texture stays bound and
    /// published with the previous R frame's contents, which transparent OIT, fur shells and
    /// opaque particles read on that frame (one frame old in screen space).
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if !self.instance && ctx.lighting.interlace().skip_contact {
            return Ok(());
        }
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
            steps: DEFAULT_STEPS,
            tunable_generation: 0,
        }));
        p.add(Box::new(ContactPass::new(gpu, state.clone())))?;
        Ok(ContactHandle { state })
    }

    /// Handoff: card 'Slots 3/3: gtao and contact instance passes in the slots extension'
    /// calls `ContactFeature::install_instance` to add `shadow.contact#<instance>` (order 45)
    /// with its own `ContactState` (the default description, no camera, frame 0, the default
    /// steps and generation 0) and gets a `ContactHandle` whose clones share that state;
    /// `ContactHandle::copy_camera_from` copies the camera of another handle into it. Today
    /// the instance never calls `LightingResources::set_contact` or `clear_contact`, publishes
    /// `contact` and `output` under its own name, and samples depth through
    /// `Targets::depth_input` with that name. Refusals, with nothing installed: `pipeline.busy`
    /// and `pipeline.duplicate_pass` naming the pass.
    pub fn install_instance(
        gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
        instance: &str,
    ) -> Result<ContactHandle, RaaError> {
        let mut p = pipeline.try_borrow_mut().map_err(|_| pipeline_busy())?;
        let name = format!("{NAME}#{instance}");
        if p.passes().iter().any(|info| info.name == name) {
            return Err(refusal(
                "pipeline.duplicate_pass",
                &format!("the pass `{name}` is already installed"),
            ));
        }
        let state = Rc::new(RefCell::new(ContactState {
            desc: ContactDesc::default(),
            camera: None,
            frame: 0,
            steps: DEFAULT_STEPS,
            tunable_generation: 0,
        }));
        p.add(Box::new(ContactPass::named(gpu, state.clone(), name, true)))?;
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

    /// Handoff: card 'Executor extension: apply Tunable bindings' calls
    /// `ContactHandle::set_tunable` through `HandleSet.contact` (crates/raa-graph/src/exec.rs)
    /// with a catalog key and its value as f64. Today it validates with
    /// `raa_pipeline::validate_tunable` against `crate::tunables()`, refuses with
    /// `shadow.invalid` without changing anything, and otherwise stores the value (integral
    /// for `steps`) in the state shared with the pass, bumping a generation when it changed so
    /// the pass rebuilds its pipeline at its next prepare. `pipeline.busy` on a re-entrant call.
    pub fn set_tunable(&self, key: &str, value: f64) -> Result<(), RaaError> {
        let spec = raa_pipeline::validate_tunable(TUNABLES, key, value).map_err(invalid)?;
        let mut st = self.state.try_borrow_mut().map_err(|_| state_busy())?;
        if spec.key == "steps" {
            let Ok(steps) = u32::try_from(value as i64) else {
                return Err(invalid(format!("`steps` out of u32 range, got {value}")));
            };
            if st.steps != steps {
                st.steps = steps;
                st.tunable_generation = st.tunable_generation.wrapping_add(1);
            }
        }
        Ok(())
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

    /// Copies `source`'s camera Option (projection, inverse and view-space sun direction)
    /// into this handle. Does nothing when both share one state or on a re-entrant borrow.
    pub fn copy_camera_from(&self, source: &ContactHandle) {
        if Rc::ptr_eq(&self.state, &source.state) {
            return;
        }
        let Ok(camera) = source.state.try_borrow().map(|st| st.camera) else {
            return;
        };
        if let Ok(mut st) = self.state.try_borrow_mut() {
            st.camera = camera;
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

    fn handle() -> ContactHandle {
        ContactHandle {
            state: Rc::new(RefCell::new(ContactState {
                desc: ContactDesc::default(),
                camera: None,
                frame: 0,
                steps: DEFAULT_STEPS,
                tunable_generation: 0,
            })),
        }
    }

    #[test]
    fn set_tunable_refusals_change_nothing() {
        let h = handle();
        for (key, value) in [("nope", 8.0), ("steps", f64::NAN), ("steps", 8.5)] {
            let e = h.set_tunable(key, value).unwrap_err();
            assert_eq!(e.code(), "shadow.invalid");
        }
        let st = h.state.borrow();
        assert_eq!((st.steps, st.tunable_generation), (24, 0));
    }

    #[test]
    fn set_tunable_stores_and_bumps_generation_once() {
        let h = handle();
        h.set_tunable("steps", 12.0).unwrap();
        h.set_tunable("steps", 12.0).unwrap();
        let st = h.state.borrow();
        assert_eq!((st.steps, st.tunable_generation), (12, 1));
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
