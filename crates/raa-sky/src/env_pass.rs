//! The `sky.env` feature pass (S12, order 10): renders the SKY_ENV source cube from the
//! common `raa_sky_radiance` seam (one fullscreen triangle per face) and hands it to the IBL
//! filter, when the sky changed or every frame per `EnvUpdate`. Its prepare drives the IBL
//! readback (`IblHandle::apply_pending`) and resets the background mode to constant; the
//! `sky.background` contributor sets it back to sky later in the same frame when enabled.
//!
//! The legacy model renders [`env_module_wgsl`]; the physical model renders
//! [`physical_env_module_wgsl`], which attaches the completed atmosphere LUTs, celestial
//! evaluator and cloud views through the private S10 group-2 layout once those resources
//! were carried by real submissions. Model and accepted source changes invalidate the IBL
//! publication before its readback so no old callback can restore the previous environment,
//! and disabling the pass invalidates it synchronously.
//!
//! The lighting's `env_specular` is the IBL output cube, never this source cube, so the env
//! render passes bind group 1 without a read/write hazard.

use raa_core::{GpuContext, RaaError};
use raa_ibl::IblHandle;
use raa_lit::LitHandle;
use raa_pipeline::{FeaturePass, HDR_FORMAT, LightingResources, PrepareCtx, RecordCtx};

use crate::atmosphere::{AtmosphereResources, PhysicalSkyGpu, SharedAtmosphere};
use crate::cloud_backend::{CloudViews, SharedCloud};
use crate::desc::{EnvUpdate, SkyModel};
use crate::shaders::{env_module_wgsl, physical_env_module_wgsl};
use crate::state::{SharedSky, SkyState};

/// S12 name of the pass.
pub(crate) const ENV_NAME: &str = "sky.env";
/// S12 order of the pass.
pub(crate) const ENV_ORDER: u32 = 10;
/// SKY_ENV: face size in pixels.
const ENV_SIZE: u32 = 256;
/// SKY_ENV: mip count (256..8).
const ENV_MIPS: u32 = 6;
/// S10 `PhysicalSkyGpu` size in bytes.
const PHYSICAL_SKY_BYTES: usize = 256;

/// The physical-sky group-2 resources both consumers bind: the private uniform, the two
/// atmosphere LUTs, the cloud panorama/screen views, the Milky Way map and the filtering
/// samplers. The bind group is rebuilt only when the view or a validity flag changes.
pub(crate) struct PhysicalSkyBinding {
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    bytes: Option<[u8; PHYSICAL_SKY_BYTES]>,
    key: Option<PhysicalKey>,
    group: Option<wgpu::BindGroup>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PhysicalKey {
    view: u32,
    size: (u32, u32),
    screen_valid: bool,
    panorama_valid: bool,
}

/// The accepted frame snapshot a physical bind-group prepare reads.
pub(crate) struct PhysicalFrame<'a> {
    pub(crate) state: &'a SkyState,
    pub(crate) view: u32,
    pub(crate) size: (u32, u32),
    pub(crate) time_s: f32,
    pub(crate) atmosphere: &'a AtmosphereResources,
    pub(crate) views: &'a CloudViews,
}

/// Whether the physical group-2 consumers may sample the atmosphere resources.
///
/// An allocated sky-view texture is only binding validity: both consumers stay inactive
/// until the view's LUT was carried by an actual submission (its acknowledged generation is
/// non-zero) and the one-time Milky Way bake was submitted too.
fn physical_inputs_ready(state: &SkyState, view: u32, atmosphere: &AtmosphereResources) -> bool {
    let submitted_generation = state
        .views
        .get(view as usize)
        .and_then(Option::as_ref)
        .map_or(0, |slot| slot.sky_view_generation);
    physical_readiness(
        atmosphere.sky_view(view).is_some(),
        submitted_generation,
        atmosphere.milky_way_ready(),
    )
}

/// The device-free readiness rule [`physical_inputs_ready`] applies.
fn physical_readiness(sky_view: bool, submitted_generation: u64, bake_ready: bool) -> bool {
    sky_view && submitted_generation > 0 && bake_ready
}

/// Whether prepare must cancel in-flight IBL work before it can publish: an accepted source
/// change since the last recorded filter (the record clears `env_dirty`, so seeing it set at
/// prepare entry means an edit replaced the source) or a switch of the evaluated model.
fn source_replaced(env_dirty: bool, prepared: Option<SkyModel>, model: SkyModel) -> bool {
    env_dirty || prepared.is_some_and(|previous| previous != model)
}

impl PhysicalSkyBinding {
    /// Creates the explicit S10 group-2 layout, the repeat-U/clamp-V sampler and the
    /// 256-byte `PhysicalSkyGpu` upload buffer.
    pub(crate) fn new(gpu: &GpuContext) -> Self {
        let device = gpu.device();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raa.sky.physical.layout"),
            entries: &physical_layout_entries(),
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("raa.sky.physical.sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa.sky.physical.uniform"),
            size: PHYSICAL_SKY_BYTES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        PhysicalSkyBinding {
            layout,
            sampler,
            uniform,
            bytes: None,
            key: None,
            group: None,
        }
    }

    pub(crate) fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    /// Uploads the current accepted state's `PhysicalSkyGpu` (with the exact cloud validity
    /// flags) and binds the completed resources. Returns false until the view's sky-view LUT
    /// and the Milky Way bake have been carried by real submissions, so the caller can keep
    /// the legacy evaluator as its explicit fallback; allocation alone is never readiness.
    pub(crate) fn prepare(&mut self, gpu: &GpuContext, frame: &PhysicalFrame<'_>) -> bool {
        let Some(slot) = frame
            .state
            .views
            .get(frame.view as usize)
            .and_then(Option::as_ref)
        else {
            return false;
        };
        if !physical_inputs_ready(frame.state, frame.view, frame.atmosphere) {
            return false;
        }
        let Some(sky_view) = frame.atmosphere.sky_view(frame.view) else {
            return false;
        };
        let mut parameters = PhysicalSkyGpu::from_state(
            frame.state,
            &slot.camera,
            frame.size,
            f64::from(frame.time_s),
        );
        parameters.flags[1] = u32::from(frame.views.screen_valid);
        parameters.flags[2] = u32::from(frame.views.panorama_valid);
        let bytes = parameters.to_bytes();
        if self.bytes != Some(bytes) {
            gpu.queue().write_buffer(&self.uniform, 0, &bytes);
            self.bytes = Some(bytes);
        }
        let key = PhysicalKey {
            view: frame.view,
            size: frame.size,
            screen_valid: frame.views.screen_valid,
            panorama_valid: frame.views.panorama_valid,
        };
        if self.key != Some(key) {
            self.group = Some(gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("raa.sky.physical.bind_group"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            frame.atmosphere.transmittance(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(sky_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&frame.views.panorama),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&frame.views.screen),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(frame.atmosphere.milky_way()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(frame.atmosphere.sampler()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            }));
            self.key = Some(key);
        }
        true
    }

    pub(crate) fn group(&self) -> Option<&wgpu::BindGroup> {
        self.group.as_ref()
    }
}

fn physical_layout_entries() -> [wgpu::BindGroupLayoutEntry; 8] {
    use wgpu::BindingType::{Buffer, Sampler, Texture};
    use wgpu::ShaderStages;
    use wgpu::{BufferBindingType, SamplerBindingType, TextureSampleType, TextureViewDimension};

    let visibility = ShaderStages::FRAGMENT;
    let texture = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: Texture {
            sample_type: TextureSampleType::Float { filterable: true },
            view_dimension: TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility,
            ty: Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: std::num::NonZeroU64::new(PHYSICAL_SKY_BYTES as u64),
            },
            count: None,
        },
        texture(1),
        texture(2),
        texture(3),
        texture(4),
        texture(5),
        wgpu::BindGroupLayoutEntry {
            binding: 6,
            visibility,
            ty: Sampler(SamplerBindingType::Filtering),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 7,
            visibility,
            ty: Sampler(SamplerBindingType::Filtering),
            count: None,
        },
    ]
}

/// Renders the sky env cube and filters it into the IBL.
pub(crate) struct SkyEnvPass {
    state: SharedSky,
    ibl: IblHandle,
    lit: LitHandle,
    atmosphere: SharedAtmosphere,
    cloud: SharedCloud,
    /// The SKY_ENV source cube (mip 0 rendered here, mips 1..5 written by the filter).
    cube: wgpu::Texture,
    /// Mip-0 views of the six faces in wgpu order +X, -X, +Y, -Y, +Z, -Z.
    faces: [wgpu::TextureView; 6],
    legacy_pipeline: wgpu::RenderPipeline,
    physical_pipeline: wgpu::RenderPipeline,
    physical: PhysicalSkyBinding,
    /// Whether the physical group-2 bind group is ready for this frame.
    physical_ready: bool,
    /// The model whose source cube and IBL the current pipeline composition renders.
    prepared_model: Option<SkyModel>,
    /// The Milky Way bake was ready at the last prepare, for the one-time env refresh.
    milky_way_ready: bool,
    /// `IblHandle::filter` takes the lighting mutably to clear the env when it refuses the
    /// source; the record context only shares the frame's lighting, so the filter gets this
    /// never-bound copy and a refusal sets `clear_env` for the next prepare instead.
    scratch: LightingResources,
    /// The filter refused the source cube: clear the frame's env at the next prepare.
    clear_env: bool,
}

/// Creates the SKY_ENV source cube with the usages raa-ibl's filter requires (plus
/// RENDER_ATTACHMENT for the face passes).
fn create_cube(gpu: &GpuContext) -> wgpu::Texture {
    gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("raa.sky.env.cube"),
        size: wgpu::Extent3d {
            width: ENV_SIZE,
            height: ENV_SIZE,
            depth_or_array_layers: 6,
        },
        mip_level_count: ENV_MIPS,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HDR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// Mip-0 2D view of cube face `face`.
fn face_view(cube: &wgpu::Texture, face: u32) -> wgpu::TextureView {
    cube.create_view(&wgpu::TextureViewDescriptor {
        label: Some("raa.sky.env.face"),
        dimension: Some(wgpu::TextureViewDimension::D2),
        base_mip_level: 0,
        mip_level_count: Some(1),
        base_array_layer: face,
        array_layer_count: Some(1),
        ..Default::default()
    })
}

/// Builds an env pipeline: groups [frame, lighting] and, for the physical model, the
/// private physical-sky group 2; no vertex buffers, HDR colour without blending, no depth,
/// single sample.
fn build_pipeline(
    gpu: &GpuContext,
    frame_layout: &wgpu::BindGroupLayout,
    lighting_layout: &wgpu::BindGroupLayout,
    physical_layout: Option<&wgpu::BindGroupLayout>,
    wgsl: String,
) -> wgpu::RenderPipeline {
    let device = gpu.device();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("raa.sky.env"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let groups: Vec<Option<&wgpu::BindGroupLayout>> = match physical_layout {
        Some(physical) => vec![Some(frame_layout), Some(lighting_layout), Some(physical)],
        None => vec![Some(frame_layout), Some(lighting_layout)],
    };
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("raa.sky.env.layout"),
        bind_group_layouts: &groups,
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("raa.sky.env"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn state_busy() -> RaaError {
    RaaError::Resource(format!("{ENV_NAME}: sky state is busy"))
}

impl SkyEnvPass {
    /// Creates the source cube, its face views and both model pipelines over the lit frame
    /// layout and the pipeline's lighting layout (`Pipeline::lighting().layout()`).
    pub(crate) fn new(
        gpu: &GpuContext,
        state: SharedSky,
        ibl: IblHandle,
        lit: LitHandle,
        lighting_layout: &wgpu::BindGroupLayout,
        atmosphere: SharedAtmosphere,
        cloud: SharedCloud,
    ) -> Self {
        let cube = create_cube(gpu);
        let faces = std::array::from_fn(|face| face_view(&cube, face as u32));
        let physical = PhysicalSkyBinding::new(gpu);
        let frame_layout = lit.frame_layout();
        let legacy_pipeline =
            build_pipeline(gpu, &frame_layout, lighting_layout, None, env_module_wgsl());
        let physical_pipeline = build_pipeline(
            gpu,
            &frame_layout,
            lighting_layout,
            Some(physical.layout()),
            physical_env_module_wgsl(),
        );
        SkyEnvPass {
            state,
            ibl,
            lit,
            atmosphere,
            cloud,
            cube,
            faces,
            legacy_pipeline,
            physical_pipeline,
            physical,
            physical_ready: false,
            prepared_model: None,
            milky_way_ready: false,
            scratch: LightingResources::new(gpu),
            clear_env: false,
        }
    }
}

impl FeaturePass for SkyEnvPass {
    fn name(&self) -> &str {
        ENV_NAME
    }

    fn order(&self) -> u32 {
        ENV_ORDER
    }

    /// Disabling invalidates the IBL publication synchronously (old callbacks cannot restore
    /// it), clears the env to flat S1 ambient and sets the background mode to constant;
    /// enabling marks the cube and the `EnvGpu` sky fields dirty so the env is rebuilt.
    fn set_enabled(&mut self, enabled: bool, lighting: &mut LightingResources) {
        if enabled {
            if let Ok(mut state) = self.state.try_borrow_mut() {
                state.env_dirty = true;
                state.gpu_dirty = true;
            }
        } else {
            self.ibl.invalidate(lighting);
            lighting.env_mut().sky_params[2] = 0.0;
            self.clear_env = false;
        }
    }

    /// Writes the `EnvGpu` sky fields when the description changed, resets the background
    /// mode to constant (S12: `sky.background` sets it to sky later in this frame when
    /// enabled), invalidates the IBL publication when the model or the accepted source
    /// changed, applies the IBL readbacks of earlier frames and decides whether the cube is
    /// rendered this frame.
    ///
    /// The physical bind group is prepared from the same accepted state with the exact
    /// `CloudViews` validity flags; both physical consumers stay inactive until the view's
    /// LUT and the Milky Way bake were actually submitted, so allocation never stands in
    /// for submitted readiness.
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        if self.clear_env {
            ctx.lighting.clear_env();
            self.clear_env = false;
        }
        let mut state = self.state.try_borrow_mut().map_err(|_| state_busy())?;
        let model = state.desc.model;
        // `record` clears `env_dirty`, so seeing it set before this prepare's own refresh
        // decisions means an accepted edit replaced the source since the recorded filter.
        // Both a same-model source replacement and a model switch must cancel the older
        // callbacks before `apply_pending` can publish them.
        if source_replaced(state.env_dirty, self.prepared_model, model) {
            self.ibl.invalidate(ctx.lighting);
        }
        self.prepared_model = Some(model);
        let bake_ready = self
            .atmosphere
            .try_borrow()
            .map(|resources| resources.milky_way_ready())
            .unwrap_or(false);
        if bake_ready && !self.milky_way_ready {
            state.env_dirty = true;
        }
        self.milky_way_ready = bake_ready;
        if state.gpu_dirty {
            state.desc.write_env(ctx.lighting.env_mut());
            state.gpu_dirty = false;
        }
        if ctx.lighting.env().sky_params[2] != 0.0 {
            ctx.lighting.env_mut().sky_params[2] = 0.0;
        }
        if self.ibl.apply_pending(ctx.lighting) || state.desc.env_update == EnvUpdate::EveryFrame {
            state.env_dirty = true;
        }
        self.physical_ready = false;
        if model == SkyModel::Physical
            && let Some(view) = state.active_view
        {
            let views = self
                .cloud
                .try_borrow()
                .ok()
                .map(|backend| backend.views(view));
            let atmosphere = self.atmosphere.try_borrow().ok();
            if let (Some(views), Some(atmosphere)) = (views, atmosphere) {
                self.physical_ready = self.physical.prepare(
                    ctx.gpu,
                    &PhysicalFrame {
                        state: &state,
                        view,
                        size: ctx.size,
                        time_s: ctx.time_s,
                        atmosphere: &atmosphere,
                        views: &views,
                    },
                );
            }
        }
        Ok(())
    }

    /// When the env is dirty: renders the six faces of the source cube (clear, then one
    /// fullscreen triangle with `instance_index` = face) and records the IBL filter of it.
    /// The physical model selects its own pipeline and group-2 bind group; a refused source
    /// keeps the explicit clear at the next prepare.
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let model = {
            let mut state = self.state.try_borrow_mut().map_err(|_| state_busy())?;
            if !state.env_dirty {
                return Ok(());
            }
            state.env_dirty = false;
            state.desc.model
        };
        let physical = model == SkyModel::Physical && self.physical_ready;
        let pipeline = if physical {
            &self.physical_pipeline
        } else {
            &self.legacy_pipeline
        };
        let frame = self.lit.frame_bind_group();
        for (face, view) in (0u32..).zip(self.faces.iter()) {
            let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("raa.sky.env.face"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &frame, &[]);
            pass.set_bind_group(1, ctx.lighting.bind_group(), &[]);
            if physical && let Some(group) = self.physical.group() {
                pass.set_bind_group(2, group, &[]);
            }
            pass.draw(0..3, face..face + 1);
        }
        let filtered = self
            .ibl
            .filter(ctx.gpu, ctx.encoder, &self.cube, &mut self.scratch);
        if filtered.is_err() {
            self.clear_env = true;
        }
        filtered
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_order_and_required_follow_s12() {
        // An instance needs a GPU; `gpu_frame_renders_and_filters_the_env` checks the same
        // through `Pipeline::passes` (where `required` is the trait default, false).
        assert_eq!((ENV_NAME, ENV_ORDER), ("sky.env", 10));
    }

    #[test]
    fn physical_readiness_waits_for_actual_submissions() {
        assert!(
            !physical_readiness(true, 0, true),
            "an allocated sky view is not a submitted LUT"
        );
        assert!(
            !physical_readiness(true, 1, false),
            "the Milky Way bake must be submitted too"
        );
        assert!(!physical_readiness(false, 1, true), "no sky-view texture");
        assert!(physical_readiness(true, 1, true));
    }

    #[test]
    fn a_same_model_source_change_invalidates_before_pending_publication() {
        // `record` clears `env_dirty`: an entry-time flag is an accepted source replacement,
        // even while the evaluated model stays the same.
        assert!(source_replaced(
            true,
            Some(SkyModel::Physical),
            SkyModel::Physical
        ));
        assert!(!source_replaced(
            false,
            Some(SkyModel::Physical),
            SkyModel::Physical
        ));
        assert!(source_replaced(
            false,
            Some(SkyModel::Legacy),
            SkyModel::Physical
        ));
        assert!(
            !source_replaced(false, None, SkyModel::Physical),
            "the first prepare has no recorded source; its dirty state decides"
        );
        assert!(
            source_replaced(true, None, SkyModel::Physical),
            "a dirty first prepare still cancels stale work"
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn physical_consumers_wait_for_submitted_luts_and_the_bake() {
        use raa_core::testing::{skip, software_gpu};

        let test = "physical_consumers_wait_for_submitted_luts_and_the_bake";
        let Some(gpu) = software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let mut resources = AtmosphereResources::new(&gpu).expect("atmosphere");
        let parameters = PhysicalSkyGpu::default();
        resources
            .prepare_view(&gpu, 0, (64, 64), &parameters)
            .expect("prepare view");
        assert!(resources.sky_view(0).is_some(), "the texture is allocated");
        let mut state = SkyState::default();
        state.views[0] = Some(crate::state::ViewSlot::new(
            0,
            raa_scene::Camera::default(),
            false,
        ));
        assert!(
            !physical_inputs_ready(&state, 0, &resources),
            "allocation and an unprepared bake are not readiness"
        );
        let destination = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa.sky.test.transmission"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let descriptor = wgpu::CommandEncoderDescriptor {
            label: Some("raa.sky.test.atmosphere"),
        };
        let mut encoder = gpu.device().create_command_encoder(&descriptor);
        resources
            .record_milky_way(&gpu, &mut encoder, 0)
            .expect("record bake");
        resources
            .record(&gpu, &mut encoder, 0, &destination)
            .expect("record luts");
        assert!(
            !physical_inputs_ready(&state, 0, &resources),
            "recording alone is pending"
        );
        gpu.queue().submit([encoder.finish()]);
        resources.observe_submission(0, 0b0000, false);
        assert!(
            !physical_inputs_ready(&state, 0, &resources),
            "an empty/failed submission acknowledges nothing"
        );
        // A released recording is re-recorded by the next invocation, exactly like the pass.
        let mut encoder = gpu.device().create_command_encoder(&descriptor);
        resources
            .record_milky_way(&gpu, &mut encoder, 0)
            .expect("re-record bake");
        gpu.queue().submit([encoder.finish()]);
        resources.observe_submission(0, 0b0001, true);
        assert!(resources.milky_way_ready(), "the submitted bake is ready");
        state.views[0]
            .as_mut()
            .expect("view slot")
            .sky_view_generation = 1;
        assert!(
            physical_inputs_ready(&state, 0, &resources),
            "a submitted sky-view LUT and bake activate the consumers"
        );
        std::mem::forget(resources);
        std::mem::forget(gpu);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn gpu_frame_renders_and_filters_the_env() {
        use std::cell::RefCell;
        use std::rc::Rc;

        use raa_core::testing::{block_on, skip, software_gpu};
        use raa_core::{FrameGraph, OffscreenTarget};
        use raa_ibl::IblFeature;
        use raa_lit::{LitOptions, LitRenderer};
        use raa_pipeline::{PassInfo, PassRole, Pipeline};
        use raa_scene::{RenderList, Scene};

        use crate::cloud_backend::create_cloud_backend;
        use crate::state::SkyState;

        let test = "gpu_frame_renders_and_filters_the_env";
        let Some(gpu) = software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        const SIZE: (u32, u32) = (32, 32);
        const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
        let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
        let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, SURFACE, false)));
        let mut lit = LitRenderer::new(&gpu, &pipeline, LitOptions::default()).expect("lit");
        let ibl = IblFeature::install(&gpu, &pipeline, &mut lit).expect("ibl");
        let state = SkyState::shared();
        let atmosphere: crate::atmosphere::SharedAtmosphere = Rc::new(RefCell::new(
            AtmosphereResources::new(&gpu).expect("atmosphere"),
        ));
        let cloud: crate::cloud_backend::SharedCloud =
            Rc::new(RefCell::new(create_cloud_backend(&gpu).expect("cloud")));
        let pass = {
            let p = pipeline.borrow();
            SkyEnvPass::new(
                &gpu,
                Rc::clone(&state),
                ibl.clone(),
                lit.handle(),
                p.lighting().layout(),
                atmosphere,
                cloud,
            )
        };
        pipeline
            .borrow_mut()
            .add(Box::new(pass))
            .expect("add sky.env");
        let info = pipeline
            .borrow()
            .passes()
            .into_iter()
            .find(|p| p.name == ENV_NAME);
        let expected = PassInfo {
            name: ENV_NAME.to_string(),
            role: PassRole::Pass,
            enabled: true,
            order: 10,
            required: false,
        };
        assert_eq!(info, Some(expected));

        let mut graph = FrameGraph::new(SURFACE);
        Pipeline::install(&pipeline, &mut graph);
        graph.compile().expect("compile");
        let backbuffer = OffscreenTarget::new(&gpu, SIZE.0, SIZE.1, SURFACE);
        let mut scene = Scene::with_capacity(1, 1);
        scene.camera.set_viewport(SIZE.0, SIZE.1);
        let mut list = RenderList::with_capacity(1);
        let mut tags = Vec::with_capacity(1);
        let mut slots = Vec::with_capacity(1);
        // Frame 0 renders and filters the cube; later frames map and apply the SH readback.
        for frame in 0..20 {
            lit.prepare(&gpu, &mut scene, &mut list, &mut tags, &mut slots)
                .expect("lit prepare");
            let commands = graph
                .execute(&gpu, backbuffer.view(), SIZE)
                .expect("execute");
            gpu.queue().submit([commands]);
            gpu.device()
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("device poll");
            if frame == 0 {
                assert!(!state.borrow().env_dirty, "the record cleared env_dirty");
            }
            if ibl.sh().is_some() {
                break;
            }
        }
        let error = block_on(errors.pop());
        assert!(error.is_none(), "GPU validation error: {error:?}");
        let sh = ibl
            .sh()
            .expect("the sky SH9 reaches the lighting within 20 frames");
        assert!(sh.iter().flatten().all(|c| c.is_finite()), "sh {sh:?}");
        assert!(
            sh[0][..3].iter().all(|&c| c > 0.0),
            "the day sky lights: {sh:?}"
        );
        let p = pipeline.borrow();
        assert_eq!(
            p.lighting().env().sky_params[2],
            0.0,
            "no background: constant mode"
        );
        drop(p);

        // Leaked like the other GPU tests (the WSL GL backend crashed dropping the device
        // after creating Rgba16Float textures).
        std::mem::forget(graph);
        std::mem::forget(backbuffer);
        std::mem::forget(ibl);
        std::mem::forget(lit);
        std::mem::forget(pipeline);
        std::mem::forget(gpu);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn same_model_source_change_cancels_an_older_ibl_completion() {
        use std::cell::RefCell;
        use std::rc::Rc;

        use raa_core::testing::{block_on, skip, software_gpu};
        use raa_core::{FrameGraph, OffscreenTarget};
        use raa_ibl::IblFeature;
        use raa_lit::{LitOptions, LitRenderer};
        use raa_pipeline::Pipeline;
        use raa_scene::{RenderList, Scene};

        use crate::cloud_backend::create_cloud_backend;
        use crate::state::SkyState;

        let test = "same_model_source_change_cancels_an_older_ibl_completion";
        let Some(gpu) = software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        const SIZE: (u32, u32) = (32, 32);
        const SURFACE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
        let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
        let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, SURFACE, false)));
        let mut lit = LitRenderer::new(&gpu, &pipeline, LitOptions::default()).expect("lit");
        let ibl = IblFeature::install(&gpu, &pipeline, &mut lit).expect("ibl");
        let state = SkyState::shared();
        // The legacy evaluator keeps the lifecycle independent of the physical LUT path.
        state.borrow_mut().desc.model = SkyModel::Legacy;
        let atmosphere: crate::atmosphere::SharedAtmosphere = Rc::new(RefCell::new(
            AtmosphereResources::new(&gpu).expect("atmosphere"),
        ));
        let cloud: crate::cloud_backend::SharedCloud =
            Rc::new(RefCell::new(create_cloud_backend(&gpu).expect("cloud")));
        let pass = {
            let p = pipeline.borrow();
            SkyEnvPass::new(
                &gpu,
                Rc::clone(&state),
                ibl.clone(),
                lit.handle(),
                p.lighting().layout(),
                atmosphere,
                cloud,
            )
        };
        pipeline
            .borrow_mut()
            .add(Box::new(pass))
            .expect("add sky.env");
        let mut graph = FrameGraph::new(SURFACE);
        Pipeline::install(&pipeline, &mut graph);
        graph.compile().expect("compile");
        let backbuffer = OffscreenTarget::new(&gpu, SIZE.0, SIZE.1, SURFACE);
        let mut scene = Scene::with_capacity(1, 1);
        scene.camera.set_viewport(SIZE.0, SIZE.1);
        let mut list = RenderList::with_capacity(1);
        let mut tags = Vec::with_capacity(1);
        let mut slots = Vec::with_capacity(1);
        let mut frame = |graph: &mut FrameGraph| {
            lit.prepare(&gpu, &mut scene, &mut list, &mut tags, &mut slots)
                .expect("lit prepare");
            let commands = graph
                .execute(&gpu, backbuffer.view(), SIZE)
                .expect("execute");
            gpu.queue().submit([commands]);
            gpu.device()
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("device poll");
        };

        // Frame 0 records the filter request of the first source (day).
        frame(&mut graph);
        assert!(ibl.sh().is_none());
        // Replace the accepted source with the same model before its completion is applied.
        {
            let mut s = state.borrow_mut();
            s.desc.zenith = [0.0, 0.0, 0.0];
            s.gpu_dirty = true;
            s.env_dirty = true;
        }
        frame(&mut graph); // invalidates before the older completion, then filters again
        frame(&mut graph); // without the invalidation this applies the replaced completion
        assert!(
            ibl.sh().is_none(),
            "a completed request of the replaced source was published"
        );
        frame(&mut graph); // the new source's completion is applied now
        let sh = ibl.sh().expect("the new source publishes");
        assert!(sh.iter().flatten().all(|c| c.is_finite()), "sh {sh:?}");
        let error = block_on(errors.pop());
        assert!(error.is_none(), "GPU validation error: {error:?}");

        std::mem::forget(graph);
        std::mem::forget(backbuffer);
        std::mem::forget(ibl);
        std::mem::forget(lit);
        std::mem::forget(pipeline);
        std::mem::forget(gpu);
    }
}
