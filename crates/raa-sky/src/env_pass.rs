//! The `sky.env` feature pass (S12, order 10): renders the SKY_ENV source cube from
//! `raa_sky_radiance` (one fullscreen triangle per face) and hands it to the IBL filter, when
//! the sky changed or every frame per `EnvUpdate`. Its prepare drives the IBL readback
//! (`IblHandle::apply_pending`) and resets the background mode to constant; the
//! `sky.background` contributor sets it back to sky later in the same frame when enabled.
//!
//! The lighting's `env_specular` is the IBL output cube, never this source cube, so the env
//! render passes bind group 1 without a read/write hazard.

use raa_core::{GpuContext, RaaError};
use raa_ibl::IblHandle;
use raa_lit::LitHandle;
use raa_pipeline::{FeaturePass, HDR_FORMAT, LightingResources, PrepareCtx, RecordCtx};

use crate::desc::EnvUpdate;
use crate::shaders::env_module_wgsl;
use crate::state::SharedSky;

/// S12 name of the pass.
pub(crate) const ENV_NAME: &str = "sky.env";
/// S12 order of the pass.
pub(crate) const ENV_ORDER: u32 = 10;
/// SKY_ENV: face size in pixels.
const ENV_SIZE: u32 = 256;
/// SKY_ENV: mip count (256..8).
const ENV_MIPS: u32 = 6;

/// Renders the sky env cube and filters it into the IBL.
#[allow(dead_code)] // consumed by SkyFeature (sky.env)
pub(crate) struct SkyEnvPass {
    state: SharedSky,
    ibl: IblHandle,
    lit: LitHandle,
    /// The SKY_ENV source cube (mip 0 rendered here, mips 1..5 written by the filter).
    cube: wgpu::Texture,
    /// Mip-0 views of the six faces in wgpu order +X, -X, +Y, -Y, +Z, -Z.
    faces: [wgpu::TextureView; 6],
    pipeline: wgpu::RenderPipeline,
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

/// Builds the env pipeline: groups [frame, lighting], no vertex buffers, HDR colour without
/// blending, no depth, single sample.
fn build_pipeline(
    gpu: &GpuContext,
    frame_layout: &wgpu::BindGroupLayout,
    lighting_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let device = gpu.device();
    let wgsl = env_module_wgsl();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("raa.sky.env"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("raa.sky.env.layout"),
        bind_group_layouts: &[Some(frame_layout), Some(lighting_layout)],
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
    /// Creates the source cube, its face views and the env pipeline over the lit frame
    /// layout and the pipeline's lighting layout (`Pipeline::lighting().layout()`).
    #[allow(dead_code)] // consumed by SkyFeature (sky.env)
    pub(crate) fn new(
        gpu: &GpuContext,
        state: SharedSky,
        ibl: IblHandle,
        lit: LitHandle,
        lighting_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let cube = create_cube(gpu);
        let faces = std::array::from_fn(|face| face_view(&cube, face as u32));
        let pipeline = build_pipeline(gpu, &lit.frame_layout(), lighting_layout);
        SkyEnvPass {
            state,
            ibl,
            lit,
            cube,
            faces,
            pipeline,
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

    /// Disabling clears the env (flat S1 ambient) and sets the background mode to constant;
    /// enabling marks the cube and the `EnvGpu` sky fields dirty so the env is rebuilt.
    fn set_enabled(&mut self, enabled: bool, lighting: &mut LightingResources) {
        if enabled {
            if let Ok(mut state) = self.state.try_borrow_mut() {
                state.env_dirty = true;
                state.gpu_dirty = true;
            }
        } else {
            lighting.clear_env();
            lighting.env_mut().sky_params[2] = 0.0;
        }
    }

    /// Writes the `EnvGpu` sky fields when the description changed, resets the background
    /// mode to constant (S12: `sky.background` sets it to sky later in this frame when
    /// enabled), applies the IBL readbacks of earlier frames and decides whether the cube is
    /// rendered this frame.
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        if self.clear_env {
            ctx.lighting.clear_env();
            self.clear_env = false;
        }
        let mut state = self.state.try_borrow_mut().map_err(|_| state_busy())?;
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
        Ok(())
    }

    /// When the env is dirty: renders the six faces of the source cube (clear, then one
    /// fullscreen triangle with `instance_index` = face) and records the IBL filter of it.
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        {
            let mut state = self.state.try_borrow_mut().map_err(|_| state_busy())?;
            if !state.env_dirty {
                return Ok(());
            }
            state.env_dirty = false;
        }
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
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &frame, &[]);
            pass.set_bind_group(1, ctx.lighting.bind_group(), &[]);
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
        let pass = {
            let p = pipeline.borrow();
            SkyEnvPass::new(
                &gpu,
                Rc::clone(&state),
                ibl.clone(),
                lit.handle(),
                p.lighting().layout(),
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
        // Frame 0 renders and filters the cube; later frames map and apply the SH readback.
        for frame in 0..20 {
            lit.prepare(&gpu, &mut scene, &mut list, &mut tags)
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
}
