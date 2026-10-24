//! `FogFeature::install` and the `FogHandle`: the `fog` render pass (S12, order 68) and the
//! `height` Fog slot provider.
//!
//! The pass writes the handle's description into the lighting's `EnvGpu` fog fields
//! (`FogDesc::write_env`) when it changed or after re-enabling, so the env upload stays
//! change-driven; transparent, fur and blended particle shading read the same fields
//! through `raa_slot_fog`. Each frame it draws a fullscreen triangle from hdr.src() into
//! hdr.dst() (group 0 = the lit frame, 1 = lighting, 2 = depth, hdr.src() and a sampler)
//! and swaps. Disabling writes the `clear` description, so the slot becomes identity.

use std::cell::RefCell;
use std::rc::Rc;

use raa_core::wgpu;
use raa_core::{GpuContext, RaaError};
use raa_lit::{LitHandle, LitRenderer};
use raa_pipeline::{
    EnvGpu, FeaturePass, HDR_FORMAT, LightingResources, Pipeline, PrepareCtx, RecordCtx, Targets,
};

use crate::desc::FogDesc;
use crate::shaders::{fog_pass_source, slot_provider};

/// S12 name of the pass.
const NAME: &str = "fog";
/// S12 order of the pass (after `ssr`, before `sunshafts`).
const ORDER: u32 = 68;

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

/// `pipeline.busy` for the shared fog state (only reachable by a re-entrant call).
fn state_busy() -> RaaError {
    refusal(
        "pipeline.busy",
        "the fog state is borrowed (re-entrant call while a frame executes)",
    )
}

/// State shared by the pass and every clone of a `FogHandle`.
struct FogState {
    desc: FogDesc,
}

type SharedFog = Rc<RefCell<FogState>>;

/// Writes `desc` into `env` unless it is the description last written (`written`).
/// Returns true when the env was written (and so marked changed by `env_mut`).
fn sync_env<'e>(
    desc: FogDesc,
    written: &mut Option<FogDesc>,
    env: impl FnOnce() -> &'e mut EnvGpu,
) -> bool {
    if *written == Some(desc) {
        return false;
    }
    desc.write_env(env());
    *written = Some(desc);
    true
}

/// The env of a disabled pass: the `clear` description (density 0, aerial 0).
fn disable_env(env: &mut EnvGpu) {
    FogDesc::default().write_env(env);
}

fn layout_entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty,
        count: None,
    }
}

/// Private group 2: depth (texture_depth_2d), hdr.src() and a filtering sampler.
fn create_private_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("raa-fog.private_layout"),
        entries: &[
            layout_entry(
                0,
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
            ),
            layout_entry(
                1,
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
            ),
            layout_entry(
                2,
                wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            ),
        ],
    })
}

fn create_pipeline(
    device: &wgpu::Device,
    layouts: [&wgpu::BindGroupLayout; 3],
) -> wgpu::RenderPipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("raa-fog.fog_pass"),
        source: wgpu::ShaderSource::Wgsl(fog_pass_source().into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("raa-fog.fog_pass"),
        bind_group_layouts: &layouts.map(Some),
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("raa-fog.fog_pass"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
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

/// The private bind group reading one texture of the HDR chain.
struct HdrGroup {
    texture: wgpu::Texture,
    group: wgpu::BindGroup,
}

/// The `fog` render pass.
struct FogPass {
    state: SharedFog,
    lit: LitHandle,
    private_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    /// The description last written into the env (None: write at the next prepare).
    written: Option<FogDesc>,
    /// One group per texture of the HDR chain (A and B), built in resize.
    groups: Vec<HdrGroup>,
}

impl FogPass {
    fn hdr_group(
        &self,
        device: &wgpu::Device,
        targets: &Targets,
        texture: &wgpu::Texture,
        view: &wgpu::TextureView,
    ) -> HdrGroup {
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-fog.private"),
            layout: &self.private_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(targets.depth()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        HdrGroup {
            texture: texture.clone(),
            group,
        }
    }
}

impl FeaturePass for FogPass {
    fn name(&self) -> &str {
        NAME
    }

    fn order(&self) -> u32 {
        ORDER
    }

    /// Rebuilds the private bind group of both HDR chain textures over the new depth.
    fn resize(&mut self, gpu: &GpuContext, targets: &Targets) {
        let device = gpu.device();
        let hdr = targets.hdr();
        self.groups = vec![
            self.hdr_group(device, targets, hdr.src_texture(), hdr.src()),
            self.hdr_group(device, targets, hdr.dst_texture(), hdr.dst()),
        ];
    }

    /// Disabling writes the `clear` description (the slot becomes identity); either way the
    /// next enabled prepare writes the handle's description again.
    fn set_enabled(&mut self, enabled: bool, lighting: &mut LightingResources) {
        if !enabled {
            disable_env(lighting.env_mut());
        }
        self.written = None;
    }

    /// Writes the description into the env fog fields when it changed (or after enabling).
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let desc = self.state.try_borrow().map_err(|_| state_busy())?.desc;
        let lighting = &mut *ctx.lighting;
        sync_env(desc, &mut self.written, || lighting.env_mut());
        Ok(())
    }

    /// Fullscreen draw from hdr.src() into hdr.dst(), then swap.
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let hdr = ctx.targets.hdr();
        let src = hdr.src_texture();
        let Some(private) = self.groups.iter().find(|g| g.texture == *src) else {
            return Err(RaaError::Resource(
                "fog: the HDR source is not a texture the pass was resized for".to_owned(),
            ));
        };
        let frame = self.lit.frame_bind_group();
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("raa-fog"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: hdr.dst(),
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
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
        pass.set_bind_group(2, &private.group, &[]);
        pass.draw(0..3, 0..1);
        drop(pass);
        hdr.swap();
        Ok(())
    }
}

/// Installs height fog and aerial perspective on a pipeline and its lit renderer.
pub struct FogFeature;

impl FogFeature {
    /// Adds the `Fog` slot provider `height` (priority 10) to `lit`, builds the fog pass
    /// pipeline (layout [lit frame, lighting, private depth + hdr + sampler]) and adds the
    /// FeaturePass `fog` (order 68) with the `clear` description.
    ///
    /// Refusals (nothing installed): `pipeline.busy` when the pipeline is borrowed,
    /// `pipeline.duplicate_pass` when `fog` is already installed, the `add_slot` refusal.
    pub fn install(
        gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
        lit: &mut LitRenderer,
    ) -> Result<FogHandle, RaaError> {
        let mut p = pipeline.try_borrow_mut().map_err(|_| pipeline_busy())?;
        if p.passes().iter().any(|info| info.name == NAME) {
            return Err(refusal(
                "pipeline.duplicate_pass",
                "the pass `fog` is already installed",
            ));
        }
        lit.add_slot(slot_provider())?;
        let device = gpu.device();
        let handle = lit.handle();
        let frame_layout = handle.frame_layout();
        let private_layout = create_private_layout(device);
        let render = create_pipeline(
            device,
            [&frame_layout, p.lighting().layout(), &private_layout],
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("raa-fog.linear_clamp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let state = Rc::new(RefCell::new(FogState {
            desc: FogDesc::default(),
        }));
        p.add(Box::new(FogPass {
            state: state.clone(),
            lit: handle,
            private_layout,
            pipeline: render,
            sampler,
            written: None,
            groups: Vec::new(),
        }))?;
        Ok(FogHandle { state })
    }
}

/// Access to the installed fog (clones share one state).
#[derive(Clone)]
pub struct FogHandle {
    state: SharedFog,
}

impl FogHandle {
    /// Replaces the description after `FogDesc::validate` (`post.invalid`; the stored
    /// description is unchanged on refusal). `pipeline.busy` on a re-entrant call.
    pub fn set(&self, d: FogDesc) -> Result<(), RaaError> {
        d.validate()?;
        self.state.try_borrow_mut().map_err(|_| state_busy())?.desc = d;
        Ok(())
    }

    /// A copy of the current description (`clear` until set).
    pub fn get(&self) -> FogDesc {
        self.state
            .try_borrow()
            .map(|st| st.desc)
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_set_refuses_invalid_and_keeps_desc() {
        let handle = FogHandle {
            state: Rc::new(RefCell::new(FogState {
                desc: FogDesc::default(),
            })),
        };
        let haze = FogDesc::preset("haze").unwrap();
        handle.set(haze).unwrap();
        let bad = FogDesc {
            density: -0.5,
            ..haze
        };
        let err = handle.set(bad).unwrap_err();
        assert_eq!(err.code(), "post.invalid");
        assert_eq!(handle.get(), haze);
        assert_eq!(handle.clone().get(), haze);
    }

    #[test]
    fn env_is_written_only_on_change_and_disable_leaves_density_zero() {
        let mut env = EnvGpu::default();
        let mut written = None;
        let haze = FogDesc::preset("haze").unwrap();
        assert!(sync_env(haze, &mut written, || &mut env));
        assert_eq!(env.fog[0], haze.density);
        assert!(!sync_env(haze, &mut written, || &mut env), "unchanged desc");
        // disabling: `clear` in the env and a rewrite on the next enabled prepare
        disable_env(&mut env);
        written = None;
        assert_eq!(env.fog[0], 0.0);
        assert_eq!(env.aerial[0], 0.0);
        assert_eq!(env.fog_color[3], 0.0);
        assert!(sync_env(haze, &mut written, || &mut env));
        assert_eq!(
            env.aerial,
            [haze.aerial_strength, haze.aerial_distance_m, 0.0, 0.0]
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    mod gpu {
        use std::cell::RefCell;
        use std::rc::Rc;

        use raa_core::testing::{block_on, skip, software_gpu};
        use raa_core::{FrameGraph, OffscreenTarget, wgpu};
        use raa_lit::{LitOptions, LitRenderer};
        use raa_pipeline::Pipeline;

        use super::super::*;

        const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
        const SIZE: u32 = 32;

        #[test]
        fn clear_fog_frame_without_geometry_validates() {
            let test = "clear_fog_frame_without_geometry_validates";
            let Some(gpu) = software_gpu() else {
                skip(test, "software_gpu returned None");
                return;
            };
            let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
            let pipeline = Rc::new(RefCell::new(Pipeline::new(&gpu, FORMAT, false)));
            let mut lit = LitRenderer::new(&gpu, &pipeline, LitOptions::default()).unwrap();
            let handle = FogFeature::install(&gpu, &pipeline, &mut lit).unwrap();
            handle.set(FogDesc::preset("clear").unwrap()).unwrap();
            let again = FogFeature::install(&gpu, &pipeline, &mut lit)
                .err()
                .unwrap();
            assert_eq!(again.code(), "pipeline.duplicate_pass");
            let target = OffscreenTarget::new(&gpu, SIZE, SIZE, FORMAT);
            let mut graph = FrameGraph::new(FORMAT);
            Pipeline::install(&pipeline, &mut graph);
            graph.compile().unwrap();
            let commands = graph.execute(&gpu, target.view(), (SIZE, SIZE)).unwrap();
            gpu.queue().submit([commands]);
            gpu.device()
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            let error = block_on(errors.pop());
            assert!(error.is_none(), "GPU validation error: {error:?}");
            assert_eq!(pipeline.borrow().lighting().env().fog[0], 0.0);
            // Leaked like the other GPU tests (GL backend crash at device drop on WSL).
            std::mem::forget(handle);
            std::mem::forget(lit);
            std::mem::forget(graph);
            std::mem::forget(target);
            std::mem::forget(pipeline);
            std::mem::forget(gpu);
        }
    }
}
