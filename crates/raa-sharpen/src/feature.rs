//! `SharpenFeature::install` and the `SharpenHandle`: the `sharpen` pass (S47, order 195,
//! installed disabled).
//!
//! Each prepared frame packs the stored sharpness into the 16 B params uniform; each recorded
//! frame draws the fullscreen sharpen kernel from `hdr.src()` into `hdr.dst()` and swaps. The
//! pass and every handle clone share one `Rc<RefCell<SharpenDesc>>`.

use std::cell::RefCell;
use std::num::NonZeroU64;
use std::rc::Rc;

use raa_core::wgpu;
use raa_core::{GpuContext, Lazy, RaaError};
use raa_pipeline::{FeaturePass, HDR_FORMAT, Pipeline, PrepareCtx, RecordCtx, Targets};

use crate::{
    FS_SHARPEN, PARAMS_SIZE, SHARPEN_ORDER, SHARPEN_PASS_NAME, SHARPEN_WGSL, SharpenDesc, VS_MAIN,
};

/// Bytes of the params uniform.
const PARAMS_BYTES: u64 = PARAMS_SIZE as u64;

fn refusal(code: &'static str, message: &str) -> RaaError {
    RaaError::FrameGraph {
        code,
        message: message.to_string(),
    }
}

/// `pipeline.busy`: the pipeline or the settings are borrowed (a call while a frame executes).
fn pipeline_busy() -> RaaError {
    refusal(
        "pipeline.busy",
        "the pipeline is borrowed (called while a frame executes)",
    )
}

/// A bind group layout entry for `binding`, visible to `visibility`.
fn layout_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::BindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty,
        count: None,
    }
}

/// A buffer binding that needs at least `min_size` bytes.
fn buffer(ty: wgpu::BufferBindingType, min_size: u64) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty,
        has_dynamic_offset: false,
        min_binding_size: NonZeroU64::new(min_size),
    }
}

/// The HDR source: a float 2D texture that is only loaded, so it is not filterable.
fn sampled() -> wgpu::BindingType {
    wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float { filterable: false },
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    }
}

/// A bind group entry binding `resource` at `binding`.
fn bind(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
}

/// Group-0 layout of the fragment draw: the params uniform and the HDR source.
fn create_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let stage = wgpu::ShaderStages::FRAGMENT;
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("raa-sharpen.layout"),
        entries: &[
            layout_entry(
                0,
                stage,
                buffer(wgpu::BufferBindingType::Uniform, PARAMS_BYTES),
            ),
            layout_entry(1, stage, sampled()),
        ],
    })
}

/// The fullscreen pipeline: `vs_main` and `fs_sharpen` into `HDR_FORMAT`, no blend.
fn create_pipeline(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> wgpu::RenderPipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("raa-sharpen"),
        source: wgpu::ShaderSource::Wgsl(SHARPEN_WGSL.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("raa-sharpen"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(FS_SHARPEN),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some(VS_MAIN),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some(FS_SHARPEN),
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

/// Writes the 16 B `SharpenParams`: sharpness f32 LE at offset 0, zeros at 4..16.
fn pack_params(out: &mut [u8; PARAMS_SIZE], sharpness: f32) {
    *out = [0; PARAMS_SIZE];
    out[..4].copy_from_slice(&sharpness.to_le_bytes());
}

/// Cached bind groups: the two own HDR chain textures plus graph-bound chain images.
const MAX_GROUPS: usize = 8;

/// One cached bind group, with the HDR chain texture whose view it binds.
struct TextureGroup {
    texture: wgpu::Texture,
    group: wgpu::BindGroup,
}

/// The `sharpen` feature pass (S47). It holds no per-view state, and it is not an overlay.
/// With m21 viewports the pipeline runs this one instance for each view, and each view has its
/// own targets; so while sharpening is enabled, the cached bind groups are rebuilt at each view
/// switch.
struct SharpenPass {
    desc: Rc<RefCell<SharpenDesc>>,
    layout: wgpu::BindGroupLayout,
    /// Built on the first sharpened frame (raa_core::Lazy says why).
    pipeline: Lazy<wgpu::RenderPipeline>,
    uniform: wgpu::Buffer,
    scratch: [u8; PARAMS_SIZE],
    /// One entry per HDR chain texture (A and B, rebuilt by `resize`) and per graph-bound
    /// source seen by `record` (at most MAX_GROUPS in all).
    groups: Vec<TextureGroup>,
}

impl SharpenPass {
    /// Builds the layout, the pipeline and the 16 B uniform (zero at creation). The bind groups
    /// wait for `resize`.
    fn new(gpu: &GpuContext, desc: Rc<RefCell<SharpenDesc>>) -> Self {
        let device = gpu.device();
        let layout = create_layout(device);
        let pipeline = {
            let (device, layout) = (device.clone(), layout.clone());
            Lazy::new(move || create_pipeline(&device, &layout))
        };
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raa-sharpen.params"),
            size: PARAMS_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        SharpenPass {
            desc,
            layout,
            pipeline,
            uniform,
            scratch: [0; PARAMS_SIZE],
            groups: Vec::new(),
        }
    }

    /// The group that binds the uniform and `view`, the view of `texture`.
    fn texture_group(
        &self,
        device: &wgpu::Device,
        texture: &wgpu::Texture,
        view: &wgpu::TextureView,
    ) -> TextureGroup {
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raa-sharpen.group"),
            layout: &self.layout,
            entries: &[
                bind(0, self.uniform.as_entire_binding()),
                bind(1, wgpu::BindingResource::TextureView(view)),
            ],
        });
        TextureGroup {
            texture: texture.clone(),
            group,
        }
    }
}

impl FeaturePass for SharpenPass {
    fn name(&self) -> &str {
        SHARPEN_PASS_NAME
    }

    fn order(&self) -> u32 {
        SHARPEN_ORDER
    }

    /// Drops the bind groups over the frame targets, which would keep replaced targets alive;
    /// `resize` rebuilds them.
    fn release(&mut self, _gpu: &GpuContext) {
        self.groups.clear();
    }

    /// Rebuilds the two groups only when the HDR chain's textures differ from the cached pair.
    fn resize(&mut self, gpu: &GpuContext, targets: &Targets) {
        let hdr = targets.hdr();
        let cached = |texture: &wgpu::Texture| self.groups.iter().any(|g| g.texture == *texture);
        if cached(hdr.src_texture()) && cached(hdr.dst_texture()) {
            return;
        }
        let device = gpu.device();
        self.groups = vec![
            self.texture_group(device, hdr.src_texture(), hdr.src()),
            self.texture_group(device, hdr.dst_texture(), hdr.dst()),
        ];
    }

    /// Packs the stored sharpness into the scratch and uploads it; nothing is allocated.
    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let sharpness = self
            .desc
            .try_borrow()
            .map_err(|_| pipeline_busy())?
            .sharpness;
        pack_params(&mut self.scratch, sharpness);
        ctx.gpu
            .queue()
            .write_buffer(&self.uniform, 0, &self.scratch);
        Ok(())
    }

    /// One fullscreen draw from `hdr.src()` into `hdr.dst()` (load Clear transparent), then swap.
    /// A render graph binds its own chain images (`HdrChain::bind`), possibly display-sized after
    /// a temporal resolve: their groups are created here on first use and kept in a small cache.
    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let hdr = ctx.targets.hdr();
        if !self.groups.iter().any(|g| g.texture == *hdr.src_texture()) {
            if self.groups.len() >= MAX_GROUPS {
                self.groups.remove(0);
            }
            let group = self.texture_group(ctx.gpu.device(), hdr.src_texture(), hdr.src());
            self.groups.push(group);
        }
        let Some(entry) = self.groups.iter().find(|g| g.texture == *hdr.src_texture()) else {
            return Err(RaaError::Resource(
                "sharpen: no bind group for the HDR source".to_owned(),
            ));
        };
        {
            let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("raa-sharpen.draw"),
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
            pass.set_pipeline(self.pipeline.get());
            pass.set_bind_group(0, &entry.group, &[]);
            pass.draw(0..3, 0..1);
        }
        hdr.swap();
        Ok(())
    }
}

/// Installs contrast-adaptive sharpening on a pipeline.
pub struct SharpenFeature;

impl SharpenFeature {
    /// Handoff: `SharpenFeature::install` is called by the downstream card
    /// "api-m34: setSharpening and sharpening on the Engine". That card then calls
    /// `SharpenHandle::get`, `SharpenHandle::set` and `SharpenHandle::set_enabled_flag`. Today
    /// `install` builds the pass, adds it as `sharpen` at order 195 and disables it.
    /// `SharpenHandle::set` validates first (the share-1 Handoff at
    /// `crates/raa-sharpen/src/desc.rs:33 SharpenDesc::validate`), then stores the whole desc,
    /// its `enabled` field included; the pipeline flag changes only through `Pipeline::set_enabled`.
    /// `SharpenHandle::set_enabled_flag` writes only the stored `enabled` field, unvalidated.
    /// `SharpenHandle::get` returns the stored copy by value.
    ///
    /// Refusals, with nothing installed: `pipeline.busy` when the pipeline is borrowed, and
    /// `pipeline.duplicate_pass` when `sharpen` is already listed.
    pub fn install(
        gpu: &GpuContext,
        pipeline: &Rc<RefCell<Pipeline>>,
    ) -> Result<SharpenHandle, RaaError> {
        let mut p = pipeline.try_borrow_mut().map_err(|_| pipeline_busy())?;
        if p.entry_enabled(SHARPEN_PASS_NAME).is_some() {
            return Err(refusal(
                "pipeline.duplicate_pass",
                "the pass `sharpen` is already installed",
            ));
        }
        let desc = Rc::new(RefCell::new(SharpenDesc::default()));
        p.add(Box::new(SharpenPass::new(gpu, Rc::clone(&desc))))?;
        p.set_enabled(SHARPEN_PASS_NAME, false)?;
        Ok(SharpenHandle { desc })
    }
}

/// Access to the installed sharpening. Clones share one description with the pass.
#[derive(Clone)]
pub struct SharpenHandle {
    desc: Rc<RefCell<SharpenDesc>>,
}

impl SharpenHandle {
    /// The stored description by value; `SharpenDesc::default()` on a re-entrant borrow.
    pub fn get(&self) -> SharpenDesc {
        self.desc.try_borrow().map(|d| *d).unwrap_or_default()
    }

    /// Stores `desc` after `SharpenDesc::validate`. A refusal (`post.invalid`) changes nothing;
    /// `pipeline.busy` on a re-entrant borrow.
    pub fn set(&self, desc: SharpenDesc) -> Result<(), RaaError> {
        desc.validate()?;
        let mut stored = self.desc.try_borrow_mut().map_err(|_| pipeline_busy())?;
        *stored = desc;
        Ok(())
    }

    /// Writes `enabled` into the stored description without validation. A re-entrant borrow
    /// leaves it unchanged. The pipeline flag of `sharpen` is set by `Pipeline::set_enabled`.
    pub fn set_enabled_flag(&self, enabled: bool) {
        if let Ok(mut stored) = self.desc.try_borrow_mut() {
            stored.enabled = enabled;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A handle over a fresh state: the default description, no frame seen.
    fn fresh_handle() -> SharpenHandle {
        SharpenHandle {
            desc: Rc::new(RefCell::new(SharpenDesc::default())),
        }
    }

    #[test]
    fn set_refuses_an_out_of_range_sharpness_and_keeps_the_stored_desc() {
        let handle = fresh_handle();
        let valid = SharpenDesc {
            enabled: false,
            sharpness: 0.5,
        };
        handle.set(valid).unwrap();
        let invalid = SharpenDesc {
            sharpness: 1.5,
            ..valid
        };
        let err = handle.set(invalid).err().unwrap();
        assert_eq!(err.code(), "post.invalid");
        assert_eq!(handle.get(), valid);
    }

    #[test]
    fn set_enabled_flag_writes_the_enabled_field_and_keeps_the_sharpness() {
        let handle = fresh_handle();
        handle
            .set(SharpenDesc {
                enabled: false,
                sharpness: 0.5,
            })
            .unwrap();
        handle.set_enabled_flag(true);
        assert_eq!(
            handle.get(),
            SharpenDesc {
                enabled: true,
                sharpness: 0.5,
            }
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    mod gpu {
        use std::cell::RefCell;
        use std::rc::Rc;

        use raa_core::testing::{skip, software_gpu};
        use raa_core::wgpu;
        use raa_pipeline::Pipeline;

        use super::super::*;

        #[test]
        fn install_lists_the_pass_disabled_at_order_195_and_refuses_a_second_install() {
            let name = "install_lists_the_pass_disabled_at_order_195_and_refuses_a_second_install";
            let Some(gpu) = software_gpu() else {
                skip(
                    name,
                    "software_gpu returned None; see the software_gpu line above",
                );
                return;
            };
            // No frame runs here, so the HDR chain and its source need no initialisation.
            let pipeline = Rc::new(RefCell::new(Pipeline::new(
                &gpu,
                wgpu::TextureFormat::Rgba8Unorm,
                false,
            )));
            let handle = SharpenFeature::install(&gpu, &pipeline).unwrap();
            let passes = pipeline.borrow().passes();
            let info = passes.iter().find(|p| p.name == SHARPEN_PASS_NAME).unwrap();
            assert_eq!(info.order, SHARPEN_ORDER);
            assert!(!info.enabled);
            assert_eq!(handle.get(), SharpenDesc::default());
            let err = SharpenFeature::install(&gpu, &pipeline).err().unwrap();
            assert_eq!(err.code(), "pipeline.duplicate_pass");
        }
    }
}
