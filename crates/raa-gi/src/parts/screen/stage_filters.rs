//! Independent a-trous and resolve stages for screen GI.
use raa_core::{GpuContext, RaaError};
use raa_pipeline::{ImageDesc, ImageDim, PrepareCtx, RecordCtx, validate_tunable};

use super::accumulate_cpu::atrous_stage_record;
use super::passes::{
    ScreenPipelines, ScreenWork, compute_pipeline, dispatch, float_entry, layout, module,
    storage_texture_entry,
};
use super::{buffer, full_view, group, texture, texture_resource};
use crate::parts::GBufferViews;
use crate::stages::{StageCx, StageImpl};

pub const STAGE_RESOLVE_ENTRY: &str = "raa_gi_stage_resolve";
pub fn stage_resolve_module_wgsl() -> String {
    include_str!("../../wgsl/screen/stage_resolve.wgsl").to_owned()
}

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const KEYS: &[&str] = &[
    "step",
    "atrousSigmaLuminance",
    "atrousNormalPower",
    "atrousPlane",
];

struct Output {
    size: (u32, u32),
    view: wgpu::TextureView,
}
impl Output {
    fn new(gpu: &GpuContext, size: (u32, u32)) -> Self {
        let usage = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC;
        Self {
            size,
            view: full_view(&texture(
                gpu,
                "raa-gi.stage.filter-output",
                size,
                1,
                FORMAT,
                usage,
            )),
        }
    }
    fn publish(
        &self,
        cx: &StageCx<'_>,
        ctx: &PrepareCtx<'_>,
        published: &mut Vec<(&'static str, ImageDesc)>,
        port: &'static str,
    ) {
        let desc = ImageDesc::new(ImageDim::D2, FORMAT, self.size.0, self.size.1, 1, 1);
        ctx.resources.publish(cx.entry, port, &self.view, desc);
        published.push((port, desc));
    }
}
fn dummy(gpu: &GpuContext) -> wgpu::TextureView {
    let tex = texture(
        gpu,
        "raa-gi.stage.filter-dummy",
        (1, 1),
        1,
        FORMAT,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    );
    gpu.queue().write_texture(
        tex.as_image_copy(),
        &[0, 0, 0, 0, 0, 0, 0, 0x3c],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(8),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    full_view(&tex)
}
fn ensure_output(slot: &mut Option<Output>, gpu: &GpuContext, size: (u32, u32)) {
    if slot.as_ref().is_none_or(|out| out.size != size) {
        *slot = Some(Output::new(gpu, size));
    }
}

#[derive(Default)]
pub(crate) struct AtrousStage {
    built: Option<(ScreenPipelines, u64, u64)>,
    output: Option<Output>,
    params: Option<wgpu::Buffer>,
    dummy: Option<wgpu::TextureView>,
}
impl StageImpl for AtrousStage {
    fn param_keys(&self) -> &'static [&'static str] {
        KEYS
    }
    fn check_value(&self, key: &str, value: f64) -> Result<(), String> {
        if key == "step" {
            if value.is_finite() && value.fract() == 0.0 && (1.0..=16.0).contains(&value) {
                Ok(())
            } else {
                Err(format!("invalid step {value}"))
            }
        } else {
            validate_tunable(crate::tunables(), key, value).map(|_| ())
        }
    }
    fn prepare(
        &mut self,
        cx: &StageCx<'_>,
        ctx: &mut PrepareCtx<'_>,
        published: &mut Vec<(&'static str, ImageDesc)>,
    ) -> Result<(), RaaError> {
        let generation = cx.shared.emissive_generation();
        if self.built.as_ref().is_none_or(|(_, values, emissive)| {
            *values != cx.values_generation || *emissive != generation
        }) {
            let pipelines = ScreenPipelines::new(
                ctx.gpu,
                &cx.shared.compute_group1_layout(),
                &cx.tunables(),
                (cx.shared.emissive_sources().screen, generation),
            );
            self.built = Some((pipelines, cx.values_generation, generation));
        }
        ensure_output(&mut self.output, ctx.gpu, ctx.size);
        let params = self.params.get_or_insert_with(|| {
            buffer(
                ctx.gpu,
                "raa-gi.stage.atrous.params",
                16,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            )
        });
        ctx.gpu.queue().write_buffer(
            params,
            0,
            &atrous_stage_record(cx.value("step", 1.0) as i32),
        );
        self.dummy.get_or_insert_with(|| dummy(ctx.gpu));
        if let Some(output) = &self.output {
            output.publish(cx, ctx, published, "dst");
        }
        Ok(())
    }
    fn record(&mut self, cx: &StageCx<'_>, ctx: &mut RecordCtx<'_>) -> Result<bool, RaaError> {
        let (
            Some(src),
            Some(frame),
            Some(group1),
            Some((built, _, _)),
            Some(output),
            Some(params),
            Some(dummy),
        ) = (
            ctx.resources.input(cx.entry, "src"),
            cx.shared.frame_buffer(),
            cx.shared.compute_group1(ctx.lighting),
            self.built.as_ref(),
            self.output.as_ref(),
            self.params.as_ref(),
            self.dummy.as_ref(),
        )
        else {
            return Ok(false);
        };
        let views = GBufferViews::resolve(ctx.targets, ctx.resources, cx.entry);
        // The optional meta input is not bound: atrous.wgsl reads variance from src alpha.
        let binding = group(
            ctx.gpu.device(),
            "raa-gi.stage.atrous",
            &built.atrous_layout,
            &[
                (0, frame.as_entire_binding()),
                (1, texture_resource(&views.depth)),
                (2, texture_resource(&views.normal)),
                (15, texture_resource(&src.view)),
                (16, texture_resource(&output.view)),
                (17, texture_resource(dummy)),
                (18, params.as_entire_binding()),
            ],
        );
        let groups = [binding.clone(), binding];
        let work = ScreenWork {
            pyramid: &[],
            pyramid_sizes: &[],
            trace: &groups[0],
            project: &groups[0],
            filter: &groups[0],
            integrate: &groups[0],
            atrous: &groups,
            atrous_passes: 1,
            probe_rays: 0,
            grid: (0, 0),
            viewport: output.size,
        };
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("raa-gi.stage.atrous"),
                timestamp_writes: None,
            });
        pass.set_bind_group(1, &group1, &[]);
        built.record_atrous(&mut pass, &work);
        Ok(true)
    }
}

#[derive(Default)]
pub(crate) struct ResolveStage {
    output: Option<Output>,
    dummy: Option<wgpu::TextureView>,
    pipeline: Option<(wgpu::BindGroupLayout, wgpu::ComputePipeline)>,
}
impl StageImpl for ResolveStage {
    fn param_keys(&self) -> &'static [&'static str] {
        &[]
    }
    fn check_value(&self, key: &str, _value: f64) -> Result<(), String> {
        Err(format!("unknown key {key}"))
    }
    fn prepare(
        &mut self,
        cx: &StageCx<'_>,
        ctx: &mut PrepareCtx<'_>,
        published: &mut Vec<(&'static str, ImageDesc)>,
    ) -> Result<(), RaaError> {
        ensure_output(&mut self.output, ctx.gpu, ctx.size);
        self.dummy.get_or_insert_with(|| dummy(ctx.gpu));
        self.pipeline.get_or_insert_with(|| {
            let device = ctx.gpu.device();
            let group = layout(
                device,
                "raa-gi.stage.resolve",
                &[
                    float_entry(0, false),
                    float_entry(1, false),
                    storage_texture_entry(2, FORMAT),
                ],
            );
            let shader = module(device, "raa-gi.stage.resolve", &stage_resolve_module_wgsl());
            let pipeline = compute_pipeline(
                device,
                "raa-gi.stage.resolve",
                &shader,
                STAGE_RESOLVE_ENTRY,
                &[Some(&group)],
                None,
            );
            (group, pipeline)
        });
        if let Some(output) = &self.output {
            output.publish(cx, ctx, published, "gi");
        }
        Ok(())
    }
    fn record(&mut self, cx: &StageCx<'_>, ctx: &mut RecordCtx<'_>) -> Result<bool, RaaError> {
        let (Some(input), Some(output), Some(dummy), Some((layout, pipeline))) = (
            ctx.resources.input(cx.entry, "irradiance"),
            self.output.as_ref(),
            self.dummy.as_ref(),
            self.pipeline.as_ref(),
        ) else {
            return Ok(false);
        };
        let occlusion = ctx.resources.input(cx.entry, "occlusion");
        let binding = group(
            ctx.gpu.device(),
            "raa-gi.stage.resolve",
            layout,
            &[
                (0, texture_resource(&input.view)),
                (
                    1,
                    texture_resource(occlusion.as_ref().map_or(dummy, |image| &image.view)),
                ),
                (2, texture_resource(&output.view)),
            ],
        );
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("raa-gi.stage.resolve"),
                timestamp_writes: None,
            });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &binding, &[]);
        dispatch(&mut pass, output.size.0, output.size.1);
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parts::GiShared;
    use crate::stages::{GiStageKind, install_stage};
    use raa_core::testing::skip;
    use raa_lit::{LitOptions, LitRenderer};
    use raa_pipeline::{LightingResources, Pipeline, ResourceTable, Targets};
    use std::{cell::RefCell, rc::Rc};

    #[test]
    fn resolve_shader_validates() {
        let source = stage_resolve_module_wgsl();
        assert!(source.contains(STAGE_RESOLVE_ENTRY));
        raa_shader::Composer::validate(&source).expect("resolve WGSL");
    }

    #[test]
    fn stages_prepare_without_inputs_and_validate_step() {
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(
                "stages_prepare_without_inputs_and_validate_step",
                "software_gpu returned None",
            );
            return;
        };
        for (kind, port) in [(GiStageKind::Atrous, "dst"), (GiStageKind::Resolve, "gi")] {
            let pipeline = Rc::new(RefCell::new(Pipeline::new(
                &gpu,
                wgpu::TextureFormat::Rgba8Unorm,
                false,
            )));
            let handle = install_stage(&gpu, &pipeline, kind, "t").expect("install");
            if kind == GiStageKind::Atrous {
                for step in [0.0, 17.0] {
                    assert!(handle.set_values(&[("step".into(), step)]).is_err());
                }
                assert!(handle.set_values(&[("step".into(), 8.0)]).is_ok());
            }
            let shared = GiShared::new(&gpu);
            shared.set_frame(Some(buffer(
                &gpu,
                "filter.test.frame",
                432,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            )));
            let lit = LitRenderer::new(&gpu, &pipeline, LitOptions::default()).expect("lit");
            handle.attach_shared(shared, lit.handle());
            let mut pass = pipeline
                .borrow_mut()
                .remove(&handle.entry())
                .expect("stage");
            let targets = Targets::new(&gpu, (8, 8));
            let mut lighting = LightingResources::new(&gpu);
            let resources = ResourceTable::new();
            pass.resize(&gpu, &targets);
            // No input is bound and no other pass ran; prepare reads no input.
            let mut ctx = PrepareCtx {
                gpu: &gpu,
                size: (8, 8),
                frame: 0,
                time_s: 0.0,
                dt_s: 0.0,
                targets: &targets,
                lighting: &mut lighting,
                resources: &resources,
            };
            pass.prepare(&mut ctx).expect("prepare");
            let desc = ImageDesc::new(ImageDim::D2, FORMAT, 8, 8, 1, 1);
            assert_eq!(handle.published(), vec![(port, desc)]);
            assert_eq!(
                resources.get(&handle.entry(), port).map(|v| v.desc),
                Some(desc)
            );
        }
    }
}
