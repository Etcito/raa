//! Independent `ssgi.trace` and `ssgi.resolve` stages. The whole `gi.ssgi` pass is unchanged.

use raa_core::{GpuContext, RaaError};
use raa_pipeline::{ImageDesc, ImageDim, PrepareCtx, RecordCtx, validate_tunable};

use super::passes::{SSGI_PARAMS_SIZE, SSGI_RESOLVE_ENTRY, SSGI_TRACE_ENTRY, SsgiPipelines};
use super::{cpu, ssgi_counts};
use crate::parts::GBufferViews;
use crate::parts::screen::passes::dispatch;
use crate::settings::SsgiQuality;
use crate::stages::{StageCx, StageImpl};
use crate::tunables;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const KEYS: &[&str] = &[
    "quality",
    "slicesLow",
    "slicesMedium",
    "slicesHigh",
    "stepsLow",
    "stepsMedium",
    "stepsHigh",
];
const RESOLVE_KEYS: &[&str] = &["changeSigma"];

fn texture(
    gpu: &GpuContext,
    label: &str,
    size: (u32, u32),
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    gpu.device().create_texture(&wgpu::TextureDescriptor {
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
        usage,
        view_formats: &[],
    })
}
fn view(tex: &wgpu::Texture) -> wgpu::TextureView {
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}
fn output(gpu: &GpuContext, label: &str, size: (u32, u32)) -> wgpu::TextureView {
    view(&texture(
        gpu,
        label,
        size,
        FORMAT,
        wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
    ))
}
fn uniform(gpu: &GpuContext, label: &str) -> wgpu::Buffer {
    gpu.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: SSGI_PARAMS_SIZE,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
fn group(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    entries: &[(u32, wgpu::BindingResource<'_>)],
) -> wgpu::BindGroup {
    let entries: Vec<_> = entries
        .iter()
        .map(|(binding, resource)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: resource.clone(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &entries,
    })
}
fn write_params(gpu: &GpuContext, params: &wgpu::Buffer, slices: u32, steps: u32, history: bool) {
    let words = [
        slices.to_le_bytes(),
        steps.to_le_bytes(),
        u32::from(history).to_le_bytes(),
        0u32.to_le_bytes(),
        cpu::RADIUS_METRES.to_le_bytes(),
        cpu::THICKNESS_METRES.to_le_bytes(),
        u32::from(history).to_le_bytes(),
        0u32.to_le_bytes(),
    ];
    let mut bytes = [0; 32];
    for (dst, src) in bytes.chunks_exact_mut(4).zip(words) {
        dst.copy_from_slice(&src);
    }
    gpu.queue().write_buffer(params, 0, &bytes);
}
fn publish(
    cx: &StageCx<'_>,
    ctx: &PrepareCtx<'_>,
    published: &mut Vec<(&'static str, ImageDesc)>,
    port: &'static str,
    image: &wgpu::TextureView,
    size: (u32, u32),
) {
    let desc = ImageDesc::new(ImageDim::D2, FORMAT, size.0, size.1, 1, 1);
    ctx.resources.publish(cx.entry, port, image, desc);
    published.push((port, desc));
}
fn quality(cx: &StageCx<'_>) -> SsgiQuality {
    match cx.value("quality", 1.0) as u32 {
        0 => SsgiQuality::Low,
        2 => SsgiQuality::High,
        _ => SsgiQuality::Medium,
    }
}
fn tuned(gpu: &GpuContext, cx: &StageCx<'_>, slot: &mut Option<(SsgiPipelines, u64)>) {
    if slot
        .as_ref()
        .is_some_and(|(_, generation)| *generation == cx.values_generation)
    {
        return;
    }
    *slot = Some((
        SsgiPipelines::new_with_tunables(gpu, &cx.shared.compute_group1_layout(), &cx.tunables()),
        cx.values_generation,
    ));
}

#[derive(Default)]
pub(crate) struct SsgiTraceStage {
    pipelines: Option<(SsgiPipelines, u64)>,
    size: Option<(u32, u32)>,
    raw: Option<wgpu::TextureView>,
    params: Option<wgpu::Buffer>,
    dummy: Option<wgpu::TextureView>,
    group: Option<TraceBound>,
    ready: bool,
}
struct TraceBound {
    views: GBufferViews,
    history: Option<wgpu::TextureView>,
    voxels: Option<u64>,
    group: wgpu::BindGroup,
}
impl StageImpl for SsgiTraceStage {
    fn param_keys(&self) -> &'static [&'static str] {
        KEYS
    }
    fn check_value(&self, key: &str, value: f64) -> Result<(), String> {
        if key == "quality" {
            return if value == 0.0 || value == 1.0 || value == 2.0 {
                Ok(())
            } else {
                Err("quality must be 0, 1 or 2".into())
            };
        }
        validate_tunable(tunables(), key, value).map(|_| ())
    }
    fn prepare(
        &mut self,
        cx: &StageCx<'_>,
        ctx: &mut PrepareCtx<'_>,
        published: &mut Vec<(&'static str, ImageDesc)>,
    ) -> Result<(), RaaError> {
        self.ready = false;
        tuned(ctx.gpu, cx, &mut self.pipelines);
        if let Some((pipelines, _)) = self.pipelines.as_mut() {
            pipelines.voxels = cx.shared.voxel_buffer();
        }
        let size = (ctx.size.0.max(1), ctx.size.1.max(1));
        if self.size != Some(size) {
            self.size = Some(size);
            self.raw = Some(output(ctx.gpu, "raa-gi.ssgi-stage.raw", size));
            self.params = Some(uniform(ctx.gpu, "raa-gi.ssgi-stage.params"));
            self.group = None;
        }
        self.dummy.get_or_insert_with(|| {
            let tex = texture(
                ctx.gpu,
                "raa-gi.ssgi-stage.dummy",
                (1, 1),
                FORMAT,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            );
            ctx.gpu.queue().write_texture(
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
            view(&tex)
        });
        let (slices, steps) = ssgi_counts(quality(cx), &cx.tunables());
        let history = cx.shared.history_valid();
        if let Some(params) = &self.params {
            write_params(ctx.gpu, params, slices, steps, history);
        }
        if let Some(raw) = &self.raw {
            publish(cx, ctx, published, "raw", raw, size);
        }
        self.ready = true;
        Ok(())
    }
    fn record(&mut self, cx: &StageCx<'_>, ctx: &mut RecordCtx<'_>) -> Result<bool, RaaError> {
        if !self.ready {
            return Ok(false);
        }
        let (Some((pipelines, _)), Some(raw), Some(params), Some(dummy), Some(frame), Some(group1)) = (
            self.pipelines.as_ref(),
            self.raw.as_ref(),
            self.params.as_ref(),
            self.dummy.as_ref(),
            cx.shared.frame_buffer(),
            cx.shared.compute_group1(ctx.lighting),
        ) else {
            return Ok(false);
        };
        let views = GBufferViews::resolve(ctx.targets, ctx.resources, cx.entry);
        let history = cx.shared.history_view();
        let voxels = pipelines.voxels.as_ref();
        let stale = self.group.as_ref().is_none_or(|old| {
            old.views != views || old.history != history || old.voxels != voxels.map(|v| v.1)
        });
        if stale {
            let fallback = ctx.gpu.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("raa-gi.ssgi-stage.voxels-dummy"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let entries = [
                (0, frame.as_entire_binding()),
                (1, wgpu::BindingResource::TextureView(&views.depth)),
                (2, wgpu::BindingResource::TextureView(&views.normal)),
                (
                    3,
                    wgpu::BindingResource::TextureView(history.as_ref().unwrap_or(dummy)),
                ),
                (5, wgpu::BindingResource::TextureView(raw)),
                (6, voxels.map_or(&fallback, |v| &v.0).as_entire_binding()),
                (8, params.as_entire_binding()),
            ];
            self.group = Some(TraceBound {
                views: views.clone(),
                history: history.clone(),
                voxels: voxels.map(|v| v.1),
                group: group(
                    ctx.gpu.device(),
                    "raa-gi.ssgi-stage.trace",
                    &pipelines.trace_layout,
                    &entries,
                ),
            });
        }
        let Some(bound) = self.group.as_ref() else {
            return Ok(false);
        };
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(SSGI_TRACE_ENTRY),
                timestamp_writes: None,
            });
        pass.set_bind_group(1, &group1, &[]);
        pass.set_pipeline(&pipelines.trace);
        pass.set_bind_group(2, &bound.group, &[]);
        dispatch(&mut pass, self.size.unwrap().0, self.size.unwrap().1);
        Ok(true)
    }
    fn resize(&mut self) {
        self.group = None;
    }
}

#[derive(Default)]
pub(crate) struct SsgiResolveStage {
    pipelines: Option<(SsgiPipelines, u64)>,
    size: Option<(u32, u32)>,
    ssgi: Option<wgpu::TextureView>,
    accumulated: Option<wgpu::TextureView>,
    params: Option<wgpu::Buffer>,
    dummy: Option<wgpu::TextureView>,
    bound: Option<ResolveBound>,
    ready: bool,
}
struct ResolveBound {
    views: GBufferViews,
    raw: wgpu::TextureView,
    probe: Option<wgpu::TextureView>,
    history: Option<wgpu::TextureView>,
    frame: wgpu::Buffer,
    group: wgpu::BindGroup,
}
impl StageImpl for SsgiResolveStage {
    fn param_keys(&self) -> &'static [&'static str] {
        RESOLVE_KEYS
    }
    fn check_value(&self, key: &str, value: f64) -> Result<(), String> {
        validate_tunable(tunables(), key, value).map(|_| ())
    }
    fn prepare(
        &mut self,
        cx: &StageCx<'_>,
        ctx: &mut PrepareCtx<'_>,
        published: &mut Vec<(&'static str, ImageDesc)>,
    ) -> Result<(), RaaError> {
        self.ready = false;
        tuned(ctx.gpu, cx, &mut self.pipelines);
        let size = (ctx.size.0.max(1), ctx.size.1.max(1));
        if self.size != Some(size) {
            self.size = Some(size);
            self.ssgi = Some(output(ctx.gpu, "raa-gi.ssgi-stage.resolve", size));
            self.accumulated = Some(output(ctx.gpu, "raa-gi.ssgi-stage.accumulated", size));
            self.params = Some(uniform(ctx.gpu, "raa-gi.ssgi-stage.resolve-params"));
            self.bound = None;
        }
        self.dummy.get_or_insert_with(|| {
            let tex = texture(
                ctx.gpu,
                "raa-gi.ssgi-stage.resolve-dummy",
                (1, 1),
                FORMAT,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            );
            ctx.gpu.queue().write_texture(
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
            view(&tex)
        });
        let history =
            ctx.resources.input(cx.entry, "history").is_some() && cx.shared.camera_valid();
        if let Some(params) = &self.params {
            write_params(ctx.gpu, params, 0, 0, history);
        }
        if let Some(ssgi) = &self.ssgi {
            publish(cx, ctx, published, "ssgi", ssgi, size);
        }
        if let Some(accumulated) = &self.accumulated {
            publish(cx, ctx, published, "accumulated", accumulated, size);
        }
        self.ready = true;
        Ok(())
    }
    fn record(&mut self, cx: &StageCx<'_>, ctx: &mut RecordCtx<'_>) -> Result<bool, RaaError> {
        if !self.ready {
            return Ok(false);
        }
        let Some(raw) = ctx.resources.input(cx.entry, "raw") else {
            return Ok(false);
        };
        let (Some((pipelines, _)), Some(ssgi), Some(accumulated), Some(params), Some(dummy)) = (
            self.pipelines.as_ref(),
            self.ssgi.as_ref(),
            self.accumulated.as_ref(),
            self.params.as_ref(),
            self.dummy.as_ref(),
        ) else {
            return Ok(false);
        };
        let views = GBufferViews::resolve(ctx.targets, ctx.resources, cx.entry);
        let probe = ctx
            .resources
            .input(cx.entry, "probe")
            .map(|image| image.view);
        let history = ctx
            .resources
            .input(cx.entry, "history")
            .map(|image| image.view);
        let frame = cx.lit.frame_buffer();
        let stale = self.bound.as_ref().is_none_or(|old| {
            old.views != views
                || old.raw != raw.view
                || old.probe != probe
                || old.history != history
                || old.frame != frame
        });
        if stale {
            let history_usable = history.is_some() && cx.shared.camera_valid();
            let bind = group(
                ctx.gpu.device(),
                "raa-gi.ssgi-stage.resolve",
                &pipelines.resolve_layout,
                &[
                    (0, frame.as_entire_binding()),
                    (1, wgpu::BindingResource::TextureView(&views.depth)),
                    (2, wgpu::BindingResource::TextureView(&views.motion)),
                    (3, wgpu::BindingResource::TextureView(&raw.view)),
                    (
                        4,
                        wgpu::BindingResource::TextureView(
                            history.as_ref().filter(|_| history_usable).unwrap_or(dummy),
                        ),
                    ),
                    (
                        5,
                        wgpu::BindingResource::TextureView(probe.as_ref().unwrap_or(dummy)),
                    ),
                    (6, wgpu::BindingResource::TextureView(accumulated)),
                    (7, wgpu::BindingResource::TextureView(ssgi)),
                    (8, params.as_entire_binding()),
                    (9, wgpu::BindingResource::TextureView(&views.normal)),
                ],
            );
            self.bound = Some(ResolveBound {
                views,
                raw: raw.view,
                probe,
                history,
                frame,
                group: bind,
            });
        }
        let Some(bound) = self.bound.as_ref() else {
            return Ok(false);
        };
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(SSGI_RESOLVE_ENTRY),
                timestamp_writes: None,
            });
        pass.set_pipeline(&pipelines.resolve);
        pass.set_bind_group(0, &bound.group, &[]);
        dispatch(&mut pass, self.size.unwrap().0, self.size.unwrap().1);
        Ok(true)
    }
    fn resize(&mut self) {
        self.bound = None;
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
    fn ssgi_stages_prepare_outputs_and_reject_invalid_quality_without_inputs() {
        let test = "ssgi_stages_prepare_outputs_and_reject_invalid_quality_without_inputs";
        let Some(gpu) = crate::parts::serial_software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let pipeline = Rc::new(RefCell::new(Pipeline::new(
            &gpu,
            wgpu::TextureFormat::Rgba8Unorm,
            false,
        )));
        let lit = LitRenderer::new(&gpu, &pipeline, LitOptions::default()).expect("lit renderer");
        let shared = GiShared::new(&gpu);
        shared.set_frame(Some(gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("ssgi-stage-test-frame"),
            size: 432,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })));
        for (kind, expected) in [
            (GiStageKind::SsgiTrace, vec!["raw"]),
            (GiStageKind::SsgiResolve, vec!["ssgi", "accumulated"]),
        ] {
            let handle = install_stage(&gpu, &pipeline, kind, "t").expect("stage install");
            handle.attach_shared(shared.clone(), lit.handle());
            if kind == GiStageKind::SsgiTrace {
                assert!(handle.set_values(&[("quality".into(), 3.0)]).is_err());
            }
            let mut pass = pipeline
                .borrow_mut()
                .remove(&handle.entry())
                .expect("stage pass");
            let targets = Targets::new(&gpu, (8, 8));
            let mut lighting = LightingResources::new(&gpu);
            let resources = ResourceTable::new();
            pass.resize(&gpu, &targets);
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
            let got = handle.published();
            assert_eq!(
                got.iter().map(|(port, _)| *port).collect::<Vec<_>>(),
                expected
            );
            assert!(
                got.iter()
                    .all(|(_, desc)| desc.format == FORMAT && desc.width == 8 && desc.height == 8)
            );
            assert!(got.iter().all(|(_, desc)| desc.dim == ImageDim::D2));
        }
    }
}
