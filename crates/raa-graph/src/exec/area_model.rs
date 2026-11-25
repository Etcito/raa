//! Runs the ticket 09 pilot on an explicitly wired stratified area-shadow branch.
use std::{cell::RefCell, rc::Rc};

use raa_core::{GpuContext, RaaError};
use raa_gi::{
    GiHandle,
    stages::{GiStageKind, stage_entry},
};
use raa_lit::{AreaStochasticFrame, LitHandle, StochasticMode, table_slot};
use raa_pipeline::{GraphFrame, MAX_STOCHASTIC_LIGHTS, PublishedImage, RecordCtx, ResourceTable};
use raa_shader::LIGHT_GPU_SIZE;

use crate::{
    catalog::{ParamValue, SignalEncoding},
    error::{Issue, Severity, Stage},
    exec::{FeatureHandles, SetResult},
    learned::{AreaPilot, AreaPilotPackage, AreaPilotRect},
    plan::{GraphPlan, InputSource, PlanNode},
};

pub(super) const SLOTS: usize = MAX_STOCHASTIC_LIGHTS;
const LAYERS: u32 = SLOTS.div_ceil(2) as u32;
const BRIDGE: &str = include_str!("area_model.wgsl");
const RAW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

pub struct AreaGraphModel {
    pub package: AreaPilotPackage,
    pub scene_size: (u32, u32),
}

pub(super) fn issue(code: &'static str, message: impl Into<String>) -> Issue {
    Issue {
        code,
        severity: Severity::Error,
        stage: Stage::Apply,
        node: None,
        port: None,
        param: None,
        link: None,
        message: message.into(),
        fix: "Use a supported stratified area.trace → area.mask → forward branch and scene extent, or clear the area model selection.".into(),
    }
}

pub(super) fn refused(issue: Issue) -> SetResult {
    SetResult {
        ok: false,
        rebuilt: 0,
        issues: vec![issue],
    }
}

pub(super) struct AreaRoute {
    pub(super) trace: String,
    pub(super) mask: String,
    pub(super) alias: Option<String>,
}

impl AreaRoute {
    pub(super) fn from_plan(plan: &GraphPlan) -> Option<Self> {
        fn producer<'a>(
            plan: &'a GraphPlan,
            node: &PlanNode,
            port: &str,
            output: &str,
        ) -> Option<&'a PlanNode> {
            let InputSource::Link {
                node,
                port,
                history: false,
            } = &node.inputs.iter().find(|input| input.port == port)?.source
            else {
                return None;
            };
            if port != output {
                return None;
            }
            plan.nodes.iter().find(|candidate| candidate.id == *node)
        }
        for required in ["pass.light.tiles", "pass.gi.volume", "gi.pyramid"] {
            if !plan.nodes.iter().any(|node| node.type_id == required) {
                return None;
            }
        }
        let forward = plan
            .nodes
            .iter()
            .find(|node| node.type_id == "pass.forward")?;
        let mask = producer(plan, forward, "areaShadow", "areaShadow")?;
        let neural = producer(plan, mask, "src", "pairs")
            .filter(|node| node.type_id == "neural.area-reconstruction");
        let trace = match neural {
            Some(neural) => producer(plan, neural, "pairs", "raw")?,
            None => producer(plan, mask, "src", "raw")?,
        };
        if mask.type_id != "area.mask"
            || trace.type_id != "area.trace"
            || mask.params.iter().any(|(name, value)| {
                name == "stochasticMode"
                    && matches!(value, ParamValue::Enum(mode) if mode != "stratified")
            })
            || plan.input_signal(mask, "src")?.encoding != SignalEncoding::StratifiedPairs
            || !(trace.order_index < mask.order_index && mask.order_index < forward.order_index)
        {
            return None;
        }
        Some(Self {
            trace: stage_entry(GiStageKind::AreaTrace, &trace.id),
            mask: stage_entry(GiStageKind::AreaMask, &mask.id),
            alias: neural.map(|node| format!("neural.area-reconstruction#{}", node.id)),
        })
    }
}

pub(super) struct AreaModelFrame {
    route: AreaRoute,
    lit: LitHandle,
    gi: GiHandle,
    package: AreaPilotPackage,
    size: (u32, u32),
    resources: Option<AreaResources>,
    previous: Option<AreaFrame>,
    pending: Option<AreaFrame>,
}

pub(super) struct AreaFrame {
    stochastic: AreaStochasticFrame,
    lights: [Option<[u8; LIGHT_GPU_SIZE]>; SLOTS],
    pub(super) slots: u32,
    jitter: [f32; 2],
}

impl AreaFrame {
    pub(super) fn capture(stochastic: AreaStochasticFrame, jitter: [f32; 2], ctx: &RecordCtx<'_>) -> Self {
        let lights = std::array::from_fn(|slot| {
            let (index, _) = table_slot(&stochastic.table, slot)?;
            let mut light = *ctx.lighting.local_light_record(index)?;
            // The hero/cone tier weights depend on camera distance, not on the light.
            light[28..32].fill(0);
            light[44..52].fill(0);
            Some(light)
        });
        let slots = (0..SLOTS)
            .filter(|&slot| lights[slot].is_some())
            .fold(0, |mask, slot| mask | (1u32 << slot));
        Self {
            stochastic,
            lights,
            slots,
            jitter,
        }
    }

    pub(super) fn same_light(&self, previous: &Self, slot: usize) -> bool {
        self.stochastic.generations[slot] == previous.stochastic.generations[slot]
            && self.lights[slot] == previous.lights[slot]
    }
}

impl AreaModelFrame {
    pub(super) fn prepare(
        model: AreaGraphModel,
        plan: &GraphPlan,
        h: &FeatureHandles<'_>,
    ) -> Result<Self, Issue> {
        let route = AreaRoute::from_plan(plan).ok_or_else(|| {
            issue(
                "graph.model_route",
                "the area model requires a direct stratified trace-to-mask link feeding forward",
            )
        })?;
        let (Some(lit), Some(gi)) = (h.lit.as_ref(), h.handles.gi.as_ref()) else {
            return Err(issue(
                "graph.model_route",
                "the area model requires the lit and GI features",
            ));
        };
        if h.viewports > 1 || !h.gpu.summary().is_hardware() {
            return Err(issue(
                "graph.model_device",
                "the area model requires a single view on a hardware adapter",
            ));
        }
        let resources = AreaResources::new(h.gpu, model.scene_size, &model.package)
            .map_err(|error| issue("graph.model_device", error.to_string()))?;
        Ok(Self {
            route,
            lit: lit.handle(),
            gi: gi.clone(),
            package: model.package,
            size: model.scene_size,
            resources: Some(resources),
            previous: None,
            pending: None,
        })
    }

    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let current = self.lit.area_stochastic();
        if current.mode != StochasticMode::Stratified || current.active == 0 {
            return Ok(());
        }
        let (Some(raw), Some(depth)) = (
            ctx.resources.get(&self.route.trace, "raw"),
            ctx.resources.get(&self.route.trace, "linearDepth"),
        ) else {
            return Ok(());
        };
        if self.size != ctx.size {
            self.size = ctx.size;
            self.previous = None;
            // A resize can exceed the device limits; the existing raw mask path remains usable.
            self.resources = AreaResources::new(ctx.gpu, ctx.size, &self.package).ok();
        }
        let Some(resources) = &self.resources else {
            return Ok(());
        };
        let current = AreaFrame::capture(current, self.lit.camera_jitter_ndc(), ctx);
        if current.slots == 0 {
            return Ok(());
        }
        let normal = ctx.targets.normal_input(ctx.resources, &self.route.trace);
        resources.record(
            ctx,
            &raw,
            &depth,
            &normal,
            &current,
            self.previous.as_ref(),
            self.gi.shared().camera_valid(),
        );
        ctx.resources.bind_input(
            &self.route.mask,
            "src",
            PublishedImage {
                view: resources.pairs.clone(),
                desc: raw.desc,
            },
        );
        self.pending = Some(current);
        Ok(())
    }
}

pub(super) struct SharedAreaFrame(pub Rc<RefCell<AreaModelFrame>>);

impl GraphFrame for SharedAreaFrame {
    fn begin(&mut self, _table: &ResourceTable) {
        self.0.borrow_mut().pending = None;
    }

    fn before_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let mut frame = self.0.borrow_mut();
        if frame.route.mask == pass {
            frame.record(ctx)?;
        }
        Ok(())
    }

    fn submitted(&mut self) {
        let mut frame = self.0.borrow_mut();
        frame.previous = frame.pending.take();
    }

    fn invalidate_history(&mut self) {
        let mut frame = self.0.borrow_mut();
        frame.pending = None;
        frame.previous = None;
    }
}

struct AreaResources {
    pilot: AreaPilot,
    rect: AreaPilotRect,
    params: [wgpu::Buffer; SLOTS],
    visibility: wgpu::Buffer,
    previous_raw: wgpu::TextureView,
    previous_depth: wgpu::TextureView,
    pairs: wgpu::TextureView,
    pack: wgpu::ComputePipeline,
    store: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
}

impl AreaResources {
    fn new(
        gpu: &GpuContext,
        size: (u32, u32),
        package: &AreaPilotPackage,
    ) -> Result<Self, RaaError> {
        let rect = AreaPilotRect::full(size.0, size.1);
        gpu.check_compute_limits(|limits| {
            rect.check_limits(limits)?;
            if limits.max_sampled_textures_per_shader_stage < 6
                || limits.max_storage_textures_per_shader_stage < 1
                || limits.max_bindings_per_bind_group < 11
                || limits.max_texture_array_layers < LAYERS
            {
                return Err(RaaError::Unsupported(
                    "area model bridge bindings exceed device limits".into(),
                ));
            }
            Ok(())
        })?;
        let pilot = AreaPilot::prepare_graph(gpu, rect, package)?;
        let device = gpu.device();
        let texture = |label, format, layers, usage| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: extent(size, layers),
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let history_usage = wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING;
        let previous_raw = texture(
            "area model previous pairs",
            RAW_FORMAT,
            LAYERS,
            history_usage,
        );
        let previous_depth = texture("area model previous depth", DEPTH_FORMAT, 1, history_usage);
        let pairs = texture(
            "area model resolved pairs",
            RAW_FORMAT,
            LAYERS,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let params = std::array::from_fn(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("area model slot parameters"),
                size: 24,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let visibility = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("area model visibility by slot"),
            size: u64::from(size.0) * u64::from(size.1) * SLOTS as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("area model graph bridge"),
            source: wgpu::ShaderSource::Wgsl(BRIDGE.into()),
        });
        use wgpu::{BufferBindingType as Buffer, TextureViewDimension as View};
        let buffer = |ty| wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let uniform = buffer(Buffer::Uniform);
        let read = buffer(Buffer::Storage { read_only: true });
        let write = buffer(Buffer::Storage { read_only: false });
        let sampled = |view_dimension| wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension,
            multisampled: false,
        };
        let pack = pipeline(
            device,
            &module,
            "pack",
            &[
                (0, sampled(View::D2Array)),
                (1, sampled(View::D2)),
                (2, sampled(View::D2)),
                (3, sampled(View::D2)),
                (4, sampled(View::D2Array)),
                (5, sampled(View::D2)),
                (6, uniform),
                (7, write),
            ],
        );
        let store = pipeline(
            device,
            &module,
            "store_visibility",
            &[(1, sampled(View::D2)), (6, uniform), (8, read), (9, write)],
        );
        let resolve = pipeline(
            device,
            &module,
            "resolve",
            &[
                (0, sampled(View::D2Array)),
                (6, uniform),
                (9, write),
                (
                    10,
                    wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: RAW_FORMAT,
                        view_dimension: View::D2Array,
                    },
                ),
            ],
        );
        Ok(Self {
            pilot,
            rect,
            params,
            visibility,
            previous_raw,
            previous_depth,
            pairs,
            pack,
            store,
            resolve,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn record(
        &self,
        ctx: &mut RecordCtx<'_>,
        raw: &PublishedImage,
        depth: &PublishedImage,
        normal: &wgpu::TextureView,
        current: &AreaFrame,
        previous: Option<&AreaFrame>,
        camera_valid: bool,
    ) {
        let [x, y] = self.rect.workgroups();
        let raw_view = wgpu::BindingResource::TextureView(&raw.view);
        let depth_view = wgpu::BindingResource::TextureView(&depth.view);
        let slots = current.slots;
        let history_offset = previous.map_or([0.0; 2], |old| {
            [
                (old.jitter[0] - current.jitter[0]) * 0.5,
                (old.jitter[1] - current.jitter[1]) * -0.5,
            ]
        });
        for slot in 0..SLOTS {
            if slots & (1 << slot) == 0 {
                continue;
            }
            let unchanged = previous.is_some_and(|old| current.same_light(old, slot));
            let bytes: Vec<_> = [
                slot as u32,
                u32::from(unchanged && camera_valid),
                u32::from(previous.is_some() && !unchanged),
                slots,
            ]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .chain(history_offset.into_iter().flat_map(f32::to_le_bytes))
            .collect();
            ctx.gpu.queue().write_buffer(&self.params[slot], 0, &bytes);
            let params = self.params[slot].as_entire_binding();
            dispatch(
                ctx,
                &self.pack,
                &[
                    (0, raw_view.clone()),
                    (1, depth_view.clone()),
                    (2, wgpu::BindingResource::TextureView(normal)),
                    (3, wgpu::BindingResource::TextureView(ctx.targets.motion())),
                    (4, wgpu::BindingResource::TextureView(&self.previous_raw)),
                    (5, wgpu::BindingResource::TextureView(&self.previous_depth)),
                    (6, params.clone()),
                    (7, self.pilot.input().as_entire_binding()),
                ],
                [x, y, 1],
            );
            self.pilot.record_real_frame(ctx.encoder, None);
            dispatch(
                ctx,
                &self.store,
                &[
                    (1, depth_view.clone()),
                    (6, params),
                    (8, self.pilot.output().as_entire_binding()),
                    (9, self.visibility.as_entire_binding()),
                ],
                [x, y, 1],
            );
        }
        let slot = slots.trailing_zeros() as usize;
        dispatch(
            ctx,
            &self.resolve,
            &[
                (0, raw_view),
                (6, self.params[slot].as_entire_binding()),
                (9, self.visibility.as_entire_binding()),
                (10, wgpu::BindingResource::TextureView(&self.pairs)),
            ],
            [x, y, raw.desc.layers],
        );
        for (image, previous) in [(raw, &self.previous_raw), (depth, &self.previous_depth)] {
            ctx.encoder.copy_texture_to_texture(
                image.view.texture().as_image_copy(),
                previous.texture().as_image_copy(),
                extent(ctx.size, image.desc.layers),
            );
        }
    }
}

fn extent((width, height): (u32, u32), depth_or_array_layers: u32) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers,
    }
}

fn pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    entry: &str,
    bindings: &[(u32, wgpu::BindingType)],
) -> wgpu::ComputePipeline {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(entry),
        entries: &bindings
            .iter()
            .map(|&(binding, ty)| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty,
                count: None,
            })
            .collect::<Vec<_>>(),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(entry),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(&layout),
        module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn dispatch(
    ctx: &mut RecordCtx<'_>,
    pipeline: &wgpu::ComputePipeline,
    bindings: &[(u32, wgpu::BindingResource<'_>)],
    [x, y, z]: [u32; 3],
) {
    let group = ctx
        .gpu
        .device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("area model bridge"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &bindings
                .iter()
                .map(|(binding, resource)| wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: resource.clone(),
                })
                .collect::<Vec<_>>(),
        });
    let mut pass = ctx
        .encoder
        .begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("area model bridge"),
            timestamp_writes: None,
        });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &group, &[]);
    pass.dispatch_workgroups(x, y, z);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn area_model_route_uses_the_direct_stratified_branch() {
        let source = include_str!("../../graphs/area-learned.json");
        let plan = crate::plan_from_json(source).expect("area pilot starter plans");
        let route = AreaRoute::from_plan(&plan).expect("area pilot route");
        assert_eq!(route.trace, "area.trace#trace");
        assert_eq!(route.mask, "area.mask#mask");

        let mut restir = plan;
        let mask = restir
            .nodes
            .iter_mut()
            .find(|node| node.id == "mask")
            .unwrap();
        let (_, mode) = mask
            .params
            .iter_mut()
            .find(|(key, _)| key == "stochasticMode")
            .unwrap();
        *mode = ParamValue::Enum("restir".into());
        assert!(AreaRoute::from_plan(&restir).is_none());

        let legacy =
            crate::plan_from_json(include_str!("../../graphs/full-gi-stages.json")).unwrap();
        assert!(AreaRoute::from_plan(&legacy).is_none());
    }
}
