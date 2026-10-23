//! Pipeline configuration with mock feature passes. CPU part: `PipelineConfig` (the
//! device-free core of `Pipeline`) lists passes and contributors in execution order,
//! enables/disables, overrides the order and refuses every invalid configuration with its
//! code and a non-empty message (Q12). GPU part (SKIP without a software adapter): `Pipeline`
//! hands the lighting resources to `set_enabled`, and its `raa.pipeline` graph pass resizes,
//! prepares and records the enabled passes in order with the frame time and the backbuffer.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use raa_core::RaaError;
use raa_pipeline::{
    FeaturePass, LightingResources, PassInfo, PassRole, PipelineConfig, PrepareCtx, RecordCtx,
    Targets,
};

/// Calls the mock passes made, in order.
type Log = Rc<RefCell<Vec<String>>>;

struct Mock {
    name: &'static str,
    order: u32,
    required: bool,
    log: Log,
}

impl Mock {
    fn boxed(name: &'static str, order: u32, required: bool, log: &Log) -> Box<Mock> {
        Box::new(Mock {
            name,
            order,
            required,
            log: Rc::clone(log),
        })
    }

    fn push(&self, entry: String) {
        self.log.borrow_mut().push(entry);
    }
}

impl FeaturePass for Mock {
    fn name(&self) -> &str {
        self.name
    }

    fn order(&self) -> u32 {
        self.order
    }

    fn required(&self) -> bool {
        self.required
    }

    fn resize(&mut self, _gpu: &raa_core::GpuContext, targets: &Targets) {
        let (w, h) = targets.size();
        self.push(format!("resize {} {w}x{h}", self.name));
    }

    fn set_enabled(&mut self, enabled: bool, lighting: &mut LightingResources) {
        if !enabled {
            lighting.clear_ao();
        }
        self.push(format!("set_enabled {} {enabled}", self.name));
    }

    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        self.push(format!(
            "prepare {} frame {} t {} dt {:.3}",
            self.name, ctx.frame, ctx.time_s, ctx.dt_s
        ));
        Ok(())
    }

    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        self.push(format!(
            "record {} {:?} {:?} srgb {}",
            self.name, ctx.size, ctx.backbuffer_format, ctx.surface_is_srgb
        ));
        Ok(())
    }
}

fn new_log() -> Log {
    Rc::new(RefCell::new(Vec::new()))
}

/// The milestone-1 required passes plus three optional ones, added out of order.
fn add_passes(config: &mut PipelineConfig, log: &Log) {
    let passes = [
        ("tonemap", 200, true),
        ("bloom", 120, false),
        ("forward", 60, true),
        ("shadow.csm", 20, false),
        ("prepass", 30, true),
        ("ao.gtao", 40, false),
    ];
    for (name, order, required) in passes {
        config
            .add(Mock::boxed(name, order, required, log))
            .expect("added");
    }
}

/// The three forward contributors of S12, added out of order; returns their flags.
fn add_contributors(config: &mut PipelineConfig) -> [Rc<Cell<bool>>; 3] {
    let flags = [(); 3].map(|_| Rc::new(Cell::new(true)));
    let entries = [
        ("sky.background", 900),
        ("fur.shells", 500),
        ("particles.opaque", 600),
    ];
    for ((name, order), flag) in entries.into_iter().zip(&flags) {
        config
            .add_contributor_entry(name, order, Rc::clone(flag))
            .expect("contributor added");
    }
    flags
}

fn full_config(log: &Log) -> (PipelineConfig, [Rc<Cell<bool>>; 3]) {
    let mut config = PipelineConfig::new();
    add_passes(&mut config, log);
    let flags = add_contributors(&mut config);
    (config, flags)
}

fn names(passes: &[PassInfo]) -> Vec<&str> {
    passes.iter().map(|p| p.name.as_str()).collect()
}

const DEFAULT_ORDER: [&str; 9] = [
    "shadow.csm",
    "prepass",
    "ao.gtao",
    "forward",
    "fur.shells",
    "particles.opaque",
    "sky.background",
    "bloom",
    "tonemap",
];

/// Asserts a refusal with `code` and a non-empty message (Q12).
fn assert_refused<T>(result: Result<T, RaaError>, code: &str) {
    let Err(err) = result else {
        panic!("expected a {code} refusal");
    };
    assert!(
        matches!(err, RaaError::FrameGraph { .. }),
        "FrameGraph variant: {err:?}"
    );
    assert_eq!(err.code(), code, "{err}");
    assert!(!err.message().is_empty(), "{code} has a message");
}

#[test]
fn passes_lists_entries_in_execution_order_with_contributors_after_forward() {
    let log = new_log();
    let (config, _flags) = full_config(&log);
    let passes = config.passes();
    assert_eq!(names(&passes), DEFAULT_ORDER);
    let forward = passes
        .iter()
        .position(|p| p.name == "forward")
        .expect("forward");
    for (i, info) in passes.iter().enumerate() {
        let contributor = (forward + 1..forward + 4).contains(&i);
        let role = if contributor {
            PassRole::Contributor
        } else {
            PassRole::Pass
        };
        assert_eq!(info.role, role, "{}", info.name);
        assert!(info.enabled, "{} starts enabled", info.name);
        let required = ["prepass", "forward", "tonemap"].contains(&info.name.as_str());
        assert_eq!(info.required, required, "{}", info.name);
    }
    assert_eq!(
        passes[4],
        PassInfo {
            name: "fur.shells".into(),
            role: PassRole::Contributor,
            enabled: true,
            order: 500,
            required: false,
        }
    );
    assert_eq!(passes[0].order, 20);
    assert_eq!(passes[8].order, 200);
}

#[test]
fn contributors_are_listed_last_while_forward_is_absent() {
    let log = new_log();
    let mut config = PipelineConfig::new();
    add_contributors(&mut config);
    config
        .add(Mock::boxed("bloom", 120, false, &log))
        .expect("added");
    let passes = config.passes();
    assert_eq!(
        names(&passes),
        ["bloom", "fur.shells", "particles.opaque", "sky.background"]
    );
    assert_eq!(passes[0].role, PassRole::Pass);
    assert!(passes[1..].iter().all(|p| p.role == PassRole::Contributor));
}

#[test]
fn equal_orders_keep_the_add_sequence() {
    let log = new_log();
    let mut config = PipelineConfig::new();
    for name in ["b", "a", "c"] {
        config
            .add(Mock::boxed(name, 50, false, &log))
            .expect("added");
    }
    config
        .add(Mock::boxed("first", 10, false, &log))
        .expect("added");
    assert_eq!(names(&config.passes()), ["first", "b", "a", "c"]);
}

#[test]
fn disabling_and_enabling_a_pass_returns_it_and_updates_passes() {
    let log = new_log();
    let (mut config, _flags) = full_config(&log);
    let pass = config.set_enabled("bloom", false).expect("disabled");
    assert_eq!(pass.map(|p| p.name().to_owned()).as_deref(), Some("bloom"));
    let bloom = |c: &PipelineConfig| c.passes().into_iter().find(|p| p.name == "bloom");
    assert_eq!(bloom(&config).map(|p| p.enabled), Some(false));
    assert_eq!(names(&config.passes()), DEFAULT_ORDER, "order unchanged");

    let pass = config.set_enabled("bloom", true).expect("enabled");
    assert!(pass.is_some());
    assert_eq!(bloom(&config).map(|p| p.enabled), Some(true));

    // enabling a required pass is allowed (it is always enabled)
    assert!(config.set_enabled("forward", true).expect("ok").is_some());
}

#[test]
fn contributor_entries_toggle_their_shared_flag() {
    let log = new_log();
    let (mut config, [sky, _fur, _particles]) = full_config(&log);
    let result = config
        .set_enabled("sky.background", false)
        .expect("disabled");
    assert!(result.is_none(), "a contributor has no pass object");
    assert!(!sky.get(), "the owner's flag follows set_enabled");
    let info = |c: &PipelineConfig| c.passes().into_iter().find(|p| p.name == "sky.background");
    assert_eq!(
        info(&config).map(|p| (p.enabled, p.role)),
        Some((false, PassRole::Contributor))
    );

    // the owner changing its flag is what passes() reports
    sky.set(true);
    assert_eq!(info(&config).map(|p| p.enabled), Some(true));
}

#[test]
fn set_order_overrides_the_execution_order() {
    let log = new_log();
    let (mut config, _flags) = full_config(&log);
    let order = [
        "prepass",
        "shadow.csm",
        "forward",
        "bloom",
        "ao.gtao",
        "tonemap",
    ];
    config.set_order(&order).expect("valid order");
    assert_eq!(
        names(&config.passes()),
        [
            "prepass",
            "shadow.csm",
            "forward",
            "fur.shells",
            "particles.opaque",
            "sky.background",
            "bloom",
            "ao.gtao",
            "tonemap",
        ]
    );
    // declared orders are reported unchanged
    let shadow = config.passes().into_iter().find(|p| p.name == "shadow.csm");
    assert_eq!(shadow.map(|p| p.order), Some(20));

    // a pass added after an override goes before the first pass of greater order
    config
        .add(Mock::boxed("sky.env", 10, false, &log))
        .expect("added");
    config
        .add(Mock::boxed("opaque.copy", 70, false, &log))
        .expect("added");
    assert_eq!(
        names(&config.passes()),
        [
            "sky.env",
            "prepass",
            "shadow.csm",
            "forward",
            "fur.shells",
            "particles.opaque",
            "sky.background",
            "opaque.copy",
            "bloom",
            "ao.gtao",
            "tonemap",
        ]
    );
}

#[test]
fn duplicate_names_are_refused_everywhere() {
    let log = new_log();
    let (mut config, _flags) = full_config(&log);
    assert_refused(
        config.add(Mock::boxed("bloom", 121, false, &log)),
        "pipeline.duplicate_pass",
    );
    assert_refused(
        config.add(Mock::boxed("sky.background", 30, false, &log)),
        "pipeline.duplicate_pass",
    );
    assert_refused(
        config.add_contributor_entry("forward", 700, Rc::new(Cell::new(true))),
        "pipeline.duplicate_pass",
    );
    assert_refused(
        config.add_contributor_entry("fur.shells", 700, Rc::new(Cell::new(true))),
        "pipeline.duplicate_pass",
    );
    let doubled = [
        "shadow.csm",
        "prepass",
        "bloom",
        "ao.gtao",
        "forward",
        "bloom",
        "tonemap",
    ];
    assert_refused(config.set_order(&doubled), "pipeline.duplicate_pass");
    assert_eq!(names(&config.passes()), DEFAULT_ORDER, "nothing changed");
}

#[test]
fn unknown_names_are_refused() {
    let log = new_log();
    let (mut config, _flags) = full_config(&log);
    assert_refused(config.set_enabled("ssr", false), "pipeline.unknown_pass");
    assert_refused(config.set_enabled("", true), "pipeline.unknown_pass");
    let unknown = [
        "shadow.csm",
        "prepass",
        "ao.gtao",
        "forward",
        "ssr",
        "tonemap",
    ];
    assert_refused(config.set_order(&unknown), "pipeline.unknown_pass");
    // contributors keep their order inside forward: they are not passes of the order
    let with_contributor = [
        "shadow.csm",
        "prepass",
        "ao.gtao",
        "forward",
        "sky.background",
        "bloom",
        "tonemap",
    ];
    assert_refused(config.set_order(&with_contributor), "pipeline.unknown_pass");
    assert_eq!(names(&config.passes()), DEFAULT_ORDER, "nothing changed");
}

#[test]
fn required_passes_cannot_be_disabled() {
    let log = new_log();
    let (mut config, _flags) = full_config(&log);
    for name in ["prepass", "forward", "tonemap"] {
        assert_refused(config.set_enabled(name, false), "pipeline.required_pass");
    }
    assert!(config.passes().iter().all(|p| p.enabled));
}

#[test]
fn an_order_missing_a_pass_is_incomplete() {
    let log = new_log();
    let (mut config, _flags) = full_config(&log);
    let missing = ["shadow.csm", "prepass", "forward", "bloom", "tonemap"];
    let err = config.set_order(&missing).expect_err("refused");
    assert_eq!(err.code(), "pipeline.order_incomplete");
    assert!(
        err.message().contains("ao.gtao"),
        "names the missing pass: {err}"
    );
    assert_refused(config.set_order(&[]), "pipeline.order_incomplete");
    assert_eq!(names(&config.passes()), DEFAULT_ORDER, "nothing changed");
}

#[test]
fn prepass_forward_tonemap_keep_their_order_and_tonemap_runs_last() {
    let log = new_log();
    let (mut config, _flags) = full_config(&log);
    let refused: [[&str; 6]; 3] = [
        // forward before prepass
        [
            "shadow.csm",
            "forward",
            "prepass",
            "ao.gtao",
            "bloom",
            "tonemap",
        ],
        // tonemap before forward
        [
            "shadow.csm",
            "prepass",
            "ao.gtao",
            "tonemap",
            "forward",
            "bloom",
        ],
        // a pass after tonemap
        [
            "shadow.csm",
            "prepass",
            "ao.gtao",
            "forward",
            "tonemap",
            "bloom",
        ],
    ];
    for order in refused {
        assert_refused(config.set_order(&order), "pipeline.required_order");
    }
    assert_eq!(names(&config.passes()), DEFAULT_ORDER, "nothing changed");

    // a pass whose order would place it after tonemap is refused when added
    assert_refused(
        config.add(Mock::boxed("late", 300, false, &log)),
        "pipeline.required_order",
    );
    // and so is a prepass ordered after forward
    let mut other = PipelineConfig::new();
    other
        .add(Mock::boxed("forward", 60, true, &log))
        .expect("added");
    assert_refused(
        other.add(Mock::boxed("prepass", 70, true, &log)),
        "pipeline.required_order",
    );
    assert_eq!(names(&other.passes()), ["forward"]);
}

#[cfg(not(target_arch = "wasm32"))]
mod gpu {
    use super::*;
    use raa_core::testing::{block_on, skip, software_gpu};
    use raa_core::{FrameGraph, GpuContext, OffscreenTarget, wgpu};
    use raa_pipeline::Pipeline;

    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

    /// `forward` mock: logs whether the HDR destination is the one of the first frame at this
    /// size (the chain is reset every frame), then swaps the chain in its record.
    struct Forward {
        inner: Mock,
        first_dst: Option<wgpu::Texture>,
    }

    impl FeaturePass for Forward {
        fn name(&self) -> &str {
            self.inner.name()
        }

        fn order(&self) -> u32 {
            self.inner.order()
        }

        fn required(&self) -> bool {
            true
        }

        fn resize(&mut self, gpu: &GpuContext, targets: &Targets) {
            self.first_dst = None;
            self.inner.resize(gpu, targets);
        }

        fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
            let dst = ctx.targets.hdr().dst_texture();
            let reset = self.first_dst.get_or_insert_with(|| dst.clone()) == dst;
            self.inner.push(format!("hdr reset {reset}"));
            self.inner.prepare(ctx)
        }

        fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
            ctx.targets.hdr().swap();
            self.inner.record(ctx)
        }
    }

    /// `tonemap` mock: clears the backbuffer to green.
    struct Tonemap {
        inner: Mock,
    }

    impl FeaturePass for Tonemap {
        fn name(&self) -> &str {
            self.inner.name()
        }

        fn order(&self) -> u32 {
            self.inner.order()
        }

        fn required(&self) -> bool {
            true
        }

        fn resize(&mut self, gpu: &GpuContext, targets: &Targets) {
            self.inner.resize(gpu, targets);
        }

        fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
            self.inner.prepare(ctx)
        }

        fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
            drop(ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tonemap-mock"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: ctx.backbuffer,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            }));
            self.inner.record(ctx)
        }
    }

    /// A pass whose prepare fails.
    struct Failing;

    impl FeaturePass for Failing {
        fn name(&self) -> &str {
            "failing"
        }

        fn order(&self) -> u32 {
            100
        }

        fn prepare(&mut self, _ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
            Err(RaaError::Gpu("mock prepare failed".into()))
        }

        fn record(&mut self, _ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
            Ok(())
        }
    }

    fn mock(name: &'static str, order: u32, log: &Log) -> Mock {
        Mock {
            name,
            order,
            required: false,
            log: Rc::clone(log),
        }
    }

    fn take(log: &Log) -> Vec<String> {
        log.borrow_mut().drain(..).collect()
    }

    fn frame(gpu: &GpuContext, graph: &mut FrameGraph, target: &OffscreenTarget) {
        let commands = graph
            .execute(gpu, target.view(), target.size())
            .expect("frame runs");
        gpu.queue().submit([commands]);
    }

    fn expected(lines: &[&str]) -> Vec<String> {
        lines.iter().map(|l| l.to_string()).collect()
    }

    fn pipeline_with_mocks(gpu: &GpuContext, log: &Log) -> (Rc<RefCell<Pipeline>>, Rc<Cell<bool>>) {
        let mut pipeline = Pipeline::new(gpu, FORMAT, false);
        let forward = Forward {
            inner: mock("forward", 60, log),
            first_dst: None,
        };
        let tonemap = Tonemap {
            inner: mock("tonemap", 200, log),
        };
        let passes: [Box<dyn FeaturePass>; 5] = [
            Box::new(tonemap),
            Box::new(mock("bloom", 120, log)),
            Box::new(forward),
            Mock::boxed("prepass", 30, true, log),
            Box::new(mock("ao.gtao", 40, log)),
        ];
        for pass in passes {
            pipeline.add(pass).expect("added");
        }
        let sky = Rc::new(Cell::new(true));
        pipeline
            .add_contributor_entry("sky.background", 900, Rc::clone(&sky))
            .expect("contributor added");
        (Rc::new(RefCell::new(pipeline)), sky)
    }

    fn configuration_goes_through_the_pipeline(gpu: &GpuContext) {
        let log = new_log();
        let (pipeline, sky) = pipeline_with_mocks(gpu, &log);
        let mut p = pipeline.borrow_mut();
        assert!(p.targets().is_none(), "targets come with the first prepare");
        assert_eq!((p.surface_format(), p.surface_is_srgb()), (FORMAT, false));
        assert_eq!(
            names(&p.passes()),
            [
                "prepass",
                "ao.gtao",
                "forward",
                "sky.background",
                "bloom",
                "tonemap"
            ]
        );

        p.set_enabled("ao.gtao", false).expect("disabled");
        assert_eq!(take(&log), ["set_enabled ao.gtao false"]);
        p.set_enabled("sky.background", false).expect("disabled");
        assert!(!sky.get());
        assert!(take(&log).is_empty(), "a contributor has no pass to call");

        assert_refused(p.set_enabled("forward", false), "pipeline.required_pass");
        assert_refused(p.set_enabled("nope", false), "pipeline.unknown_pass");
        assert_refused(
            p.add(Box::new(mock("bloom", 1, &log))),
            "pipeline.duplicate_pass",
        );
        assert_refused(
            p.add_contributor_entry("tonemap", 1, Rc::new(Cell::new(true))),
            "pipeline.duplicate_pass",
        );
        assert_refused(p.set_order(&["prepass"]), "pipeline.order_incomplete");
        assert_refused(
            p.set_order(&["forward", "prepass", "ao.gtao", "bloom", "tonemap"]),
            "pipeline.required_order",
        );
        assert!(take(&log).is_empty(), "refusals call no pass");

        let infos = p.passes();
        let enabled: Vec<(&str, bool)> =
            infos.iter().map(|i| (i.name.as_str(), i.enabled)).collect();
        assert_eq!(
            enabled,
            [
                ("prepass", true),
                ("ao.gtao", false),
                ("forward", true),
                ("sky.background", false),
                ("bloom", true),
                ("tonemap", true),
            ]
        );
    }

    fn graph_pass_runs_the_enabled_passes_in_order(gpu: &GpuContext) {
        let scope = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
        let log = new_log();
        let (pipeline, _sky) = pipeline_with_mocks(gpu, &log);
        pipeline
            .borrow_mut()
            .set_enabled("ao.gtao", false)
            .expect("disabled");
        take(&log);

        let mut graph = FrameGraph::new(FORMAT);
        Pipeline::install(&pipeline, &mut graph);
        graph.compile().expect("compiles");
        assert_eq!(graph.pass_names(), ["raa.pipeline"]);

        // frame 0: every enabled pass is resized, prepared, then recorded, in order
        let target = OffscreenTarget::new(gpu, 32, 16, FORMAT);
        pipeline.borrow_mut().set_time(1000.0);
        frame(gpu, &mut graph, &target);
        let rec = "(32, 16) Rgba8Unorm srgb false";
        assert_eq!(
            take(&log),
            expected(&[
                "resize prepass 32x16",
                "prepare prepass frame 0 t 1 dt 0.000",
                "resize forward 32x16",
                "hdr reset true",
                "prepare forward frame 0 t 1 dt 0.000",
                "resize bloom 32x16",
                "prepare bloom frame 0 t 1 dt 0.000",
                "resize tonemap 32x16",
                "prepare tonemap frame 0 t 1 dt 0.000",
                &format!("record prepass {rec}"),
                &format!("record forward {rec}"),
                &format!("record bloom {rec}"),
                &format!("record tonemap {rec}"),
            ])
        );
        let size = pipeline.borrow().targets().map(|t| t.size());
        assert_eq!(size, Some((32, 16)));
        let pixels = target.read_rgba8(gpu).expect("readback");
        assert!(
            pixels.chunks_exact(4).all(|px| px == [0, 255, 0, 255]),
            "tonemap mock cleared the backbuffer"
        );

        // frame 1: the time delta, no resize, the HDR chain reset after forward's swap
        pipeline.borrow_mut().set_time(1016.0);
        frame(gpu, &mut graph, &target);
        let lines = take(&log);
        assert_eq!(lines.len(), 9, "{lines:?}");
        assert_eq!(lines[0], "prepare prepass frame 1 t 1.016 dt 0.016");
        assert_eq!(lines[1], "hdr reset true");

        // a pass enabled later is resized before its first prepare
        pipeline
            .borrow_mut()
            .set_enabled("ao.gtao", true)
            .expect("enabled");
        assert_eq!(take(&log), ["set_enabled ao.gtao true"]);
        frame(gpu, &mut graph, &target);
        let lines = take(&log);
        assert_eq!(lines[1], "resize ao.gtao 32x16", "{lines:?}");
        // dt_s is the delta of the latest set_time
        assert_eq!(lines[2], "prepare ao.gtao frame 2 t 1.016 dt 0.016");

        // a new size recreates the targets and resizes every enabled pass
        let small = OffscreenTarget::new(gpu, 16, 8, FORMAT);
        frame(gpu, &mut graph, &small);
        let resized: Vec<String> = take(&log)
            .into_iter()
            .filter(|l| l.starts_with("resize"))
            .collect();
        assert_eq!(
            resized,
            expected(&[
                "resize prepass 16x8",
                "resize ao.gtao 16x8",
                "resize forward 16x8",
                "resize bloom 16x8",
                "resize tonemap 16x8",
            ])
        );
        let size = pipeline.borrow().targets().map(|t| t.size());
        assert_eq!(size, Some((16, 8)));

        // an order override changes the record order
        pipeline
            .borrow_mut()
            .set_order(&["prepass", "forward", "bloom", "ao.gtao", "tonemap"])
            .expect("valid order");
        frame(gpu, &mut graph, &small);
        let records: Vec<String> = take(&log)
            .into_iter()
            .filter_map(|l| {
                l.strip_prefix("record ")
                    .map(|r| r.split(' ').next().unwrap_or("").to_string())
            })
            .collect();
        assert_eq!(
            records,
            ["prepass", "forward", "bloom", "ao.gtao", "tonemap"]
        );

        gpu.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let error = block_on(scope.pop());
        assert!(error.is_none(), "validation error: {error:?}");
        // the pipeline holds Rgba16Float targets: leaked with the device (see below)
        std::mem::forget(graph);
        std::mem::forget(pipeline);
    }

    fn a_failing_pass_fails_the_frame(gpu: &GpuContext) {
        let log = new_log();
        let (pipeline, _sky) = pipeline_with_mocks(gpu, &log);
        pipeline.borrow_mut().add(Box::new(Failing)).expect("added");
        let mut graph = FrameGraph::new(FORMAT);
        Pipeline::install(&pipeline, &mut graph);
        graph.compile().expect("compiles");
        let target = OffscreenTarget::new(gpu, 8, 8, FORMAT);
        let err = graph
            .execute(gpu, target.view(), target.size())
            .expect_err("the failing prepare stops the frame");
        assert_eq!(err.code(), "core.gpu");
        assert!(
            take(&log).iter().all(|l| !l.starts_with("record")),
            "nothing records after a failed prepare"
        );
        std::mem::forget(graph);
        std::mem::forget(pipeline);
    }

    /// One GPU context for every check, leaked like in tests/targets.rs (the GL backend of
    /// the WSL driver crashed at exit after dropping a device that created Rgba16Float
    /// textures).
    #[test]
    fn pipeline_on_the_software_gpu() {
        let Some(gpu) = software_gpu() else {
            skip(
                "pipeline_on_the_software_gpu",
                "software_gpu returned None; see the software_gpu line above",
            );
            return;
        };
        configuration_goes_through_the_pipeline(&gpu);
        graph_pass_runs_the_enabled_passes_in_order(&gpu);
        a_failing_pass_fails_the_frame(&gpu);
        std::mem::forget(gpu);
    }
}
