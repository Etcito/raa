use raa_core::{
    BACKBUFFER, ClearPass, FrameGraph, PassBuilder, RaaError, RecordContext, RenderPass,
    ResourceId, TargetDesc, TargetSize,
};

struct TestPass {
    name: String,
    reads: Vec<ResourceId>,
    writes: Vec<ResourceId>,
}

impl TestPass {
    fn new(name: &str, reads: &[ResourceId], writes: &[ResourceId]) -> Self {
        Self {
            name: name.into(),
            reads: reads.to_vec(),
            writes: writes.to_vec(),
        }
    }
}

impl RenderPass for TestPass {
    fn name(&self) -> &str {
        &self.name
    }
    fn declare(&self, b: &mut PassBuilder) {
        for &r in &self.reads {
            b.read(r);
        }
        for &r in &self.writes {
            b.write(r);
        }
    }
    fn record(&mut self, _: &mut RecordContext<'_>) -> Result<(), RaaError> {
        Ok(())
    }
}

fn pass(name: &str, reads: &[ResourceId], writes: &[ResourceId]) -> Box<dyn RenderPass> {
    Box::new(TestPass::new(name, reads, writes))
}

fn graph() -> FrameGraph {
    FrameGraph::new(wgpu::TextureFormat::Bgra8Unorm)
}

fn target(g: &mut FrameGraph) -> ResourceId {
    g.create_target(TargetDesc {
        label: "test",
        format: wgpu::TextureFormat::Rgba8Unorm,
        size: TargetSize::Surface,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
    })
}

fn code(result: Result<(), RaaError>) -> &'static str {
    result.err().map_or("ok", |e| e.code())
}

#[test]
fn compile_reports_each_rule_error_code() {
    let mut duplicate = graph();
    duplicate.add_pass(pass("same", &[], &[BACKBUFFER]));
    duplicate.add_pass(pass("same", &[], &[BACKBUFFER]));
    assert_eq!(code(duplicate.compile()), "framegraph.duplicate_pass");

    let mut unknown = graph();
    unknown.add_pass(pass("bad", &[], &[ResourceId::from_raw(99)]));
    assert_eq!(code(unknown.compile()), "framegraph.unknown_resource");

    let mut before_write = graph();
    let r = target(&mut before_write);
    before_write.add_pass(pass("read", &[r], &[BACKBUFFER]));
    assert_eq!(code(before_write.compile()), "framegraph.read_before_write");

    let mut no_output = graph();
    no_output.add_pass(pass("offscreen", &[], &[]));
    assert_eq!(code(no_output.compile()), "framegraph.no_output");
}

#[test]
fn placement_reports_unknown_pass_for_both_relations() {
    let mut g = graph();
    assert_eq!(
        g.add_pass_after("missing", pass("x", &[], &[]))
            .unwrap_err()
            .code(),
        "framegraph.unknown_pass"
    );
    assert_eq!(
        g.add_pass_before("missing", pass("x", &[], &[]))
            .unwrap_err()
            .code(),
        "framegraph.unknown_pass"
    );
}

#[test]
fn placement_and_mutation_have_expected_order_and_compilation_state() {
    let mut g = graph();
    g.add_pass(pass("a", &[], &[BACKBUFFER]));
    g.add_pass(pass("c", &[], &[BACKBUFFER]));
    g.add_pass_after("a", pass("b", &[], &[BACKBUFFER]))
        .unwrap();
    assert_eq!(g.pass_names(), ["a", "b", "c"]);
    g.add_pass_before("a", pass("start", &[], &[BACKBUFFER]))
        .unwrap();
    assert_eq!(g.pass_names(), ["start", "a", "b", "c"]);
    assert!(g.compile().is_ok());
    assert!(g.is_compiled());
    g.add_pass(pass("tail", &[], &[BACKBUFFER]));
    assert!(!g.is_compiled());
}

#[test]
fn first_write_is_only_true_for_the_first_writer() {
    let mut g = graph();
    let r = target(&mut g);
    g.add_pass(pass("first", &[], &[r, BACKBUFFER]));
    g.add_pass(pass("second", &[], &[r, BACKBUFFER]));
    g.compile().unwrap();
    assert_eq!(g.first_write_of("first", r), Some(true));
    assert_eq!(g.first_write_of("second", r), Some(false));
}

#[test]
fn clear_pass_declares_backbuffer_write_and_compiles_alone() {
    let mut g = graph();
    g.add_pass(Box::new(ClearPass::new(
        "clear",
        BACKBUFFER,
        wgpu::Color::BLACK,
    )));
    assert_eq!(g.pass_names(), ["clear"]);
    assert!(g.compile().is_ok());
    assert_eq!(g.first_write_of("clear", BACKBUFFER), Some(true));
}
