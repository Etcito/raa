//! A minimal frame graph: passes declare the resources they read and write,
//! [`FrameGraph::compile`] validates the declarations once, and [`FrameGraph::execute`] records
//! every pass in order into one command encoder.

use crate::{GpuContext, RaaError};

/// A frame graph resource: [`BACKBUFFER`] or a target made by [`FrameGraph::create_target`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceId(u32);

/// The backbuffer (id 0), imported each frame as the view given to [`FrameGraph::execute`].
pub const BACKBUFFER: ResourceId = ResourceId(0);

impl ResourceId {
    /// Creates a resource id from its numeric graph id (0 is [`BACKBUFFER`]).
    pub const fn from_raw(id: u32) -> Self {
        Self(id)
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

/// How big a graph-owned target is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetSize {
    /// Follows the size given to [`FrameGraph::execute`] and is recreated when it changes.
    Surface,
    /// Fixed width and height in pixels (zero clamps to 1).
    Fixed(u32, u32),
}

/// Description of a 2D texture owned by the frame graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetDesc {
    /// Debug label of the texture, also used in error messages.
    pub label: &'static str,
    /// Texture format.
    pub format: wgpu::TextureFormat,
    /// Surface-sized or fixed.
    pub size: TargetSize,
    /// Texture usages (`RENDER_ATTACHMENT` for a target a pass draws into).
    pub usage: wgpu::TextureUsages,
}

/// Identifies a pass added to a [`FrameGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PassId(u32);

/// Collects the resources a pass declares in [`RenderPass::declare`].
#[derive(Debug)]
pub struct PassBuilder {
    reads: Vec<ResourceId>,
    writes: Vec<ResourceId>,
}

impl PassBuilder {
    fn new() -> Self {
        PassBuilder {
            reads: Vec::new(),
            writes: Vec::new(),
        }
    }

    /// Declares that the pass reads `r` (samples or loads it).
    pub fn read(&mut self, r: ResourceId) {
        self.reads.push(r);
    }

    /// Declares that the pass writes `r` (renders into it).
    pub fn write(&mut self, r: ResourceId) {
        self.writes.push(r);
    }

    /// Empties the declarations and keeps the capacity.
    fn clear(&mut self) {
        self.reads.clear();
        self.writes.clear();
    }
}

/// One step of a frame: declares its resources once, then prepares and records every frame.
pub trait RenderPass {
    /// Name of the pass, unique in its graph (used for placement and in error messages).
    fn name(&self) -> &str;

    /// Declares every resource the pass reads and writes; called by [`FrameGraph::compile`].
    fn declare(&self, builder: &mut PassBuilder);

    /// CPU work and uploads, run for every pass before any pass records. Default: nothing.
    fn prepare(&mut self, _ctx: &PrepareContext<'_>) -> Result<(), RaaError> {
        Ok(())
    }

    /// Records the pass's commands into `ctx.encoder`.
    fn record(&mut self, ctx: &mut RecordContext<'_>) -> Result<(), RaaError>;

    /// Called on the pass's first frame and whenever the size changes, after Surface-sized
    /// targets are recreated. Default: nothing.
    fn resize(&mut self, _gpu: &GpuContext, _size: (u32, u32)) {}
}

/// What [`RenderPass::prepare`] receives.
#[derive(Debug, Clone, Copy)]
pub struct PrepareContext<'a> {
    /// Device and queue.
    pub gpu: &'a GpuContext,
    /// Frame size in pixels.
    pub size: (u32, u32),
    /// Frame index, 0 for the first executed frame.
    pub frame: u64,
}

/// What [`RenderPass::record`] receives: the shared encoder and the graph's resources.
pub struct RecordContext<'a> {
    /// Device and queue.
    pub gpu: &'a GpuContext,
    /// The one encoder every pass of the frame records into.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// Frame size in pixels.
    pub size: (u32, u32),
    /// Frame index, 0 for the first executed frame.
    pub frame: u64,
    backbuffer: &'a wgpu::TextureView,
    backbuffer_format: wgpu::TextureFormat,
    targets: &'a [Target],
    first_writes: &'a [bool],
}

impl<'a> RecordContext<'a> {
    /// The view of `r`: the backbuffer view for [`BACKBUFFER`], else the graph-owned target view.
    /// The view borrows the graph, not the context, so `ctx.encoder` stays usable next to it.
    pub fn view(&self, r: ResourceId) -> Result<&'a wgpu::TextureView, RaaError> {
        if r == BACKBUFFER {
            return Ok(self.backbuffer);
        }
        let targets: &'a [Target] = self.targets;
        target_of(targets, r)
            .and_then(|t| t.allocated.as_ref())
            .map(|(_, view)| view)
            .ok_or_else(|| {
                let what = resource_name(targets, r);
                graph_error(
                    "framegraph.unknown_resource",
                    format!("{what} has no view in this frame graph"),
                )
            })
    }

    /// The format of `r`. An id unknown to this graph reports the backbuffer format.
    pub fn format(&self, r: ResourceId) -> wgpu::TextureFormat {
        match target_of(self.targets, r) {
            Some(t) => t.desc.format,
            None => self.backbuffer_format,
        }
    }

    /// True when this pass is the first of the frame to write `r`: it must clear `r` then, and
    /// load it otherwise.
    pub fn is_first_write(&self, r: ResourceId) -> bool {
        self.first_writes.get(r.index()) == Some(&true)
    }
}

/// A graph-owned target and, once [`FrameGraph::execute`] has run, its texture and view.
struct Target {
    desc: TargetDesc,
    allocated: Option<(wgpu::Texture, wgpu::TextureView)>,
}

impl Target {
    fn allocate(&mut self, gpu: &GpuContext, surface_size: (u32, u32)) {
        let (width, height) = match self.desc.size {
            TargetSize::Surface => surface_size,
            TargetSize::Fixed(w, h) => (w.max(1), h.max(1)),
        };
        let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some(self.desc.label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.desc.format,
            usage: self.desc.usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.allocated = Some((texture, view));
    }
}

/// The target behind `r`; `None` for [`BACKBUFFER`] and unknown ids.
fn target_of(targets: &[Target], r: ResourceId) -> Option<&Target> {
    r.index().checked_sub(1).and_then(|i| targets.get(i))
}

/// How error messages name a resource.
fn resource_name(targets: &[Target], r: ResourceId) -> String {
    if r == BACKBUFFER {
        return "BACKBUFFER".to_string();
    }
    match target_of(targets, r) {
        Some(t) => format!("target `{}` (resource {})", t.desc.label, r.0),
        None => format!("unknown resource {}", r.0),
    }
}

fn graph_error(code: &'static str, message: String) -> RaaError {
    RaaError::FrameGraph { code, message }
}

/// A pass with its reusable declaration and first-write buffers.
struct PassSlot {
    pass: Box<dyn RenderPass>,
    decl: PassBuilder,
    /// Indexed by resource id: true when this pass is the first writer of the resource.
    first_writes: Vec<bool>,
    /// True until the pass has received its first `resize`.
    needs_resize: bool,
}

/// Passes in execution order plus the targets they use. Build it, [`compile`](Self::compile)
/// it, then [`execute`](Self::execute) it every frame.
pub struct FrameGraph {
    backbuffer_format: wgpu::TextureFormat,
    targets: Vec<Target>,
    /// Kept in execution order: placement inserts in place, so compile never reorders.
    passes: Vec<PassSlot>,
    next_pass: u32,
    compiled: bool,
    /// Compile scratch, indexed by resource id: written by an earlier pass.
    written: Vec<bool>,
    size: Option<(u32, u32)>,
    frame: u64,
}

impl FrameGraph {
    /// An empty graph whose [`BACKBUFFER`] has `backbuffer_format`.
    pub fn new(backbuffer_format: wgpu::TextureFormat) -> Self {
        FrameGraph {
            backbuffer_format,
            targets: Vec::new(),
            passes: Vec::new(),
            next_pass: 0,
            compiled: false,
            written: Vec::new(),
            size: None,
            frame: 0,
        }
    }

    /// Registers a graph-owned target. Its texture is created by [`execute`](Self::execute).
    pub fn create_target(&mut self, desc: TargetDesc) -> ResourceId {
        self.targets.push(Target {
            desc,
            allocated: None,
        });
        // Target ids start at 1 because BACKBUFFER is 0.
        ResourceId(self.targets.len() as u32)
    }

    /// Appends `pass` at the end of the order.
    pub fn add_pass(&mut self, pass: Box<dyn RenderPass>) -> PassId {
        self.insert_pass(self.passes.len(), pass)
    }

    /// Inserts `pass` right after the pass named `after` (`framegraph.unknown_pass` otherwise).
    pub fn add_pass_after(
        &mut self,
        after: &str,
        pass: Box<dyn RenderPass>,
    ) -> Result<PassId, RaaError> {
        let at = self.position_of(after, "after", pass.as_ref())?;
        Ok(self.insert_pass(at + 1, pass))
    }

    /// Inserts `pass` right before the pass named `before` (`framegraph.unknown_pass` otherwise).
    pub fn add_pass_before(
        &mut self,
        before: &str,
        pass: Box<dyn RenderPass>,
    ) -> Result<PassId, RaaError> {
        let at = self.position_of(before, "before", pass.as_ref())?;
        Ok(self.insert_pass(at, pass))
    }

    /// Validates the declarations and caches the first-write flags. Rules, in order: unique pass
    /// names (`framegraph.duplicate_pass`), known resources (`framegraph.unknown_resource`), every
    /// read of a target written by an earlier pass (`framegraph.read_before_write`), and at least
    /// one pass writing [`BACKBUFFER`] (`framegraph.no_output`).
    pub fn compile(&mut self) -> Result<(), RaaError> {
        self.compiled = false;
        let resource_count = self.targets.len() + 1;
        for slot in &mut self.passes {
            slot.decl.clear();
            slot.pass.declare(&mut slot.decl);
        }

        for (i, slot) in self.passes.iter().enumerate() {
            let name = slot.pass.name();
            if self.passes.iter().take(i).any(|s| s.pass.name() == name) {
                return Err(graph_error(
                    "framegraph.duplicate_pass",
                    format!("pass `{name}` is added more than once"),
                ));
            }
        }

        for slot in &self.passes {
            let decl = &slot.decl;
            for &r in decl.reads.iter().chain(&decl.writes) {
                if r.index() >= resource_count {
                    let name = slot.pass.name();
                    let what = resource_name(&self.targets, r);
                    return Err(graph_error(
                        "framegraph.unknown_resource",
                        format!("pass `{name}` uses {what}"),
                    ));
                }
            }
        }

        // From here every declared id is below resource_count (checked above), and both
        // `written` and `first_writes` hold resource_count flags, so indexing cannot panic.
        self.written.clear();
        self.written.resize(resource_count, false);
        for slot in &mut self.passes {
            for &r in &slot.decl.reads {
                if r != BACKBUFFER && !self.written[r.index()] {
                    let name = slot.pass.name();
                    let what = resource_name(&self.targets, r);
                    return Err(graph_error(
                        "framegraph.read_before_write",
                        format!("pass `{name}` reads {what} with no earlier writer"),
                    ));
                }
            }
            slot.first_writes.clear();
            slot.first_writes.resize(resource_count, false);
            for &w in &slot.decl.writes {
                if !self.written[w.index()] {
                    self.written[w.index()] = true;
                    slot.first_writes[w.index()] = true;
                }
            }
        }

        if !self.written[BACKBUFFER.index()] {
            let passes = self.pass_names().join(", ");
            return Err(graph_error(
                "framegraph.no_output",
                format!("no pass writes BACKBUFFER (passes: {passes})"),
            ));
        }
        self.compiled = true;
        Ok(())
    }

    /// Pass names in execution order.
    pub fn pass_names(&self) -> Vec<&str> {
        self.passes.iter().map(|s| s.pass.name()).collect()
    }

    /// Whether the current pass list has passed [`Self::compile`].
    pub fn is_compiled(&self) -> bool {
        self.compiled
    }

    /// Whether `pass` is the first writer of resource `r`; `None` if pass is unknown or uncompiled.
    pub fn first_write_of(&self, pass: &str, r: ResourceId) -> Option<bool> {
        if !self.compiled || r.index() > self.targets.len() {
            return None;
        }
        self.passes
            .iter()
            .find(|slot| slot.pass.name() == pass)
            .map(|slot| slot.first_writes.get(r.index()).copied().unwrap_or(false))
    }

    /// Runs one frame into `backbuffer` and returns the command buffer to submit. Surface-sized
    /// targets are (re)created and passes resized on the first frame and on a size change (zero
    /// sizes clamp to 1); then every pass prepares, then every pass records in order into one
    /// encoder. Gives `framegraph.not_compiled` when the graph is not compiled.
    pub fn execute(
        &mut self,
        gpu: &GpuContext,
        backbuffer: &wgpu::TextureView,
        size: (u32, u32),
    ) -> Result<wgpu::CommandBuffer, RaaError> {
        if !self.compiled {
            return Err(graph_error(
                "framegraph.not_compiled",
                "execute called before compile".to_string(),
            ));
        }
        let size = (size.0.max(1), size.1.max(1));
        let resized = self.size != Some(size);
        for target in &mut self.targets {
            if target.allocated.is_none() || (resized && target.desc.size == TargetSize::Surface) {
                target.allocate(gpu, size);
            }
        }
        self.size = Some(size);
        for slot in &mut self.passes {
            if resized || slot.needs_resize {
                slot.pass.resize(gpu, size);
                slot.needs_resize = false;
            }
        }

        let prepare = PrepareContext {
            gpu,
            size,
            frame: self.frame,
        };
        for slot in &mut self.passes {
            slot.pass.prepare(&prepare)?;
        }

        let desc = wgpu::CommandEncoderDescriptor {
            label: Some("raa-frame-graph"),
        };
        let mut encoder = gpu.device().create_command_encoder(&desc);
        for slot in &mut self.passes {
            let mut ctx = RecordContext {
                gpu,
                encoder: &mut encoder,
                size,
                frame: self.frame,
                backbuffer,
                backbuffer_format: self.backbuffer_format,
                targets: &self.targets,
                first_writes: &slot.first_writes,
            };
            slot.pass.record(&mut ctx)?;
        }
        self.frame = self.frame.wrapping_add(1);
        Ok(encoder.finish())
    }

    fn position_of(
        &self,
        anchor: &str,
        relation: &str,
        pass: &dyn RenderPass,
    ) -> Result<usize, RaaError> {
        let name = pass.name();
        self.passes
            .iter()
            .position(|s| s.pass.name() == anchor)
            .ok_or_else(|| {
                graph_error(
                    "framegraph.unknown_pass",
                    format!("cannot add `{name}` {relation} `{anchor}`: no such pass"),
                )
            })
    }

    fn insert_pass(&mut self, at: usize, pass: Box<dyn RenderPass>) -> PassId {
        self.passes.insert(
            at,
            PassSlot {
                pass,
                decl: PassBuilder::new(),
                first_writes: Vec::new(),
                needs_resize: true,
            },
        );
        self.compiled = false;
        let id = PassId(self.next_pass);
        self.next_pass = self.next_pass.wrapping_add(1);
        id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pass that only declares resources; it never touches a GPU.
    struct Dummy {
        name: &'static str,
        reads: Vec<ResourceId>,
        writes: Vec<ResourceId>,
    }

    impl RenderPass for Dummy {
        fn name(&self) -> &str {
            self.name
        }

        fn declare(&self, builder: &mut PassBuilder) {
            for &r in &self.reads {
                builder.read(r);
            }
            for &w in &self.writes {
                builder.write(w);
            }
        }

        fn record(&mut self, _ctx: &mut RecordContext<'_>) -> Result<(), RaaError> {
            Ok(())
        }
    }

    fn pass(name: &'static str, reads: &[ResourceId], writes: &[ResourceId]) -> Box<Dummy> {
        Box::new(Dummy {
            name,
            reads: reads.to_vec(),
            writes: writes.to_vec(),
        })
    }

    fn graph() -> FrameGraph {
        FrameGraph::new(wgpu::TextureFormat::Bgra8Unorm)
    }

    fn target(graph: &mut FrameGraph, label: &'static str) -> ResourceId {
        graph.create_target(TargetDesc {
            label,
            format: wgpu::TextureFormat::Rgba16Float,
            size: TargetSize::Surface,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        })
    }

    fn compile_code(graph: &mut FrameGraph) -> &'static str {
        graph.compile().err().map_or("ok", |e| e.code())
    }

    #[test]
    fn backbuffer_is_resource_zero_and_targets_follow() {
        let mut g = graph();
        assert_eq!(BACKBUFFER, ResourceId(0));
        assert_eq!(target(&mut g, "hdr"), ResourceId(1));
        assert_eq!(target(&mut g, "bloom"), ResourceId(2));
    }

    #[test]
    fn compile_succeeds_for_one_pass_writing_backbuffer() {
        let mut g = graph();
        g.add_pass(pass("clear", &[], &[BACKBUFFER]));
        assert_eq!(g.compile(), Ok(()));
        assert!(g.compiled);
        assert_eq!(g.passes[0].first_writes, [true]);
    }

    #[test]
    fn add_pass_before_places_the_new_pass_first() {
        let mut g = graph();
        g.add_pass(pass("a", &[], &[BACKBUFFER]));
        let new = pass("new", &[], &[BACKBUFFER]);
        g.add_pass_before("a", new).unwrap();
        assert_eq!(g.pass_names(), ["new", "a"]);
        assert_eq!(g.compile(), Ok(()));
        assert_eq!(g.pass_names(), ["new", "a"]);
    }

    #[test]
    fn add_pass_after_places_the_new_pass_next() {
        let mut g = graph();
        g.add_pass(pass("a", &[], &[BACKBUFFER]));
        g.add_pass(pass("b", &[], &[BACKBUFFER]));
        let new = pass("new", &[], &[BACKBUFFER]);
        g.add_pass_after("a", new).unwrap();
        assert_eq!(g.pass_names(), ["a", "new", "b"]);
    }

    #[test]
    fn placing_next_to_an_unknown_pass_is_refused() {
        let mut g = graph();
        g.add_pass(pass("a", &[], &[BACKBUFFER]));
        let err = g.add_pass_after("x", pass("new", &[], &[])).unwrap_err();
        assert_eq!(err.code(), "framegraph.unknown_pass");
        assert!(err.message().contains("`new`"));
        assert!(err.message().contains("`x`"));
        let err = g.add_pass_before("y", pass("new", &[], &[])).unwrap_err();
        assert_eq!(err.code(), "framegraph.unknown_pass");
        assert_eq!(g.pass_names(), ["a"]);
    }

    #[test]
    fn pass_ids_are_distinct() {
        let mut g = graph();
        let a = g.add_pass(pass("a", &[], &[BACKBUFFER]));
        let b = g.add_pass_before("a", pass("b", &[], &[])).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn adding_a_pass_invalidates_compilation() {
        let mut g = graph();
        g.add_pass(pass("a", &[], &[BACKBUFFER]));
        g.compile().unwrap();
        g.add_pass(pass("b", &[BACKBUFFER], &[BACKBUFFER]));
        assert!(!g.compiled);
        g.compile().unwrap();
        g.add_pass_after("a", pass("c", &[], &[])).unwrap();
        assert!(!g.compiled);
        g.compile().unwrap();
        g.add_pass_before("a", pass("d", &[], &[])).unwrap();
        assert!(!g.compiled);
    }

    #[test]
    fn only_the_first_writer_of_a_resource_gets_the_first_write_flag() {
        let mut g = graph();
        let hdr = target(&mut g, "hdr");
        g.add_pass(pass("scene", &[], &[hdr]));
        g.add_pass(pass("decals", &[hdr], &[hdr]));
        g.add_pass(pass("tonemap", &[hdr], &[BACKBUFFER]));
        g.add_pass(pass("ui", &[], &[BACKBUFFER]));
        g.compile().unwrap();
        let flags: Vec<&[bool]> = g.passes.iter().map(|s| &s.first_writes[..]).collect();
        let expected: [&[bool]; 4] = [
            &[false, true],
            &[false, false],
            &[true, false],
            &[false, false],
        ];
        assert_eq!(flags, expected);
    }

    #[test]
    fn recompiling_reuses_the_declaration_buffers() {
        let mut g = graph();
        g.add_pass(pass("a", &[BACKBUFFER], &[BACKBUFFER]));
        g.compile().unwrap();
        g.compile().unwrap();
        assert_eq!(g.passes[0].decl.reads, [BACKBUFFER]);
        assert_eq!(g.passes[0].decl.writes, [BACKBUFFER]);
    }

    #[test]
    fn each_compile_rule_reports_its_code_naming_pass_and_resource() {
        let mut g = graph();
        g.add_pass(pass("a", &[], &[BACKBUFFER]));
        g.add_pass(pass("a", &[], &[BACKBUFFER]));
        let err = g.compile().unwrap_err();
        assert_eq!(err.code(), "framegraph.duplicate_pass");
        assert!(err.message().contains("`a`"));

        let mut other = graph();
        target(&mut other, "t1");
        let foreign = target(&mut other, "t2");
        let mut g = graph();
        g.add_pass(pass("draw", &[], &[BACKBUFFER, foreign]));
        let err = g.compile().unwrap_err();
        assert_eq!(err.code(), "framegraph.unknown_resource");
        assert!(err.message().contains("`draw`"));
        assert!(err.message().contains("resource 2"));

        let mut g = graph();
        let hdr = target(&mut g, "hdr");
        g.add_pass(pass("tonemap", &[hdr], &[BACKBUFFER]));
        g.add_pass(pass("scene", &[], &[hdr]));
        let err = g.compile().unwrap_err();
        assert_eq!(err.code(), "framegraph.read_before_write");
        assert!(err.message().contains("`tonemap`"));
        assert!(err.message().contains("`hdr`"));

        let mut g = graph();
        let hdr = target(&mut g, "hdr");
        g.add_pass(pass("scene", &[], &[hdr]));
        let err = g.compile().unwrap_err();
        assert_eq!(err.code(), "framegraph.no_output");
        assert!(err.message().contains("BACKBUFFER"));
        assert!(err.message().contains("scene"));
        assert!(!g.compiled);
    }

    #[test]
    fn rules_apply_in_the_stated_order() {
        let mut g = graph();
        let hdr = target(&mut g, "hdr");
        g.add_pass(pass("a", &[hdr], &[ResourceId(9)]));
        g.add_pass(pass("a", &[], &[]));
        assert_eq!(compile_code(&mut g), "framegraph.duplicate_pass");

        let mut g = graph();
        let hdr = target(&mut g, "hdr");
        g.add_pass(pass("a", &[hdr], &[]));
        g.add_pass(pass("b", &[], &[ResourceId(9)]));
        assert_eq!(compile_code(&mut g), "framegraph.unknown_resource");

        let mut g = graph();
        let hdr = target(&mut g, "hdr");
        g.add_pass(pass("a", &[hdr], &[]));
        assert_eq!(compile_code(&mut g), "framegraph.read_before_write");
    }

    #[test]
    fn reading_the_backbuffer_needs_no_earlier_write() {
        let mut g = graph();
        g.add_pass(pass("overlay", &[BACKBUFFER], &[BACKBUFFER]));
        assert_eq!(compile_code(&mut g), "ok");
    }

    #[test]
    fn a_pass_reading_what_it_alone_writes_is_a_read_before_write() {
        let mut g = graph();
        let hdr = target(&mut g, "hdr");
        g.add_pass(pass("feedback", &[hdr], &[hdr, BACKBUFFER]));
        assert_eq!(compile_code(&mut g), "framegraph.read_before_write");
    }

    #[test]
    fn an_empty_graph_has_no_output() {
        assert_eq!(compile_code(&mut graph()), "framegraph.no_output");
    }
}
