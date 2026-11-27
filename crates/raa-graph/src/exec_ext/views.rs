//! Executor extension `views` (S55): the sixteen `view.image` shader variants, the node input
//! checks the views share applies, the compositor `ViewPass` each executed view node runs, the
//! `views` graph frame that binds their inputs, and the statistics of view and pass nodes.

use raa_core::{GpuContext, RaaError};
use raa_pipeline::{
    FeaturePass, GraphFrame, ImageDesc, ImageDim, PrepareCtx, RecordCtx, SampleKind,
};

use crate::catalog::{Catalog, GraphFormat, NodeTypeSpec, ParamValue, PortSpec, PortType};
use crate::error::{Issue, Severity, Stage};
use crate::exec::{ExecExtension, FeatureHandles, NodeStats, ResourceStat};
use crate::plan::{GraphPlan, InputSource, PlanInput, PlanNode};

#[allow(dead_code)] // the compositor card and the tests call these; this share installs nothing
/// The sixteen (dimension, sample kind) pairs `view.image` has a shader for, in this order.
pub(crate) const IMAGE_VARIANTS: [(ImageDim, SampleKind); 16] = [
    (ImageDim::D2, SampleKind::Float),
    (ImageDim::D2, SampleKind::UnfilterableFloat),
    (ImageDim::D2, SampleKind::Depth),
    (ImageDim::D2, SampleKind::Uint),
    (ImageDim::D2, SampleKind::Sint),
    (ImageDim::D2Array, SampleKind::Float),
    (ImageDim::D2Array, SampleKind::UnfilterableFloat),
    (ImageDim::D2Array, SampleKind::Depth),
    (ImageDim::D2Array, SampleKind::Uint),
    (ImageDim::D2Array, SampleKind::Sint),
    (ImageDim::D3, SampleKind::Float),
    (ImageDim::D3, SampleKind::UnfilterableFloat),
    (ImageDim::D3, SampleKind::Uint),
    (ImageDim::D3, SampleKind::Sint),
    (ImageDim::Cube, SampleKind::Float),
    (ImageDim::Cube, SampleKind::UnfilterableFloat),
];

#[allow(dead_code)] // the compositor card and the tests call these; this share installs nothing
/// The WGSL shared by every variant; `//#SOURCE` marks where the variant's binding and fetch go.
const TEMPLATE: &str = include_str!("../wgsl/view_image.wgsl");

#[allow(dead_code)] // the compositor card and the tests call these; this share installs nothing
/// The `view_fetch` body of the cube variants: face `view.layer` in WebGPU order (+X, -X, +Y, -Y,
/// +Z, -Z), sampled at the screen `uv`.
const CUBE_FETCH: &str = "\
    let s = uv * 2.0 - 1.0;
    let face = view.layer % 6u;
    var dir = vec3<f32>(-s.x, -s.y, -1.0);
    if (face == 0u) { dir = vec3<f32>(1.0, -s.y, -s.x); }
    else if (face == 1u) { dir = vec3<f32>(-1.0, -s.y, s.x); }
    else if (face == 2u) { dir = vec3<f32>(s.x, 1.0, s.y); }
    else if (face == 3u) { dir = vec3<f32>(s.x, -1.0, -s.y); }
    else if (face == 4u) { dir = vec3<f32>(s.x, -s.y, 1.0); }
    return textureSampleLevel(view_source, view_sampler, dir, f32(view.mip));
";

#[allow(dead_code)] // the compositor card and the tests call these; this share installs nothing
/// The texture declaration and `textureLoad` call of a variant, `None` outside [`IMAGE_VARIANTS`].
fn variant(dim: ImageDim, sample: SampleKind) -> Option<(&'static str, &'static str)> {
    use ImageDim::{Cube, D2, D2Array, D3};
    use SampleKind::{Depth, Float, Sint, Uint, UnfilterableFloat};
    let ty = match (dim, sample) {
        (D2, Float | UnfilterableFloat) => "texture_2d<f32>",
        (D2, Depth) => "texture_depth_2d",
        (D2, Uint) => "texture_2d<u32>",
        (D2, Sint) => "texture_2d<i32>",
        (D2Array, Float | UnfilterableFloat) => "texture_2d_array<f32>",
        (D2Array, Depth) => "texture_depth_2d_array",
        (D2Array, Uint) => "texture_2d_array<u32>",
        (D2Array, Sint) => "texture_2d_array<i32>",
        (D3, Float | UnfilterableFloat) => "texture_3d<f32>",
        (D3, Uint) => "texture_3d<u32>",
        (D3, Sint) => "texture_3d<i32>",
        (Cube, Float | UnfilterableFloat) => "texture_cube<f32>",
        _ => return None,
    };
    let load = match dim {
        D2 => "textureLoad(view_source, p, i32(view.mip))",
        D2Array => "textureLoad(view_source, p, i32(view.layer), i32(view.mip))",
        D3 => "textureLoad(view_source, vec3<i32>(p, i32(view.slice)), i32(view.mip))",
        Cube => "",
    };
    Some((ty, load))
}

/// The lane `channels` reads: `r`/`rgb`/`rgba-alpha` read x, `g` y, `b` z and `a` w.
fn integer_body(load: &str, signed: bool) -> String {
    let lane = |component: char| {
        if signed {
            format!("bitcast<u32>(texel.{component})")
        } else {
            format!("texel.{component}")
        }
    };
    format!(
        "    let texel = {load};\n    var lane = {};\n    if (view.channels == 2u) {{ lane = {}; }}\n    else if (view.channels == 3u) {{ lane = {}; }}\n    else if (view.channels == 4u) {{ lane = {}; }}\n    var v = f32(lane);\n    if (view.decode == 1u) {{ v = unpack2x16float(lane).x; }}\n    else if (view.decode == 2u) {{ v = unpack2x16float(lane).y; }}\n    return vec4<f32>(v, v, v, 1.0);\n",
        lane('x'),
        lane('y'),
        lane('z'),
        lane('w'),
    )
}

#[allow(dead_code)] // the compositor card and the tests call these; this share installs nothing
/// Handoff: `view_image_source` is called by card 'View nodes 4/6: ViewPass compositor FeaturePass'
/// to build the module of one variant. It returns the text of `wgsl/view_image.wgsl` with
/// `//#SOURCE` replaced by the `(dim, sample)` binding and `view_fetch`, `None` outside
/// [`IMAGE_VARIANTS`]; no state is passed.
pub(crate) fn view_image_source(dim: ImageDim, sample: SampleKind) -> Option<String> {
    let (ty, load) = variant(dim, sample)?;
    let fetch = if dim == ImageDim::Cube {
        CUBE_FETCH.to_string()
    } else {
        let head = "    let dims = vec2<i32>(textureDimensions(view_source, view.mip).xy);\n    let p = clamp(px, vec2<i32>(0), dims - 1);\n";
        let tail = match sample {
            SampleKind::Float | SampleKind::UnfilterableFloat => format!("    return {load};\n"),
            SampleKind::Depth => {
                format!("    let d = {load};\n    return vec4<f32>(d, d, d, 1.0);\n")
            }
            SampleKind::Uint => integer_body(load, false),
            SampleKind::Sint => integer_body(load, true),
        };
        format!("{head}{tail}")
    };
    let source = format!(
        "@group(0) @binding(1) var view_source: {ty};\n\nfn view_fetch(px: vec2<i32>, uv: vec2<f32>) -> vec4<f32> {{\n{fetch}}}\n"
    );
    Some(TEMPLATE.replace("//#SOURCE", &source))
}

/// Handoff: `ViewsExtension` is extended by card 'View nodes 5/6: views graph frame, pass
/// installation and node stats', which extends `ViewsExtension::apply` and `ViewsExtension::clear`
/// and calls `check_plan`; the generated registry attaches the extension through `extension()`.
/// State: the extension's own `Catalog`, built once in `ViewsExtension::new()`.
pub(crate) struct ViewsExtension {
    catalog: Catalog,
    /// The view passes this extension installed; the next apply or clear removes them.
    installed: Vec<String>,
    /// The table entries of the latest frame, captured by `frame` for `stats`.
    images: Vec<(String, String, ImageDesc)>,
    buffers: Vec<(String, String, u64)>,
}

impl ViewsExtension {
    /// The extension with its own catalog and nothing installed.
    pub(crate) fn new() -> ViewsExtension {
        ViewsExtension {
            catalog: Catalog::build(),
            installed: Vec::new(),
            images: Vec::new(),
            buffers: Vec::new(),
        }
    }

    /// Removes the view passes this extension installed and the `views` graph frame.
    fn uninstall(&mut self, h: &mut FeatureHandles<'_>) {
        let mut pipeline = h.pipeline.borrow_mut();
        for name in self.installed.drain(..) {
            let _ = pipeline.remove(&name);
        }
        let _ = pipeline.remove_graph_frame(VIEWS_FRAME);
    }

    /// Installs the view passes of `plan` and the `views` graph frame. A refusal removes what this
    /// call installed and returns its `graph.order` issue.
    fn install(&mut self, plan: &GraphPlan, h: &mut FeatureHandles<'_>) -> Result<(), Vec<Issue>> {
        if !plan
            .nodes
            .iter()
            .any(|node| node.type_id.starts_with("view."))
        {
            return Ok(());
        }
        let mut bindings = Vec::new();
        for Placement {
            node,
            spec,
            bindings: inputs,
        } in placements(plan, &self.catalog)
        {
            let name = spec.name.clone();
            let added = h.pipeline.borrow_mut().add(Box::new(ViewPass::new(spec)));
            if let Err(error) = added {
                self.uninstall(h);
                return Err(vec![order_issue(Some(node.as_str()), &error)]);
            }
            self.installed.push(name);
            bindings.extend(inputs);
        }
        let frame = ViewsFrame { bindings };
        let added = h
            .pipeline
            .borrow_mut()
            .add_graph_frame(VIEWS_FRAME, Box::new(frame));
        if let Err(error) = added {
            self.uninstall(h);
            return Err(vec![order_issue(None, &error)]);
        }
        Ok(())
    }
}

/// The `views` extension, discovered by build.rs.
pub fn extension() -> Box<dyn ExecExtension> {
    Box::new(ViewsExtension::new())
}

impl ExecExtension for ViewsExtension {
    fn name(&self) -> &'static str {
        "views"
    }

    fn apply(
        &mut self,
        plan: &GraphPlan,
        previous: Option<&GraphPlan>,
        h: &mut FeatureHandles<'_>,
    ) -> Result<(), Vec<Issue>> {
        let issues = check_plan(plan, &self.catalog);
        if !issues.is_empty() {
            return Err(issues);
        }
        self.uninstall(h);
        let Err(issues) = self.install(plan, h) else {
            return Ok(());
        };
        // The executor does not undo a refused extension, so the previous views are restored here.
        if let Some(before) = previous {
            let _ = self.install(before, h);
        }
        Err(issues)
    }

    fn clear(&mut self, h: &mut FeatureHandles<'_>) {
        self.uninstall(h);
        self.images.clear();
        self.buffers.clear();
    }

    fn frame(&mut self, h: &mut FeatureHandles<'_>) -> Result<(), RaaError> {
        let pipeline = h.pipeline.borrow();
        let table = pipeline.resources();
        self.images = table.entries();
        self.buffers = table.buffer_entries();
        Ok(())
    }

    fn stats(&self, out: &mut Vec<NodeStats>) {
        for node in out.iter_mut() {
            if node.type_id.starts_with("view.") {
                let pass = format!("{VIEW_PREFIX}{}", node.node);
                let colour = self
                    .images
                    .iter()
                    .find(|(name, port, _)| *name == pass && port == "color");
                node.ran = colour.is_some();
                node.resources = colour
                    .map(|(_, port, desc)| resource_stat(port, desc))
                    .into_iter()
                    .collect();
            } else if let Some(pass) = self.catalog.node(&node.type_id).and_then(|spec| spec.pass) {
                let images = self
                    .images
                    .iter()
                    .filter(|(name, _, _)| name == pass)
                    .map(|(_, port, desc)| resource_stat(port, desc));
                let buffers = self
                    .buffers
                    .iter()
                    .filter(|(name, _, _)| name == pass)
                    .map(|(_, port, size)| buffer_stat(port, *size));
                node.resources = images.chain(buffers).collect();
            }
        }
    }
}

/// The name prefix of every view pass: `view#<nodeId>`.
const VIEW_PREFIX: &str = "view#";
/// The graph frame that binds the view passes' inputs.
const VIEWS_FRAME: &str = "views";

/// One bound input of a view pass: `(view pass name, input port, source pass, source port)`.
type Binding = (String, &'static str, String, String);

/// The `views` graph frame (S55): before a view pass records, each of its linked inputs is bound to
/// the image its source pass published in this frame, when one was published. Every other pass
/// is ignored, and the other hooks keep their defaults.
struct ViewsFrame {
    bindings: Vec<Binding>,
}

impl GraphFrame for ViewsFrame {
    fn before_record(&mut self, pass: &str, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        if !pass.starts_with(VIEW_PREFIX) {
            return Ok(());
        }
        for (view, port, source, source_port) in &self.bindings {
            if view == pass
                && let Some(image) = ctx.resources.get(source, source_port)
            {
                ctx.resources.bind_input(view, port, image);
            }
        }
        Ok(())
    }
}

/// The `graph.order` issue of a view pass the pipeline refused, on `node` when it has one.
fn order_issue(node: Option<&str>, error: &RaaError) -> Issue {
    Issue {
        code: "graph.order",
        severity: Severity::Error,
        stage: Stage::Apply,
        node: node.map(str::to_owned),
        port: None,
        param: None,
        link: None,
        message: error.message().to_owned(),
        fix: "Unlink the view node, or link it to a pass the pipeline's order can place."
            .to_owned(),
    }
}

/// One apply-stage error of `node` on `port`.
fn issue(node: &str, port: &str, code: &'static str, message: String, fix: &str) -> Issue {
    Issue {
        code,
        severity: Severity::Error,
        stage: Stage::Apply,
        node: Some(node.to_string()),
        port: Some(port.to_string()),
        param: None,
        link: None,
        message,
        fix: fix.to_string(),
    }
}

/// The catalog output spec that feeds a planned input, `None` when it is not a link.
fn output_spec<'a>(
    plan: &'a GraphPlan,
    catalog: &'a Catalog,
    node: &str,
    port: &str,
) -> Option<&'a PortSpec> {
    let producer = plan.nodes.iter().find(|candidate| candidate.id == node)?;
    catalog
        .node(&producer.type_id)?
        .outputs
        .iter()
        .find(|spec| spec.key == port)
}

/// The input checks of the view nodes (contract 'view-shader-image'): `view.image` links exactly
/// one of its six inputs, `view.grid` at least one tile, and `view.compare`/`view.grid` read
/// float-sampleable image2d sources only. Every issue is an apply-stage error.
pub(crate) fn check_plan(plan: &GraphPlan, catalog: &Catalog) -> Vec<Issue> {
    let mut issues = Vec::new();
    for node in &plan.nodes {
        let Some(spec) = catalog.node(&node.type_id) else {
            continue;
        };
        let linked: Vec<&PlanInput> = node
            .inputs
            .iter()
            .filter(|input| matches!(input.source, InputSource::Link { .. }))
            .collect();
        if spec.id == "view.image" {
            if linked.is_empty() {
                issues.push(issue(
                    &node.id,
                    "src",
                    "graph.input_missing",
                    format!(
                        "node `{}` (view.image) links none of its image inputs src, srcDepth, srcArray, srcDepthArray, src3d and srcCube, and exactly one must be linked",
                        node.id
                    ),
                    "Link one image into the node's `src` input.",
                ));
            } else if linked.len() > 1 {
                let (first, second) = (linked[0].port.as_str(), linked[1].port.as_str());
                issues.push(issue(
                    &node.id,
                    second,
                    "graph.input_taken",
                    format!(
                        "node `{}` (view.image) links two image inputs, `{first}` and `{second}`, and exactly one is allowed",
                        node.id
                    ),
                    &format!("Unlink `{second}` so that only `{first}` stays linked."),
                ));
            }
        }
        if spec.id == "view.grid" && linked.is_empty() {
            issues.push(issue(
                &node.id,
                "t0",
                "graph.input_missing",
                format!(
                    "node `{}` (view.grid) links none of its tile inputs t0..t15, so it has nothing to show",
                    node.id
                ),
                "Link an image into the node's `t0` input.",
            ));
        }
        if spec.id != "view.compare" && spec.id != "view.grid" {
            continue;
        }
        for input in linked {
            let InputSource::Link {
                node: source_node,
                port: source_port,
                ..
            } = &input.source
            else {
                continue;
            };
            let Some(source) = output_spec(plan, catalog, source_node, source_port) else {
                continue;
            };
            let refused = source.ty != PortType::Image2d
                || matches!(
                    source.format,
                    Some(
                        GraphFormat::R32Uint
                            | GraphFormat::Rg32Uint
                            | GraphFormat::Rgba32Uint
                            | GraphFormat::Depth32Float
                    )
                );
            if refused {
                issues.push(issue(
                    &node.id,
                    &input.port,
                    "graph.link_type",
                    format!(
                        "node `{}` input `{}` reads `{source_node}.{source_port}`, a {:?} with format {}, and view.compare and view.grid take float-sampleable image2d sources only",
                        node.id,
                        input.port,
                        source.ty,
                        source.format.as_ref().map_or("none", GraphFormat::name)
                    ),
                    &format!(
                        "Insert a view.image node between `{source_node}.{source_port}` and `{}.{}`.",
                        node.id, input.port
                    ),
                ));
            }
        }
    }
    issues
}

/// Returns the `view.compare` module (crates/raa-graph/src/wgsl/view_compare.wgsl) unchanged.
/// Handoff: `view_compare_source`, `view_grid_source` and `pack_label` are called by card
/// 'View nodes 4/6: ViewPass compositor FeaturePass'.
#[allow(dead_code)] // called by the compositor and the tests
pub(crate) fn view_compare_source() -> String {
    include_str!("../wgsl/view_compare.wgsl").to_string()
}

/// Returns the `view.grid` module (crates/raa-graph/src/wgsl/view_grid.wgsl) unchanged.
#[allow(dead_code)] // called by the compositor and the tests
pub(crate) fn view_grid_source() -> String {
    include_str!("../wgsl/view_grid.wgsl").to_string()
}

/// Packs the first 32 bytes of `text` into `out`, 4 bytes per word little-endian, zero after the
/// end; a char outside ASCII 32..=126 becomes one `?`.
#[allow(dead_code)] // called by the compositor and the tests
pub(crate) fn pack_label(text: &str, out: &mut [u32; 8]) {
    *out = [0; 8];
    for (i, c) in text.chars().take(32).enumerate() {
        let byte = if (' '..='~').contains(&c) {
            c as u32
        } else {
            u32::from(b'?')
        };
        out[i / 4] |= byte << ((i % 4) * 8);
    }
}

/// The colour format of a view pass's published `color`.
const COLOUR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// The colour a view pass clears to when it has no input to draw.
const GREY: wgpu::Color = wgpu::Color {
    r: 0.1,
    g: 0.1,
    b: 0.1,
    a: 1.0,
};
/// Uniform block sizes: ViewParams, CompareParams, and GridParams (a 16 B header, then 16 tiles of
/// 32 B label words).
const IMAGE_BYTES: usize = 48;
const COMPARE_BYTES: usize = 16;
const GRID_BYTES: usize = 528;
/// The tiles a `view.grid` shows.
const GRID_TILES: usize = 16;
/// The image inputs of a `view.image` node, in the order the first bound one is read.
const IMAGE_PORTS: [&str; 6] = [
    "src",
    "srcDepth",
    "srcArray",
    "srcDepthArray",
    "src3d",
    "srcCube",
];
/// The parameters of the backbuffer draw: rgb, range 0..1, no exposure.
const IDENTITY: ImageParams = ImageParams {
    channels: 0,
    linearize: false,
    layer: 0,
    slice: 0,
    mip: 0,
    decode: 0,
    range_min: 0.0,
    range_max: 1.0,
    exposure_ev: 0.0,
};

/// The codes of a `view.image` node (contract 'view-shader-image', ViewParams).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ImageParams {
    pub(crate) channels: u32,
    pub(crate) linearize: bool,
    pub(crate) layer: u32,
    pub(crate) slice: u32,
    pub(crate) mip: u32,
    pub(crate) decode: u32,
    pub(crate) range_min: f32,
    pub(crate) range_max: f32,
    pub(crate) exposure_ev: f32,
}

/// What a view pass shows (contract 'view-pass').
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ViewKind {
    /// One image of `view.image`, drawn with its codes.
    Image(ImageParams),
    /// `view.compare`: mode 0 split, 1 side by side, 2 difference, and the split position.
    Compare { mode: u32, position: f32 },
    /// `view.grid`: the linked tiles in port order with their labels, at most 16.
    Grid {
        columns: u32,
        labels: bool,
        tiles: Vec<(&'static str, String)>,
    },
}

/// The spec of one view pass: its pass name `view#<nodeId>`, its order, whether it is an output
/// overlay after tone mapping, whether it draws its colour onto the backbuffer, and what it shows.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ViewPassSpec {
    pub(crate) name: String,
    pub(crate) order: u32,
    pub(crate) overlay: bool,
    pub(crate) display: bool,
    pub(crate) kind: ViewKind,
}

/// A shader program of the views; its cache key is the variant or the fixed compare and grid module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Program {
    Image(ImageDim, SampleKind),
    Compare,
    Grid,
}

/// The module and bind group layout of one program, fragment stage only.
struct Shader {
    module: wgpu::ShaderModule,
    layout: wgpu::BindGroupLayout,
}

/// The colour target of a view pass, recreated when the frame size changes.
struct Colour {
    size: (u32, u32),
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

/// Handoff: `ViewPass` is the compositor FeaturePass of one executed view node. Card 'View nodes
/// 5/6: views graph frame, pass installation and node stats' builds it with `ViewPass::new(spec:
/// ViewPassSpec) -> ViewPass` and attaches it with `Pipeline::add`. Today `prepare` creates the
/// colour texture and the uniform buffer on first use and publishes `(name, "color")`; `record`
/// reads the inputs bound in the resource table, and an unbound image or compare input clears the
/// colour to grey without drawing.
pub(crate) struct ViewPass {
    spec: ViewPassSpec,
    colour: Option<Colour>,
    params: Option<wgpu::Buffer>,
    identity: Option<wgpu::Buffer>,
    sampler: Option<wgpu::Sampler>,
    dummy: Option<wgpu::TextureView>,
    shaders: Vec<(Program, Shader)>,
    pipelines: Vec<((Program, wgpu::TextureFormat), wgpu::RenderPipeline)>,
}

impl ViewPass {
    /// The pass of `spec`. No GPU work happens here: `prepare` and `record` create what they use.
    pub(crate) fn new(spec: ViewPassSpec) -> ViewPass {
        ViewPass {
            spec,
            colour: None,
            params: None,
            identity: None,
            sampler: None,
            dummy: None,
            shaders: Vec::new(),
            pipelines: Vec::new(),
        }
    }
}

impl Colour {
    fn new(gpu: &GpuContext, size: (u32, u32)) -> Colour {
        let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("raa.views.colour"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOUR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Colour {
            size,
            _texture: texture,
            view,
        }
    }
}

impl FeaturePass for ViewPass {
    fn name(&self) -> &str {
        &self.spec.name
    }

    fn order(&self) -> u32 {
        self.spec.order
    }

    fn output_overlay(&self) -> bool {
        self.spec.overlay
    }

    fn prepare(&mut self, ctx: &mut PrepareCtx<'_>) -> Result<(), RaaError> {
        let gpu = ctx.gpu;
        let size = ctx.size;
        let colour = match self.colour.take() {
            Some(colour) if colour.size == size => colour,
            _ => Colour::new(gpu, size),
        };
        let desc = ImageDesc::new(ImageDim::D2, COLOUR_FORMAT, size.0, size.1, 1, 1);
        ctx.resources
            .publish(&self.spec.name, "color", &colour.view, desc);
        self.colour = Some(colour);
        let buffer = self
            .params
            .get_or_insert_with(|| uniform_buffer(gpu.device(), kind_bytes(&self.spec.kind)));
        write_params(gpu.queue(), buffer, &self.spec.kind);
        Ok(())
    }

    fn record(&mut self, ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
        let gpu = ctx.gpu;
        let device = gpu.device();
        let name = self.spec.name.as_str();
        let (Some(colour), Some(params)) = (self.colour.as_ref(), self.params.as_ref()) else {
            return Err(RaaError::Resource(format!(
                "view pass `{name}` was recorded before its prepare"
            )));
        };
        let sampler: &wgpu::Sampler = self.sampler.get_or_insert_with(|| nearest_sampler(device));
        let inputs: Option<(Program, Vec<wgpu::TextureView>)> = match &self.spec.kind {
            ViewKind::Image(_) => IMAGE_PORTS
                .iter()
                .find_map(|port| ctx.resources.input(name, port))
                .map(|image| {
                    (
                        Program::Image(image.desc.dim, image.desc.sample),
                        vec![image.view],
                    )
                }),
            ViewKind::Compare { .. } => {
                match (
                    ctx.resources.input(name, "a"),
                    ctx.resources.input(name, "b"),
                ) {
                    (Some(a), Some(b)) => Some((Program::Compare, vec![a.view, b.view])),
                    _ => None,
                }
            }
            ViewKind::Grid { tiles, .. } => {
                let dummy = self.dummy.get_or_insert_with(|| dummy_view(gpu));
                let views = (0..GRID_TILES)
                    .map(|slot| {
                        tiles
                            .get(slot)
                            .and_then(|(port, _)| ctx.resources.input(name, port))
                            .map_or_else(|| dummy.clone(), |image| image.view)
                    })
                    .collect();
                Some((Program::Grid, views))
            }
        };
        let target = &colour.view;
        match inputs {
            Some((program, views)) => {
                let shader = cached(&mut self.shaders, program, || build_shader(device, program))?;
                let pipeline = cached(&mut self.pipelines, (program, COLOUR_FORMAT), || {
                    Ok(build_pipeline(device, shader, COLOUR_FORMAT))
                })?;
                let refs: Vec<&wgpu::TextureView> = views.iter().collect();
                let bound_sampler = matches!(program, Program::Image(..)).then_some(sampler);
                let group = bind_group(device, shader, params, &refs, bound_sampler);
                let clear = wgpu::LoadOp::Clear(wgpu::Color::BLACK);
                draw_pass(ctx.encoder, target, clear, Some((pipeline, &group)));
            }
            None => draw_pass(ctx.encoder, target, wgpu::LoadOp::Clear(GREY), None),
        }
        if self.spec.display {
            let program = Program::Image(ImageDim::D2, SampleKind::Float);
            let format = ctx.backbuffer_format;
            let shader = cached(&mut self.shaders, program, || build_shader(device, program))?;
            let pipeline = cached(&mut self.pipelines, (program, format), || {
                Ok(build_pipeline(device, shader, format))
            })?;
            let identity = self.identity.get_or_insert_with(|| {
                let buffer = uniform_buffer(device, IMAGE_BYTES as u64);
                gpu.queue()
                    .write_buffer(&buffer, 0, &image_bytes(&IDENTITY));
                buffer
            });
            let group = bind_group(device, shader, identity, &[&colour.view], Some(sampler));
            let load = wgpu::LoadOp::Load;
            draw_pass(ctx.encoder, ctx.backbuffer, load, Some((pipeline, &group)));
        }
        Ok(())
    }
}

/// The entry of `cache` under `key`, built by `make` on first use.
fn cached<K: PartialEq, V>(
    cache: &mut Vec<(K, V)>,
    key: K,
    make: impl FnOnce() -> Result<V, RaaError>,
) -> Result<&V, RaaError> {
    let index = match cache.iter().position(|(known, _)| *known == key) {
        Some(index) => index,
        None => {
            let value = make()?;
            cache.push((key, value));
            cache.len() - 1
        }
    };
    Ok(&cache[index].1)
}

/// The WGSL of `program` and its bind group layout: a `view.image` variant (its binding 1 texture,
/// 2 the sampler), or the compare or grid module with float textures from binding 1.
fn build_shader(device: &wgpu::Device, program: Program) -> Result<Shader, RaaError> {
    let uniform = wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let (source, entries) = match program {
        Program::Image(dim, sample) => {
            let source = view_image_source(dim, sample).ok_or_else(|| RaaError::FrameGraph {
                code: "graph.view_variant",
                message: format!("no view.image shader for {dim:?} {sample:?}"),
            })?;
            let view_dimension = match dim {
                ImageDim::D2 => wgpu::TextureViewDimension::D2,
                ImageDim::D2Array => wgpu::TextureViewDimension::D2Array,
                ImageDim::D3 => wgpu::TextureViewDimension::D3,
                ImageDim::Cube => wgpu::TextureViewDimension::Cube,
            };
            let sample_type = match sample {
                SampleKind::Float => wgpu::TextureSampleType::Float { filterable: true },
                SampleKind::UnfilterableFloat => {
                    wgpu::TextureSampleType::Float { filterable: false }
                }
                SampleKind::Depth => wgpu::TextureSampleType::Depth,
                SampleKind::Uint => wgpu::TextureSampleType::Uint,
                SampleKind::Sint => wgpu::TextureSampleType::Sint,
            };
            let texture = wgpu::BindingType::Texture {
                sample_type,
                view_dimension,
                multisampled: false,
            };
            let sampler = wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering);
            let entries = vec![
                layout_entry(0, uniform),
                layout_entry(1, texture),
                layout_entry(2, sampler),
            ];
            (source, entries)
        }
        Program::Compare => {
            let entries = vec![layout_entry(0, uniform), float_layout(1), float_layout(2)];
            (view_compare_source(), entries)
        }
        Program::Grid => {
            let uniform = layout_entry(0, uniform);
            let tiles = (1..=GRID_TILES as u32).map(float_layout);
            (
                view_grid_source(),
                std::iter::once(uniform).chain(tiles).collect(),
            )
        }
    };
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("raa.views.shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("raa.views.layout"),
        entries: &entries,
    });
    Ok(Shader { module, layout })
}

/// A fragment-stage bind group layout entry.
fn layout_entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty,
        count: None,
    }
}

/// A non-multisampled 2D float texture the compare and grid modules load with `textureLoad`.
fn float_layout(binding: u32) -> wgpu::BindGroupLayoutEntry {
    layout_entry(
        binding,
        wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
    )
}

/// The pipeline of `shader` drawing to one `format`: stages `vs` and `fs`, no vertex buffers.
fn build_pipeline(
    device: &wgpu::Device,
    shader: &Shader,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("raa.views.pipeline_layout"),
        bind_group_layouts: &[Some(&shader.layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("raa.views"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader.module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader.module,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// The bind group of a program: `buffer` at binding 0, `views` from 1, the sampler after them.
fn bind_group(
    device: &wgpu::Device,
    shader: &Shader,
    buffer: &wgpu::Buffer,
    views: &[&wgpu::TextureView],
    sampler: Option<&wgpu::Sampler>,
) -> wgpu::BindGroup {
    let mut entries = vec![wgpu::BindGroupEntry {
        binding: 0,
        resource: buffer.as_entire_binding(),
    }];
    for (binding, view) in (1..).zip(views) {
        entries.push(wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::TextureView(view),
        });
    }
    if let Some(sampler) = sampler {
        entries.push(wgpu::BindGroupEntry {
            binding: views.len() as u32 + 1,
            resource: wgpu::BindingResource::Sampler(sampler),
        });
    }
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("raa.views.group"),
        layout: &shader.layout,
        entries: &entries,
    })
}

/// Records one render pass onto `target` that clears or loads it and, when `draw` is given, draws
/// the full-screen triangle with its pipeline and bind group.
fn draw_pass(
    encoder: &mut wgpu::CommandEncoder,
    target: &wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
    draw: Option<(&wgpu::RenderPipeline, &wgpu::BindGroup)>,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("raa.views"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    if let Some((pipeline, group)) = draw {
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// A uniform buffer of `size` bytes that the pass rewrites each frame.
fn uniform_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("raa.views.params"),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// The uniform block size of a pass kind.
fn kind_bytes(kind: &ViewKind) -> u64 {
    match kind {
        ViewKind::Image(_) => IMAGE_BYTES as u64,
        ViewKind::Compare { .. } => COMPARE_BYTES as u64,
        ViewKind::Grid { .. } => GRID_BYTES as u64,
    }
}

/// Writes the uniform block of `kind` into `buffer`.
fn write_params(queue: &wgpu::Queue, buffer: &wgpu::Buffer, kind: &ViewKind) {
    match kind {
        ViewKind::Image(params) => queue.write_buffer(buffer, 0, &image_bytes(params)),
        ViewKind::Compare { mode, position } => {
            queue.write_buffer(buffer, 0, &compare_bytes(*mode, *position));
        }
        ViewKind::Grid {
            columns,
            labels,
            tiles,
        } => queue.write_buffer(buffer, 0, &grid_bytes(*columns, *labels, tiles)),
    }
}

/// Writes `words` into `out` from its start, four little-endian bytes each.
fn put_words(out: &mut [u8], words: &[u32]) {
    for (chunk, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(words) {
        chunk.copy_from_slice(&word.to_le_bytes());
    }
}

/// ViewParams: channels, linearize, layer, slice, mip, decode, range min, range max, exposure, then
/// three zero words.
fn image_bytes(params: &ImageParams) -> [u8; IMAGE_BYTES] {
    let mut out = [0; IMAGE_BYTES];
    put_words(
        &mut out,
        &[
            params.channels,
            u32::from(params.linearize),
            params.layer,
            params.slice,
            params.mip,
            params.decode,
            params.range_min.to_bits(),
            params.range_max.to_bits(),
            params.exposure_ev.to_bits(),
            0,
            0,
            0,
        ],
    );
    out
}

/// CompareParams: mode, split position, two zero words.
fn compare_bytes(mode: u32, position: f32) -> [u8; COMPARE_BYTES] {
    let mut out = [0; COMPARE_BYTES];
    put_words(&mut out, &[mode, position.to_bits(), 0, 0]);
    out
}

/// GridParams: columns, tile count, labels flag, a zero word, then 16 tiles of 8 label words each
/// (`pack_label`).
fn grid_bytes(columns: u32, labels: bool, tiles: &[(&'static str, String)]) -> [u8; GRID_BYTES] {
    let mut out = [0; GRID_BYTES];
    let count = tiles.len().min(GRID_TILES) as u32;
    put_words(&mut out, &[columns, count, u32::from(labels), 0]);
    for (slot, (_, label)) in tiles.iter().take(GRID_TILES).enumerate() {
        let mut words = [0; 8];
        pack_label(label, &mut words);
        put_words(&mut out[16 + 32 * slot..], &words);
    }
    out
}

/// The 1x1 tile of an unbound grid slot: grey 0.1 (binary16 0x2E66 per colour channel), alpha 1.
fn dummy_view(gpu: &GpuContext) -> wgpu::TextureView {
    let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("raa.views.dummy"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: COLOUR_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    gpu.queue().write_texture(
        texture.as_image_copy(),
        &[0x66, 0x2e, 0x66, 0x2e, 0x66, 0x2e, 0x00, 0x3c],
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
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// The nearest, clamped sampler of the `view.image` variants (NonFiltering).
fn nearest_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("raa.views.sampler"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    })
}

/// The compare inputs of `view.compare`.
const COMPARE_PORTS: [&str; 2] = ["a", "b"];
/// The tile inputs of `view.grid`, in port order.
const TILE_PORTS: [&str; 16] = [
    "t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11", "t12", "t13", "t14",
    "t15",
];
/// Every format a `GraphFormat` names; a published texture format takes the name of the first one
/// that maps to it.
const GRAPH_FORMATS: [GraphFormat; 17] = [
    GraphFormat::R8Unorm,
    GraphFormat::Rg8Unorm,
    GraphFormat::Rgba8Unorm,
    GraphFormat::Rgba8UnormSrgb,
    GraphFormat::Bgra8Unorm,
    GraphFormat::R16Float,
    GraphFormat::Rg16Float,
    GraphFormat::Rgba16Float,
    GraphFormat::R32Float,
    GraphFormat::Rg32Float,
    GraphFormat::Rgba32Float,
    GraphFormat::R32Uint,
    GraphFormat::Rg32Uint,
    GraphFormat::Rgba32Uint,
    GraphFormat::Rgb10a2Unorm,
    GraphFormat::Rg11b10Ufloat,
    GraphFormat::Depth32Float,
];

/// The placement of one view node: its pass, and the bindings its graph frame keeps.
struct Placement {
    node: String,
    spec: ViewPassSpec,
    bindings: Vec<Binding>,
}

/// A linked input of a view node, with what its placement reads from the source node.
struct Source {
    port: &'static str,
    pass: String,
    source_port: String,
    /// The source node's plan order index.
    index: u32,
    /// The source pass's default order, or the placed order of a source view.
    order: u32,
    /// The source is a tonemap or overlay colour, or a display-space view.
    display_space: bool,
    /// The grid label: the `label` of a `view.image` source, else its node id.
    label: String,
}

/// The parameters of a planned node: the values the plan wrote, or the catalog default.
struct Params<'a> {
    node: &'a PlanNode,
    spec: &'a NodeTypeSpec,
}

impl<'a> Params<'a> {
    fn get(&self, key: &str) -> Option<&'a ParamValue> {
        let (node, spec) = (self.node, self.spec);
        node.params
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
            .or_else(|| {
                spec.params
                    .iter()
                    .find(|param| param.key == key)
                    .map(|param| &param.default)
            })
    }

    fn number(&self, key: &str) -> f32 {
        match self.get(key) {
            Some(ParamValue::Number(value)) => *value as f32,
            _ => 0.0,
        }
    }

    fn int(&self, key: &str) -> u32 {
        match self.get(key) {
            Some(ParamValue::Int(value)) => u32::try_from(*value).unwrap_or(0),
            _ => 0,
        }
    }

    fn flag(&self, key: &str) -> bool {
        matches!(self.get(key), Some(ParamValue::Bool(true)))
    }

    /// The index of an enum value among its declared options: the code the shaders read.
    fn code(&self, key: &str) -> u32 {
        let declared = self.spec.params.iter().find(|param| param.key == key);
        match (self.get(key), declared) {
            (Some(ParamValue::Enum(choice)), Some(param)) => param
                .options
                .iter()
                .position(|option| *option == choice.as_str())
                .map_or(0, |index| index as u32),
            _ => 0,
        }
    }

    fn text(&self, key: &str) -> &'a str {
        match self.get(key) {
            Some(ParamValue::String(text)) => text.as_str(),
            _ => "",
        }
    }
}

/// The static name of a view node's input port, `None` for any other port.
fn static_port(port: &str) -> Option<&'static str> {
    IMAGE_PORTS
        .iter()
        .chain(&COMPARE_PORTS)
        .chain(&TILE_PORTS)
        .find(|known| **known == port)
        .copied()
}

/// The source of a linked input of a view node; `None` for an input that is not linked or whose
/// source node is not placed.
fn source_of(
    plan: &GraphPlan,
    catalog: &Catalog,
    placed: &[Placement],
    input: &PlanInput,
) -> Option<Source> {
    let InputSource::Link {
        node: from,
        port: from_port,
        ..
    } = &input.source
    else {
        return None;
    };
    let port = static_port(&input.port)?;
    let node = plan.nodes.iter().find(|candidate| candidate.id == *from)?;
    let spec = catalog.node(&node.type_id)?;
    let (pass, order, view_display) = if node.type_id == "compute.wgsl" {
        (format!("compute#{}", node.id), spec.default_order, false)
    } else if node.type_id.starts_with("view.") {
        let name = format!("{VIEW_PREFIX}{}", node.id);
        let view = placed.iter().find(|view| view.spec.name == name)?;
        (name, view.spec.order, view.spec.overlay)
    } else {
        let pass = spec
            .pass
            .map_or_else(|| node.type_id.clone(), str::to_owned);
        (pass, spec.default_order, false)
    };
    let colour = matches!(pass.as_str(), "tonemap" | "overlay") && from_port == "color";
    let text = Params { node, spec }.text("label");
    let label = if node.type_id == "view.image" && !text.is_empty() {
        text.to_owned()
    } else {
        node.id.clone()
    };
    Some(Source {
        port,
        pass,
        source_port: from_port.clone(),
        index: node.order_index,
        order,
        display_space: colour || view_display,
        label,
    })
}

/// The kind a view node's pass shows, from its parameters and linked inputs (grid tiles in port order).
fn view_kind(params: &Params<'_>, sources: &[Source]) -> ViewKind {
    match params.spec.id {
        "view.compare" => ViewKind::Compare {
            mode: params.code("mode"),
            position: params.number("position"),
        },
        "view.grid" => ViewKind::Grid {
            columns: params.int("columns"),
            labels: params.flag("labels"),
            tiles: TILE_PORTS
                .iter()
                .filter_map(|port| {
                    sources
                        .iter()
                        .find(|source| source.port == *port)
                        .map(|source| (*port, source.label.clone()))
                })
                .collect(),
        },
        _ => ViewKind::Image(ImageParams {
            channels: params.code("channels"),
            linearize: params.flag("linearizeDepth"),
            layer: params.int("layer"),
            slice: params.int("slice"),
            mip: params.int("mip"),
            decode: params.code("decode"),
            range_min: params.number("rangeMin"),
            range_max: params.number("rangeMax"),
            exposure_ev: params.number("exposureEv"),
        }),
    }
}

/// The placement of every view node of `plan`, in plan order. A display-space node (on the display,
/// or reading a tonemap or overlay colour or a display-space view) runs as an output overlay at
/// `1000 + order_index`. Any other node takes the default order of its latest pass source, so the
/// pipeline puts it right after that pass; a source view contributes its own order.
fn placements(plan: &GraphPlan, catalog: &Catalog) -> Vec<Placement> {
    let mut placed: Vec<Placement> = Vec::new();
    for node in plan
        .nodes
        .iter()
        .filter(|node| node.type_id.starts_with("view."))
    {
        let Some(spec) = catalog.node(&node.type_id) else {
            continue;
        };
        let sources: Vec<Source> = node
            .inputs
            .iter()
            .filter_map(|input| source_of(plan, catalog, &placed, input))
            .collect();
        let display = plan.output.node == node.id;
        let display_space = display || sources.iter().any(|source| source.display_space);
        let order = if display_space {
            1000 + node.order_index
        } else {
            sources
                .iter()
                .max_by_key(|source| source.index)
                .map_or(spec.default_order, |source| source.order)
        };
        let name = format!("{VIEW_PREFIX}{}", node.id);
        let bindings = sources
            .iter()
            .map(|source| {
                (
                    name.clone(),
                    source.port,
                    source.pass.clone(),
                    source.source_port.clone(),
                )
            })
            .collect();
        let kind = view_kind(&Params { node, spec }, &sources);
        placed.push(Placement {
            node: node.id.clone(),
            spec: ViewPassSpec {
                name,
                order,
                overlay: display_space,
                display,
                kind,
            },
            bindings,
        });
    }
    placed
}

/// The pass name, order, overlay and display flag of every view pass `plan` installs, in plan order.
#[cfg_attr(not(test), allow(dead_code))] // the placement tests pin the rule through this
pub(crate) fn placement(plan: &GraphPlan, catalog: &Catalog) -> Vec<(String, u32, bool, bool)> {
    placements(plan, catalog)
        .into_iter()
        .map(|placed| {
            (
                placed.spec.name,
                placed.spec.order,
                placed.spec.overlay,
                placed.spec.display,
            )
        })
        .collect()
}

/// The lowercase WebGPU name of `format`: the first `GraphFormat` that maps to it, else its Debug name.
fn format_name(format: wgpu::TextureFormat) -> String {
    GRAPH_FORMATS
        .iter()
        .find(|graph| graph.to_wgpu(format) == format)
        .map_or_else(
            || format!("{format:?}").to_lowercase(),
            |graph| graph.name().to_owned(),
        )
}

/// The statistics of one published image (S53 `ResourceStat`): its extent and format, and the bytes
/// of its mip chain over its layers. A depth format counts its depth aspect; a format without a
/// copy size counts 0 bytes.
pub(crate) fn resource_stat(port: &str, desc: &ImageDesc) -> ResourceStat {
    let aspect = desc
        .format
        .has_depth_aspect()
        .then_some(wgpu::TextureAspect::DepthOnly);
    let texel = desc.format.block_copy_size(aspect).unwrap_or(0);
    let bytes = (0..desc.mips)
        .map(|mip| {
            let width = u64::from(desc.width.checked_shr(mip).unwrap_or(0).max(1));
            let height = u64::from(desc.height.checked_shr(mip).unwrap_or(0).max(1));
            width * height * u64::from(desc.layers) * u64::from(texel)
        })
        .sum();
    ResourceStat {
        port: port.to_owned(),
        width: desc.width,
        height: desc.height,
        layers: desc.layers,
        mips: desc.mips,
        format: format_name(desc.format),
        bytes,
    }
}

/// The statistics of one published buffer: no extent, one layer and one mip, its size in bytes.
fn buffer_stat(port: &str, size: u64) -> ResourceStat {
    ResourceStat {
        port: port.to_owned(),
        width: 0,
        height: 0,
        layers: 1,
        mips: 1,
        format: "buffer".to_owned(),
        bytes: size,
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use raa_core::testing::{block_on, skip, software_gpu};
    use raa_loader::json::{JsonValue, parse, write_pretty};
    use raa_pipeline::{LightingResources, Pipeline, PublishedImage, ResourceTable, Targets};
    use raa_shader::Composer;

    use crate::exec::HandleSet;
    use crate::plan::plan_from_json;

    use super::*;

    #[test]
    fn compute_image_sources_bind_to_the_compute_pass() {
        let plan = plan_from_json(r#"{"format":"raa.render-graph","version":1,"name":"compute view","nodes":[{"id":"prepass","type":"pass.prepass"},{"id":"forward","type":"pass.forward"},{"id":"tonemap","type":"pass.tonemap"},{"id":"out","type":"output.display"},{"id":"calc","type":"compute.wgsl","inputs":[{"key":"src","type":"image2d"}],"outputs":[{"key":"dst","type":"image2d","format":"rgba16float"}],"math":null,"wgsl":"fn rg_main(p: vec2<u32>) {}"},{"id":"view","type":"view.image"}],"links":[{"from":"prepass.normal","to":"calc.src"},{"from":"calc.dst","to":"view.src"},{"from":"view.color","to":"tonemap.color"},{"from":"tonemap.color","to":"out.color"}]}"#).unwrap();
        let placements = placements(&plan, &Catalog::build());
        assert!(placements[0].bindings.contains(&(
            "view#view".into(),
            "src",
            "compute#calc".into(),
            "dst".into()
        )));
    }

    #[test]
    fn every_image_variant_validates() {
        for (dim, sample) in IMAGE_VARIANTS {
            let text = view_image_source(dim, sample)
                .unwrap_or_else(|| panic!("{dim:?} {sample:?} has a variant"));
            Composer::validate(&text).unwrap_or_else(|error| panic!("{dim:?} {sample:?}: {error}"));
        }
        assert_eq!(IMAGE_VARIANTS.len(), 16);
        assert!(view_image_source(ImageDim::D3, SampleKind::Depth).is_none());
        assert!(view_image_source(ImageDim::Cube, SampleKind::Uint).is_none());
    }

    /// The Simple forward starter as JSON, from the generated registry table.
    fn starter() -> JsonValue {
        let (_, _, text) = crate::registry::STARTERS
            .iter()
            .find(|(_, stem, _)| *stem == "simple-forward")
            .expect("the simple-forward starter is registered");
        parse(text).expect("the Simple forward starter parses")
    }

    /// The `key` field of a JSON object.
    fn text_field<'a>(value: &'a JsonValue, key: &str) -> &'a str {
        let JsonValue::Object(fields) = value else {
            panic!("a JSON object is expected")
        };
        match fields
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
        {
            Some(JsonValue::String(text)) => text,
            _ => panic!("the object names `{key}` as a string"),
        }
    }

    /// The array stored under `key` of the starter document.
    fn array<'a>(doc: &'a mut JsonValue, key: &str) -> &'a mut Vec<JsonValue> {
        let JsonValue::Object(fields) = doc else {
            panic!("the starter is a JSON object")
        };
        match fields.iter_mut().find(|(name, _)| name == key) {
            Some((_, JsonValue::Array(items))) => items,
            _ => panic!("the starter holds a `{key}` array"),
        }
    }

    fn link(from: &str, to: &str) -> JsonValue {
        JsonValue::Object(vec![
            ("from".to_string(), JsonValue::String(from.to_string())),
            ("to".to_string(), JsonValue::String(to.to_string())),
        ])
    }

    /// A Simple forward plan with the nodes `nodes` (id, type) and the links `links` (from, to) added,
    /// and `output.display` reading `display` in place of `tonemap`.
    fn plan_of(nodes: &[(&str, &str)], links: &[(&str, &str)], display: &str) -> GraphPlan {
        let mut doc = starter();
        let doc_links = array(&mut doc, "links");
        let into_display = doc_links
            .iter()
            .position(|link| text_field(link, "to") == "out.color")
            .expect("the starter links into out.color");
        let JsonValue::Object(fields) = &mut doc_links[into_display] else {
            panic!("a link is a JSON object")
        };
        let from = fields
            .iter_mut()
            .find(|(name, _)| name == "from")
            .expect("a link names its source");
        from.1 = JsonValue::String(display.to_string());
        doc_links.extend(links.iter().map(|(from, to)| link(from, to)));
        array(&mut doc, "nodes").extend(nodes.iter().map(|(id, node_type)| {
            JsonValue::Object(vec![
                ("id".to_string(), JsonValue::String(id.to_string())),
                ("type".to_string(), JsonValue::String(node_type.to_string())),
            ])
        }));
        let mut text = String::new();
        write_pretty(&doc, &mut text);
        crate::plan::plan_from_json(&text).expect("the test document plans")
    }

    /// A Simple forward plan with a `view` node of `view_type` displayed in place of `tonemap`,
    /// reading `inputs`.
    fn plan_with(view_type: &str, inputs: &[(&str, &str)]) -> GraphPlan {
        plan_of(&[("view1", view_type)], inputs, "view1.color")
    }

    /// The order index of the plan node `id`.
    fn order_index(plan: &GraphPlan, id: &str) -> u32 {
        plan.nodes
            .iter()
            .find(|node| node.id == id)
            .expect("the node is planned")
            .order_index
    }

    #[test]
    fn one_linked_image_input_is_accepted() {
        let plan = plan_with("view.image", &[("tonemap.color", "view1.src")]);
        assert_eq!(check_plan(&plan, &Catalog::build()), Vec::new());
    }

    #[test]
    fn a_second_linked_image_input_is_taken() {
        let plan = plan_with(
            "view.image",
            &[
                ("tonemap.color", "view1.src"),
                ("prepass.depth", "view1.srcDepth"),
            ],
        );
        let issues = check_plan(&plan, &Catalog::build());
        assert_eq!(issues.len(), 1, "{issues:?}");
        let issue = &issues[0];
        assert_eq!(issue.code, "graph.input_taken");
        assert_eq!(issue.node.as_deref(), Some("view1"));
        assert_eq!(issue.port.as_deref(), Some("srcDepth"));
        assert_eq!(issue.stage, Stage::Apply);
        assert_eq!(issue.severity, Severity::Error);
    }

    #[test]
    fn missing_view_inputs_are_named() {
        let grid = plan_with("view.grid", &[]);
        let issues = check_plan(&grid, &Catalog::build());
        assert_eq!(issues.len(), 1, "{issues:?}");
        assert_eq!(issues[0].code, "graph.input_missing");
        assert_eq!(issues[0].port.as_deref(), Some("t0"));

        let image = plan_with("view.image", &[]);
        let issues = check_plan(&image, &Catalog::build());
        assert_eq!(issues.len(), 1, "{issues:?}");
        assert_eq!(issues[0].code, "graph.input_missing");
        assert_eq!(issues[0].port.as_deref(), Some("src"));
    }

    #[test]
    fn compare_and_grid_sources_validate() {
        Composer::validate(&view_compare_source()).unwrap_or_else(|e| panic!("compare: {e}"));
        Composer::validate(&view_grid_source()).unwrap_or_else(|e| panic!("grid: {e}"));
    }

    #[test]
    fn labels_pack_little_endian_and_truncate() {
        let mut out = [7u32; 8];
        pack_label("ao", &mut out);
        assert_eq!(out, [0x6f61, 0, 0, 0, 0, 0, 0, 0]);
        let long: String = (0..40u8).map(|i| char::from(b'A' + i % 26)).collect();
        pack_label(&long, &mut out);
        assert_eq!(out[7] >> 24, u32::from(long.as_bytes()[31]));
        assert_eq!(out[0], u32::from_le_bytes(*b"ABCD"));
        pack_label("\u{e9}", &mut out);
        assert_eq!(out, [0x3f, 0, 0, 0, 0, 0, 0, 0]);
    }

    fn font_glyphs() -> Vec<u64> {
        let source = view_grid_source();
        let start = source.find("const VIEW_FONT").expect("VIEW_FONT");
        let mut glyphs = Vec::new();
        for line in source[start..].lines().skip(1) {
            let Some(rest) = line.trim().strip_prefix("vec2<u32>(") else {
                break;
            };
            let inner = &rest[..rest.find(')').expect("closing parenthesis")];
            let lanes: Vec<u64> = inner
                .split(',')
                .map(|n| {
                    let hex = n.trim().trim_start_matches("0x").trim_end_matches('u');
                    u64::from_str_radix(hex, 16).expect("hex lane")
                })
                .collect();
            glyphs.push(lanes[0] | (lanes[1] << 32));
        }
        glyphs
    }

    #[test]
    fn font_has_blank_space_and_visible_digits_and_capitals() {
        let glyphs = font_glyphs();
        assert_eq!(glyphs.len(), 95);
        assert_eq!(glyphs[0], 0);
        for c in ('0'..='9').chain('A'..='Z') {
            assert_ne!(glyphs[c as usize - 32], 0, "glyph {c}");
        }
    }

    #[test]
    fn view_pass_draws_an_image_and_publishes_its_colour() {
        let test = "view_pass_draws_an_image_and_publishes_its_colour";
        let Some(gpu) = software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let size = (8, 8);
        let targets = Targets::new(&gpu, size);
        let mut lighting = LightingResources::new(&gpu);
        let resources = ResourceTable::new();
        let backbuffer = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("view test backbuffer"),
            size: wgpu::Extent3d {
                width: 8,
                height: 8,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let backbuffer_view = backbuffer.create_view(&wgpu::TextureViewDescriptor::default());
        let mut image = ViewPass::new(ViewPassSpec {
            name: "view#t".to_string(),
            order: 1,
            overlay: false,
            display: true,
            kind: ViewKind::Image(IDENTITY),
        });
        let mut grid = ViewPass::new(ViewPassSpec {
            name: "view#grid".to_string(),
            order: 2,
            overlay: false,
            display: false,
            kind: ViewKind::Grid {
                columns: 2,
                labels: true,
                tiles: vec![("t0", "depth".to_string()), ("t1", "colour".to_string())],
            },
        });
        let errors = gpu.device().push_error_scope(wgpu::ErrorFilter::Validation);
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        // No mesh pass writes depth in this test, so clear it to 1.0 (far) before the image reads it.
        drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("view test depth clear"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: targets.depth(),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        }));
        {
            let mut prepare = PrepareCtx {
                gpu: &gpu,
                size,
                frame: 0,
                time_s: 0.0,
                dt_s: 0.0,
                targets: &targets,
                lighting: &mut lighting,
                resources: &resources,
            };
            image
                .prepare(&mut prepare)
                .expect("the image pass prepares");
            grid.prepare(&mut prepare).expect("the grid pass prepares");
        }
        // The views graph frame binds the inputs before each record: the depth target to the image,
        // and the image's published colour to both grid tiles.
        let depth = PublishedImage {
            view: targets.depth().clone(),
            desc: ImageDesc::new(ImageDim::D2, wgpu::TextureFormat::Depth32Float, 8, 8, 1, 1),
        };
        resources.bind_input("view#t", "srcDepth", depth);
        let colour = resources
            .get("view#t", "color")
            .expect("the image pass publishes its colour");
        resources.bind_input("view#grid", "t0", colour.clone());
        resources.bind_input("view#grid", "t1", colour);
        {
            let mut record = RecordCtx {
                gpu: &gpu,
                encoder: &mut encoder,
                size,
                frame: 0,
                time_s: 0.0,
                targets: &targets,
                lighting: &lighting,
                resources: &resources,
                backbuffer: &backbuffer_view,
                backbuffer_format: wgpu::TextureFormat::Rgba8Unorm,
                surface_is_srgb: false,
            };
            image.record(&mut record).expect("the image pass records");
            grid.record(&mut record).expect("the grid pass records");
        }
        gpu.queue().submit(Some(encoder.finish()));
        let _ = gpu.device().poll(wgpu::PollType::wait_indefinitely());
        let error = block_on(errors.pop());
        assert!(error.is_none(), "view frame: {error:?}");
        let colour_desc = ImageDesc::new(ImageDim::D2, COLOUR_FORMAT, 8, 8, 1, 1);
        let entries = resources.entries();
        assert!(
            entries.contains(&("view#t".to_string(), "color".to_string(), colour_desc)),
            "{entries:?}"
        );
    }

    #[test]
    fn a_colour_mip_chain_sums_its_texels_by_hand() {
        let desc = ImageDesc::new(ImageDim::D2, wgpu::TextureFormat::Rgba16Float, 4, 2, 1, 3);
        let stat = resource_stat("color", &desc);
        // Mips 4x2, 2x1 and 1x1 hold 8 + 2 + 1 texels of 8 bytes each.
        assert_eq!(stat.bytes, 88);
        assert_eq!(stat.format, "rgba16float");
        assert_eq!(
            (stat.width, stat.height, stat.layers, stat.mips),
            (4, 2, 1, 3)
        );
    }

    #[test]
    fn a_depth_image_counts_its_depth_aspect_over_its_layers() {
        let desc = ImageDesc::new(
            ImageDim::D2Array,
            wgpu::TextureFormat::Depth32Float,
            2,
            2,
            4,
            1,
        );
        // 2x2 texels of 4 bytes in each of 4 layers.
        let stat = resource_stat("depth", &desc);
        assert_eq!(stat.bytes, 64);
        assert_eq!(stat.format, "depth32float");
    }

    #[test]
    fn a_view_on_the_display_runs_as_an_overlay_after_tonemap() {
        let plan = plan_of(
            &[("show", "view.image")],
            &[("prepass.depth", "show.srcDepth")],
            "show.color",
        );
        let order = 1000 + order_index(&plan, "show");
        assert_eq!(
            placement(&plan, &Catalog::build()),
            vec![("view#show".to_string(), order, true, true)]
        );
    }

    #[test]
    fn a_view_off_the_display_runs_right_after_its_source_pass() {
        let plan = plan_of(
            &[("depth", "view.image"), ("grid", "view.grid")],
            &[
                ("prepass.depth", "depth.srcDepth"),
                ("depth.color", "grid.t0"),
            ],
            "grid.color",
        );
        let placed = placement(&plan, &Catalog::build());
        // prepass runs at order 30; the view takes that order and sits right after it.
        assert!(
            placed.contains(&("view#depth".to_string(), 30, false, false)),
            "{placed:?}"
        );
        let grid = 1000 + order_index(&plan, "grid");
        assert!(
            placed.contains(&("view#grid".to_string(), grid, true, true)),
            "{placed:?}"
        );
    }

    /// A stand-in pass: the pipeline needs only its name and order to place it.
    struct Stand(&'static str, u32);

    impl FeaturePass for Stand {
        fn name(&self) -> &str {
            self.0
        }

        fn order(&self) -> u32 {
            self.1
        }

        fn record(&mut self, _ctx: &mut RecordCtx<'_>) -> Result<(), RaaError> {
            Ok(())
        }
    }

    #[test]
    fn a_refused_plan_restores_the_previous_views_and_clear_removes_them() {
        let test = "a_refused_plan_restores_the_previous_views_and_clear_removes_them";
        let Some(gpu) = software_gpu() else {
            skip(test, "software_gpu returned None");
            return;
        };
        let pipeline = Rc::new(RefCell::new(Pipeline::new(
            &gpu,
            wgpu::TextureFormat::Rgba8Unorm,
            false,
        )));
        {
            // A foreign entry named like the late view makes the pipeline refuse that view's pass.
            let mut pipeline = pipeline.borrow_mut();
            pipeline
                .add(Box::new(Stand("tonemap", 200)))
                .expect("tonemap is added");
            pipeline
                .add(Box::new(Stand("view#late", 30)))
                .expect("the foreign entry is added");
        }
        let mut handles = HandleSet::default();
        let mut h = FeatureHandles {
            gpu: &gpu,
            pipeline: &pipeline,
            handles: &mut handles,
            viewports: 1,
            msaa: 1,
            lit: None,
        };
        let mut views = ViewsExtension::new();
        let shown = plan_of(
            &[("show", "view.image")],
            &[("prepass.depth", "show.srcDepth")],
            "show.color",
        );
        views
            .apply(&shown, None, &mut h)
            .expect("the display view installs");
        let late = plan_of(
            &[("late", "view.image")],
            &[("prepass.depth", "late.srcDepth")],
            "late.color",
        );
        let issues = views
            .apply(&late, Some(&shown), &mut h)
            .expect_err("the foreign entry refuses the late view");
        assert_eq!(
            (issues[0].code, issues[0].node.as_deref()),
            ("graph.order", Some("late"))
        );
        // The refusal changes nothing: the previous view is installed again, the foreign entry stays.
        assert_eq!(pipeline.borrow().entry_enabled("view#show"), Some(true));
        assert_eq!(pipeline.borrow().entry_enabled("view#late"), Some(true));
        views.clear(&mut h);
        assert_eq!(pipeline.borrow().entry_enabled("view#show"), None);
        assert!(
            pipeline
                .borrow_mut()
                .remove_graph_frame(VIEWS_FRAME)
                .is_none()
        );
    }
}
