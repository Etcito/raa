//! The render graph catalog types and functions (S51): node types, their ports,
//! parameters and budgets, the built catalog and its JSON form.
//!
//! `crates/raa-graph/src/lib.rs`: out of scope for this share: consumers use the module
//! path (`raa_graph::catalog::Catalog`).

use raa_loader::json::JsonValue;

/// The kind of image or buffer a port carries.
#[derive(Clone, Debug, PartialEq)]
pub enum PortType {
    Image2d,
    Image2dArray,
    Image3d,
    ImageCube,
    Depth2d,
    Depth2dArray,
    Buffer,
}

/// A texture format named by a port or a compute declaration.
#[derive(Clone, Debug, PartialEq)]
pub enum GraphFormat {
    R8Unorm,
    Rg8Unorm,
    Rgba8Unorm,
    Rgba8UnormSrgb,
    Bgra8Unorm,
    R16Float,
    Rg16Float,
    Rgba16Float,
    R32Float,
    Rg32Float,
    Rgba32Float,
    R32Uint,
    Rg32Uint,
    Rgba32Uint,
    Rgb10a2Unorm,
    Rg11b10Ufloat,
    Depth32Float,
    Surface,
}

impl GraphFormat {
    /// The WebGPU name of the format (`surface` for the canvas format).
    pub fn name(&self) -> &'static str {
        match self {
            GraphFormat::R8Unorm => "r8unorm",
            GraphFormat::Rg8Unorm => "rg8unorm",
            GraphFormat::Rgba8Unorm => "rgba8unorm",
            GraphFormat::Rgba8UnormSrgb => "rgba8unorm-srgb",
            GraphFormat::Bgra8Unorm => "bgra8unorm",
            GraphFormat::R16Float => "r16float",
            GraphFormat::Rg16Float => "rg16float",
            GraphFormat::Rgba16Float => "rgba16float",
            GraphFormat::R32Float => "r32float",
            GraphFormat::Rg32Float => "rg32float",
            GraphFormat::Rgba32Float => "rgba32float",
            GraphFormat::R32Uint => "r32uint",
            GraphFormat::Rg32Uint => "rg32uint",
            GraphFormat::Rgba32Uint => "rgba32uint",
            GraphFormat::Rgb10a2Unorm => "rgb10a2unorm",
            GraphFormat::Rg11b10Ufloat => "rg11b10ufloat",
            GraphFormat::Depth32Float => "depth32float",
            GraphFormat::Surface => "surface",
        }
    }

    /// The format of that WebGPU name, `None` when absent.
    pub fn from_name(name: &str) -> Option<GraphFormat> {
        Some(match name {
            "r8unorm" => GraphFormat::R8Unorm,
            "rg8unorm" => GraphFormat::Rg8Unorm,
            "rgba8unorm" => GraphFormat::Rgba8Unorm,
            "rgba8unorm-srgb" => GraphFormat::Rgba8UnormSrgb,
            "bgra8unorm" => GraphFormat::Bgra8Unorm,
            "r16float" => GraphFormat::R16Float,
            "rg16float" => GraphFormat::Rg16Float,
            "rgba16float" => GraphFormat::Rgba16Float,
            "r32float" => GraphFormat::R32Float,
            "rg32float" => GraphFormat::Rg32Float,
            "rgba32float" => GraphFormat::Rgba32Float,
            "r32uint" => GraphFormat::R32Uint,
            "rg32uint" => GraphFormat::Rg32Uint,
            "rgba32uint" => GraphFormat::Rgba32Uint,
            "rgb10a2unorm" => GraphFormat::Rgb10a2Unorm,
            "rg11b10ufloat" => GraphFormat::Rg11b10Ufloat,
            "depth32float" => GraphFormat::Depth32Float,
            "surface" => GraphFormat::Surface,
            _ => return None,
        })
    }

    /// The wgpu format; `Surface` is the canvas format passed in.
    pub fn to_wgpu(&self, surface: wgpu::TextureFormat) -> wgpu::TextureFormat {
        match self {
            GraphFormat::R8Unorm => wgpu::TextureFormat::R8Unorm,
            GraphFormat::Rg8Unorm => wgpu::TextureFormat::Rg8Unorm,
            GraphFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8Unorm,
            GraphFormat::Rgba8UnormSrgb => wgpu::TextureFormat::Rgba8UnormSrgb,
            GraphFormat::Bgra8Unorm => wgpu::TextureFormat::Bgra8Unorm,
            GraphFormat::R16Float => wgpu::TextureFormat::R16Float,
            GraphFormat::Rg16Float => wgpu::TextureFormat::Rg16Float,
            GraphFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
            GraphFormat::R32Float => wgpu::TextureFormat::R32Float,
            GraphFormat::Rg32Float => wgpu::TextureFormat::Rg32Float,
            GraphFormat::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
            GraphFormat::R32Uint => wgpu::TextureFormat::R32Uint,
            GraphFormat::Rg32Uint => wgpu::TextureFormat::Rg32Uint,
            GraphFormat::Rgba32Uint => wgpu::TextureFormat::Rgba32Uint,
            GraphFormat::Rgb10a2Unorm => wgpu::TextureFormat::Rgb10a2Unorm,
            GraphFormat::Rg11b10Ufloat => wgpu::TextureFormat::Rg11b10Ufloat,
            GraphFormat::Depth32Float => wgpu::TextureFormat::Depth32Float,
            GraphFormat::Surface => surface,
        }
    }
}

/// The JSON names of the port types, in S51 order.
const PORT_TYPES: [&str; 7] = [
    "image2d",
    "image2dArray",
    "image3d",
    "imageCube",
    "depth2d",
    "depth2dArray",
    "buffer",
];

fn port_type_name(ty: &PortType) -> &'static str {
    match ty {
        PortType::Image2d => "image2d",
        PortType::Image2dArray => "image2dArray",
        PortType::Image3d => "image3d",
        PortType::ImageCube => "imageCube",
        PortType::Depth2d => "depth2d",
        PortType::Depth2dArray => "depth2dArray",
        PortType::Buffer => "buffer",
    }
}

/// How the size of a resource is chosen.
#[derive(Clone, Debug, PartialEq)]
pub enum SizeSpec {
    Scale(f32),
    Fixed { width: u32, height: u32 },
    FromParam(&'static str),
    Input(&'static str),
}

/// The kind of a parameter value.
#[derive(Clone, Debug, PartialEq)]
pub enum ParamKind {
    Number,
    Int,
    Bool,
    Enum,
    Vec2,
    Vec3,
    Vec4,
    String,
}

/// A parameter value.
#[derive(Clone, Debug, PartialEq)]
pub enum ParamValue {
    Number(f64),
    Int(i64),
    Bool(bool),
    Enum(String),
    Vec2([f64; 2]),
    Vec3([f64; 3]),
    Vec4([f64; 4]),
    String(String),
}

impl ParamValue {
    /// The S51 JSON form of the value.
    pub fn to_json(&self) -> JsonValue {
        match self {
            ParamValue::Number(n) => JsonValue::Number(*n),
            ParamValue::Int(i) => JsonValue::Number(*i as f64),
            ParamValue::Bool(b) => JsonValue::Bool(*b),
            ParamValue::Enum(s) | ParamValue::String(s) => JsonValue::String(s.clone()),
            ParamValue::Vec2(v) => {
                JsonValue::Array(v.iter().map(|n| JsonValue::Number(*n)).collect())
            }
            ParamValue::Vec3(v) => {
                JsonValue::Array(v.iter().map(|n| JsonValue::Number(*n)).collect())
            }
            ParamValue::Vec4(v) => {
                JsonValue::Array(v.iter().map(|n| JsonValue::Number(*n)).collect())
            }
        }
    }

    /// Converts one JSON value following the S50 rules of `kind`. Range and option checks
    /// stay with the validator.
    pub fn from_json(kind: &ParamKind, v: &JsonValue) -> Result<ParamValue, String> {
        match (kind, v) {
            (ParamKind::Number, JsonValue::Number(n)) if finite_as_f32(*n) => {
                Ok(ParamValue::Number(*n))
            }
            (ParamKind::Int, JsonValue::Number(n))
                if n.is_finite()
                    && n.fract() == 0.0
                    && *n >= -2147483648.0
                    && *n < 2147483648.0 =>
            {
                Ok(ParamValue::Int(*n as i64))
            }
            (ParamKind::Bool, JsonValue::Bool(b)) => Ok(ParamValue::Bool(*b)),
            (ParamKind::Enum, JsonValue::String(s)) => Ok(ParamValue::Enum(s.clone())),
            (ParamKind::String, JsonValue::String(s)) => Ok(ParamValue::String(s.clone())),
            (ParamKind::Vec2, value) => Ok(ParamValue::Vec2(float_array::<2>(value, kind)?)),
            (ParamKind::Vec3, value) => Ok(ParamValue::Vec3(float_array::<3>(value, kind)?)),
            (ParamKind::Vec4, value) => Ok(ParamValue::Vec4(float_array::<4>(value, kind)?)),
            _ => Err(expected_value(kind_name(kind), v)),
        }
    }
}

/// How a change of a parameter is applied.
#[derive(Clone, Debug, PartialEq)]
pub enum Rebuild {
    None,
    Pipeline,
    Resources,
    Engine,
}

/// The feature that owns a parameter.
#[derive(Clone, Debug, PartialEq)]
pub enum FeatureId {
    Pipeline,
    Tonemap,
    Shadow,
    Vg,
    GiQuality,
    Gi,
    AreaShadows,
    AreaShading,
    Gtao,
    Contact,
    Ssr,
    Fog,
    Shafts,
    Oit,
    Taa,
    Dof,
    Bloom,
    Grading,
    Cel,
    Sky,
    Other(&'static str),
}

/// Where a parameter's value is written.
#[derive(Clone, Debug, PartialEq)]
pub enum ParamBinding {
    Setting {
        feature: FeatureId,
        key: &'static str,
    },
    Tunable {
        feature: FeatureId,
        key: &'static str,
    },
    Engine {
        option: &'static str,
    },
    Node,
}

/// The catalog category of a node type.
#[derive(Clone, Debug, PartialEq)]
pub enum Category {
    Geometry,
    Lighting,
    Gi,
    Shadows,
    Post,
    Compute,
    View,
    Output,
}

impl Category {
    /// The menu and catalog order of the category.
    pub fn rank(&self) -> u8 {
        match self {
            Category::Geometry => 0,
            Category::Lighting => 1,
            Category::Gi => 2,
            Category::Shadows => 3,
            Category::Post => 4,
            Category::Compute => 5,
            Category::View => 6,
            Category::Output => 7,
        }
    }

    /// The S51 JSON name of the category.
    pub fn name(&self) -> &'static str {
        match self {
            Category::Geometry => "geometry",
            Category::Lighting => "lighting",
            Category::Gi => "gi",
            Category::Shadows => "shadows",
            Category::Post => "post",
            Category::Compute => "compute",
            Category::View => "view",
            Category::Output => "output",
        }
    }
}

/// The categories in rank order, as the catalog JSON lists them.
const CATEGORIES: [Category; 8] = [
    Category::Geometry,
    Category::Lighting,
    Category::Gi,
    Category::Shadows,
    Category::Post,
    Category::Compute,
    Category::View,
    Category::Output,
];

/// The physical encoding of an area shadow signal, independent of its texture format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignalEncoding {
    /// Two per-slot (weighted visibility, weight) pairs per layer.
    StratifiedPairs,
    /// One (weighted visibility, weight) aggregate for all slots in layer zero.
    RestirAggregate,
    /// Four resolved per-slot visibility values per layer.
    PackedVisibility,
}

impl SignalEncoding {
    pub const fn name(self) -> &'static str {
        match self {
            Self::StratifiedPairs => "stratifiedPairs",
            Self::RestirAggregate => "restirAggregate",
            Self::PackedVisibility => "packedVisibility",
        }
    }
}

/// Which frame supplies the signal to an input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignalCadence {
    Current,
    Previous,
}

/// Coordinates used by the valid texels of an area shadow signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignalRect {
    FullScene,
}

/// Physical contract projected before resources are allocated. Output usage is provisioned
/// usage, while input usage is the subset the consumer needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SignalContract {
    pub version: u32,
    pub encoding: SignalEncoding,
    pub usage: wgpu::TextureUsages,
    pub samples: u32,
    pub min_layers: u32,
    pub max_layers: u32,
    pub cadence: SignalCadence,
    pub active_rect: SignalRect,
}

/// Signal contracts accepted at a port. Outputs publish exactly one contract.
/// Incompatible contract revisions must also bump the corresponding node type version
/// so older frozen plans are refused by the existing version check.
pub fn port_signals(node: &str, port: &str, output: bool) -> &'static [SignalContract] {
    use SignalEncoding::{PackedVisibility, RestirAggregate, StratifiedPairs};
    const SAMPLE: wgpu::TextureUsages = wgpu::TextureUsages::TEXTURE_BINDING;
    const COPY: wgpu::TextureUsages = wgpu::TextureUsages::COPY_SRC;
    const STORAGE: wgpu::TextureUsages = wgpu::TextureUsages::STORAGE_BINDING;
    const SAMPLE_COPY: wgpu::TextureUsages = SAMPLE.union(COPY);
    const AREA_OUTPUT: wgpu::TextureUsages = SAMPLE_COPY.union(STORAGE);
    const SPATIAL_MASK: wgpu::TextureUsages = COPY.union(STORAGE);
    const LEGACY_MASK: wgpu::TextureUsages = SAMPLE.union(STORAGE);
    const PUBLISHED_MASK: wgpu::TextureUsages = AREA_OUTPUT.union(wgpu::TextureUsages::COPY_DST);
    const PAIRS: SignalContract = SignalContract {
        version: 1,
        encoding: StratifiedPairs,
        usage: SAMPLE,
        samples: 1,
        min_layers: 2,
        max_layers: (raa_pipeline::MAX_STOCHASTIC_LIGHTS as u32).div_ceil(2),
        cadence: SignalCadence::Current,
        active_rect: SignalRect::FullScene,
    };
    const AGGREGATE: SignalContract = SignalContract {
        encoding: RestirAggregate,
        ..PAIRS
    };
    const MASK: SignalContract = SignalContract {
        encoding: PackedVisibility,
        min_layers: raa_pipeline::AREA_SHADOW_MASK_LAYERS,
        max_layers: raa_pipeline::AREA_SHADOW_MASK_LAYERS,
        ..PAIRS
    };

    match (node, port, output) {
        ("area.trace", "raw", true) | ("area.temporal", "filtered", true) => &[SignalContract {
            usage: AREA_OUTPUT,
            ..PAIRS
        }],
        ("area.restir.spatial", "raw", true) => &[SignalContract {
            usage: AREA_OUTPUT,
            ..AGGREGATE
        }],
        ("area.spatial", "mask", true) => &[SignalContract {
            usage: SPATIAL_MASK,
            ..MASK
        }],
        ("area.mask", "areaShadow", true) => &[SignalContract {
            usage: PUBLISHED_MASK,
            ..MASK
        }],
        ("pass.gi.area-shadow", "areaShadow" | "mask", true) => &[SignalContract {
            usage: LEGACY_MASK,
            ..MASK
        }],
        ("neural.area-reconstruction", "pairs", true) => &[SignalContract {
            usage: AREA_OUTPUT,
            ..PAIRS
        }],
        ("neural.area-reconstruction", "pairs", false) => &[PAIRS],
        ("area.temporal", "raw", false) | ("area.spatial", "src", false) => &[PAIRS],
        ("area.temporal", "history", false) => &[SignalContract {
            cadence: SignalCadence::Previous,
            ..PAIRS
        }],
        ("area.mask", "src", false) => &[
            PAIRS,
            AGGREGATE,
            SignalContract {
                usage: COPY,
                ..MASK
            },
        ],
        ("pass.forward", "areaShadow", false) => &[MASK],
        _ => &[],
    }
}

/// An input or output port of a node type.
#[derive(Clone, Debug, PartialEq)]
pub struct PortSpec {
    pub key: &'static str,
    pub label: &'static str,
    pub ty: PortType,
    pub format: Option<GraphFormat>,
    pub size: SizeSpec,
    pub mips: u32,
    pub layers: u32,
    pub required: bool,
    pub default_source: &'static [&'static str],
    pub absent_ok: bool,
    pub generic: bool,
    pub history: bool,
    pub slot: Option<u32>,
}

/// A parameter of a node type.
#[derive(Clone, Debug, PartialEq)]
pub struct ParamSpec {
    pub key: &'static str,
    pub label: &'static str,
    pub group: &'static str,
    pub kind: ParamKind,
    pub default: ParamValue,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub step: Option<f64>,
    pub options: &'static [&'static str],
    pub unit: &'static str,
    pub rebuild: Rebuild,
    pub primary: bool,
    pub unstated_range: bool,
    pub binding: ParamBinding,
    pub source: &'static str,
}

/// The bindings a node type uses in one shader stage.
#[derive(Clone, Debug, PartialEq)]
pub struct StageBudget {
    pub stage: ShaderStage,
    pub sampled_textures: u32,
    pub samplers: u32,
    pub storage_textures: u32,
    pub storage_buffers: u32,
    pub uniform_buffers: u32,
}

/// The shader stage a budget counts.
#[derive(Clone, Debug, PartialEq)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

/// The signature of a node type's parameter upgrade.
type UpgradeFn = fn(from: u32, params: &mut Vec<(String, ParamValue)>) -> Result<(), String>;

/// One node type of the catalog.
#[derive(Clone, Debug)]
pub struct NodeTypeSpec {
    pub id: &'static str,
    pub version: u32,
    pub category: Category,
    pub label: &'static str,
    pub description: &'static str,
    pub pass: Option<&'static str>,
    pub contributor: bool,
    pub default_order: u32,
    pub multi_instance: bool,
    pub side_effect: bool,
    pub required_in_graph: bool,
    pub inputs: Vec<PortSpec>,
    pub outputs: Vec<PortSpec>,
    pub params: Vec<ParamSpec>,
    pub budgets: Vec<StageBudget>,
    pub upgrade: UpgradeFn,
    pub family: &'static str,
}

/// A family of node types the catalog is built from, with validation hooks (S51).
pub trait NodeFamily {
    fn name(&self) -> &'static str;
    fn node_types(&self) -> Vec<NodeTypeSpec>;
    fn extend_types(&self, _types: &mut Vec<NodeTypeSpec>) {}
    fn validate_graph(&self, _doc: &crate::format::GraphDoc, _issues: &mut Vec<crate::error::Issue>) {}
    fn validate_node(
        &self,
        _node: &crate::format::NodeDoc,
        _spec: &NodeTypeSpec,
        _ctx: &crate::validate::ValidateCtx<'_>,
        _issues: &mut Vec<crate::error::Issue>,
    ) {
    }
}

/// The static catalog plus the starter graphs, presets and frozen pipelines.
#[derive(Clone, Debug, PartialEq)]
pub struct Catalog {
    pub nodes: Vec<NodeTypeSpec>,
    pub starters: Vec<Starter>,
    pub presets: Vec<Starter>,
    pub frozen: Vec<&'static str>,
}

/// A starter graph or compute preset shipped with the catalog.
#[derive(Clone, Debug, PartialEq)]
pub struct Starter {
    pub name: &'static str,
    pub file: &'static str,
    pub json: &'static str,
}

/// The `format` value of the catalog JSON (RG_CATALOG_FORMAT).
pub const CATALOG_FORMAT: &str = "raa.render-graph-catalog";

impl Catalog {
    /// Builds the catalog from every registered family, in registry order.
    ///
    /// Panics on a repeated node-type id, or on a repeated input, output or parameter key
    /// within one type: the catalog is static data, so this is a programming error.
    pub fn build() -> Catalog {
        Catalog::build_from(crate::registry::families_all())
    }

    fn build_from(families: Vec<Box<dyn NodeFamily>>) -> Catalog {
        let mut nodes = Vec::new();
        for family in &families {
            nodes.extend(family.node_types());
        }
        for family in &families {
            family.extend_types(&mut nodes);
        }
        for (index, node) in nodes.iter().enumerate() {
            if let Some(earlier) = nodes[..index].iter().find(|earlier| earlier.id == node.id) {
                panic!(
                    "duplicate node type id `{}` (also declared by family `{}`)",
                    node.id, earlier.family
                );
            }
            check_duplicate_keys(node);
        }
        nodes.sort_by(|a, b| {
            a.category
                .rank()
                .cmp(&b.category.rank())
                .then(a.default_order.cmp(&b.default_order))
                .then_with(|| a.id.as_bytes().cmp(b.id.as_bytes()))
        });
        Catalog {
            nodes,
            starters: starters_from(crate::registry::STARTERS),
            presets: starters_from(crate::registry::PRESETS),
            frozen: crate::frozen_names(),
        }
    }

    /// The node type with that id, `None` when absent.
    ///
    /// Handoff: card 'raa-graph validate 1/2: nodes, params, links and canonical_doc'
    /// calls `Catalog::node` to resolve a document node's type and `ParamValue::from_json`
    /// to convert its parameter values; the state passed is `Catalog.nodes`, the node
    /// types by id.
    pub fn node(&self, id: &str) -> Option<&NodeTypeSpec> {
        self.nodes.iter().find(|node| node.id == id)
    }

    /// The S51 catalog JSON (`renderGraphCatalog()`), `available` deciding the flag.
    pub fn to_json(&self, available: &dyn Fn(&NodeTypeSpec) -> bool) -> JsonValue {
        let mut entries: Vec<(String, JsonValue)> = vec![(
            "format".to_string(),
            JsonValue::String(CATALOG_FORMAT.to_string()),
        )];
        entries.push(("version".to_string(), JsonValue::Number(1.0)));
        entries.push(("graphVersion".to_string(), JsonValue::Number(1.0)));
        entries.push((
            "categories".to_string(),
            JsonValue::Array(
                CATEGORIES
                    .iter()
                    .map(|category| JsonValue::String(category.name().to_string()))
                    .collect(),
            ),
        ));
        entries.push((
            "portTypes".to_string(),
            JsonValue::Array(
                PORT_TYPES
                    .iter()
                    .map(|name| JsonValue::String((*name).to_string()))
                    .collect(),
            ),
        ));
        entries.push((
            "nodes".to_string(),
            JsonValue::Array(
                self.nodes
                    .iter()
                    .map(|node| node_json(node, available(node)))
                    .collect(),
            ),
        ));
        entries.push(("starters".to_string(), starters_json(&self.starters)));
        entries.push(("presets".to_string(), starters_json(&self.presets)));
        entries.push((
            "frozen".to_string(),
            JsonValue::Array(
                self.frozen
                    .iter()
                    .map(|name| JsonValue::String((*name).to_string()))
                    .collect(),
            ),
        ));
        JsonValue::Object(entries)
    }
}

fn starters_from(table: &'static [(&'static str, &'static str, &'static str)]) -> Vec<Starter> {
    table
        .iter()
        .map(|&(name, file, json)| Starter { name, file, json })
        .collect()
}

fn check_duplicate_keys(node: &NodeTypeSpec) {
    for (index, port) in node.inputs.iter().enumerate() {
        assert!(
            node.inputs[..index]
                .iter()
                .all(|earlier| earlier.key != port.key),
            "duplicate input port key `{}` in node type `{}`",
            port.key,
            node.id
        );
    }
    for (index, port) in node.outputs.iter().enumerate() {
        assert!(
            node.outputs[..index]
                .iter()
                .all(|earlier| earlier.key != port.key),
            "duplicate output port key `{}` in node type `{}`",
            port.key,
            node.id
        );
    }
    for (index, param) in node.params.iter().enumerate() {
        assert!(
            node.params[..index]
                .iter()
                .all(|earlier| earlier.key != param.key),
            "duplicate parameter key `{}` in node type `{}`",
            param.key,
            node.id
        );
    }
}

fn starters_json(starters: &[Starter]) -> JsonValue {
    JsonValue::Array(
        starters
            .iter()
            .map(|starter| {
                JsonValue::Object(vec![
                    (
                        "name".to_string(),
                        JsonValue::String(starter.name.to_string()),
                    ),
                    (
                        "file".to_string(),
                        JsonValue::String(starter.file.to_string()),
                    ),
                    (
                        "json".to_string(),
                        JsonValue::String(starter.json.to_string()),
                    ),
                ])
            })
            .collect(),
    )
}

fn node_json(node: &NodeTypeSpec, available: bool) -> JsonValue {
    let mut fields = vec![
        ("id".to_string(), JsonValue::String(node.id.to_string())),
        (
            "version".to_string(),
            JsonValue::Number(f64::from(node.version)),
        ),
        (
            "category".to_string(),
            JsonValue::String(node.category.name().to_string()),
        ),
        (
            "label".to_string(),
            JsonValue::String(node.label.to_string()),
        ),
        (
            "description".to_string(),
            JsonValue::String(node.description.to_string()),
        ),
        (
            "pass".to_string(),
            node.pass
                .map_or(JsonValue::Null, |pass| JsonValue::String(pass.to_string())),
        ),
        ("contributor".to_string(), JsonValue::Bool(node.contributor)),
        (
            "defaultOrder".to_string(),
            JsonValue::Number(f64::from(node.default_order)),
        ),
        (
            "multiInstance".to_string(),
            JsonValue::Bool(node.multi_instance),
        ),
        ("sideEffect".to_string(), JsonValue::Bool(node.side_effect)),
        (
            "requiredInGraph".to_string(),
            JsonValue::Bool(node.required_in_graph),
        ),
        ("available".to_string(), JsonValue::Bool(available)),
        (
            "inputs".to_string(),
            JsonValue::Array(
                node.inputs
                    .iter()
                    .map(|port| port_json(node.id, port, false))
                    .collect(),
            ),
        ),
        (
            "outputs".to_string(),
            JsonValue::Array(
                node.outputs
                    .iter()
                    .map(|port| port_json(node.id, port, true))
                    .collect(),
            ),
        ),
        (
            "params".to_string(),
            JsonValue::Array(node.params.iter().map(|param| {
                let mut value = param_json(param);
                if node.family == "neural" && param.key == "model"
                    && let JsonValue::Object(fields) = &mut value
                {
                    let asset = if node.id == "neural.frame-generation" { "generated-frame-model" } else { "neural-model" };
                    fields.push(("asset".into(), JsonValue::String(asset.into())));
                }
                value
            }).collect()),
        ),
    ];
    let capability = match node.id {
        "neural.area-reconstruction" | "neural.super-resolution"
        | "neural.reflection-reconstruction" | "neural.frame-generation" => "experimental-learned",
        _ => return JsonValue::Object(fields),
    };
    fields.push(("capability".into(), JsonValue::String(capability.into())));
    JsonValue::Object(fields)
}

fn port_json(node: &str, port: &PortSpec, output: bool) -> JsonValue {
    let mut json = if output {
        output_json(port)
    } else {
        input_json(port)
    };
    let signals = port_signals(node, port.key, output);
    if let JsonValue::Object(fields) = &mut json
        && !signals.is_empty()
    {
        fields.push((
            "signals".to_string(),
            JsonValue::Array(signals.iter().map(signal_json).collect()),
        ));
    }
    json
}

fn signal_json(signal: &SignalContract) -> JsonValue {
    let usage = [
        (wgpu::TextureUsages::TEXTURE_BINDING, "textureBinding"),
        (wgpu::TextureUsages::STORAGE_BINDING, "storageBinding"),
        (wgpu::TextureUsages::COPY_SRC, "copySrc"),
        (wgpu::TextureUsages::COPY_DST, "copyDst"),
    ]
    .into_iter()
    .filter(|(flag, _)| signal.usage.contains(*flag))
    .map(|(_, name)| JsonValue::String(name.to_string()))
    .collect();
    let cadence = match signal.cadence {
        SignalCadence::Current => "current",
        SignalCadence::Previous => "previous",
    };
    let active_rect = match signal.active_rect {
        SignalRect::FullScene => "fullScene",
    };
    JsonValue::Object(vec![
        (
            "version".to_string(),
            JsonValue::Number(f64::from(signal.version)),
        ),
        (
            "encoding".to_string(),
            JsonValue::String(signal.encoding.name().to_string()),
        ),
        ("usage".to_string(), JsonValue::Array(usage)),
        (
            "samples".to_string(),
            JsonValue::Number(f64::from(signal.samples)),
        ),
        (
            "minLayers".to_string(),
            JsonValue::Number(f64::from(signal.min_layers)),
        ),
        (
            "maxLayers".to_string(),
            JsonValue::Number(f64::from(signal.max_layers)),
        ),
        (
            "cadence".to_string(),
            JsonValue::String(cadence.to_string()),
        ),
        (
            "activeRect".to_string(),
            JsonValue::String(active_rect.to_string()),
        ),
    ])
}

fn input_json(port: &PortSpec) -> JsonValue {
    JsonValue::Object(vec![
        ("key".to_string(), JsonValue::String(port.key.to_string())),
        (
            "label".to_string(),
            JsonValue::String(port.label.to_string()),
        ),
        (
            "type".to_string(),
            JsonValue::String(port_type_name(&port.ty).to_string()),
        ),
        (
            "format".to_string(),
            option_string(port.format.as_ref().map(GraphFormat::name)),
        ),
        ("required".to_string(), JsonValue::Bool(port.required)),
        (
            "defaultSource".to_string(),
            JsonValue::Array(
                port.default_source
                    .iter()
                    .map(|source| JsonValue::String((*source).to_string()))
                    .collect(),
            ),
        ),
        ("absentOk".to_string(), JsonValue::Bool(port.absent_ok)),
        ("generic".to_string(), JsonValue::Bool(port.generic)),
        ("history".to_string(), JsonValue::Bool(port.history)),
        ("slot".to_string(), option_number(port.slot.map(f64::from))),
    ])
}

fn output_json(port: &PortSpec) -> JsonValue {
    JsonValue::Object(vec![
        ("key".to_string(), JsonValue::String(port.key.to_string())),
        (
            "label".to_string(),
            JsonValue::String(port.label.to_string()),
        ),
        (
            "type".to_string(),
            JsonValue::String(port_type_name(&port.ty).to_string()),
        ),
        (
            "format".to_string(),
            option_string(port.format.as_ref().map(GraphFormat::name)),
        ),
        ("size".to_string(), size_json(&port.size)),
        ("mips".to_string(), JsonValue::Number(f64::from(port.mips))),
        (
            "layers".to_string(),
            JsonValue::Number(f64::from(port.layers)),
        ),
    ])
}

fn size_json(size: &SizeSpec) -> JsonValue {
    match size {
        SizeSpec::Scale(scale) => JsonValue::Object(vec![(
            "scale".to_string(),
            JsonValue::Number(f64::from(*scale)),
        )]),
        SizeSpec::Fixed { width, height } => JsonValue::Object(vec![
            ("width".to_string(), JsonValue::Number(f64::from(*width))),
            ("height".to_string(), JsonValue::Number(f64::from(*height))),
        ]),
        SizeSpec::FromParam(param) => JsonValue::Object(vec![(
            "param".to_string(),
            JsonValue::String((*param).to_string()),
        )]),
        SizeSpec::Input(input) => JsonValue::Object(vec![(
            "input".to_string(),
            JsonValue::String((*input).to_string()),
        )]),
    }
}

fn param_json(param: &ParamSpec) -> JsonValue {
    JsonValue::Object(vec![
        ("key".to_string(), JsonValue::String(param.key.to_string())),
        (
            "label".to_string(),
            JsonValue::String(param.label.to_string()),
        ),
        (
            "group".to_string(),
            JsonValue::String(param.group.to_string()),
        ),
        (
            "kind".to_string(),
            JsonValue::String(kind_name(&param.kind).to_string()),
        ),
        ("default".to_string(), param.default.to_json()),
        ("min".to_string(), option_number(param.min)),
        ("max".to_string(), option_number(param.max)),
        ("step".to_string(), option_number(param.step)),
        (
            "options".to_string(),
            if param.options.is_empty() {
                JsonValue::Null
            } else {
                JsonValue::Array(
                    param
                        .options
                        .iter()
                        .map(|option| JsonValue::String((*option).to_string()))
                        .collect(),
                )
            },
        ),
        (
            "unit".to_string(),
            JsonValue::String(param.unit.to_string()),
        ),
        (
            "rebuild".to_string(),
            JsonValue::String(rebuild_name(&param.rebuild).to_string()),
        ),
        ("primary".to_string(), JsonValue::Bool(param.primary)),
        (
            "unstatedRange".to_string(),
            JsonValue::Bool(param.unstated_range),
        ),
        (
            "source".to_string(),
            JsonValue::String(param.source.to_string()),
        ),
    ])
}

fn option_string(value: Option<&'static str>) -> JsonValue {
    value.map_or(JsonValue::Null, |text| JsonValue::String(text.to_string()))
}

fn option_number(value: Option<f64>) -> JsonValue {
    value.map_or(JsonValue::Null, JsonValue::Number)
}

fn kind_name(kind: &ParamKind) -> &'static str {
    match kind {
        ParamKind::Number => "number",
        ParamKind::Int => "int",
        ParamKind::Bool => "bool",
        ParamKind::Enum => "enum",
        ParamKind::Vec2 => "vec2",
        ParamKind::Vec3 => "vec3",
        ParamKind::Vec4 => "vec4",
        ParamKind::String => "string",
    }
}

fn rebuild_name(rebuild: &Rebuild) -> &'static str {
    match rebuild {
        Rebuild::None => "none",
        Rebuild::Pipeline => "pipeline",
        Rebuild::Resources => "resources",
        Rebuild::Engine => "engine",
    }
}

fn finite_as_f32(n: f64) -> bool {
    n.is_finite() && (n as f32).is_finite()
}

fn float_array<const N: usize>(value: &JsonValue, kind: &ParamKind) -> Result<[f64; N], String> {
    let JsonValue::Array(items) = value else {
        return Err(expected_value(kind_name(kind), value));
    };
    if items.len() != N {
        return Err(format!(
            "expected {} ({} numbers), found {} elements",
            kind_name(kind),
            N,
            items.len()
        ));
    }
    let mut out = [0.0; N];
    for (slot, item) in out.iter_mut().zip(items) {
        let JsonValue::Number(n) = item else {
            return Err(expected_value(kind_name(kind), item));
        };
        if !finite_as_f32(*n) {
            return Err(expected_value(kind_name(kind), item));
        }
        *slot = *n;
    }
    Ok(out)
}

fn expected_value(expected: &str, v: &JsonValue) -> String {
    format!("expected {expected}, found {}", json_describe(v))
}

fn json_describe(v: &JsonValue) -> String {
    match v {
        JsonValue::Null => "null".to_string(),
        JsonValue::Bool(b) => format!("boolean {b}"),
        JsonValue::Number(n) => format!("number {}", number_text(*n)),
        JsonValue::String(s) => format!("string {s:?}"),
        JsonValue::Array(items) => format!("an array of {} elements", items.len()),
        JsonValue::Object(_) => "an object".to_string(),
    }
}

fn number_text(n: f64) -> String {
    let mut out = String::new();
    raa_loader::json::format_number(n, &mut out);
    out
}

// Manual `PartialEq`: comparing the `upgrade` function pointer by address is
// unpredictable (`unpredictable_function_pointer_comparisons`), so equality compares
// the data of the spec and ignores the pointer.
impl PartialEq for NodeTypeSpec {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.version == other.version
            && self.category == other.category
            && self.label == other.label
            && self.description == other.description
            && self.pass == other.pass
            && self.contributor == other.contributor
            && self.default_order == other.default_order
            && self.multi_instance == other.multi_instance
            && self.side_effect == other.side_effect
            && self.required_in_graph == other.required_in_graph
            && self.inputs == other.inputs
            && self.outputs == other.outputs
            && self.params == other.params
            && self.budgets == other.budgets
            && self.family == other.family
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> NodeTypeSpec {
        NodeTypeSpec {
            id: "test.node",
            version: 2,
            category: Category::Compute,
            label: "Test node",
            description: "A node for the catalog JSON test.",
            pass: None,
            contributor: false,
            default_order: 7,
            multi_instance: true,
            side_effect: false,
            required_in_graph: false,
            inputs: vec![PortSpec {
                key: "in",
                label: "In",
                ty: PortType::Image2d,
                format: Some(GraphFormat::Rgba16Float),
                size: SizeSpec::Scale(1.0),
                mips: 1,
                layers: 1,
                required: true,
                default_source: &[],
                absent_ok: false,
                generic: false,
                history: false,
                slot: None,
            }],
            outputs: vec![PortSpec {
                key: "out",
                label: "Out",
                ty: PortType::Image2d,
                format: None,
                size: SizeSpec::Fixed {
                    width: 8,
                    height: 4,
                },
                mips: 0,
                layers: 0,
                required: false,
                default_source: &[],
                absent_ok: true,
                generic: true,
                history: true,
                slot: None,
            }],
            params: vec![ParamSpec {
                key: "amount",
                label: "Amount",
                group: "Test",
                kind: ParamKind::Number,
                default: ParamValue::Number(0.5),
                min: Some(0.0),
                max: None,
                step: None,
                options: &[],
                unit: "",
                rebuild: Rebuild::None,
                primary: true,
                unstated_range: false,
                binding: ParamBinding::Node,
                source: "test",
            }],
            budgets: Vec::new(),
            upgrade: |_, _| Ok(()),
            family: "test",
        }
    }

    fn object_keys(value: &JsonValue) -> Vec<&str> {
        let JsonValue::Object(entries) = value else {
            panic!("expected a JSON object, found {value:?}");
        };
        entries.iter().map(|(key, _)| key.as_str()).collect()
    }

    fn field<'a>(value: &'a JsonValue, key: &str) -> &'a JsonValue {
        let JsonValue::Object(entries) = value else {
            panic!("expected a JSON object, found {value:?}");
        };
        let (_, found) = entries
            .iter()
            .find(|(name, _)| name == key)
            .unwrap_or_else(|| panic!("missing key {key}"));
        found
    }

    fn strings(items: &[&str]) -> JsonValue {
        JsonValue::Array(
            items
                .iter()
                .map(|item| JsonValue::String((*item).to_string()))
                .collect(),
        )
    }

    #[test]
    fn to_json_has_the_s51_keys_and_null_encodings() {
        let catalog = Catalog {
            nodes: vec![spec()],
            starters: vec![Starter {
                name: "Simple",
                file: "simple",
                json: "{}",
            }],
            presets: Vec::new(),
            frozen: vec!["simple_forward"],
        };
        let json = catalog.to_json(&|_| false);
        assert_eq!(
            object_keys(&json),
            [
                "format",
                "version",
                "graphVersion",
                "categories",
                "portTypes",
                "nodes",
                "starters",
                "presets",
                "frozen"
            ]
        );
        assert_eq!(
            field(&json, "format"),
            &JsonValue::String(CATALOG_FORMAT.to_string())
        );
        assert_eq!(field(&json, "version"), &JsonValue::Number(1.0));
        assert_eq!(field(&json, "graphVersion"), &JsonValue::Number(1.0));
        assert_eq!(
            field(&json, "categories"),
            &strings(&[
                "geometry", "lighting", "gi", "shadows", "post", "compute", "view", "output"
            ])
        );
        assert_eq!(
            field(&json, "portTypes"),
            &strings(&[
                "image2d",
                "image2dArray",
                "image3d",
                "imageCube",
                "depth2d",
                "depth2dArray",
                "buffer"
            ])
        );

        let JsonValue::Array(nodes) = field(&json, "nodes") else {
            panic!("nodes is not an array");
        };
        let node = &nodes[0];
        assert_eq!(
            object_keys(node),
            [
                "id",
                "version",
                "category",
                "label",
                "description",
                "pass",
                "contributor",
                "defaultOrder",
                "multiInstance",
                "sideEffect",
                "requiredInGraph",
                "available",
                "inputs",
                "outputs",
                "params"
            ]
        );
        assert_eq!(
            field(node, "id"),
            &JsonValue::String("test.node".to_string())
        );
        assert_eq!(field(node, "version"), &JsonValue::Number(2.0));
        assert_eq!(
            field(node, "category"),
            &JsonValue::String("compute".to_string())
        );
        assert_eq!(field(node, "pass"), &JsonValue::Null);
        assert_eq!(field(node, "defaultOrder"), &JsonValue::Number(7.0));
        assert_eq!(field(node, "available"), &JsonValue::Bool(false));

        let JsonValue::Array(inputs) = field(node, "inputs") else {
            panic!("inputs is not an array");
        };
        assert_eq!(
            object_keys(&inputs[0]),
            [
                "key",
                "label",
                "type",
                "format",
                "required",
                "defaultSource",
                "absentOk",
                "generic",
                "history",
                "slot"
            ]
        );
        assert_eq!(
            field(&inputs[0], "type"),
            &JsonValue::String("image2d".to_string())
        );
        assert_eq!(
            field(&inputs[0], "format"),
            &JsonValue::String("rgba16float".to_string())
        );
        assert_eq!(field(&inputs[0], "slot"), &JsonValue::Null);

        let JsonValue::Array(outputs) = field(node, "outputs") else {
            panic!("outputs is not an array");
        };
        assert_eq!(
            object_keys(&outputs[0]),
            ["key", "label", "type", "format", "size", "mips", "layers"]
        );
        assert_eq!(field(&outputs[0], "format"), &JsonValue::Null);
        assert_eq!(
            field(&outputs[0], "size"),
            &JsonValue::Object(vec![
                ("width".to_string(), JsonValue::Number(8.0)),
                ("height".to_string(), JsonValue::Number(4.0)),
            ])
        );

        let JsonValue::Array(params) = field(node, "params") else {
            panic!("params is not an array");
        };
        assert_eq!(
            object_keys(&params[0]),
            [
                "key",
                "label",
                "group",
                "kind",
                "default",
                "min",
                "max",
                "step",
                "options",
                "unit",
                "rebuild",
                "primary",
                "unstatedRange",
                "source"
            ]
        );
        assert_eq!(
            field(&params[0], "kind"),
            &JsonValue::String("number".to_string())
        );
        assert_eq!(field(&params[0], "default"), &JsonValue::Number(0.5));
        assert_eq!(field(&params[0], "min"), &JsonValue::Number(0.0));
        assert_eq!(field(&params[0], "max"), &JsonValue::Null);
        assert_eq!(field(&params[0], "step"), &JsonValue::Null);
        assert_eq!(field(&params[0], "options"), &JsonValue::Null);

        let JsonValue::Array(starters) = field(&json, "starters") else {
            panic!("starters is not an array");
        };
        assert_eq!(object_keys(&starters[0]), ["name", "file", "json"]);
        assert_eq!(field(&json, "presets"), &JsonValue::Array(Vec::new()));
        assert_eq!(field(&json, "frozen"), &strings(&["simple_forward"]));
    }

    #[test]
    fn format_names_round_trip() {
        let formats = [
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
            GraphFormat::Surface,
        ];
        let names = [
            "r8unorm",
            "rg8unorm",
            "rgba8unorm",
            "rgba8unorm-srgb",
            "bgra8unorm",
            "r16float",
            "rg16float",
            "rgba16float",
            "r32float",
            "rg32float",
            "rgba32float",
            "r32uint",
            "rg32uint",
            "rgba32uint",
            "rgb10a2unorm",
            "rg11b10ufloat",
            "depth32float",
            "surface",
        ];
        for (format, name) in formats.into_iter().zip(names) {
            assert_eq!(format.name(), name);
            assert_eq!(GraphFormat::from_name(name), Some(format));
        }
        assert_eq!(GraphFormat::from_name("not-a-format"), None);
    }

    #[test]
    fn surface_maps_to_the_canvas_format() {
        let surface = wgpu::TextureFormat::Bgra8UnormSrgb;
        assert_eq!(GraphFormat::Surface.to_wgpu(surface), surface);
        assert_eq!(
            GraphFormat::Rgba16Float.to_wgpu(surface),
            wgpu::TextureFormat::Rgba16Float
        );
    }

    #[test]
    fn from_json_accepts_each_kind() {
        let cases = [
            (
                ParamKind::Number,
                JsonValue::Number(0.5),
                ParamValue::Number(0.5),
            ),
            (ParamKind::Int, JsonValue::Number(-7.0), ParamValue::Int(-7)),
            (
                ParamKind::Bool,
                JsonValue::Bool(true),
                ParamValue::Bool(true),
            ),
            (
                ParamKind::Enum,
                JsonValue::String("agx".to_string()),
                ParamValue::Enum("agx".to_string()),
            ),
            (
                ParamKind::Vec2,
                JsonValue::Array(vec![JsonValue::Number(1.0), JsonValue::Number(2.0)]),
                ParamValue::Vec2([1.0, 2.0]),
            ),
            (
                ParamKind::Vec3,
                JsonValue::Array(vec![
                    JsonValue::Number(1.0),
                    JsonValue::Number(2.0),
                    JsonValue::Number(3.0),
                ]),
                ParamValue::Vec3([1.0, 2.0, 3.0]),
            ),
            (
                ParamKind::Vec4,
                JsonValue::Array((1..=4).map(|n| JsonValue::Number(f64::from(n))).collect()),
                ParamValue::Vec4([1.0, 2.0, 3.0, 4.0]),
            ),
            (
                ParamKind::String,
                JsonValue::String("hello".to_string()),
                ParamValue::String("hello".to_string()),
            ),
        ];
        for (kind, value, expected) in cases {
            assert_eq!(
                ParamValue::from_json(&kind, &value),
                Ok(expected),
                "{kind:?} from {value:?}"
            );
        }
    }

    #[test]
    fn from_json_refuses_bad_values() {
        assert!(ParamValue::from_json(&ParamKind::Number, &JsonValue::Number(3.5e38)).is_err());
        assert!(ParamValue::from_json(&ParamKind::Int, &JsonValue::Number(2147483648.0)).is_err());
        assert!(ParamValue::from_json(&ParamKind::Int, &JsonValue::Number(1.5)).is_err());
        let two_numbers = JsonValue::Array(vec![JsonValue::Number(1.0), JsonValue::Number(2.0)]);
        assert!(ParamValue::from_json(&ParamKind::Vec3, &two_numbers).is_err());
        assert!(ParamValue::from_json(&ParamKind::Bool, &JsonValue::Number(1.0)).is_err());
        assert!(ParamValue::from_json(&ParamKind::Enum, &JsonValue::Null).is_err());
        assert!(ParamValue::from_json(&ParamKind::String, &JsonValue::Array(Vec::new())).is_err());
    }

    #[test]
    fn build_with_no_families_returns_no_nodes() {
        let catalog = Catalog::build_from(Vec::new());
        assert!(catalog.nodes.is_empty());
        assert!(catalog.node("pass.forward").is_none());
    }
}
