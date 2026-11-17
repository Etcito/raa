//! The render graph catalog types (S51): node types, their ports, parameters and budgets.

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
