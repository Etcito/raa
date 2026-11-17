//! Family `view`: the view nodes `view.image`, `view.compare` and `view.grid` (S55). This file
//! declares their catalog data; the `views` extension (`exec_ext/views.rs`) draws them.

use crate::catalog::{
    Category, GraphFormat, NodeFamily, NodeTypeSpec, ParamBinding, ParamKind, ParamSpec,
    ParamValue, PortSpec, PortType, Rebuild, ShaderStage, SizeSpec, StageBudget,
};

/// The `view` family, discovered by build.rs.
pub fn family() -> Box<dyn NodeFamily> {
    Box::new(ViewFamily)
}

struct ViewFamily;

/// The sixteen tile inputs of `view.grid`, in port order.
const TILES: [&str; 16] = [
    "t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11", "t12", "t13", "t14",
    "t15",
];

impl NodeFamily for ViewFamily {
    fn name(&self) -> &'static str {
        "view"
    }

    fn node_types(&self) -> Vec<NodeTypeSpec> {
        vec![
            view_type(
                "view.image",
                "View image",
                "Shows one image as colour: a channel, a range, an exposure, a mip or a layer.",
                300,
                image_inputs(),
                image_params(),
                fragment(1, 1),
            ),
            view_type(
                "view.compare",
                "View compare",
                "Compares two images: split at a position, side by side or as a difference.",
                301,
                compare_inputs(),
                compare_params(),
                fragment(2, 0),
            ),
            view_type(
                "view.grid",
                "View grid",
                "Tiles up to sixteen images into one grid, each tile labelled.",
                302,
                grid_inputs(),
                grid_params(),
                fragment(16, 0),
            ),
        ]
    }

    /// The required passes always run, so a graph whose display reads a view keeps them planned.
    fn extend_types(&self, types: &mut Vec<NodeTypeSpec>) {
        for spec in types.iter_mut() {
            if matches!(spec.id, "pass.prepass" | "pass.forward" | "pass.tonemap") {
                spec.side_effect = true;
            }
        }
    }
}

/// A view node type: multi-instance, one `color` output, no pass of its own.
fn view_type(
    id: &'static str,
    label: &'static str,
    description: &'static str,
    default_order: u32,
    inputs: Vec<PortSpec>,
    params: Vec<ParamSpec>,
    budgets: Vec<StageBudget>,
) -> NodeTypeSpec {
    NodeTypeSpec {
        id,
        version: 1,
        category: Category::View,
        label,
        description,
        pass: None,
        contributor: false,
        default_order,
        multi_instance: true,
        side_effect: false,
        required_in_graph: false,
        inputs,
        outputs: vec![color()],
        params,
        budgets,
        upgrade: no_upgrade,
        family: "view",
    }
}

#[allow(clippy::ptr_arg)]
fn no_upgrade(from: u32, _params: &mut Vec<(String, ParamValue)>) -> Result<(), String> {
    Err(format!("no upgrade from version {from}"))
}

/// The fragment budget of a view node: its sampled textures and samplers, one uniform buffer.
fn fragment(sampled_textures: u32, samplers: u32) -> Vec<StageBudget> {
    vec![StageBudget {
        stage: ShaderStage::Fragment,
        sampled_textures,
        samplers,
        storage_textures: 0,
        storage_buffers: 0,
        uniform_buffers: 1,
    }]
}

/// The `color` output: a full-size Rgba16Float image.
fn color() -> PortSpec {
    PortSpec {
        format: Some(GraphFormat::Rgba16Float),
        absent_ok: false,
        generic: false,
        ..input("color", "Color", PortType::Image2d, false)
    }
}

/// An image input of a view node: any format, generic, absent unless `required`.
fn input(key: &'static str, label: &'static str, ty: PortType, required: bool) -> PortSpec {
    PortSpec {
        key,
        label,
        ty,
        format: None,
        size: SizeSpec::Scale(1.0),
        mips: 1,
        layers: 1,
        required,
        default_source: &[],
        absent_ok: !required,
        generic: true,
        history: false,
        slot: None,
    }
}

fn image_inputs() -> Vec<PortSpec> {
    vec![
        input("src", "Image", PortType::Image2d, false),
        input("srcDepth", "Depth", PortType::Depth2d, false),
        input("srcArray", "Array", PortType::Image2dArray, false),
        input(
            "srcDepthArray",
            "Depth array",
            PortType::Depth2dArray,
            false,
        ),
        input("src3d", "3D", PortType::Image3d, false),
        input("srcCube", "Cube", PortType::ImageCube, false),
    ]
}

fn compare_inputs() -> Vec<PortSpec> {
    vec![
        input("a", "A", PortType::Image2d, true),
        input("b", "B", PortType::Image2d, true),
    ]
}

fn grid_inputs() -> Vec<PortSpec> {
    TILES
        .iter()
        .map(|&key| input(key, key, PortType::Image2d, false))
        .collect()
}

/// A view parameter: group "View", bound to its node, with no rebuild, step or unit.
fn param(
    key: &'static str,
    label: &'static str,
    default: ParamValue,
    range: (Option<f64>, Option<f64>),
    source: &'static str,
) -> ParamSpec {
    let kind = match default {
        ParamValue::Number(_) => ParamKind::Number,
        ParamValue::Int(_) => ParamKind::Int,
        ParamValue::Bool(_) => ParamKind::Bool,
        ParamValue::String(_) => ParamKind::String,
        _ => ParamKind::Enum,
    };
    let (min, max) = range;
    ParamSpec {
        key,
        label,
        group: "View",
        kind,
        default,
        min,
        max,
        step: None,
        options: &[],
        unit: "",
        rebuild: Rebuild::None,
        primary: false,
        unstated_range: false,
        binding: ParamBinding::Node,
        source,
    }
}

fn image_params() -> Vec<ParamSpec> {
    vec![
        ParamSpec {
            options: &["rgb", "r", "g", "b", "a", "rgba-alpha"],
            primary: true,
            ..param(
                "channels",
                "Channels",
                ParamValue::Enum("rgb".to_string()),
                (None, None),
                "raa-graph/src/nodes/view.rs:217",
            )
        },
        param(
            "rangeMin",
            "Range min",
            ParamValue::Number(0.0),
            (None, None),
            "raa-graph/src/nodes/view.rs:225",
        ),
        param(
            "rangeMax",
            "Range max",
            ParamValue::Number(1.0),
            (None, None),
            "raa-graph/src/nodes/view.rs:232",
        ),
        ParamSpec {
            unit: "EV",
            ..param(
                "exposureEv",
                "Exposure",
                ParamValue::Number(0.0),
                (None, None),
                "raa-graph/src/nodes/view.rs:241",
            )
        },
        param(
            "linearizeDepth",
            "Linearize",
            ParamValue::Bool(false),
            (None, None),
            "raa-graph/src/nodes/view.rs:249",
        ),
        param(
            "layer",
            "Layer",
            ParamValue::Int(0),
            (Some(0.0), None),
            "raa-graph/src/nodes/view.rs:256",
        ),
        param(
            "slice",
            "Slice",
            ParamValue::Int(0),
            (Some(0.0), None),
            "raa-graph/src/nodes/view.rs:263",
        ),
        param(
            "mip",
            "Mip",
            ParamValue::Int(0),
            (Some(0.0), None),
            "raa-graph/src/nodes/view.rs:270",
        ),
        ParamSpec {
            options: &["none", "half2lo", "half2hi"],
            ..param(
                "decode",
                "Decode",
                ParamValue::Enum("none".to_string()),
                (None, None),
                "raa-graph/src/nodes/view.rs:279",
            )
        },
        ParamSpec {
            primary: true,
            ..param(
                "label",
                "Label",
                ParamValue::String(String::new()),
                (None, None),
                "raa-graph/src/nodes/view.rs:289",
            )
        },
    ]
}

fn compare_params() -> Vec<ParamSpec> {
    vec![
        ParamSpec {
            options: &["split", "sideBySide", "difference"],
            primary: true,
            ..param(
                "mode",
                "Mode",
                ParamValue::Enum("split".to_string()),
                (None, None),
                "raa-graph/src/nodes/view.rs:305",
            )
        },
        param(
            "position",
            "Position",
            ParamValue::Number(0.5),
            (Some(0.0), Some(1.0)),
            "raa-graph/src/nodes/view.rs:313",
        ),
    ]
}

fn grid_params() -> Vec<ParamSpec> {
    vec![
        ParamSpec {
            primary: true,
            ..param(
                "columns",
                "Columns",
                ParamValue::Int(4),
                (Some(1.0), Some(4.0)),
                "raa-graph/src/nodes/view.rs:327",
            )
        },
        param(
            "labels",
            "Labels",
            ParamValue::Bool(true),
            (None, None),
            "raa-graph/src/nodes/view.rs:335",
        ),
    ]
}
