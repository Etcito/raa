//! Device-free module generator of the `compute.wgsl` node (S59, contract rg-compute-module):
//! the RgParams struct, the shared prelude, the declared ports, the user WGSL and the entry
//! point. The file names its types only through `super::`, so it compiles twice: inside
//! `nodes/compute.rs`, and in `tests/compute.rs`, which includes it with `#[path]`.

use super::*;

/// The module prelude: `RgFrame` and the group 0 bindings (S59).
const PRELUDE: &str = include_str!("../../wgsl/compute_prelude.wgsl");

/// The entry point appended after the user text (S59).
const ENTRY: &str = concat!(
    "@compute @workgroup_size(8, 8)\n",
    "fn rg_entry(@builtin(global_invocation_id) id: vec3<u32>) {\n",
    "  if (any(id.xy >= rg_frame.size)) {\n",
    "    return;\n",
    "  }\n",
    "  rg_main(id.xy);\n",
    "}\n",
);

/// The eight storage formats a compute output may declare, in contract order.
pub const STORAGE_FORMATS: [GraphFormat; 8] = [
    GraphFormat::Rgba8Unorm,
    GraphFormat::Rgba16Float,
    GraphFormat::R32Float,
    GraphFormat::Rg32Float,
    GraphFormat::Rgba32Float,
    GraphFormat::R32Uint,
    GraphFormat::Rg32Uint,
    GraphFormat::Rgba32Uint,
];

/// The WGSL texel name of a storage format; `None` outside `STORAGE_FORMATS`.
pub fn storage_format_name(format: &GraphFormat) -> Option<&'static str> {
    match format {
        GraphFormat::Rgba8Unorm => Some("rgba8unorm"),
        GraphFormat::Rgba16Float => Some("rgba16float"),
        GraphFormat::R32Float => Some("r32float"),
        GraphFormat::Rg32Float => Some("rg32float"),
        GraphFormat::Rgba32Float => Some("rgba32float"),
        GraphFormat::R32Uint => Some("r32uint"),
        GraphFormat::Rg32Uint => Some("rg32uint"),
        GraphFormat::Rgba32Uint => Some("rgba32uint"),
        _ => None,
    }
}

/// The parameter value of a JSON number, bool or array of 2, 3 or 4 numbers; `None` otherwise.
pub fn infer_param(value: &JsonValue) -> Option<ParamValue> {
    match value {
        JsonValue::Number(number) => Some(ParamValue::Number(*number)),
        JsonValue::Bool(flag) => Some(ParamValue::Bool(*flag)),
        JsonValue::Array(items) => {
            let mut numbers = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    JsonValue::Number(number) => numbers.push(*number),
                    _ => return None,
                }
            }
            match numbers.as_slice() {
                [x, y] => Some(ParamValue::Vec2([*x, *y])),
                [x, y, z] => Some(ParamValue::Vec3([*x, *y, *z])),
                [x, y, z, w] => Some(ParamValue::Vec4([*x, *y, *z, *w])),
                _ => None,
            }
        }
        _ => None,
    }
}

/// The WGSL type of an inferred parameter value; `None` for kinds a compute parameter cannot be.
fn param_type(value: &ParamValue) -> Option<&'static str> {
    match value {
        ParamValue::Number(_) => Some("f32"),
        ParamValue::Bool(_) => Some("u32"),
        ParamValue::Vec2(_) => Some("vec2<f32>"),
        ParamValue::Vec3(_) => Some("vec3<f32>"),
        ParamValue::Vec4(_) => Some("vec4<f32>"),
        _ => None,
    }
}

/// An S52 structure error on `node`, with the port or the parameter it names.
fn issue(
    node: &NodeDoc,
    code: &'static str,
    port: Option<String>,
    param: Option<String>,
    message: String,
    fix: &str,
) -> Issue {
    Issue {
        code,
        severity: Severity::Error,
        stage: Stage::Structure,
        node: Some(node.id.clone()),
        port,
        param,
        link: None,
        message,
        fix: fix.to_string(),
    }
}

/// Handoff: card 'raa-graph compute 5/7: compute pass, pipeline, bind groups and dispatch' calls
/// `generate_module` to build one compute pass's module, and card 'raa-graph compute 2/7:
/// declaration and WGSL validation' calls it (and `infer_param`) in its validation hook. Today
/// it takes the NodeDoc's document-order params and its ComputeDecl, writes the RgParams struct
/// (or its `rg_unused` member), the prelude, one binding per declared input and output, the user
/// WGSL with a final LF and the `rg_entry` entry point, and returns that text with the line
/// offset of the user text.
#[allow(clippy::result_large_err)] // the S59 signature fixes the unboxed `Issue` error
pub fn generate_module(node: &NodeDoc) -> Result<(String, u32), Issue> {
    let Some(compute) = &node.compute else {
        return Err(issue(
            node,
            "graph.wgsl",
            None,
            None,
            "the node has no compute declaration".to_string(),
            "add inputs, outputs, math and wgsl to the compute node",
        ));
    };
    let mut text = String::from("struct RgParams {\n");
    if node.params.is_empty() {
        text.push_str("  rg_unused: vec4<f32>,\n");
    } else {
        for (key, value) in &node.params {
            let ty = infer_param(value).as_ref().and_then(param_type);
            let Some(ty) = ty else {
                return Err(issue(
                    node,
                    "graph.param_kind",
                    None,
                    Some(key.clone()),
                    format!(
                        "parameter `{key}` is not a number, a bool or an array of 2..4 numbers"
                    ),
                    "write a number, a bool or an array of 2, 3 or 4 numbers",
                ));
            };
            text.push_str(&format!("  {key}: {ty},\n"));
        }
    }
    text.push_str("}\n");
    text.push_str(PRELUDE);
    for (index, input) in compute.inputs.iter().enumerate() {
        let key = &input.key;
        let line = match input.ty {
            PortType::Image2d => {
                format!("@group(1) @binding({index}) var rg_in_{key}: texture_2d<f32>;\n")
            }
            PortType::Depth2d => {
                format!("@group(1) @binding({index}) var rg_in_{key}: texture_depth_2d;\n")
            }
            PortType::Buffer => {
                format!("@group(1) @binding({index}) var<storage, read> rg_in_{key}: array<u32>;\n")
            }
            _ => {
                return Err(issue(
                    node,
                    "graph.link_type",
                    Some(key.clone()),
                    None,
                    format!("input `{key}` is not image2d, depth2d or buffer"),
                    "declare the input as image2d, depth2d or buffer",
                ));
            }
        };
        text.push_str(&line);
    }
    for (index, output) in compute.outputs.iter().enumerate() {
        let key = &output.key;
        let name = if output.ty == PortType::Image2d {
            output.format.as_ref().and_then(storage_format_name)
        } else {
            None
        };
        let Some(name) = name else {
            return Err(issue(
                node,
                "graph.link_type",
                Some(key.clone()),
                None,
                format!("output `{key}` is not an image2d with a storage format"),
                "declare the output as image2d with one of the storage formats",
            ));
        };
        text.push_str(&format!(
            "@group(2) @binding({index}) var rg_out_{key}: texture_storage_2d<{name}, write>;\n"
        ));
    }
    let offset = text.bytes().filter(|byte| *byte == b'\n').count() as u32;
    text.push_str(&compute.wgsl);
    if !compute.wgsl.ends_with('\n') {
        text.push('\n');
    }
    text.push_str(ENTRY);
    Ok((text, offset))
}
