//! The compute node (S59): the module text `generate_module` writes, its parameter inference,
//! the storage formats and the catalog entry of `compute.wgsl`. The expected module is written
//! literally and the generated text is validated with `raa_shader::Composer::validate`.

#[allow(dead_code)]
#[path = "../src/nodes/compute/codegen.rs"]
mod codegen;

use codegen::{STORAGE_FORMATS, generate_module, infer_param, storage_format_name};
use raa_graph::Catalog;
use raa_graph::catalog::{Category, GraphFormat, ParamValue, PortType};
use raa_graph::error::{Issue, Severity, Stage};
use raa_graph::format::{ComputeDecl, ComputePortDecl, NodeDoc};
use raa_loader::json::JsonValue;

/// A compute node with the given params, declaration inputs, outputs and user WGSL.
fn node(
    params: Vec<(&str, JsonValue)>,
    inputs: Vec<ComputePortDecl>,
    outputs: Vec<ComputePortDecl>,
    wgsl: &str,
) -> NodeDoc {
    NodeDoc {
        id: "blur".to_string(),
        type_id: "compute.wgsl".to_string(),
        version: None,
        params: params
            .into_iter()
            .map(|(key, value)| (key.to_string(), value))
            .collect(),
        compute: Some(ComputeDecl {
            inputs,
            outputs,
            math: JsonValue::Null,
            wgsl: wgsl.to_string(),
        }),
    }
}

fn image_input(key: &str) -> ComputePortDecl {
    ComputePortDecl {
        key: key.to_string(),
        ty: PortType::Image2d,
        format: None,
        size: None,
        extent: None,
    }
}

fn depth_input(key: &str) -> ComputePortDecl {
    ComputePortDecl {
        key: key.to_string(),
        ty: PortType::Depth2d,
        format: None,
        size: None,
        extent: None,
    }
}

fn image_output(key: &str, format: GraphFormat) -> ComputePortDecl {
    ComputePortDecl {
        key: key.to_string(),
        ty: PortType::Image2d,
        format: Some(format),
        size: None,
        extent: None,
    }
}

fn vec_value(items: &[f64]) -> JsonValue {
    JsonValue::Array(items.iter().copied().map(JsonValue::Number).collect())
}

#[test]
fn generate_module_writes_the_full_text_and_its_offset() {
    let node = node(
        vec![
            ("gain", JsonValue::Number(0.5)),
            ("tint", vec_value(&[1.0, 0.5, 0.25])),
        ],
        vec![image_input("src"), depth_input("depth")],
        vec![image_output("dst", GraphFormat::Rgba16Float)],
        "fn rg_main(p: vec2<u32>) {\n  textureStore(rg_out_dst, p, textureLoad(rg_in_src, vec2<i32>(p), 0) * rg_params.gain);\n}\n",
    );
    let (text, offset) = generate_module(&node).expect("the module generates");
    let expected = concat!(
        "struct RgParams {\n",
        "  gain: f32,\n",
        "  tint: vec3<f32>,\n",
        "}\n",
        "struct RgFrame {\n",
        "  size: vec2<u32>,\n",
        "  inv_size: vec2<f32>,\n",
        "  frame_index: u32,\n",
        "  time: f32,\n",
        "  view: mat4x4<f32>,\n",
        "  proj: mat4x4<f32>,\n",
        "  inv_proj: mat4x4<f32>,\n",
        "  inv_view: mat4x4<f32>,\n",
        "  previous_view_proj: mat4x4<f32>,\n",
        "  history_valid: u32,\n",
        "  interlace: u32,\n",
        "}\n",
        "@group(0) @binding(0) var<uniform> rg_params: RgParams;\n",
        "@group(0) @binding(1) var<uniform> rg_frame: RgFrame;\n",
        "@group(0) @binding(2) var rg_linear: sampler;\n",
        "@group(1) @binding(0) var rg_in_src: texture_2d<f32>;\n",
        "@group(1) @binding(1) var rg_in_depth: texture_depth_2d;\n",
        "@group(2) @binding(0) var rg_out_dst: texture_storage_2d<rgba16float, write>;\n",
        "fn rg_main(p: vec2<u32>) {\n",
        "  textureStore(rg_out_dst, p, textureLoad(rg_in_src, vec2<i32>(p), 0) * rg_params.gain);\n",
        "}\n",
        "@compute @workgroup_size(8, 8)\n",
        "fn rg_entry(@builtin(global_invocation_id) id: vec3<u32>) {\n",
        "  if (any(id.xy >= rg_frame.size)) {\n",
        "    return;\n",
        "  }\n",
        "  rg_main(id.xy);\n",
        "}\n",
    );
    assert_eq!(text.as_str(), expected);
    assert_eq!(offset, 22);
    raa_shader::Composer::validate(&text).expect("the generated module validates");
}

#[test]
fn no_params_writes_the_placeholder_member_and_offset() {
    let node = node(
        Vec::new(),
        vec![image_input("src")],
        vec![image_output("dst", GraphFormat::Rgba16Float)],
        "fn rg_main(p: vec2<u32>) {\n}\n",
    );
    let (text, offset) = generate_module(&node).expect("the module generates");
    assert!(text.contains("struct RgParams {\n  rg_unused: vec4<f32>,\n}\n"));
    assert_eq!(offset, 20);
}

#[test]
fn infer_param_maps_each_kind_and_generate_refuses_bad_values() {
    assert_eq!(
        infer_param(&JsonValue::Number(0.5)),
        Some(ParamValue::Number(0.5))
    );
    assert_eq!(
        infer_param(&JsonValue::Bool(true)),
        Some(ParamValue::Bool(true))
    );
    assert_eq!(
        infer_param(&vec_value(&[1.0, 2.0])),
        Some(ParamValue::Vec2([1.0, 2.0]))
    );
    assert_eq!(
        infer_param(&vec_value(&[1.0, 2.0, 3.0])),
        Some(ParamValue::Vec3([1.0, 2.0, 3.0]))
    );
    assert_eq!(
        infer_param(&vec_value(&[1.0, 2.0, 3.0, 4.0])),
        Some(ParamValue::Vec4([1.0, 2.0, 3.0, 4.0]))
    );
    assert_eq!(infer_param(&vec_value(&[1.0])), None);
    assert_eq!(infer_param(&vec_value(&[1.0, 2.0, 3.0, 4.0, 5.0])), None);
    assert_eq!(infer_param(&JsonValue::String("x".to_string())), None);
    assert_eq!(infer_param(&JsonValue::Null), None);

    assert_eq!(STORAGE_FORMATS.len(), 8);
    assert_eq!(
        storage_format_name(&GraphFormat::Rgba16Float),
        Some("rgba16float")
    );
    assert_eq!(storage_format_name(&GraphFormat::R8Unorm), None);

    let string_param = node(
        vec![("label", JsonValue::String("x".to_string()))],
        Vec::new(),
        Vec::new(),
        "fn rg_main(p: vec2<u32>) {\n}\n",
    );
    let err = generate_module(&string_param).expect_err("a string parameter is refused");
    assert_eq!(err.code, "graph.param_kind");
    assert_eq!(err.severity, Severity::Error);
    assert_eq!(err.stage, Stage::Structure);
    assert_eq!(err.node.as_deref(), Some("blur"));
    assert_eq!(err.param.as_deref(), Some("label"));

    let bad_output = node(
        Vec::new(),
        Vec::new(),
        vec![image_output("dst", GraphFormat::R8Unorm)],
        "fn rg_main(p: vec2<u32>) {\n}\n",
    );
    let err = generate_module(&bad_output).expect_err("a non-storage format is refused");
    assert_eq!(err.code, "graph.link_type");
    assert_eq!(err.port.as_deref(), Some("dst"));
}

#[test]
fn catalog_declares_the_compute_node() {
    let catalog = Catalog::build();
    let spec = catalog
        .node("compute.wgsl")
        .expect("compute.wgsl is in the catalog");
    assert_eq!(spec.pass, None);
    assert!(spec.multi_instance);
    assert_eq!(spec.default_order, 46);
    assert_eq!(spec.category, Category::Compute);
}
