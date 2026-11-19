//! Ambient-occlusion stage nodes and shared post-stage spec builders.
use crate::catalog::{
    Category, GraphFormat, NodeFamily, NodeTypeSpec, ParamBinding, ParamKind, ParamSpec,
    ParamValue, PortSpec, PortType, Rebuild, ShaderStage, SizeSpec, StageBudget,
};
use raa_pipeline::{TunableSpec, find_tunable};

pub fn family() -> Box<dyn NodeFamily> {
    Box::new(AoStages)
}

struct AoStages;

impl NodeFamily for AoStages {
    fn name(&self) -> &'static str {
        "ao_stages"
    }

    fn node_types(&self) -> Vec<NodeTypeSpec> {
        let ao = raa_gtao::AoDesc::default();
        vec![
            post_stage(
                "ao.trace",
                "ao_stages",
                Category::Lighting,
                "AO trace",
                40,
                vec![gbuffer_in("depth"), gbuffer_in("normal")],
                vec![image_out(
                    "raw",
                    "Raw AO",
                    GraphFormat::R32Float,
                    SizeSpec::Scale(1.0),
                    1,
                )],
                vec![
                    node_param(
                        "radius",
                        "Radius",
                        "AO",
                        ParamKind::Number,
                        ParamValue::Number(f64::from(ao.radius_m)),
                        Some(0.0),
                        Some(10.0),
                        None,
                        "",
                        "raa-gtao/src/desc.rs:25",
                    ),
                    node_param(
                        "intensity",
                        "Intensity",
                        "AO",
                        ParamKind::Number,
                        ParamValue::Number(f64::from(ao.intensity)),
                        Some(0.0),
                        Some(1.0),
                        None,
                        "",
                        "raa-gtao/src/desc.rs:26",
                    ),
                    node_param(
                        "power",
                        "Power",
                        "AO",
                        ParamKind::Number,
                        ParamValue::Number(f64::from(ao.power)),
                        Some(0.0),
                        Some(8.0),
                        None,
                        "",
                        "raa-gtao/src/desc.rs:27",
                    ),
                    tunable_param(raa_gtao::tunables(), "slices", "Slices", "AO trace"),
                    tunable_param(raa_gtao::tunables(), "steps", "Steps", "AO trace"),
                ],
                vec![stage_budget(ShaderStage::Compute, 2, 0, 1, 0, 1)],
            ),
            post_stage(
                "ao.blur",
                "ao_stages",
                Category::Lighting,
                "AO blur",
                40,
                vec![
                    required_in("src", "Source", GraphFormat::R32Float),
                    gbuffer_in("depth"),
                ],
                vec![
                    image_out(
                        "ao",
                        "AO",
                        GraphFormat::Rgba16Float,
                        SizeSpec::Scale(1.0),
                        1,
                    ),
                    image_out(
                        "blur",
                        "Blurred AO",
                        GraphFormat::R32Float,
                        SizeSpec::Scale(1.0),
                        1,
                    ),
                ],
                vec![],
                vec![stage_budget(ShaderStage::Compute, 2, 0, 2, 0, 1)],
            ),
        ]
    }
}

/// Handoff: card 'ssr_stages node family' and card 'bloom_stages node family' call this
/// builder for their catalog entries; it supplies the common stage-node flags and no-op upgrade.
#[allow(clippy::too_many_arguments)]
pub(crate) fn post_stage(
    id: &'static str,
    family: &'static str,
    category: Category,
    label: &'static str,
    default_order: u32,
    inputs: Vec<PortSpec>,
    outputs: Vec<PortSpec>,
    params: Vec<ParamSpec>,
    budgets: Vec<StageBudget>,
) -> NodeTypeSpec {
    NodeTypeSpec {
        id,
        version: 1,
        category,
        label,
        description: label,
        pass: None,
        contributor: false,
        default_order,
        multi_instance: true,
        side_effect: false,
        required_in_graph: false,
        inputs,
        outputs,
        params,
        budgets,
        upgrade,
        family,
    }
}

fn upgrade(_: u32, _: &mut Vec<(String, ParamValue)>) -> Result<(), String> {
    Ok(())
}

pub(crate) fn gbuffer_in(key: &'static str) -> PortSpec {
    let (label, ty, format, sources) = match key {
        "depth" => (
            "Depth",
            PortType::Depth2d,
            GraphFormat::Depth32Float,
            &["pass.prepass.depth", "pass.vg.gbuffer.depth"][..],
        ),
        "normal" => (
            "Normal",
            PortType::Image2d,
            GraphFormat::Rgba16Float,
            &["pass.prepass.normal", "pass.vg.gbuffer.normal"][..],
        ),
        "material" => (
            "Material",
            PortType::Image2d,
            GraphFormat::Rgba8Unorm,
            &["pass.prepass.material", "pass.vg.gbuffer.material"][..],
        ),
        _ => panic!("ao_stages: unknown G-buffer input `{key}`"),
    };
    input(key, label, ty, format, false, sources)
}

pub(crate) fn required_in(key: &'static str, label: &'static str, format: GraphFormat) -> PortSpec {
    input(key, label, PortType::Image2d, format, true, &[])
}

#[allow(dead_code)] // consumed by the discovered bloom_stages family when it lands
pub(crate) fn optional_in(key: &'static str, label: &'static str, format: GraphFormat) -> PortSpec {
    let mut port = input(key, label, PortType::Image2d, format, false, &[]);
    port.absent_ok = true;
    port
}

fn input(
    key: &'static str,
    label: &'static str,
    ty: PortType,
    format: GraphFormat,
    required: bool,
    default_source: &'static [&'static str],
) -> PortSpec {
    PortSpec {
        key,
        label,
        ty,
        format: Some(format),
        size: SizeSpec::Scale(1.0),
        mips: 1,
        layers: 1,
        required,
        default_source,
        absent_ok: false,
        generic: true,
        history: false,
        slot: None,
    }
}

pub(crate) fn image_out(
    key: &'static str,
    label: &'static str,
    format: GraphFormat,
    size: SizeSpec,
    mips: u32,
) -> PortSpec {
    PortSpec {
        key,
        label,
        ty: PortType::Image2d,
        format: Some(format),
        size,
        mips,
        layers: 1,
        required: false,
        default_source: &[],
        absent_ok: false,
        generic: false,
        history: false,
        slot: None,
    }
}

pub(crate) fn stage_budget(
    stage: ShaderStage,
    sampled: u32,
    samplers: u32,
    storage_textures: u32,
    storage_buffers: u32,
    uniform: u32,
) -> StageBudget {
    StageBudget {
        stage,
        sampled_textures: sampled,
        samplers,
        storage_textures,
        storage_buffers,
        uniform_buffers: uniform,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn node_param(
    key: &'static str,
    label: &'static str,
    group: &'static str,
    kind: ParamKind,
    default: ParamValue,
    min: Option<f64>,
    max: Option<f64>,
    step: Option<f64>,
    unit: &'static str,
    source: &'static str,
) -> ParamSpec {
    ParamSpec {
        key,
        label,
        group,
        kind,
        default,
        min,
        max,
        step,
        options: &[],
        unit,
        rebuild: Rebuild::None,
        primary: false,
        unstated_range: false,
        binding: ParamBinding::Node,
        source,
    }
}

pub(crate) fn tunable_param(
    specs: &'static [TunableSpec],
    key: &'static str,
    label: &'static str,
    group: &'static str,
) -> ParamSpec {
    let spec =
        find_tunable(specs, key).unwrap_or_else(|| panic!("ao_stages: missing tunable `{key}`"));
    let (kind, default) = if spec.integer {
        (ParamKind::Int, ParamValue::Int(spec.default as i64))
    } else {
        (ParamKind::Number, ParamValue::Number(spec.default))
    };
    ParamSpec {
        key,
        label,
        group,
        kind,
        default,
        min: spec.min,
        max: spec.max,
        step: None,
        options: &[],
        unit: "",
        rebuild: if spec.rebuild == "pipeline" {
            Rebuild::Pipeline
        } else {
            Rebuild::None
        },
        primary: false,
        unstated_range: spec.min.is_none() && spec.max.is_none(),
        binding: ParamBinding::Node,
        source: spec.source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ao_stage_catalog_contract() {
        let catalog = crate::Catalog::build();
        for id in ["ao.trace", "ao.blur"] {
            let node = catalog.node(id).expect("AO stage node");
            assert_eq!(node.family, "ao_stages");
            assert_eq!(node.pass, None);
            assert_eq!(node.category, Category::Lighting);
            assert_eq!(node.default_order, 40);
            assert!(node.multi_instance);
        }
        let trace = catalog.node("ao.trace").unwrap();
        assert_eq!(
            trace
                .params
                .iter()
                .map(|param| param.key)
                .collect::<Vec<_>>(),
            ["radius", "intensity", "power", "slices", "steps"]
        );
        let pass = catalog.node("pass.ao.gtao").unwrap();
        for key in ["radius", "intensity", "power"] {
            let mut actual = trace
                .params
                .iter()
                .find(|param| param.key == key)
                .unwrap()
                .clone();
            let mut expected = pass
                .params
                .iter()
                .find(|param| param.key == key)
                .unwrap()
                .clone();
            actual.binding = ParamBinding::Node;
            expected.binding = ParamBinding::Node;
            assert_eq!(actual, expected);
        }
        let blur = catalog.node("ao.blur").unwrap();
        assert_eq!(
            blur.inputs
                .iter()
                .map(|port| (port.key, port.required, port.format.clone()))
                .collect::<Vec<_>>(),
            [
                ("src", true, Some(GraphFormat::R32Float)),
                ("depth", false, Some(GraphFormat::Depth32Float)),
            ]
        );
        assert_eq!(
            blur.outputs
                .iter()
                .map(|port| (port.key, port.format.clone()))
                .collect::<Vec<_>>(),
            [
                ("ao", Some(GraphFormat::Rgba16Float)),
                ("blur", Some(GraphFormat::R32Float)),
            ]
        );
        assert!(
            catalog
                .nodes
                .iter()
                .filter(|node| node.family == "ao_stages")
                .flat_map(|node| &node.inputs)
                .all(|port| port.generic)
        );
    }
}
