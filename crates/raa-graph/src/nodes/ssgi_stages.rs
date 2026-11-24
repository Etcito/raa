//! Screen-space GI stage node types.
use crate::catalog::{
    Category, GraphFormat as F, NodeFamily, NodeTypeSpec, ParamBinding, ParamKind, ParamSpec,
    ParamValue, PortSpec, PortType as T, Rebuild, ShaderStage, SizeSpec, StageBudget,
};
use raa_pipeline::{TunableSpec, find_tunable};

pub fn family() -> Box<dyn NodeFamily> {
    Box::new(SsgiStages)
}
struct SsgiStages;
impl NodeFamily for SsgiStages {
    fn name(&self) -> &'static str {
        "ssgi_stages"
    }
    fn node_types(&self) -> Vec<NodeTypeSpec> {
        let depth = gbuffer("depth");
        let normal = gbuffer("normal");
        let motion = gbuffer("motion");
        vec![
            node(
                "ssgi.trace",
                "SSGI trace",
                vec![depth.clone(), normal.clone()],
                vec![output("raw")],
                &[
                    "quality",
                    "slicesLow",
                    "slicesMedium",
                    "slicesHigh",
                    "stepsLow",
                    "stepsMedium",
                    "stepsHigh",
                ],
                5,
                0,
                2,
            ),
            node(
                "ssgi.resolve",
                "SSGI resolve",
                vec![
                    depth,
                    normal,
                    motion,
                    required(input("raw", T::Image2d, Some(F::Rgba16Float), true, false)),
                    input("probe", T::Image2d, Some(F::Rgba16Float), false, false),
                    input("history", T::Image2d, Some(F::Rgba16Float), false, true),
                ],
                vec![output("ssgi"), output("accumulated")],
                &["changeSigma"],
                6,
                0,
                2,
            ),
        ]
    }
}
#[allow(clippy::too_many_arguments)]
fn node(
    id: &'static str,
    label: &'static str,
    inputs: Vec<PortSpec>,
    outputs: Vec<PortSpec>,
    keys: &[&'static str],
    sampled: u32,
    samplers: u32,
    storage_textures: u32,
) -> NodeTypeSpec {
    let params = keys.iter().map(|key| param(key)).collect();
    NodeTypeSpec {
        id,
        version: 1,
        category: Category::Gi,
        label,
        description: label,
        pass: None,
        contributor: false,
        default_order: 44,
        multi_instance: true,
        side_effect: false,
        required_in_graph: false,
        inputs,
        outputs,
        params,
        budgets: vec![StageBudget {
            stage: ShaderStage::Compute,
            sampled_textures: sampled,
            samplers,
            storage_textures,
            storage_buffers: 0,
            uniform_buffers: 2,
        }],
        upgrade,
        family: "ssgi_stages",
    }
}
fn upgrade(_: u32, _: &mut Vec<(String, ParamValue)>) -> Result<(), String> {
    Ok(())
}
fn param(key: &'static str) -> ParamSpec {
    if key == "quality" {
        return ParamSpec {
            key: "quality",
            label: "Quality",
            group: "SSGI trace",
            kind: ParamKind::Enum,
            default: ParamValue::Enum(
                raa_gi::GiSettings::DEFAULT
                    .screen_space_quality
                    .as_str()
                    .to_string(),
            ),
            min: None,
            max: None,
            step: None,
            options: &["low", "medium", "high"],
            unit: "",
            rebuild: Rebuild::None,
            primary: false,
            unstated_range: false,
            binding: ParamBinding::Node,
            source: "raa-gi/src/settings.rs:353",
        };
    }
    let spec = find_tunable(raa_gi::tunables(), key)
        .unwrap_or_else(|| panic!("missing SSGI tunable `{key}`"));
    let (group, label) = match key {
        "slicesLow" => ("SSGI trace", "Slices (low)"),
        "slicesMedium" => ("SSGI trace", "Slices (medium)"),
        "slicesHigh" => ("SSGI trace", "Slices (high)"),
        "stepsLow" => ("SSGI trace", "Steps (low)"),
        "stepsMedium" => ("SSGI trace", "Steps (medium)"),
        "stepsHigh" => ("SSGI trace", "Steps (high)"),
        "changeSigma" => ("SSGI resolve", "Change sigma"),
        _ => panic!("unknown SSGI key `{key}`"),
    };
    tunable(key, spec, group, label)
}
fn tunable(
    key: &'static str,
    spec: &TunableSpec,
    group: &'static str,
    label: &'static str,
) -> ParamSpec {
    let (kind, default, step) = if spec.integer {
        (
            ParamKind::Int,
            ParamValue::Int(spec.default as i64),
            Some(1.0),
        )
    } else {
        (ParamKind::Number, ParamValue::Number(spec.default), None)
    };
    ParamSpec {
        key,
        label,
        group,
        kind,
        default,
        min: spec.min,
        max: spec.max,
        step,
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
fn port(key: &'static str, ty: T, format: Option<F>, required: bool, history: bool) -> PortSpec {
    PortSpec {
        key,
        label: key,
        ty: ty.clone(),
        format,
        size: SizeSpec::Scale(1.0),
        mips: 1,
        layers: 1,
        required,
        default_source: &[],
        absent_ok: !required,
        generic: true,
        history,
        slot: None,
    }
}
fn input(key: &'static str, ty: T, f: Option<F>, required: bool, history: bool) -> PortSpec {
    port(key, ty, f, required, history)
}
fn output(key: &'static str) -> PortSpec {
    PortSpec {
        absent_ok: false,
        ..port(key, T::Image2d, Some(F::Rgba16Float), false, false)
    }
}
fn gbuffer(key: &'static str) -> PortSpec {
    let (label, ty, format, sources) = match key {
        "depth" => (
            "Depth",
            T::Depth2d,
            F::Depth32Float,
            &["pass.prepass.depth", "pass.vg.gbuffer.depth"][..],
        ),
        "normal" => (
            "Normal",
            T::Image2d,
            F::Rgba16Float,
            &["pass.prepass.normal", "pass.vg.gbuffer.normal"][..],
        ),
        _ => (
            "Motion",
            T::Image2d,
            F::Rg16Float,
            &["pass.prepass.motion", "pass.vg.gbuffer.motion"][..],
        ),
    };
    PortSpec {
        label,
        ty: ty.clone(),
        format: Some(format.clone()),
        default_source: sources,
        absent_ok: false,
        ..port(key, ty, Some(format), false, false)
    }
}
fn required(mut p: PortSpec) -> PortSpec {
    p.required = true;
    p.absent_ok = false;
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Catalog;
    #[test]
    fn ssgi_stage_catalog_contract() {
        let c = Catalog::build();
        for id in ["ssgi.trace", "ssgi.resolve"] {
            let n = c.node(id).unwrap();
            assert_eq!(n.family, "ssgi_stages");
            assert_eq!(n.pass, None);
            assert!(n.multi_instance);
        }
        let history: Vec<_> = c
            .nodes
            .iter()
            .filter(|n| n.family == "ssgi_stages")
            .flat_map(|n| {
                n.inputs
                    .iter()
                    .filter(|p| p.history)
                    .map(move |p| (n.id, p.key))
            })
            .collect();
        assert_eq!(history, [("ssgi.resolve", "history")]);
        let quality = c
            .node("ssgi.trace")
            .unwrap()
            .params
            .iter()
            .find(|p| p.key == "quality")
            .unwrap();
        assert_eq!(quality.options, ["low", "medium", "high"]);
        assert_eq!(
            quality.default,
            ParamValue::Enum(
                raa_gi::GiSettings::DEFAULT
                    .screen_space_quality
                    .as_str()
                    .to_string()
            )
        );
    }
}
