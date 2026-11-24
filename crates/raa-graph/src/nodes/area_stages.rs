//! Area-shadow stage node types.
use crate::catalog::{
    Category, FeatureId, GraphFormat as F, NodeFamily, NodeTypeSpec, ParamBinding, ParamKind,
    ParamSpec, ParamValue, PortSpec, PortType as T, Rebuild, ShaderStage, SizeSpec, StageBudget,
};
use raa_pipeline::{TunableSpec, find_tunable};

pub fn family() -> Box<dyn NodeFamily> {
    Box::new(AreaStages)
}
struct AreaStages;

impl NodeFamily for AreaStages {
    fn name(&self) -> &'static str {
        "area_stages"
    }
    fn node_types(&self) -> Vec<NodeTypeSpec> {
        let (depth, normal, motion) = (gbuffer("depth"), gbuffer("normal"), gbuffer("motion"));
        let array = || output("raw", T::Image2dArray, F::Rgba16Float, 2);
        vec![
            node(
                "area.trace",
                "Area shadow trace",
                vec![
                    depth.clone(),
                    normal.clone(),
                    port(
                        "pyramid",
                        T::Image2d,
                        Some(F::R32Float),
                        false,
                        false,
                        false,
                        true,
                    ),
                ],
                vec![array(), output("linearDepth", T::Image2d, F::R32Float, 1)],
                &[],
                5,
                1,
                2,
                0,
                4,
            ),
            node(
                "area.temporal",
                "Area shadow temporal",
                vec![
                    depth.clone(),
                    normal.clone(),
                    motion.clone(),
                    input(
                        "linearDepth",
                        T::Image2d,
                        Some(F::R32Float),
                        true,
                        false,
                        false,
                        true,
                    ),
                    required(array_port("raw", true)),
                    input(
                        "history",
                        T::Image2dArray,
                        Some(F::Rgba16Float),
                        false,
                        true,
                        false,
                        true,
                    ),
                    input(
                        "depthHistory",
                        T::Image2d,
                        Some(F::R32Float),
                        false,
                        true,
                        false,
                        true,
                    ),
                ],
                vec![
                    array_port("filtered", false),
                    output("depth", T::Image2d, F::R32Float, 1),
                ],
                &["temporalAlpha", "depthTolerance", "antiLag"],
                5,
                1,
                2,
                0,
                4,
            ),
            node(
                "area.spatial",
                "Area shadow spatial",
                vec![
                    depth.clone(),
                    normal.clone(),
                    required(input(
                        "linearDepth",
                        T::Image2d,
                        Some(F::R32Float),
                        true,
                        false,
                        false,
                        true,
                    )),
                    required(array_port("src", true)),
                ],
                vec![array_port("mask", false)],
                &["spatialStride", "normalPower"],
                5,
                1,
                2,
                0,
                4,
            ),
            node(
                "area.restir.initial",
                "Area ReSTIR initial",
                vec![depth.clone(), normal.clone()],
                vec![buffer("reservoirs")],
                &["restirCandidates", "restirDepthTolerance"],
                5,
                1,
                2,
                0,
                4,
            ),
            node(
                "area.restir.temporal",
                "Area ReSTIR temporal",
                vec![
                    depth.clone(),
                    normal.clone(),
                    motion.clone(),
                    required(buffer("initial")),
                ],
                vec![buffer("reservoirs")],
                &["restirMCap", "restirBoilingFactor", "restirNormalCos"],
                5,
                1,
                2,
                0,
                4,
            ),
            node(
                "area.restir.spatial",
                "Area ReSTIR spatial",
                vec![depth, normal, required(buffer("reservoirs"))],
                vec![array_port("raw", false), buffer("reservoirs")],
                &[
                    "restirSpatialSamples",
                    "restirSpatialRadius",
                    "restirNormalCos",
                ],
                5,
                1,
                2,
                0,
                4,
            ),
            node(
                "area.mask",
                "Area shadow mask",
                vec![required(array_port("src", true))],
                vec![output("areaShadow", T::Image2dArray, F::Rgba16Float, 2)],
                &[],
                5,
                1,
                2,
                0,
                4,
            ),
        ]
    }

    fn extend_types(&self, types: &mut Vec<NodeTypeSpec>) {
        let params = types
            .iter()
            .find(|n| n.id == "pass.gi.area-shadow")
            .expect("area shadow pass node")
            .params
            .clone();
        let mask = types
            .iter_mut()
            .find(|n| n.id == "area.mask")
            .expect("area.mask node");
        mask.params.extend(
            params
                .iter()
                .filter(|p| {
                    matches!(
                        p.binding,
                        ParamBinding::Setting {
                            feature: FeatureId::AreaShadows,
                            ..
                        }
                    )
                })
                .cloned(),
        );
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
    storage_buffers: u32,
    uniform_buffers: u32,
) -> NodeTypeSpec {
    let params = keys
        .iter()
        .map(|key| stage_param("pass.gi.area-shadow", key))
        .collect();
    NodeTypeSpec {
        id,
        version: 1,
        category: Category::Shadows,
        label,
        description: label,
        pass: None,
        contributor: false,
        default_order: 43,
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
            storage_buffers,
            uniform_buffers,
        }],
        upgrade,
        family: "area_stages",
    }
}
fn upgrade(_: u32, _: &mut Vec<(String, ParamValue)>) -> Result<(), String> {
    Ok(())
}
fn stage_param(pass: &str, key: &'static str) -> ParamSpec {
    let binding_key = match (pass, key) {
        ("pass.gi.area-shadow", "restirCandidates") => "areaRestirCandidates",
        ("pass.gi.area-shadow", "restirSpatialSamples") => "areaRestirSpatialSamples",
        ("pass.gi.area-shadow", "restirSpatialRadius") => "areaRestirSpatialRadius",
        ("pass.gi.area-shadow", "restirMCap") => "areaRestirMCap",
        ("pass.gi.area-shadow", "restirBoilingFactor") => "areaRestirBoilingFactor",
        ("pass.gi.area-shadow", "restirNormalCos") => "areaRestirNormalCos",
        _ => key,
    };
    let anti_lag = &raa_gi::parts::area_shadow::passes::AREA_ANTI_LAG;
    let spec = if key == anti_lag.key {
        anti_lag
    } else {
        find_tunable(raa_gi::tunables(), binding_key).expect("stage tunable spec")
    };
    let (group, label) = match key {
        "temporalAlpha" => ("Area shadow temporal", "Temporal alpha"),
        "antiLag" => ("Area shadow temporal", "Anti-lag"),
        "depthTolerance" => ("Area shadow temporal", "Depth tolerance"),
        "spatialStride" => ("Area shadow spatial", "Spatial stride"),
        "normalPower" => ("Area shadow spatial", "Normal power"),
        "restirDepthTolerance" => ("Area ReSTIR initial", "Restir depth tolerance"),
        "restirCandidates" => ("Area ReSTIR initial", "Candidates"),
        "restirMCap" => ("Area ReSTIR temporal", "M cap"),
        "restirBoilingFactor" => ("Area ReSTIR temporal", "Boiling factor"),
        "restirNormalCos" => ("Area ReSTIR temporal", "Normal cosine"),
        "restirSpatialSamples" => ("Area ReSTIR spatial", "Spatial samples"),
        "restirSpatialRadius" => ("Area ReSTIR spatial", "Spatial radius"),
        _ => panic!("no area stage metadata for {key}"),
    };
    tunable_param(key, spec, group, label)
}
fn tunable_param(
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
fn port(
    key: &'static str,
    ty: T,
    format: Option<F>,
    required: bool,
    history: bool,
    absent_ok: bool,
    generic: bool,
) -> PortSpec {
    PortSpec {
        key,
        label: key,
        ty,
        format,
        size: SizeSpec::Scale(1.0),
        mips: 1,
        layers: 1,
        required,
        default_source: &[],
        absent_ok,
        generic,
        history,
        slot: None,
    }
}
fn input(
    key: &'static str,
    ty: T,
    f: Option<F>,
    required: bool,
    history: bool,
    absent_ok: bool,
    generic: bool,
) -> PortSpec {
    port(key, ty, f, required, history, absent_ok, generic)
}
fn output(key: &'static str, ty: T, f: F, layers: u32) -> PortSpec {
    PortSpec {
        layers,
        ..port(key, ty, Some(f), false, false, false, false)
    }
}
fn gbuffer(key: &'static str) -> PortSpec {
    let (label, f, sources) = match key {
        "depth" => (
            "Depth",
            F::Depth32Float,
            &["pass.prepass.depth", "pass.vg.gbuffer.depth"][..],
        ),
        "normal" => (
            "Normal",
            F::Rgba16Float,
            &["pass.prepass.normal", "pass.vg.gbuffer.normal"][..],
        ),
        _ => (
            "Motion",
            F::Rg16Float,
            &["pass.prepass.motion", "pass.vg.gbuffer.motion"][..],
        ),
    };
    PortSpec {
        label,
        ty: if key == "depth" {
            T::Depth2d
        } else {
            T::Image2d
        },
        format: Some(f.clone()),
        default_source: sources,
        absent_ok: false,
        generic: true,
        ..port(
            key,
            if key == "depth" {
                T::Depth2d
            } else {
                T::Image2d
            },
            Some(f),
            false,
            false,
            false,
            true,
        )
    }
}
fn required(mut p: PortSpec) -> PortSpec {
    p.required = true;
    p.absent_ok = false;
    p
}
fn array_port(key: &'static str, req: bool) -> PortSpec {
    let mut p = port(
        key,
        T::Image2dArray,
        Some(F::Rgba16Float),
        req,
        false,
        !req,
        true,
    );
    p.layers = 2;
    p
}
fn buffer(key: &'static str) -> PortSpec {
    port(key, T::Buffer, None, false, false, true, true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Catalog;
    #[test]
    fn area_stage_catalog_contract() {
        let c = Catalog::build();
        for id in [
            "area.trace",
            "area.temporal",
            "area.spatial",
            "area.restir.initial",
            "area.restir.temporal",
            "area.restir.spatial",
            "area.mask",
        ] {
            let n = c.node(id).unwrap();
            assert_eq!(n.family, "area_stages");
            assert_eq!(n.pass, None);
            assert!(n.multi_instance);
        }
        assert!(
            c.nodes
                .iter()
                .filter(|n| n.family == "area_stages")
                .all(|n| { n.inputs.iter().all(|input| input.generic) })
        );
        let linear_depth = c
            .node("area.spatial")
            .unwrap()
            .inputs
            .iter()
            .find(|input| input.key == "linearDepth")
            .unwrap();
        assert!(linear_depth.generic && linear_depth.required && !linear_depth.absent_ok);
        let history: Vec<_> = c
            .nodes
            .iter()
            .filter(|n| n.family == "area_stages")
            .flat_map(|n| {
                n.inputs
                    .iter()
                    .filter(|p| p.history)
                    .map(move |p| (n.id, p.key))
            })
            .collect();
        assert_eq!(
            history,
            [
                ("area.temporal", "history"),
                ("area.temporal", "depthHistory")
            ]
        );
        let mask = c.node("area.mask").unwrap();
        let out = mask.outputs.iter().find(|p| p.key == "areaShadow").unwrap();
        assert_eq!(
            (out.ty.clone(), out.format.clone(), out.layers),
            (T::Image2dArray, Some(F::Rgba16Float), 2)
        );
        let pass = c.node("pass.gi.area-shadow").unwrap();
        let settings: Vec<_> = pass
            .params
            .iter()
            .filter(|p| {
                matches!(
                    p.binding,
                    ParamBinding::Setting {
                        feature: FeatureId::AreaShadows,
                        ..
                    }
                )
            })
            .collect();
        assert!(!settings.is_empty());
        assert_eq!(mask.params.len(), settings.len());
        for (a, b) in mask.params.iter().zip(settings) {
            assert_eq!(a, b);
            assert!(matches!(
                a.binding,
                ParamBinding::Setting {
                    feature: FeatureId::AreaShadows,
                    ..
                }
            ));
        }
    }
}
