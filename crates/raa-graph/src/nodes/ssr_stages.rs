//! Screen-space reflection stage nodes.

use super::ao_stages::{
    gbuffer_in, image_out, node_param, post_stage, required_in, stage_budget, tunable_param,
};
use crate::catalog::{
    Category, GraphFormat, NodeFamily, NodeTypeSpec, ParamKind, ParamValue, ShaderStage, SizeSpec,
};

pub fn family() -> Box<dyn NodeFamily> {
    Box::new(SsrStages)
}

struct SsrStages;

impl NodeFamily for SsrStages {
    fn name(&self) -> &'static str {
        "ssr_stages"
    }

    fn node_types(&self) -> Vec<NodeTypeSpec> {
        let desc = raa_ssr::SsrDesc::default();
        let pass = super::passes::family().node_types();
        let ssr = pass.iter().find(|node| node.id == "pass.ssr").unwrap();
        let param = |key: &str| ssr.params.iter().find(|spec| spec.key == key).unwrap();
        let copied = |key: &'static str, default: f64| {
            let source = param(key);
            node_param(
                key,
                source.label,
                source.group,
                ParamKind::Number,
                ParamValue::Number(default),
                source.min,
                source.max,
                source.step,
                source.unit,
                source.source,
            )
        };
        let fragment = ssr
            .budgets
            .iter()
            .find(|budget| budget.stage == ShaderStage::Fragment)
            .unwrap();
        let fragment_budget = |sampled| {
            stage_budget(
                ShaderStage::Fragment,
                sampled,
                fragment.samplers,
                fragment.storage_textures,
                fragment.storage_buffers,
                fragment.uniform_buffers,
            )
        };
        let hiz_mips = ssr
            .outputs
            .iter()
            .find(|port| port.key == "hiz")
            .unwrap()
            .mips;

        vec![
            post_stage(
                "ssr.hiz",
                "ssr_stages",
                Category::Post,
                "SSR Hi-Z",
                65,
                vec![gbuffer_in("depth")],
                vec![image_out(
                    "hiz",
                    "Hi-Z pyramid",
                    GraphFormat::R32Float,
                    SizeSpec::Scale(1.0),
                    hiz_mips,
                )],
                vec![],
                vec![stage_budget(ShaderStage::Compute, 1, 0, 1, 0, 1)],
            ),
            post_stage(
                "ssr.trace",
                "ssr_stages",
                Category::Post,
                "SSR trace",
                65,
                vec![
                    gbuffer_in("depth"),
                    gbuffer_in("normal"),
                    required_in("hiz", "Hi-Z pyramid", GraphFormat::R32Float),
                ],
                vec![image_out(
                    "trace",
                    "Trace",
                    GraphFormat::Rgba16Float,
                    SizeSpec::Scale(1.0),
                    1,
                )],
                vec![
                    copied("maxRoughness", f64::from(desc.max_roughness)),
                    copied("thickness", f64::from(desc.thickness_m)),
                    tunable_param(raa_ssr::tunables(), "maxSteps", "Max steps", "SSR trace"),
                    tunable_param(
                        raa_ssr::tunables(),
                        "refineSteps",
                        "Refine steps",
                        "SSR trace",
                    ),
                    tunable_param(raa_ssr::tunables(), "edgeFade", "Edge fade", "SSR trace"),
                ],
                vec![fragment_budget(14)],
            ),
            post_stage(
                "ssr.composite",
                "ssr_stages",
                Category::Post,
                "SSR composite",
                65,
                vec![
                    required_in("color", "Color", GraphFormat::Rgba16Float),
                    gbuffer_in("depth"),
                    gbuffer_in("normal"),
                    gbuffer_in("material"),
                    required_in("trace", "Trace", GraphFormat::Rgba16Float),
                ],
                vec![image_out(
                    "color",
                    "Color",
                    GraphFormat::Rgba16Float,
                    SizeSpec::Scale(1.0),
                    1,
                )],
                vec![
                    copied("intensity", f64::from(desc.intensity)),
                    copied("maxRoughness", f64::from(desc.max_roughness)),
                ],
                vec![fragment_budget(16)],
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{Catalog, ParamBinding};
    use crate::validate::GraphLimits;

    #[test]
    fn ssr_stages_catalog_contract() {
        let catalog = Catalog::build();
        let pass = catalog.node("pass.ssr").unwrap();
        let limits = GraphLimits::CORE;
        for id in ["ssr.hiz", "ssr.trace", "ssr.composite"] {
            let node = catalog.node(id).unwrap();
            assert_eq!(node.family, "ssr_stages");
            assert_eq!(node.category, Category::Post);
            assert_eq!(node.default_order, 65);
            assert!(node.multi_instance);
            assert_eq!(node.pass, None);
        }

        let trace = catalog.node("ssr.trace").unwrap();
        assert_eq!(
            trace
                .params
                .iter()
                .map(|param| param.key)
                .collect::<Vec<_>>(),
            [
                "maxRoughness",
                "thickness",
                "maxSteps",
                "refineSteps",
                "edgeFade"
            ]
        );
        let composite = catalog.node("ssr.composite").unwrap();
        assert_eq!(
            composite
                .params
                .iter()
                .map(|param| param.key)
                .collect::<Vec<_>>(),
            ["intensity", "maxRoughness"]
        );
        for (node, keys) in [
            (trace, &["maxRoughness", "thickness"][..]),
            (composite, &["intensity", "maxRoughness"][..]),
        ] {
            for key in keys {
                let actual = node.params.iter().find(|param| param.key == *key).unwrap();
                let expected = pass.params.iter().find(|param| param.key == *key).unwrap();
                let mut expected = expected.clone();
                expected.binding = ParamBinding::Node;
                assert_eq!(actual, &expected);
            }
        }
        let inputs = &composite.inputs;
        assert_eq!(
            inputs.iter().map(|port| port.key).collect::<Vec<_>>(),
            ["color", "depth", "normal", "material", "trace"]
        );
        assert!(
            inputs
                .iter()
                .find(|port| port.key == "color")
                .unwrap()
                .required
        );
        assert!(
            inputs
                .iter()
                .find(|port| port.key == "trace")
                .unwrap()
                .required
        );
        assert_eq!(trace.budgets[0].sampled_textures, 14);
        assert_eq!(composite.budgets[0].sampled_textures, 16);
        assert!(trace.budgets[0].sampled_textures <= limits.sampled_textures);
        assert!(composite.budgets[0].sampled_textures <= limits.sampled_textures);
    }
}
