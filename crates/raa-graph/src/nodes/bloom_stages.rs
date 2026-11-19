//! Repeatable bloom pyramid stage nodes.
use super::ao_stages::{image_out, node_param, optional_in, post_stage, required_in, stage_budget};
use crate::catalog::{
    Category, GraphFormat, NodeFamily, NodeTypeSpec, ParamKind, ParamValue, ShaderStage, SizeSpec,
};

pub fn family() -> Box<dyn NodeFamily> {
    Box::new(BloomStages)
}

struct BloomStages;

impl NodeFamily for BloomStages {
    fn name(&self) -> &'static str {
        "bloom_stages"
    }

    fn node_types(&self) -> Vec<NodeTypeSpec> {
        let bloom = raa_bloom::BloomDesc::default();
        let mut types = vec![
            post_stage(
                "bloom.down",
                "bloom_stages",
                Category::Post,
                "Bloom down",
                120,
                vec![required_in("src", "Source", GraphFormat::Rgba16Float)],
                vec![image_out(
                    "level",
                    "Level",
                    GraphFormat::Rgba16Float,
                    SizeSpec::Input("src"),
                    1,
                )],
                vec![node_param(
                    "radius",
                    "Radius",
                    "Bloom",
                    ParamKind::Number,
                    ParamValue::Number(f64::from(bloom.radius)),
                    Some(0.0),
                    Some(1.0),
                    None,
                    "",
                    "crates/raa-bloom/src/desc.rs:21",
                )],
                vec![stage_budget(ShaderStage::Fragment, 1, 1, 0, 0, 1)],
            ),
            post_stage(
                "bloom.up",
                "bloom_stages",
                Category::Post,
                "Bloom up",
                120,
                vec![
                    required_in("src", "Source", GraphFormat::Rgba16Float),
                    required_in("base", "Base", GraphFormat::Rgba16Float),
                    optional_in("color", "Color", GraphFormat::Rgba16Float),
                ],
                vec![
                    image_out(
                        "level",
                        "Level",
                        GraphFormat::Rgba16Float,
                        SizeSpec::Input("base"),
                        1,
                    ),
                    image_out(
                        "color",
                        "Color",
                        GraphFormat::Rgba16Float,
                        SizeSpec::Scale(1.0),
                        1,
                    ),
                ],
                vec![
                    node_param(
                        "radius",
                        "Radius",
                        "Bloom",
                        ParamKind::Number,
                        ParamValue::Number(f64::from(bloom.radius)),
                        Some(0.0),
                        Some(1.0),
                        None,
                        "",
                        "crates/raa-bloom/src/desc.rs:21",
                    ),
                    node_param(
                        "strength",
                        "Strength",
                        "Bloom",
                        ParamKind::Number,
                        ParamValue::Number(f64::from(bloom.strength)),
                        Some(0.0),
                        Some(1.0),
                        None,
                        "",
                        "crates/raa-bloom/src/desc.rs:20",
                    ),
                ],
                vec![stage_budget(ShaderStage::Fragment, 2, 1, 0, 0, 1)],
            ),
        ];
        types[0].description = "Allocates half the source size for this bloom level.";
        types[1].description = "Allocates the base size for this bloom level.";
        types
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{ParamBinding, PortType};

    #[test]
    fn bloom_stage_catalog_contract() {
        let catalog = crate::Catalog::build();
        let down = catalog.node("bloom.down").expect("bloom down node");
        let up = catalog.node("bloom.up").expect("bloom up node");

        for node in [down, up] {
            assert_eq!(node.family, "bloom_stages");
            assert_eq!(node.category, Category::Post);
            assert_eq!(node.default_order, 120);
            assert_eq!(node.pass, None);
            assert!(node.multi_instance);
            assert_eq!(node.version, 1);
            assert!(!node.contributor);
            assert!(!node.side_effect);
            assert!(!node.required_in_graph);
        }
        assert!(down.description.contains("half the source size"));
        assert!(up.description.contains("base size"));

        assert_eq!(
            down.inputs
                .iter()
                .map(|port| (port.key, port.ty.clone(), port.required, port.absent_ok))
                .collect::<Vec<_>>(),
            [("src", PortType::Image2d, true, false)]
        );
        assert_eq!(
            down.outputs
                .iter()
                .map(|port| (port.key, port.format.clone(), port.size.clone()))
                .collect::<Vec<_>>(),
            [(
                "level",
                Some(GraphFormat::Rgba16Float),
                SizeSpec::Input("src")
            )]
        );
        assert_eq!(
            up.inputs
                .iter()
                .map(|port| (port.key, port.ty.clone(), port.required, port.absent_ok))
                .collect::<Vec<_>>(),
            [
                ("src", PortType::Image2d, true, false),
                ("base", PortType::Image2d, true, false),
                ("color", PortType::Image2d, false, true),
            ]
        );
        assert_eq!(
            up.outputs
                .iter()
                .map(|port| (port.key, port.format.clone(), port.size.clone()))
                .collect::<Vec<_>>(),
            [
                (
                    "level",
                    Some(GraphFormat::Rgba16Float),
                    SizeSpec::Input("base")
                ),
                (
                    "color",
                    Some(GraphFormat::Rgba16Float),
                    SizeSpec::Scale(1.0)
                ),
            ]
        );
        assert_eq!(
            down.params
                .iter()
                .map(|param| param.key)
                .collect::<Vec<_>>(),
            ["radius"]
        );
        assert_eq!(
            up.params.iter().map(|param| param.key).collect::<Vec<_>>(),
            ["radius", "strength"]
        );

        let pass = catalog.node("pass.bloom").expect("bloom pass node");
        for (node, keys) in [(down, &["radius"][..]), (up, &["radius", "strength"][..])] {
            for key in keys {
                let mut actual = node
                    .params
                    .iter()
                    .find(|param| param.key == *key)
                    .unwrap()
                    .clone();
                let mut expected = pass
                    .params
                    .iter()
                    .find(|param| param.key == *key)
                    .unwrap()
                    .clone();
                actual.binding = ParamBinding::Node;
                expected.binding = ParamBinding::Node;
                assert_eq!(actual, expected, "parameter {key}");
            }
        }
    }
}
