//! Editor-facing nodes for the experimental shared reconstruction package.

use super::ao_stages::{image_out, node_param, post_stage, required_in, stage_budget};
use crate::catalog::{
    Category, GraphFormat, NodeFamily, NodeTypeSpec, ParamKind, ParamValue, PortType, ShaderStage,
    SizeSpec,
};

pub fn family() -> Box<dyn NodeFamily> {
    Box::new(Neural)
}

struct Neural;

fn model() -> crate::catalog::ParamSpec {
    node_param(
        "model",
        "Experimental model ID",
        "Model",
        ParamKind::String,
        ParamValue::String(String::new()),
        None,
        None,
        None,
        "",
        "Registered real-frame FP32 package. Empty uses the analytic path.",
    )
}

impl NodeFamily for Neural {
    fn name(&self) -> &'static str {
        "neural"
    }

    fn node_types(&self) -> Vec<NodeTypeSpec> {
        let mut pairs = required_in(
            "pairs",
            "Weighted visibility pairs",
            GraphFormat::Rgba16Float,
        );
        pairs.ty = PortType::Image2dArray;
        pairs.layers = 2;
        let mut pairs_out = image_out(
            "pairs",
            "Reconstructed visibility pairs",
            GraphFormat::Rgba16Float,
            SizeSpec::Scale(1.0),
            1,
        );
        pairs_out.ty = PortType::Image2dArray;
        pairs_out.layers = 2;
        let mut area = post_stage(
            "neural.area-reconstruction",
            "neural",
            Category::Gi,
            "Area ray reconstruction (experimental)",
            43,
            vec![pairs],
            vec![pairs_out],
            vec![model()],
            vec![stage_budget(ShaderStage::Compute, 3, 0, 1, 4, 2)],
        );
        area.multi_instance = false;
        area.description = "Experimental joint FP32 area head for stratified pairs. An empty model ID passes the original pairs through. Trained package availability is checked by the engine.";

        let incident = required_in(
            "incident",
            "Linear RGB and hit confidence",
            GraphFormat::Rgba32Float,
        );
        let reconstructed = image_out(
            "incident",
            "Reconstructed incident radiance",
            GraphFormat::Rgba32Float,
            SizeSpec::Scale(1.0),
            1,
        );
        let mut reflection = post_stage(
            "neural.reflection-reconstruction",
            "neural",
            Category::Post,
            "Reflection ray reconstruction (operator only)",
            65,
            vec![incident],
            vec![reconstructed],
            vec![model()],
            vec![stage_budget(ShaderStage::Compute, 3, 0, 1, 4, 2)],
        );
        reflection.multi_instance = false;
        reflection.description = "Experimental screen-hit incident-radiance operator. The current fitted package has no trained reflection head. The empty model ID passes native incident radiance through.";

        let mut color = required_in("color", "Current real HDR", GraphFormat::Rgba16Float);
        color.generic = true;
        let mut generation = post_stage(
            "neural.frame-generation",
            "neural",
            Category::Output,
            "Frame generation (experimental)",
            109,
            vec![color],
            vec![],
            vec![node_param(
                "model",
                "Experimental model ID",
                "Model",
                ParamKind::String,
                ParamValue::String(String::new()),
                None,
                None,
                None,
                "",
                "Registered generated-frame FP32 package. Empty uses analytic camera reprojection.",
            )],
            vec![],
        );
        generation.multi_instance = false;
        generation.side_effect = true;
        generation.description = "Opt-in presentation at the real HDR endpoint for supported camera history. A fitted package adds a bounded learned HDR residual. An empty model ID uses analytic reprojection. The fitted camera profile is experimental.";

        vec![area, reflection, generation]
    }

    fn extend_types(&self, types: &mut Vec<NodeTypeSpec>) {
        let mut sr = types
            .iter()
            .find(|spec| spec.id == "pass.taa")
            .expect("TAA specification")
            .clone();
        sr.id = "neural.super-resolution";
        sr.family = self.name();
        sr.multi_instance = false;
        sr.label = "Temporal super-resolution (experimental)";
        sr.description = "Experimental continuous-offset SR from shared recurrent features. With an empty model ID the analytic temporal resolve runs.";
        sr.params.push(model());
        types.push(sr);
    }
}
