//! Real-time denoiser nodes in the spirit of NRD SIGMA (shadow visibility) and ReLAX
//! (radiance). Clean-room implementations from SVGF and the public ReBLUR/ReLAX/SIGMA
//! descriptions; no NRD source is used (its licence forbids derivatives in MIT code).
//!
//! Like the experimental `neural.*` nodes, these are executed by a graph frame that runs in
//! front of the consumer pass of their output and rebinds that pass's input, so their GPU time
//! is reported inside the consumer pass.

use super::ao_stages::{gbuffer_in, image_out, node_param, post_stage, required_in, stage_budget};
use crate::catalog::{
    Category, GraphFormat, NodeFamily, NodeTypeSpec, ParamKind, ParamSpec, ParamValue, PortSpec,
    PortType, ShaderStage, SizeSpec,
};

pub fn family() -> Box<dyn NodeFamily> {
    Box::new(Denoise)
}

struct Denoise;

fn number(
    key: &'static str,
    label: &'static str,
    group: &'static str,
    default: f64,
    min: f64,
    max: f64,
    source: &'static str,
) -> ParamSpec {
    node_param(
        key,
        label,
        group,
        ParamKind::Number,
        ParamValue::Number(default),
        Some(min),
        Some(max),
        None,
        "",
        source,
    )
}

fn int(
    key: &'static str,
    label: &'static str,
    group: &'static str,
    default: i64,
    min: f64,
    max: f64,
    source: &'static str,
) -> ParamSpec {
    node_param(
        key,
        label,
        group,
        ParamKind::Int,
        ParamValue::Int(default),
        Some(min),
        Some(max),
        Some(1.0),
        "",
        source,
    )
}

fn optional(key: &'static str, label: &'static str, format: GraphFormat) -> PortSpec {
    let mut port = required_in(key, label, format);
    port.required = false;
    port.absent_ok = true;
    port
}

fn motion_in() -> PortSpec {
    let mut port = optional("motion", "Motion", GraphFormat::Rg16Float);
    port.default_source = &["pass.prepass.motion", "pass.vg.gbuffer.motion"];
    port.absent_ok = false;
    port
}

fn gbuffers() -> Vec<PortSpec> {
    vec![gbuffer_in("depth"), gbuffer_in("normal"), motion_in()]
}

impl NodeFamily for Denoise {
    fn name(&self) -> &'static str {
        "denoise"
    }

    fn node_types(&self) -> Vec<NodeTypeSpec> {
        let mut pairs = required_in(
            "signal",
            "Weighted visibility pairs",
            GraphFormat::Rgba16Float,
        );
        pairs.ty = PortType::Image2dArray;
        pairs.layers = 2;
        let mut pairs_out = image_out(
            "signal",
            "Denoised visibility pairs",
            GraphFormat::Rgba16Float,
            SizeSpec::Scale(1.0),
            1,
        );
        pairs_out.ty = PortType::Image2dArray;
        pairs_out.layers = 2;
        let mut sigma_inputs = vec![pairs];
        sigma_inputs.extend(gbuffers());
        let mut sigma = post_stage(
            "denoise.sigma",
            "denoise",
            Category::Shadows,
            "Shadow denoiser (SIGMA-style)",
            43,
            sigma_inputs,
            vec![pairs_out],
            vec![
                int(
                    "historyFrames",
                    "History frames",
                    "Temporal",
                    8,
                    0.0,
                    64.0,
                    "Maximum accumulated frames per pixel and light slot; 0 disables history.",
                ),
                number(
                    "radius",
                    "Blur radius",
                    "Spatial",
                    4.0,
                    0.0,
                    16.0,
                    "Penumbra blur radius in pixels before history shrinks it.",
                ),
                number(
                    "normalPower",
                    "Normal power",
                    "Spatial",
                    32.0,
                    0.0,
                    256.0,
                    "Exponent of the normal edge-stopping weight.",
                ),
                number(
                    "depthSigma",
                    "Depth sigma",
                    "Spatial",
                    1.0,
                    0.0,
                    16.0,
                    "Depth edge-stopping tolerance in depth-gradient units.",
                ),
                number(
                    "clampSigma",
                    "History clamp",
                    "Temporal",
                    1.5,
                    0.0,
                    8.0,
                    "Neighbourhood visibility clamp width in standard deviations.",
                ),
            ],
            vec![stage_budget(ShaderStage::Compute, 7, 0, 2, 0, 1)],
        );
        sigma.description = "Denoises stratified area-light visibility pairs (S, U per slot): short reprojected history with per-slot light-change reset and neighbourhood clamp, then a penumbra-adaptive edge-aware blur in pair space. Clean-room design after the public SIGMA description.";

        let signal = required_in(
            "signal",
            "Linear RGB and confidence",
            GraphFormat::Rgba32Float,
        );
        let mut relax_inputs = vec![signal];
        relax_inputs.extend(gbuffers());
        relax_inputs.push(optional(
            "hitMeta",
            "Hit metadata (class, validity, distance)",
            GraphFormat::Rgba32Float,
        ));
        relax_inputs.push(optional(
            "trace",
            "Hit UV, confidence, cone",
            GraphFormat::Rgba16Float,
        ));
        let mut relax = post_stage(
            "denoise.relax",
            "denoise",
            Category::Post,
            "Radiance denoiser (ReLAX-style)",
            65,
            relax_inputs,
            vec![image_out(
                "signal",
                "Denoised radiance",
                GraphFormat::Rgba32Float,
                SizeSpec::Scale(1.0),
                1,
            )],
            vec![
                int(
                    "historyFrames",
                    "History frames",
                    "Temporal",
                    16,
                    0.0,
                    64.0,
                    "Maximum accumulated frames for rough surfaces; mirrors use a quarter.",
                ),
                int(
                    "iterations",
                    "A-trous iterations",
                    "Spatial",
                    2,
                    0.0,
                    5.0,
                    "Number of edge-aware a-trous passes (step 1, 2, 4, ...).",
                ),
                number(
                    "phiLuminance",
                    "Luminance sigma",
                    "Spatial",
                    4.0,
                    0.0,
                    64.0,
                    "Luminance edge-stopping width in standard deviations.",
                ),
                number(
                    "normalPower",
                    "Normal power",
                    "Spatial",
                    64.0,
                    0.0,
                    256.0,
                    "Exponent of the normal weight at roughness 0 (scaled down with roughness).",
                ),
                number(
                    "depthSigma",
                    "Depth sigma",
                    "Spatial",
                    1.0,
                    0.0,
                    16.0,
                    "Depth edge-stopping tolerance in depth-gradient units.",
                ),
                number(
                    "clampSigma",
                    "History clamp",
                    "Temporal",
                    2.0,
                    0.0,
                    8.0,
                    "Colour box clamp of history in standard deviations; 0 disables it.",
                ),
            ],
            vec![stage_budget(ShaderStage::Compute, 8, 0, 2, 0, 1)],
        );
        relax.description = "Denoises a radiance signal whose alpha is a confidence or validity weight (SSR incident radiance, noisy GI): reprojected accumulation with roughness-dependent history, specular history validated by hit reprojection when a trace is linked, colour clamp, then SVGF a-trous with luminance-variance, normal and depth weights. Pixels with zero alpha (or invalid hit metadata) pass through unchanged. Clean-room design after SVGF and the public ReLAX description.";

        vec![sigma, relax]
    }
}

#[cfg(test)]
mod tests {
    use crate::catalog::Catalog;

    #[test]
    fn denoise_catalog_contract() {
        let c = Catalog::build();
        let sigma = c.node("denoise.sigma").expect("sigma node");
        assert_eq!(sigma.family, "denoise");
        assert_eq!(sigma.pass, None);
        let input = sigma.inputs.iter().find(|p| p.key == "signal").unwrap();
        let output = sigma.outputs.iter().find(|p| p.key == "signal").unwrap();
        assert!(input.required);
        assert_eq!((input.layers, output.layers), (2, 2));
        let relax = c.node("denoise.relax").expect("relax node");
        assert!(relax.inputs.iter().filter(|p| p.required).map(|p| p.key).eq(["signal"]));
        for key in ["hitMeta", "trace"] {
            let port = relax.inputs.iter().find(|p| p.key == key).unwrap();
            assert!(port.absent_ok && !port.required);
        }
    }
}
