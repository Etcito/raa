//! Family `lighting_reconstruct`: `pass.lighting.reconstruct`, the full-resolution accumulation
//! and composition of the forward lightingOutput "layers-interlaced" signals (no network: the
//! analytic ablation of the neural target, `nt_model.step` with an identity kernel and constant
//! alphas). Executed by the `lighting_reconstruct` extension as the pipeline entry
//! `pass.lighting.reconstruct#<id>`, a member of the HDR colour chain.

use super::ao_stages::{gbuffer_in, image_out, node_param, post_stage, required_in, stage_budget};
use crate::catalog::{
    Category, GraphFormat, NodeFamily, NodeTypeSpec, ParamKind, ParamSpec, ParamValue, PortSpec,
    ShaderStage, SizeSpec,
};

/// The node type id.
pub const TYPE: &str = "pass.lighting.reconstruct";

/// The `debug` options, in shader code order (0 = off).
pub const DEBUG_VIEWS: [&str; 6] = ["off", "shadow", "ao", "irradiance", "confidence", "holes"];

/// `(key, label, default, source)` of the number parameters; the defaults are the learned
/// constants of the analytic ablation (`analytic-constants.json`).
const NUMBERS: [(&str, &str, f64, &str); 6] = [
    ("alphaSHit", "Shadow alpha (sampled pixel)", 0.985_325_634_479_522_7, "alpha_s_hit"),
    ("alphaSMiss", "Shadow alpha (upsampled pixel)", 0.980_347_156_524_658_2, "alpha_s_miss"),
    ("alphaEHit", "Irradiance/AO alpha (sampled pixel)", 0.988_787_829_875_946, "alpha_ea_hit"),
    ("alphaEMiss", "Irradiance/AO alpha (upsampled pixel)", 0.979_948_699_474_334_7, "alpha_ea_miss"),
    ("qMiss", "Upsampled sample quality", 0.3, "q_miss"),
    ("decay", "Confidence decay per frame", 0.98, "decay"),
];

pub fn family() -> Box<dyn NodeFamily> {
    Box::new(LightingReconstruct)
}

struct LightingReconstruct;

fn motion_in() -> PortSpec {
    let mut port = required_in("motion", "Motion", GraphFormat::Rg16Float);
    port.required = false;
    port.default_source = &["pass.prepass.motion", "pass.vg.gbuffer.motion"];
    port
}

fn params() -> Vec<ParamSpec> {
    let mut params: Vec<ParamSpec> = NUMBERS
        .iter()
        .map(|&(key, label, default, source)| {
            node_param(
                key,
                label,
                "Reconstruction",
                ParamKind::Number,
                ParamValue::Number(default),
                Some(0.0),
                Some(1.0),
                None,
                "",
                source,
            )
        })
        .collect();
    params.push(node_param(
        "jitterCorrect",
        "Jitter correction",
        "Reconstruction",
        ParamKind::Bool,
        ParamValue::Bool(false),
        None,
        None,
        None,
        "",
        "Adds (previous - current) camera jitter to the history reprojection.",
    ));
    params.push(node_param(
        "historyClamp",
        "History clamp",
        "Reconstruction",
        ParamKind::Bool,
        ParamValue::Bool(true),
        None,
        None,
        None,
        "",
        "On R frames an upsampled pixel keeps its reprojected history clamped to the range of the neighbouring phase samples (full-resolution shadow edges) instead of their interpolation.",
    ));
    let mut debug = node_param(
        "debug",
        "Debug view",
        "Debug",
        ParamKind::Enum,
        ParamValue::Enum("off".into()),
        None,
        None,
        None,
        "",
        "Replaces the output colour by a signal visualisation.",
    );
    debug.options = &DEBUG_VIEWS;
    params.push(debug);
    params
}

impl NodeFamily for LightingReconstruct {
    fn name(&self) -> &'static str {
        "lighting_reconstruct"
    }

    fn node_types(&self) -> Vec<NodeTypeSpec> {
        let mut color = required_in("color", "Forward colour (d0, a = s)", GraphFormat::Rgba16Float);
        color.generic = true;
        let mut inputs = vec![
            color,
            required_in("layerKd", "Diffuse reflectance (a = AO)", GraphFormat::Rgba16Float),
            required_in("layerS0", "AO ambient at AO 1 (a = sky code)", GraphFormat::Rgba16Float),
            required_in("layerEm", "Emission and contributors", GraphFormat::Rgba16Float),
            required_in("layerE", "Probe irradiance x w (a = w)", GraphFormat::Rgba16Float),
        ];
        inputs.extend([gbuffer_in("depth"), gbuffer_in("normal"), motion_in()]);
        let mut output = image_out(
            "color",
            "Composited HDR colour",
            GraphFormat::Rgba16Float,
            SizeSpec::Scale(1.0),
            1,
        );
        output.generic = true;
        let mut node = post_stage(
            TYPE,
            "lighting_reconstruct",
            Category::Lighting,
            "Interlaced lighting reconstruction",
            61,
            inputs,
            vec![output],
            params(),
            // 8 inputs + 2 history images, 3 storage outputs (H0', H1', colour), params and frame.
            vec![stage_budget(ShaderStage::Compute, 10, 1, 3, 0, 2)],
        );
        node.description = "Accumulates the 30 Hz half-resolution signals of forward lightingOutput \"layers-interlaced\" (s, AO, probe irradiance at one phase pixel per 2x2 block on R frames) into a full-resolution history reprojected by the prepass motion, fills G-frame disocclusions from a 12-tap ring, and composites C = d0*s + (kd*E + s0)*a + em. One full-resolution compute dispatch per frame; feeds pass.taa / ssr / fog / tonemap.";
        vec![node]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Catalog;

    #[test]
    fn reconstruct_catalog_contract() {
        let c = Catalog::build();
        let node = c.node(TYPE).expect("reconstruct node");
        assert_eq!(node.pass, None);
        assert_eq!(
            node.inputs.iter().map(|p| p.key).collect::<Vec<_>>(),
            ["color", "layerKd", "layerS0", "layerEm", "layerE", "depth", "normal", "motion"]
        );
        assert_eq!(node.outputs.iter().map(|p| p.key).collect::<Vec<_>>(), ["color"]);
        assert_eq!(
            node.params.iter().map(|p| p.key).collect::<Vec<_>>(),
            [
                "alphaSHit",
                "alphaSMiss",
                "alphaEHit",
                "alphaEMiss",
                "qMiss",
                "decay",
                "jitterCorrect",
                "historyClamp",
                "debug"
            ]
        );
    }
}
