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

/// The `shadowRate` options (raa-pipeline ShadowRate): s at the phase pixel of R frames, at
/// every pixel of R frames, at every pixel of every frame.
pub const SHADOW_RATES: [&str; 3] = ["phase", "real", "every"];
/// The `irradianceRate` options (raa-pipeline IrradianceRate): E at the phase pixel of R frames
/// (30 Hz), at the phase pixel of every frame (60 Hz interlaced), at every pixel of every frame.
pub const IRRADIANCE_RATES: [&str; 3] = ["phase30", "phase60", "full"];
/// Defaults: the signal-sweep rank 1 (s at every pixel every frame, E 1/2 60 Hz interlaced).
pub const DEFAULT_SHADOW_RATE: usize = 2;
pub const DEFAULT_IRRADIANCE_RATE: usize = 1;

/// `(key, label, default, source)` of the number parameters; the defaults are the learned
/// constants of the analytic ablation (`analytic-constants.json`), except the sampled-pixel alphas
/// (0.5, measured on the 3090 against the 4K label, raa-taa-quality 2026-11-26): the static
/// per-pixel PCF rotation sampled once per 2x2 block left 2 px stairs on moving shadow edges at 0.985.
const NUMBERS: [(&str, &str, f64, &str); 6] = [
    ("alphaSHit", "Shadow alpha (sampled pixel)", 0.5, "alpha_s_hit"),
    ("alphaSMiss", "Shadow alpha (upsampled pixel)", 0.980_347_156_524_658_2, "alpha_s_miss"),
    ("alphaEHit", "Irradiance/AO alpha (sampled pixel)", 0.5, "alpha_ea_hit"),
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
        ParamValue::Bool(true),
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
        "An upsampled pixel keeps its reprojected history clamped to the range of the neighbouring phase samples (full-resolution shadow edges) instead of their interpolation.",
    ));
    params.push(node_param(
        "historyCubic",
        "Cubic history",
        "Reconstruction",
        ParamKind::Bool,
        ParamValue::Bool(true),
        None,
        None,
        None,
        "",
        "Resamples the accumulated signals with a 5-tap Catmull-Rom instead of bilinear (sharper edges in motion).",
    ));
    let mut shadow_rate = node_param(
        "shadowRate",
        "Sun visibility sampling",
        "Signals",
        ParamKind::Enum,
        ParamValue::Enum(SHADOW_RATES[DEFAULT_SHADOW_RATE].into()),
        None,
        None,
        None,
        "",
        "Where the forward evaluates the sun visibility s: \"phase\" one pixel per 2x2 block on R frames (30 Hz), \"real\" every pixel of R frames, \"every\" every pixel of every frame (G frames sample the R frame's cascade maps; shadow.contact runs every frame).",
    );
    shadow_rate.options = &SHADOW_RATES;
    params.push(shadow_rate);
    let mut irradiance_rate = node_param(
        "irradianceRate",
        "Probe irradiance sampling",
        "Signals",
        ParamKind::Enum,
        ParamValue::Enum(IRRADIANCE_RATES[DEFAULT_IRRADIANCE_RATE].into()),
        None,
        None,
        None,
        "",
        "Where the forward evaluates the probe irradiance E: \"phase30\" one pixel per 2x2 block on R frames, \"phase60\" one pixel per block on every frame (G frames take the next unvisited phase; the DDGI screen resolve runs every frame), \"full\" every pixel of every frame (the DDGI screen resolve runs at scale 1 every frame). AO stays at the R-frame phase pixel.",
    );
    irradiance_rate.options = &IRRADIANCE_RATES;
    params.push(irradiance_rate);
    params.push(node_param(
        "shadowFull",
        "Sun visibility at every pixel (legacy)",
        "Signals",
        ParamKind::Bool,
        ParamValue::Bool(false),
        None,
        None,
        None,
        "",
        "Signal-sweep study switch: with shadowRate \"phase\", R frames evaluate s at every pixel (= shadowRate \"real\").",
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
        node.description = "Accumulates the signals of forward lightingOutput \"layers-interlaced\" into a full-resolution history reprojected by the prepass motion, fills G-frame disocclusions from a 12-tap ring, and composites C = d0*s + (kd*E + s0)*a + em. The node sets the forward's signal rates: by default s at every pixel of every frame (cascade maps at 30 Hz), probe irradiance E at one phase pixel per 2x2 block on every frame (60 Hz interlaced), AO at one phase pixel per block on R frames (30 Hz); and a camera jitter period of 15 while installed. One full-resolution compute dispatch per frame; feeds pass.taa / ssr / fog / tonemap.";
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
                "historyCubic",
                "shadowRate",
                "irradianceRate",
                "shadowFull",
                "debug"
            ]
        );
    }
}
