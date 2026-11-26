//! Family `exposure`: the automatic exposure node `pass.exposure.auto`, which places the S46
//! `exposure.auto` entry (raa-exposure, order 125: after bloom, before grading) in the colour
//! chain. Its parameters are node-bound; the executor extension `exposure` (exec_ext/exposure.rs)
//! converts them with [`desc_from_params`] and writes them to every auto-exposure handle.
//!
//! The pass measures a luminance histogram of every 4th pixel, averages the log luminance over a
//! percentile band (the darkest `low` and brightest `1 - high` fractions are left out, which keeps
//! a bright sky or sun patch from driving it), adapts that average over real time with separate
//! time constants towards a brighter and a darker scene, and multiplies the HDR colour by
//! `2^compensation * 0.18 / average` (clamped to `[2^minEv, 2^maxEv]`). The tone mapper's own
//! exposure still applies after it. With `enabled` off the node is a pass-through: the exposure
//! is clamped to exactly 1, so the image equals the graph without the node.

use raa_exposure::AutoExposureDesc;

use crate::catalog::{
    Category, GraphFormat, NodeFamily, NodeTypeSpec, ParamBinding, ParamKind, ParamSpec,
    ParamValue, PortSpec, PortType, Rebuild, ShaderStage, SizeSpec, StageBudget,
};

pub(crate) const EXPOSURE_TYPE: &str = "pass.exposure.auto";
const SOURCE: &str = "crates/raa-exposure/src/desc.rs";

/// Default adaptation time constant towards a brighter scene (seconds).
pub(crate) const DEFAULT_BRIGHTER_S: f64 = 0.3;
/// Default adaptation time constant towards a darker scene (seconds).
pub(crate) const DEFAULT_DARKER_S: f64 = 0.6;

/// The `color` sources of the node: the chain entries before order 125, nearest first.
const COLOR_SOURCES: &[&str] = &[
    "pass.bloom.color",
    "pass.dof.color",
    "pass.taa.color",
    "pass.particles.blend.color",
    "pass.transparent.oit.color",
    "pass.opaque.copy.color",
    "pass.sunshafts.color",
    "pass.fog.color",
    "pass.ssr.color",
    "pass.gi.history.color",
    "pass.forward.color",
];

pub fn family() -> Box<dyn NodeFamily> {
    Box::new(Exposure)
}

struct Exposure;

#[allow(clippy::too_many_arguments)]
fn param(
    key: &'static str,
    label: &'static str,
    kind: ParamKind,
    default: ParamValue,
    range: Option<(f64, f64)>,
    step: Option<f64>,
    unit: &'static str,
) -> ParamSpec {
    ParamSpec {
        key,
        label,
        group: "Auto exposure",
        kind,
        default,
        min: range.map(|r| r.0),
        max: range.map(|r| r.1),
        step,
        options: &[],
        unit,
        rebuild: Rebuild::None,
        primary: false,
        unstated_range: false,
        binding: ParamBinding::Node,
        source: SOURCE,
    }
}

fn color_port(key: &'static str, sources: &'static [&'static str], required: bool) -> PortSpec {
    PortSpec {
        key,
        label: "Color",
        ty: PortType::Image2d,
        format: Some(GraphFormat::Rgba16Float),
        size: SizeSpec::Scale(1.0),
        mips: 1,
        layers: 1,
        required,
        default_source: sources,
        absent_ok: false,
        generic: false,
        history: false,
        slot: None,
    }
}

impl NodeFamily for Exposure {
    fn name(&self) -> &'static str {
        "exposure"
    }

    fn node_types(&self) -> Vec<NodeTypeSpec> {
        let d = AutoExposureDesc::default();
        let number = |key, label, value: f64, range: (f64, f64), step: f64, unit| {
            param(
                key,
                label,
                ParamKind::Number,
                ParamValue::Number(value),
                Some(range),
                Some(step),
                unit,
            )
        };
        let params = vec![
            param(
                "enabled",
                "Adapt the exposure (off: pass-through, the fixed tone-mapper exposure only)",
                ParamKind::Bool,
                ParamValue::Bool(false),
                None,
                None,
                "",
            ),
            number(
                "compensationEv",
                "Compensation: the band average maps to middle grey (0.18) times 2^compensation",
                f64::from(d.compensation_ev),
                (-8.0, 8.0),
                0.1,
                "EV",
            ),
            number(
                "brighterSeconds",
                "Adaptation time constant towards a brighter scene",
                DEFAULT_BRIGHTER_S,
                (0.01, 30.0),
                0.05,
                "s",
            ),
            number(
                "darkerSeconds",
                "Adaptation time constant towards a darker scene",
                DEFAULT_DARKER_S,
                (0.01, 30.0),
                0.05,
                "s",
            ),
            number(
                "lowPercent",
                "Metering band start (histogram fraction; darker pixels are ignored)",
                f64::from(d.low_percent),
                (0.0, 0.99),
                0.01,
                "",
            ),
            number(
                "highPercent",
                "Metering band end (histogram fraction; brighter pixels, sky and sun patches, are ignored)",
                f64::from(d.high_percent),
                (0.01, 1.0),
                0.01,
                "",
            ),
            number(
                "minEv",
                "Lowest adapted exposure",
                f64::from(d.min_ev),
                (-16.0, 16.0),
                0.5,
                "EV",
            ),
            number(
                "maxEv",
                "Highest adapted exposure",
                f64::from(d.max_ev),
                (-16.0, 16.0),
                0.5,
                "EV",
            ),
        ];
        vec![NodeTypeSpec {
            id: EXPOSURE_TYPE,
            version: 1,
            category: Category::Post,
            label: "Auto exposure",
            description: "Automatic exposure from a luminance histogram (percentile band, asymmetric adaptation over real time) before grading and tone mapping; off by default.",
            pass: Some(raa_exposure::AUTO_EXPOSURE_PASS_NAME),
            contributor: false,
            default_order: raa_exposure::AUTO_EXPOSURE_ORDER,
            multi_instance: false,
            side_effect: false,
            required_in_graph: false,
            inputs: vec![color_port("color", COLOR_SOURCES, true)],
            outputs: vec![color_port("color", &[], false)],
            params,
            budgets: vec![
                // The compute pair (raa-exposure/src/feature.rs): params, HDR source, histogram, state.
                StageBudget {
                    stage: ShaderStage::Compute,
                    sampled_textures: 1,
                    samplers: 0,
                    storage_textures: 0,
                    storage_buffers: 2,
                    uniform_buffers: 1,
                },
                // The apply draw: params, HDR source, state (read-only).
                StageBudget {
                    stage: ShaderStage::Fragment,
                    sampled_textures: 1,
                    samplers: 0,
                    storage_textures: 0,
                    storage_buffers: 1,
                    uniform_buffers: 1,
                },
            ],
            upgrade: |from, _| Err(format!("no upgrade from version {from}")),
            family: self.name(),
        }]
    }
}

/// The description a node's parameters select; absent or mistyped values keep the defaults.
/// `enabled` off clamps the exposure to exactly 1 (pass-through). The result may still be
/// refused by [`AutoExposureDesc::validate`] (an inverted band).
pub(crate) fn desc_from_params(params: &[(String, ParamValue)]) -> AutoExposureDesc {
    let mut d = AutoExposureDesc {
        enabled: true,
        speed_up: (1.0 / DEFAULT_BRIGHTER_S) as f32,
        speed_down: (1.0 / DEFAULT_DARKER_S) as f32,
        ..AutoExposureDesc::default()
    };
    let mut adapt = false;
    for (key, value) in params {
        let number = match value {
            ParamValue::Number(x) => Some(*x),
            ParamValue::Int(n) => Some(*n as f64),
            _ => None,
        };
        match (key.as_str(), value, number) {
            ("enabled", ParamValue::Bool(b), _) => adapt = *b,
            ("compensationEv", _, Some(x)) => d.compensation_ev = x as f32,
            ("brighterSeconds", _, Some(x)) => d.speed_up = (1.0 / x.max(1e-3)) as f32,
            ("darkerSeconds", _, Some(x)) => d.speed_down = (1.0 / x.max(1e-3)) as f32,
            ("lowPercent", _, Some(x)) => d.low_percent = x as f32,
            ("highPercent", _, Some(x)) => d.high_percent = x as f32,
            ("minEv", _, Some(x)) => d.min_ev = x as f32,
            ("maxEv", _, Some(x)) => d.max_ev = x as f32,
            _ => {}
        }
    }
    if !adapt {
        d.min_ev = 0.0;
        d.max_ev = 0.0;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_is_an_exact_pass_through_and_on_takes_the_time_constants() {
        let off = desc_from_params(&[]);
        assert_eq!((off.min_ev, off.max_ev), (0.0, 0.0));
        assert!(off.validate().is_ok());
        let on = desc_from_params(&[
            ("enabled".into(), ParamValue::Bool(true)),
            ("brighterSeconds".into(), ParamValue::Number(0.25)),
            ("darkerSeconds".into(), ParamValue::Number(0.5)),
            ("compensationEv".into(), ParamValue::Number(-1.0)),
        ]);
        assert_eq!((on.speed_up, on.speed_down, on.compensation_ev), (4.0, 2.0, -1.0));
        assert_eq!((on.min_ev, on.max_ev), (-6.0, 6.0));
    }
}
