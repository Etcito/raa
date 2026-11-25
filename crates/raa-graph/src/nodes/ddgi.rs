//! Family `ddgi`: the DDGI probe volume node `pass.gi.ddgi` (Majercik et al. 2019 with the 2021
//! relocation and classification). It can replace `pass.gi.spherical`: forward and VG shading read
//! its volume through the same lighting binding. Its parameters are node-bound; the executor
//! extension `ddgi` (exec_ext/ddgi.rs) converts them with [`settings_from_params`].

use raa_sdf_trace::mesh::{DDGI_GI_NAME, DdgiDebug, DdgiSettings};

use crate::catalog::{
    Category, NodeFamily, NodeTypeSpec, ParamBinding, ParamKind, ParamSpec, ParamValue, Rebuild,
    ShaderStage, StageBudget,
};
use crate::error::{Issue, Severity, Stage};
use crate::format::NodeDoc;
use crate::validate::ValidateCtx;

pub(crate) const DDGI_TYPE: &str = "pass.gi.ddgi";
const SOURCE: &str = "crates/raa-sdf-trace/src/mesh/ddgi.rs";
const DEBUG_OPTIONS: &[&str] = &["off", "irradiance", "coverage", "resolve"];

pub fn family() -> Box<dyn NodeFamily> {
    Box::new(Ddgi)
}

struct Ddgi;

#[allow(clippy::too_many_arguments)]
fn param(
    key: &'static str,
    label: &'static str,
    kind: ParamKind,
    default: ParamValue,
    range: Option<(f64, f64)>,
    step: Option<f64>,
    unit: &'static str,
    rebuild: Rebuild,
) -> ParamSpec {
    ParamSpec {
        key,
        label,
        group: "DDGI",
        kind,
        default,
        min: range.map(|r| r.0),
        max: range.map(|r| r.1),
        step,
        options: &[],
        unit,
        rebuild,
        primary: false,
        unstated_range: false,
        binding: ParamBinding::Node,
        source: SOURCE,
    }
}

fn number(value: f32) -> ParamValue {
    ParamValue::Number(f64::from(value))
}

impl NodeFamily for Ddgi {
    fn name(&self) -> &'static str {
        "ddgi"
    }

    fn node_types(&self) -> Vec<NodeTypeSpec> {
        let d = DdgiSettings::default();
        let count = |key, label, value: u32| {
            param(
                key,
                label,
                ParamKind::Int,
                ParamValue::Int(i64::from(value)),
                Some((2.0, 128.0)),
                Some(1.0),
                "",
                Rebuild::Resources,
            )
        };
        let flag = |key, label, value: bool| {
            param(
                key,
                label,
                ParamKind::Bool,
                ParamValue::Bool(value),
                None,
                None,
                "",
                Rebuild::None,
            )
        };
        let mut debug = param(
            "debug",
            "DDGI diagnostic",
            ParamKind::Enum,
            ParamValue::Enum("off".into()),
            None,
            None,
            "",
            Rebuild::None,
        );
        debug.options = DEBUG_OPTIONS;
        let params = vec![
            count("probeCountX", "Probes X", d.counts[0]),
            count("probeCountY", "Probes Y (up)", d.counts[1]),
            count("probeCountZ", "Probes Z", d.counts[2]),
            param(
                "probeSpacing",
                "Probe spacing",
                ParamKind::Number,
                number(d.spacing),
                Some((0.05, 100.0)),
                Some(0.05),
                "m",
                Rebuild::Resources,
            ),
            param(
                "raysPerProbe",
                "Rays per probe",
                ParamKind::Int,
                ParamValue::Int(i64::from(d.rays_per_probe)),
                Some((16.0, 256.0)),
                Some(16.0),
                "",
                Rebuild::None,
            ),
            param(
                "hysteresis",
                "Hysteresis",
                ParamKind::Number,
                number(d.hysteresis),
                Some((0.0, 0.995)),
                Some(0.005),
                "",
                Rebuild::None,
            ),
            param(
                "updateFraction",
                "Max probes updated per frame (fraction)",
                ParamKind::Number,
                number(d.update_fraction),
                Some((0.01, 1.0)),
                Some(0.01),
                "",
                Rebuild::None,
            ),
            param(
                "budgetMs",
                "Update GPU budget",
                ParamKind::Number,
                number(d.budget_ms),
                Some((0.05, 20.0)),
                Some(0.05),
                "ms",
                Rebuild::None,
            ),
            param(
                "burstMs",
                "Burst GPU budget (unconverged visible probes)",
                ParamKind::Number,
                number(d.burst_ms),
                Some((0.05, 20.0)),
                Some(0.05),
                "ms",
                Rebuild::None,
            ),
            param(
                "intensity",
                "Intensity",
                ParamKind::Number,
                number(d.intensity),
                Some((0.0, 16.0)),
                Some(0.05),
                "",
                Rebuild::None,
            ),
            param(
                "normalBias",
                "Normal bias (x spacing)",
                ParamKind::Number,
                number(d.normal_bias),
                Some((0.0, 1.0)),
                Some(0.01),
                "",
                Rebuild::None,
            ),
            param(
                "viewBias",
                "View bias (x spacing)",
                ParamKind::Number,
                number(d.view_bias),
                Some((0.0, 1.0)),
                Some(0.01),
                "",
                Rebuild::None,
            ),
            flag("relocation", "Probe relocation", d.relocation),
            flag("classification", "Probe classification", d.classification),
            flag("bounce", "Multiple bounces", d.bounce),
            flag("screenResolve", "Screen resolve pass", d.screen_resolve),
            debug,
        ];
        vec![NodeTypeSpec {
            id: DDGI_TYPE,
            version: 1,
            category: Category::Gi,
            label: "DDGI",
            description: "Ray-traced irradiance probe volume (DDGI) around the camera; replaces the spherical grid for forward lighting.",
            pass: Some(DDGI_GI_NAME),
            contributor: false,
            default_order: 57,
            multi_instance: false,
            side_effect: true,
            required_in_graph: false,
            inputs: vec![],
            outputs: vec![],
            params,
            budgets: vec![StageBudget {
                stage: ShaderStage::Compute,
                sampled_textures: 2,
                samplers: 1,
                storage_textures: 0,
                storage_buffers: 5,
                uniform_buffers: 4,
            }],
            upgrade: |from, _| Err(format!("no upgrade from version {from}")),
            family: self.name(),
        }]
    }

    fn validate_node(
        &self,
        node: &NodeDoc,
        spec: &NodeTypeSpec,
        _: &ValidateCtx<'_>,
        issues: &mut Vec<Issue>,
    ) {
        if spec.id != DDGI_TYPE {
            return;
        }
        let mut values = Vec::new();
        for (key, value) in &node.params {
            let Some(param) = spec.params.iter().find(|param| param.key == key) else {
                return;
            };
            let Ok(value) = ParamValue::from_json(&param.kind, value) else {
                return;
            };
            values.push((key.clone(), value));
        }
        if let Err(error) = settings_from_params(&values).validate() {
            issues.push(Issue {
                code: "graph.param_range",
                severity: Severity::Error,
                stage: Stage::Structure,
                node: Some(node.id.clone()),
                port: None,
                param: Some("probeCountX".into()),
                link: None,
                message: error.message().into(),
                fix: "Use at most 32768 probes, 2 to 128 per axis.".into(),
            });
        }
    }
}

/// The settings a node's parameters select; absent or mistyped values keep the defaults.
/// `enabled` is set: the node's presence turns the volume on.
pub(crate) fn settings_from_params(params: &[(String, ParamValue)]) -> DdgiSettings {
    let mut s = DdgiSettings {
        enabled: true,
        ..DdgiSettings::default()
    };
    for (key, value) in params {
        match (key.as_str(), value) {
            ("probeCountX", ParamValue::Int(n)) => s.counts[0] = (*n).clamp(0, 1 << 16) as u32,
            ("probeCountY", ParamValue::Int(n)) => s.counts[1] = (*n).clamp(0, 1 << 16) as u32,
            ("probeCountZ", ParamValue::Int(n)) => s.counts[2] = (*n).clamp(0, 1 << 16) as u32,
            ("probeSpacing", ParamValue::Number(x)) => s.spacing = *x as f32,
            ("raysPerProbe", ParamValue::Int(n)) => {
                s.rays_per_probe = (*n).clamp(0, 1 << 16) as u32
            }
            ("hysteresis", ParamValue::Number(x)) => s.hysteresis = *x as f32,
            ("updateFraction", ParamValue::Number(x)) => s.update_fraction = *x as f32,
            ("budgetMs", ParamValue::Number(x)) => s.budget_ms = *x as f32,
            ("burstMs", ParamValue::Number(x)) => s.burst_ms = *x as f32,
            ("intensity", ParamValue::Number(x)) => s.intensity = *x as f32,
            ("normalBias", ParamValue::Number(x)) => s.normal_bias = *x as f32,
            ("viewBias", ParamValue::Number(x)) => s.view_bias = *x as f32,
            ("relocation", ParamValue::Bool(b)) => s.relocation = *b,
            ("classification", ParamValue::Bool(b)) => s.classification = *b,
            ("bounce", ParamValue::Bool(b)) => s.bounce = *b,
            ("screenResolve", ParamValue::Bool(b)) => s.screen_resolve = *b,
            ("debug", ParamValue::Enum(name)) => {
                s.debug = match name.as_str() {
                    "irradiance" => DdgiDebug::Irradiance,
                    "coverage" => DdgiDebug::Coverage,
                    "resolve" => DdgiDebug::Resolve,
                    _ => DdgiDebug::Off,
                }
            }
            _ => {}
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_params_round_trip_to_valid_settings() {
        let spec = &Ddgi.node_types()[0];
        let params: Vec<_> = spec
            .params
            .iter()
            .map(|p| (p.key.to_string(), p.default.clone()))
            .collect();
        let settings = settings_from_params(&params)
            .validate()
            .expect("valid defaults");
        assert!(settings.enabled);
        assert_eq!(
            DdgiSettings {
                enabled: true,
                ..DdgiSettings::default()
            },
            settings
        );
    }
}
