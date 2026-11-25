//! Family `nrc`: the neural radiance cache node `neural.radiance-cache` (Müller et al. 2021).
//! It can replace `pass.gi.spherical` or `pass.gi.ddgi`: forward shading reads its resolved
//! indirect diffuse through the GI screen slot. Its parameters are node-bound; the executor
//! extension `nrc` (exec_ext/nrc.rs) converts them with [`settings_from_params`].

use raa_sdf_trace::mesh::{NRC_GI_NAME, NrcSettings};

use crate::catalog::{
    Category, NodeFamily, NodeTypeSpec, ParamBinding, ParamKind, ParamSpec, ParamValue, Rebuild,
    ShaderStage, StageBudget,
};
use crate::error::{Issue, Severity, Stage};
use crate::format::NodeDoc;
use crate::validate::ValidateCtx;

pub(crate) const NRC_TYPE: &str = "neural.radiance-cache";
const SOURCE: &str = "crates/raa-sdf-trace/src/mesh/nrc.rs";
const SCALE_OPTIONS: &[&str] = &["eighth", "quarter", "half", "full"];
const SAMPLE_OPTIONS: &[&str] = &["4096", "8192", "16384", "32768", "65536"];

pub fn family() -> Box<dyn NodeFamily> {
    Box::new(Nrc)
}

struct Nrc;

fn param(
    key: &'static str,
    label: &'static str,
    kind: ParamKind,
    default: ParamValue,
    range: Option<(f64, f64)>,
    step: Option<f64>,
    rebuild: Rebuild,
) -> ParamSpec {
    ParamSpec {
        key,
        label,
        group: "Radiance cache",
        kind,
        default,
        min: range.map(|r| r.0),
        max: range.map(|r| r.1),
        step,
        options: &[],
        unit: "",
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

fn scale_name(divisor: u32) -> &'static str {
    match divisor {
        1 => "full",
        2 => "half",
        4 => "quarter",
        _ => "eighth",
    }
}

impl NodeFamily for Nrc {
    fn name(&self) -> &'static str {
        "nrc"
    }

    fn node_types(&self) -> Vec<NodeTypeSpec> {
        let d = NrcSettings::default();
        let mut scale = param(
            "scale",
            "Query resolution",
            ParamKind::Enum,
            ParamValue::Enum(scale_name(d.divisor).into()),
            None,
            None,
            Rebuild::Resources,
        );
        scale.options = SCALE_OPTIONS;
        let mut samples = param(
            "trainSamples",
            "Training samples per frame",
            ParamKind::Enum,
            ParamValue::Enum(d.train_samples.to_string()),
            None,
            None,
            Rebuild::Resources,
        );
        samples.options = SAMPLE_OPTIONS;
        let params = vec![
            param(
                "hiddenLayers",
                "Hidden layers (64 wide)",
                ParamKind::Int,
                ParamValue::Int(i64::from(d.hidden_layers)),
                Some((1.0, 5.0)),
                Some(1.0),
                Rebuild::Resources,
            ),
            samples,
            param(
                "learningRate",
                "Learning rate",
                ParamKind::Number,
                number(d.learning_rate),
                Some((0.0, 0.1)),
                Some(0.0005),
                Rebuild::None,
            ),
            param(
                "ema",
                "Weight averaging",
                ParamKind::Number,
                number(d.ema),
                Some((0.0, 0.9999)),
                Some(0.01),
                Rebuild::None,
            ),
            param(
                "train",
                "Train online",
                ParamKind::Bool,
                ParamValue::Bool(d.train),
                None,
                None,
                Rebuild::None,
            ),
            scale,
            param(
                "bounces",
                "Bounces before the cache",
                ParamKind::Int,
                ParamValue::Int(i64::from(d.bounces)),
                Some((0.0, 1.0)),
                Some(1.0),
                Rebuild::None,
            ),
            param(
                "temporalBlend",
                "Current frame weight",
                ParamKind::Number,
                number(d.temporal_blend),
                Some((0.01, 1.0)),
                Some(0.01),
                Rebuild::None,
            ),
        ];
        vec![NodeTypeSpec {
            id: NRC_TYPE,
            version: 1,
            category: Category::Gi,
            label: "Neural radiance cache (experimental)",
            description: "Online-trained neural radiance cache (Mueller et al. 2021) on the mesh BVH; supplies indirect diffuse to forward lighting in place of the spherical grid or DDGI.",
            pass: Some(NRC_GI_NAME),
            contributor: false,
            default_order: 56,
            multi_instance: false,
            side_effect: true,
            required_in_graph: false,
            inputs: vec![],
            outputs: vec![],
            params,
            budgets: vec![StageBudget {
                stage: ShaderStage::Compute,
                sampled_textures: 4,
                samplers: 1,
                storage_textures: 1,
                storage_buffers: 6,
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
        if spec.id != NRC_TYPE {
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
                param: Some("hiddenLayers".into()),
                link: None,
                message: error.message().into(),
                fix: "Use 1 to 5 hidden layers and the listed sample counts.".into(),
            });
        }
    }
}

/// The settings a node's parameters select; absent or mistyped values keep the defaults.
/// `enabled` is set: the node's presence turns the cache on.
pub(crate) fn settings_from_params(params: &[(String, ParamValue)]) -> NrcSettings {
    let mut s = NrcSettings {
        enabled: true,
        ..NrcSettings::default()
    };
    for (key, value) in params {
        match (key.as_str(), value) {
            ("hiddenLayers", ParamValue::Int(n)) => s.hidden_layers = (*n).clamp(0, 64) as u32,
            ("trainSamples", ParamValue::Enum(name)) => {
                s.train_samples = name.parse().unwrap_or(0);
            }
            ("learningRate", ParamValue::Number(x)) => s.learning_rate = *x as f32,
            ("ema", ParamValue::Number(x)) => s.ema = *x as f32,
            ("train", ParamValue::Bool(b)) => s.train = *b,
            ("scale", ParamValue::Enum(name)) => {
                s.divisor = match name.as_str() {
                    "full" => 1,
                    "half" => 2,
                    "quarter" => 4,
                    "eighth" => 8,
                    _ => 0,
                }
            }
            ("bounces", ParamValue::Int(n)) => s.bounces = (*n).clamp(0, 64) as u32,
            ("temporalBlend", ParamValue::Number(x)) => s.temporal_blend = *x as f32,
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
        let spec = &Nrc.node_types()[0];
        let params: Vec<_> = spec
            .params
            .iter()
            .map(|p| (p.key.to_string(), p.default.clone()))
            .collect();
        let settings = settings_from_params(&params)
            .validate()
            .expect("valid defaults");
        assert_eq!(
            NrcSettings {
                enabled: true,
                ..NrcSettings::default()
            },
            settings
        );
    }

    #[test]
    fn enum_params_select_resolution_and_samples() {
        let params = vec![
            ("scale".to_string(), ParamValue::Enum("half".into())),
            ("trainSamples".to_string(), ParamValue::Enum("65536".into())),
            ("bounces".to_string(), ParamValue::Int(0)),
        ];
        let settings = settings_from_params(&params).validate().expect("valid");
        assert_eq!(
            (settings.divisor, settings.train_samples, settings.bounces),
            (2, 65_536, 0)
        );
    }
}
