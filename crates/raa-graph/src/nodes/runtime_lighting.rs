//! Graph controls for the existing spherical capture and forward lighting.

use raa_sdf_trace::mesh::{SPHERICAL_GI_NAME, SphericalGiSettings};

use crate::catalog::{
    Category, FeatureId, NodeFamily, NodeTypeSpec, ParamBinding, ParamKind, ParamSpec, ParamValue,
    PortSpec, PortType, Rebuild, ShaderStage, SizeSpec, StageBudget,
};
use crate::error::{Issue, Severity, Stage};
use crate::format::{GraphDoc, NodeDoc};
use crate::validate::ValidateCtx;

use crate::runtime_lighting::{ISOLATION, SPHERICAL, SPHERICAL_TYPE, spherical_get, spherical_set};

const LOOKUP: &str = "lookup";
const FORWARD: &str = "pass.forward";
const FORWARD_LOOKUP: &str = "sphericalLookup";
const DEBUG_OPTIONS: &[&str] = &["off", "boundaries", "occupancy", "age", "light-hash"];

pub fn family() -> Box<dyn NodeFamily> {
    Box::new(RuntimeLighting)
}

struct RuntimeLighting;

impl NodeFamily for RuntimeLighting {
    fn name(&self) -> &'static str {
        "runtime_lighting"
    }

    fn node_types(&self) -> Vec<NodeTypeSpec> {
        let defaults = SphericalGiSettings::default();
        let params = [
            (
                "sphereSubdivision",
                "Sphere subdivision",
                ParamKind::Int,
                Some((0.0, 6.0)),
                &[][..],
                Rebuild::Resources,
            ),
            (
                "exponentialPower",
                "Radial exponent",
                ParamKind::Number,
                Some((0.0, 2.0)),
                &[][..],
                Rebuild::Resources,
            ),
            (
                "clipStart",
                "Inner radius",
                ParamKind::Number,
                Some((0.0, f64::from(f32::MAX))),
                &[][..],
                Rebuild::Resources,
            ),
            (
                "zMax",
                "Outer radius",
                ParamKind::Number,
                Some((0.0, f64::from(f32::MAX))),
                &[][..],
                Rebuild::Resources,
            ),
            (
                "channelMask",
                "Sky, sun and local light mask",
                ParamKind::Int,
                Some((0.0, 7.0)),
                &[][..],
                Rebuild::None,
            ),
            (
                "debug",
                "Spherical diagnostic",
                ParamKind::Enum,
                None,
                DEBUG_OPTIONS,
                Rebuild::None,
            ),
        ];
        vec![NodeTypeSpec {
            id: SPHERICAL_TYPE,
            version: 1,
            category: Category::Gi,
            label: "Spherical GI",
            description: "Capture and sample a camera-centred spherical diffuse grid.",
            pass: Some(SPHERICAL_GI_NAME),
            contributor: false,
            default_order: 58,
            multi_instance: false,
            side_effect: false,
            required_in_graph: false,
            inputs: vec![],
            outputs: vec![lookup_port(LOOKUP, &[], false)],
            params: params
                .into_iter()
                .map(|(key, label, kind, range, options, rebuild)| {
                    let mut spec = setting(
                        SPHERICAL,
                        key,
                        label,
                        kind,
                        spherical_get(defaults, key).expect("spherical setting"),
                        options,
                    );
                    if let Some((min, max)) = range {
                        spec.min = Some(min);
                        spec.max = Some(max);
                    }
                    if matches!(spec.kind, ParamKind::Int) {
                        spec.step = Some(1.0);
                    }
                    if matches!(key, "clipStart" | "zMax") {
                        spec.unit = "m";
                    }
                    spec.rebuild = rebuild;
                    spec
                })
                .collect(),
            // The spherical capture kernel is the larger of the two compute entry points.
            budgets: vec![StageBudget {
                stage: ShaderStage::Compute,
                sampled_textures: 1,
                samplers: 1,
                storage_textures: 0,
                storage_buffers: 7,
                uniform_buffers: 2,
            }],
            upgrade: |from, _| Err(format!("no upgrade from version {from}")),
            family: self.name(),
        }]
    }

    fn extend_types(&self, types: &mut Vec<NodeTypeSpec>) {
        if let Some(forward) = types.iter_mut().find(|node| node.id == FORWARD) {
            forward.inputs.push(lookup_port(
                FORWARD_LOOKUP,
                &["pass.gi.spherical.lookup"],
                true,
            ));
            forward.params.push(setting(
                ISOLATION,
                "lightingOutput",
                "Lighting output",
                ParamKind::Enum,
                ParamValue::Enum("combined".into()),
                &["combined", "direct", "indirect"],
            ));
        }
        if let Some(ao) = types.iter_mut().find(|node| node.id == "pass.ao.gtao") {
            ao.params.push(setting(
                FeatureId::Gtao,
                "indirectDiffuse",
                "Attenuate indirect diffuse",
                ParamKind::Bool,
                ParamValue::Bool(false),
                &[],
            ));
        }
    }

    fn validate_graph(&self, doc: &GraphDoc, issues: &mut Vec<Issue>) {
        validate_links(doc, issues);
    }

    fn validate_node(
        &self,
        node: &NodeDoc,
        spec: &NodeTypeSpec,
        _: &ValidateCtx<'_>,
        issues: &mut Vec<Issue>,
    ) {
        if spec.id != SPHERICAL_TYPE {
            return;
        }
        let mut settings = SphericalGiSettings::default();
        for (key, value) in &node.params {
            let Some(param) = spec.params.iter().find(|param| param.key == key) else {
                return;
            };
            let Ok(value) = ParamValue::from_json(&param.kind, value) else {
                return;
            };
            if spherical_set(&mut settings, key, &value).is_err() {
                // The catalog reports individual kind, key and range errors.
                return;
            }
        }
        if let Err(error) = settings.validate() {
            issues.push(Issue {
                code: "graph.param_range",
                severity: Severity::Error,
                stage: Stage::Structure,
                node: Some(node.id.clone()),
                port: None,
                param: Some("zMax".into()),
                link: None,
                message: error.message().into(),
                fix: "Set zMax greater than clipStart and use a representable radial grid.".into(),
            });
        }
    }
}

fn setting(
    feature: FeatureId,
    key: &'static str,
    label: &'static str,
    kind: ParamKind,
    default: ParamValue,
    options: &'static [&'static str],
) -> ParamSpec {
    ParamSpec {
        key,
        label,
        group: "Lighting",
        kind,
        default,
        min: None,
        max: None,
        step: None,
        options,
        unit: "",
        rebuild: Rebuild::None,
        primary: false,
        unstated_range: false,
        binding: ParamBinding::Setting { feature, key },
        source: "runtime_lighting.rs",
    }
}

fn lookup_port(
    key: &'static str,
    default_source: &'static [&'static str],
    absent_ok: bool,
) -> PortSpec {
    PortSpec {
        key,
        label: "Spherical irradiance lookup",
        ty: PortType::Buffer,
        format: None,
        size: SizeSpec::Scale(1.0),
        mips: 1,
        layers: 1,
        required: false,
        default_source,
        absent_ok,
        generic: false,
        history: false,
        slot: None,
    }
}

/// The buffer is published directly by the spherical pass to the shared lighting bind group.
fn validate_links(doc: &GraphDoc, issues: &mut Vec<Issue>) {
    for (index, link) in doc.links.iter().enumerate() {
        let source = doc.nodes.iter().find(|node| node.id == link.from.node);
        if link.from.port != LOOKUP || !source.is_some_and(|node| node.type_id == SPHERICAL_TYPE) {
            continue;
        }
        let target = doc.nodes.iter().find(|node| node.id == link.to.node);
        if link.to.port == FORWARD_LOOKUP && target.is_some_and(|node| node.type_id == FORWARD) {
            continue;
        }
        issues.push(Issue {
            code: "graph.fixed_wiring",
            severity: Severity::Error,
            stage: Stage::Structure,
            node: Some(link.to.node.clone()),
            port: Some(link.to.port.clone()),
            param: None,
            link: Some(index as u32),
            message: "The spherical lookup buffer is private to the forward lighting bind group."
                .into(),
            fix: "Connect lookup to the forward node's sphericalLookup input.".into(),
        });
    }
}
