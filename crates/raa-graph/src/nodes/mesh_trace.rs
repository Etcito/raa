//! Live mesh BVH render stages.

use super::ao_stages::{gbuffer_in, image_out, node_param, post_stage, required_in, stage_budget};
use super::runtime_lighting::spherical_lookup_input;
use crate::catalog::{
    Category, GraphFormat, NodeFamily, NodeTypeSpec, ParamKind, ParamSpec, ParamValue, PortType,
    ShaderStage, SizeSpec,
};
use crate::mesh_indirect::{INHERIT_GI, VOLUME_PORT, VOLUME_SOURCE};
use raa_sdf_trace::mesh::{MAX_BOUNCES, MAX_LIGHTING_DEPTH, MeshTraceSettings, REFERENCE_SPP};

pub fn family() -> Box<dyn NodeFamily> {
    Box::new(MeshTrace)
}

struct MeshTrace;

impl NodeFamily for MeshTrace {
    fn name(&self) -> &'static str {
        "mesh_trace"
    }

    fn node_types(&self) -> Vec<NodeTypeSpec> {
        let default = MeshTraceSettings::default();
        let screen_inputs = || {
            let mut depth = gbuffer_in("depth");
            depth.required = true;
            let mut normal = gbuffer_in("normal");
            normal.required = true;
            vec![
                depth,
                normal,
                required_in("motion", "Motion and mirror mask", GraphFormat::Rg16Float),
                required_in(
                    "screenTrace",
                    "Screen hit UV and confidence",
                    GraphFormat::Rgba16Float,
                ),
                required_in("hiz", "Current Hi-Z pyramid", GraphFormat::R32Float),
                required_in(
                    "screenColor",
                    "Unreflected scene HDR",
                    GraphFormat::Rgba16Float,
                ),
                spherical_lookup_input(),
                volume_atlas_input(),
            ]
        };
        let reflection_bounces = || {
            int(
                "reflectionBounces",
                "Mirror bounces",
                default.reflection_bounces,
                1,
                MAX_BOUNCES,
            )
        };
        let screen_params = || {
            vec![
                node_param(
                    "confidenceThreshold",
                    "Screen confidence threshold",
                    "Mesh trace",
                    ParamKind::Number,
                    ParamValue::Number(f64::from(default.confidence_threshold)),
                    Some(0.0),
                    Some(1.0),
                    Some(0.01),
                    "",
                    "MeshTraceSettings",
                ),
                reflection_bounces(),
                node_param(
                    INHERIT_GI,
                    "Inherit unlinked GI grids",
                    "Mesh trace",
                    ParamKind::Bool,
                    ParamValue::Bool(true),
                    None,
                    None,
                    None,
                    "",
                    "Mesh reflection grid routing",
                ),
            ]
        };
        let outputs = |scale| {
            vec![
                image_out(
                    "color",
                    "Scene HDR",
                    GraphFormat::Rgba16Float,
                    SizeSpec::Scale(1.0),
                    1,
                ),
                image_out(
                    "radiance",
                    "Traced radiance",
                    GraphFormat::Rgba16Float,
                    SizeSpec::Scale(scale),
                    1,
                ),
            ]
        };
        let mesh = |id, label, inputs, params, scale, fitted: bool| {
            post_stage(
                id,
                self.name(),
                Category::Post,
                label,
                66,
                inputs,
                outputs(scale),
                params,
                vec![
                    stage_budget(
                        ShaderStage::Compute,
                        8 + u32::from(fitted),
                        1,
                        4,
                        if fitted { 8 } else { 5 },
                        4 + u32::from(fitted),
                    ),
                    stage_budget(ShaderStage::Fragment, 6, 0, 0, 0, 1),
                ],
            )
        };
        let mut fitted_params = screen_params();
        fitted_params.push(choice(
            "lightingOutput",
            "Lighting output",
            &[
                "combined",
                "direct",
                "indirect",
                "diffuse-indirect",
                "rough-specular",
                "rough-grid",
            ],
        ));
        let mut reflection_inputs = screen_inputs();
        reflection_inputs.push(required_in(
            "color",
            "Base HDR with ordinary SSR",
            GraphFormat::Rgba16Float,
        ));
        vec![
            mesh(
                "mesh.reflection",
                "Geometry reflections",
                reflection_inputs,
                screen_params(),
                0.5,
                false,
            ),
            mesh(
                "mesh.reference",
                "Mesh path traced reference",
                vec![required_in(
                    "color",
                    "Upstream HDR to replace",
                    GraphFormat::Rgba16Float,
                )],
                vec![
                    reflection_bounces(),
                    int(
                        "pathBounces",
                        "Path bounces",
                        default.path_bounces,
                        0,
                        MAX_BOUNCES,
                    ),
                    int("samples", "Samples per pixel", REFERENCE_SPP, 1, 32),
                    int("seed", "Sample seed", 0, 0, u32::MAX),
                    int("lightingDepth", "Lighting depth", 0, 0, MAX_LIGHTING_DEPTH),
                    choice(
                        "lightingOutput",
                        "Lighting output",
                        &["combined", "direct", "indirect", "bounce"],
                    ),
                    choice(
                        "panelSampling",
                        "Panel sampling",
                        &["stable", "single", "independent"],
                    ),
                ],
                1.0,
                false,
            ),
            mesh(
                "mesh.fitted-specular",
                "Fitted specular path pilot",
                screen_inputs(),
                fitted_params,
                0.5,
                true,
            ),
        ]
    }
}

fn volume_atlas_input() -> crate::catalog::PortSpec {
    let mut port = required_in(
        VOLUME_PORT,
        "Volume irradiance atlas",
        GraphFormat::Rgba16Float,
    );
    port.ty = PortType::Image3d;
    port.default_source = VOLUME_SOURCE;
    port.required = false;
    port.absent_ok = true;
    port.generic = false;
    port
}

fn int(key: &'static str, label: &'static str, default: u32, min: u32, max: u32) -> ParamSpec {
    node_param(
        key,
        label,
        "Mesh trace",
        ParamKind::Int,
        ParamValue::Int(i64::from(default)),
        Some(f64::from(min)),
        Some(f64::from(max)),
        Some(1.0),
        "",
        "MeshTraceSettings",
    )
}

fn choice(key: &'static str, label: &'static str, options: &'static [&'static str]) -> ParamSpec {
    ParamSpec {
        options,
        ..node_param(
            key,
            label,
            "Mesh trace",
            ParamKind::Enum,
            ParamValue::Enum(options[0].into()),
            None,
            None,
            None,
            "",
            "MeshTraceSettings",
        )
    }
}
