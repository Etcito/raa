//! Live mesh BVH render stages.

use super::ao_stages::{gbuffer_in, image_out, node_param, post_stage, required_in, stage_budget};
use crate::catalog::{
    Category, GraphFormat, NodeFamily, NodeTypeSpec, ParamKind, ParamSpec, ParamValue, ShaderStage,
    SizeSpec,
};
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
                        if fitted { 6 } else { 4 },
                        4 + u32::from(fitted),
                    ),
                    stage_budget(ShaderStage::Fragment, 6, 0, 0, 0, 1),
                ],
            )
        };
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
                screen_params(),
                0.5,
                true,
            ),
        ]
    }
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
