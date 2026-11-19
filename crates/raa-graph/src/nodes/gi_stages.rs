//! The eight screen-space GI stage node types (contract `rg-gi-stage-types`).
use crate::catalog::{
    Category, GraphFormat as F, NodeFamily, NodeTypeSpec, ParamBinding, ParamKind, ParamSpec,
    ParamValue, PortSpec, PortType as P, Rebuild, ShaderStage, SizeSpec, StageBudget,
};
use raa_pipeline::{TunableSpec, find_tunable};

pub fn family() -> Box<dyn NodeFamily> {
    Box::new(GiStages)
}
struct GiStages;
impl NodeFamily for GiStages {
    fn name(&self) -> &'static str {
        "gi_stages"
    }
    fn node_types(&self) -> Vec<NodeTypeSpec> {
        vec![
            stage(
                "gi.pyramid",
                "GI depth pyramid",
                vec![gbuffer("depth")],
                vec![image("pyramid", "Pyramid", F::R32Float)],
                vec![],
                budget(3, 0, 1, 0, 1),
            ),
            stage(
                "gi.probes",
                "GI screen probes",
                vec![
                    gbuffer("depth"),
                    gbuffer("normal"),
                    required(image_in(
                        "pyramid",
                        "Pyramid",
                        P::Image2d,
                        Some(F::R32Float),
                        &[],
                        true,
                    )),
                    image_in(
                        "volumeAtlas",
                        "Volume atlas",
                        P::Image3d,
                        Some(F::Rgba16Float),
                        &["pass.gi.volume.volumeAtlas"],
                        true,
                    ),
                ],
                vec![
                    image("atlas", "Atlas", F::Rgba16Float),
                    buffer("probesRaw", "Raw probes"),
                    buffer("probesFiltered", "Filtered probes"),
                ],
                params(
                    "GI screen probes",
                    &[
                        "screenTraces",
                        "probeScreenDistance",
                        "probeScreenSteps",
                        "probeScreenThickness",
                        "probeVolumeDistance",
                        "probeVolumeSteps",
                        "probeVolumeHitVoxels",
                        "screenEdgeFade",
                    ],
                ),
                budget(5, 0, 2, 2, 2),
            ),
            stage(
                "gi.restir.initial",
                "ReSTIR GI initial",
                vec![
                    gbuffer("depth"),
                    gbuffer("normal"),
                    required(image_in(
                        "pyramid",
                        "Pyramid",
                        P::Image2d,
                        Some(F::R32Float),
                        &[],
                        true,
                    )),
                ],
                vec![buffer("reservoirs", "Reservoirs")],
                params(
                    "ReSTIR GI initial",
                    &[
                        "screenTraces",
                        "restirJacobianMax",
                        "restirVisibilitySteps",
                        "restirVisibilityDistance",
                        "restirScreenDistance",
                        "restirScreenSteps",
                        "restirScreenThickness",
                        "restirVolumeDistance",
                        "restirVolumeSteps",
                        "restirVolumeHitVoxels",
                    ],
                ),
                budget(6, 0, 1, 2, 2),
            ),
            stage(
                "gi.restir.temporal",
                "ReSTIR GI temporal",
                vec![
                    gbuffer("depth"),
                    gbuffer("normal"),
                    gbuffer("motion"),
                    required(input("initial", "Initial", P::Buffer, None, &[], true)),
                ],
                vec![
                    buffer("reservoirs", "Reservoirs"),
                    image("irradiance", "Irradiance", F::Rgba16Float),
                ],
                params(
                    "ReSTIR GI temporal",
                    &[
                        "restirMCap",
                        "restirBoilingFactor",
                        "restirNormalCos",
                        "restirJacobianMax",
                        "restirVisibilitySteps",
                        "restirVisibilityDistance",
                    ],
                ),
                budget(5, 0, 1, 2, 2),
            ),
            stage(
                "gi.restir.spatial",
                "ReSTIR GI spatial",
                vec![
                    gbuffer("depth"),
                    gbuffer("normal"),
                    required(input(
                        "reservoirs",
                        "Reservoirs",
                        P::Buffer,
                        None,
                        &[],
                        true,
                    )),
                ],
                vec![
                    buffer("reservoirs", "Reservoirs"),
                    image("irradiance", "Irradiance", F::Rgba16Float),
                ],
                params(
                    "ReSTIR GI spatial",
                    &[
                        "restirSpatialSamples",
                        "restirSpatialRadius",
                        "restirNormalCos",
                        "restirJacobianMax",
                        "restirVisibilitySteps",
                        "restirVisibilityDistance",
                    ],
                ),
                budget(5, 0, 1, 2, 2),
            ),
            stage(
                "gi.accumulate",
                "GI accumulate",
                vec![
                    gbuffer("depth"),
                    gbuffer("normal"),
                    gbuffer("motion"),
                    input("probes", "Probes", P::Buffer, None, &[], true),
                    input(
                        "irradiance",
                        "Irradiance",
                        P::Image2d,
                        Some(F::Rgba16Float),
                        &[],
                        true,
                    ),
                    history("history", "History", F::Rgba16Float),
                    history("historyMeta", "History meta", F::Rgba32Uint),
                ],
                vec![
                    image("output", "Output", F::Rgba16Float),
                    image("meta", "Meta", F::Rgba32Uint),
                    image("filter", "Filter", F::Rgba16Float),
                    image("debug", "Debug", F::Rgba32Uint),
                ],
                params(
                    "GI accumulate",
                    &[
                        "historyMax",
                        "historyMin",
                        "clampGamma",
                        "accDepthTolerance",
                        "accNormalCos",
                        "accMinWeight",
                        "accChangeSlack",
                        "accChangeMin",
                        "occlusionDirections",
                        "occlusionSteps",
                        "occlusionRadius",
                    ],
                ),
                budget(6, 0, 2, 1, 2),
            ),
            stage(
                "gi.atrous",
                "GI a-trous",
                vec![
                    gbuffer("depth"),
                    gbuffer("normal"),
                    required(image_in(
                        "src",
                        "Source",
                        P::Image2d,
                        Some(F::Rgba16Float),
                        &[],
                        true,
                    )),
                    image_in("meta", "Meta", P::Image2d, Some(F::Rgba32Uint), &[], true),
                ],
                vec![image("dst", "Destination", F::Rgba16Float)],
                {
                    let mut p = params(
                        "GI a-trous",
                        &["atrousSigmaLuminance", "atrousNormalPower", "atrousPlane"],
                    );
                    p.insert(0, step_param());
                    p
                },
                budget(4, 0, 1, 0, 1),
            ),
            stage(
                "gi.resolve",
                "GI resolve",
                vec![
                    required(image_in(
                        "irradiance",
                        "Irradiance",
                        P::Image2d,
                        Some(F::Rgba16Float),
                        &[],
                        true,
                    )),
                    image_in(
                        "occlusion",
                        "Occlusion",
                        P::Image2d,
                        Some(F::Rgba16Float),
                        &[],
                        true,
                    ),
                ],
                vec![image("gi", "GI", F::Rgba16Float)],
                vec![],
                budget(2, 0, 1, 0, 1),
            ),
        ]
    }
}

fn stage(
    id: &'static str,
    label: &'static str,
    inputs: Vec<PortSpec>,
    outputs: Vec<PortSpec>,
    params: Vec<ParamSpec>,
    compute: StageBudget,
) -> NodeTypeSpec {
    NodeTypeSpec {
        id,
        version: 1,
        category: Category::Gi,
        label,
        description: label,
        pass: None,
        contributor: false,
        default_order: 42,
        multi_instance: true,
        side_effect: false,
        required_in_graph: false,
        inputs,
        outputs,
        params,
        budgets: vec![compute],
        upgrade,
        family: "gi_stages",
    }
}
fn upgrade(_: u32, _: &mut Vec<(String, ParamValue)>) -> Result<(), String> {
    Ok(())
}
fn budget(
    sampled_textures: u32,
    samplers: u32,
    storage_textures: u32,
    storage_buffers: u32,
    uniform_buffers: u32,
) -> StageBudget {
    StageBudget {
        stage: ShaderStage::Compute,
        sampled_textures,
        samplers,
        storage_textures,
        storage_buffers,
        uniform_buffers,
    }
}
fn input(
    key: &'static str,
    label: &'static str,
    ty: P,
    format: Option<F>,
    sources: &'static [&'static str],
    absent_ok: bool,
) -> PortSpec {
    PortSpec {
        key,
        label,
        ty,
        format,
        size: SizeSpec::Scale(1.0),
        mips: 1,
        layers: 1,
        required: false,
        default_source: sources,
        absent_ok,
        generic: true,
        history: false,
        slot: None,
    }
}
fn required(mut p: PortSpec) -> PortSpec {
    p.required = true;
    p.absent_ok = false;
    p
}
fn gbuffer(key: &'static str) -> PortSpec {
    let (label, ty, format, sources) = match key {
        "depth" => (
            "Depth",
            P::Depth2d,
            F::Depth32Float,
            &["pass.prepass.depth", "pass.vg.gbuffer.depth"][..],
        ),
        "normal" => (
            "Normal",
            P::Image2d,
            F::Rgba16Float,
            &["pass.prepass.normal", "pass.vg.gbuffer.normal"][..],
        ),
        _ => (
            "Motion",
            P::Image2d,
            F::Rg16Float,
            &["pass.prepass.motion", "pass.vg.gbuffer.motion"][..],
        ),
    };
    input(key, label, ty, Some(format), sources, false)
}
fn image_in(
    key: &'static str,
    label: &'static str,
    ty: P,
    format: Option<F>,
    sources: &'static [&'static str],
    absent_ok: bool,
) -> PortSpec {
    input(key, label, ty, format, sources, absent_ok)
}
fn output(key: &'static str, label: &'static str, ty: P, format: Option<F>) -> PortSpec {
    PortSpec {
        key,
        label,
        ty,
        format,
        size: SizeSpec::Scale(1.0),
        mips: 1,
        layers: 1,
        required: false,
        default_source: &[],
        absent_ok: false,
        generic: false,
        history: false,
        slot: None,
    }
}
fn image(key: &'static str, label: &'static str, format: F) -> PortSpec {
    output(key, label, P::Image2d, Some(format))
}
fn buffer(key: &'static str, label: &'static str) -> PortSpec {
    output(key, label, P::Buffer, None)
}
fn history(key: &'static str, label: &'static str, format: F) -> PortSpec {
    let mut p = input(key, label, P::Image2d, Some(format), &[], true);
    p.history = true;
    p
}
/// Reads one tunable's complete metadata from its canonical GI spec by binding key.
fn tunable(key: &'static str, label: &'static str, group: &'static str) -> ParamSpec {
    let spec: &TunableSpec = find_tunable(raa_gi::tunables(), key)
        .unwrap_or_else(|| panic!("gi_stages: missing tunable `{key}`"));
    let (kind, default) = if spec.integer {
        (ParamKind::Int, ParamValue::Int(spec.default as i64))
    } else {
        (ParamKind::Number, ParamValue::Number(spec.default))
    };
    ParamSpec {
        key,
        label,
        group,
        kind,
        default,
        min: spec.min,
        max: spec.max,
        step: None,
        options: &[],
        unit: "",
        rebuild: if spec.rebuild == "pipeline" {
            Rebuild::Pipeline
        } else {
            Rebuild::None
        },
        primary: false,
        unstated_range: spec.min.is_none() && spec.max.is_none(),
        binding: ParamBinding::Node,
        source: spec.source,
    }
}
fn param_label(key: &str) -> &'static str {
    match key {
        "probeScreenDistance" => "Probe screen distance",
        "probeScreenSteps" => "Probe screen steps",
        "probeScreenThickness" => "Probe screen thickness",
        "probeVolumeDistance" => "Probe volume distance",
        "probeVolumeSteps" => "Probe volume steps",
        "probeVolumeHitVoxels" => "Probe volume hit voxels",
        "screenEdgeFade" => "Screen edge fade",
        "restirJacobianMax" => "Restir jacobian max",
        "restirVisibilitySteps" => "Restir visibility steps",
        "restirVisibilityDistance" => "Restir visibility distance",
        "restirScreenDistance" => "Restir screen distance",
        "restirScreenSteps" => "Restir screen steps",
        "restirScreenThickness" => "Restir screen thickness",
        "restirVolumeDistance" => "Restir volume distance",
        "restirVolumeSteps" => "Restir volume steps",
        "restirVolumeHitVoxels" => "Restir volume hit voxels",
        "restirSpatialSamples" => "Spatial samples",
        "restirSpatialRadius" => "Spatial radius",
        "restirBoilingFactor" => "Boiling factor",
        "restirNormalCos" => "Restir normal cosine",
        "atrousSigmaLuminance" => "Atrous sigma luminance",
        "atrousNormalPower" => "Atrous normal power",
        "atrousPlane" => "Atrous plane",
        "accDepthTolerance" => "Acc depth tolerance",
        "accNormalCos" => "Acc normal cos",
        "accMinWeight" => "Acc min weight",
        "accChangeSlack" => "Acc change slack",
        "accChangeMin" => "Acc change min",
        "occlusionDirections" => "Occlusion directions",
        "occlusionSteps" => "Occlusion steps",
        "occlusionRadius" => "Occlusion radius",
        "screenTraces" => "Screen traces",
        "historyMax" => "History",
        "historyMin" => "History after change",
        "clampGamma" => "Clamp gamma",
        _ => key_label(key),
    }
}
fn key_label(key: &str) -> &'static str {
    match key {
        "restirMCap" => "M cap",
        _ => "GI parameter",
    }
}
fn params(group: &'static str, keys: &[&'static str]) -> Vec<ParamSpec> {
    keys.iter()
        .map(|key| match *key {
            "screenTraces" => setting_bool(
                key,
                "Screen traces",
                group,
                raa_gi::GiSettings::default().screen_traces,
                "raa-gi/src/settings.rs:350",
            ),
            "historyMax" => setting_int(
                key,
                "History",
                group,
                raa_gi::GiSettings::default().history_max,
                1.0,
                256.0,
                "raa-gi/src/settings.rs:355",
            ),
            "historyMin" => setting_int(
                key,
                "History after change",
                group,
                raa_gi::GiSettings::default().history_min,
                1.0,
                256.0,
                "raa-gi/src/settings.rs:360",
            ),
            "clampGamma" => setting_num(
                key,
                "Clamp gamma",
                group,
                raa_gi::GiSettings::default().clamp_gamma as f64,
                0.5,
                10.0,
                "raa-gi/src/settings.rs:356",
            ),
            "restirMCap" => setting_num(
                key,
                "ReSTIR M cap",
                group,
                raa_gi::GiSettings::default().restir_m_cap as f64,
                1.0,
                64.0,
                "raa-gi/src/settings.rs:358",
            ),
            other => tunable(other, param_label(other), group),
        })
        .collect()
}
fn setting_base(
    key: &'static str,
    label: &'static str,
    group: &'static str,
    kind: ParamKind,
    default: ParamValue,
    range: Option<(f64, f64)>,
    source: &'static str,
) -> ParamSpec {
    let (min, max) = range.map_or((None, None), |(min, max)| (Some(min), Some(max)));
    ParamSpec {
        key,
        label,
        group,
        kind,
        default,
        min,
        max,
        step: None,
        options: &[],
        unit: "",
        rebuild: Rebuild::None,
        primary: false,
        unstated_range: false,
        binding: ParamBinding::Node,
        source,
    }
}
fn setting_bool(
    key: &'static str,
    label: &'static str,
    group: &'static str,
    value: bool,
    source: &'static str,
) -> ParamSpec {
    setting_base(
        key,
        label,
        group,
        ParamKind::Bool,
        ParamValue::Bool(value),
        None,
        source,
    )
}
fn setting_int(
    key: &'static str,
    label: &'static str,
    group: &'static str,
    value: u32,
    min: f64,
    max: f64,
    source: &'static str,
) -> ParamSpec {
    setting_base(
        key,
        label,
        group,
        ParamKind::Int,
        ParamValue::Int(value.into()),
        Some((min, max)),
        source,
    )
}
fn setting_num(
    key: &'static str,
    label: &'static str,
    group: &'static str,
    value: f64,
    min: f64,
    max: f64,
    source: &'static str,
) -> ParamSpec {
    setting_base(
        key,
        label,
        group,
        ParamKind::Number,
        ParamValue::Number(value),
        Some((min, max)),
        source,
    )
}
fn step_param() -> ParamSpec {
    setting_base(
        "step",
        "Step",
        "GI a-trous",
        ParamKind::Int,
        ParamValue::Int(1),
        Some((1.0, 16.0)),
        "raa-gi/src/parts/screen/passes.rs:115 (`atrous_record`, at 536486a78ae2)",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Catalog;

    #[test]
    fn stage_types_have_contract_metadata() {
        let catalog = Catalog::build();
        let ids = [
            "gi.pyramid",
            "gi.probes",
            "gi.restir.initial",
            "gi.restir.temporal",
            "gi.restir.spatial",
            "gi.accumulate",
            "gi.atrous",
            "gi.resolve",
        ];
        for id in ids {
            let node = catalog.node(id).expect(id);
            assert_eq!(
                (
                    node.family,
                    node.pass,
                    node.category.clone(),
                    node.default_order,
                    node.multi_instance
                ),
                ("gi_stages", None, Category::Gi, 42, true)
            );
        }
    }

    #[test]
    fn only_accumulate_history_inputs_accept_history() {
        let catalog = Catalog::build();
        for node in catalog
            .nodes
            .iter()
            .filter(|node| node.family == "gi_stages")
        {
            for port in &node.inputs {
                assert_eq!(
                    port.history,
                    node.id == "gi.accumulate" && matches!(port.key, "history" | "historyMeta"),
                    "{}.{}",
                    node.id,
                    port.key
                );
            }
        }
    }

    #[test]
    fn atrous_step_and_tunable_default_are_pinned() {
        let catalog = Catalog::build();
        let atrous = catalog.node("gi.atrous").unwrap();
        let step = atrous
            .params
            .iter()
            .find(|param| param.key == "step")
            .unwrap();
        assert_eq!(step.default, ParamValue::Int(1));
        assert_eq!((step.min, step.max), (Some(1.0), Some(16.0)));
        let sigma = atrous
            .params
            .iter()
            .find(|param| param.key == "atrousSigmaLuminance")
            .unwrap();
        assert_eq!(sigma.binding, ParamBinding::Node);
        let spec = find_tunable(raa_gi::tunables(), sigma.key).unwrap();
        assert_eq!(sigma.default, ParamValue::Number(spec.default));
    }

    #[test]
    fn probes_params_keep_contract_order() {
        let catalog = Catalog::build();
        let probes = catalog.node("gi.probes").unwrap();
        assert_eq!(
            probes
                .params
                .iter()
                .map(|param| param.key)
                .collect::<Vec<_>>(),
            [
                "screenTraces",
                "probeScreenDistance",
                "probeScreenSteps",
                "probeScreenThickness",
                "probeVolumeDistance",
                "probeVolumeSteps",
                "probeVolumeHitVoxels",
                "screenEdgeFade"
            ]
        );
    }
}
