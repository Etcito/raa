//! Family `passes`: one node type per lit-pipeline entry (S51), with the private helpers of
//! contract rg-pass-node-helpers.

use crate::catalog::{
    Category, FeatureId, GraphFormat, NodeFamily, NodeTypeSpec, ParamBinding, ParamKind, ParamSpec,
    ParamValue, PortSpec, PortType, Rebuild, ShaderStage, SizeSpec, StageBudget,
};

/// The `passes` family, discovered by build.rs.
pub fn family() -> Box<dyn NodeFamily> {
    Box::new(PassesFamily)
}

struct PassesFamily;

impl NodeFamily for PassesFamily {
    fn name(&self) -> &'static str {
        "passes"
    }

    fn node_types(&self) -> Vec<NodeTypeSpec> {
        let mut types = geometry_lighting_types();
        types.extend(shadow_gi_types());
        types.extend(post_types());
        types
    }
}

/// The colour-chain entries with their default orders.
#[allow(dead_code)]
const CHAIN: &[(&str, u32)] = &[
    ("forward", 60),
    ("gi.history", 64),
    ("ssr", 65),
    ("fog", 68),
    ("sunshafts", 69),
    ("opaque.copy", 70),
    ("transparent.oit", 80),
    ("particles.blend", 85),
    ("taa", 100),
    ("dof", 110),
    ("bloom", 120),
    ("grading", 130),
    ("cel.resolve", 190),
    ("tonemap", 200),
    ("overlay", 220),
];

/// The `color` sources of a chain pass at `order`: every CHAIN entry of smaller order,
/// nearest first (one static table per caller).
#[allow(dead_code)]
fn chain_sources(order: u32) -> &'static [&'static str] {
    const S: [&str; 15] = [
        "pass.overlay.color",
        "pass.tonemap.color",
        "pass.cel.resolve.color",
        "pass.grading.color",
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
    const ALL: &[&str] = &S;
    let earlier = CHAIN.iter().filter(|(_, o)| *o < order).count();
    match earlier {
        0 => &[],
        n => &ALL[ALL.len() - n..],
    }
}

#[allow(dead_code)]
fn color_input(order: u32) -> PortSpec {
    PortSpec {
        required: true,
        ..input(
            "color",
            "Color",
            PortType::Image2d,
            Some(GraphFormat::Rgba16Float),
            chain_sources(order),
            false,
            None,
        )
    }
}

fn color_output() -> PortSpec {
    output(
        "color",
        "Color",
        PortType::Image2d,
        Some(GraphFormat::Rgba16Float),
        SizeSpec::Scale(1.0),
        1,
        1,
    )
}

fn gbuffer_input(key: &'static str) -> PortSpec {
    let (label, ty, format, sources): (_, _, _, &'static [&'static str]) = match key {
        "depth" => (
            "Depth",
            PortType::Depth2d,
            GraphFormat::Depth32Float,
            &["pass.prepass.depth", "pass.vg.gbuffer.depth"],
        ),
        "normal" => (
            "Normal",
            PortType::Image2d,
            GraphFormat::Rgba16Float,
            &["pass.prepass.normal", "pass.vg.gbuffer.normal"],
        ),
        "material" => (
            "Material",
            PortType::Image2d,
            GraphFormat::Rgba8Unorm,
            &["pass.prepass.material", "pass.vg.gbuffer.material"],
        ),
        _ => (
            "Motion",
            PortType::Image2d,
            GraphFormat::Rg16Float,
            &["pass.prepass.motion", "pass.vg.gbuffer.motion"],
        ),
    };
    input(key, label, ty, Some(format), sources, false, None)
}

fn input(
    key: &'static str,
    label: &'static str,
    ty: PortType,
    format: Option<GraphFormat>,
    default_source: &'static [&'static str],
    absent_ok: bool,
    slot: Option<u32>,
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
        default_source,
        absent_ok,
        generic: false,
        history: false,
        slot,
    }
}

fn output(
    key: &'static str,
    label: &'static str,
    ty: PortType,
    format: Option<GraphFormat>,
    size: SizeSpec,
    mips: u32,
    layers: u32,
) -> PortSpec {
    PortSpec {
        key,
        label,
        ty,
        format,
        size,
        mips,
        layers,
        required: false,
        default_source: &[],
        absent_ok: false,
        generic: false,
        history: false,
        slot: None,
    }
}

#[allow(clippy::too_many_arguments)]
fn setting(
    feature: FeatureId,
    key: &'static str,
    label: &'static str,
    group: &'static str,
    kind: ParamKind,
    default: ParamValue,
    min: Option<f64>,
    max: Option<f64>,
    step: Option<f64>,
    options: &'static [&'static str],
    unit: &'static str,
    rebuild: Rebuild,
    source: &'static str,
) -> ParamSpec {
    let unstated_range = min.is_none()
        && max.is_none()
        && options.is_empty()
        && matches!(kind, ParamKind::Number | ParamKind::Int);
    ParamSpec {
        key,
        label,
        group,
        kind,
        default,
        min,
        max,
        step,
        options,
        unit,
        rebuild,
        primary: false,
        unstated_range,
        binding: ParamBinding::Setting { feature, key },
        source,
    }
}

#[allow(clippy::ptr_arg)]
fn no_upgrade(from: u32, _params: &mut Vec<(String, ParamValue)>) -> Result<(), String> {
    Err(format!("no upgrade from version {from}"))
}

#[allow(clippy::too_many_arguments)]
fn pass_type(
    id: &'static str,
    category: Category,
    label: &'static str,
    description: &'static str,
    pass: &'static str,
    default_order: u32,
    inputs: Vec<PortSpec>,
    outputs: Vec<PortSpec>,
    params: Vec<ParamSpec>,
    budgets: Vec<StageBudget>,
) -> NodeTypeSpec {
    NodeTypeSpec {
        id,
        version: 1,
        category,
        label,
        description,
        pass: Some(pass),
        contributor: false,
        default_order,
        multi_instance: false,
        side_effect: false,
        required_in_graph: matches!(pass, "prepass" | "forward" | "tonemap"),
        inputs,
        outputs,
        params,
        budgets,
        upgrade: no_upgrade,
        family: "passes",
    }
}

fn budget(
    stage: ShaderStage,
    sampled_textures: u32,
    samplers: u32,
    storage_textures: u32,
    storage_buffers: u32,
    uniform_buffers: u32,
) -> StageBudget {
    StageBudget {
        stage,
        sampled_textures,
        samplers,
        storage_textures,
        storage_buffers,
        uniform_buffers,
    }
}

/// `pass_type` with `side_effect` set (contract 'Side-effect pass types').
fn side_effect(mut spec: NodeTypeSpec) -> NodeTypeSpec {
    spec.side_effect = true;
    spec
}

use GraphFormat as F;
use PortType as P;
use ShaderStage::{Compute, Fragment, Vertex};

/// Groups 0 and 1 of a forward-class draw. Vertex: group 0 (`raa-lit/src/handle.rs:207`,
/// atlas, sampler, 6 storage, frame uniform) plus group-1 0-11 (`raa-pipeline/src/lighting.rs:541`).
/// Fragment: group-1 textures 0, 3, 5, 7-10, 13-15, 17, 20, samplers 1, 6, uniforms 2, 4,
/// 11, 12, 16, 18, 19 (`raa-pipeline/src/lighting.rs:498`) plus the group-0 atlas, sampler,
/// 5 storage buffers and frame uniform.
fn forward_class() -> Vec<StageBudget> {
    vec![
        budget(Vertex, 8, 3, 0, 6, 4),
        budget(Fragment, 13, 3, 0, 5, 8),
    ]
}

/// The S44 opaque group 1 (`raa-pipeline/src/lighting.rs:531`, 10 textures, 2 samplers,
/// 7 uniforms) plus group 0 and the vg surface group 2 (`raa-vg/src/resolve.rs:435`).
fn vg_surface_draw() -> Vec<StageBudget> {
    vec![budget(Fragment, 15, 3, 0, 5, 9)]
}

fn image(key: &'static str, label: &'static str, format: GraphFormat) -> PortSpec {
    output(
        key,
        label,
        P::Image2d,
        Some(format),
        SizeSpec::Scale(1.0),
        1,
        1,
    )
}

fn fixed(key: &'static str, label: &'static str, width: u32, height: u32) -> PortSpec {
    let size = SizeSpec::Fixed { width, height };
    output(key, label, P::Image2d, Some(F::Rgba16Float), size, 1, 1)
}

fn buffer(key: &'static str, label: &'static str) -> PortSpec {
    output(key, label, P::Buffer, None, SizeSpec::Scale(1.0), 1, 1)
}

fn gbuffer_outputs() -> Vec<PortSpec> {
    vec![
        output(
            "depth",
            "Depth",
            P::Depth2d,
            Some(F::Depth32Float),
            SizeSpec::Scale(1.0),
            1,
            1,
        ),
        image("normal", "Normal", F::Rgba16Float),
        image("material", "Material", F::Rgba8Unorm),
        image("motion", "Motion", F::Rg16Float),
    ]
}

fn vg_settings() -> Vec<ParamSpec> {
    use raa_vg::settings::{
        ERROR_PIXELS_MAX, ERROR_PIXELS_MIN, MAX_VISIBLE_CLUSTERS_MAX, MAX_VISIBLE_CLUSTERS_MIN,
    };
    let d = raa_vg::VgSettings::default();
    let toggle = |key, label, value: bool, source| {
        let v = ParamValue::Bool(value);
        setting(
            FeatureId::Vg,
            key,
            label,
            "Virtual geometry",
            ParamKind::Bool,
            v,
            None,
            None,
            None,
            &[],
            "",
            Rebuild::None,
            source,
        )
    };
    vec![
        setting(
            FeatureId::Vg,
            "errorPixels",
            "Error",
            "Virtual geometry",
            ParamKind::Number,
            ParamValue::Number(f64::from(d.error_pixels)),
            Some(f64::from(ERROR_PIXELS_MIN)),
            Some(f64::from(ERROR_PIXELS_MAX)),
            None,
            &[],
            "px",
            Rebuild::None,
            "crates/raa-vg/src/settings.rs:109",
        ),
        toggle(
            "softwareRaster",
            "Software raster",
            d.software_raster,
            "crates/raa-vg/src/settings.rs:110",
        ),
        toggle(
            "occlusion",
            "Occlusion",
            d.occlusion,
            "crates/raa-vg/src/settings.rs:111",
        ),
        toggle(
            "shadows",
            "Shadows",
            d.shadows,
            "crates/raa-vg/src/settings.rs:112",
        ),
        setting(
            FeatureId::Vg,
            "maxVisibleClusters",
            "Max visible clusters",
            "Virtual geometry",
            ParamKind::Int,
            ParamValue::Int(i64::from(d.max_visible_clusters)),
            Some(f64::from(MAX_VISIBLE_CLUSTERS_MIN)),
            Some(f64::from(MAX_VISIBLE_CLUSTERS_MAX)),
            Some(1.0),
            &[],
            "",
            Rebuild::Resources,
            "crates/raa-vg/src/settings.rs:115",
        ),
        setting(
            FeatureId::Vg,
            "debugView",
            "Debug view",
            "Virtual geometry",
            ParamKind::Enum,
            ParamValue::Enum(d.debug_view.name().to_string()),
            None,
            None,
            None,
            &[
                "none",
                "clusters",
                "lod",
                "raster",
                "occlusion",
                "instances",
            ],
            "",
            Rebuild::None,
            "crates/raa-vg/src/settings.rs:116",
        ),
    ]
}

/// The forward inputs: each a fixed group-1 slot (`raa-pipeline/src/lighting.rs:541`).
fn forward_inputs() -> Vec<PortSpec> {
    vec![
        input(
            "lightTiles",
            "Light tiles",
            P::Image2dArray,
            Some(F::R32Uint),
            &["pass.light.tiles.lightTiles"],
            true,
            Some(13),
        ),
        input(
            "shadowMap",
            "Shadow map",
            P::Depth2dArray,
            Some(F::Depth32Float),
            &["pass.shadow.csm.shadowMap"],
            true,
            Some(0),
        ),
        input(
            "envSpecular",
            "Environment",
            P::ImageCube,
            Some(F::Rgba16Float),
            &["pass.sky.env.envSpecular"],
            true,
            Some(3),
        ),
        input(
            "ao",
            "AO",
            P::Image2d,
            Some(F::Rgba16Float),
            &["pass.ao.gtao.ao"],
            true,
            Some(7),
        ),
        input(
            "contact",
            "Contact shadows",
            P::Image2d,
            Some(F::R8Unorm),
            &["pass.shadow.contact.contact"],
            true,
            Some(8),
        ),
        input(
            "gi",
            "GI",
            P::Image2d,
            Some(F::Rgba16Float),
            &["pass.gi.screen.gi"],
            true,
            Some(14),
        ),
        input(
            "areaShadow",
            "Area shadow",
            P::Image2dArray,
            None,
            &["pass.gi.area-shadow.areaShadow"],
            true,
            Some(20),
        ),
        input(
            "cloudTransmittance",
            "Cloud transmittance",
            P::Image2dArray,
            Some(F::Rgba16Float),
            &["pass.sky.cloud-lighting.cloudTransmittance"],
            true,
            Some(15),
        ),
        // Opaque copy runs after forward. Transparency reads its same-frame copy;
        // forward only reads these slots when a graph explicitly supplies a source.
        input(
            "opaqueColor",
            "Opaque color",
            P::Image2d,
            Some(F::Rgba16Float),
            &[],
            true,
            Some(9),
        ),
        input(
            "opaqueDepth",
            "Opaque depth",
            P::Image2d,
            Some(F::R32Float),
            &[],
            true,
            Some(10),
        ),
    ]
}

/// `color` and the lightingOutput "layers" / "layers-signals" / "layers-interlaced" targets
/// (raa-lit `ForwardLayers::PORTS`, published only in those modes; `layerE` in
/// "layers-signals" and "layers-interlaced", `layerR` only in "layers-signals"). Layer
/// formulas: raa-shader `opaque_layers.wgsl`; the interlaced cadence:
/// `pass_forward_layers_interlaced.wgsl`.
fn forward_outputs() -> Vec<PortSpec> {
    let mut outputs = vec![color_output()];
    for (key, label) in [
        ("layerKd", "Layer Kd (DDGI diffuse factor)"),
        ("layerS0", "Layer S0 (AO-scaled ambient at AO 1, alpha sky visibility code)"),
        ("layerEm", "Layer Em (emissive, local, contributors)"),
        ("layerE", "Signal E (probe GI x w, alpha w)"),
        ("layerR", "Residual of the recomposition"),
    ] {
        outputs.push(image(key, label, F::Rgba16Float));
    }
    outputs
}

fn forward_params() -> Vec<ParamSpec> {
    vec![
        ParamSpec {
            key: "msaa",
            label: "MSAA",
            group: "Forward",
            kind: ParamKind::Int,
            default: ParamValue::Int(4),
            min: None,
            max: None,
            step: None,
            options: &["1", "4"],
            unit: "",
            rebuild: Rebuild::Engine,
            primary: false,
            unstated_range: false,
            binding: ParamBinding::Engine { option: "msaa" },
            source: "crates/raa-js/src/convert.rs:316",
        },
        setting(
            FeatureId::AreaShading,
            "mode",
            "Area light shading",
            "Area lights",
            ParamKind::Enum,
            ParamValue::Enum("ltc".to_string()),
            None,
            None,
            None,
            &["ltc", "formFactor"],
            "",
            Rebuild::None,
            "crates/raa-js/src/api/m28_lights.rs:235",
        ),
    ]
}

fn contributor(
    id: &'static str,
    category: Category,
    label: &'static str,
    pass: &'static str,
    order: u32,
    budgets: Vec<StageBudget>,
) -> NodeTypeSpec {
    let mut spec = pass_type(
        id,
        category,
        label,
        "Draws into the forward pass while it runs.",
        pass,
        order,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        budgets,
    );
    spec.contributor = true;
    spec.side_effect = true;
    spec
}

/// The geometry and lighting pass node types of S51, the output node excepted.
///
/// Handoff: `geometry_lighting_types` is the first list `PassesFamily::node_types` returns;
/// the helpers `color_input`, `color_output`, `gbuffer_input`, `input`, `output`, `setting`,
/// `pass_type` and `budget` and the `CHAIN` table (entry name to default order, from which
/// `color_input` builds the colour default sources) are called by card 'raa-graph pass
/// nodes 2/3: shadow, AO and GI nodes', which appends `shadow_gi_types()` after this list,
/// and by card 'raa-graph pass nodes 3/3: post and tonemap nodes', which appends
/// `post_types()` last.
fn geometry_lighting_types() -> Vec<NodeTypeSpec> {
    use Category::{Geometry, Lighting};
    let none = Vec::new;
    vec![
        side_effect(pass_type(
            "pass.sky.atmosphere",
            Lighting,
            "Sky atmosphere",
            "Milky Way bake, atmosphere LUTs and directional transmission.",
            "sky.atmosphere",
            5,
            none(),
            vec![
                fixed("transmittanceLut", "Transmittance LUT", 256, 64),
                fixed("multiScatteringLut", "Multiple scattering LUT", 32, 32),
                fixed("skyView", "Sky view", 256, 144),
            ],
            Vec::new(),
            // raa-sky/src/atmosphere.rs:1179 (LUT layout) and :1225 (transmission buffer).
            vec![budget(Compute, 2, 1, 1, 1, 1)],
        )),
        pass_type(
            "pass.sky.cloud-lighting",
            Lighting,
            "Cloud lighting",
            "Cloud panorama and ground transmittance.",
            "sky.cloud-lighting",
            8,
            none(),
            vec![
                fixed("cloudPanorama", "Cloud panorama", 192, 108),
                output(
                    "cloudTransmittance",
                    "Cloud transmittance",
                    P::Image2dArray,
                    Some(F::Rgba16Float),
                    SizeSpec::Fixed {
                        width: 256,
                        height: 256,
                    },
                    1,
                    2,
                ),
            ],
            Vec::new(),
            // raa-sky/src/clouds/passes.rs:138 (march_entries).
            vec![budget(Compute, 5, 2, 1, 1, 2)],
        ),
        pass_type(
            "pass.sky.env",
            Lighting,
            "Sky environment",
            "Sky environment cube for IBL.",
            "sky.env",
            10,
            none(),
            vec![output(
                "envSpecular",
                "Environment specular",
                P::ImageCube,
                Some(F::Rgba16Float),
                SizeSpec::Fixed {
                    width: 256,
                    height: 256,
                },
                6,
                1,
            )],
            Vec::new(),
            // raa-sky/src/env_pass.rs:277 (physical_layout_entries).
            vec![budget(Fragment, 5, 2, 0, 0, 1)],
        ),
        side_effect(pass_type(
            "pass.skin",
            Geometry,
            "Skinning",
            "GPU skinning of posed meshes.",
            "skin",
            15,
            none(),
            vec![buffer("skinned", "Skinned vertices")],
            Vec::new(),
            // raa-skin/src/skin.wgsl:3-7 (group 0: two read, two read-write buffers).
            vec![budget(Compute, 0, 0, 0, 4, 0)],
        )),
        pass_type(
            "pass.vg.cull",
            Geometry,
            "VG cull",
            "Virtualized-geometry culling, phase one.",
            "vg.cull",
            21,
            none(),
            vec![buffer("cullList", "Cull list")],
            vg_settings(),
            // raa-vg/src/culling.rs:442 (scene group 2) and :466 (work group 3).
            vec![budget(Compute, 1, 0, 0, 7, 1)],
        ),
        pass_type(
            "pass.vg.raster",
            Geometry,
            "VG raster",
            "Hardware and software cluster raster, occlusion pyramid, cascades.",
            "vg.raster",
            22,
            vec![
                input(
                    "cullList",
                    "Cull list",
                    P::Buffer,
                    None,
                    &["pass.vg.cull.cullList"],
                    false,
                    None,
                ),
                input(
                    "shadowMap",
                    "Shadow map",
                    P::Depth2dArray,
                    Some(F::Depth32Float),
                    &["pass.shadow.csm.shadowMap"],
                    true,
                    None,
                ),
            ],
            vec![
                buffer("visibility", "Visibility"),
                output(
                    "hzb",
                    "Occlusion pyramid",
                    P::Image2d,
                    Some(F::R32Float),
                    SizeSpec::Scale(1.0),
                    1,
                    1,
                ),
            ],
            Vec::new(),
            // raa-vg/src/raster.rs:364 (scene group) and :452 (cascade group).
            vec![
                budget(Compute, 0, 0, 0, 7, 1),
                budget(Vertex, 0, 0, 0, 6, 1),
            ],
        ),
        pass_type(
            "pass.vg.resolve",
            Geometry,
            "VG resolve",
            "Fills the vg surface targets.",
            "vg.resolve",
            23,
            vec![input(
                "visibility",
                "Visibility",
                P::Buffer,
                None,
                &["pass.vg.raster.visibility"],
                false,
                None,
            )],
            vec![
                image("surfaceIds", "Surface ids", F::Rgba32Uint),
                image("surfaceNormal", "Surface normal", F::Rgba16Float),
                image("surfaceUvMotion", "Surface uv and motion", F::Rgba32Float),
                image("surfaceAlbedo", "Surface albedo", F::Rgba8Unorm),
            ],
            Vec::new(),
            // raa-vg/src/resolve.rs:218 (uniform, 8 storage, 4 storage targets, id texture).
            vec![budget(Compute, 1, 0, 4, 8, 1)],
        ),
        pass_type(
            "pass.prepass",
            Geometry,
            "Prepass",
            "Depth, normal, material and motion.",
            "prepass",
            30,
            none(),
            gbuffer_outputs(),
            Vec::new(),
            forward_class(),
        ),
        side_effect(pass_type(
            "pass.vg.gbuffer",
            Geometry,
            "VG G-buffer",
            "Writes the G-buffer targets from the vg surface.",
            "vg.gbuffer",
            32,
            vec![
                input(
                    "surfaceIds",
                    "Surface ids",
                    P::Image2d,
                    Some(F::Rgba32Uint),
                    &["pass.vg.resolve.surfaceIds"],
                    false,
                    None,
                ),
                input(
                    "surfaceNormal",
                    "Surface normal",
                    P::Image2d,
                    Some(F::Rgba16Float),
                    &["pass.vg.resolve.surfaceNormal"],
                    false,
                    None,
                ),
                input(
                    "surfaceUvMotion",
                    "Surface uv and motion",
                    P::Image2d,
                    Some(F::Rgba32Float),
                    &["pass.vg.resolve.surfaceUvMotion"],
                    false,
                    None,
                ),
                input(
                    "surfaceAlbedo",
                    "Surface albedo",
                    P::Image2d,
                    Some(F::Rgba8Unorm),
                    &["pass.vg.resolve.surfaceAlbedo"],
                    false,
                    None,
                ),
            ],
            gbuffer_outputs(),
            Vec::new(),
            vg_surface_draw(),
        )),
        pass_type(
            "pass.light.tiles",
            Lighting,
            "Light tiles",
            "Local lights per 16x16 tile.",
            "light.tiles",
            35,
            vec![gbuffer_input("depth")],
            vec![output(
                "lightTiles",
                "Light tiles",
                P::Image2dArray,
                Some(F::R32Uint),
                SizeSpec::Scale(1.0 / 16.0),
                1,
                128,
            )],
            Vec::new(),
            // raa-lit/src/wgsl/light_tiles.wgsl:7-12 (group 2).
            vec![budget(Compute, 1, 0, 1, 1, 1)],
        ),
        side_effect(pass_type(
            "pass.particles.sim",
            Geometry,
            "Particle simulation",
            "Simulates the particle pool.",
            "particles.sim",
            50,
            none(),
            vec![buffer("particles", "Particles")],
            Vec::new(),
            // raa-particles/src/sim.rs:138 (5 storage buffers and the step uniform).
            vec![budget(Compute, 0, 0, 0, 5, 1)],
        )),
        side_effect(pass_type(
            "pass.trace.sdf",
            Geometry,
            "SDF trace",
            "Traces SDF volumes.",
            "trace.sdf",
            55,
            vec![gbuffer_input("depth")],
            vec![image("trace", "Trace", F::Rgba16Float)],
            Vec::new(),
            // raa-sdf-trace/src/pass.rs:706 (frame and volume groups).
            vec![budget(Compute, 2, 1, 1, 8, 2)],
        )),
        side_effect(pass_type(
            "pass.sky.cloud-screen",
            Lighting,
            "Cloud screen march",
            "Half-resolution screen-space cloud march.",
            "sky.cloud-screen",
            55,
            vec![gbuffer_input("depth")],
            vec![output(
                "cloudScreen",
                "Cloud screen",
                P::Image2d,
                Some(F::Rgba16Float),
                SizeSpec::Scale(0.5),
                1,
                1,
            )],
            Vec::new(),
            // raa-sky/src/clouds/passes.rs:181 (march entries plus history and depth).
            vec![budget(Compute, 7, 2, 1, 1, 2)],
        )),
        pass_type(
            "pass.forward",
            Lighting,
            "Forward",
            "Opaque lighting into the HDR chain.",
            "forward",
            60,
            forward_inputs(),
            forward_outputs(),
            forward_params(),
            forward_class(),
        ),
        contributor(
            "pass.vg.shade",
            Geometry,
            "VG shade",
            "vg.shade",
            450,
            vg_surface_draw(),
        ),
        contributor(
            "pass.fur.shells",
            Geometry,
            "Fur shells",
            "fur.shells",
            500,
            forward_class(),
        ),
        // The composite runs on the display after tone mapping (raa-sdf-trace::CompositePass).
        side_effect(pass_type(
            "pass.trace.composite",
            Category::Output,
            "Trace composite",
            "Composites the traced SDF scene on the display.",
            raa_sdf_trace::pass::COMPOSITE_NAME,
            raa_sdf_trace::pass::COMPOSITE_ORDER,
            vec![input(
                "trace",
                "Traced SDF",
                P::Image2d,
                Some(F::Rgba16Float),
                &["pass.trace.sdf.trace"],
                true,
                None,
            )],
            Vec::new(),
            Vec::new(),
            vec![
                budget(Vertex, 0, 0, 0, 0, 1),
                budget(Fragment, 2, 0, 0, 0, 1),
            ],
        )),
        contributor(
            "pass.particles.opaque",
            Geometry,
            "Opaque particles",
            "particles.opaque",
            600,
            forward_class(),
        ),
        // raa-sky/src/env_pass.rs:277 (sky group 2).
        contributor(
            "pass.sky.background",
            Lighting,
            "Sky background",
            "sky.background",
            900,
            vec![budget(Fragment, 5, 2, 0, 0, 1)],
        ),
    ]
}

/// A Number, Int, Bool or Enum Setting of `feature` with no unit and `Rebuild::None`.
#[allow(clippy::too_many_arguments)]
fn value(
    feature: FeatureId,
    key: &'static str,
    label: &'static str,
    group: &'static str,
    default: ParamValue,
    range: (Option<f64>, Option<f64>),
    options: &'static [&'static str],
    source: &'static str,
) -> ParamSpec {
    let kind = match default {
        ParamValue::Number(_) => ParamKind::Number,
        ParamValue::Int(_) => ParamKind::Int,
        ParamValue::Bool(_) => ParamKind::Bool,
        _ => ParamKind::Enum,
    };
    let (min, max) = range;
    let step = (kind == ParamKind::Int && options.is_empty()).then_some(1.0);
    setting(
        feature,
        key,
        label,
        group,
        kind,
        default,
        min,
        max,
        step,
        options,
        "",
        Rebuild::None,
        source,
    )
}

fn num(v: f32) -> ParamValue {
    ParamValue::Number(f64::from(v))
}

fn int(v: u32) -> ParamValue {
    ParamValue::Int(i64::from(v))
}

fn name(v: &str) -> ParamValue {
    ParamValue::Enum(v.to_string())
}

/// `spec` with `rebuild` set.
fn rebuilds(spec: ParamSpec, rebuild: Rebuild) -> ParamSpec {
    ParamSpec { rebuild, ..spec }
}

fn ranged(min: f64, max: f64) -> (Option<f64>, Option<f64>) {
    (Some(min), Some(max))
}

const OPEN: (Option<f64>, Option<f64>) = (None, None);
const POSITIVE: (Option<f64>, Option<f64>) = (Some(0.0), None);

/// The shadow, AO and GI pass node types of S51; `PassesFamily::node_types` appends them after
/// `geometry_lighting_types` (helpers per its Handoff doc comment, passes.rs:606).
fn shadow_gi_types() -> Vec<NodeTypeSpec> {
    use Category::{Gi, Lighting, Shadows};
    use FeatureId as Fid;
    let shadow = raa_shadow::ShadowDesc::default();
    let ao = raa_gtao::AoDesc::default();
    let contact = raa_contact::ContactDesc::default();
    let gi = raa_gi::GiSettings::default();
    let q = raa_gi::GiQuality::default();
    let area = raa_lit::AreaShadowSettings::default();
    let st = raa_lit::StochasticShadowSettings::standard();
    let g = |key, label, default, range, options, line| {
        value(Fid::Gi, key, label, "GI", default, range, options, line)
    };
    let gq = |key, label, v: u32, options, line| {
        let spec = value(
            Fid::GiQuality,
            key,
            label,
            "GI quality",
            int(v),
            OPEN,
            options,
            line,
        );
        rebuilds(spec, Rebuild::Resources)
    };
    let a = |key, label, default, range, options, line| {
        value(
            Fid::AreaShadows,
            key,
            label,
            "Area shadows",
            default,
            range,
            options,
            line,
        )
    };
    let image3d = |key, label| {
        output(
            key,
            label,
            P::Image3d,
            Some(F::Rgba16Float),
            SizeSpec::Scale(1.0),
            1,
            1,
        )
    };
    let array = |key, label, format, layers| {
        output(
            key,
            label,
            P::Image2dArray,
            Some(format),
            SizeSpec::Scale(1.0),
            1,
            layers,
        )
    };
    vec![
        pass_type(
            "pass.shadow.csm",
            Shadows,
            "Cascaded shadows",
            "Four sun shadow cascades.",
            "shadow.csm",
            20,
            Vec::new(),
            vec![output(
                "shadowMap",
                "Shadow map",
                P::Depth2dArray,
                Some(F::Depth32Float),
                SizeSpec::FromParam("resolution"),
                1,
                4,
            )],
            vec![
                value(
                    Fid::Shadow,
                    "softness",
                    "Softness",
                    "Shadows",
                    num(shadow.softness),
                    ranged(0.0, 16.0),
                    &[],
                    "raa-shadow/src/desc.rs:19",
                ),
                value(
                    Fid::Shadow,
                    "distance",
                    "Distance",
                    "Shadows",
                    num(shadow.distance_m),
                    POSITIVE,
                    &[],
                    "raa-shadow/src/desc.rs:20",
                ),
                value(
                    Fid::Shadow,
                    "cover",
                    "Cover scene",
                    "Shadows",
                    ParamValue::Bool(shadow.cover),
                    OPEN,
                    &[],
                    "raa-shadow/src/desc.rs:22",
                ),
                rebuilds(
                    value(
                        Fid::Shadow,
                        "resolution",
                        "Resolution",
                        "Shadows",
                        int(shadow.resolution),
                        OPEN,
                        &["1024", "2048", "4096"],
                        "raa-shadow/src/desc.rs:21",
                    ),
                    Rebuild::Resources,
                ),
                value(
                    Fid::Shadow,
                    "cascades",
                    "Cascades",
                    "Shadows",
                    int(shadow.cascades),
                    OPEN,
                    &["1", "2", "3", "4"],
                    "raa-shadow/src/desc.rs:22",
                ),
                value(
                    Fid::Shadow,
                    "filter",
                    "Filter radius (m)",
                    "Shadows",
                    num(shadow.filter_m),
                    ranged(0.0, 0.5),
                    &[],
                    "raa-shadow/src/desc.rs:26",
                ),
                value(
                    Fid::Shadow,
                    "split",
                    "Split distribution",
                    "Shadows",
                    num(shadow.split_lambda),
                    ranged(0.0, 1.0),
                    &[],
                    "raa-shadow/src/desc.rs:30",
                ),
                value(
                    Fid::Shadow,
                    "spread",
                    "Spread over scene",
                    "Shadows",
                    ParamValue::Bool(shadow.spread),
                    OPEN,
                    &[],
                    "raa-shadow/src/desc.rs:34",
                ),
            ],
            // raa-shadow/src/caster.rs:177 (group 2 uniform) over the lit group 0 vertex
            // bindings (raa-lit/src/handle.rs:207: atlas, sampler, 6 storage + palette, frame).
            vec![budget(Vertex, 1, 1, 0, 7, 2)],
        ),
        side_effect(pass_type(
            "pass.gi.volume",
            Gi,
            "GI volume",
            "Voxel clipmap volume of the global illumination.",
            "gi.volume",
            25,
            vec![input(
                "surfaceIds",
                "Surface ids",
                P::Image2d,
                Some(F::Rgba32Uint),
                &["pass.vg.resolve.surfaceIds"],
                true,
                None,
            )],
            vec![image3d("volumeAtlas", "Volume atlas")],
            vec![
                gq(
                    "resolution",
                    "Resolution",
                    q.resolution,
                    &["64", "96"],
                    "raa-gi/src/settings.rs:525",
                ),
                gq(
                    "levels",
                    "Levels",
                    q.levels,
                    &["3", "4", "5"],
                    "raa-gi/src/settings.rs:526",
                ),
                gq(
                    "probeRays",
                    "Probe rays",
                    q.probe_rays,
                    &["6", "8"],
                    "raa-gi/src/settings.rs:527",
                ),
                g(
                    "intensity",
                    "Intensity",
                    num(gi.intensity),
                    ranged(0.0, 16.0),
                    &[],
                    "raa-gi/src/settings.rs:346",
                ),
                g(
                    "environmentDiffuse",
                    "Environment diffuse",
                    num(gi.environment_diffuse),
                    ranged(0.0, 1.0),
                    &[],
                    "raa-gi/src/settings.rs:347",
                ),
                g(
                    "localLightShadows",
                    "Local light shadows",
                    ParamValue::Bool(gi.local_light_shadows),
                    OPEN,
                    &[],
                    "raa-gi/src/settings.rs:348",
                ),
                g(
                    "voxelFullRelight",
                    "Full voxel relight",
                    ParamValue::Bool(gi.voxel_full_relight),
                    OPEN,
                    &[],
                    "raa-gi/src/settings.rs:359",
                ),
                g(
                    "voxelHistoryMax",
                    "Voxel history",
                    int(gi.voxel_history_max),
                    ranged(1.0, 255.0),
                    &[],
                    "raa-gi/src/settings.rs:361",
                ),
                g(
                    "voxelRays",
                    "Voxel rays",
                    int(gi.voxel_rays),
                    ranged(1.0, 15.0),
                    &[],
                    "raa-gi/src/settings.rs:362",
                ),
            ],
            // raa-gi/src/wgsl/volume_write.wgsl:51-56 and voxel_light.wgsl:44-49 (group 2).
            vec![budget(Compute, 0, 0, 1, 3, 2)],
        )),
        pass_type(
            "pass.ao.gtao",
            Lighting,
            "GTAO",
            "Horizon-based ambient occlusion.",
            "ao.gtao",
            40,
            vec![gbuffer_input("depth"), gbuffer_input("normal")],
            vec![
                image("ao", "AO", F::Rgba16Float),
                image("raw", "Raw AO", F::R32Float),
                image("blur", "Blurred AO", F::R32Float),
            ],
            vec![
                value(
                    Fid::Gtao,
                    "radius",
                    "Radius",
                    "AO",
                    num(ao.radius_m),
                    ranged(0.0, 10.0),
                    &[],
                    "raa-gtao/src/desc.rs:25",
                ),
                value(
                    Fid::Gtao,
                    "intensity",
                    "Intensity",
                    "AO",
                    num(ao.intensity),
                    ranged(0.0, 1.0),
                    &[],
                    "raa-gtao/src/desc.rs:26",
                ),
                value(
                    Fid::Gtao,
                    "power",
                    "Power",
                    "AO",
                    num(ao.power),
                    ranged(0.0, 8.0),
                    &[],
                    "raa-gtao/src/desc.rs:27",
                ),
            ],
            // raa-gtao/src/wgsl/gtao.wgsl:27-30 and denoise.wgsl:17-21 (group 0).
            vec![budget(Compute, 2, 0, 2, 0, 1)],
        ),
        pass_type(
            "pass.gi.screen",
            Gi,
            "GI screen",
            "Screen probes, ReSTIR GI and the denoised screen irradiance.",
            "gi.screen",
            42,
            vec![
                gbuffer_input("depth"),
                gbuffer_input("normal"),
                gbuffer_input("material"),
                gbuffer_input("motion"),
                input(
                    "volumeAtlas",
                    "Volume atlas",
                    P::Image3d,
                    Some(F::Rgba16Float),
                    &["pass.gi.volume.volumeAtlas"],
                    true,
                    None,
                ),
            ],
            vec![
                image("gi", "GI", F::Rgba16Float),
                image("pyramid", "Depth pyramid", F::R32Float),
                image("atlas", "Probe atlas", F::Rgba16Float),
                buffer("probesRaw", "Raw probes"),
                buffer("probesFiltered", "Filtered probes"),
                image("output", "Accumulated", F::Rgba16Float),
                image("meta", "Moments", F::Rgba32Uint),
                image("filter", "A-trous", F::Rgba16Float),
                image("display", "Denoised", F::Rgba16Float),
                image("debug", "Debug record", F::Rgba32Uint),
            ],
            vec![
                g(
                    "indirect",
                    "Indirect",
                    name(gi.indirect.as_str()),
                    OPEN,
                    &["probes", "restir"],
                    "raa-gi/src/settings.rs:354",
                ),
                g(
                    "screenTraces",
                    "Screen traces",
                    ParamValue::Bool(gi.screen_traces),
                    OPEN,
                    &[],
                    "raa-gi/src/settings.rs:350",
                ),
                g(
                    "reflections",
                    "Reflections",
                    ParamValue::Bool(gi.reflections),
                    OPEN,
                    &[],
                    "raa-gi/src/settings.rs:349",
                ),
                g(
                    "historyMax",
                    "History",
                    int(gi.history_max),
                    ranged(1.0, 256.0),
                    &[],
                    "raa-gi/src/settings.rs:355",
                ),
                g(
                    "historyMin",
                    "History after change",
                    int(gi.history_min),
                    ranged(1.0, 256.0),
                    &[],
                    "raa-gi/src/settings.rs:360",
                ),
                g(
                    "clampGamma",
                    "Clamp gamma",
                    num(gi.clamp_gamma),
                    ranged(0.5, 10.0),
                    &[],
                    "raa-gi/src/settings.rs:356",
                ),
                g(
                    "spatialPasses",
                    "Spatial passes",
                    int(gi.spatial_passes),
                    ranged(0.0, 5.0),
                    &[],
                    "raa-gi/src/settings.rs:357",
                ),
                g(
                    "restirMCap",
                    "ReSTIR M cap",
                    num(gi.restir_m_cap),
                    ranged(1.0, 64.0),
                    &[],
                    "raa-gi/src/settings.rs:358",
                ),
                g(
                    "debugView",
                    "Debug view",
                    name(gi.debug_view.as_str()),
                    OPEN,
                    &[
                        "none",
                        "voxels",
                        "radiance",
                        "distance",
                        "levels",
                        "probes",
                        "hitType",
                        "history",
                        "historyLength",
                        "reservoirs",
                        "rawDenoised",
                    ],
                    "raa-gi/src/settings.rs:351",
                ),
            ],
            // raa-gi/src/parts/screen/restir_gi.rs:83,114 and screen/passes.rs:310 (group 2,
            // per-field maximum over the entries).
            vec![budget(Compute, 6, 0, 4, 2, 3)],
        ),
        pass_type(
            "pass.gi.area-shadow",
            Shadows,
            "Area shadows",
            "Stochastic and ReSTIR soft shadows of area lights.",
            "gi.area-shadow",
            43,
            vec![
                gbuffer_input("depth"),
                gbuffer_input("normal"),
                gbuffer_input("motion"),
            ],
            vec![
                array("areaShadow", "Area shadow", F::Rgba16Float, 2),
                array("raw", "Raw", F::Rgba16Float, 2),
                image("depthHistory", "Depth history", F::R32Float),
                array("mask", "Mask", F::Rgba16Float, 2),
                image("restirAtlas", "ReSTIR atlas", F::Rgba16Float),
            ],
            vec![
                a(
                    "tier",
                    "Tier",
                    name(area.tier.as_str()),
                    OPEN,
                    &["auto", "hero", "cone", "fade"],
                    "raa-lit/src/area_shadow.rs:88",
                ),
                a(
                    "heroCount",
                    "Hero count",
                    int(area.hero_count),
                    ranged(0.0, 4.0),
                    &[],
                    "raa-lit/src/area_shadow.rs:89",
                ),
                a(
                    "heroDistance",
                    "Hero distance",
                    num(area.hero_distance),
                    POSITIVE,
                    &[],
                    "raa-lit/src/area_shadow.rs:90",
                ),
                // `> heroDistance`: no fixed bound.
                a(
                    "fadeDistance",
                    "Fade distance",
                    num(area.fade_distance),
                    OPEN,
                    &[],
                    "raa-lit/src/area_shadow.rs:91",
                ),
                a(
                    "band",
                    "Band",
                    num(area.band),
                    POSITIVE,
                    &[],
                    "raa-lit/src/area_shadow.rs:92",
                ),
                a(
                    "stochasticCount",
                    "Stochastic count",
                    int(st.count),
                    ranged(0.0, 8.0),
                    &[],
                    "raa-lit/src/area_stochastic.rs:100",
                ),
                a(
                    "stochasticDistance",
                    "Stochastic distance",
                    num(st.distance),
                    POSITIVE,
                    &[],
                    "raa-lit/src/area_stochastic.rs:101",
                ),
                a(
                    "stochasticCells",
                    "Stochastic cells",
                    int(st.cells),
                    OPEN,
                    &["4", "9"],
                    "raa-lit/src/area_stochastic.rs:102",
                ),
                a(
                    "stochasticMode",
                    "Stochastic mode",
                    name(st.mode.as_str()),
                    OPEN,
                    &["stratified", "restir"],
                    "raa-lit/src/area_stochastic.rs:103",
                ),
            ],
            // raa-gi/src/parts/area_shadow/passes.rs:14,55 (per-field maximum).
            vec![budget(Compute, 5, 1, 2, 0, 4)],
        ),
        side_effect(pass_type(
            "pass.gi.ssgi",
            Gi,
            "SSGI",
            "Screen-space indirect diffuse with a visibility bitmask.",
            "gi.ssgi",
            44,
            vec![
                gbuffer_input("depth"),
                gbuffer_input("normal"),
                gbuffer_input("motion"),
            ],
            vec![
                image("ssgi", "SSGI", F::Rgba16Float),
                image("raw", "Raw", F::Rgba16Float),
                image("history", "History", F::Rgba16Float),
                image("output", "Output", F::Rgba16Float),
            ],
            vec![g(
                "screenSpaceQuality",
                "Quality",
                name(gi.screen_space_quality.as_str()),
                OPEN,
                &["low", "medium", "high"],
                "raa-gi/src/settings.rs:353",
            )],
            // raa-gi/src/parts/ssgi/passes.rs:59,84 (per-field maximum).
            vec![budget(Compute, 5, 0, 2, 0, 2)],
        )),
        pass_type(
            "pass.shadow.contact",
            Shadows,
            "Contact shadows",
            "Screen-space contact shadows.",
            "shadow.contact",
            45,
            vec![gbuffer_input("depth")],
            vec![
                image("contact", "Contact", F::Rgba16Float),
                image("output", "Output", F::Rgba16Float),
            ],
            vec![
                value(
                    Fid::Contact,
                    "distance",
                    "Distance",
                    "Contact shadows",
                    num(contact.distance_m),
                    ranged(0.0, 5.0),
                    &[],
                    "raa-contact/src/desc.rs:20",
                ),
                value(
                    Fid::Contact,
                    "thickness",
                    "Thickness",
                    "Contact shadows",
                    num(contact.thickness_m),
                    ranged(0.0, 1.0),
                    &[],
                    "raa-contact/src/desc.rs:21",
                ),
                value(
                    Fid::Contact,
                    "intensity",
                    "Intensity",
                    "Contact shadows",
                    num(contact.intensity),
                    ranged(0.0, 1.0),
                    &[],
                    "raa-contact/src/desc.rs:22",
                ),
            ],
            // raa-contact/src/wgsl/contact.wgsl:19-21 (group 0).
            vec![budget(Compute, 1, 0, 1, 0, 1)],
        ),
        pass_type(
            "pass.gi.history",
            Gi,
            "GI history",
            "Copy of the lit frame for the next frame's screen traces.",
            "gi.history",
            64,
            vec![color_input(64)],
            vec![color_output(), image("history", "History", F::Rgba16Float)],
            Vec::new(),
            // raa-gi/src/feature.rs:118: a texture copy, no bind group.
            Vec::new(),
        ),
    ]
}

/// A Vec3 Setting in linear RGB (unit `linear-rgb`); `min` bounds each component.
fn colour(
    feature: FeatureId,
    group: &'static str,
    key: &'static str,
    label: &'static str,
    default: [f32; 3],
    min: Option<f64>,
    source: &'static str,
) -> ParamSpec {
    setting(
        feature,
        key,
        label,
        group,
        ParamKind::Vec3,
        ParamValue::Vec3(default.map(f64::from)),
        min,
        None,
        None,
        &[],
        "linear-rgb",
        Rebuild::None,
        source,
    )
}

/// `spec` with its unit set.
fn with_unit(spec: ParamSpec, unit: &'static str) -> ParamSpec {
    ParamSpec { unit, ..spec }
}

/// The post, tonemap and output pass node types of S51, appended last by
/// `PassesFamily::node_types`. The colour chain and the helpers follow the Handoff doc comment
/// on `geometry_lighting_types` (passes.rs:614). A budget counts the declared layout entries
/// visible to its stage, with the lit groups where the pass's pipeline layout holds them.
fn post_types() -> Vec<NodeTypeSpec> {
    use Category::{Output, Post};
    use FeatureId as Fid;
    use raa_tonemap::{Look, ToneMapper};
    /// Upper bound of a surface-sized mip chain: the levels of an 8192 px side, the WebGPU
    /// default maximum dimension (`raa_ssr::hiz::mip_count` gives 14 there).
    const SURFACE_MIPS: u32 = 14;
    let ssr = raa_ssr::SsrDesc::default();
    let fog = raa_fog::FogDesc::default();
    let shafts = raa_shafts::ShaftsDesc::default();
    let oit = raa_oit::OitSettings::default();
    let dof = raa_dof::DofDesc::default();
    let bloom = raa_bloom::BloomDesc::default();
    let grading = raa_grading::GradingDesc::default();
    let cel = raa_cel::CelSettings::default();
    let tone = raa_tonemap::TonemapControl::default().get();
    let ssr_p = |key, label, default, range, source| {
        value(
            Fid::Ssr,
            key,
            label,
            "Reflections",
            default,
            range,
            &[],
            source,
        )
    };
    let fog_p = |key, label, default, range, source| {
        value(Fid::Fog, key, label, "Fog", default, range, &[], source)
    };
    let shafts_p = |key, label, default, range, source| {
        value(
            Fid::Shafts,
            key,
            label,
            "Sun shafts",
            default,
            range,
            &[],
            source,
        )
    };
    let dof_p = |key, label, default, range, source| {
        value(
            Fid::Dof,
            key,
            label,
            "Depth of field",
            default,
            range,
            &[],
            source,
        )
    };
    let bloom_p = |key, label, default, range, source| {
        value(Fid::Bloom, key, label, "Bloom", default, range, &[], source)
    };
    let grading_p = |key, label, default, range, source| {
        value(
            Fid::Grading,
            key,
            label,
            "Colour grading",
            default,
            range,
            &[],
            source,
        )
    };
    let cel_p = |key, label, default, range, options, source| {
        value(
            Fid::Cel,
            key,
            label,
            "Cel shading",
            default,
            range,
            options,
            source,
        )
    };
    let tone_p = |key, label, default, range, options, source| {
        value(
            Fid::Tonemap,
            key,
            label,
            "Tone mapping",
            default,
            range,
            options,
            source,
        )
    };
    let mapper = match tone.mapper {
        ToneMapper::Agx => "agx",
        ToneMapper::Aces => "aces",
        ToneMapper::Linear => "linear",
    };
    let look = match tone.look {
        Look::None => "none",
        Look::Punchy => "punchy",
        Look::Golden => "golden",
    };
    vec![
        pass_type(
            "pass.ssr",
            Post,
            "Screen-space reflections",
            "Screen-space reflections composited over the colour chain; also publishes the Hi-Z pyramid and the trace.",
            "ssr",
            65,
            vec![
                color_input(65),
                gbuffer_input("depth"),
                gbuffer_input("normal"),
                gbuffer_input("material"),
            ],
            vec![
                color_output(),
                output(
                    "hiz",
                    "Hi-Z pyramid",
                    P::Image2d,
                    Some(F::R32Float),
                    SizeSpec::Scale(1.0),
                    SURFACE_MIPS,
                    1,
                ),
                output(
                    "trace",
                    "Trace",
                    P::Image2d,
                    Some(F::Rgba16Float),
                    SizeSpec::Scale(1.0),
                    1,
                    1,
                ),
            ],
            vec![
                ssr_p(
                    "intensity",
                    "Intensity",
                    num(ssr.intensity),
                    ranged(0.0, 2.0),
                    "crates/raa-ssr/src/desc.rs:19",
                ),
                ssr_p(
                    "maxRoughness",
                    "Max roughness",
                    num(ssr.max_roughness),
                    ranged(0.0, 1.0),
                    "crates/raa-ssr/src/desc.rs:20",
                ),
                with_unit(
                    ssr_p(
                        "thickness",
                        "Thickness",
                        num(ssr.thickness_m),
                        ranged(0.0, 10.0),
                        "crates/raa-ssr/src/desc.rs:21",
                    ),
                    "m",
                ),
            ],
            vec![
                // Hi-Z copy and reduce: depth or unfilterable texture, storage, params uniform
                // (raa-ssr/src/resources.rs:259-276).
                budget(Compute, 1, 0, 1, 0, 1),
                // VS_FULLSCREEN: the opaque group 1 entries visible to the vertex stage.
                budget(Vertex, 5, 2, 0, 0, 3),
                // fs_composite: composite_layout (resources.rs:295-307: 5 textures, 1 sampler,
                // 1 uniform) with the opaque group 1 (`LightingResources::opaque_layout`,
                // raa-pipeline/src/lighting.rs:952). The 16 textures are as resources.rs:283-284
                // states; the opaque layout's 10 textures would give 15, for the owner to confirm.
                budget(Fragment, 16, 3, 0, 0, 8),
            ],
        ),
        pass_type(
            "pass.fog",
            Post,
            "Height fog",
            "Exponential height fog and aerial perspective over the colour chain.",
            "fog",
            68,
            vec![color_input(68), gbuffer_input("depth")],
            vec![color_output()],
            vec![
                fog_p(
                    "density",
                    "Density",
                    num(fog.density),
                    ranged(0.0, 1.0),
                    "crates/raa-fog/src/desc.rs:32",
                ),
                with_unit(
                    fog_p(
                        "heightFalloff",
                        "Height falloff",
                        num(fog.height_falloff),
                        ranged(0.0, 10.0),
                        "crates/raa-fog/src/desc.rs:33",
                    ),
                    "1/m",
                ),
                with_unit(
                    fog_p(
                        "baseHeight",
                        "Base height",
                        num(fog.base_height_m),
                        OPEN,
                        "crates/raa-fog/src/desc.rs:34",
                    ),
                    "m",
                ),
                with_unit(
                    fog_p(
                        "start",
                        "Start",
                        num(fog.start_m),
                        (Some(0.0), None),
                        "crates/raa-fog/src/desc.rs:35",
                    ),
                    "m",
                ),
                colour(
                    Fid::Fog,
                    "Fog",
                    "color",
                    "Colour",
                    fog.color,
                    Some(0.0),
                    "crates/raa-fog/src/desc.rs:36",
                ),
                fog_p(
                    "inscatter",
                    "Inscatter",
                    num(fog.inscatter),
                    ranged(0.0, 4.0),
                    "crates/raa-fog/src/desc.rs:37",
                ),
                fog_p(
                    "aerialStrength",
                    "Aerial strength",
                    num(fog.aerial_strength),
                    ranged(0.0, 2.0),
                    "crates/raa-fog/src/desc.rs:38",
                ),
                with_unit(
                    fog_p(
                        "aerialDistance",
                        "Aerial distance",
                        num(fog.aerial_distance_m),
                        POSITIVE,
                        "crates/raa-fog/src/desc.rs:39",
                    ),
                    "m",
                ),
            ],
            vec![
                // Group 0 frame (LitHandle, as forward), the base group 1 (`base_layout`,
                // lighting.rs:933) and the private depth, colour and sampler (feature.rs:113).
                budget(Vertex, 8, 3, 0, 6, 4),
                budget(Fragment, 12, 4, 0, 5, 6),
            ],
        ),
        pass_type(
            "pass.sunshafts",
            Post,
            "Sun shafts",
            "Light shafts marched towards the sun over the colour chain.",
            "sunshafts",
            69,
            vec![color_input(69), gbuffer_input("depth")],
            vec![color_output()],
            vec![
                shafts_p(
                    "intensity",
                    "Intensity",
                    num(shafts.intensity),
                    ranged(0.0, 8.0),
                    "crates/raa-shafts/src/desc.rs:14",
                ),
                shafts_p(
                    "density",
                    "Density",
                    num(shafts.density),
                    ranged(0.0, 1.0),
                    "crates/raa-shafts/src/desc.rs:15",
                ),
                with_unit(
                    shafts_p(
                        "maxDistance",
                        "Max distance",
                        num(shafts.max_distance_m),
                        ranged(0.0, 1000.0),
                        "crates/raa-shafts/src/desc.rs:16",
                    ),
                    "m",
                ),
            ],
            vec![
                // Group 1 is the opaque layout (lighting.rs:952); the private group 0 (depth,
                // two colour textures, sampler, uniform) is fragment-only (feature.rs:148).
                budget(Vertex, 5, 2, 0, 0, 3),
                budget(Fragment, 13, 3, 0, 0, 8),
            ],
        ),
        pass_type(
            "pass.opaque.copy",
            Post,
            "Opaque copy",
            "Copies the opaque colour with its mips and the opaque depth for transmission.",
            "opaque.copy",
            70,
            vec![color_input(70), gbuffer_input("depth")],
            vec![
                color_output(),
                output(
                    "opaqueColor",
                    "Opaque colour",
                    P::Image2d,
                    Some(F::Rgba16Float),
                    SizeSpec::Scale(1.0),
                    SURFACE_MIPS,
                    1,
                ),
                output(
                    "opaqueDepth",
                    "Opaque depth",
                    P::Image2d,
                    Some(F::R32Float),
                    SizeSpec::Scale(1.0),
                    1,
                    1,
                ),
            ],
            Vec::new(),
            // The down pass (filterable texture, sampler) and the depth pass (texture), each with
            // its own layout (raa-transmission/src/copy.rs:178 and :191).
            vec![budget(Fragment, 1, 1, 0, 0, 0)],
        ),
        pass_type(
            "pass.transparent.oit",
            Post,
            "Transparency (OIT)",
            "Order-independent transparency: the linked-list fragments resolve over the colour chain.",
            "transparent.oit",
            80,
            vec![
                color_input(80),
                gbuffer_input("depth"),
                // Optional: without the copy the dummy binds (absent_ok).
                input(
                    "opaqueColor",
                    "Opaque colour",
                    P::Image2d,
                    Some(F::Rgba16Float),
                    &["pass.opaque.copy.opaqueColor"],
                    true,
                    None,
                ),
                input(
                    "opaqueDepth",
                    "Opaque depth",
                    P::Image2d,
                    Some(F::R32Float),
                    &["pass.opaque.copy.opaqueDepth"],
                    true,
                    None,
                ),
            ],
            vec![color_output()],
            vec![
                rebuilds(
                    value(
                        Fid::Oit,
                        "maxFragments",
                        "Max fragments",
                        "Transparency",
                        int(oit.max_fragments),
                        ranged(4.0, 32.0),
                        &[],
                        "crates/raa-oit/src/feature.rs:54",
                    ),
                    Rebuild::Resources,
                ),
                rebuilds(
                    value(
                        Fid::Oit,
                        "poolFactor",
                        "Pool factor",
                        "Transparency",
                        int(oit.pool_factor),
                        ranged(2.0, 32.0),
                        &[],
                        "crates/raa-oit/src/feature.rs:55",
                    ),
                    Rebuild::Resources,
                ),
            ],
            vec![
                // The clear (compute: 2 storage buffers, 1 uniform) and resolve (fragment: 1
                // texture, 2 storage buffers, 1 uniform) layouts (raa-oit/src/buffers.rs:276, :285).
                budget(Compute, 0, 0, 0, 2, 1),
                budget(Fragment, 1, 0, 0, 2, 1),
            ],
        ),
        pass_type(
            "pass.particles.blend",
            Post,
            "Particles (blend)",
            "Blended particles drawn over the colour chain.",
            "particles.blend",
            85,
            vec![
                color_input(85),
                gbuffer_input("depth"),
                input(
                    "particles",
                    "Particles",
                    P::Buffer,
                    None,
                    &["pass.particles.sim.particles"],
                    false,
                    None,
                ),
            ],
            vec![color_output()],
            Vec::new(),
            // The frame group 0 (LitHandle), the full group 1 (`LightingResources::layout`,
            // lighting.rs:922) and the group 2 particle storage (vertex) and depth (fragment)
            // (raa-particles/src/blend.rs:96, :192).
            vec![
                budget(Vertex, 8, 3, 0, 8, 4),
                budget(Fragment, 14, 3, 0, 5, 8),
            ],
        ),
        pass_type(
            "pass.taa",
            Post,
            "TAA",
            "Temporal anti-aliasing over the colour chain, with its history output.",
            "taa",
            100,
            vec![
                color_input(100),
                gbuffer_input("depth"),
                gbuffer_input("motion"),
            ],
            vec![
                color_output(),
                output(
                    "history",
                    "History",
                    P::Image2d,
                    Some(F::Rgba16Float),
                    SizeSpec::Scale(1.0),
                    1,
                    1,
                ),
            ],
            Vec::new(),
            // The resolve compute group 0 (raa-taa/src/resources.rs:84-96).
            vec![budget(Compute, 4, 1, 2, 0, 2)],
        ),
        pass_type(
            "pass.dof",
            Post,
            "Depth of field",
            "Thin-lens depth of field over the colour chain; also publishes its scratch textures.",
            "dof",
            110,
            vec![color_input(110), gbuffer_input("depth")],
            vec![
                color_output(),
                output(
                    "coc",
                    "CoC",
                    P::Image2d,
                    Some(F::R32Float),
                    SizeSpec::Scale(1.0),
                    1,
                    1,
                ),
                // Tiles are 16 px (raa-dof/src/shaders.rs:25), so 1/16 of the surface.
                output(
                    "tiles",
                    "Tiles",
                    P::Image2d,
                    Some(F::Rg32Float),
                    SizeSpec::Scale(0.0625),
                    1,
                    1,
                ),
                output(
                    "dilated",
                    "Dilated tiles",
                    P::Image2d,
                    Some(F::Rg32Float),
                    SizeSpec::Scale(0.0625),
                    1,
                    1,
                ),
                // The half-size prefilter and gather layers (raa-dof/src/resources.rs:389-393).
                output(
                    "nearPrefilter",
                    "Near prefilter",
                    P::Image2d,
                    Some(F::Rgba16Float),
                    SizeSpec::Scale(0.5),
                    1,
                    1,
                ),
                output(
                    "farPrefilter",
                    "Far prefilter",
                    P::Image2d,
                    Some(F::Rgba16Float),
                    SizeSpec::Scale(0.5),
                    1,
                    1,
                ),
                output(
                    "nearGather",
                    "Near gather",
                    P::Image2d,
                    Some(F::Rgba16Float),
                    SizeSpec::Scale(0.5),
                    1,
                    1,
                ),
                output(
                    "farGather",
                    "Far gather",
                    P::Image2d,
                    Some(F::Rgba16Float),
                    SizeSpec::Scale(0.5),
                    1,
                    1,
                ),
            ],
            vec![
                with_unit(
                    dof_p(
                        "focusDistance",
                        "Focus distance",
                        num(dof.focus_distance_m),
                        POSITIVE,
                        "crates/raa-dof/src/desc.rs:35",
                    ),
                    "m",
                ),
                dof_p(
                    "fStop",
                    "Aperture",
                    num(dof.f_stop),
                    ranged(0.7, 64.0),
                    "crates/raa-dof/src/desc.rs:36",
                ),
                with_unit(
                    dof_p(
                        "focalLength",
                        "Focal length",
                        num(dof.focal_length_mm),
                        ranged(8.0, 600.0),
                        "crates/raa-dof/src/desc.rs:37",
                    ),
                    "mm",
                ),
                with_unit(
                    dof_p(
                        "maxCoc",
                        "Max circle of confusion",
                        num(dof.max_coc_px),
                        ranged(0.0, 64.0),
                        "crates/raa-dof/src/desc.rs:39",
                    ),
                    "px",
                ),
            ],
            vec![
                // Compute: autofocus, CoC, tiles and dilate (raa-dof/src/resources.rs:238-270).
                budget(Compute, 1, 0, 1, 1, 1),
                // Fragment: composite (raa-dof/src/resources.rs:292-305).
                budget(Fragment, 5, 0, 0, 0, 1),
            ],
        ),
        pass_type(
            "pass.bloom",
            Post,
            "Bloom",
            "Bloom over the colour chain; also publishes the bloom chain levels.",
            "bloom",
            120,
            vec![color_input(120)],
            vec![
                color_output(),
                output(
                    "chain",
                    "Bloom chain",
                    P::Image2d,
                    Some(F::Rgba16Float),
                    SizeSpec::Scale(0.5),
                    raa_bloom::MAX_MIPS as u32,
                    1,
                ),
            ],
            vec![
                bloom_p(
                    "strength",
                    "Strength",
                    num(bloom.strength),
                    ranged(0.0, 1.0),
                    "crates/raa-bloom/src/desc.rs:20",
                ),
                bloom_p(
                    "radius",
                    "Radius",
                    num(bloom.radius),
                    ranged(0.0, 1.0),
                    "crates/raa-bloom/src/desc.rs:21",
                ),
            ],
            // The composite layout (raa-bloom/src/feature.rs:129): 2 textures, 1 sampler, 1 uniform.
            vec![budget(Fragment, 2, 1, 0, 0, 1)],
        ),
        pass_type(
            "pass.grading",
            Post,
            "Colour grading",
            "White balance, exposure, contrast, saturation and split toning over the colour chain.",
            "grading",
            130,
            vec![color_input(130)],
            vec![color_output()],
            vec![
                with_unit(
                    grading_p(
                        "temperature",
                        "Temperature",
                        num(grading.temperature_k),
                        ranged(1500.0, 15000.0),
                        "crates/raa-grading/src/desc.rs:40",
                    ),
                    "K",
                ),
                grading_p(
                    "tint",
                    "Tint",
                    num(grading.tint),
                    ranged(-1.0, 1.0),
                    "crates/raa-grading/src/desc.rs:41",
                ),
                with_unit(
                    grading_p(
                        "exposure",
                        "Exposure",
                        num(grading.exposure_ev),
                        ranged(-10.0, 10.0),
                        "crates/raa-grading/src/desc.rs:42",
                    ),
                    "EV",
                ),
                grading_p(
                    "contrast",
                    "Contrast",
                    num(grading.contrast),
                    ranged(0.0, 4.0),
                    "crates/raa-grading/src/desc.rs:43",
                ),
                grading_p(
                    "saturation",
                    "Saturation",
                    num(grading.saturation),
                    ranged(0.0, 4.0),
                    "crates/raa-grading/src/desc.rs:44",
                ),
                grading_p(
                    "vibrance",
                    "Vibrance",
                    num(grading.vibrance),
                    ranged(-1.0, 1.0),
                    "crates/raa-grading/src/desc.rs:45",
                ),
                colour(
                    Fid::Grading,
                    "Colour grading",
                    "lift",
                    "Lift",
                    grading.lift,
                    None,
                    "crates/raa-grading/src/desc.rs:46",
                ),
                colour(
                    Fid::Grading,
                    "Colour grading",
                    "gamma",
                    "Gamma",
                    grading.gamma,
                    Some(0.0),
                    "crates/raa-grading/src/desc.rs:47",
                ),
                colour(
                    Fid::Grading,
                    "Colour grading",
                    "gain",
                    "Gain",
                    grading.gain,
                    Some(0.0),
                    "crates/raa-grading/src/desc.rs:48",
                ),
                colour(
                    Fid::Grading,
                    "Colour grading",
                    "shadowsTint",
                    "Shadows tint",
                    grading.shadows_tint,
                    Some(0.0),
                    "crates/raa-grading/src/desc.rs:49",
                ),
                colour(
                    Fid::Grading,
                    "Colour grading",
                    "highlightsTint",
                    "Highlights tint",
                    grading.highlights_tint,
                    Some(0.0),
                    "crates/raa-grading/src/desc.rs:50",
                ),
                grading_p(
                    "splitBalance",
                    "Split balance",
                    num(grading.split_balance),
                    ranged(-1.0, 1.0),
                    "crates/raa-grading/src/desc.rs:51",
                ),
                grading_p(
                    "vignette",
                    "Vignette",
                    num(grading.vignette),
                    ranged(0.0, 1.0),
                    "crates/raa-grading/src/desc.rs:52",
                ),
            ],
            // The grading layout (raa-grading/src/feature.rs:101): uniform, texture, sampler.
            vec![budget(Fragment, 1, 1, 0, 0, 1)],
        ),
        // The S47 `sharpen` entry (order 195). Its strength stays the engine's `setSharpening`
        // value; the node only places the pass in the colour chain. It is a fullscreen post
        // pass, so after a display-sized resolve it runs at the display extent.
        pass_type(
            "pass.sharpen",
            Post,
            "Sharpen",
            "Contrast-adaptive sharpening over the colour chain (strength: setSharpening).",
            "sharpen",
            195,
            vec![color_input(195)],
            vec![color_output()],
            Vec::new(),
            // The sharpen layout (raa-sharpen/src/feature.rs:75): uniform and HDR source.
            vec![budget(Fragment, 1, 0, 0, 0, 1)],
        ),
        pass_type(
            "pass.cel.resolve",
            Post,
            "Cel shading",
            "Toon bands and ink outlines resolved from colour, depth and normal.",
            "cel.resolve",
            190,
            vec![
                color_input(190),
                gbuffer_input("depth"),
                gbuffer_input("normal"),
            ],
            vec![color_output()],
            vec![
                cel_p(
                    "bands",
                    "Bands",
                    int(cel.bands),
                    ranged(2.0, 32.0),
                    &[],
                    "crates/raa-cel/src/desc.rs:63",
                ),
                cel_p(
                    "hues",
                    "Hues",
                    int(cel.hues),
                    ranged(4.0, 32.0),
                    &[],
                    "crates/raa-cel/src/desc.rs:64",
                ),
                cel_p(
                    "chroma",
                    "Chroma",
                    int(cel.chroma),
                    ranged(2.0, 12.0),
                    &[],
                    "crates/raa-cel/src/desc.rs:65",
                ),
                cel_p(
                    "dither",
                    "Dither",
                    name(cel.dither.name()),
                    OPEN,
                    &[
                        "bayer2", "bayer4", "bayer8", "bayer16", "ign", "white", "blue",
                    ],
                    "crates/raa-cel/src/desc.rs:66",
                ),
                cel_p(
                    "ditherAmount",
                    "Dither amount",
                    num(cel.dither_amount),
                    ranged(0.0, 1.0),
                    &[],
                    "crates/raa-cel/src/desc.rs:67",
                ),
                with_unit(
                    cel_p(
                        "outlineWidth",
                        "Outline width",
                        num(cel.outline_width),
                        ranged(0.5, 6.0),
                        &[],
                        "crates/raa-cel/src/desc.rs:68",
                    ),
                    "px",
                ),
                cel_p(
                    "outlineSharpness",
                    "Outline sharpness",
                    num(cel.outline_sharpness),
                    ranged(0.0, 1.0),
                    &[],
                    "crates/raa-cel/src/desc.rs:69",
                ),
                cel_p(
                    "interiorEdges",
                    "Interior edges",
                    ParamValue::Bool(cel.interior_edges),
                    OPEN,
                    &[],
                    "crates/raa-cel/src/desc.rs:70",
                ),
                cel_p(
                    "edgeRetention",
                    "Edge retention",
                    num(cel.edge_retention),
                    ranged(0.0, 1.0),
                    &[],
                    "crates/raa-cel/src/desc.rs:71",
                ),
                cel_p(
                    "worldSpace",
                    "World space",
                    ParamValue::Bool(cel.world_space),
                    OPEN,
                    &[],
                    "crates/raa-cel/src/desc.rs:72",
                ),
                with_unit(
                    cel_p(
                        "worldScale",
                        "World scale",
                        num(cel.world_scale),
                        ranged(1.0, 512.0),
                        &[],
                        "crates/raa-cel/src/desc.rs:73",
                    ),
                    "1/m",
                ),
                cel_p(
                    "softness",
                    "Softness",
                    num(cel.softness),
                    ranged(0.0, 1.0),
                    &[],
                    "crates/raa-cel/src/desc.rs:74",
                ),
                cel_p(
                    "specular",
                    "Specular",
                    num(cel.specular),
                    ranged(0.0, 2.0),
                    &[],
                    "crates/raa-cel/src/desc.rs:75",
                ),
                cel_p(
                    "toon",
                    "Toon",
                    ParamValue::Bool(cel.toon),
                    OPEN,
                    &[],
                    "crates/raa-cel/src/desc.rs:76",
                ),
                cel_p(
                    "palette",
                    "Palette",
                    ParamValue::Bool(cel.palette),
                    OPEN,
                    &[],
                    "crates/raa-cel/src/desc.rs:77",
                ),
                cel_p(
                    "outlines",
                    "Outlines",
                    ParamValue::Bool(cel.outlines),
                    OPEN,
                    &[],
                    "crates/raa-cel/src/desc.rs:78",
                ),
                cel_p(
                    "ditherOutlines",
                    "Dither outlines",
                    ParamValue::Bool(cel.dither_outlines),
                    OPEN,
                    &[],
                    "crates/raa-cel/src/desc.rs:79",
                ),
            ],
            // The cel layout (raa-cel/src/feature.rs:61): 3 textures and 2 uniforms.
            vec![budget(Fragment, 3, 0, 0, 0, 2)],
        ),
        pass_type(
            "pass.tonemap",
            Post,
            "Tone mapping",
            "Tone mapping to the surface format; a required node of every graph.",
            "tonemap",
            200,
            vec![color_input(200), gbuffer_input("depth")],
            vec![output(
                "color",
                "Color",
                P::Image2d,
                Some(F::Surface),
                SizeSpec::Scale(1.0),
                1,
                1,
            )],
            vec![
                tone_p(
                    "mapper",
                    "Mapper",
                    name(mapper),
                    OPEN,
                    &["agx", "aces", "linear"],
                    "crates/raa-tonemap/src/lib.rs:75",
                ),
                with_unit(
                    tone_p(
                        "exposureEv",
                        "Exposure",
                        num(tone.exposure_ev),
                        OPEN,
                        &[],
                        "crates/raa-tonemap/src/lib.rs:76",
                    ),
                    "EV",
                ),
                tone_p(
                    "look",
                    "Look",
                    name(look),
                    OPEN,
                    &["none", "punchy", "golden"],
                    "crates/raa-tonemap/src/lib.rs:77",
                ),
                tone_p(
                    "dither",
                    "Dither",
                    ParamValue::Bool(tone.dither),
                    OPEN,
                    &[],
                    "crates/raa-tonemap/src/lib.rs:78",
                ),
            ],
            // The tone-mapping layout (raa-tonemap/src/pass.rs:90): 2 textures and 2 uniforms.
            vec![budget(Fragment, 2, 0, 0, 0, 2)],
        ),
        side_effect(pass_type(
            "pass.overlay",
            Output,
            "Overlay",
            "Line overlay drawn onto the tone-mapped output; a side-effect node.",
            "overlay",
            220,
            vec![PortSpec {
                required: true,
                ..input(
                    "color",
                    "Display color",
                    P::Image2d,
                    Some(F::Surface),
                    &["pass.tonemap.color"],
                    false,
                    None,
                )
            }],
            vec![output(
                "color",
                "Display color",
                P::Image2d,
                Some(F::Surface),
                SizeSpec::Scale(1.0),
                1,
                1,
            )],
            Vec::new(),
            // The lit frame group 0 that the overlay lines bind (raa-js/src/api/m07_pipeline.rs:65).
            vec![
                budget(Vertex, 1, 1, 0, 6, 1),
                budget(Fragment, 1, 1, 0, 5, 1),
            ],
        )),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Catalog;

    #[test]
    fn catalog_holds_this_share() {
        let catalog = Catalog::build();
        for spec in geometry_lighting_types() {
            assert!(catalog.node(spec.id).is_some(), "{}", spec.id);
        }
        assert!(catalog.node("output.display").is_some());
        let node = |id| catalog.node(id).unwrap();
        assert!(node("pass.prepass").required_in_graph);
        assert!(node("pass.forward").required_in_graph);
        assert!(node("pass.sky.atmosphere").side_effect);
        assert!(node("pass.sky.cloud-screen").side_effect);
        // The view family marks the required passes as side-effect nodes.
        assert!(node("pass.prepass").side_effect);
        let msaa = node("pass.forward")
            .params
            .iter()
            .find(|p| p.key == "msaa")
            .unwrap();
        assert_eq!(msaa.binding, ParamBinding::Engine { option: "msaa" });
    }

    #[test]
    fn catalog_holds_shadow_gi_share() {
        let catalog = Catalog::build();
        for id in [
            "pass.shadow.csm",
            "pass.gi.volume",
            "pass.ao.gtao",
            "pass.gi.screen",
            "pass.gi.area-shadow",
            "pass.gi.ssgi",
            "pass.shadow.contact",
            "pass.gi.history",
        ] {
            assert!(catalog.node(id).is_some(), "{id}");
        }
        let node = |id| catalog.node(id).unwrap();
        let radius = node("pass.ao.gtao")
            .params
            .iter()
            .find(|p| p.key == "radius")
            .unwrap();
        let expected = f64::from(raa_gtao::AoDesc::default().radius_m);
        assert_eq!(radius.default, ParamValue::Number(expected));
        assert!(node("pass.gi.ssgi").side_effect);
        assert!(node("pass.gi.volume").side_effect);
        assert!(!node("pass.gi.screen").side_effect);
    }

    #[test]
    fn color_sources_follow_chain() {
        assert_eq!(
            color_input(65).default_source,
            ["pass.gi.history.color", "pass.forward.color"]
        );
        assert_eq!(color_input(64).default_source, ["pass.forward.color"]);
        assert!(color_input(60).default_source.is_empty());
        assert_eq!(
            gbuffer_input("depth").default_source[1],
            "pass.vg.gbuffer.depth"
        );
    }

    #[test]
    fn catalog_holds_post_share() {
        let catalog = Catalog::build();
        for id in [
            "pass.ssr",
            "pass.fog",
            "pass.sunshafts",
            "pass.opaque.copy",
            "pass.transparent.oit",
            "pass.particles.blend",
            "pass.taa",
            "pass.dof",
            "pass.bloom",
            "pass.grading",
            "pass.cel.resolve",
            "pass.tonemap",
            "pass.overlay",
        ] {
            assert!(catalog.node(id).is_some(), "{id}");
        }
        let tonemap = catalog.node("pass.tonemap").unwrap();
        assert_eq!(tonemap.outputs[0].format, Some(GraphFormat::Surface));
        assert_eq!(
            tonemap.inputs[0].default_source[0],
            "pass.cel.resolve.color"
        );
        assert!(tonemap.required_in_graph);
        // The view family marks the required passes as side-effect nodes.
        assert!(tonemap.side_effect);
        assert!(catalog.node("pass.overlay").unwrap().side_effect);
    }
}
