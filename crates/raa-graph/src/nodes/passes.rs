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
            Some(F::R8Unorm),
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
        input(
            "opaqueColor",
            "Opaque color",
            P::Image2d,
            Some(F::Rgba16Float),
            &["pass.opaque.copy.opaqueColor"],
            true,
            Some(9),
        ),
        input(
            "opaqueDepth",
            "Opaque depth",
            P::Image2d,
            Some(F::R32Float),
            &["pass.opaque.copy.opaqueDepth"],
            true,
            Some(10),
        ),
    ]
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
            vec![color_output()],
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
        // raa-sdf-trace/src/pass.rs:1065 (frame uniform) and :1088 (two textures).
        contributor(
            "pass.trace.composite",
            Geometry,
            "Trace composite",
            "trace.composite",
            560,
            vec![
                budget(Vertex, 0, 0, 0, 0, 1),
                budget(Fragment, 2, 0, 0, 0, 1),
            ],
        ),
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
                image("ao", "AO", F::R8Unorm),
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
                image("contact", "Contact", F::R8Unorm),
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
        assert!(!node("pass.prepass").side_effect);
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
}
